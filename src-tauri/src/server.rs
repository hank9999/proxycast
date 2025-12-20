//! HTTP API 服务器
use crate::config::{
    Config, ConfigChangeEvent, ConfigChangeKind, ConfigManager, FileWatcher, HotReloadManager,
    ReloadResult,
};
use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::converter::openai_to_antigravity::{
    convert_antigravity_to_openai_response, convert_openai_to_antigravity,
};
use crate::credential::CredentialSyncService;
use crate::database::dao::provider_pool::ProviderPoolDao;
use crate::database::DbConnection;
use crate::injection::Injector;
use crate::logger::LogStore;
use crate::models::anthropic::*;
use crate::models::openai::*;
use crate::models::route_model::{RouteInfo, RouteListResponse};
use crate::processor::{RequestContext, RequestProcessor};
use crate::providers::antigravity::AntigravityProvider;
use crate::providers::claude_custom::ClaudeCustomProvider;
use crate::providers::gemini::GeminiProvider;
use crate::providers::kiro::KiroProvider;
use crate::providers::openai_custom::OpenAICustomProvider;
use crate::providers::qwen::QwenProvider;
use crate::providers::vertex::VertexProvider;
use crate::services::provider_pool_service::ProviderPoolService;
use crate::services::token_cache_service::TokenCacheService;
use crate::telemetry::{RequestLog, RequestStatus};
use crate::websocket::{WsConfig, WsConnectionManager, WsStats};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::stream;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};

/// 安全截断字符串到指定字符数，避免 UTF-8 边界问题
fn safe_truncate(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        chars[..max_chars].iter().collect()
    }
}

/// 计算 MessageContent 的字符长度
fn message_content_len(content: &crate::models::openai::MessageContent) -> usize {
    use crate::models::openai::{ContentPart, MessageContent};
    match content {
        MessageContent::Text(s) => s.len(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text { text } = p {
                    Some(text.len())
                } else {
                    None
                }
            })
            .sum(),
    }
}

/// 记录请求统计到遥测系统
fn record_request_telemetry(
    state: &AppState,
    ctx: &RequestContext,
    status: crate::telemetry::RequestStatus,
    error_message: Option<String>,
) {
    use crate::telemetry::RequestLog;

    let provider = ctx.provider.unwrap_or(crate::ProviderType::Kiro);
    let mut log = RequestLog::new(
        ctx.request_id.clone(),
        provider,
        ctx.resolved_model.clone(),
        ctx.is_stream,
    );

    // 设置状态和持续时间
    match status {
        crate::telemetry::RequestStatus::Success => log.mark_success(ctx.elapsed_ms(), 200),
        crate::telemetry::RequestStatus::Failed => log.mark_failed(
            ctx.elapsed_ms(),
            None,
            error_message.clone().unwrap_or_default(),
        ),
        crate::telemetry::RequestStatus::Timeout => log.mark_timeout(ctx.elapsed_ms()),
        crate::telemetry::RequestStatus::Cancelled => log.mark_cancelled(ctx.elapsed_ms()),
        crate::telemetry::RequestStatus::Retrying => {
            log.duration_ms = ctx.elapsed_ms();
        }
    }

    // 设置凭证 ID
    if let Some(cred_id) = &ctx.credential_id {
        log.set_credential_id(cred_id.clone());
    }

    // 设置重试次数
    log.retry_count = ctx.retry_count;

    // 记录到统计聚合器
    {
        let stats = state.processor.stats.write();
        stats.record(log.clone());
    }

    // 记录到请求日志记录器（用于前端日志列表显示）
    if let Some(logger) = &state.request_logger {
        let _ = logger.record(log.clone());
    }

    tracing::info!(
        "[TELEMETRY] request_id={} provider={:?} model={} status={:?} duration_ms={}",
        ctx.request_id,
        provider,
        ctx.resolved_model,
        status,
        ctx.elapsed_ms()
    );
}

/// 记录 Token 使用量到遥测系统
fn record_token_usage(
    state: &AppState,
    ctx: &RequestContext,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
) {
    use crate::telemetry::{TokenSource, TokenUsageRecord};

    // 只有当至少有一个 Token 值时才记录
    if input_tokens.is_none() && output_tokens.is_none() {
        return;
    }

    let provider = ctx.provider.unwrap_or(crate::ProviderType::Kiro);
    let record = TokenUsageRecord::new(
        uuid::Uuid::new_v4().to_string(),
        provider,
        ctx.resolved_model.clone(),
        input_tokens.unwrap_or(0),
        output_tokens.unwrap_or(0),
        TokenSource::Actual,
    )
    .with_request_id(ctx.request_id.clone());

    // 记录到 Token 追踪器
    {
        let tokens = state.processor.tokens.write();
        tokens.record(record);
    }

    tracing::debug!(
        "[TOKEN] request_id={} input={} output={}",
        ctx.request_id,
        input_tokens.unwrap_or(0),
        output_tokens.unwrap_or(0)
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub running: bool,
    pub host: String,
    pub port: u16,
    pub requests: u64,
    pub uptime_secs: u64,
}

pub struct ServerState {
    pub config: Config,
    pub running: bool,
    pub requests: u64,
    pub start_time: Option<std::time::Instant>,
    pub kiro_provider: KiroProvider,
    pub gemini_provider: GeminiProvider,
    pub qwen_provider: QwenProvider,
    pub openai_custom_provider: OpenAICustomProvider,
    pub claude_custom_provider: ClaudeCustomProvider,
    pub default_provider_ref: Arc<RwLock<String>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// 服务器运行时使用的 API key（启动时从配置复制）
    /// 用于 test_api 命令，确保测试使用的 API key 和服务器一致
    pub running_api_key: Option<String>,
}

impl ServerState {
    pub fn new(config: Config) -> Self {
        let kiro = KiroProvider::new();
        let gemini = GeminiProvider::new();
        let qwen = QwenProvider::new();
        let openai_custom = OpenAICustomProvider::new();
        let claude_custom = ClaudeCustomProvider::new();
        let default_provider_ref = Arc::new(RwLock::new(config.default_provider.clone()));

        Self {
            config,
            running: false,
            requests: 0,
            start_time: None,
            kiro_provider: kiro,
            gemini_provider: gemini,
            qwen_provider: qwen,
            openai_custom_provider: openai_custom,
            claude_custom_provider: claude_custom,
            default_provider_ref,
            shutdown_tx: None,
            running_api_key: None,
        }
    }

    pub fn status(&self) -> ServerStatus {
        ServerStatus {
            running: self.running,
            host: self.config.server.host.clone(),
            port: self.config.server.port,
            requests: self.requests,
            uptime_secs: self.start_time.map(|t| t.elapsed().as_secs()).unwrap_or(0),
        }
    }

    pub async fn start(
        &mut self,
        logs: Arc<RwLock<LogStore>>,
        pool_service: Arc<ProviderPoolService>,
        token_cache: Arc<TokenCacheService>,
        db: Option<DbConnection>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.start_with_telemetry(logs, pool_service, token_cache, db, None, None, None)
            .await
    }

    /// 启动服务器（使用共享的遥测实例）
    ///
    /// 这允许服务器与 TelemetryState 共享同一个 StatsAggregator、TokenTracker 和 RequestLogger，
    /// 使得请求处理过程中记录的统计数据能够在前端监控页面中显示。
    pub async fn start_with_telemetry(
        &mut self,
        logs: Arc<RwLock<LogStore>>,
        pool_service: Arc<ProviderPoolService>,
        token_cache: Arc<TokenCacheService>,
        db: Option<DbConnection>,
        shared_stats: Option<Arc<parking_lot::RwLock<crate::telemetry::StatsAggregator>>>,
        shared_tokens: Option<Arc<parking_lot::RwLock<crate::telemetry::TokenTracker>>>,
        shared_logger: Option<Arc<crate::telemetry::RequestLogger>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.running {
            return Ok(());
        }

        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        let host = self.config.server.host.clone();
        let port = self.config.server.port;
        let api_key = self.config.server.api_key.clone();
        let api_key_for_state = api_key.clone(); // 用于保存到 running_api_key
        let default_provider_ref = self.default_provider_ref.clone();

        if api_key.trim().is_empty() {
            return Err("API Key 不能为空".into());
        }

        if !is_localhost_host(&host) {
            return Err("当前版本仅支持本地监听，请使用 127.0.0.1/localhost/::1".into());
        }

        if (!is_localhost_host(&host) || self.config.remote_management.allow_remote)
            && crate::config::is_default_api_key(&api_key)
        {
            return Err("非本地访问场景下禁止使用默认 API Key，请设置强口令".into());
        }

        if self.config.server.tls.enable {
            return Err("当前版本暂不支持 TLS，请关闭 TLS 配置".into());
        }

        if self.config.remote_management.allow_remote {
            return Err("当前版本未启用 TLS，禁止开启远程管理".into());
        }

        // 重新加载凭证
        let _ = self.kiro_provider.load_credentials().await;
        let kiro = self.kiro_provider.clone();

        // 创建参数注入器
        let injection_enabled = self.config.injection.enabled;
        let injector = Injector::with_rules(
            self.config
                .injection
                .rules
                .iter()
                .map(|r| r.clone().into())
                .collect(),
        );

        // 获取配置和配置路径用于热重载
        let config = self.config.clone();
        let config_path = crate::config::ConfigManager::default_config_path();

        tokio::spawn(async move {
            if let Err(e) = run_server(
                &host,
                port,
                &api_key,
                default_provider_ref,
                kiro,
                logs,
                rx,
                pool_service,
                token_cache,
                db,
                injector,
                injection_enabled,
                shared_stats,
                shared_tokens,
                shared_logger,
                Some(config),
                Some(config_path),
            )
            .await
            {
                tracing::error!("Server error: {}", e);
            }
        });

        self.running = true;
        self.start_time = Some(std::time::Instant::now());
        // 保存服务器运行时使用的 API key，用于 test_api 命令
        self.running_api_key = Some(api_key_for_state);
        Ok(())
    }

    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        self.running = false;
        self.start_time = None;
        self.running_api_key = None;
    }
}

fn is_localhost_host(host: &str) -> bool {
    if host == "localhost" {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|addr| addr.is_loopback())
        .unwrap_or(false)
}

impl Clone for KiroProvider {
    fn clone(&self) -> Self {
        Self {
            credentials: self.credentials.clone(),
            client: reqwest::Client::new(),
            creds_path: self.creds_path.clone(),
        }
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct AppState {
    api_key: String,
    base_url: String,
    default_provider: Arc<RwLock<String>>,
    config: Option<Config>,
    config_manager: Option<Arc<std::sync::RwLock<ConfigManager>>>,
    start_time: std::time::Instant,
    kiro: Arc<RwLock<KiroProvider>>,
    logs: Arc<RwLock<LogStore>>,
    kiro_refresh_lock: Arc<tokio::sync::Mutex<()>>,
    gemini_refresh_lock: Arc<tokio::sync::Mutex<()>>,
    qwen_refresh_lock: Arc<tokio::sync::Mutex<()>>,
    pool_service: Arc<ProviderPoolService>,
    token_cache: Arc<TokenCacheService>,
    db: Option<DbConnection>,
    /// 参数注入器
    injector: Arc<RwLock<Injector>>,
    /// 是否启用参数注入
    injection_enabled: Arc<RwLock<bool>>,
    /// 请求处理器
    processor: Arc<RequestProcessor>,
    /// WebSocket 连接管理器
    ws_manager: Arc<WsConnectionManager>,
    /// WebSocket 统计信息
    ws_stats: Arc<WsStats>,
    /// 热重载管理器
    hot_reload_manager: Option<Arc<HotReloadManager>>,
    /// 请求日志记录器（与 TelemetryState 共享）
    request_logger: Option<Arc<crate::telemetry::RequestLogger>>,
    /// Amp CLI 路由器
    amp_router: Arc<crate::router::AmpRouter>,
}

/// 启动配置文件监控
///
/// 监控配置文件变化并触发热重载。
///
/// # 连接保持
///
/// 热重载过程不会中断现有连接：
/// - 配置更新在独立的 tokio 任务中异步执行
/// - 使用 RwLock 进行原子性更新，不会阻塞正在处理的请求
/// - 服务器继续运行，不需要重启
/// - HTTP 和 WebSocket 连接保持活跃
async fn start_config_watcher(
    config_path: PathBuf,
    hot_reload_manager: Option<Arc<HotReloadManager>>,
    processor: Arc<RequestProcessor>,
    logs: Arc<RwLock<LogStore>>,
    db: Option<DbConnection>,
    config_manager: Option<Arc<std::sync::RwLock<ConfigManager>>>,
) -> Option<FileWatcher> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfigChangeEvent>();

    // 创建文件监控器
    let mut watcher = match FileWatcher::new(&config_path, tx) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!("[HOT_RELOAD] 创建文件监控器失败: {}", e);
            return None;
        }
    };

    // 启动监控
    if let Err(e) = watcher.start() {
        tracing::error!("[HOT_RELOAD] 启动文件监控失败: {}", e);
        return None;
    }

    tracing::info!("[HOT_RELOAD] 配置文件监控已启动: {:?}", config_path);

    // 启动事件处理任务
    let hot_reload_manager_clone = hot_reload_manager.clone();
    let processor_clone = processor.clone();
    let logs_clone = logs.clone();
    let db_clone = db.clone();
    let config_manager_clone = config_manager.clone();

    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            // 只处理修改事件
            if event.kind != ConfigChangeKind::Modified {
                continue;
            }

            tracing::info!("[HOT_RELOAD] 检测到配置文件变更: {:?}", event.path);
            logs_clone.write().await.add(
                "info",
                &format!("[HOT_RELOAD] 检测到配置文件变更: {:?}", event.path),
            );

            // 执行热重载
            if let Some(ref manager) = hot_reload_manager_clone {
                let result = manager.reload();
                match &result {
                    ReloadResult::Success { .. } => {
                        tracing::info!("[HOT_RELOAD] 配置热重载成功");
                        logs_clone
                            .write()
                            .await
                            .add("info", "[HOT_RELOAD] 配置热重载成功");

                        // 更新处理器中的组件
                        let new_config = manager.config();
                        update_processor_config(&processor_clone, &new_config).await;

                        // 同步凭证池
                        if let (Some(ref db), Some(ref cfg_manager)) =
                            (&db_clone, &config_manager_clone)
                        {
                            match sync_credential_pool_from_config(db, cfg_manager, &logs_clone)
                                .await
                            {
                                Ok(count) => {
                                    tracing::info!(
                                        "[HOT_RELOAD] 凭证池同步完成，共 {} 个凭证",
                                        count
                                    );
                                    logs_clone.write().await.add(
                                        "info",
                                        &format!(
                                            "[HOT_RELOAD] 凭证池同步完成，共 {} 个凭证",
                                            count
                                        ),
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!("[HOT_RELOAD] 凭证池同步失败: {}", e);
                                    logs_clone.write().await.add(
                                        "warn",
                                        &format!("[HOT_RELOAD] 凭证池同步失败: {}", e),
                                    );
                                }
                            }
                        }
                    }
                    ReloadResult::RolledBack { error, .. } => {
                        tracing::warn!("[HOT_RELOAD] 配置热重载失败，已回滚: {}", error);
                        logs_clone.write().await.add(
                            "warn",
                            &format!("[HOT_RELOAD] 配置热重载失败，已回滚: {}", error),
                        );
                    }
                    ReloadResult::Failed {
                        error,
                        rollback_error,
                        ..
                    } => {
                        tracing::error!(
                            "[HOT_RELOAD] 配置热重载失败: {}, 回滚错误: {:?}",
                            error,
                            rollback_error
                        );
                        logs_clone.write().await.add(
                            "error",
                            &format!(
                                "[HOT_RELOAD] 配置热重载失败: {}, 回滚错误: {:?}",
                                error, rollback_error
                            ),
                        );
                    }
                }
            }
        }
    });

    Some(watcher)
}

/// 更新处理器配置
///
/// 当配置热重载成功后，更新 RequestProcessor 中的各个组件。
///
/// # 原子性更新
///
/// 每个组件的更新都是原子性的，使用 RwLock 确保：
/// - 正在处理的请求不会看到部分更新的状态
/// - 更新过程不会阻塞新请求的处理
/// - 现有连接不受影响
async fn update_processor_config(processor: &RequestProcessor, config: &Config) {
    // 更新注入器规则
    {
        let mut injector = processor.injector.write().await;
        injector.clear();
        for rule in &config.injection.rules {
            injector.add_rule(rule.clone().into());
        }
        tracing::debug!(
            "[HOT_RELOAD] 注入器规则已更新: {} 条规则",
            config.injection.rules.len()
        );
    }

    // 更新路由器规则
    {
        let mut router = processor.router.write().await;
        router.clear_rules();
        for rule in &config.routing.rules {
            // 解析 provider 字符串为 ProviderType
            if let Ok(provider_type) = rule.provider.parse::<crate::ProviderType>() {
                router.add_rule(crate::router::RoutingRule {
                    pattern: rule.pattern.clone(),
                    target_provider: provider_type,
                    priority: rule.priority,
                    enabled: true,
                });
            } else {
                tracing::warn!("[HOT_RELOAD] 无法解析 provider: {}", rule.provider);
            }
        }
        tracing::debug!(
            "[HOT_RELOAD] 路由规则已更新: {} 条规则",
            config.routing.rules.len()
        );
    }

    // 更新模型映射器
    {
        let mut mapper = processor.mapper.write().await;
        mapper.clear();
        for (alias, model) in &config.routing.model_aliases {
            mapper.add_alias(alias, model);
        }
        tracing::debug!(
            "[HOT_RELOAD] 模型别名已更新: {} 个别名",
            config.routing.model_aliases.len()
        );
    }

    // 注意：重试配置目前不支持热更新，因为 Retrier 是不可变的
    // 如果需要更新重试配置，需要重启服务器
    tracing::debug!(
        "[HOT_RELOAD] 重试配置: max_retries={}, base_delay={}ms (需重启生效)",
        config.retry.max_retries,
        config.retry.base_delay_ms
    );

    tracing::info!("[HOT_RELOAD] 处理器配置更新完成");
}

/// 从配置同步凭证池
///
/// 当配置热重载成功后，从 YAML 配置中加载凭证并同步到数据库。
///
/// # 同步策略
///
/// - 从配置中加载所有凭证
/// - 对于配置中存在但数据库中不存在的凭证，添加到数据库
/// - 对于配置中存在且数据库中也存在的凭证，更新数据库中的记录
/// - 对于数据库中存在但配置中不存在的凭证，保留（不删除，避免丢失运行时状态）
async fn sync_credential_pool_from_config(
    db: &DbConnection,
    config_manager: &Arc<std::sync::RwLock<ConfigManager>>,
    _logs: &Arc<RwLock<LogStore>>,
) -> Result<usize, String> {
    // 创建凭证同步服务
    let sync_service = CredentialSyncService::new(config_manager.clone());

    // 从配置加载凭证
    let credentials = sync_service.load_from_config().map_err(|e| e.to_string())?;

    let conn = db.lock().map_err(|e| e.to_string())?;
    let mut synced_count = 0;

    for cred in &credentials {
        // 检查凭证是否已存在
        let existing =
            ProviderPoolDao::get_by_uuid(&conn, &cred.uuid).map_err(|e| e.to_string())?;

        if existing.is_some() {
            // 更新现有凭证
            ProviderPoolDao::update(&conn, cred).map_err(|e| e.to_string())?;
            tracing::debug!(
                "[HOT_RELOAD] 更新凭证: {} ({})",
                cred.uuid,
                cred.provider_type
            );
        } else {
            // 添加新凭证
            ProviderPoolDao::insert(&conn, cred).map_err(|e| e.to_string())?;
            tracing::debug!(
                "[HOT_RELOAD] 添加凭证: {} ({})",
                cred.uuid,
                cred.provider_type
            );
        }
        synced_count += 1;
    }

    Ok(synced_count)
}

async fn run_server(
    host: &str,
    port: u16,
    api_key: &str,
    default_provider: Arc<RwLock<String>>,
    kiro: KiroProvider,
    logs: Arc<RwLock<LogStore>>,
    shutdown: oneshot::Receiver<()>,
    pool_service: Arc<ProviderPoolService>,
    token_cache: Arc<TokenCacheService>,
    db: Option<DbConnection>,
    injector: Injector,
    injection_enabled: bool,
    shared_stats: Option<Arc<parking_lot::RwLock<crate::telemetry::StatsAggregator>>>,
    shared_tokens: Option<Arc<parking_lot::RwLock<crate::telemetry::TokenTracker>>>,
    shared_logger: Option<Arc<crate::telemetry::RequestLogger>>,
    config: Option<Config>,
    config_path: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let base_url = format!("http://{}:{}", host, port);

    // 创建请求处理器（使用共享的遥测实例或默认实例）
    let processor = match (shared_stats, shared_tokens) {
        (Some(stats), Some(tokens)) => Arc::new(RequestProcessor::with_shared_telemetry(
            pool_service.clone(),
            stats,
            tokens,
        )),
        _ => Arc::new(RequestProcessor::with_defaults(pool_service.clone())),
    };

    // 将注入器规则同步到处理器
    {
        let mut proc_injector = processor.injector.write().await;
        for rule in injector.rules() {
            proc_injector.add_rule(rule.clone());
        }
    }

    // 初始化 WebSocket 管理器
    let ws_manager = Arc::new(WsConnectionManager::new(WsConfig::default()));
    let ws_stats = ws_manager.stats().clone();

    // 初始化热重载管理器
    let hot_reload_manager = match (&config, &config_path) {
        (Some(cfg), Some(path)) => Some(Arc::new(HotReloadManager::new(cfg.clone(), path.clone()))),
        _ => None,
    };

    // 初始化配置管理器（用于凭证池同步）
    let config_manager: Option<Arc<std::sync::RwLock<ConfigManager>>> =
        match (&config, &config_path) {
            (Some(cfg), Some(path)) => Some(Arc::new(std::sync::RwLock::new(
                ConfigManager::with_config(cfg.clone(), path.clone()),
            ))),
            _ => None,
        };

    let logs_clone = logs.clone();
    let db_clone = db.clone();

    // 初始化 Amp CLI 路由器
    let amp_router = Arc::new(crate::router::AmpRouter::new(
        config
            .as_ref()
            .map(|c| c.ampcode.clone())
            .unwrap_or_default(),
    ));

    let state = AppState {
        api_key: api_key.to_string(),
        base_url,
        default_provider,
        config: config.clone(),
        config_manager: config_manager.clone(),
        start_time: std::time::Instant::now(),
        kiro: Arc::new(RwLock::new(kiro)),
        logs,
        kiro_refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        gemini_refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        qwen_refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        pool_service,
        token_cache,
        db,
        injector: Arc::new(RwLock::new(injector)),
        injection_enabled: Arc::new(RwLock::new(injection_enabled)),
        processor: processor.clone(),
        ws_manager,
        ws_stats,
        hot_reload_manager: hot_reload_manager.clone(),
        request_logger: shared_logger,
        amp_router,
    };

    // 启动配置文件监控
    let _file_watcher = if let Some(path) = config_path {
        start_config_watcher(
            path,
            hot_reload_manager,
            processor,
            logs_clone,
            db_clone,
            config_manager,
        )
        .await
    } else {
        None
    };

    // 设置请求体大小限制为 100MB，支持大型上下文请求（如 Claude Code 的 /compact 命令）
    let body_limit = 100 * 1024 * 1024; // 100MB

    // 创建管理 API 路由（带认证中间件）
    let management_config = config
        .as_ref()
        .map(|c| c.remote_management.clone())
        .unwrap_or_default();

    let management_routes = Router::new()
        .route("/v0/management/status", get(management_status))
        .route(
            "/v0/management/credentials",
            get(management_list_credentials),
        )
        .route(
            "/v0/management/credentials",
            post(management_add_credential),
        )
        .route("/v0/management/config", get(management_get_config))
        .route(
            "/v0/management/config",
            axum::routing::put(management_update_config),
        )
        .layer(crate::middleware::ManagementAuthLayer::new(
            management_config,
        ));

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/routes", get(list_routes))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        // WebSocket 路由
        .route("/v1/ws", get(ws_upgrade_handler))
        .route("/ws", get(ws_upgrade_handler))
        // 多供应商路由
        .route(
            "/:selector/v1/messages",
            post(anthropic_messages_with_selector),
        )
        .route(
            "/:selector/v1/chat/completions",
            post(chat_completions_with_selector),
        )
        // Amp CLI 路由
        .route(
            "/api/provider/:provider/v1/chat/completions",
            post(amp_chat_completions),
        )
        .route("/api/provider/:provider/v1/messages", post(amp_messages))
        // Amp CLI 管理代理路由
        .route(
            "/api/auth/*path",
            axum::routing::any(amp_management_proxy_auth),
        )
        .route(
            "/api/user/*path",
            axum::routing::any(amp_management_proxy_user),
        )
        // 管理 API 路由
        .merge(management_routes)
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("Server listening on {}", addr);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let _ = shutdown.await;
    })
    .await?;

    Ok(())
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let (db_ready, pool_stats) = match &state.db {
        Some(db) => {
            let db_ok = db
                .lock()
                .map(|conn| conn.query_row::<i32, _, _>("SELECT 1", [], |row| row.get(0)))
                .is_ok();
            let stats = if db_ok {
                state.pool_service.get_overview(db).ok().map(|items| {
                    let mut total = 0usize;
                    let mut healthy = 0usize;
                    let mut disabled = 0usize;
                    for item in items {
                        total += item.stats.total_count;
                        healthy += item.stats.healthy_count;
                        disabled += item.stats.disabled_count;
                    }
                    serde_json::json!({
                        "total": total,
                        "healthy": healthy,
                        "disabled": disabled
                    })
                })
            } else {
                None
            };
            (db_ok, stats)
        }
        None => (false, None),
    };

    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "db_ready": db_ready,
        "pool": pool_stats
    }))
}

async fn models() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            // Kiro/Claude models
            {"id": "claude-sonnet-4-5", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-3-7-sonnet-20250219", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-3-5-sonnet-latest", "object": "model", "owned_by": "anthropic"},
            // Gemini models
            {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-flash-lite", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-pro-preview-06-05", "object": "model", "owned_by": "google"},
            {"id": "gemini-3-pro-preview", "object": "model", "owned_by": "google"},
            // Qwen models
            {"id": "qwen3-coder-plus", "object": "model", "owned_by": "alibaba"},
            {"id": "qwen3-coder-flash", "object": "model", "owned_by": "alibaba"}
        ]
    }))
}

async fn verify_api_key(
    headers: &HeaderMap,
    expected_key: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let auth = headers
        .get("authorization")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": {"message": "No API key provided"}})),
            ))
        }
    };

    if key != expected_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "Invalid API key"}})),
        ));
    }

    Ok(())
}

/// Anthropic 格式的 API key 验证
/// 返回 Anthropic 标准错误格式：{"type": "error", "error": {"type": "...", "message": "..."}}
async fn verify_api_key_anthropic(
    headers: &HeaderMap,
    expected_key: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let auth = headers
        .get("x-api-key")
        .or_else(|| headers.get("authorization"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "No API key provided. Please set the x-api-key header."
                    }
                })),
            ))
        }
    };

    if key != expected_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key"
                }
            })),
        ));
    }

    Ok(())
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/chat/completions");
        return e.into_response();
    }

    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    state.logs.write().await.add(
        "info",
        &format!(
            "POST /v1/chat/completions request_id={} model={} stream={}",
            ctx.request_id, request.model, request.stream
        ),
    );

    // 使用 RequestProcessor 解析模型别名和路由
    let provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
        state.logs.write().await.add(
            "info",
            &format!(
                "[MAPPER] request_id={} alias={} -> model={}",
                ctx.request_id, ctx.original_model, ctx.resolved_model
            ),
        );
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INJECT] request_id={} applied_rules={:?} injected_params={:?}",
                    ctx.request_id, result.applied_rules, result.injected_params
                ),
            );
            // 更新请求
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取当前默认 provider（用于凭证池选择）
    let default_provider = state.default_provider.read().await.clone();

    // 记录路由结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[ROUTE] request_id={} model={} provider={}",
            ctx.request_id, ctx.resolved_model, provider
        ),
    );

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => state
            .pool_service
            .select_credential(db, &default_provider, Some(&request.model))
            .ok()
            .flatten(),
        None => None,
    };

    // 如果找到凭证池中的凭证，使用它
    if let Some(cred) = credential {
        state.logs.write().await.add(
            "info",
            &format!(
                "[ROUTE] Using pool credential: type={} name={:?} uuid={}",
                cred.provider_type,
                cred.name,
                &cred.uuid[..8]
            ),
        );
        let response = call_provider_openai(&state, &cred, &request).await;

        // 记录请求统计
        let is_success = response.status().is_success();
        let status = if is_success {
            crate::telemetry::RequestStatus::Success
        } else {
            crate::telemetry::RequestStatus::Failed
        };
        record_request_telemetry(&state, &ctx, status, None);

        // 如果成功，记录估算的 Token 使用量
        if is_success {
            let estimated_input_tokens = request
                .messages
                .iter()
                .map(|m| {
                    let content_len = match &m.content {
                        Some(c) => message_content_len(c),
                        None => 0,
                    };
                    content_len / 4
                })
                .sum::<usize>() as u32;
            // 输出 Token 使用估算值（假设平均响应长度）
            let estimated_output_tokens = 100u32;
            record_token_usage(
                &state,
                &ctx,
                Some(estimated_input_tokens),
                Some(estimated_output_tokens),
            );
        }

        return response;
    }

    // 回退到旧的单凭证模式
    state.logs.write().await.add(
        "debug",
        &format!(
            "[ROUTE] No pool credential found for '{}', using legacy mode",
            default_provider
        ),
    );

    // 检查是否需要刷新 token（无 token 或即将过期）
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("Token refresh failed: {e}"));
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                ).into_response();
            }
        }
    }

    let kiro = state.kiro.read().await;

    match kiro.call_api(&request).await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.text().await {
                    Ok(body) => {
                        let parsed = parse_cw_response(&body);
                        let has_tool_calls = !parsed.tool_calls.is_empty();

                        state.logs.write().await.add(
                            "info",
                            &format!(
                                "Request completed: content_len={}, tool_calls={}",
                                parsed.content.len(),
                                parsed.tool_calls.len()
                            ),
                        );

                        // 构建消息
                        let message = if has_tool_calls {
                            serde_json::json!({
                                "role": "assistant",
                                "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                    serde_json::json!({
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    })
                                }).collect::<Vec<_>>()
                            })
                        } else {
                            serde_json::json!({
                                "role": "assistant",
                                "content": parsed.content
                            })
                        };

                        // 估算 Token 数量（基于字符数，约 4 字符 = 1 token）
                        let estimated_output_tokens = (parsed.content.len() / 4) as u32;
                        // 估算输入 Token（基于请求消息）
                        let estimated_input_tokens = request
                            .messages
                            .iter()
                            .map(|m| {
                                let content_len = match &m.content {
                                    Some(c) => message_content_len(c),
                                    None => 0,
                                };
                                content_len / 4
                            })
                            .sum::<usize>()
                            as u32;

                        let response = serde_json::json!({
                            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                            "object": "chat.completion",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "message": message,
                                "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                            }],
                            "usage": {
                                "prompt_tokens": estimated_input_tokens,
                                "completion_tokens": estimated_output_tokens,
                                "total_tokens": estimated_input_tokens + estimated_output_tokens
                            }
                        });
                        // 记录成功请求统计
                        record_request_telemetry(
                            &state,
                            &ctx,
                            crate::telemetry::RequestStatus::Success,
                            None,
                        );
                        // 记录 Token 使用量
                        record_token_usage(
                            &state,
                            &ctx,
                            Some(estimated_input_tokens),
                            Some(estimated_output_tokens),
                        );
                        Json(response).into_response()
                    }
                    Err(e) => {
                        // 记录失败请求统计
                        record_request_telemetry(
                            &state,
                            &ctx,
                            crate::telemetry::RequestStatus::Failed,
                            Some(e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 403 || status.as_u16() == 402 {
                // Token 过期或账户问题，尝试重新加载凭证并刷新
                drop(kiro);
                let _guard = state.kiro_refresh_lock.lock().await;
                let mut kiro = state.kiro.write().await;
                state.logs.write().await.add(
                    "warn",
                    &format!(
                        "[AUTH] Got {}, reloading credentials and attempting token refresh...",
                        status.as_u16()
                    ),
                );

                // 先重新加载凭证文件（可能用户换了账户）
                if let Err(e) = kiro.load_credentials().await {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AUTH] Failed to reload credentials: {e}"),
                    );
                }

                match kiro.refresh_token().await {
                    Ok(_) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("info", "[AUTH] Token refreshed successfully after reload");
                        // 重试请求
                        drop(kiro);
                        let kiro = state.kiro.read().await;
                        match kiro.call_api(&request).await {
                            Ok(retry_resp) => {
                                if retry_resp.status().is_success() {
                                    match retry_resp.text().await {
                                        Ok(body) => {
                                            let parsed = parse_cw_response(&body);
                                            let has_tool_calls = !parsed.tool_calls.is_empty();

                                            let message = if has_tool_calls {
                                                serde_json::json!({
                                                    "role": "assistant",
                                                    "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                                    "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                                        serde_json::json!({
                                                            "id": tc.id,
                                                            "type": "function",
                                                            "function": {
                                                                "name": tc.function.name,
                                                                "arguments": tc.function.arguments
                                                            }
                                                        })
                                                    }).collect::<Vec<_>>()
                                                })
                                            } else {
                                                serde_json::json!({
                                                    "role": "assistant",
                                                    "content": parsed.content
                                                })
                                            };

                                            let response = serde_json::json!({
                                                "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                                "object": "chat.completion",
                                                "created": std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_secs(),
                                                "model": request.model,
                                                "choices": [{
                                                    "index": 0,
                                                    "message": message,
                                                    "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                                                }],
                                                "usage": {
                                                    "prompt_tokens": 0,
                                                    "completion_tokens": 0,
                                                    "total_tokens": 0
                                                }
                                            });
                                            return Json(response).into_response();
                                        }
                                        Err(e) => return (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                        ).into_response(),
                                    }
                                }
                                let body = retry_resp.text().await.unwrap_or_default();
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                                ).into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                        (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                state.logs.write().await.add(
                    "error",
                    &format!("Upstream error {}: {}", status, safe_truncate(&body, 200)),
                );
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}))
                ).into_response()
            }
        }
        Err(e) => {
            state
                .logs
                .write()
                .await
                .add("error", &format!("API call failed: {e}"));
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response()
        }
    }
}

async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<AnthropicMessagesRequest>,
) -> Response {
    // 使用 Anthropic 格式的认证验证（优先检查 x-api-key）
    if let Err(e) = verify_api_key_anthropic(&headers, &state.api_key).await {
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/messages");
        return e.into_response();
    }

    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    // 详细记录请求信息
    let msg_count = request.messages.len();
    let has_tools = request.tools.as_ref().map(|t| t.len()).unwrap_or(0);
    let has_system = request.system.is_some();
    state.logs.write().await.add(
        "info",
        &format!(
            "[REQ] POST /v1/messages request_id={} model={} stream={} messages={} tools={} has_system={}",
            ctx.request_id, request.model, request.stream, msg_count, has_tools, has_system
        ),
    );

    // 使用 RequestProcessor 解析模型别名和路由
    let provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
        state.logs.write().await.add(
            "info",
            &format!(
                "[MAPPER] request_id={} alias={} -> model={}",
                ctx.request_id, ctx.original_model, ctx.resolved_model
            ),
        );
    }

    // 记录最后一条消息的角色和内容预览
    if let Some(last_msg) = request.messages.last() {
        let content_preview = match &last_msg.content {
            serde_json::Value::String(s) => s.chars().take(100).collect::<String>(),
            serde_json::Value::Array(arr) => {
                if let Some(first) = arr.first() {
                    if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                        text.chars().take(100).collect::<String>()
                    } else {
                        format!("[{} blocks]", arr.len())
                    }
                } else {
                    "[empty]".to_string()
                }
            }
            _ => "[unknown]".to_string(),
        };
        state.logs.write().await.add(
            "debug",
            &format!(
                "[REQ] request_id={} last_message: role={} content={}",
                ctx.request_id, last_msg.role, content_preview
            ),
        );
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INJECT] request_id={} applied_rules={:?} injected_params={:?}",
                    ctx.request_id, result.applied_rules, result.injected_params
                ),
            );
            // 更新请求
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取当前默认 provider（用于凭证池选择）
    let default_provider = state.default_provider.read().await.clone();

    // 记录路由结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[ROUTE] request_id={} model={} provider={}",
            ctx.request_id, ctx.resolved_model, provider
        ),
    );

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => {
            // 根据 default_provider 配置选择凭证
            state
                .pool_service
                .select_credential(db, &default_provider, Some(&request.model))
                .ok()
                .flatten()
        }
        None => None,
    };

    // 如果找到凭证池中的凭证，使用它
    if let Some(cred) = credential {
        state.logs.write().await.add(
            "info",
            &format!(
                "[ROUTE] Using pool credential: type={} name={:?} uuid={}",
                cred.provider_type,
                cred.name,
                &cred.uuid[..8]
            ),
        );
        let response = call_provider_anthropic(&state, &cred, &request).await;

        // 记录请求统计
        let is_success = response.status().is_success();
        let status = if is_success {
            crate::telemetry::RequestStatus::Success
        } else {
            crate::telemetry::RequestStatus::Failed
        };
        record_request_telemetry(&state, &ctx, status, None);

        // 如果成功，记录估算的 Token 使用量
        if is_success {
            let estimated_input_tokens = request
                .messages
                .iter()
                .map(|m| {
                    let content_len = match &m.content {
                        serde_json::Value::String(s) => s.len(),
                        serde_json::Value::Array(arr) => arr
                            .iter()
                            .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                            .map(|s| s.len())
                            .sum(),
                        _ => 0,
                    };
                    content_len / 4
                })
                .sum::<usize>() as u32;
            // 输出 Token 使用估算值
            let estimated_output_tokens = 100u32;
            record_token_usage(
                &state,
                &ctx,
                Some(estimated_input_tokens),
                Some(estimated_output_tokens),
            );
        }

        return response;
    }

    // 回退到旧的单凭证模式
    state.logs.write().await.add(
        "debug",
        &format!(
            "[ROUTE] No pool credential found for '{}', using legacy mode",
            default_provider
        ),
    );

    // 检查是否需要刷新 token（无 token 或即将过期）
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            state.logs.write().await.add(
                "info",
                "[AUTH] No access token or token expiring soon, attempting refresh...",
            );
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                )
                    .into_response();
            }
            state
                .logs
                .write()
                .await
                .add("info", "[AUTH] Token refreshed successfully");
        }
    }

    // 转换为 OpenAI 格式
    let openai_request = convert_anthropic_to_openai(&request);

    // 记录转换后的请求信息
    state.logs.write().await.add(
        "debug",
        &format!(
            "[CONVERT] OpenAI format: messages={} tools={} stream={}",
            openai_request.messages.len(),
            openai_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            openai_request.stream
        ),
    );

    let kiro = state.kiro.read().await;

    match kiro.call_api(&openai_request).await {
        Ok(resp) => {
            let status = resp.status();
            state
                .logs
                .write()
                .await
                .add("info", &format!("[RESP] Upstream status: {status}"));

            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        // 使用 lossy 转换，避免无效 UTF-8 导致崩溃
                        let body = String::from_utf8_lossy(&bytes).to_string();

                        // 记录原始响应长度
                        state.logs.write().await.add(
                            "debug",
                            &format!("[RESP] Raw body length: {} bytes", bytes.len()),
                        );

                        // 保存原始响应到文件用于调试
                        let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
                        state.logs.read().await.log_raw_response(&request_id, &body);
                        state.logs.write().await.add(
                            "debug",
                            &format!("[RESP] Raw response saved to raw_response_{request_id}.txt"),
                        );

                        // 记录响应的前200字符用于调试（减少日志量）
                        let preview: String =
                            body.chars().filter(|c| !c.is_control()).take(200).collect();
                        state
                            .logs
                            .write()
                            .await
                            .add("debug", &format!("[RESP] Body preview: {preview}"));

                        let parsed = parse_cw_response(&body);

                        // 详细记录解析结果
                        state.logs.write().await.add(
                            "info",
                            &format!(
                                "[RESP] Parsed: content_len={}, tool_calls={}, content_preview={}",
                                parsed.content.len(),
                                parsed.tool_calls.len(),
                                parsed.content.chars().take(100).collect::<String>()
                            ),
                        );

                        // 记录 tool calls 详情
                        for (i, tc) in parsed.tool_calls.iter().enumerate() {
                            state.logs.write().await.add(
                                "debug",
                                &format!(
                                    "[RESP] Tool call {}: name={} id={}",
                                    i, tc.function.name, tc.id
                                ),
                            );
                        }

                        // 如果请求流式响应，返回 SSE 格式
                        if request.stream {
                            return build_anthropic_stream_response(&request.model, &parsed);
                        }

                        // 非流式响应
                        build_anthropic_response(&request.model, &parsed)
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[ERROR] Response body read failed: {e}"));
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 403 || status.as_u16() == 402 {
                // Token 过期或账户问题，尝试重新加载凭证并刷新
                drop(kiro);
                let _guard = state.kiro_refresh_lock.lock().await;
                let mut kiro = state.kiro.write().await;
                state.logs.write().await.add(
                    "warn",
                    &format!(
                        "[AUTH] Got {}, reloading credentials and attempting token refresh...",
                        status.as_u16()
                    ),
                );

                // 先重新加载凭证文件（可能用户换了账户）
                if let Err(e) = kiro.load_credentials().await {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AUTH] Failed to reload credentials: {e}"),
                    );
                }

                match kiro.refresh_token().await {
                    Ok(_) => {
                        state.logs.write().await.add(
                            "info",
                            "[AUTH] Token refreshed successfully, retrying request...",
                        );
                        drop(kiro);
                        let kiro = state.kiro.read().await;
                        match kiro.call_api(&openai_request).await {
                            Ok(retry_resp) => {
                                let retry_status = retry_resp.status();
                                state.logs.write().await.add(
                                    "info",
                                    &format!("[RETRY] Response status: {retry_status}"),
                                );
                                if retry_resp.status().is_success() {
                                    match retry_resp.bytes().await {
                                        Ok(bytes) => {
                                            let body = String::from_utf8_lossy(&bytes).to_string();
                                            let parsed = parse_cw_response(&body);
                                            state.logs.write().await.add(
                                                "info",
                                                &format!(
                                                "[RETRY] Success: content_len={}, tool_calls={}",
                                                parsed.content.len(), parsed.tool_calls.len()
                                            ),
                                            );
                                            if request.stream {
                                                return build_anthropic_stream_response(
                                                    &request.model,
                                                    &parsed,
                                                );
                                            }
                                            return build_anthropic_response(
                                                &request.model,
                                                &parsed,
                                            );
                                        }
                                        Err(e) => {
                                            state.logs.write().await.add(
                                                "error",
                                                &format!("[RETRY] Body read failed: {e}"),
                                            );
                                            return (
                                                StatusCode::INTERNAL_SERVER_ERROR,
                                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                            )
                                                .into_response();
                                        }
                                    }
                                }
                                let body = retry_resp
                                    .bytes()
                                    .await
                                    .map(|b| String::from_utf8_lossy(&b).to_string())
                                    .unwrap_or_default();
                                state.logs.write().await.add(
                                    "error",
                                    &format!(
                                        "[RETRY] Failed with status {retry_status}: {}",
                                        safe_truncate(&body, 500)
                                    ),
                                );
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                                )
                                    .into_response()
                            }
                            Err(e) => {
                                state
                                    .logs
                                    .write()
                                    .await
                                    .add("error", &format!("[RETRY] Request failed: {e}"));
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                        (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                state.logs.write().await.add(
                    "error",
                    &format!(
                        "[ERROR] Upstream error HTTP {}: {}",
                        status,
                        safe_truncate(&body, 500)
                    ),
                );
                (
                    StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(
                        serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}),
                    ),
                )
                    .into_response()
            }
        }
        Err(e) => {
            // 详细记录网络/连接错误
            let error_details = format!("{e:?}");
            state
                .logs
                .write()
                .await
                .add("error", &format!("[ERROR] Kiro API call failed: {e}"));
            state.logs.write().await.add(
                "debug",
                &format!("[ERROR] Full error details: {error_details}"),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response()
        }
    }
}

/// 构建 Anthropic 非流式响应
fn build_anthropic_response(model: &str, parsed: &CWParsedResponse) -> Response {
    let has_tool_calls = !parsed.tool_calls.is_empty();
    let mut content_array: Vec<serde_json::Value> = Vec::new();

    if !parsed.content.is_empty() {
        content_array.push(serde_json::json!({
            "type": "text",
            "text": parsed.content
        }));
    }

    for tc in &parsed.tool_calls {
        let input: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
        content_array.push(serde_json::json!({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.function.name,
            "input": input
        }));
    }

    if content_array.is_empty() {
        content_array.push(serde_json::json!({"type": "text", "text": ""}));
    }

    // 估算 output tokens: 基于响应内容长度 (约 4 字符 = 1 token)
    let mut output_tokens: u32 = (parsed.content.len() / 4) as u32;
    for tc in &parsed.tool_calls {
        output_tokens += (tc.function.arguments.len() / 4) as u32;
    }
    // 从 context_usage_percentage 估算 input tokens
    // 假设 100% = 200k tokens (Claude 的上下文窗口)
    let input_tokens = ((parsed.context_usage_percentage / 100.0) * 200000.0) as u32;

    let response = serde_json::json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4()),
        "type": "message",
        "role": "assistant",
        "content": content_array,
        "model": model,
        "stop_reason": if has_tool_calls { "tool_use" } else { "end_turn" },
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    });
    Json(response).into_response()
}

/// 构建 Anthropic 流式响应 (SSE)
fn build_anthropic_stream_response(model: &str, parsed: &CWParsedResponse) -> Response {
    let has_tool_calls = !parsed.tool_calls.is_empty();
    let message_id = format!("msg_{}", uuid::Uuid::new_v4());
    let model = model.to_string();
    let content = parsed.content.clone();
    let tool_calls = parsed.tool_calls.clone();

    // 估算 output tokens: 基于响应内容长度 (约 4 字符 = 1 token)
    let mut output_tokens: u32 = (parsed.content.len() / 4) as u32;
    for tc in &parsed.tool_calls {
        output_tokens += (tc.function.arguments.len() / 4) as u32;
    }
    // 从 context_usage_percentage 估算 input tokens
    let input_tokens = ((parsed.context_usage_percentage / 100.0) * 200000.0) as u32;

    // 构建 SSE 事件流
    let mut events: Vec<String> = Vec::new();

    // 1. message_start
    let message_start = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}
        }
    });
    events.push(format!("event: message_start\ndata: {message_start}\n\n"));

    let mut block_index = 0;

    // 2. 文本内容块 - 即使为空也要发送，Claude Code 需要至少一个 content block
    // content_block_start
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {"type": "text", "text": ""}
    });
    events.push(format!(
        "event: content_block_start\ndata: {block_start}\n\n"
    ));

    if !content.is_empty() {
        // content_block_delta - 发送完整内容
        let block_delta = serde_json::json!({
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "text_delta", "text": content}
        });
        events.push(format!(
            "event: content_block_delta\ndata: {block_delta}\n\n"
        ));
    }

    // content_block_stop
    let block_stop = serde_json::json!({
        "type": "content_block_stop",
        "index": block_index
    });
    events.push(format!("event: content_block_stop\ndata: {block_stop}\n\n"));

    block_index += 1;

    // 3. Tool use 块
    for tc in &tool_calls {
        // content_block_start
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": {}
            }
        });
        events.push(format!(
            "event: content_block_start\ndata: {block_start}\n\n"
        ));

        // content_block_delta - input_json_delta
        // 注意：partial_json 应该是原始 JSON 字符串，不是再次序列化的
        let partial_json = if tc.function.arguments.is_empty() {
            "{}".to_string()
        } else {
            tc.function.arguments.clone()
        };
        let block_delta = serde_json::json!({
            "type": "content_block_delta",
            "index": block_index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": partial_json
            }
        });
        events.push(format!(
            "event: content_block_delta\ndata: {block_delta}\n\n"
        ));

        // content_block_stop
        let block_stop = serde_json::json!({
            "type": "content_block_stop",
            "index": block_index
        });
        events.push(format!("event: content_block_stop\ndata: {block_stop}\n\n"));

        block_index += 1;
    }

    // 4. message_delta
    let message_delta = serde_json::json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": if has_tool_calls { "tool_use" } else { "end_turn" },
            "stop_sequence": null
        },
        "usage": {"output_tokens": output_tokens}
    });
    events.push(format!("event: message_delta\ndata: {message_delta}\n\n"));

    // 5. message_stop
    let message_stop = serde_json::json!({"type": "message_stop"});
    events.push(format!("event: message_stop\ndata: {message_stop}\n\n"));

    // 创建 SSE 响应
    let body_stream = stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
    let body = Body::from_stream(body_stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap_or_else(|e| {
            tracing::error!("Failed to build SSE response: {}", e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .unwrap_or_default()
        })
}

async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(_request): Json<serde_json::Value>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        return e.into_response();
    }

    // Claude Code 需要这个端点，返回估算值
    Json(serde_json::json!({
        "input_tokens": 100
    }))
    .into_response()
}

/// CodeWhisperer 响应解析结果
#[derive(Debug, Default)]
struct CWParsedResponse {
    content: String,
    tool_calls: Vec<ToolCall>,
    usage_credits: f64,
    context_usage_percentage: f64,
}

/// 解析 CodeWhisperer AWS Event Stream 响应
/// AWS Event Stream 是二进制格式，JSON payload 嵌入在二进制头部之间
fn parse_cw_response(body: &str) -> CWParsedResponse {
    let mut result = CWParsedResponse::default();
    // 使用 HashMap 来跟踪多个并发的 tool calls
    // key: toolUseId, value: (name, input_accumulated)
    let mut tool_map: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();

    // 将字符串转换为字节，因为 AWS Event Stream 包含二进制数据
    let bytes = body.as_bytes();

    // 搜索所有 JSON 对象的模式
    // AWS Event Stream 格式: [binary headers]{"content":"..."}[binary trailer]
    let json_patterns: &[&[u8]] = &[
        b"{\"content\":",
        b"{\"name\":",
        b"{\"input\":",
        b"{\"stop\":",
        b"{\"followupPrompt\":",
        b"{\"toolUseId\":",
        b"{\"unit\":",                   // meteringEvent
        b"{\"contextUsagePercentage\":", // contextUsageEvent
    ];

    let mut pos = 0;
    while pos < bytes.len() {
        // 找到下一个 JSON 对象的开始
        let mut next_start: Option<usize> = None;

        for pattern in json_patterns {
            if let Some(idx) = find_subsequence(&bytes[pos..], pattern) {
                let abs_pos = pos + idx;
                if next_start.is_none_or(|start| abs_pos < start) {
                    next_start = Some(abs_pos);
                }
            }
        }

        let start = match next_start {
            Some(s) => s,
            None => break,
        };

        // 从 start 位置提取完整的 JSON 对象
        if let Some(json_str) = extract_json_from_bytes(&bytes[start..]) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_str) {
                // 处理 content 事件
                if let Some(content) = value.get("content").and_then(|v| v.as_str()) {
                    // 跳过 followupPrompt
                    if value.get("followupPrompt").is_none() {
                        result.content.push_str(content);
                    }
                }
                // 处理 tool use 事件 (包含 toolUseId)
                else if let Some(tool_use_id) = value.get("toolUseId").and_then(|v| v.as_str()) {
                    let name = value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input_chunk = value
                        .get("input")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let is_stop = value.get("stop").and_then(|v| v.as_bool()).unwrap_or(false);

                    // 获取或创建 tool entry
                    let entry = tool_map
                        .entry(tool_use_id.to_string())
                        .or_insert_with(|| (String::new(), String::new()));

                    // 更新 name（如果有）
                    if !name.is_empty() {
                        entry.0 = name;
                    }

                    // 累积 input
                    entry.1.push_str(&input_chunk);

                    // 如果是 stop 事件，完成这个 tool call
                    if is_stop {
                        if let Some((name, input)) = tool_map.remove(tool_use_id) {
                            if !name.is_empty() {
                                result.tool_calls.push(ToolCall {
                                    id: tool_use_id.to_string(),
                                    call_type: "function".to_string(),
                                    function: FunctionCall {
                                        name,
                                        arguments: input,
                                    },
                                });
                            }
                        }
                    }
                }
                // 处理独立的 stop 事件（没有 toolUseId）
                else if value.get("stop").and_then(|v| v.as_bool()).unwrap_or(false) {
                    // 这种情况不应该发生，但以防万一
                }
                // 处理 meteringEvent: {"unit":"credit","unitPlural":"credits","usage":0.34}
                else if let Some(usage) = value.get("usage").and_then(|v| v.as_f64()) {
                    result.usage_credits = usage;
                }
                // 处理 contextUsageEvent: {"contextUsagePercentage":54.36}
                else if let Some(ctx_usage) =
                    value.get("contextUsagePercentage").and_then(|v| v.as_f64())
                {
                    result.context_usage_percentage = ctx_usage;
                }
            }
            pos = start + json_str.len();
        } else {
            pos = start + 1;
        }
    }

    // 处理未完成的 tool calls（没有收到 stop 事件的）
    for (id, (name, input)) in tool_map {
        if !name.is_empty() {
            result.tool_calls.push(ToolCall {
                id,
                call_type: "function".to_string(),
                function: FunctionCall {
                    name,
                    arguments: input,
                },
            });
        }
    }

    // 解析 bracket 格式的 tool calls: [Called xxx with args: {...}]
    parse_bracket_tool_calls(&mut result);

    result
}

/// 在字节数组中查找子序列
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// 从字节数组中提取 JSON 对象字符串
fn extract_json_from_bytes(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() || bytes[0] != b'{' {
        return None;
    }

    let mut brace_count = 0;
    let mut in_string = false;
    let mut escape_next = false;
    let mut end_pos = None;

    for (i, &b) in bytes.iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match b {
            b'\\' if in_string => escape_next = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => brace_count += 1,
            b'}' if !in_string => {
                brace_count -= 1;
                if brace_count == 0 {
                    end_pos = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }

    end_pos.and_then(|end| String::from_utf8(bytes[..end].to_vec()).ok())
}

/// 从字符串中提取完整的 JSON 对象 (保留用于兼容)
#[allow(dead_code)]
fn extract_json_object(s: &str) -> Option<&str> {
    if !s.starts_with('{') {
        return None;
    }

    let mut brace_count = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, c) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match c {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => brace_count += 1,
            '}' if !in_string => {
                brace_count -= 1;
                if brace_count == 0 {
                    return Some(&s[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// 解析 bracket 格式的 tool calls
fn parse_bracket_tool_calls(result: &mut CWParsedResponse) {
    let re =
        regex::Regex::new(r"\[Called\s+(\w+)\s+with\s+args:\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\})\]")
            .ok();

    if let Some(re) = re {
        let mut to_remove = Vec::new();
        for cap in re.captures_iter(&result.content) {
            if let (Some(name), Some(args)) = (cap.get(1), cap.get(2)) {
                let tool_id = format!(
                    "call_{}",
                    &uuid::Uuid::new_v4().to_string().replace('-', "")[..8]
                );
                result.tool_calls.push(ToolCall {
                    id: tool_id,
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.as_str().to_string(),
                        arguments: args.as_str().to_string(),
                    },
                });
                if let Some(full_match) = cap.get(0) {
                    to_remove.push(full_match.as_str().to_string());
                }
            }
        }
        // 从 content 中移除 tool call 文本
        for s in to_remove {
            result.content = result.content.replace(&s, "");
        }
        result.content = result.content.trim().to_string();
    }
}

/// 列出所有可用路由
async fn list_routes(State(state): State<AppState>) -> impl IntoResponse {
    let routes = match &state.db {
        Some(db) => state
            .pool_service
            .get_available_routes(db, &state.base_url)
            .unwrap_or_default(),
        None => Vec::new(),
    };

    // 添加默认路由
    let mut all_routes = vec![RouteInfo {
        selector: "default".to_string(),
        provider_type: "kiro".to_string(),
        credential_count: 1,
        endpoints: vec![
            crate::models::route_model::RouteEndpoint {
                path: "/v1/messages".to_string(),
                protocol: "claude".to_string(),
                url: format!("{}/v1/messages", state.base_url),
            },
            crate::models::route_model::RouteEndpoint {
                path: "/v1/chat/completions".to_string(),
                protocol: "openai".to_string(),
                url: format!("{}/v1/chat/completions", state.base_url),
            },
        ],
        tags: vec!["默认".to_string()],
        enabled: true,
    }];
    all_routes.extend(routes);

    let response = RouteListResponse {
        base_url: state.base_url.clone(),
        default_provider: "kiro".to_string(),
        routes: all_routes,
    };

    Json(response)
}

/// 带选择器的 Anthropic messages 处理
async fn anthropic_messages_with_selector(
    State(state): State<AppState>,
    Path(selector): Path<String>,
    headers: HeaderMap,
    Json(request): Json<AnthropicMessagesRequest>,
) -> Response {
    // 使用 Anthropic 格式的认证验证
    if let Err(e) = verify_api_key_anthropic(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!("Unauthorized request to /{}/v1/messages", selector),
        );
        return e.into_response();
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[REQ] POST /{}/v1/messages model={} stream={}",
            selector, request.model, request.stream
        ),
    );

    // 尝试解析凭证
    let credential = match &state.db {
        Some(db) => {
            // 首先尝试按名称查找
            if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &selector) {
                Some(cred)
            }
            // 然后尝试按 UUID 查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &selector) {
                Some(cred)
            }
            // 最后尝试按 provider 类型轮询
            else if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &selector, Some(&request.model))
            {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[ROUTE] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );

            // 根据凭证类型调用相应的 Provider
            call_provider_anthropic(&state, &cred, &request).await
        }
        None => {
            // 回退到默认 Kiro provider
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[ROUTE] Credential not found for selector '{}', falling back to default",
                    selector
                ),
            );
            // 调用原有的 Kiro 处理逻辑
            anthropic_messages_internal(&state, &request).await
        }
    }
}

/// 带选择器的 OpenAI chat completions 处理
async fn chat_completions_with_selector(
    State(state): State<AppState>,
    Path(selector): Path<String>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!("Unauthorized request to /{}/v1/chat/completions", selector),
        );
        return e.into_response();
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[REQ] POST /{}/v1/chat/completions model={} stream={}",
            selector, request.model, request.stream
        ),
    );

    // 尝试解析凭证
    let credential = match &state.db {
        Some(db) => {
            if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &selector) {
                Some(cred)
            } else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &selector) {
                Some(cred)
            } else if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &selector, Some(&request.model))
            {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[ROUTE] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );

            call_provider_openai(&state, &cred, &request).await
        }
        None => {
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[ROUTE] Credential not found for selector '{}', falling back to default",
                    selector
                ),
            );
            chat_completions_internal(&state, &request).await
        }
    }
}

// ============ Amp CLI 路由处理 ============

/// Amp CLI chat completions 处理
///
/// 处理 `/api/provider/:provider/v1/chat/completions` 路由
/// 支持模型映射，将不可用模型映射到可用替代
async fn amp_chat_completions(
    State(state): State<AppState>,
    Path(provider): Path<String>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!(
                "Unauthorized request to /api/provider/{}/v1/chat/completions",
                provider
            ),
        );
        return e.into_response();
    }

    // 应用模型映射
    let original_model = request.model.clone();
    let mapped_model = state.amp_router.apply_model_mapping(&request.model);
    if mapped_model != original_model {
        state.logs.write().await.add(
            "info",
            &format!(
                "[AMP] Model mapping applied: {} -> {}",
                original_model, mapped_model
            ),
        );
        request.model = mapped_model;
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[AMP] POST /api/provider/{}/v1/chat/completions model={} stream={}",
            provider, request.model, request.stream
        ),
    );

    // 尝试根据 provider 名称选择凭证
    let credential = match &state.db {
        Some(db) => {
            // 首先尝试按 provider 类型选择
            if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &provider, Some(&request.model))
            {
                Some(cred)
            }
            // 然后尝试按名称查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &provider) {
                Some(cred)
            }
            // 最后尝试按 UUID 查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &provider) {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[AMP] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );
            call_provider_openai(&state, &cred, &request).await
        }
        None => {
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[AMP] Credential not found for provider '{}', falling back to default",
                    provider
                ),
            );
            chat_completions_internal(&state, &request).await
        }
    }
}

/// Amp CLI messages 处理
///
/// 处理 `/api/provider/:provider/v1/messages` 路由
/// 支持模型映射，将不可用模型映射到可用替代
async fn amp_messages(
    State(state): State<AppState>,
    Path(provider): Path<String>,
    headers: HeaderMap,
    Json(mut request): Json<AnthropicMessagesRequest>,
) -> Response {
    // 使用 Anthropic 格式的认证验证
    if let Err(e) = verify_api_key_anthropic(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!(
                "Unauthorized request to /api/provider/{}/v1/messages",
                provider
            ),
        );
        return e.into_response();
    }

    // 应用模型映射
    let original_model = request.model.clone();
    let mapped_model = state.amp_router.apply_model_mapping(&request.model);
    if mapped_model != original_model {
        state.logs.write().await.add(
            "info",
            &format!(
                "[AMP] Model mapping applied: {} -> {}",
                original_model, mapped_model
            ),
        );
        request.model = mapped_model;
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[AMP] POST /api/provider/{}/v1/messages model={} stream={}",
            provider, request.model, request.stream
        ),
    );

    // 尝试根据 provider 名称选择凭证
    let credential = match &state.db {
        Some(db) => {
            // 首先尝试按 provider 类型选择
            if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &provider, Some(&request.model))
            {
                Some(cred)
            }
            // 然后尝试按名称查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &provider) {
                Some(cred)
            }
            // 最后尝试按 UUID 查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &provider) {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[AMP] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );
            call_provider_anthropic(&state, &cred, &request).await
        }
        None => {
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[AMP] Credential not found for provider '{}', falling back to default",
                    provider
                ),
            );
            anthropic_messages_internal(&state, &request).await
        }
    }
}

/// Amp CLI 管理代理 - auth 路由
///
/// 处理 `/api/auth/*` 路由，将请求代理到上游 URL
async fn amp_management_proxy_auth(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> Response {
    amp_management_proxy_internal(state, &format!("auth/{}", path), headers, method, body).await
}

/// Amp CLI 管理代理 - user 路由
///
/// 处理 `/api/user/*` 路由，将请求代理到上游 URL
async fn amp_management_proxy_user(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> Response {
    amp_management_proxy_internal(state, &format!("user/{}", path), headers, method, body).await
}

/// Amp CLI 管理代理内部实现
///
/// 处理 `/api/auth/*` 和 `/api/user/*` 路由
/// 将请求代理到上游 URL
///
/// # 参数
/// - `path`: 请求路径（不含 /api/ 前缀，如 "auth/login" 或 "user/profile"）
async fn amp_management_proxy_internal(
    state: AppState,
    path: &str,
    headers: HeaderMap,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> Response {
    let full_path = format!("/api/{}", path);

    // 检查是否是管理路由
    if !state.amp_router.is_management_route(&full_path) {
        state.logs.write().await.add(
            "warn",
            &format!("[AMP] Invalid management route: {}", full_path),
        );
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"message": "Not found"}})),
        )
            .into_response();
    }

    // 检查 localhost 限制
    if state.amp_router.restrict_management_to_localhost() {
        // 从 headers 中获取客户端 IP
        let client_ip = headers
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(',').next().unwrap_or("").trim().to_string())
            .or_else(|| {
                headers
                    .get("x-real-ip")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string())
            });

        if let Some(ip) = &client_ip {
            let is_localhost = ip == "127.0.0.1" || ip == "::1" || ip == "localhost";
            if !is_localhost {
                state.logs.write().await.add(
                    "warn",
                    &format!("[AMP] Management proxy blocked from non-localhost: {}", ip),
                );
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({"error": {"message": "Management endpoints are restricted to localhost"}})),
                )
                    .into_response();
            }
        }
    }

    // 获取上游 URL
    let upstream_url = match state.amp_router.get_management_upstream_path(&full_path) {
        Some(url) => url,
        None => {
            state.logs.write().await.add(
                "warn",
                &format!("[AMP] No upstream URL configured for management proxy"),
            );
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": {"message": "Upstream URL not configured"}})),
            )
                .into_response();
        }
    };

    state.logs.write().await.add(
        "info",
        &format!(
            "[AMP] Proxying management request: {} {} -> {}",
            method, full_path, upstream_url
        ),
    );

    // 创建 HTTP 客户端
    let client = reqwest::Client::new();

    // 构建请求
    let mut request_builder = match method {
        axum::http::Method::GET => client.get(&upstream_url),
        axum::http::Method::POST => client.post(&upstream_url),
        axum::http::Method::PUT => client.put(&upstream_url),
        axum::http::Method::DELETE => client.delete(&upstream_url),
        axum::http::Method::PATCH => client.patch(&upstream_url),
        axum::http::Method::HEAD => client.head(&upstream_url),
        axum::http::Method::OPTIONS => client.request(reqwest::Method::OPTIONS, &upstream_url),
        _ => {
            return (
                StatusCode::METHOD_NOT_ALLOWED,
                Json(serde_json::json!({"error": {"message": "Method not allowed"}})),
            )
                .into_response();
        }
    };

    // 复制请求头（排除 host 和 content-length）
    for (name, value) in headers.iter() {
        let name_str = name.as_str().to_lowercase();
        if name_str != "host" && name_str != "content-length" {
            if let Ok(value_str) = value.to_str() {
                request_builder = request_builder.header(name.as_str(), value_str);
            }
        }
    }

    // 添加请求体
    if !body.is_empty() {
        request_builder = request_builder.body(body.to_vec());
    }

    // 发送请求
    match request_builder.send().await {
        Ok(response) => {
            let status = response.status();
            let response_headers = response.headers().clone();

            match response.bytes().await {
                Ok(response_body) => {
                    let mut builder = Response::builder().status(status.as_u16());

                    // 复制响应头
                    for (name, value) in response_headers.iter() {
                        let name_str = name.as_str().to_lowercase();
                        // 排除 transfer-encoding 和 content-length（axum 会自动处理）
                        if name_str != "transfer-encoding" && name_str != "content-length" {
                            builder = builder.header(name.as_str(), value.to_str().unwrap_or(""));
                        }
                    }

                    builder
                        .body(Body::from(response_body.to_vec()))
                        .unwrap_or_else(|_| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": "Failed to build response"}})),
                            )
                                .into_response()
                        })
                }
                Err(e) => {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AMP] Failed to read upstream response: {}", e),
                    );
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(serde_json::json!({"error": {"message": format!("Failed to read upstream response: {}", e)}})),
                    )
                        .into_response()
                }
            }
        }
        Err(e) => {
            state.logs.write().await.add(
                "error",
                &format!("[AMP] Failed to proxy request to upstream: {}", e),
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {"message": format!("Failed to connect to upstream: {}", e)}})),
            )
                .into_response()
        }
    }
}

/// 内部 Anthropic messages 处理 (使用默认 Kiro)
async fn anthropic_messages_internal(
    state: &AppState,
    request: &AnthropicMessagesRequest,
) -> Response {
    // 检查 token
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                )
                    .into_response();
            }
        }
    }

    let openai_request = convert_anthropic_to_openai(request);
    let kiro = state.kiro.read().await;

    match kiro.call_api(&openai_request).await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        let body = String::from_utf8_lossy(&bytes).to_string();
                        let parsed = parse_cw_response(&body);
                        if request.stream {
                            build_anthropic_stream_response(&request.model, &parsed)
                        } else {
                            build_anthropic_response(&request.model, &parsed)
                        }
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response(),
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}})),
                )
                    .into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
            .into_response(),
    }
}

/// 内部 OpenAI chat completions 处理 (使用默认 Kiro)
async fn chat_completions_internal(state: &AppState, request: &ChatCompletionRequest) -> Response {
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            if let Err(e) = kiro.refresh_token().await {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                )
                    .into_response();
            }
        }
    }

    let kiro = state.kiro.read().await;
    match kiro.call_api(request).await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.text().await {
                    Ok(body) => {
                        let parsed = parse_cw_response(&body);
                        let has_tool_calls = !parsed.tool_calls.is_empty();

                        let message = if has_tool_calls {
                            serde_json::json!({
                                "role": "assistant",
                                "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                    serde_json::json!({
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    })
                                }).collect::<Vec<_>>()
                            })
                        } else {
                            serde_json::json!({
                                "role": "assistant",
                                "content": parsed.content
                            })
                        };

                        let response = serde_json::json!({
                            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                            "object": "chat.completion",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "message": message,
                                "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        });
                        Json(response).into_response()
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response(),
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}})),
                )
                    .into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
            .into_response(),
    }
}

use crate::models::provider_pool_model::{CredentialData, ProviderCredential};

/// 根据凭证调用 Provider (Anthropic 格式)
async fn call_provider_anthropic(
    state: &AppState,
    credential: &ProviderCredential,
    request: &AnthropicMessagesRequest,
) -> Response {
    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            // 使用 TokenCacheService 获取有效 token
            let db = match &state.db {
                Some(db) => db,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": "Database not available"}})),
                    )
                        .into_response();
                }
            };

            // 获取缓存的 token
            let token = match state
                .token_cache
                .get_valid_token(db, &credential.uuid)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("[POOL] Token cache miss, loading from source: {}", e);
                    // 回退到从源文件加载
                    let mut kiro = KiroProvider::new();
                    if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                        // 记录凭证加载失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Failed to load credentials: {}", e)),
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                        )
                            .into_response();
                    }
                    if let Err(e) = kiro.refresh_token().await {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                    kiro.credentials.access_token.unwrap_or_default()
                }
            };

            // 使用获取到的 token 创建 KiroProvider
            let mut kiro = KiroProvider::new();
            kiro.credentials.access_token = Some(token);
            // 从源文件加载其他配置（region, profile_arn 等）
            let _ = kiro.load_credentials_from_path(creds_file_path).await;

            let openai_request = convert_anthropic_to_openai(request);
            let resp = match kiro.call_api(&openai_request).await {
                Ok(r) => r,
                Err(e) => {
                    // 记录 API 调用失败
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&e.to_string()),
                    );
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response();
                }
            };

            let status = resp.status();
            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        let body = String::from_utf8_lossy(&bytes).to_string();
                        let parsed = parse_cw_response(&body);
                        // 记录成功
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                        if request.stream {
                            build_anthropic_stream_response(&request.model, &parsed)
                        } else {
                            build_anthropic_response(&request.model, &parsed)
                        }
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                // Token 过期，强制刷新并重试
                tracing::info!(
                    "[POOL] Got {}, forcing token refresh for {}",
                    status,
                    &credential.uuid[..8]
                );

                let new_token = match state
                    .token_cache
                    .refresh_and_cache(db, &credential.uuid, true)
                    .await
                {
                    Ok(t) => t,
                    Err(e) => {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                };

                // 使用新 token 重试
                kiro.credentials.access_token = Some(new_token);
                match kiro.call_api(&openai_request).await {
                    Ok(retry_resp) => {
                        if retry_resp.status().is_success() {
                            match retry_resp.bytes().await {
                                Ok(bytes) => {
                                    let body = String::from_utf8_lossy(&bytes).to_string();
                                    let parsed = parse_cw_response(&body);
                                    // 记录重试成功
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                    if request.stream {
                                        build_anthropic_stream_response(&request.model, &parsed)
                                    } else {
                                        build_anthropic_response(&request.model, &parsed)
                                    }
                                }
                                Err(e) => {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                    )
                                        .into_response()
                                }
                            }
                        } else {
                            let body = retry_resp.text().await.unwrap_or_default();
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&format!("Retry failed: {}", body)),
                            );
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                            )
                                .into_response()
                        }
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                let _ = state
                    .pool_service
                    .mark_unhealthy(db, &credential.uuid, Some(&body));
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": body}})),
                )
                    .into_response()
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            // Gemini OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented. Use /v1/messages with Gemini models instead."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            // Qwen OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented. Use /v1/messages with Qwen models instead."}})),
            )
                .into_response()
        }
        CredentialData::AntigravityOAuth {
            creds_file_path,
            project_id,
        } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity
                .load_credentials_from_path(creds_file_path)
                .await
            {
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }

            // 检查并刷新 token
            if antigravity.is_token_expiring_soon() {
                if let Err(e) = antigravity.refresh_token().await {
                    // 记录 Token 刷新失败
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                    }
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                    )
                        .into_response();
                }
            }

            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }

            // 先转换为 OpenAI 格式，再转换为 Antigravity 格式
            let openai_request = convert_anthropic_to_openai(request);
            let antigravity_request = convert_openai_to_antigravity(&openai_request);

            match antigravity
                .generate_content(&request.model, &antigravity_request)
                .await
            {
                Ok(resp) => {
                    // 转换为 OpenAI 格式，再构建 Anthropic 响应
                    let content = resp["candidates"][0]["content"]["parts"][0]["text"]
                        .as_str()
                        .unwrap_or("");
                    let parsed = CWParsedResponse {
                        content: content.to_string(),
                        tool_calls: Vec::new(),
                        usage_credits: 0.0,
                        context_usage_percentage: 0.0,
                    };
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    if request.stream {
                        build_anthropic_stream_response(&request.model, &parsed)
                    } else {
                        build_anthropic_response(&request.model, &parsed)
                    }
                }
                Err(e) => {
                    // 记录 API 调用失败
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            let openai_request = convert_anthropic_to_openai(request);
            match openai.call_api(&openai_request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(openai_resp) =
                                    serde_json::from_str::<serde_json::Value>(&body)
                                {
                                    let content = openai_resp["choices"][0]["message"]["content"]
                                        .as_str()
                                        .unwrap_or("");
                                    let parsed = CWParsedResponse {
                                        content: content.to_string(),
                                        tool_calls: Vec::new(),
                                        usage_credits: 0.0,
                                        context_usage_percentage: 0.0,
                                    };
                                    // 记录成功
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_healthy(
                                            db,
                                            &credential.uuid,
                                            Some(&request.model),
                                        );
                                        let _ =
                                            state.pool_service.record_usage(db, &credential.uuid);
                                    }
                                    if request.stream {
                                        build_anthropic_stream_response(&request.model, &parsed)
                                    } else {
                                        build_anthropic_response(&request.model, &parsed)
                                    }
                                } else {
                                    // 记录解析失败
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_unhealthy(
                                            db,
                                            &credential.uuid,
                                            Some("Failed to parse OpenAI response"),
                                        );
                                    }
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Failed to parse OpenAI response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&body),
                            );
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            let request_url = claude.get_base_url();
            state.logs.write().await.add(
                "info",
                &format!(
                    "[CLAUDE] 使用 Claude API 代理: base_url={} -> {}/v1/messages credential_uuid={}",
                    actual_base_url,
                    request_url,
                    &credential.uuid[..8]
                ),
            );
            // 打印请求参数
            let request_json = serde_json::to_string(request).unwrap_or_default();
            state.logs.write().await.add(
                "debug",
                &format!(
                    "[CLAUDE] 请求参数: {}",
                    &request_json.chars().take(500).collect::<String>()
                ),
            );
            match claude.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    // 打印响应状态
                    state.logs.write().await.add(
                        "info",
                        &format!(
                            "[CLAUDE] 响应状态: status={} model={}",
                            status,
                            request.model
                        ),
                    );
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                // 打印响应内容预览
                                state.logs.write().await.add(
                                    "debug",
                                    &format!(
                                        "[CLAUDE] 响应内容: {}",
                                        &body.chars().take(500).collect::<String>()
                                    ),
                                );
                                // 记录成功
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({"error": {"message": "Failed to build response"}})),
                                        )
                                            .into_response()
                                    })
                            } else {
                                state.logs.write().await.add(
                                    "error",
                                    &format!(
                                        "[CLAUDE] 请求失败: status={} body={}",
                                        status,
                                        &body.chars().take(200).collect::<String>()
                                    ),
                                );
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&body),
                                    );
                                }
                                (
                                    StatusCode::from_u16(status.as_u16())
                                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                                    Json(serde_json::json!({"error": {"message": body}})),
                                )
                                    .into_response()
                            }
                        }
                        Err(e) => {
                            state.logs.write().await.add(
                                "error",
                                &format!("[CLAUDE] 读取响应失败: {}", e),
                            );
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(
                                    db,
                                    &credential.uuid,
                                    Some(&e.to_string()),
                                );
                            }
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response()
                        }
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::VertexKey { api_key, base_url, .. } => {
            // Vertex AI uses Gemini-compatible API, convert Anthropic to OpenAI format first
            let openai_request = convert_anthropic_to_openai(request);
            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&openai_request).unwrap_or_default()).await {
                Ok(resp) => {
                    let status = resp.status();
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Failed to build response"}}))).into_response()
                                    })
                            } else {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&body));
                                }
                                (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(serde_json::json!({"error": {"message": body}}))).into_response()
                            }
                        }
                        Err(e) => {
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                            }
                            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                        }
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                }
            }
        }
        // Gemini API Key credentials - not supported for Anthropic format
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support Anthropic format"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 Anthropic 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support Anthropic format yet"}})),
            )
                .into_response()
        }
    }
}

/// 根据凭证调用 Provider (OpenAI 格式)
async fn call_provider_openai(
    state: &AppState,
    credential: &ProviderCredential,
    request: &ChatCompletionRequest,
) -> Response {
    let start_time = std::time::Instant::now();

    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            let mut kiro = KiroProvider::new();
            if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                )
                    .into_response();
            }
            if let Err(e) = kiro.refresh_token().await {
                // 记录 Token 刷新失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&format!("Token refresh failed: {}", e)));
                }
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                )
                    .into_response();
            }

            match kiro.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // 记录成功
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                            let _ = state.pool_service.record_usage(db, &credential.uuid);
                        }
                        match resp.text().await {
                            Ok(body) => {
                                let parsed = parse_cw_response(&body);
                                let has_tool_calls = !parsed.tool_calls.is_empty();

                                let message = if has_tool_calls {
                                    serde_json::json!({
                                        "role": "assistant",
                                        "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                        "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                            serde_json::json!({
                                                "id": tc.id,
                                                "type": "function",
                                                "function": {
                                                    "name": tc.function.name,
                                                    "arguments": tc.function.arguments
                                                }
                                            })
                                        }).collect::<Vec<_>>()
                                    })
                                } else {
                                    serde_json::json!({
                                        "role": "assistant",
                                        "content": parsed.content
                                    })
                                };

                                Json(serde_json::json!({
                                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                    "object": "chat.completion",
                                    "created": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs(),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "message": message,
                                        "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                                    }],
                                    "usage": {
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "total_tokens": 0
                                    }
                                }))
                                .into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        // 记录 API 调用失败
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&format!("HTTP {}: {}", status, safe_truncate(&body, 100))));
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    // 记录请求错误
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::AntigravityOAuth { creds_file_path, project_id } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity.load_credentials_from_path(creds_file_path).await {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }

            // 检查并刷新 token
            if antigravity.is_token_expiring_soon() {
                if let Err(e) = antigravity.refresh_token().await {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                    )
                        .into_response();
                }
            }

            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }

            // 转换请求格式
            let antigravity_request = convert_openai_to_antigravity(request);

            match antigravity.generate_content(&request.model, &antigravity_request).await {
                Ok(resp) => {
                    let openai_response = convert_antigravity_to_openai_response(&resp, &request.model);
                    Json(openai_response).into_response()
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            match openai.call_api(request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Invalid JSON response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={}",
                actual_base_url,
                &credential.uuid[..8]
            );
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            match claude.call_openai_api(request).await {
                Ok(resp) => Json(resp).into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::VertexKey { api_key, base_url, model_aliases } => {
            // Resolve model alias if present
            let resolved_model = model_aliases.get(&request.model).cloned().unwrap_or_else(|| request.model.clone());
            let mut modified_request = request.clone();
            modified_request.model = resolved_model;

            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&modified_request).unwrap_or_default()).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Invalid JSON response"}}))).into_response()
                                }
                            }
                            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": body}}))).into_response()
                    }
                }
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
            }
        }
        // Gemini API Key credentials - not supported for OpenAI format yet
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support OpenAI format yet"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 OpenAI 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support OpenAI format yet"}})),
            )
                .into_response()
        }
    }
}

// ========== WebSocket 处理 ==========

use crate::websocket::{
    WsApiRequest, WsApiResponse, WsEndpoint, WsError, WsMessage as WsProtoMessage,
};
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use futures::{SinkExt, StreamExt as FuturesStreamExt};

/// WebSocket 升级处理器
async fn ws_upgrade_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // 验证 API 密钥
    let auth = headers
        .get("authorization")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return axum::http::Response::builder()
                .status(401)
                .body(Body::from("No API key provided"))
                .unwrap()
                .into_response();
        }
    };

    if key != state.api_key {
        return axum::http::Response::builder()
            .status(401)
            .body(Body::from("Invalid API key"))
            .unwrap()
            .into_response();
    }

    // 获取客户端信息
    let client_info = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    ws.on_upgrade(move |socket| handle_websocket(socket, state, client_info))
}

/// 处理 WebSocket 连接
async fn handle_websocket(socket: WebSocket, state: AppState, client_info: Option<String>) {
    let conn_id = uuid::Uuid::new_v4().to_string();

    // 注册连接
    if let Err(e) = state
        .ws_manager
        .register(conn_id.clone(), client_info.clone())
    {
        state.logs.write().await.add(
            "error",
            &format!("[WS] Failed to register connection: {}", e.message),
        );
        return;
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[WS] New connection: {} (client: {:?})",
            &conn_id[..8],
            client_info
        ),
    );

    let (mut sender, mut receiver) = socket.split();

    // 消息处理循环
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(WsMessage::Text(text)) => {
                state.ws_manager.on_message();
                state.ws_manager.increment_request_count(&conn_id);

                match serde_json::from_str::<WsProtoMessage>(&text) {
                    Ok(ws_msg) => {
                        let response = handle_ws_message(&state, &conn_id, ws_msg).await;
                        if let Some(resp) = response {
                            let resp_text = serde_json::to_string(&resp).unwrap_or_default();
                            if sender
                                .send(WsMessage::Text(resp_text.into()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        state.ws_manager.on_error();
                        let error = WsProtoMessage::Error(WsError::invalid_message(format!(
                            "Failed to parse message: {}",
                            e
                        )));
                        let error_text = serde_json::to_string(&error).unwrap_or_default();
                        if sender
                            .send(WsMessage::Text(error_text.into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
            Ok(WsMessage::Binary(_)) => {
                state.ws_manager.on_error();
                let error = WsProtoMessage::Error(WsError::invalid_message(
                    "Binary messages not supported",
                ));
                let error_text = serde_json::to_string(&error).unwrap_or_default();
                if sender
                    .send(WsMessage::Text(error_text.into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Ok(WsMessage::Ping(data)) => {
                if sender.send(WsMessage::Pong(data)).await.is_err() {
                    break;
                }
            }
            Ok(WsMessage::Pong(_)) => {
                // 收到 pong，连接正常
            }
            Ok(WsMessage::Close(_)) => {
                break;
            }
            Err(e) => {
                state.logs.write().await.add(
                    "error",
                    &format!("[WS] Connection {} error: {}", &conn_id[..8], e),
                );
                break;
            }
        }
    }

    // 清理连接
    state.ws_manager.unregister(&conn_id);
    state.logs.write().await.add(
        "info",
        &format!("[WS] Connection closed: {}", &conn_id[..8]),
    );
}

/// 处理 WebSocket 消息
async fn handle_ws_message(
    state: &AppState,
    conn_id: &str,
    msg: WsProtoMessage,
) -> Option<WsProtoMessage> {
    match msg {
        WsProtoMessage::Ping { timestamp } => Some(WsProtoMessage::Pong { timestamp }),
        WsProtoMessage::Pong { .. } => None,
        WsProtoMessage::Request(request) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[WS] Request from {}: id={} endpoint={:?}",
                    &conn_id[..8],
                    request.request_id,
                    request.endpoint
                ),
            );

            // 处理 API 请求
            let response = handle_ws_api_request(state, &request).await;
            Some(response)
        }
        WsProtoMessage::Response(_)
        | WsProtoMessage::StreamChunk(_)
        | WsProtoMessage::StreamEnd(_) => Some(WsProtoMessage::Error(WsError::invalid_request(
            None,
            "Invalid message type from client",
        ))),
        WsProtoMessage::Error(_) => None,
    }
}

/// 处理 WebSocket API 请求
async fn handle_ws_api_request(state: &AppState, request: &WsApiRequest) -> WsProtoMessage {
    match request.endpoint {
        WsEndpoint::Models => {
            // 返回模型列表
            let models = serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "claude-sonnet-4-5", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-3-7-sonnet-20250219", "object": "model", "owned_by": "anthropic"},
                    {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
                    {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
                    {"id": "qwen3-coder-plus", "object": "model", "owned_by": "alibaba"},
                ]
            });
            WsProtoMessage::Response(WsApiResponse {
                request_id: request.request_id.clone(),
                payload: models,
            })
        }
        WsEndpoint::ChatCompletions => {
            // 解析 ChatCompletionRequest
            match serde_json::from_value::<ChatCompletionRequest>(request.payload.clone()) {
                Ok(chat_request) => {
                    handle_ws_chat_completions(state, &request.request_id, chat_request).await
                }
                Err(e) => WsProtoMessage::Error(WsError::invalid_request(
                    Some(request.request_id.clone()),
                    format!("Invalid chat completion request: {}", e),
                )),
            }
        }
        WsEndpoint::Messages => {
            // 解析 AnthropicMessagesRequest
            match serde_json::from_value::<AnthropicMessagesRequest>(request.payload.clone()) {
                Ok(messages_request) => {
                    handle_ws_anthropic_messages(state, &request.request_id, messages_request).await
                }
                Err(e) => WsProtoMessage::Error(WsError::invalid_request(
                    Some(request.request_id.clone()),
                    format!("Invalid messages request: {}", e),
                )),
            }
        }
    }
}

/// 处理 WebSocket chat completions 请求
async fn handle_ws_chat_completions(
    state: &AppState,
    request_id: &str,
    mut request: ChatCompletionRequest,
) -> WsProtoMessage {
    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    // 使用 RequestProcessor 解析模型别名和路由
    let _provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取默认 provider
    let default_provider = state.default_provider.read().await.clone();

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => state
            .pool_service
            .select_credential(db, &default_provider, Some(&request.model))
            .ok()
            .flatten(),
        None => None,
    };

    // 如果找到凭证，使用它调用 API
    if let Some(cred) = credential {
        // 简化实现：直接调用 provider 并返回结果
        // 实际实现应该复用 call_provider_openai 的逻辑
        match call_provider_openai_for_ws(state, &cred, &request).await {
            Ok(response) => WsProtoMessage::Response(WsApiResponse {
                request_id: request_id.to_string(),
                payload: response,
            }),
            Err(e) => WsProtoMessage::Error(WsError::upstream(Some(request_id.to_string()), e)),
        }
    } else {
        // 回退到 Kiro provider
        let kiro = state.kiro.read().await;
        match kiro.call_api(&request).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text().await {
                        Ok(body) => {
                            let parsed = parse_cw_response(&body);
                            let has_tool_calls = !parsed.tool_calls.is_empty();

                            let message = if has_tool_calls {
                                serde_json::json!({
                                    "role": "assistant",
                                    "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                    "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                        serde_json::json!({
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments
                                            }
                                        })
                                    }).collect::<Vec<_>>()
                                })
                            } else {
                                serde_json::json!({
                                    "role": "assistant",
                                    "content": parsed.content
                                })
                            };

                            let response = serde_json::json!({
                                "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                "object": "chat.completion",
                                "created": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "message": message,
                                    "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                                }],
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0
                                }
                            });

                            WsProtoMessage::Response(WsApiResponse {
                                request_id: request_id.to_string(),
                                payload: response,
                            })
                        }
                        Err(e) => WsProtoMessage::Error(WsError::internal(
                            Some(request_id.to_string()),
                            e.to_string(),
                        )),
                    }
                } else {
                    let body = resp.text().await.unwrap_or_default();
                    WsProtoMessage::Error(WsError::upstream(
                        Some(request_id.to_string()),
                        format!("Upstream error: {}", body),
                    ))
                }
            }
            Err(e) => WsProtoMessage::Error(WsError::internal(
                Some(request_id.to_string()),
                e.to_string(),
            )),
        }
    }
}

/// 处理 WebSocket anthropic messages 请求
async fn handle_ws_anthropic_messages(
    state: &AppState,
    request_id: &str,
    mut request: AnthropicMessagesRequest,
) -> WsProtoMessage {
    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    // 使用 RequestProcessor 解析模型别名和路由
    let _provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取默认 provider
    let default_provider = state.default_provider.read().await.clone();

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => state
            .pool_service
            .select_credential(db, &default_provider, Some(&request.model))
            .ok()
            .flatten(),
        None => None,
    };

    // 如果找到凭证，使用它调用 API
    if let Some(cred) = credential {
        match call_provider_anthropic_for_ws(state, &cred, &request).await {
            Ok(response) => WsProtoMessage::Response(WsApiResponse {
                request_id: request_id.to_string(),
                payload: response,
            }),
            Err(e) => WsProtoMessage::Error(WsError::upstream(Some(request_id.to_string()), e)),
        }
    } else {
        // 回退到 Kiro provider
        let kiro = state.kiro.read().await;

        // 转换为 OpenAI 格式
        let openai_request = convert_anthropic_to_openai(&request);

        match kiro.call_api(&openai_request).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text().await {
                        Ok(body) => {
                            let parsed = parse_cw_response(&body);

                            // 转换为 Anthropic 格式响应
                            let response = serde_json::json!({
                                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "text",
                                    "text": parsed.content
                                }],
                                "model": request.model,
                                "stop_reason": "end_turn",
                                "usage": {
                                    "input_tokens": 0,
                                    "output_tokens": 0
                                }
                            });

                            WsProtoMessage::Response(WsApiResponse {
                                request_id: request_id.to_string(),
                                payload: response,
                            })
                        }
                        Err(e) => WsProtoMessage::Error(WsError::internal(
                            Some(request_id.to_string()),
                            e.to_string(),
                        )),
                    }
                } else {
                    let body = resp.text().await.unwrap_or_default();
                    WsProtoMessage::Error(WsError::upstream(
                        Some(request_id.to_string()),
                        format!("Upstream error: {}", body),
                    ))
                }
            }
            Err(e) => WsProtoMessage::Error(WsError::internal(
                Some(request_id.to_string()),
                e.to_string(),
            )),
        }
    }
}

/// WebSocket 专用的 OpenAI 格式 Provider 调用
async fn call_provider_openai_for_ws(
    state: &AppState,
    credential: &ProviderCredential,
    request: &ChatCompletionRequest,
) -> Result<serde_json::Value, String> {
    use crate::models::provider_pool_model::CredentialData;

    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            let mut kiro = KiroProvider::new();
            if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return Err(e.to_string());
            }
            if let Err(e) = kiro.refresh_token().await {
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Token refresh failed: {}", e)),
                    );
                }
                return Err(e.to_string());
            }

            let resp = match kiro.call_api(request).await {
                Ok(r) => r,
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    return Err(e.to_string());
                }
            };
            if resp.status().is_success() {
                let body = resp.text().await.map_err(|e| e.to_string())?;
                let parsed = parse_cw_response(&body);
                let has_tool_calls = !parsed.tool_calls.is_empty();

                // 记录成功
                if let Some(db) = &state.db {
                    let _ =
                        state
                            .pool_service
                            .mark_healthy(db, &credential.uuid, Some(&request.model));
                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                }

                let message = if has_tool_calls {
                    serde_json::json!({
                        "role": "assistant",
                        "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                        "tool_calls": parsed.tool_calls.iter().map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            })
                        }).collect::<Vec<_>>()
                    })
                } else {
                    serde_json::json!({
                        "role": "assistant",
                        "content": parsed.content
                    })
                };

                Ok(serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion",
                    "created": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }))
            } else {
                let body = resp.text().await.unwrap_or_default();
                if let Some(db) = &state.db {
                    let _ = state
                        .pool_service
                        .mark_unhealthy(db, &credential.uuid, Some(&body));
                }
                Err(format!("Upstream error: {}", body))
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let provider = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            let resp = match provider.call_api(request).await {
                Ok(r) => r,
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    return Err(e.to_string());
                }
            };
            if resp.status().is_success() {
                // 记录成功
                if let Some(db) = &state.db {
                    let _ =
                        state
                            .pool_service
                            .mark_healthy(db, &credential.uuid, Some(&request.model));
                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                }
                resp.json::<serde_json::Value>()
                    .await
                    .map_err(|e| e.to_string())
            } else {
                let body = resp.text().await.unwrap_or_default();
                if let Some(db) = &state.db {
                    let _ = state
                        .pool_service
                        .mark_unhealthy(db, &credential.uuid, Some(&body));
                }
                Err(format!("Upstream error: {}", body))
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={}",
                actual_base_url,
                &credential.uuid[..8]
            );
            let provider = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            match provider.call_openai_api(request).await {
                Ok(result) => {
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    Ok(result)
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    Err(e.to_string())
                }
            }
        }
        CredentialData::AntigravityOAuth {
            creds_file_path, ..
        } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity
                .load_credentials_from_path(creds_file_path)
                .await
            {
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return Err(e.to_string());
            }
            if !antigravity.is_token_valid() {
                if let Err(e) = antigravity.refresh_token().await {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                    }
                    return Err(e.to_string());
                }
            }
            let antigravity_request = convert_openai_to_antigravity(request);
            match antigravity
                .call_api("generateContent", &antigravity_request)
                .await
            {
                Ok(resp) => {
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    Ok(convert_antigravity_to_openai_response(
                        &resp,
                        &request.model,
                    ))
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    Err(e.to_string())
                }
            }
        }
        // GeminiOAuth 和 QwenOAuth 暂不支持 WebSocket，需要使用 HTTP 端点
        _ => Err(
            "This credential type is not yet supported via WebSocket. Please use HTTP endpoints."
                .to_string(),
        ),
    }
}

/// WebSocket 专用的 Anthropic 格式 Provider 调用
async fn call_provider_anthropic_for_ws(
    state: &AppState,
    credential: &ProviderCredential,
    request: &AnthropicMessagesRequest,
) -> Result<serde_json::Value, String> {
    use crate::models::provider_pool_model::CredentialData;

    match &credential.credential {
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={}",
                actual_base_url,
                &credential.uuid[..8]
            );
            let provider = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            let resp = match provider.call_api(request).await {
                Ok(r) => r,
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    return Err(e.to_string());
                }
            };
            if resp.status().is_success() {
                // 记录成功
                if let Some(db) = &state.db {
                    let _ =
                        state
                            .pool_service
                            .mark_healthy(db, &credential.uuid, Some(&request.model));
                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                }
                resp.json::<serde_json::Value>()
                    .await
                    .map_err(|e| e.to_string())
            } else {
                let body = resp.text().await.unwrap_or_default();
                if let Some(db) = &state.db {
                    let _ = state
                        .pool_service
                        .mark_unhealthy(db, &credential.uuid, Some(&body));
                }
                Err(format!("Upstream error: {}", body))
            }
        }
        _ => {
            // 转换为 OpenAI 格式并调用（健康状态更新在 call_provider_openai_for_ws 中处理）
            let openai_request = convert_anthropic_to_openai(request);
            let result = call_provider_openai_for_ws(state, credential, &openai_request).await?;

            // 转换响应为 Anthropic 格式
            Ok(serde_json::json!({
                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": result.get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                }],
                "model": request.model,
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }))
        }
    }
}

// ============ Management API Types and Handlers ============

/// 管理 API 状态响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementStatusResponse {
    /// 服务器是否运行中
    pub running: bool,
    /// 监听地址
    pub host: String,
    /// 监听端口
    pub port: u16,
    /// 处理的请求数
    pub requests: u64,
    /// 运行时间（秒）
    pub uptime_secs: u64,
    /// 版本号
    pub version: String,
    /// TLS 是否启用
    pub tls_enabled: bool,
    /// 默认 Provider
    pub default_provider: String,
}

/// 凭证信息（用于列表显示）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialInfo {
    /// 凭证 ID
    pub id: String,
    /// Provider 类型
    pub provider_type: String,
    /// 是否禁用
    pub disabled: bool,
    /// 是否有效
    pub is_valid: bool,
}

/// 凭证列表响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialsListResponse {
    /// 凭证列表
    pub credentials: Vec<CredentialInfo>,
    /// 总数
    pub total: usize,
}

/// 添加凭证请求
#[derive(Debug, Clone, Deserialize)]
pub struct AddCredentialRequest {
    /// Provider 类型
    pub provider_type: String,
    /// 凭证 ID
    pub id: String,
    /// API Key（用于 API Key 类型的凭证）
    #[serde(default)]
    pub api_key: Option<String>,
    /// Token 文件路径（用于 OAuth 类型的凭证）
    #[serde(default)]
    pub token_file: Option<String>,
    /// Base URL
    #[serde(default)]
    pub base_url: Option<String>,
    /// 代理 URL
    #[serde(default)]
    pub proxy_url: Option<String>,
}

/// 添加凭证响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddCredentialResponse {
    /// 是否成功
    pub success: bool,
    /// 消息
    pub message: String,
    /// 凭证 ID
    pub id: Option<String>,
}

/// 配置响应（简化版，不包含敏感信息）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementConfigResponse {
    /// 服务器配置
    pub server: ManagementServerConfigInfo,
    /// 路由配置
    pub routing: ManagementRoutingConfigInfo,
    /// 重试配置
    pub retry: ManagementRetryConfigInfo,
    /// 远程管理配置（不包含 secret_key）
    pub remote_management: ManagementRemoteInfo,
}

/// 服务器配置信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementServerConfigInfo {
    pub host: String,
    pub port: u16,
    pub tls_enabled: bool,
}

/// 路由配置信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementRoutingConfigInfo {
    pub default_provider: String,
    pub rules_count: usize,
}

/// 重试配置信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementRetryConfigInfo {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

/// 远程管理配置信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementRemoteInfo {
    pub allow_remote: bool,
    pub has_secret_key: bool,
    pub disable_control_panel: bool,
}

/// 更新配置请求
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateConfigRequest {
    /// 默认 Provider
    #[serde(default)]
    pub default_provider: Option<String>,
    /// 是否允许远程访问
    #[serde(default)]
    pub allow_remote: Option<bool>,
}

/// 更新配置响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfigResponse {
    pub success: bool,
    pub message: String,
}

fn snapshot_config(state: &AppState) -> Option<Config> {
    if let Some(manager) = &state.config_manager {
        if let Ok(guard) = manager.read() {
            return Some(guard.config().clone());
        }
    }
    state.config.clone()
}

/// GET /v0/management/status - 获取服务器状态
pub async fn management_status(State(state): State<AppState>) -> impl IntoResponse {
    let default_provider = state.default_provider.read().await.clone();

    // 获取请求数量
    let requests = state.processor.stats.read().len() as u64;
    let config = snapshot_config(&state).unwrap_or_default();
    let uptime_secs = state.start_time.elapsed().as_secs();

    let response = ManagementStatusResponse {
        running: true,
        host: config.server.host,
        port: config.server.port,
        requests,
        uptime_secs,
        version: env!("CARGO_PKG_VERSION").to_string(),
        tls_enabled: config.server.tls.enable,
        default_provider,
    };

    Json(response)
}

/// GET /v0/management/credentials - 获取凭证列表
pub async fn management_list_credentials(State(state): State<AppState>) -> impl IntoResponse {
    let mut credentials = Vec::new();

    // 从数据库获取凭证列表
    if let Some(ref db) = state.db {
        if let Ok(conn) = db.lock() {
            if let Ok(pool_credentials) = ProviderPoolDao::get_all(&conn) {
                for cred in pool_credentials {
                    credentials.push(CredentialInfo {
                        id: cred.uuid.clone(),
                        provider_type: cred.provider_type.to_string(),
                        disabled: cred.is_disabled,
                        is_valid: cred.is_healthy,
                    });
                }
            }
        }
    }

    let total = credentials.len();
    Json(CredentialsListResponse { credentials, total })
}

/// POST /v0/management/credentials - 添加凭证
pub async fn management_add_credential(
    State(state): State<AppState>,
    Json(request): Json<AddCredentialRequest>,
) -> impl IntoResponse {
    use crate::models::provider_pool_model::{
        CredentialData, PoolProviderType, ProviderCredential,
    };

    // 验证请求
    if request.id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(AddCredentialResponse {
                success: false,
                message: "Credential ID is required".to_string(),
                id: None,
            }),
        );
    }

    if request.provider_type.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(AddCredentialResponse {
                success: false,
                message: "Provider type is required".to_string(),
                id: None,
            }),
        );
    }

    // 解析 provider 类型
    let provider_type: PoolProviderType = match request.provider_type.parse() {
        Ok(pt) => pt,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(AddCredentialResponse {
                    success: false,
                    message: format!("Invalid provider type: {}", request.provider_type),
                    id: None,
                }),
            );
        }
    };

    // 根据 provider 类型创建凭证数据
    let credential_data = match provider_type {
        PoolProviderType::OpenAI => {
            if let Some(api_key) = request.api_key {
                CredentialData::OpenAIKey {
                    api_key,
                    base_url: request.base_url,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "API key is required for OpenAI provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::Claude => {
            if let Some(api_key) = request.api_key {
                CredentialData::ClaudeKey {
                    api_key,
                    base_url: request.base_url,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "API key is required for Claude provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::Vertex => {
            if let Some(api_key) = request.api_key {
                CredentialData::VertexKey {
                    api_key,
                    base_url: request.base_url,
                    model_aliases: std::collections::HashMap::new(),
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "API key is required for Vertex provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::Kiro => {
            if let Some(token_file) = request.token_file {
                CredentialData::KiroOAuth {
                    creds_file_path: token_file,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "Token file is required for Kiro provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::Gemini => {
            if let Some(token_file) = request.token_file {
                CredentialData::GeminiOAuth {
                    creds_file_path: token_file,
                    project_id: None,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "Token file is required for Gemini provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::Qwen => {
            if let Some(token_file) = request.token_file {
                CredentialData::QwenOAuth {
                    creds_file_path: token_file,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "Token file is required for Qwen provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::Antigravity => {
            if let Some(token_file) = request.token_file {
                CredentialData::AntigravityOAuth {
                    creds_file_path: token_file,
                    project_id: None,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "Token file is required for Antigravity provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::GeminiApiKey => {
            if let Some(api_key) = request.api_key {
                CredentialData::GeminiApiKey {
                    api_key,
                    base_url: request.base_url,
                    excluded_models: Vec::new(),
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "API key is required for Gemini API Key provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::Codex => {
            if let Some(token_file) = request.token_file {
                CredentialData::CodexOAuth {
                    creds_file_path: token_file,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "Token file is required for Codex provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::ClaudeOAuth => {
            if let Some(token_file) = request.token_file {
                CredentialData::ClaudeOAuth {
                    creds_file_path: token_file,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "Token file is required for Claude OAuth provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
        PoolProviderType::IFlow => {
            if let Some(token_file) = request.token_file {
                // 默认使用 OAuth 类型，Cookie 类型需要通过其他方式添加
                CredentialData::IFlowOAuth {
                    creds_file_path: token_file,
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(AddCredentialResponse {
                        success: false,
                        message: "Token file is required for iFlow provider".to_string(),
                        id: None,
                    }),
                );
            }
        }
    };

    // 创建凭证
    let mut credential = ProviderCredential::new(provider_type, credential_data);
    credential.uuid = request.id.clone();
    credential.name = Some(request.id.clone());

    // 添加凭证到数据库
    if let Some(ref db) = state.db {
        if let Ok(conn) = db.lock() {
            match ProviderPoolDao::insert(&conn, &credential) {
                Ok(_) => {
                    tracing::info!(
                        "[MANAGEMENT] Added credential: {} ({})",
                        request.id,
                        request.provider_type
                    );
                    return (
                        StatusCode::CREATED,
                        Json(AddCredentialResponse {
                            success: true,
                            message: "Credential added successfully".to_string(),
                            id: Some(request.id),
                        }),
                    );
                }
                Err(e) => {
                    tracing::error!("[MANAGEMENT] Failed to add credential: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(AddCredentialResponse {
                            success: false,
                            message: format!("Failed to add credential: {}", e),
                            id: None,
                        }),
                    );
                }
            }
        }
    }

    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(AddCredentialResponse {
            success: false,
            message: "Database not available".to_string(),
            id: None,
        }),
    )
}

/// GET /v0/management/config - 获取配置
pub async fn management_get_config(State(state): State<AppState>) -> impl IntoResponse {
    let default_provider = state.default_provider.read().await.clone();

    // 获取路由规则数量
    let rules_count = state.processor.router.read().await.rules().len();
    let config = snapshot_config(&state).unwrap_or_default();

    let response = ManagementConfigResponse {
        server: ManagementServerConfigInfo {
            host: config.server.host,
            port: config.server.port,
            tls_enabled: config.server.tls.enable,
        },
        routing: ManagementRoutingConfigInfo {
            default_provider,
            rules_count,
        },
        retry: ManagementRetryConfigInfo {
            max_retries: config.retry.max_retries,
            base_delay_ms: config.retry.base_delay_ms,
            max_delay_ms: config.retry.max_delay_ms,
        },
        remote_management: ManagementRemoteInfo {
            allow_remote: config.remote_management.allow_remote,
            has_secret_key: config
                .remote_management
                .secret_key
                .as_ref()
                .map(|key| !key.is_empty())
                .unwrap_or(false),
            disable_control_panel: config.remote_management.disable_control_panel,
        },
    };

    Json(response)
}

/// PUT /v0/management/config - 更新配置
pub async fn management_update_config(
    State(state): State<AppState>,
    Json(request): Json<UpdateConfigRequest>,
) -> impl IntoResponse {
    let mut updated = false;
    let mut needs_restart = false;

    // 更新默认 Provider
    if let Some(provider) = request.default_provider {
        // 验证 provider 类型
        if provider.parse::<crate::ProviderType>().is_ok() {
            let mut dp = state.default_provider.write().await;
            *dp = provider.clone();
            tracing::info!("[MANAGEMENT] Updated default_provider to: {}", provider);
            if let Some(manager) = &state.config_manager {
                if let Ok(mut guard) = manager.write() {
                    guard.config_mut().default_provider = provider.clone();
                    guard.config_mut().routing.default_provider = provider.clone();
                    if let Err(err) = guard.save() {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(UpdateConfigResponse {
                                success: false,
                                message: format!("Failed to save config: {}", err),
                            }),
                        );
                    }
                } else {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(UpdateConfigResponse {
                            success: false,
                            message: "Failed to lock config manager".to_string(),
                        }),
                    );
                }
            }
            updated = true;
        } else {
            return (
                StatusCode::BAD_REQUEST,
                Json(UpdateConfigResponse {
                    success: false,
                    message: format!("Invalid provider type: {}", provider),
                }),
            );
        }
    }

    // 更新是否允许远程访问（需要重启生效）
    if let Some(allow_remote) = request.allow_remote {
        if allow_remote {
            return (
                StatusCode::BAD_REQUEST,
                Json(UpdateConfigResponse {
                    success: false,
                    message: "当前版本未启用 TLS，禁止开启远程管理".to_string(),
                }),
            );
        }
        let manager = match &state.config_manager {
            Some(manager) => manager,
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(UpdateConfigResponse {
                        success: false,
                        message: "Config manager is not available".to_string(),
                    }),
                );
            }
        };
        if let Ok(mut guard) = manager.write() {
            guard.config_mut().remote_management.allow_remote = allow_remote;
            if let Err(err) = guard.save() {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(UpdateConfigResponse {
                        success: false,
                        message: format!("Failed to save config: {}", err),
                    }),
                );
            }
        } else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(UpdateConfigResponse {
                    success: false,
                    message: "Failed to lock config manager".to_string(),
                }),
            );
        }
        tracing::info!(
            "[MANAGEMENT] Updated remote_management.allow_remote to: {}",
            allow_remote
        );
        updated = true;
        needs_restart = true;
    }

    if updated {
        (
            StatusCode::OK,
            Json(UpdateConfigResponse {
                success: true,
                message: if needs_restart {
                    "Configuration updated. Restart required to apply all changes.".to_string()
                } else {
                    "Configuration updated successfully".to_string()
                },
            }),
        )
    } else {
        (
            StatusCode::OK,
            Json(UpdateConfigResponse {
                success: true,
                message: "No changes applied".to_string(),
            }),
        )
    }
}
