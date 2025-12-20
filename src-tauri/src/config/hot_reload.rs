//! 配置热重载模块
//!
//! 提供配置文件监控和热重载功能
//! - 使用 `notify` crate 监控配置文件变化
//! - 支持原子性配置更新
//! - 失败时自动回滚到之前的配置

use super::types::Config;
use super::yaml::ConfigManager;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// 热重载错误类型
#[derive(Debug, Clone)]
pub enum HotReloadError {
    /// 文件监控错误
    WatchError(String),
    /// 配置加载错误
    LoadError(String),
    /// 配置验证错误
    ValidationError(String),
    /// 回滚错误
    RollbackError(String),
    /// 通道错误
    ChannelError(String),
}

impl std::fmt::Display for HotReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HotReloadError::WatchError(msg) => write!(f, "文件监控错误: {}", msg),
            HotReloadError::LoadError(msg) => write!(f, "配置加载错误: {}", msg),
            HotReloadError::ValidationError(msg) => write!(f, "配置验证错误: {}", msg),
            HotReloadError::RollbackError(msg) => write!(f, "回滚错误: {}", msg),
            HotReloadError::ChannelError(msg) => write!(f, "通道错误: {}", msg),
        }
    }
}

impl std::error::Error for HotReloadError {}

/// 热重载结果
#[derive(Debug, Clone)]
pub enum ReloadResult {
    /// 重载成功
    Success {
        /// 重载时间戳
        timestamp: Instant,
    },
    /// 重载失败，已回滚
    RolledBack {
        /// 错误信息
        error: String,
        /// 回滚时间戳
        timestamp: Instant,
    },
    /// 重载失败，回滚也失败
    Failed {
        /// 原始错误
        error: String,
        /// 回滚错误
        rollback_error: Option<String>,
        /// 失败时间戳
        timestamp: Instant,
    },
}

/// 配置变更事件
#[derive(Debug, Clone)]
pub struct ConfigChangeEvent {
    /// 变更的文件路径
    pub path: PathBuf,
    /// 事件类型
    pub kind: ConfigChangeKind,
    /// 事件时间戳
    pub timestamp: Instant,
}

/// 配置变更类型
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigChangeKind {
    /// 文件被修改
    Modified,
    /// 文件被创建
    Created,
    /// 文件被删除
    Removed,
}

/// 文件监控器
///
/// 监控配置文件变化并触发回调
pub struct FileWatcher {
    /// 内部监控器
    watcher: RecommendedWatcher,
    /// 监控的路径
    watched_path: PathBuf,
    /// 是否正在运行
    running: Arc<AtomicBool>,
}

impl FileWatcher {
    /// 创建新的文件监控器
    ///
    /// # Arguments
    /// * `path` - 要监控的文件路径
    /// * `tx` - 事件发送通道
    pub fn new(
        path: &Path,
        tx: mpsc::UnboundedSender<ConfigChangeEvent>,
    ) -> Result<Self, HotReloadError> {
        let watched_path = path.to_path_buf();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        // 防抖动：记录最后一次事件时间
        let last_event = Arc::new(RwLock::new(Instant::now() - Duration::from_secs(10)));
        let debounce_duration = Duration::from_millis(500);

        let watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if !running_clone.load(Ordering::SeqCst) {
                return;
            }

            match res {
                Ok(event) => {
                    // 检查是否需要防抖动
                    let now = Instant::now();
                    {
                        let last = last_event.read();
                        if now.duration_since(*last) < debounce_duration {
                            return;
                        }
                    }

                    // 更新最后事件时间
                    {
                        let mut last = last_event.write();
                        *last = now;
                    }

                    // 转换事件类型
                    let kind = match event.kind {
                        EventKind::Create(_) => Some(ConfigChangeKind::Created),
                        EventKind::Modify(_) => Some(ConfigChangeKind::Modified),
                        EventKind::Remove(_) => Some(ConfigChangeKind::Removed),
                        _ => None,
                    };

                    if let Some(kind) = kind {
                        for path in event.paths {
                            let change_event = ConfigChangeEvent {
                                path,
                                kind: kind.clone(),
                                timestamp: now,
                            };
                            let _ = tx.send(change_event);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("文件监控错误: {:?}", e);
                }
            }
        })
        .map_err(|e| HotReloadError::WatchError(e.to_string()))?;

        Ok(Self {
            watcher,
            watched_path,
            running,
        })
    }

    /// 开始监控
    pub fn start(&mut self) -> Result<(), HotReloadError> {
        // 监控文件所在目录（因为某些编辑器会删除并重建文件）
        let watch_path = self.watched_path.parent().unwrap_or(&self.watched_path);

        self.watcher
            .watch(watch_path, RecursiveMode::NonRecursive)
            .map_err(|e| HotReloadError::WatchError(e.to_string()))?;

        self.running.store(true, Ordering::SeqCst);
        tracing::info!("开始监控配置文件: {:?}", self.watched_path);
        Ok(())
    }

    /// 停止监控
    pub fn stop(&mut self) -> Result<(), HotReloadError> {
        self.running.store(false, Ordering::SeqCst);

        let watch_path = self.watched_path.parent().unwrap_or(&self.watched_path);

        self.watcher
            .unwatch(watch_path)
            .map_err(|e| HotReloadError::WatchError(e.to_string()))?;

        tracing::info!("停止监控配置文件: {:?}", self.watched_path);
        Ok(())
    }

    /// 检查是否正在运行
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// 获取监控的路径
    pub fn watched_path(&self) -> &Path {
        &self.watched_path
    }
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// 热重载管理器
///
/// 管理配置的热重载，支持原子性更新和失败回滚
pub struct HotReloadManager {
    /// 当前配置
    current_config: Arc<RwLock<Config>>,
    /// 备份配置（用于回滚）
    backup_config: Arc<RwLock<Option<Config>>>,
    /// 配置文件路径
    config_path: PathBuf,
    /// 最后重载时间
    last_reload: Arc<RwLock<Option<Instant>>>,
    /// 重载状态
    reload_in_progress: Arc<AtomicBool>,
}

impl HotReloadManager {
    /// 创建新的热重载管理器
    pub fn new(config: Config, config_path: PathBuf) -> Self {
        Self {
            current_config: Arc::new(RwLock::new(config)),
            backup_config: Arc::new(RwLock::new(None)),
            config_path,
            last_reload: Arc::new(RwLock::new(None)),
            reload_in_progress: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 获取当前配置
    pub fn config(&self) -> Config {
        self.current_config.read().clone()
    }

    /// 获取配置的引用
    pub fn config_ref(&self) -> Arc<RwLock<Config>> {
        self.current_config.clone()
    }

    /// 获取最后重载时间
    pub fn last_reload_time(&self) -> Option<Instant> {
        *self.last_reload.read()
    }

    /// 检查是否正在重载
    pub fn is_reloading(&self) -> bool {
        self.reload_in_progress.load(Ordering::SeqCst)
    }

    /// 执行热重载
    ///
    /// 原子性地更新配置，失败时自动回滚
    pub fn reload(&self) -> ReloadResult {
        // 检查是否已经在重载中
        if self
            .reload_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return ReloadResult::Failed {
                error: "重载已在进行中".to_string(),
                rollback_error: None,
                timestamp: Instant::now(),
            };
        }

        let result = self.do_reload();

        // 重置重载状态
        self.reload_in_progress.store(false, Ordering::SeqCst);

        result
    }

    /// 内部重载逻辑
    fn do_reload(&self) -> ReloadResult {
        let now = Instant::now();

        // 1. 备份当前配置
        {
            let current = self.current_config.read().clone();
            let mut backup = self.backup_config.write();
            *backup = Some(current);
        }

        // 2. 尝试加载新配置
        let new_config = match self.load_config_from_file() {
            Ok(config) => config,
            Err(e) => {
                // 加载失败，清除备份（无需回滚，因为当前配置未变）
                let mut backup = self.backup_config.write();
                *backup = None;
                return ReloadResult::RolledBack {
                    error: e.to_string(),
                    timestamp: now,
                };
            }
        };

        // 3. 验证新配置
        if let Err(e) = self.validate_config(&new_config) {
            // 验证失败，清除备份
            let mut backup = self.backup_config.write();
            *backup = None;
            return ReloadResult::RolledBack {
                error: e.to_string(),
                timestamp: now,
            };
        }

        // 4. 原子性地应用新配置
        {
            let mut current = self.current_config.write();
            *current = new_config;
        }

        // 5. 更新最后重载时间
        {
            let mut last = self.last_reload.write();
            *last = Some(now);
        }

        // 6. 清除备份
        {
            let mut backup = self.backup_config.write();
            *backup = None;
        }

        tracing::info!("配置热重载成功");
        ReloadResult::Success { timestamp: now }
    }

    /// 从文件加载配置
    fn load_config_from_file(&self) -> Result<Config, HotReloadError> {
        if !self.config_path.exists() {
            return Err(HotReloadError::LoadError(format!(
                "配置文件不存在: {:?}",
                self.config_path
            )));
        }

        let content = std::fs::read_to_string(&self.config_path)
            .map_err(|e| HotReloadError::LoadError(e.to_string()))?;

        ConfigManager::parse_yaml(&content).map_err(|e| HotReloadError::LoadError(e.to_string()))
    }

    /// 验证配置
    fn validate_config(&self, config: &Config) -> Result<(), HotReloadError> {
        let is_localhost = is_localhost_host(&config.server.host);

        // 验证端口范围
        if config.server.port == 0 {
            return Err(HotReloadError::ValidationError(
                "端口号不能为 0".to_string(),
            ));
        }

        if !is_localhost {
            return Err(HotReloadError::ValidationError(
                "当前版本仅支持本地监听，请使用 127.0.0.1/localhost/::1".to_string(),
            ));
        }

        // 验证重试配置
        if config.retry.max_retries > 100 {
            return Err(HotReloadError::ValidationError(
                "最大重试次数不能超过 100".to_string(),
            ));
        }

        if config.retry.base_delay_ms == 0 {
            return Err(HotReloadError::ValidationError(
                "基础延迟不能为 0".to_string(),
            ));
        }

        // 验证日志保留天数
        if config.logging.retention_days == 0 {
            return Err(HotReloadError::ValidationError(
                "日志保留天数不能为 0".to_string(),
            ));
        }

        if config.server.api_key.trim().is_empty() {
            return Err(HotReloadError::ValidationError(
                "API Key 不能为空".to_string(),
            ));
        }

        if (!is_localhost || config.remote_management.allow_remote)
            && crate::config::is_default_api_key(&config.server.api_key)
        {
            return Err(HotReloadError::ValidationError(
                "非本地访问场景下禁止使用默认 API Key，请设置强口令".to_string(),
            ));
        }

        if config.server.tls.enable {
            return Err(HotReloadError::ValidationError(
                "当前版本暂不支持 TLS，请关闭 TLS 配置".to_string(),
            ));
        }

        if config.remote_management.allow_remote {
            return Err(HotReloadError::ValidationError(
                "当前版本未启用 TLS，禁止开启远程管理".to_string(),
            ));
        }

        Ok(())
    }

    /// 手动回滚到备份配置
    pub fn rollback(&self) -> Result<(), HotReloadError> {
        let backup = {
            let backup = self.backup_config.read();
            backup.clone()
        };

        match backup {
            Some(config) => {
                let mut current = self.current_config.write();
                *current = config;

                // 清除备份
                let mut backup = self.backup_config.write();
                *backup = None;

                tracing::info!("配置已回滚");
                Ok(())
            }
            None => Err(HotReloadError::RollbackError(
                "没有可用的备份配置".to_string(),
            )),
        }
    }

    /// 更新配置（用于外部更新）
    pub fn update_config(&self, config: Config) {
        let mut current = self.current_config.write();
        *current = config;
    }

    /// 获取配置文件路径
    pub fn config_path(&self) -> &Path {
        &self.config_path
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

/// 热重载状态
#[derive(Debug, Clone, serde::Serialize)]
pub struct HotReloadStatus {
    /// 是否启用
    pub enabled: bool,
    /// 是否正在监控
    pub watching: bool,
    /// 最后重载时间（毫秒时间戳）
    pub last_reload_ms: Option<u64>,
    /// 配置文件路径
    pub config_path: String,
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_hot_reload_manager_new() {
        let config = Config::default();
        let path = PathBuf::from("/tmp/test_config.yaml");
        let manager = HotReloadManager::new(config.clone(), path.clone());

        assert_eq!(manager.config(), config);
        assert_eq!(manager.config_path(), path);
        assert!(manager.last_reload_time().is_none());
        assert!(!manager.is_reloading());
    }

    #[test]
    fn test_hot_reload_manager_update_config() {
        let config = Config::default();
        let path = PathBuf::from("/tmp/test_config.yaml");
        let manager = HotReloadManager::new(config, path);

        let mut new_config = Config::default();
        new_config.server.port = 9999;
        manager.update_config(new_config.clone());

        assert_eq!(manager.config().server.port, 9999);
    }

    #[test]
    fn test_hot_reload_manager_reload_file_not_exists() {
        let config = Config::default();
        let path = PathBuf::from("/tmp/nonexistent_config_12345.yaml");
        let manager = HotReloadManager::new(config, path);

        let result = manager.reload();
        match result {
            ReloadResult::RolledBack { error, .. } => {
                assert!(error.contains("不存在"));
            }
            _ => panic!("Expected RolledBack result"),
        }
    }

    #[test]
    fn test_hot_reload_manager_reload_success() {
        // 创建临时配置文件
        let mut temp_file = NamedTempFile::new().unwrap();
        let yaml_content = r#"
server:
  host: "127.0.0.1"
  port: 9000
  api_key: "test-key"
retry:
  max_retries: 5
  base_delay_ms: 2000
  max_delay_ms: 60000
  auto_switch_provider: true
logging:
  enabled: true
  level: "debug"
  retention_days: 14
"#;
        temp_file.write_all(yaml_content.as_bytes()).unwrap();

        let config = Config::default();
        let manager = HotReloadManager::new(config, temp_file.path().to_path_buf());

        let result = manager.reload();
        match result {
            ReloadResult::Success { .. } => {
                let new_config = manager.config();
                assert_eq!(new_config.server.port, 9000);
                assert_eq!(new_config.retry.max_retries, 5);
                assert_eq!(new_config.logging.level, "debug");
            }
            _ => panic!("Expected Success result"),
        }
    }

    #[test]
    fn test_hot_reload_manager_reload_invalid_yaml() {
        // 创建临时配置文件（无效 YAML）
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"invalid: yaml: content:").unwrap();

        let config = Config::default();
        let manager = HotReloadManager::new(config.clone(), temp_file.path().to_path_buf());

        let result = manager.reload();
        match result {
            ReloadResult::RolledBack { .. } => {
                // 配置应该保持不变
                assert_eq!(manager.config(), config);
            }
            _ => panic!("Expected RolledBack result"),
        }
    }

    #[test]
    fn test_hot_reload_manager_validation_error() {
        // 创建临时配置文件（端口为 0）
        let mut temp_file = NamedTempFile::new().unwrap();
        let yaml_content = r#"
server:
  host: "127.0.0.1"
  port: 0
  api_key: "test-key"
"#;
        temp_file.write_all(yaml_content.as_bytes()).unwrap();

        let config = Config::default();
        let manager = HotReloadManager::new(config.clone(), temp_file.path().to_path_buf());

        let result = manager.reload();
        match result {
            ReloadResult::RolledBack { error, .. } => {
                assert!(error.contains("端口号"));
                // 配置应该保持不变
                assert_eq!(manager.config(), config);
            }
            _ => panic!("Expected RolledBack result"),
        }
    }

    #[test]
    fn test_config_change_kind_eq() {
        assert_eq!(ConfigChangeKind::Modified, ConfigChangeKind::Modified);
        assert_ne!(ConfigChangeKind::Modified, ConfigChangeKind::Created);
    }

    #[test]
    fn test_hot_reload_error_display() {
        let err = HotReloadError::WatchError("test error".to_string());
        assert!(err.to_string().contains("文件监控错误"));
        assert!(err.to_string().contains("test error"));

        let err = HotReloadError::LoadError("load error".to_string());
        assert!(err.to_string().contains("配置加载错误"));

        let err = HotReloadError::ValidationError("validation error".to_string());
        assert!(err.to_string().contains("配置验证错误"));

        let err = HotReloadError::RollbackError("rollback error".to_string());
        assert!(err.to_string().contains("回滚错误"));
    }
}
