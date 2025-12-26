//! Flow Monitor Tauri 命令
//!
//! 提供 LLM Flow Monitor 的 Tauri 命令接口，用于前端访问 Flow 数据。
//!
//! **Validates: Requirements 10.1-10.7**

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;

use crate::flow_monitor::monitor::{NotificationConfig, NotificationSettings};
use crate::flow_monitor::{
    get_filter_help, BatchOperation, BatchOperations, BatchResult, DiffConfig, ExportFormat,
    ExportOptions, FilterExpr, FilterParser, FlowAnnotations, FlowDiff, FlowDiffResult,
    FlowExporter, FlowFilter, FlowMonitor, FlowQueryResult, FlowQueryService, FlowSearchResult,
    FlowSortBy, FlowStats, LLMFlow, FILTER_HELP,
};

// ============================================================================
// 状态封装
// ============================================================================

/// FlowMonitor 状态封装
pub struct FlowMonitorState(pub Arc<FlowMonitor>);

/// FlowQueryService 状态封装
pub struct FlowQueryServiceState(pub Arc<FlowQueryService>);

// ============================================================================
// 请求/响应类型
// ============================================================================

/// 查询 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFlowsRequest {
    /// 过滤条件
    #[serde(default)]
    pub filter: FlowFilter,
    /// 排序字段
    #[serde(default)]
    pub sort_by: FlowSortBy,
    /// 是否降序
    #[serde(default = "default_true")]
    pub sort_desc: bool,
    /// 页码（从 1 开始）
    #[serde(default = "default_page")]
    pub page: usize,
    /// 每页大小
    #[serde(default = "default_page_size")]
    pub page_size: usize,
}

fn default_true() -> bool {
    true
}

fn default_page() -> usize {
    1
}

fn default_page_size() -> usize {
    20
}

impl Default for QueryFlowsRequest {
    fn default() -> Self {
        Self {
            filter: FlowFilter::default(),
            sort_by: FlowSortBy::default(),
            sort_desc: true,
            page: 1,
            page_size: 20,
        }
    }
}

/// 搜索 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFlowsRequest {
    /// 搜索关键词
    pub query: String,
    /// 最大返回数量
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize {
    50
}

/// 导出 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFlowsRequest {
    /// 导出格式
    pub format: ExportFormat,
    /// 过滤条件
    #[serde(default)]
    pub filter: Option<FlowFilter>,
    /// 是否包含原始请求/响应体
    #[serde(default = "default_true")]
    pub include_raw: bool,
    /// 是否包含流式 chunks
    #[serde(default)]
    pub include_stream_chunks: bool,
    /// 是否脱敏敏感数据
    #[serde(default)]
    pub redact_sensitive: bool,
    /// Flow ID 列表（如果指定，则只导出这些 Flow）
    #[serde(default)]
    pub flow_ids: Option<Vec<String>>,
}

/// 导出结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFlowsResponse {
    /// 导出的数据（JSON 字符串）
    pub data: String,
    /// 导出的 Flow 数量
    pub count: usize,
    /// 导出格式
    pub format: ExportFormat,
}

/// 更新标注请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateAnnotationsRequest {
    /// Flow ID
    pub flow_id: String,
    /// 标注信息
    pub annotations: FlowAnnotations,
}

/// 清理 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupFlowsRequest {
    /// 清理类型
    pub cleanup_type: CleanupType,
    /// 保留天数（清理此天数之前的数据）- 仅当 cleanup_type 为 ByTime 时使用
    pub retention_days: Option<u32>,
    /// 保留小时数（清理此小时数之前的数据）- 仅当 cleanup_type 为 ByTime 时使用
    pub retention_hours: Option<u32>,
    /// 保留的最大记录数 - 仅当 cleanup_type 为 ByCount 时使用
    pub max_records: Option<usize>,
    /// 要清理的状态列表 - 仅当 cleanup_type 为 ByStatus 时使用
    pub target_states: Option<Vec<String>>,
    /// 要清理的 Provider 列表 - 仅当 cleanup_type 为 ByProvider 时使用
    pub target_providers: Option<Vec<String>>,
    /// 最大存储大小（字节）- 仅当 cleanup_type 为 BySize 时使用
    pub max_storage_bytes: Option<u64>,
}

/// 清理类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupType {
    /// 删除所有日志
    All,
    /// 按时间清理（保留最近的数据）
    ByTime,
    /// 按数量清理（只保留最近N条记录）
    ByCount,
    /// 按状态清理（删除特定状态的Flow）
    ByStatus,
    /// 按Provider清理（删除特定Provider的日志）
    ByProvider,
    /// 按存储大小清理（当超过指定大小时清理最旧的数据）
    BySize,
}

/// 清理结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupFlowsResponse {
    /// 清理的 Flow 数量
    pub cleaned_count: usize,
    /// 清理的文件数量
    pub cleaned_files: usize,
    /// 释放的空间（字节）
    pub freed_bytes: u64,
}

// ============================================================================
// Tauri 命令实现
// ============================================================================

/// 查询 Flow 列表
///
/// **Validates: Requirements 10.1**
///
/// # Arguments
/// * `request` - 查询请求参数
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(FlowQueryResult)` - 成功时返回查询结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn query_flows(
    request: QueryFlowsRequest,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<FlowQueryResult, String> {
    query_service
        .0
        .query(
            request.filter,
            request.sort_by,
            request.sort_desc,
            request.page,
            request.page_size,
        )
        .await
        .map_err(|e| format!("查询 Flow 失败: {}", e))
}

/// 获取单个 Flow 详情
///
/// **Validates: Requirements 10.2**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(Some(LLMFlow))` - 成功时返回 Flow 详情
/// * `Ok(None)` - Flow 不存在
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_flow_detail(
    flow_id: String,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<Option<LLMFlow>, String> {
    query_service
        .0
        .get_flow(&flow_id)
        .await
        .map_err(|e| format!("获取 Flow 详情失败: {}", e))
}

/// 全文搜索 Flow
///
/// **Validates: Requirements 10.3**
///
/// # Arguments
/// * `request` - 搜索请求参数
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(Vec<FlowSearchResult>)` - 成功时返回搜索结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn search_flows(
    request: SearchFlowsRequest,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<Vec<FlowSearchResult>, String> {
    query_service
        .0
        .search(&request.query, request.limit)
        .await
        .map_err(|e| format!("搜索 Flow 失败: {}", e))
}

/// 获取 Flow 统计信息
///
/// **Validates: Requirements 10.4**
///
/// # Arguments
/// * `filter` - 过滤条件（可选）
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(FlowStats)` - 成功时返回统计信息
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_flow_stats(
    filter: Option<FlowFilter>,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<FlowStats, String> {
    let filter = filter.unwrap_or_default();
    Ok(query_service.0.get_stats(&filter).await)
}

/// 导出 Flow
///
/// **Validates: Requirements 10.5**
///
/// # Arguments
/// * `request` - 导出请求参数
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(ExportFlowsResponse)` - 成功时返回导出结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn export_flows(
    request: ExportFlowsRequest,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<ExportFlowsResponse, String> {
    // 获取要导出的 Flow
    let flows = if let Some(flow_ids) = request.flow_ids {
        // 按 ID 列表获取
        let mut flows = Vec::new();
        for id in flow_ids {
            if let Ok(Some(flow)) = query_service.0.get_flow(&id).await {
                flows.push(flow);
            }
        }
        flows
    } else {
        // 按过滤条件获取
        let filter = request.filter.unwrap_or_default();
        let result = query_service
            .0
            .query(filter, FlowSortBy::CreatedAt, true, 1, 10000)
            .await
            .map_err(|e| format!("查询 Flow 失败: {}", e))?;
        result.flows
    };

    let count = flows.len();

    // 创建导出器
    let options = ExportOptions {
        format: request.format,
        filter: None,
        include_raw: request.include_raw,
        include_stream_chunks: request.include_stream_chunks,
        redact_sensitive: request.redact_sensitive,
        redaction_rules: Vec::new(),
        compress: false,
    };
    let exporter = FlowExporter::new(options);

    // 导出数据
    let data = match request.format {
        ExportFormat::HAR => {
            let har = exporter.export_har(&flows);
            serde_json::to_string_pretty(&har).map_err(|e| format!("序列化 HAR 失败: {}", e))?
        }
        ExportFormat::JSON => {
            let json = exporter.export_json(&flows);
            serde_json::to_string_pretty(&json).map_err(|e| format!("序列化 JSON 失败: {}", e))?
        }
        ExportFormat::JSONL => exporter.export_jsonl(&flows),
        ExportFormat::Markdown => exporter.export_markdown_multiple(&flows),
        ExportFormat::CSV => exporter.export_csv(&flows),
    };

    Ok(ExportFlowsResponse {
        data,
        count,
        format: request.format,
    })
}

/// 更新 Flow 标注
///
/// **Validates: Requirements 10.6**
///
/// # Arguments
/// * `request` - 更新标注请求参数
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否更新成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn update_flow_annotations(
    request: UpdateAnnotationsRequest,
    monitor: State<'_, FlowMonitorState>,
) -> Result<bool, String> {
    let updated = monitor
        .0
        .update_annotations(&request.flow_id, request.annotations)
        .await;
    Ok(updated)
}

/// 切换 Flow 收藏状态
///
/// **Validates: Requirements 10.6**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否更新成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn toggle_flow_starred(
    flow_id: String,
    monitor: State<'_, FlowMonitorState>,
) -> Result<bool, String> {
    let updated = monitor.0.toggle_starred(&flow_id).await;
    Ok(updated)
}

/// 添加 Flow 评论
///
/// **Validates: Requirements 10.6**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `comment` - 评论内容
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否更新成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn add_flow_comment(
    flow_id: String,
    comment: String,
    monitor: State<'_, FlowMonitorState>,
) -> Result<bool, String> {
    let updated = monitor.0.add_comment(&flow_id, comment).await;
    Ok(updated)
}

/// 添加 Flow 标签
///
/// **Validates: Requirements 10.6**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `tag` - 标签
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否更新成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn add_flow_tag(
    flow_id: String,
    tag: String,
    monitor: State<'_, FlowMonitorState>,
) -> Result<bool, String> {
    let updated = monitor.0.add_tag(&flow_id, tag).await;
    Ok(updated)
}

/// 移除 Flow 标签
///
/// **Validates: Requirements 10.6**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `tag` - 标签
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否更新成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn remove_flow_tag(
    flow_id: String,
    tag: String,
    monitor: State<'_, FlowMonitorState>,
) -> Result<bool, String> {
    let updated = monitor.0.remove_tag(&flow_id, &tag).await;
    Ok(updated)
}

/// 设置 Flow 标记
///
/// **Validates: Requirements 10.6**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `marker` - 标记（如 ⭐、🔴、🟢，None 表示清除标记）
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否更新成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn set_flow_marker(
    flow_id: String,
    marker: Option<String>,
    monitor: State<'_, FlowMonitorState>,
) -> Result<bool, String> {
    let updated = monitor.0.set_marker(&flow_id, marker).await;
    Ok(updated)
}

/// 清理旧的 Flow 数据
///
/// **Validates: Requirements 10.7**
///
/// # Arguments
/// * `request` - 清理请求参数
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(CleanupFlowsResponse)` - 成功时返回清理结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn cleanup_flows(
    request: CleanupFlowsRequest,
    monitor: State<'_, FlowMonitorState>,
) -> Result<CleanupFlowsResponse, String> {
    let mut cleaned_count = 0;
    let mut cleaned_files = 0;
    let mut freed_bytes = 0u64;

    // 根据清理类型执行不同的清理逻辑
    match request.cleanup_type {
        CleanupType::All => {
            // 清理所有数据 - 使用当前时间 + 1天，确保删除所有数据
            let before = chrono::Utc::now() + chrono::Duration::days(1);

            // 清理文件存储
            if let Some(file_store) = monitor.0.file_store() {
                match file_store.cleanup(before) {
                    Ok(result) => {
                        cleaned_count = result.flows_deleted;
                        cleaned_files = result.files_deleted;
                        freed_bytes = result.bytes_freed;
                    }
                    Err(e) => {
                        tracing::error!("清理所有数据失败: {}", e);
                        return Err(format!("清理所有数据失败: {}", e));
                    }
                }
            }

            // 清理内存存储
            {
                let memory_store = monitor.0.memory_store();
                let mut store = memory_store.write().await;
                store.clear();
                tracing::info!("已清理内存存储");
            }
        }

        CleanupType::ByTime => {
            // 按时间清理
            let before = if let Some(hours) = request.retention_hours {
                chrono::Utc::now() - chrono::Duration::hours(hours as i64)
            } else if let Some(days) = request.retention_days {
                chrono::Utc::now() - chrono::Duration::days(days as i64)
            } else {
                return Err("时间清理需要指定保留时间".to_string());
            };

            // 清理文件存储
            if let Some(file_store) = monitor.0.file_store() {
                match file_store.cleanup(before) {
                    Ok(result) => {
                        cleaned_count = result.flows_deleted;
                        cleaned_files = result.files_deleted;
                        freed_bytes = result.bytes_freed;
                    }
                    Err(e) => {
                        tracing::error!("按时间清理失败: {}", e);
                        return Err(format!("按时间清理失败: {}", e));
                    }
                }
            }

            // 清理内存存储
            {
                let memory_store = monitor.0.memory_store();
                let mut store = memory_store.write().await;
                let memory_cleaned = store.cleanup_before(before);
                cleaned_count += memory_cleaned;
                tracing::info!("已清理内存存储 {} 条记录", memory_cleaned);
            }
        }

        CleanupType::ByCount => {
            // 按数量清理 - 使用retention清理，这是一个简化实现
            if let Some(file_store) = monitor.0.file_store() {
                match file_store.cleanup_by_retention() {
                    Ok(result) => {
                        cleaned_count = result.flows_deleted;
                        cleaned_files = result.files_deleted;
                        freed_bytes = result.bytes_freed;
                    }
                    Err(e) => {
                        tracing::error!("按数量清理失败: {}", e);
                        return Err(format!("按数量清理失败: {}", e));
                    }
                }
            }

            // 清理内存存储 - 对于按数量清理，清空所有内存数据
            // 因为文件存储已按保留策略清理，内存应保持一致
            {
                let memory_store = monitor.0.memory_store();
                let mut store = memory_store.write().await;
                store.clear();
                tracing::info!("已清理内存存储");
            }
        }

        CleanupType::ByStatus => {
            // 按状态清理 - 暂时不支持，返回错误提示
            return Err("按状态清理功能暂未实现，请使用按时间清理".to_string());
        }

        CleanupType::ByProvider => {
            // 按Provider清理 - 暂时不支持，返回错误提示
            return Err("按Provider清理功能暂未实现，请使用按时间清理".to_string());
        }

        CleanupType::BySize => {
            // 按大小清理 - 暂时不支持，返回错误提示
            return Err("按大小清理功能暂未实现，请使用按时间清理".to_string());
        }
    }

    Ok(CleanupFlowsResponse {
        cleaned_count,
        cleaned_files,
        freed_bytes,
    })
}

/// 获取最近的 Flow 列表
///
/// **Validates: Requirements 10.1**
///
/// # Arguments
/// * `limit` - 最大返回数量
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(Vec<LLMFlow>)` - 成功时返回 Flow 列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_recent_flows(
    limit: Option<usize>,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<Vec<LLMFlow>, String> {
    let limit = limit.unwrap_or(20);
    Ok(query_service.0.get_recent(limit).await)
}

/// 获取 Flow Monitor 状态
///
/// **Validates: Requirements 10.1**
///
/// # Arguments
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(FlowMonitorStatus)` - 成功时返回监控状态
/// * `Err(String)` - 失败时返回错误消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMonitorStatus {
    /// 是否启用
    pub enabled: bool,
    /// 活跃 Flow 数量
    pub active_flow_count: usize,
    /// 内存中的 Flow 数量
    pub memory_flow_count: usize,
    /// 最大内存 Flow 数量
    pub max_memory_flows: usize,
}

#[tauri::command]
pub async fn get_flow_monitor_status(
    monitor: State<'_, FlowMonitorState>,
) -> Result<FlowMonitorStatus, String> {
    let config = monitor.0.config().await;
    Ok(FlowMonitorStatus {
        enabled: monitor.0.is_enabled().await,
        active_flow_count: monitor.0.active_flow_count().await,
        memory_flow_count: monitor.0.memory_flow_count().await,
        max_memory_flows: config.max_memory_flows,
    })
}

/// 获取 Flow Monitor 状态（调试用）
///
/// **Validates: Requirements 10.1**
///
/// # Arguments
/// * `monitor` - Flow 监控服务状态
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(FlowMonitorDebugInfo)` - 成功时返回调试信息
/// * `Err(String)` - 失败时返回错误消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMonitorDebugInfo {
    /// 是否启用
    pub enabled: bool,
    /// 活跃 Flow 数量
    pub active_flow_count: usize,
    /// 内存中的 Flow 数量
    pub memory_flow_count: usize,
    /// 最大内存 Flow 数量
    pub max_memory_flows: usize,
    /// 内存中的 Flow ID 列表（最多显示10个）
    pub memory_flow_ids: Vec<String>,
    /// 配置信息
    pub config_enabled: bool,
}

#[tauri::command]
pub async fn get_flow_monitor_debug_info(
    monitor: State<'_, FlowMonitorState>,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<FlowMonitorDebugInfo, String> {
    let config = monitor.0.config().await;
    let recent_flows = query_service.0.get_recent(10).await;

    Ok(FlowMonitorDebugInfo {
        enabled: monitor.0.is_enabled().await,
        active_flow_count: monitor.0.active_flow_count().await,
        memory_flow_count: monitor.0.memory_flow_count().await,
        max_memory_flows: config.max_memory_flows,
        memory_flow_ids: recent_flows.into_iter().map(|f| f.id).collect(),
        config_enabled: config.enabled,
    })
}

/// 启用 Flow Monitor
///
/// **Validates: Requirements 10.1**
///
/// # Arguments
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn enable_flow_monitor(monitor: State<'_, FlowMonitorState>) -> Result<(), String> {
    monitor.0.enable().await;
    Ok(())
}

/// 创建测试 Flow 数据（仅用于调试）
///
/// **Validates: Requirements 10.1**
///
/// # Arguments
/// * `count` - 要创建的测试 Flow 数量
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(usize)` - 成功创建的 Flow 数量
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn create_test_flows(
    count: Option<usize>,
    monitor: State<'_, FlowMonitorState>,
) -> Result<usize, String> {
    use crate::flow_monitor::{
        ClientInfo, FlowMetadata, LLMRequest, Message, MessageRole, ProviderType,
        RequestParameters, RoutingInfo,
    };
    use chrono::Utc;

    let count = count.unwrap_or(5);
    let mut created = 0;

    for i in 0..count {
        // 创建测试请求
        let request = LLMRequest {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            headers: std::collections::HashMap::new(),
            body: serde_json::json!({
                "model": format!("gpt-4-test-{}", i),
                "messages": [{"role": "user", "content": format!("测试消息 {}", i)}]
            }),
            messages: vec![Message {
                role: MessageRole::User,
                content: crate::flow_monitor::MessageContent::Text(format!("测试消息 {}", i)),
                tool_calls: None,
                tool_result: None,
                name: None,
            }],
            system_prompt: None,
            tools: None,
            model: format!("gpt-4-test-{}", i),
            original_model: None,
            parameters: RequestParameters {
                temperature: Some(0.7),
                top_p: Some(1.0),
                max_tokens: Some(1000),
                stop: None,
                stream: false,
                extra: std::collections::HashMap::new(),
            },
            size_bytes: 100 + i * 10,
            timestamp: Utc::now(),
        };

        // 创建测试元数据
        let metadata = FlowMetadata {
            provider: ProviderType::OpenAI,
            credential_id: Some(format!("test-cred-{}", i)),
            credential_name: Some(format!("测试凭证 {}", i)),
            retry_count: 0,
            client_info: ClientInfo {
                ip: Some("127.0.0.1".to_string()),
                user_agent: Some("test-agent".to_string()),
                request_id: Some(format!("test-req-{}", i)),
            },
            routing_info: RoutingInfo {
                target_url: Some("https://api.openai.com".to_string()),
                route_rule: None,
                load_balance_strategy: None,
            },
            injected_params: None,
            context_usage_percentage: Some(50.0),
        };

        // 启动 Flow
        if let Some(flow_id) = monitor.0.start_flow(request, metadata).await {
            // 模拟完成 Flow
            let response = crate::flow_monitor::LLMResponse {
                status_code: 200,
                status_text: "OK".to_string(),
                headers: std::collections::HashMap::new(),
                body: serde_json::json!({
                    "choices": [{"message": {"role": "assistant", "content": format!("测试响应 {}", i)}}]
                }),
                content: format!("测试响应 {}", i),
                thinking: None,
                tool_calls: Vec::new(),
                usage: crate::flow_monitor::TokenUsage {
                    input_tokens: 10 + i as u32,
                    output_tokens: 20 + i as u32,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    thinking_tokens: None,
                    total_tokens: 30 + i as u32 * 2,
                },
                stop_reason: Some(crate::flow_monitor::StopReason::Stop),
                size_bytes: 200 + i * 15,
                timestamp_start: Utc::now(),
                timestamp_end: Utc::now(),
                stream_info: None,
            };

            monitor.0.complete_flow(&flow_id, Some(response)).await;
            created += 1;
        }
    }

    Ok(created)
}

/// 禁用 Flow Monitor
///
/// **Validates: Requirements 10.1**
///
/// # Arguments
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn disable_flow_monitor(monitor: State<'_, FlowMonitorState>) -> Result<(), String> {
    monitor.0.disable().await;
    Ok(())
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_flows_request_default() {
        let request = QueryFlowsRequest::default();
        assert_eq!(request.page, 1);
        assert_eq!(request.page_size, 20);
        assert!(request.sort_desc);
    }

    #[test]
    fn test_search_flows_request_default_limit() {
        let request = SearchFlowsRequest {
            query: "test".to_string(),
            limit: default_search_limit(),
        };
        assert_eq!(request.limit, 50);
    }

    #[test]
    fn test_export_flows_request_serialization() {
        let request = ExportFlowsRequest {
            format: ExportFormat::JSON,
            filter: None,
            include_raw: true,
            include_stream_chunks: false,
            redact_sensitive: false,
            flow_ids: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ExportFlowsRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.format, ExportFormat::JSON);
        assert!(deserialized.include_raw);
    }
}

// ============================================================================
// 实时事件订阅命令
// ============================================================================

use tauri::{AppHandle, Emitter};

/// 订阅 Flow 实时事件
///
/// 启动一个后台任务，将 Flow 事件通过 Tauri 事件系统推送到前端。
/// 前端可以通过 `listen("flow-event", ...)` 来接收事件。
///
/// # Arguments
/// * `app` - Tauri AppHandle
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(())` - 成功启动订阅
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn subscribe_flow_events(
    app: AppHandle,
    monitor: State<'_, FlowMonitorState>,
) -> Result<(), String> {
    let mut receiver = monitor.0.subscribe();

    // 启动后台任务来转发事件
    tokio::spawn(async move {
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    // 将事件发送到前端
                    if let Err(e) = app.emit("flow-event", &event) {
                        tracing::warn!("发送 Flow 事件到前端失败: {}", e);
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("Flow 事件接收器落后 {} 条消息", n);
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    tracing::debug!("Flow 事件通道已关闭");
                    break;
                }
            }
        }
    });

    Ok(())
}

/// 获取所有可用的 Flow 标签
///
/// # Arguments
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(Vec<String>)` - 成功时返回标签列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_all_flow_tags(
    _query_service: State<'_, FlowQueryServiceState>,
) -> Result<Vec<String>, String> {
    // TODO: 实现从存储中获取所有标签
    // 目前返回空列表
    Ok(Vec::new())
}

// ============================================================================
// 过滤表达式相关命令
// ============================================================================

/// 过滤表达式解析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseFilterResult {
    /// 是否有效
    pub valid: bool,
    /// 错误信息（如果无效）
    pub error: Option<String>,
    /// 解析后的表达式（序列化为 JSON）
    pub expr: Option<FilterExpr>,
}

/// 过滤表达式帮助信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterHelpItem {
    /// 语法
    pub syntax: String,
    /// 描述
    pub description: String,
}

/// 解析过滤表达式
///
/// **Validates: Requirements 1.1-1.17**
///
/// 验证并解析过滤表达式字符串，返回解析结果。
/// 如果表达式有效，返回解析后的 AST；如果无效，返回错误信息。
///
/// # Arguments
/// * `expression` - 过滤表达式字符串
///
/// # Returns
/// * `Ok(ParseFilterResult)` - 解析结果
#[tauri::command]
pub async fn parse_filter(expression: String) -> Result<ParseFilterResult, String> {
    match FilterParser::parse(&expression) {
        Ok(expr) => Ok(ParseFilterResult {
            valid: true,
            error: None,
            expr: Some(expr),
        }),
        Err(e) => Ok(ParseFilterResult {
            valid: false,
            error: Some(e.to_string()),
            expr: None,
        }),
    }
}

/// 验证过滤表达式
///
/// **Validates: Requirements 1.17**
///
/// 仅验证过滤表达式语法是否正确，不返回解析后的 AST。
///
/// # Arguments
/// * `expression` - 过滤表达式字符串
///
/// # Returns
/// * `Ok(bool)` - 表达式是否有效
/// * `Err(String)` - 验证过程中的错误
#[tauri::command]
pub async fn validate_filter(expression: String) -> Result<bool, String> {
    Ok(FilterParser::validate(&expression).is_ok())
}

/// 获取过滤表达式帮助信息
///
/// **Validates: Requirements 1.1-1.16**
///
/// 返回所有支持的过滤表达式语法和描述。
///
/// # Returns
/// * `Ok(Vec<FilterHelpItem>)` - 帮助信息列表
#[tauri::command]
pub async fn get_filter_help_items() -> Result<Vec<FilterHelpItem>, String> {
    let items: Vec<FilterHelpItem> = FILTER_HELP
        .iter()
        .map(|(syntax, desc)| FilterHelpItem {
            syntax: syntax.to_string(),
            description: desc.to_string(),
        })
        .collect();
    Ok(items)
}

/// 获取过滤表达式帮助文本
///
/// **Validates: Requirements 1.1-1.16**
///
/// 返回格式化的帮助文本，包含所有支持的过滤表达式语法和示例。
///
/// # Returns
/// * `Ok(String)` - 帮助文本
#[tauri::command]
pub async fn get_filter_help_text() -> Result<String, String> {
    Ok(get_filter_help())
}

/// 使用过滤表达式查询 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFlowsWithExpressionRequest {
    /// 过滤表达式
    pub filter_expr: String,
    /// 排序字段
    #[serde(default)]
    pub sort_by: FlowSortBy,
    /// 是否降序
    #[serde(default = "default_true")]
    pub sort_desc: bool,
    /// 页码（从 1 开始）
    #[serde(default = "default_page")]
    pub page: usize,
    /// 每页大小
    #[serde(default = "default_page_size")]
    pub page_size: usize,
}

/// 使用过滤表达式查询 Flow
///
/// **Validates: Requirements 1.1-1.16**
///
/// 使用类似 mitmproxy 的过滤表达式语法查询 Flow。
///
/// # Arguments
/// * `request` - 查询请求参数
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(FlowQueryResult)` - 成功时返回查询结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn query_flows_with_expression(
    request: QueryFlowsWithExpressionRequest,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<FlowQueryResult, String> {
    query_service
        .0
        .query_with_expression(
            &request.filter_expr,
            request.sort_by,
            request.sort_desc,
            request.page,
            request.page_size,
        )
        .await
        .map_err(|e| format!("查询 Flow 失败: {}", e))
}

// ============================================================================
// 拦截器相关命令
// ============================================================================

use crate::flow_monitor::{
    FlowInterceptor, InterceptConfig, InterceptEvent, InterceptedFlow, InterceptorError,
    ModifiedData, TimeoutAction,
};

use crate::flow_monitor::{
    BatchReplayResult, FlowReplayer, ReplayConfig, ReplayResult, RequestModification,
};

/// 拦截器状态封装
pub struct FlowInterceptorState(pub Arc<FlowInterceptor>);

/// 重放器状态封装
pub struct FlowReplayerState(pub Arc<FlowReplayer>);

/// 获取拦截器配置
///
/// **Validates: Requirements 2.7, 2.8**
///
/// # Arguments
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(InterceptConfig)` - 成功时返回拦截器配置
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn intercept_config_get(
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<InterceptConfig, String> {
    Ok(interceptor.0.config().await)
}

/// 设置拦截器配置
///
/// **Validates: Requirements 2.7, 2.8**
///
/// # Arguments
/// * `config` - 新的拦截器配置
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn intercept_config_set(
    config: InterceptConfig,
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<(), String> {
    interceptor
        .0
        .update_config(config)
        .await
        .map_err(|e| format!("设置拦截器配置失败: {}", e))
}

/// 继续处理被拦截的 Flow
///
/// **Validates: Requirements 2.3, 2.5**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `modified_request` - 修改后的请求（可选）
/// * `modified_response` - 修改后的响应（可选）
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn intercept_continue(
    flow_id: String,
    modified_request: Option<crate::flow_monitor::LLMRequest>,
    modified_response: Option<crate::flow_monitor::LLMResponse>,
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<(), String> {
    // 确定修改数据
    let modified = if let Some(req) = modified_request {
        Some(ModifiedData::Request(req))
    } else if let Some(resp) = modified_response {
        Some(ModifiedData::Response(resp))
    } else {
        None
    };

    interceptor
        .0
        .continue_flow(&flow_id, modified)
        .await
        .map_err(|e| format!("继续处理 Flow 失败: {}", e))
}

/// 取消被拦截的 Flow
///
/// **Validates: Requirements 2.4**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn intercept_cancel(
    flow_id: String,
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<(), String> {
    interceptor
        .0
        .cancel_flow(&flow_id)
        .await
        .map_err(|e| format!("取消 Flow 失败: {}", e))
}

/// 获取被拦截的 Flow 详情
///
/// **Validates: Requirements 2.1**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(Option<InterceptedFlow>)` - 成功时返回被拦截的 Flow 详情
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn intercept_get_flow(
    flow_id: String,
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<Option<InterceptedFlow>, String> {
    Ok(interceptor.0.get_intercepted_flow(&flow_id).await)
}

/// 获取所有被拦截的 Flow 列表
///
/// **Validates: Requirements 2.1**
///
/// # Arguments
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(Vec<InterceptedFlow>)` - 成功时返回被拦截的 Flow 列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn intercept_list_flows(
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<Vec<InterceptedFlow>, String> {
    Ok(interceptor.0.list_intercepted_flows().await)
}

/// 获取被拦截的 Flow 数量
///
/// **Validates: Requirements 2.1**
///
/// # Arguments
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(usize)` - 成功时返回被拦截的 Flow 数量
#[tauri::command]
pub async fn intercept_count(
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<usize, String> {
    Ok(interceptor.0.intercepted_count().await)
}

/// 检查拦截是否启用
///
/// **Validates: Requirements 2.1**
///
/// # Arguments
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回拦截是否启用
#[tauri::command]
pub async fn intercept_is_enabled(
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<bool, String> {
    Ok(interceptor.0.is_enabled().await)
}

/// 启用拦截
///
/// **Validates: Requirements 2.1**
///
/// # Arguments
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(())` - 成功
#[tauri::command]
pub async fn intercept_enable(interceptor: State<'_, FlowInterceptorState>) -> Result<(), String> {
    interceptor.0.enable().await;
    Ok(())
}

/// 禁用拦截
///
/// **Validates: Requirements 2.1**
///
/// # Arguments
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(())` - 成功
#[tauri::command]
pub async fn intercept_disable(interceptor: State<'_, FlowInterceptorState>) -> Result<(), String> {
    interceptor.0.disable().await;
    Ok(())
}

/// 设置 Flow 为编辑状态
///
/// **Validates: Requirements 2.2**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn intercept_set_editing(
    flow_id: String,
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<(), String> {
    interceptor
        .0
        .set_editing(&flow_id)
        .await
        .map_err(|e| format!("设置编辑状态失败: {}", e))
}

/// 订阅拦截事件
///
/// **Validates: Requirements 2.1**
///
/// 启动一个后台任务，将拦截事件通过 Tauri 事件系统推送到前端。
/// 前端可以通过 `listen("intercept-event", ...)` 来接收事件。
///
/// # Arguments
/// * `app` - Tauri AppHandle
/// * `interceptor` - 拦截器状态
///
/// # Returns
/// * `Ok(())` - 成功启动订阅
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn subscribe_intercept_events(
    app: AppHandle,
    interceptor: State<'_, FlowInterceptorState>,
) -> Result<(), String> {
    let mut receiver = interceptor.0.subscribe();

    // 启动后台任务来转发事件
    tokio::spawn(async move {
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    // 将事件发送到前端
                    if let Err(e) = app.emit("intercept-event", &event) {
                        tracing::warn!("发送拦截事件到前端失败: {}", e);
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("拦截事件接收器落后 {} 条消息", n);
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    tracing::debug!("拦截事件通道已关闭");
                    break;
                }
            }
        }
    });

    Ok(())
}

// ============================================================================
// 重放器相关命令
// ============================================================================

/// 重放 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFlowRequest {
    /// 要重放的 Flow ID
    pub flow_id: String,
    /// 重放配置
    #[serde(default)]
    pub config: ReplayConfig,
}

/// 批量重放 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFlowsBatchRequest {
    /// 要重放的 Flow ID 列表
    pub flow_ids: Vec<String>,
    /// 重放配置
    #[serde(default)]
    pub config: ReplayConfig,
}

/// 重放单个 Flow
///
/// **Validates: Requirements 3.1, 3.3, 3.4**
///
/// # Arguments
/// * `request` - 重放请求参数
/// * `replayer` - 重放器状态
///
/// # Returns
/// * `Ok(ReplayResult)` - 成功时返回重放结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn replay_flow(
    request: ReplayFlowRequest,
    replayer: State<'_, FlowReplayerState>,
) -> Result<ReplayResult, String> {
    replayer
        .0
        .replay(&request.flow_id, request.config)
        .await
        .map_err(|e| format!("重放 Flow 失败: {}", e))
}

/// 批量重放多个 Flow
///
/// **Validates: Requirements 3.6, 3.7**
///
/// # Arguments
/// * `request` - 批量重放请求参数
/// * `replayer` - 重放器状态
///
/// # Returns
/// * `Ok(BatchReplayResult)` - 成功时返回批量重放结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn replay_flows_batch(
    request: ReplayFlowsBatchRequest,
    replayer: State<'_, FlowReplayerState>,
) -> Result<BatchReplayResult, String> {
    Ok(replayer
        .0
        .replay_batch(&request.flow_ids, request.config)
        .await)
}

// ============================================================================
// 差异对比命令
// ============================================================================

/// 差异对比请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffFlowsRequest {
    /// 左侧 Flow ID
    pub left_flow_id: String,
    /// 右侧 Flow ID
    pub right_flow_id: String,
    /// 差异配置
    #[serde(default)]
    pub config: DiffConfig,
}

/// 对比两个 Flow 的差异
///
/// **Validates: Requirements 4.1, 4.2**
///
/// # Arguments
/// * `request` - 差异对比请求参数
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(FlowDiffResult)` - 成功时返回差异结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn diff_flows(
    request: DiffFlowsRequest,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<FlowDiffResult, String> {
    // 获取左侧 Flow
    let left_flow = query_service
        .0
        .get_flow(&request.left_flow_id)
        .await
        .map_err(|e| format!("获取左侧 Flow 失败: {}", e))?
        .ok_or_else(|| format!("左侧 Flow 不存在: {}", request.left_flow_id))?;

    // 获取右侧 Flow
    let right_flow = query_service
        .0
        .get_flow(&request.right_flow_id)
        .await
        .map_err(|e| format!("获取右侧 Flow 失败: {}", e))?
        .ok_or_else(|| format!("右侧 Flow 不存在: {}", request.right_flow_id))?;

    // 执行差异对比
    let result = FlowDiff::diff(&left_flow, &right_flow, &request.config);

    Ok(result)
}

// ============================================================================
// 重放器测试模块
// ============================================================================

#[cfg(test)]
mod replayer_tests {
    use super::*;

    #[test]
    fn test_replay_flow_request_serialization() {
        let request = ReplayFlowRequest {
            flow_id: "test-flow-id".to_string(),
            config: ReplayConfig::default(),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ReplayFlowRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.flow_id, "test-flow-id");
        assert!(deserialized.config.credential_id.is_none());
    }

    #[test]
    fn test_replay_flows_batch_request_serialization() {
        let request = ReplayFlowsBatchRequest {
            flow_ids: vec!["flow-1".to_string(), "flow-2".to_string()],
            config: ReplayConfig {
                credential_id: Some("cred-1".to_string()),
                modify_request: None,
                interval_ms: 500,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ReplayFlowsBatchRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.flow_ids.len(), 2);
        assert_eq!(deserialized.config.interval_ms, 500);
        assert_eq!(
            deserialized.config.credential_id,
            Some("cred-1".to_string())
        );
    }

    #[test]
    fn test_replay_config_default() {
        let config = ReplayConfig::default();
        assert!(config.credential_id.is_none());
        assert!(config.modify_request.is_none());
        assert_eq!(config.interval_ms, 1000);
    }
}

// ============================================================================
// 差异对比测试模块
// ============================================================================

#[cfg(test)]
mod diff_tests {
    use super::*;

    #[test]
    fn test_diff_flows_request_serialization() {
        let request = DiffFlowsRequest {
            left_flow_id: "flow-1".to_string(),
            right_flow_id: "flow-2".to_string(),
            config: DiffConfig::default(),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: DiffFlowsRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.left_flow_id, "flow-1");
        assert_eq!(deserialized.right_flow_id, "flow-2");
        assert!(deserialized.config.ignore_timestamps);
        assert!(deserialized.config.ignore_ids);
    }

    #[test]
    fn test_diff_flows_request_with_custom_config() {
        let request = DiffFlowsRequest {
            left_flow_id: "flow-a".to_string(),
            right_flow_id: "flow-b".to_string(),
            config: DiffConfig {
                ignore_fields: vec!["custom_field".to_string()],
                ignore_timestamps: false,
                ignore_ids: false,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: DiffFlowsRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.config.ignore_fields.len(), 1);
        assert!(!deserialized.config.ignore_timestamps);
        assert!(!deserialized.config.ignore_ids);
    }
}

// ============================================================================
// 会话管理命令
// ============================================================================

use crate::flow_monitor::{AutoSessionConfig, FlowSession, SessionExportResult, SessionManager};

/// 会话管理器状态封装
pub struct SessionManagerState(pub Arc<SessionManager>);

/// 创建会话请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    /// 会话名称
    pub name: String,
    /// 会话描述（可选）
    #[serde(default)]
    pub description: Option<String>,
}

/// 更新会话请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateSessionRequest {
    /// 会话 ID
    pub session_id: String,
    /// 新名称（可选）
    #[serde(default)]
    pub name: Option<String>,
    /// 新描述（可选，None 表示不更新，Some(None) 表示清除描述）
    #[serde(default)]
    pub description: Option<Option<String>>,
}

/// 导出会话请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSessionRequest {
    /// 会话 ID
    pub session_id: String,
    /// 导出格式
    #[serde(default)]
    pub format: ExportFormat,
}

/// 创建新会话
///
/// **Validates: Requirements 5.1**
///
/// # Arguments
/// * `request` - 创建会话请求参数
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(FlowSession)` - 成功时返回新创建的会话
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn create_session(
    request: CreateSessionRequest,
    session_manager: State<'_, SessionManagerState>,
) -> Result<FlowSession, String> {
    session_manager
        .0
        .create_session(&request.name, request.description.as_deref())
        .map_err(|e| format!("创建会话失败: {}", e))
}

/// 获取会话详情
///
/// **Validates: Requirements 5.3**
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(Option<FlowSession>)` - 成功时返回会话详情
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_session(
    session_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<Option<FlowSession>, String> {
    session_manager
        .0
        .get_session(&session_id)
        .map_err(|e| format!("获取会话失败: {}", e))
}

/// 列出所有会话
///
/// **Validates: Requirements 5.3**
///
/// # Arguments
/// * `include_archived` - 是否包含已归档的会话
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(Vec<FlowSession>)` - 成功时返回会话列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn list_sessions(
    include_archived: Option<bool>,
    session_manager: State<'_, SessionManagerState>,
) -> Result<Vec<FlowSession>, String> {
    session_manager
        .0
        .list_sessions(include_archived.unwrap_or(false))
        .map_err(|e| format!("列出会话失败: {}", e))
}

/// 添加 Flow 到会话
///
/// **Validates: Requirements 5.2**
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `flow_id` - Flow ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn add_flow_to_session(
    session_id: String,
    flow_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager
        .0
        .add_flow(&session_id, &flow_id)
        .map_err(|e| format!("添加 Flow 到会话失败: {}", e))
}

/// 从会话移除 Flow
///
/// **Validates: Requirements 5.2**
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `flow_id` - Flow ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn remove_flow_from_session(
    session_id: String,
    flow_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager
        .0
        .remove_flow(&session_id, &flow_id)
        .map_err(|e| format!("从会话移除 Flow 失败: {}", e))
}

/// 更新会话信息
///
/// **Validates: Requirements 5.5**
///
/// # Arguments
/// * `request` - 更新会话请求参数
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn update_session(
    request: UpdateSessionRequest,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager
        .0
        .update_session(
            &request.session_id,
            request.name.as_deref(),
            request.description.as_ref().map(|d| d.as_deref()),
        )
        .map_err(|e| format!("更新会话失败: {}", e))
}

/// 归档会话
///
/// **Validates: Requirements 5.7**
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn archive_session(
    session_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager
        .0
        .archive_session(&session_id)
        .map_err(|e| format!("归档会话失败: {}", e))
}

/// 取消归档会话
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn unarchive_session(
    session_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager
        .0
        .unarchive_session(&session_id)
        .map_err(|e| format!("取消归档会话失败: {}", e))
}

/// 删除会话
///
/// **Validates: Requirements 5.7**
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn delete_session(
    session_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager
        .0
        .delete_session(&session_id)
        .map_err(|e| format!("删除会话失败: {}", e))
}

/// 导出会话
///
/// **Validates: Requirements 5.6**
///
/// # Arguments
/// * `request` - 导出会话请求参数
/// * `session_manager` - 会话管理器状态
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(SessionExportResult)` - 成功时返回导出结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn export_session(
    request: ExportSessionRequest,
    session_manager: State<'_, SessionManagerState>,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<SessionExportResult, String> {
    // 获取会话中的 Flow ID
    let flow_ids = session_manager
        .0
        .get_session_flow_ids(&request.session_id)
        .map_err(|e| format!("获取会话 Flow 列表失败: {}", e))?;

    // 获取所有 Flow
    let mut flows = Vec::new();
    for flow_id in &flow_ids {
        if let Ok(Some(flow)) = query_service.0.get_flow(flow_id).await {
            flows.push(flow);
        }
    }

    // 导出会话
    session_manager
        .0
        .export_session(&request.session_id, &flows, request.format)
        .map_err(|e| format!("导出会话失败: {}", e))
}

/// 获取会话中的 Flow 数量
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(usize)` - 成功时返回 Flow 数量
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_session_flow_count(
    session_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<usize, String> {
    session_manager
        .0
        .get_session_flow_count(&session_id)
        .map_err(|e| format!("获取会话 Flow 数量失败: {}", e))
}

/// 检查 Flow 是否在会话中
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `flow_id` - Flow ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否在会话中
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn is_flow_in_session(
    session_id: String,
    flow_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<bool, String> {
    session_manager
        .0
        .is_flow_in_session(&session_id, &flow_id)
        .map_err(|e| format!("检查 Flow 是否在会话中失败: {}", e))
}

/// 获取 Flow 所属的会话列表
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(Vec<String>)` - 成功时返回会话 ID 列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_sessions_for_flow(
    flow_id: String,
    session_manager: State<'_, SessionManagerState>,
) -> Result<Vec<String>, String> {
    session_manager
        .0
        .get_sessions_for_flow(&flow_id)
        .map_err(|e| format!("获取 Flow 所属会话失败: {}", e))
}

/// 获取自动会话检测配置
///
/// **Validates: Requirements 5.4**
///
/// # Arguments
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(AutoSessionConfig)` - 成功时返回配置
#[tauri::command]
pub async fn get_auto_session_config(
    session_manager: State<'_, SessionManagerState>,
) -> Result<AutoSessionConfig, String> {
    Ok(session_manager.0.get_auto_config())
}

/// 设置自动会话检测配置
///
/// **Validates: Requirements 5.4**
///
/// # Arguments
/// * `config` - 新配置
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
#[tauri::command]
pub async fn set_auto_session_config(
    config: AutoSessionConfig,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager.0.set_auto_config(config);
    Ok(())
}

/// 注册活跃会话（用于自动检测）
///
/// # Arguments
/// * `session_id` - 会话 ID
/// * `client_key` - 客户端标识（可选）
/// * `session_manager` - 会话管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
#[tauri::command]
pub async fn register_active_session(
    session_id: String,
    client_key: Option<String>,
    session_manager: State<'_, SessionManagerState>,
) -> Result<(), String> {
    session_manager
        .0
        .register_active_session(&session_id, client_key.as_deref());
    Ok(())
}

// ============================================================================
// 快速过滤器命令
// ============================================================================

use crate::flow_monitor::{QuickFilter, QuickFilterManager, QuickFilterUpdate};

/// 快速过滤器管理器状态封装
pub struct QuickFilterManagerState(pub Arc<QuickFilterManager>);

/// 保存快速过滤器请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveQuickFilterRequest {
    /// 过滤器名称
    pub name: String,
    /// 过滤表达式
    pub filter_expr: String,
    /// 描述（可选）
    #[serde(default)]
    pub description: Option<String>,
    /// 分组（可选）
    #[serde(default)]
    pub group: Option<String>,
}

/// 更新快速过滤器请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateQuickFilterRequest {
    /// 过滤器 ID
    pub id: String,
    /// 新名称（可选）
    #[serde(default)]
    pub name: Option<String>,
    /// 新描述（可选）
    #[serde(default)]
    pub description: Option<Option<String>>,
    /// 新过滤表达式（可选）
    #[serde(default)]
    pub filter_expr: Option<String>,
    /// 新分组（可选）
    #[serde(default)]
    pub group: Option<Option<String>>,
    /// 新排序顺序（可选）
    #[serde(default)]
    pub order: Option<i32>,
}

/// 导入快速过滤器请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportQuickFiltersRequest {
    /// JSON 格式的导入数据
    pub data: String,
    /// 是否覆盖同名过滤器
    #[serde(default)]
    pub overwrite: bool,
}

/// 保存快速过滤器
///
/// **Validates: Requirements 6.1**
///
/// # Arguments
/// * `request` - 保存快速过滤器请求参数
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(QuickFilter)` - 成功时返回新创建的快速过滤器
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn save_quick_filter(
    request: SaveQuickFilterRequest,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<QuickFilter, String> {
    quick_filter_manager
        .0
        .save(
            &request.name,
            &request.filter_expr,
            request.description.as_deref(),
            request.group.as_deref(),
        )
        .map_err(|e| format!("保存快速过滤器失败: {}", e))
}

/// 获取快速过滤器
///
/// # Arguments
/// * `id` - 过滤器 ID
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(Option<QuickFilter>)` - 成功时返回快速过滤器
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_quick_filter(
    id: String,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<Option<QuickFilter>, String> {
    quick_filter_manager
        .0
        .get(&id)
        .map_err(|e| format!("获取快速过滤器失败: {}", e))
}

/// 更新快速过滤器
///
/// **Validates: Requirements 6.4**
///
/// # Arguments
/// * `request` - 更新快速过滤器请求参数
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(QuickFilter)` - 成功时返回更新后的快速过滤器
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn update_quick_filter(
    request: UpdateQuickFilterRequest,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<QuickFilter, String> {
    let updates = QuickFilterUpdate {
        name: request.name,
        description: request.description,
        filter_expr: request.filter_expr,
        group: request.group,
        order: request.order,
    };

    quick_filter_manager
        .0
        .update(&request.id, updates)
        .map_err(|e| format!("更新快速过滤器失败: {}", e))
}

/// 删除快速过滤器
///
/// **Validates: Requirements 6.4**
///
/// # Arguments
/// * `id` - 过滤器 ID
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn delete_quick_filter(
    id: String,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<(), String> {
    quick_filter_manager
        .0
        .delete(&id)
        .map_err(|e| format!("删除快速过滤器失败: {}", e))
}

/// 列出所有快速过滤器
///
/// **Validates: Requirements 6.2, 6.5**
///
/// # Arguments
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(Vec<QuickFilter>)` - 成功时返回快速过滤器列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn list_quick_filters(
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<Vec<QuickFilter>, String> {
    quick_filter_manager
        .0
        .list()
        .map_err(|e| format!("列出快速过滤器失败: {}", e))
}

/// 按分组列出快速过滤器
///
/// **Validates: Requirements 6.5**
///
/// # Arguments
/// * `group` - 分组名称（可选，None 表示无分组的过滤器）
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(Vec<QuickFilter>)` - 成功时返回快速过滤器列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn list_quick_filters_by_group(
    group: Option<String>,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<Vec<QuickFilter>, String> {
    quick_filter_manager
        .0
        .list_by_group(group.as_deref())
        .map_err(|e| format!("按分组列出快速过滤器失败: {}", e))
}

/// 列出所有分组
///
/// **Validates: Requirements 6.5**
///
/// # Arguments
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(Vec<String>)` - 成功时返回分组名称列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn list_quick_filter_groups(
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<Vec<String>, String> {
    quick_filter_manager
        .0
        .list_groups()
        .map_err(|e| format!("列出快速过滤器分组失败: {}", e))
}

/// 导出快速过滤器
///
/// **Validates: Requirements 6.7**
///
/// # Arguments
/// * `include_presets` - 是否包含预设过滤器
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(String)` - 成功时返回 JSON 格式的导出数据
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn export_quick_filters(
    include_presets: Option<bool>,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<String, String> {
    quick_filter_manager
        .0
        .export(include_presets.unwrap_or(false))
        .map_err(|e| format!("导出快速过滤器失败: {}", e))
}

/// 导入快速过滤器
///
/// **Validates: Requirements 6.7**
///
/// # Arguments
/// * `request` - 导入快速过滤器请求参数
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(Vec<QuickFilter>)` - 成功时返回导入的快速过滤器列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn import_quick_filters(
    request: ImportQuickFiltersRequest,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<Vec<QuickFilter>, String> {
    quick_filter_manager
        .0
        .import(&request.data, request.overwrite)
        .map_err(|e| format!("导入快速过滤器失败: {}", e))
}

/// 按名称查找快速过滤器
///
/// # Arguments
/// * `name` - 过滤器名称
/// * `quick_filter_manager` - 快速过滤器管理器状态
///
/// # Returns
/// * `Ok(Option<QuickFilter>)` - 成功时返回快速过滤器
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn find_quick_filter_by_name(
    name: String,
    quick_filter_manager: State<'_, QuickFilterManagerState>,
) -> Result<Option<QuickFilter>, String> {
    quick_filter_manager
        .0
        .find_by_name(&name)
        .map_err(|e| format!("查找快速过滤器失败: {}", e))
}

// ============================================================================
// 代码导出命令
// ============================================================================

use crate::flow_monitor::{CodeExporter, CodeFormat};

/// 代码导出请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFlowAsCodeRequest {
    /// Flow ID
    pub flow_id: String,
    /// 导出格式
    pub format: CodeFormat,
}

/// 代码导出响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFlowAsCodeResponse {
    /// 导出的代码
    pub code: String,
    /// 导出格式
    pub format: CodeFormat,
}

/// 将 Flow 导出为代码
///
/// **Validates: Requirements 7.7, 7.8**
///
/// 将指定的 Flow 导出为可执行的代码格式（curl、Python、TypeScript、JavaScript）。
///
/// # Arguments
/// * `request` - 代码导出请求参数
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(ExportFlowAsCodeResponse)` - 成功时返回导出的代码
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn export_flow_as_code(
    request: ExportFlowAsCodeRequest,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<ExportFlowAsCodeResponse, String> {
    // 获取 Flow
    let flow = query_service
        .0
        .get_flow(&request.flow_id)
        .await
        .map_err(|e| format!("获取 Flow 失败: {}", e))?
        .ok_or_else(|| format!("Flow 不存在: {}", request.flow_id))?;

    // 导出为代码
    let code = CodeExporter::export(&flow, request.format);

    Ok(ExportFlowAsCodeResponse {
        code,
        format: request.format,
    })
}

/// 批量导出 Flow 为代码
///
/// **Validates: Requirements 7.7, 7.8**
///
/// 将多个 Flow 导出为可执行的代码格式。
///
/// # Arguments
/// * `flow_ids` - Flow ID 列表
/// * `format` - 导出格式
/// * `query_service` - 查询服务状态
///
/// # Returns
/// * `Ok(Vec<ExportFlowAsCodeResponse>)` - 成功时返回导出的代码列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn export_flows_as_code(
    flow_ids: Vec<String>,
    format: CodeFormat,
    query_service: State<'_, FlowQueryServiceState>,
) -> Result<Vec<ExportFlowAsCodeResponse>, String> {
    let mut results = Vec::new();

    for flow_id in flow_ids {
        if let Ok(Some(flow)) = query_service.0.get_flow(&flow_id).await {
            let code = CodeExporter::export(&flow, format);
            results.push(ExportFlowAsCodeResponse { code, format });
        }
    }

    Ok(results)
}

/// 获取支持的代码导出格式
///
/// **Validates: Requirements 7.7, 7.8**
///
/// 返回所有支持的代码导出格式列表。
///
/// # Returns
/// * `Ok(Vec<CodeFormatInfo>)` - 成功时返回格式列表
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFormatInfo {
    /// 格式标识
    pub format: CodeFormat,
    /// 格式名称
    pub name: String,
    /// 格式描述
    pub description: String,
}

#[tauri::command]
pub async fn get_code_export_formats() -> Result<Vec<CodeFormatInfo>, String> {
    Ok(vec![
        CodeFormatInfo {
            format: CodeFormat::Curl,
            name: "curl".to_string(),
            description: "curl 命令行工具".to_string(),
        },
        CodeFormatInfo {
            format: CodeFormat::Python,
            name: "Python".to_string(),
            description: "Python requests 库".to_string(),
        },
        CodeFormatInfo {
            format: CodeFormat::TypeScript,
            name: "TypeScript".to_string(),
            description: "TypeScript fetch API".to_string(),
        },
        CodeFormatInfo {
            format: CodeFormat::JavaScript,
            name: "JavaScript".to_string(),
            description: "JavaScript fetch API".to_string(),
        },
    ])
}

// ============================================================================
// 书签管理命令
// ============================================================================

use crate::flow_monitor::{BookmarkExport, BookmarkManager, FlowBookmark};

/// 书签管理器状态封装
pub struct BookmarkManagerState(pub Arc<BookmarkManager>);

/// 添加书签请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddBookmarkRequest {
    /// Flow ID
    pub flow_id: String,
    /// 书签名称（可选）
    #[serde(default)]
    pub name: Option<String>,
    /// 分组名称（可选）
    #[serde(default)]
    pub group: Option<String>,
}

/// 更新书签请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateBookmarkRequest {
    /// 书签 ID
    pub bookmark_id: String,
    /// 新名称（可选）
    #[serde(default)]
    pub name: Option<Option<String>>,
    /// 新分组（可选）
    #[serde(default)]
    pub group: Option<Option<String>>,
}

/// 导入书签请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportBookmarksRequest {
    /// JSON 格式的导入数据
    pub data: String,
    /// 是否覆盖已存在的书签
    #[serde(default)]
    pub overwrite: bool,
}

/// 添加书签
///
/// **Validates: Requirements 8.1**
///
/// # Arguments
/// * `request` - 添加书签请求参数
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(FlowBookmark)` - 成功时返回新创建的书签
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn add_bookmark(
    request: AddBookmarkRequest,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<FlowBookmark, String> {
    bookmark_manager
        .0
        .add(
            &request.flow_id,
            request.name.as_deref(),
            request.group.as_deref(),
        )
        .map_err(|e| format!("添加书签失败: {}", e))
}

/// 获取书签
///
/// # Arguments
/// * `bookmark_id` - 书签 ID
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(Option<FlowBookmark>)` - 成功时返回书签
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_bookmark(
    bookmark_id: String,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<Option<FlowBookmark>, String> {
    bookmark_manager
        .0
        .get(&bookmark_id)
        .map_err(|e| format!("获取书签失败: {}", e))
}

/// 根据 Flow ID 获取书签
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(Option<FlowBookmark>)` - 成功时返回书签
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_bookmark_by_flow_id(
    flow_id: String,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<Option<FlowBookmark>, String> {
    bookmark_manager
        .0
        .get_by_flow_id(&flow_id)
        .map_err(|e| format!("获取书签失败: {}", e))
}

/// 移除书签
///
/// **Validates: Requirements 8.1**
///
/// # Arguments
/// * `bookmark_id` - 书签 ID
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn remove_bookmark(
    bookmark_id: String,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<(), String> {
    bookmark_manager
        .0
        .remove(&bookmark_id)
        .map_err(|e| format!("移除书签失败: {}", e))
}

/// 根据 Flow ID 移除书签
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn remove_bookmark_by_flow_id(
    flow_id: String,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<(), String> {
    bookmark_manager
        .0
        .remove_by_flow_id(&flow_id)
        .map_err(|e| format!("移除书签失败: {}", e))
}

/// 更新书签
///
/// # Arguments
/// * `request` - 更新书签请求参数
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(FlowBookmark)` - 成功时返回更新后的书签
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn update_bookmark(
    request: UpdateBookmarkRequest,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<FlowBookmark, String> {
    bookmark_manager
        .0
        .update(
            &request.bookmark_id,
            request.name.as_ref().map(|n| n.as_deref()),
            request.group.as_ref().map(|g| g.as_deref()),
        )
        .map_err(|e| format!("更新书签失败: {}", e))
}

/// 列出所有书签
///
/// **Validates: Requirements 8.3**
///
/// # Arguments
/// * `group` - 分组名称（可选，None 表示所有书签）
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(Vec<FlowBookmark>)` - 成功时返回书签列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn list_bookmarks(
    group: Option<String>,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<Vec<FlowBookmark>, String> {
    bookmark_manager
        .0
        .list(group.as_deref())
        .map_err(|e| format!("列出书签失败: {}", e))
}

/// 列出所有书签分组
///
/// **Validates: Requirements 8.3**
///
/// # Arguments
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(Vec<String>)` - 成功时返回分组名称列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn list_bookmark_groups(
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<Vec<String>, String> {
    bookmark_manager
        .0
        .list_groups()
        .map_err(|e| format!("列出书签分组失败: {}", e))
}

/// 检查 Flow 是否已添加书签
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否已添加书签
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn is_flow_bookmarked(
    flow_id: String,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<bool, String> {
    bookmark_manager
        .0
        .is_bookmarked(&flow_id)
        .map_err(|e| format!("检查书签状态失败: {}", e))
}

/// 获取书签数量
///
/// # Arguments
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(usize)` - 成功时返回书签数量
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_bookmark_count(
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<usize, String> {
    bookmark_manager
        .0
        .count()
        .map_err(|e| format!("获取书签数量失败: {}", e))
}

/// 导出书签
///
/// **Validates: Requirements 8.6**
///
/// # Arguments
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(String)` - 成功时返回 JSON 格式的导出数据
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn export_bookmarks(
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<String, String> {
    bookmark_manager
        .0
        .export()
        .map_err(|e| format!("导出书签失败: {}", e))
}

/// 导入书签
///
/// **Validates: Requirements 8.6**
///
/// # Arguments
/// * `request` - 导入书签请求参数
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(Vec<FlowBookmark>)` - 成功时返回导入的书签列表
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn import_bookmarks(
    request: ImportBookmarksRequest,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<Vec<FlowBookmark>, String> {
    bookmark_manager
        .0
        .import(&request.data, request.overwrite)
        .map_err(|e| format!("导入书签失败: {}", e))
}

/// 切换书签状态
///
/// 如果 Flow 已添加书签则移除，否则添加书签。
///
/// **Validates: Requirements 8.1**
///
/// # Arguments
/// * `flow_id` - Flow ID
/// * `name` - 书签名称（可选，仅在添加时使用）
/// * `group` - 分组名称（可选，仅在添加时使用）
/// * `bookmark_manager` - 书签管理器状态
///
/// # Returns
/// * `Ok(Option<FlowBookmark>)` - 成功时返回书签（如果添加）或 None（如果移除）
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn toggle_bookmark(
    flow_id: String,
    name: Option<String>,
    group: Option<String>,
    bookmark_manager: State<'_, BookmarkManagerState>,
) -> Result<Option<FlowBookmark>, String> {
    let is_bookmarked = bookmark_manager
        .0
        .is_bookmarked(&flow_id)
        .map_err(|e| format!("检查书签状态失败: {}", e))?;

    if is_bookmarked {
        bookmark_manager
            .0
            .remove_by_flow_id(&flow_id)
            .map_err(|e| format!("移除书签失败: {}", e))?;
        Ok(None)
    } else {
        let bookmark = bookmark_manager
            .0
            .add(&flow_id, name.as_deref(), group.as_deref())
            .map_err(|e| format!("添加书签失败: {}", e))?;
        Ok(Some(bookmark))
    }
}

// ============================================================================
// 增强统计相关命令
// ============================================================================

use crate::flow_monitor::{
    Distribution, EnhancedStats, EnhancedStatsService, ReportFormat, StatsTimeRange,
    TimeSeriesPoint, TrendData,
};

/// 增强统计服务状态封装
pub struct EnhancedStatsServiceState(pub Arc<EnhancedStatsService>);

/// 获取增强统计请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetEnhancedStatsRequest {
    /// 过滤条件
    #[serde(default)]
    pub filter: FlowFilter,
    /// 时间范围
    #[serde(default)]
    pub time_range: StatsTimeRange,
}

/// 获取请求趋势请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetRequestTrendRequest {
    /// 过滤条件
    #[serde(default)]
    pub filter: FlowFilter,
    /// 时间范围
    #[serde(default)]
    pub time_range: StatsTimeRange,
    /// 时间间隔（如 "1h", "30m", "1d"）
    #[serde(default = "default_interval")]
    pub interval: String,
}

fn default_interval() -> String {
    "1h".to_string()
}

/// 获取延迟直方图请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetLatencyHistogramRequest {
    /// 过滤条件
    #[serde(default)]
    pub filter: FlowFilter,
    /// 时间范围
    #[serde(default)]
    pub time_range: StatsTimeRange,
    /// 直方图桶边界（毫秒）
    #[serde(default = "default_latency_buckets")]
    pub buckets: Vec<u64>,
}

fn default_latency_buckets() -> Vec<u64> {
    vec![100, 500, 1000, 2000, 5000, 10000]
}

/// 导出统计报告请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStatsReportRequest {
    /// 过滤条件
    #[serde(default)]
    pub filter: FlowFilter,
    /// 时间范围
    #[serde(default)]
    pub time_range: StatsTimeRange,
    /// 报告格式
    #[serde(default)]
    pub format: ReportFormat,
}

/// 获取增强统计
///
/// **Validates: Requirements 9.1-9.5**
///
/// # Arguments
/// * `request` - 获取增强统计请求参数
/// * `stats_service` - 增强统计服务状态
///
/// # Returns
/// * `Ok(EnhancedStats)` - 成功时返回增强统计结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_enhanced_stats(
    request: GetEnhancedStatsRequest,
    stats_service: State<'_, EnhancedStatsServiceState>,
) -> Result<EnhancedStats, String> {
    Ok(stats_service
        .0
        .get_stats(&request.filter, &request.time_range)
        .await)
}

/// 获取请求趋势
///
/// **Validates: Requirements 9.1**
///
/// # Arguments
/// * `request` - 获取请求趋势请求参数
/// * `stats_service` - 增强统计服务状态
///
/// # Returns
/// * `Ok(TrendData)` - 成功时返回趋势数据
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_request_trend(
    request: GetRequestTrendRequest,
    stats_service: State<'_, EnhancedStatsServiceState>,
) -> Result<TrendData, String> {
    Ok(stats_service
        .0
        .get_request_trend(&request.filter, &request.time_range, &request.interval)
        .await)
}

/// 获取 Token 分布
///
/// **Validates: Requirements 9.2**
///
/// # Arguments
/// * `request` - 获取增强统计请求参数（复用）
/// * `stats_service` - 增强统计服务状态
///
/// # Returns
/// * `Ok(Distribution)` - 成功时返回 Token 分布数据
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_token_distribution(
    request: GetEnhancedStatsRequest,
    stats_service: State<'_, EnhancedStatsServiceState>,
) -> Result<Distribution, String> {
    Ok(stats_service
        .0
        .get_token_distribution(&request.filter, &request.time_range)
        .await)
}

/// 获取延迟直方图
///
/// **Validates: Requirements 9.4**
///
/// # Arguments
/// * `request` - 获取延迟直方图请求参数
/// * `stats_service` - 增强统计服务状态
///
/// # Returns
/// * `Ok(Distribution)` - 成功时返回延迟直方图数据
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_latency_histogram(
    request: GetLatencyHistogramRequest,
    stats_service: State<'_, EnhancedStatsServiceState>,
) -> Result<Distribution, String> {
    Ok(stats_service
        .0
        .get_latency_histogram(&request.filter, &request.time_range, &request.buckets)
        .await)
}

/// 导出统计报告
///
/// **Validates: Requirements 9.7**
///
/// # Arguments
/// * `request` - 导出统计报告请求参数
/// * `stats_service` - 增强统计服务状态
///
/// # Returns
/// * `Ok(String)` - 成功时返回格式化的报告字符串
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn export_stats_report(
    request: ExportStatsReportRequest,
    stats_service: State<'_, EnhancedStatsServiceState>,
) -> Result<String, String> {
    Ok(stats_service
        .0
        .export_report(&request.filter, &request.time_range, &request.format)
        .await)
}
// ============================================================================
// 批量操作状态封装
// ============================================================================

/// BatchOperations 状态封装
pub struct BatchOperationsState(pub Arc<BatchOperations>);

// ============================================================================
// 批量操作请求/响应类型
// ============================================================================

/// 批量收藏 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStarFlowsRequest {
    /// Flow ID 列表
    pub flow_ids: Vec<String>,
}

/// 批量取消收藏 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUnstarFlowsRequest {
    /// Flow ID 列表
    pub flow_ids: Vec<String>,
}

/// 批量添加标签请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAddTagsRequest {
    /// Flow ID 列表
    pub flow_ids: Vec<String>,
    /// 要添加的标签列表
    pub tags: Vec<String>,
}

/// 批量移除标签请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRemoveTagsRequest {
    /// Flow ID 列表
    pub flow_ids: Vec<String>,
    /// 要移除的标签列表
    pub tags: Vec<String>,
}

/// 批量导出 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExportFlowsRequest {
    /// Flow ID 列表
    pub flow_ids: Vec<String>,
    /// 导出格式
    pub format: ExportFormat,
}

/// 批量删除 Flow 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchDeleteFlowsRequest {
    /// Flow ID 列表
    pub flow_ids: Vec<String>,
}

/// 批量添加到会话请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAddToSessionRequest {
    /// Flow ID 列表
    pub flow_ids: Vec<String>,
    /// 会话 ID
    pub session_id: String,
}

// ============================================================================
// 批量操作 Tauri 命令
// ============================================================================

/// 批量收藏 Flow
///
/// **Validates: Requirements 11.2**
///
/// # Arguments
/// * `request` - 批量收藏请求参数
/// * `batch_ops` - 批量操作服务状态
///
/// # Returns
/// * `Ok(BatchResult)` - 成功时返回批量操作结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn batch_star_flows(
    request: BatchStarFlowsRequest,
    batch_ops: State<'_, BatchOperationsState>,
) -> Result<BatchResult, String> {
    Ok(batch_ops
        .0
        .execute(&request.flow_ids, BatchOperation::Star)
        .await)
}

/// 批量取消收藏 Flow
///
/// **Validates: Requirements 11.2**
///
/// # Arguments
/// * `request` - 批量取消收藏请求参数
/// * `batch_ops` - 批量操作服务状态
///
/// # Returns
/// * `Ok(BatchResult)` - 成功时返回批量操作结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn batch_unstar_flows(
    request: BatchUnstarFlowsRequest,
    batch_ops: State<'_, BatchOperationsState>,
) -> Result<BatchResult, String> {
    Ok(batch_ops
        .0
        .execute(&request.flow_ids, BatchOperation::Unstar)
        .await)
}

/// 批量添加标签
///
/// **Validates: Requirements 11.3**
///
/// # Arguments
/// * `request` - 批量添加标签请求参数
/// * `batch_ops` - 批量操作服务状态
///
/// # Returns
/// * `Ok(BatchResult)` - 成功时返回批量操作结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn batch_add_tags(
    request: BatchAddTagsRequest,
    batch_ops: State<'_, BatchOperationsState>,
) -> Result<BatchResult, String> {
    Ok(batch_ops
        .0
        .execute(
            &request.flow_ids,
            BatchOperation::AddTags { tags: request.tags },
        )
        .await)
}

/// 批量移除标签
///
/// **Validates: Requirements 11.4**
///
/// # Arguments
/// * `request` - 批量移除标签请求参数
/// * `batch_ops` - 批量操作服务状态
///
/// # Returns
/// * `Ok(BatchResult)` - 成功时返回批量操作结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn batch_remove_tags(
    request: BatchRemoveTagsRequest,
    batch_ops: State<'_, BatchOperationsState>,
) -> Result<BatchResult, String> {
    Ok(batch_ops
        .0
        .execute(
            &request.flow_ids,
            BatchOperation::RemoveTags { tags: request.tags },
        )
        .await)
}

/// 批量导出 Flow
///
/// **Validates: Requirements 11.5**
///
/// # Arguments
/// * `request` - 批量导出请求参数
/// * `batch_ops` - 批量操作服务状态
///
/// # Returns
/// * `Ok(BatchResult)` - 成功时返回批量操作结果（包含导出数据）
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn batch_export_flows(
    request: BatchExportFlowsRequest,
    batch_ops: State<'_, BatchOperationsState>,
) -> Result<BatchResult, String> {
    Ok(batch_ops
        .0
        .execute(
            &request.flow_ids,
            BatchOperation::Export {
                format: request.format,
            },
        )
        .await)
}

/// 批量删除 Flow
///
/// **Validates: Requirements 11.6**
///
/// # Arguments
/// * `request` - 批量删除请求参数
/// * `batch_ops` - 批量操作服务状态
///
/// # Returns
/// * `Ok(BatchResult)` - 成功时返回批量操作结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn batch_delete_flows(
    request: BatchDeleteFlowsRequest,
    batch_ops: State<'_, BatchOperationsState>,
) -> Result<BatchResult, String> {
    Ok(batch_ops
        .0
        .execute(&request.flow_ids, BatchOperation::Delete)
        .await)
}

/// 批量添加到会话
///
/// **Validates: Requirements 11.2-11.6**
///
/// # Arguments
/// * `request` - 批量添加到会话请求参数
/// * `batch_ops` - 批量操作服务状态
///
/// # Returns
/// * `Ok(BatchResult)` - 成功时返回批量操作结果
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn batch_add_to_session(
    request: BatchAddToSessionRequest,
    batch_ops: State<'_, BatchOperationsState>,
) -> Result<BatchResult, String> {
    Ok(batch_ops
        .0
        .execute(
            &request.flow_ids,
            BatchOperation::AddToSession {
                session_id: request.session_id,
            },
        )
        .await)
}

// ============================================================================
// 实时监控增强命令
// ============================================================================

use crate::flow_monitor::{ThresholdCheckResult, ThresholdConfig};

/// 阈值配置响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfigResponse {
    /// 是否启用阈值检测
    pub enabled: bool,
    /// 延迟阈值（毫秒）
    pub latency_threshold_ms: u64,
    /// Token 使用量阈值
    pub token_threshold: u32,
    /// 输入 Token 阈值（可选）
    pub input_token_threshold: Option<u32>,
    /// 输出 Token 阈值（可选）
    pub output_token_threshold: Option<u32>,
}

impl From<ThresholdConfig> for ThresholdConfigResponse {
    fn from(config: ThresholdConfig) -> Self {
        Self {
            enabled: config.enabled,
            latency_threshold_ms: config.latency_threshold_ms,
            token_threshold: config.token_threshold,
            input_token_threshold: config.input_token_threshold,
            output_token_threshold: config.output_token_threshold,
        }
    }
}

/// 请求速率响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRateResponse {
    /// 请求速率（每秒）
    pub rate: f64,
    /// 时间窗口内的请求数量
    pub count: usize,
    /// 时间窗口（秒）
    pub window_seconds: i64,
}

/// 获取阈值配置
///
/// **Validates: Requirements 10.3, 10.4**
///
/// # Arguments
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(ThresholdConfigResponse)` - 成功时返回阈值配置
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_threshold_config(
    monitor: State<'_, FlowMonitorState>,
) -> Result<ThresholdConfigResponse, String> {
    let config = monitor.0.threshold_config().await;
    Ok(ThresholdConfigResponse::from(config))
}

/// 更新阈值配置
///
/// **Validates: Requirements 10.3, 10.4**
///
/// # Arguments
/// * `config` - 新的阈值配置
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn update_threshold_config(
    config: ThresholdConfig,
    monitor: State<'_, FlowMonitorState>,
) -> Result<(), String> {
    monitor.0.update_threshold_config(config).await;
    Ok(())
}

/// 获取请求速率
///
/// **Validates: Requirements 10.7**
///
/// # Arguments
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(RequestRateResponse)` - 成功时返回请求速率信息
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_request_rate(
    monitor: State<'_, FlowMonitorState>,
) -> Result<RequestRateResponse, String> {
    let rate = monitor.0.get_request_rate().await;
    let count = monitor.0.get_request_count().await;

    Ok(RequestRateResponse {
        rate,
        count,
        window_seconds: 60, // 默认 60 秒窗口
    })
}

/// 设置请求速率追踪器的时间窗口
///
/// **Validates: Requirements 10.7**
///
/// # Arguments
/// * `window_seconds` - 时间窗口（秒）
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn set_rate_window(
    window_seconds: i64,
    monitor: State<'_, FlowMonitorState>,
) -> Result<(), String> {
    if window_seconds <= 0 {
        return Err("时间窗口必须大于 0".to_string());
    }
    monitor.0.set_rate_window(window_seconds).await;
    Ok(())
}
// ============================================================================
// 通知配置命令
// ============================================================================

/*
/// 通知配置响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfigResponse {
    /// 是否启用通知
    pub enabled: bool,
    /// 新 Flow 通知配置
    pub new_flow: NotificationSettingsResponse,
    /// 错误 Flow 通知配置
    pub error_flow: NotificationSettingsResponse,
    /// 延迟警告通知配置
    pub latency_warning: NotificationSettingsResponse,
    /// Token 警告通知配置
    pub token_warning: NotificationSettingsResponse,
}

/// 通知设置响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettingsResponse {
    /// 是否启用
    pub enabled: bool,
    /// 是否显示桌面通知
    pub desktop: bool,
    /// 是否播放声音
    pub sound: bool,
    /// 声音文件路径（可选）
    pub sound_file: Option<String>,
}

impl From<NotificationSettings> for NotificationSettingsResponse {
    fn from(settings: NotificationSettings) -> Self {
        Self {
            enabled: settings.enabled,
            desktop: settings.desktop,
            sound: settings.sound,
            sound_file: settings.sound_file,
        }
    }
}

impl From<NotificationConfig> for NotificationConfigResponse {
    fn from(config: NotificationConfig) -> Self {
        Self {
            enabled: config.enabled,
            new_flow: NotificationSettingsResponse::from(config.new_flow),
            error_flow: NotificationSettingsResponse::from(config.error_flow),
            latency_warning: NotificationSettingsResponse::from(config.latency_warning),
            token_warning: NotificationSettingsResponse::from(config.token_warning),
        }
    }
}

/// 获取通知配置
///
/// **Validates: Requirements 10.1, 10.2**
///
/// # Arguments
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(NotificationConfigResponse)` - 成功时返回通知配置
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_notification_config(
    monitor: State<'_, FlowMonitorState>,
) -> Result<NotificationConfigResponse, String> {
    let config = monitor.0.notification_config().await;
    Ok(NotificationConfigResponse::from(config))
}

/// 更新通知配置
///
/// **Validates: Requirements 10.1, 10.2**
///
/// # Arguments
/// * `config` - 新的通知配置
/// * `monitor` - Flow 监控服务状态
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn update_notification_config(
    config: NotificationConfig,
    monitor: State<'_, FlowMonitorState>,
) -> Result<(), String> {
    monitor.0.update_notification_config(config).await;
    Ok(())
}
*/
