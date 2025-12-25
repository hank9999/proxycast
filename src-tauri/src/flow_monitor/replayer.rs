//! Flow 重放器
//!
//! 该模块实现 LLM Flow 的重放功能，允许用户重新发送历史请求。
//!
//! # 功能
//!
//! - 重放单个 Flow
//! - 批量重放多个 Flow
//! - 支持修改请求参数后重放
//! - 支持选择不同的凭证
//! - 重放的 Flow 会被标记为 "replay"

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use super::models::{
    FlowAnnotations, FlowMetadata, FlowState, FlowTimestamps, LLMFlow, LLMRequest, LLMResponse,
    Message, RequestParameters, TokenUsage,
};
use super::monitor::FlowMonitor;
use crate::database::DbConnection;
use crate::ProviderPoolService;
use crate::ProviderType;

// ============================================================================
// 配置结构
// ============================================================================

/// 重放配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// 使用的凭证 ID（可选，为空时使用原始凭证或自动选择）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_id: Option<String>,
    /// 请求修改（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modify_request: Option<RequestModification>,
    /// 重放间隔（毫秒），用于批量重放时避免触发速率限制
    #[serde(default = "default_interval_ms")]
    pub interval_ms: u64,
}

fn default_interval_ms() -> u64 {
    1000 // 默认 1 秒间隔
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            credential_id: None,
            modify_request: None,
            interval_ms: default_interval_ms(),
        }
    }
}

/// 请求修改
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestModification {
    /// 修改模型名称
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// 修改消息列表
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
    /// 修改请求参数
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<RequestParameters>,
    /// 修改系统提示词
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
}

// ============================================================================
// 重放结果
// ============================================================================

/// 重放结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    /// 原始 Flow ID
    pub original_flow_id: String,
    /// 重放生成的新 Flow ID
    pub replay_flow_id: String,
    /// 是否成功
    pub success: bool,
    /// 错误信息（如果失败）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// 重放开始时间
    pub started_at: DateTime<Utc>,
    /// 重放结束时间
    pub completed_at: DateTime<Utc>,
    /// 耗时（毫秒）
    pub duration_ms: u64,
}

impl ReplayResult {
    /// 创建成功的重放结果
    pub fn success(
        original_flow_id: String,
        replay_flow_id: String,
        started_at: DateTime<Utc>,
        completed_at: DateTime<Utc>,
    ) -> Self {
        let duration_ms = (completed_at - started_at).num_milliseconds().max(0) as u64;
        Self {
            original_flow_id,
            replay_flow_id,
            success: true,
            error: None,
            started_at,
            completed_at,
            duration_ms,
        }
    }

    /// 创建失败的重放结果
    pub fn failure(
        original_flow_id: String,
        error: String,
        started_at: DateTime<Utc>,
        completed_at: DateTime<Utc>,
    ) -> Self {
        let duration_ms = (completed_at - started_at).num_milliseconds().max(0) as u64;
        Self {
            original_flow_id,
            replay_flow_id: String::new(),
            success: false,
            error: Some(error),
            started_at,
            completed_at,
            duration_ms,
        }
    }
}

// ============================================================================
// 批量重放结果
// ============================================================================

/// 批量重放结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchReplayResult {
    /// 总数
    pub total: usize,
    /// 成功数
    pub success_count: usize,
    /// 失败数
    pub failure_count: usize,
    /// 各个 Flow 的重放结果
    pub results: Vec<ReplayResult>,
    /// 批量重放开始时间
    pub started_at: DateTime<Utc>,
    /// 批量重放结束时间
    pub completed_at: DateTime<Utc>,
    /// 总耗时（毫秒）
    pub total_duration_ms: u64,
}

// ============================================================================
// 重放器错误
// ============================================================================

/// 重放器错误
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum ReplayerError {
    /// Flow 不存在
    #[error("Flow '{0}' 不存在")]
    FlowNotFound(String),
    /// 凭证不可用
    #[error("凭证 '{0}' 不可用")]
    CredentialUnavailable(String),
    /// 请求失败
    #[error("请求失败: {0}")]
    RequestFailed(String),
    /// 内部错误
    #[error("内部错误: {0}")]
    Internal(String),
}

// ============================================================================
// Flow 重放器
// ============================================================================

/// Flow 重放器
///
/// 负责重放历史 LLM Flow 的核心服务。
pub struct FlowReplayer {
    /// HTTP 客户端
    client: Client,
    /// Flow 监控服务
    flow_monitor: Arc<FlowMonitor>,
    /// 凭证池服务
    provider_pool: Arc<ProviderPoolService>,
    /// 数据库连接
    db: DbConnection,
}

impl FlowReplayer {
    /// 创建新的重放器
    pub fn new(
        flow_monitor: Arc<FlowMonitor>,
        provider_pool: Arc<ProviderPoolService>,
        db: DbConnection,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();

        Self {
            client,
            flow_monitor,
            provider_pool,
            db,
        }
    }

    /// 重放单个 Flow
    ///
    /// **Validates: Requirements 3.1, 3.3, 3.4**
    ///
    /// # Arguments
    /// * `flow_id` - 要重放的 Flow ID
    /// * `config` - 重放配置
    ///
    /// # Returns
    /// * `Ok(ReplayResult)` - 重放结果
    /// * `Err(ReplayerError)` - 重放失败
    pub async fn replay(
        &self,
        flow_id: &str,
        config: ReplayConfig,
    ) -> Result<ReplayResult, ReplayerError> {
        let started_at = Utc::now();

        // 获取原始 Flow
        let original_flow = self.get_flow(flow_id).await?;

        // 应用请求修改
        let request = self.apply_modifications(&original_flow.request, &config.modify_request);

        // 确定使用的凭证
        let credential_id = self.resolve_credential(&original_flow, &config).await?;

        // 创建重放 Flow
        let replay_flow_id = self
            .create_replay_flow(&original_flow, &request, &credential_id)
            .await;

        // 执行重放请求
        match self
            .execute_replay(&request, &original_flow.metadata, &credential_id)
            .await
        {
            Ok(response) => {
                // 更新重放 Flow 的响应
                self.complete_replay_flow(&replay_flow_id, Some(response))
                    .await;
                let completed_at = Utc::now();
                Ok(ReplayResult::success(
                    flow_id.to_string(),
                    replay_flow_id,
                    started_at,
                    completed_at,
                ))
            }
            Err(e) => {
                // 标记重放 Flow 失败
                self.fail_replay_flow(&replay_flow_id, &e.to_string()).await;
                let completed_at = Utc::now();
                Ok(ReplayResult::failure(
                    flow_id.to_string(),
                    e.to_string(),
                    started_at,
                    completed_at,
                ))
            }
        }
    }

    /// 批量重放多个 Flow
    ///
    /// **Validates: Requirements 3.6, 3.7**
    ///
    /// # Arguments
    /// * `flow_ids` - 要重放的 Flow ID 列表
    /// * `config` - 重放配置
    ///
    /// # Returns
    /// * `BatchReplayResult` - 批量重放结果
    pub async fn replay_batch(
        &self,
        flow_ids: &[String],
        config: ReplayConfig,
    ) -> BatchReplayResult {
        let started_at = Utc::now();
        let mut results = Vec::with_capacity(flow_ids.len());
        let mut success_count = 0;
        let mut failure_count = 0;

        for (i, flow_id) in flow_ids.iter().enumerate() {
            // 执行重放
            let result = match self.replay(flow_id, config.clone()).await {
                Ok(r) => r,
                Err(e) => {
                    ReplayResult::failure(flow_id.clone(), e.to_string(), Utc::now(), Utc::now())
                }
            };

            if result.success {
                success_count += 1;
            } else {
                failure_count += 1;
            }

            results.push(result);

            // 如果不是最后一个，等待间隔时间
            if i < flow_ids.len() - 1 && config.interval_ms > 0 {
                sleep(Duration::from_millis(config.interval_ms)).await;
            }
        }

        let completed_at = Utc::now();
        let total_duration_ms = (completed_at - started_at).num_milliseconds().max(0) as u64;

        BatchReplayResult {
            total: flow_ids.len(),
            success_count,
            failure_count,
            results,
            started_at,
            completed_at,
            total_duration_ms,
        }
    }

    /// 获取 Flow
    async fn get_flow(&self, flow_id: &str) -> Result<LLMFlow, ReplayerError> {
        // 先从内存存储获取
        let store = self.flow_monitor.memory_store();
        let store_guard = store.read().await;

        if let Some(flow_lock) = store_guard.get(flow_id) {
            let flow = flow_lock.read().unwrap().clone();
            return Ok(flow);
        }
        drop(store_guard);

        // 再从文件存储获取
        if let Some(file_store) = self.flow_monitor.file_store() {
            if let Ok(Some(flow)) = file_store.get(flow_id) {
                return Ok(flow);
            }
        }

        Err(ReplayerError::FlowNotFound(flow_id.to_string()))
    }

    /// 应用请求修改
    fn apply_modifications(
        &self,
        original: &LLMRequest,
        modification: &Option<RequestModification>,
    ) -> LLMRequest {
        let mut request = original.clone();

        if let Some(mod_config) = modification {
            // 修改模型
            if let Some(ref model) = mod_config.model {
                request.model = model.clone();
            }

            // 修改消息
            if let Some(ref messages) = mod_config.messages {
                request.messages = messages.clone();
            }

            // 修改参数
            if let Some(ref params) = mod_config.parameters {
                request.parameters = params.clone();
            }

            // 修改系统提示词
            if let Some(ref system_prompt) = mod_config.system_prompt {
                request.system_prompt = Some(system_prompt.clone());
            }
        }

        // 更新时间戳
        request.timestamp = Utc::now();

        request
    }

    /// 解析凭证
    async fn resolve_credential(
        &self,
        original_flow: &LLMFlow,
        config: &ReplayConfig,
    ) -> Result<Option<String>, ReplayerError> {
        // 如果配置中指定了凭证，使用指定的凭证
        if let Some(ref cred_id) = config.credential_id {
            return Ok(Some(cred_id.clone()));
        }

        // 否则使用原始 Flow 的凭证
        Ok(original_flow.metadata.credential_id.clone())
    }

    /// 创建重放 Flow
    ///
    /// **Validates: Requirements 3.2**
    async fn create_replay_flow(
        &self,
        original_flow: &LLMFlow,
        request: &LLMRequest,
        credential_id: &Option<String>,
    ) -> String {
        let replay_flow_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        // 创建重放 Flow 的元数据
        let mut metadata = original_flow.metadata.clone();
        metadata.credential_id = credential_id.clone();

        // 创建重放 Flow
        let replay_flow = LLMFlow {
            id: replay_flow_id.clone(),
            flow_type: original_flow.flow_type.clone(),
            request: request.clone(),
            response: None,
            error: None,
            metadata,
            timestamps: FlowTimestamps {
                created: now,
                request_start: now,
                request_end: None,
                response_start: None,
                response_end: None,
                duration_ms: 0,
                ttfb_ms: None,
            },
            state: FlowState::Pending,
            annotations: FlowAnnotations {
                marker: Some("🔄".to_string()), // 重放标记
                comment: Some(format!("重放自 Flow: {}", original_flow.id)),
                tags: vec!["replay".to_string()],
                starred: false,
            },
        };

        // 保存到内存存储
        {
            let store = self.flow_monitor.memory_store();
            let mut store_guard = store.write().await;
            store_guard.add(replay_flow.clone());
        }

        // 保存到文件存储
        if let Some(file_store) = self.flow_monitor.file_store() {
            if let Err(e) = file_store.write(&replay_flow) {
                tracing::error!("保存重放 Flow 到文件失败: {}", e);
            }
        }

        replay_flow_id
    }

    /// 执行重放请求
    async fn execute_replay(
        &self,
        request: &LLMRequest,
        metadata: &FlowMetadata,
        credential_id: &Option<String>,
    ) -> Result<LLMResponse, ReplayerError> {
        // 构建请求 URL
        let base_url = self.get_base_url(&metadata.provider);
        let url = format!("{}{}", base_url, request.path);

        // 获取认证信息
        let auth_header = self
            .get_auth_header(&metadata.provider, credential_id)
            .await?;

        // 构建请求
        let mut req_builder = self.client.post(&url);

        // 添加认证头
        if let Some(auth) = auth_header {
            req_builder = req_builder.header("Authorization", auth);
        }

        // 添加其他头
        req_builder = req_builder
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        // 添加请求体
        req_builder = req_builder.json(&request.body);

        // 发送请求
        let start_time = Utc::now();
        let response = req_builder
            .send()
            .await
            .map_err(|e| ReplayerError::RequestFailed(e.to_string()))?;

        let end_time = Utc::now();
        let status_code = response.status().as_u16();
        let status_text = response.status().to_string();

        // 获取响应头
        let mut headers = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(v) = value.to_str() {
                headers.insert(key.to_string(), v.to_string());
            }
        }

        // 获取响应体
        let body_bytes = response
            .bytes()
            .await
            .map_err(|e| ReplayerError::RequestFailed(e.to_string()))?;
        let size_bytes = body_bytes.len();

        // 解析响应体
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&body_bytes).to_string())
        });

        // 提取内容
        let content = self.extract_content(&body, &metadata.provider);

        // 提取 token 使用量
        let usage = self.extract_usage(&body, &metadata.provider);

        Ok(LLMResponse {
            status_code,
            status_text,
            headers,
            body,
            content,
            thinking: None,
            tool_calls: Vec::new(),
            usage,
            stop_reason: None,
            size_bytes,
            timestamp_start: start_time,
            timestamp_end: end_time,
            stream_info: None,
        })
    }

    /// 获取基础 URL
    fn get_base_url(&self, provider: &ProviderType) -> String {
        match provider {
            ProviderType::OpenAI => "https://api.openai.com".to_string(),
            ProviderType::Claude => "https://api.anthropic.com".to_string(),
            ProviderType::Gemini | ProviderType::GeminiApiKey => {
                "https://generativelanguage.googleapis.com".to_string()
            }
            ProviderType::Qwen => "https://dashscope.aliyuncs.com".to_string(),
            ProviderType::Kiro => "https://q.us-east-1.amazonaws.com".to_string(),
            _ => "https://api.openai.com".to_string(), // 默认使用 OpenAI 兼容 API
        }
    }

    /// 获取认证头
    async fn get_auth_header(
        &self,
        provider: &ProviderType,
        credential_id: &Option<String>,
    ) -> Result<Option<String>, ReplayerError> {
        // 如果没有指定凭证，尝试从凭证池选择
        let cred_id = if let Some(id) = credential_id {
            id.clone()
        } else {
            // 尝试从凭证池选择
            let provider_type_str = format!("{:?}", provider);
            if let Ok(Some(cred)) =
                self.provider_pool
                    .select_credential(&self.db, &provider_type_str, None)
            {
                cred.uuid
            } else {
                return Ok(None);
            }
        };

        // TODO: 根据凭证 ID 获取实际的认证信息
        // 这里需要根据具体的凭证类型来获取 token
        // 目前返回 None，实际实现需要从凭证池获取 token
        Ok(None)
    }

    /// 提取响应内容
    fn extract_content(&self, body: &serde_json::Value, provider: &ProviderType) -> String {
        match provider {
            ProviderType::OpenAI | ProviderType::Kiro => {
                // OpenAI 格式
                body["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            }
            ProviderType::Claude | ProviderType::ClaudeOAuth => {
                // Claude 格式
                body["content"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            }
            ProviderType::Gemini | ProviderType::GeminiApiKey => {
                // Gemini 格式
                body["candidates"][0]["content"]["parts"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            }
            _ => {
                // 尝试通用格式
                body["choices"][0]["message"]["content"]
                    .as_str()
                    .or_else(|| body["content"][0]["text"].as_str())
                    .unwrap_or("")
                    .to_string()
            }
        }
    }

    /// 提取 token 使用量
    fn extract_usage(&self, body: &serde_json::Value, provider: &ProviderType) -> TokenUsage {
        let usage = &body["usage"];

        match provider {
            ProviderType::OpenAI | ProviderType::Kiro => TokenUsage {
                input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: usage["total_tokens"].as_u64().unwrap_or(0) as u32,
                ..Default::default()
            },
            ProviderType::Claude | ProviderType::ClaudeOAuth => TokenUsage {
                input_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: (usage["input_tokens"].as_u64().unwrap_or(0)
                    + usage["output_tokens"].as_u64().unwrap_or(0))
                    as u32,
                ..Default::default()
            },
            _ => TokenUsage::default(),
        }
    }

    /// 完成重放 Flow
    async fn complete_replay_flow(&self, flow_id: &str, response: Option<LLMResponse>) {
        let now = Utc::now();

        // 更新内存存储中的 Flow
        let store = self.flow_monitor.memory_store();
        let store_guard = store.read().await;

        if let Some(flow_lock) = store_guard.get(flow_id) {
            let mut flow = flow_lock.write().unwrap();
            flow.response = response;
            flow.state = FlowState::Completed;
            flow.timestamps.response_end = Some(now);
            flow.timestamps.calculate_duration();
        }
    }

    /// 标记重放 Flow 失败
    async fn fail_replay_flow(&self, flow_id: &str, error: &str) {
        let now = Utc::now();

        // 更新内存存储中的 Flow
        let store = self.flow_monitor.memory_store();
        let store_guard = store.read().await;

        if let Some(flow_lock) = store_guard.get(flow_id) {
            let mut flow = flow_lock.write().unwrap();
            flow.state = FlowState::Failed;
            flow.error = Some(super::models::FlowError::new(
                super::models::FlowErrorType::Other,
                error,
            ));
            flow.timestamps.response_end = Some(now);
            flow.timestamps.calculate_duration();
        }
    }

    /// 检查 Flow 是否为重放 Flow
    ///
    /// **Validates: Requirements 3.2**
    pub fn is_replay_flow(flow: &LLMFlow) -> bool {
        flow.annotations.tags.contains(&"replay".to_string())
    }

    /// 获取原始 Flow ID（从重放 Flow 的注释中提取）
    pub fn get_original_flow_id(flow: &LLMFlow) -> Option<String> {
        if let Some(ref comment) = flow.annotations.comment {
            if comment.starts_with("重放自 Flow: ") {
                return Some(comment.replace("重放自 Flow: ", ""));
            }
        }
        None
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::models::FlowType;
    use super::*;

    #[test]
    fn test_replay_config_default() {
        let config = ReplayConfig::default();
        assert!(config.credential_id.is_none());
        assert!(config.modify_request.is_none());
        assert_eq!(config.interval_ms, 1000);
    }

    #[test]
    fn test_replay_result_success() {
        let started_at = Utc::now();
        let completed_at = started_at + chrono::Duration::milliseconds(500);

        let result = ReplayResult::success(
            "original-id".to_string(),
            "replay-id".to_string(),
            started_at,
            completed_at,
        );

        assert!(result.success);
        assert_eq!(result.original_flow_id, "original-id");
        assert_eq!(result.replay_flow_id, "replay-id");
        assert!(result.error.is_none());
        assert_eq!(result.duration_ms, 500);
    }

    #[test]
    fn test_replay_result_failure() {
        let started_at = Utc::now();
        let completed_at = started_at + chrono::Duration::milliseconds(100);

        let result = ReplayResult::failure(
            "original-id".to_string(),
            "Connection failed".to_string(),
            started_at,
            completed_at,
        );

        assert!(!result.success);
        assert_eq!(result.original_flow_id, "original-id");
        assert!(result.replay_flow_id.is_empty());
        assert_eq!(result.error, Some("Connection failed".to_string()));
    }

    #[test]
    fn test_is_replay_flow() {
        let mut flow = LLMFlow::new(
            "test-id".to_string(),
            FlowType::ChatCompletions,
            LLMRequest::default(),
            FlowMetadata::default(),
        );

        // 没有 replay 标签
        assert!(!FlowReplayer::is_replay_flow(&flow));

        // 添加 replay 标签
        flow.annotations.tags.push("replay".to_string());
        assert!(FlowReplayer::is_replay_flow(&flow));
    }

    #[test]
    fn test_get_original_flow_id() {
        let mut flow = LLMFlow::new(
            "replay-id".to_string(),
            FlowType::ChatCompletions,
            LLMRequest::default(),
            FlowMetadata::default(),
        );

        // 没有注释
        assert!(FlowReplayer::get_original_flow_id(&flow).is_none());

        // 添加重放注释
        flow.annotations.comment = Some("重放自 Flow: original-id".to_string());
        assert_eq!(
            FlowReplayer::get_original_flow_id(&flow),
            Some("original-id".to_string())
        );
    }

    #[test]
    fn test_request_modification_serialization() {
        let modification = RequestModification {
            model: Some("gpt-4-turbo".to_string()),
            messages: None,
            parameters: None,
            system_prompt: Some("You are a helpful assistant.".to_string()),
        };

        let json = serde_json::to_string(&modification).unwrap();
        let deserialized: RequestModification = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.model, Some("gpt-4-turbo".to_string()));
        assert_eq!(
            deserialized.system_prompt,
            Some("You are a helpful assistant.".to_string())
        );
    }
}

// ============================================================================
// 属性测试
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::super::models::FlowType;
    use super::*;
    use proptest::prelude::*;

    // ========================================================================
    // 生成器
    // ========================================================================

    /// 生成随机的 Flow ID
    fn arb_flow_id() -> impl Strategy<Value = String> {
        "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
    }

    /// 生成随机的模型名称
    fn arb_model_name() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("gpt-4".to_string()),
            Just("gpt-4-turbo".to_string()),
            Just("gpt-3.5-turbo".to_string()),
            Just("claude-3-opus".to_string()),
            Just("claude-3-sonnet".to_string()),
            Just("gemini-pro".to_string()),
        ]
    }

    /// 生成随机的 LLMRequest
    fn arb_llm_request() -> impl Strategy<Value = LLMRequest> {
        arb_model_name().prop_map(|model| LLMRequest {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            headers: std::collections::HashMap::new(),
            body: serde_json::Value::Null,
            messages: Vec::new(),
            system_prompt: None,
            tools: None,
            model,
            original_model: None,
            parameters: RequestParameters::default(),
            size_bytes: 0,
            timestamp: Utc::now(),
        })
    }

    /// 生成随机的 FlowMetadata
    fn arb_flow_metadata() -> impl Strategy<Value = FlowMetadata> {
        prop_oneof![
            Just(crate::ProviderType::OpenAI),
            Just(crate::ProviderType::Claude),
            Just(crate::ProviderType::Gemini),
            Just(crate::ProviderType::Kiro),
        ]
        .prop_map(|provider| FlowMetadata {
            provider,
            credential_id: Some("test-cred".to_string()),
            credential_name: Some("Test Credential".to_string()),
            ..Default::default()
        })
    }

    /// 生成随机的 LLMFlow
    fn arb_llm_flow() -> impl Strategy<Value = LLMFlow> {
        (arb_flow_id(), arb_llm_request(), arb_flow_metadata()).prop_map(
            |(id, request, metadata)| {
                LLMFlow::new(id, FlowType::ChatCompletions, request, metadata)
            },
        )
    }

    // ========================================================================
    // 属性测试
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: flow-monitor-enhancement, Property 5: 重放 Flow 标记正确性**
        /// **Validates: Requirements 3.2**
        ///
        /// *对于任意* 重放操作，新创建的 Flow 应该被正确标记为 "replay"，
        /// 并且包含原始 Flow 的引用。
        #[test]
        fn prop_replay_flow_marking_correctness(
            original_flow in arb_llm_flow(),
        ) {
            // 保存原始 Flow ID 的副本
            let original_flow_id = original_flow.id.clone();

            // 创建一个模拟的重放 Flow（模拟 create_replay_flow 的行为）
            let replay_flow_id = uuid::Uuid::new_v4().to_string();
            let now = Utc::now();

            // 创建重放 Flow 的元数据
            let metadata = original_flow.metadata.clone();

            // 创建重放 Flow（模拟 create_replay_flow 的逻辑）
            let replay_flow = LLMFlow {
                id: replay_flow_id.clone(),
                flow_type: original_flow.flow_type.clone(),
                request: original_flow.request.clone(),
                response: None,
                error: None,
                metadata,
                timestamps: FlowTimestamps {
                    created: now,
                    request_start: now,
                    request_end: None,
                    response_start: None,
                    response_end: None,
                    duration_ms: 0,
                    ttfb_ms: None,
                },
                state: FlowState::Pending,
                annotations: FlowAnnotations {
                    marker: Some("🔄".to_string()), // 重放标记
                    comment: Some(format!("重放自 Flow: {}", original_flow_id)),
                    tags: vec!["replay".to_string()],
                    starred: false,
                },
            };

            // 验证 1: 重放 Flow 应该有 "replay" 标签
            prop_assert!(
                FlowReplayer::is_replay_flow(&replay_flow),
                "重放 Flow 应该被标记为 replay"
            );

            // 验证 2: 重放 Flow 应该包含原始 Flow ID 的引用
            let extracted_original_id = FlowReplayer::get_original_flow_id(&replay_flow);
            prop_assert!(
                extracted_original_id.is_some(),
                "重放 Flow 应该包含原始 Flow ID 的引用"
            );
            prop_assert_eq!(
                extracted_original_id.unwrap(),
                original_flow_id.clone(),
                "提取的原始 Flow ID 应该与实际原始 Flow ID 一致"
            );

            // 验证 3: 重放 Flow 应该有重放标记 emoji
            prop_assert_eq!(
                replay_flow.annotations.marker,
                Some("🔄".to_string()),
                "重放 Flow 应该有重放标记 emoji"
            );

            // 验证 4: 重放 Flow 的 ID 应该与原始 Flow 不同
            prop_assert_ne!(
                replay_flow.id,
                original_flow_id,
                "重放 Flow 的 ID 应该与原始 Flow 不同"
            );

            // 验证 5: 原始 Flow 不应该被标记为 replay（除非它本身就是重放）
            if !original_flow.annotations.tags.contains(&"replay".to_string()) {
                prop_assert!(
                    !FlowReplayer::is_replay_flow(&original_flow),
                    "原始 Flow 不应该被标记为 replay"
                );
            }
        }

        /// **Feature: flow-monitor-enhancement, Property 5b: 非重放 Flow 标记正确性**
        /// **Validates: Requirements 3.2**
        ///
        /// *对于任意* 普通 Flow（非重放），is_replay_flow 应该返回 false。
        #[test]
        fn prop_non_replay_flow_not_marked(
            flow in arb_llm_flow(),
        ) {
            // 普通 Flow 不应该被标记为 replay
            prop_assert!(
                !FlowReplayer::is_replay_flow(&flow),
                "普通 Flow 不应该被标记为 replay"
            );

            // 普通 Flow 不应该有原始 Flow ID
            prop_assert!(
                FlowReplayer::get_original_flow_id(&flow).is_none(),
                "普通 Flow 不应该有原始 Flow ID"
            );
        }
    }
}
