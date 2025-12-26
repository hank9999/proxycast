//! Flow 内存存储
//!
//! 该模块实现 LLM Flow 的内存缓存存储，支持 LRU 驱逐策略。
//! 提供快速的 Flow 访问和查询功能。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

use super::models::{FlowState, FlowType, LLMFlow};
use crate::ProviderType;

// ============================================================================
// 过滤器结构
// ============================================================================

/// 时间范围
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// 开始时间
    pub start: Option<DateTime<Utc>>,
    /// 结束时间
    pub end: Option<DateTime<Utc>>,
}

impl TimeRange {
    /// 创建新的时间范围
    pub fn new(start: Option<DateTime<Utc>>, end: Option<DateTime<Utc>>) -> Self {
        Self { start, end }
    }

    /// 检查时间是否在范围内
    pub fn contains(&self, time: &DateTime<Utc>) -> bool {
        let after_start = self.start.map_or(true, |s| time >= &s);
        let before_end = self.end.map_or(true, |e| time <= &e);
        after_start && before_end
    }
}

/// Token 范围
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRange {
    /// 最小 Token 数
    pub min: Option<u32>,
    /// 最大 Token 数
    pub max: Option<u32>,
}

impl TokenRange {
    /// 检查 Token 数是否在范围内
    pub fn contains(&self, tokens: u32) -> bool {
        let above_min = self.min.map_or(true, |m| tokens >= m);
        let below_max = self.max.map_or(true, |m| tokens <= m);
        above_min && below_max
    }
}

/// 延迟范围
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRange {
    /// 最小延迟（毫秒）
    pub min_ms: Option<u64>,
    /// 最大延迟（毫秒）
    pub max_ms: Option<u64>,
}

impl LatencyRange {
    /// 检查延迟是否在范围内
    pub fn contains(&self, latency_ms: u64) -> bool {
        let above_min = self.min_ms.map_or(true, |m| latency_ms >= m);
        let below_max = self.max_ms.map_or(true, |m| latency_ms <= m);
        above_min && below_max
    }
}

/// Flow 过滤器
///
/// 支持多维度过滤条件，用于查询 Flow。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowFilter {
    /// 时间范围
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_range: Option<TimeRange>,
    /// 提供商类型列表
    #[serde(skip_serializing_if = "Option::is_none")]
    pub providers: Option<Vec<ProviderType>>,
    /// 模型名称列表（支持通配符 *）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    /// 状态列表
    #[serde(skip_serializing_if = "Option::is_none")]
    pub states: Option<Vec<FlowState>>,
    /// 是否有错误
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_error: Option<bool>,
    /// 是否有工具调用
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_tool_calls: Option<bool>,
    /// 是否有思维链
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_thinking: Option<bool>,
    /// 是否是流式响应
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_streaming: Option<bool>,
    /// 内容搜索（响应内容）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_search: Option<String>,
    /// 请求搜索（请求内容）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_search: Option<String>,
    /// Token 范围
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_range: Option<TokenRange>,
    /// 延迟范围
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_range: Option<LatencyRange>,
    /// 标签列表
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// 仅收藏
    #[serde(default)]
    pub starred_only: bool,
    /// 凭证 ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_id: Option<String>,
    /// Flow 类型
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow_types: Option<Vec<FlowType>>,
}

impl FlowFilter {
    /// 创建空过滤器（匹配所有）
    pub fn new() -> Self {
        Self::default()
    }

    /// 检查 Flow 是否匹配过滤条件
    pub fn matches(&self, flow: &LLMFlow) -> bool {
        // 时间范围过滤
        if let Some(ref time_range) = self.time_range {
            if !time_range.contains(&flow.timestamps.created) {
                return false;
            }
        }

        // 提供商过滤
        if let Some(ref providers) = self.providers {
            if !providers.contains(&flow.metadata.provider) {
                return false;
            }
        }

        // 模型过滤（支持通配符）
        if let Some(ref models) = self.models {
            let model_matches = models
                .iter()
                .any(|pattern| Self::match_pattern(pattern, &flow.request.model));
            if !model_matches {
                return false;
            }
        }

        // 状态过滤
        if let Some(ref states) = self.states {
            if !states.contains(&flow.state) {
                return false;
            }
        }

        // 错误过滤
        if let Some(has_error) = self.has_error {
            let flow_has_error = flow.error.is_some();
            if has_error != flow_has_error {
                return false;
            }
        }

        // 工具调用过滤
        if let Some(has_tool_calls) = self.has_tool_calls {
            let flow_has_tool_calls = flow
                .response
                .as_ref()
                .map_or(false, |r| !r.tool_calls.is_empty());
            if has_tool_calls != flow_has_tool_calls {
                return false;
            }
        }

        // 思维链过滤
        if let Some(has_thinking) = self.has_thinking {
            let flow_has_thinking = flow
                .response
                .as_ref()
                .map_or(false, |r| r.thinking.is_some());
            if has_thinking != flow_has_thinking {
                return false;
            }
        }

        // 流式响应过滤
        if let Some(is_streaming) = self.is_streaming {
            let flow_is_streaming = flow.request.parameters.stream;
            if is_streaming != flow_is_streaming {
                return false;
            }
        }

        // 内容搜索（搜索响应内容、模型名称、提供商名称）
        if let Some(ref search) = self.content_search {
            let search_lower = search.to_lowercase();

            // 搜索响应内容
            let content = flow
                .response
                .as_ref()
                .map_or(String::new(), |r| r.content.clone());
            let content_matches = content.to_lowercase().contains(&search_lower);

            // 搜索模型名称
            let model_matches = flow.request.model.to_lowercase().contains(&search_lower);

            // 搜索提供商名称
            let provider_name = format!("{:?}", flow.metadata.provider).to_lowercase();
            let provider_matches = provider_name.contains(&search_lower);

            // 任一匹配即可
            if !content_matches && !model_matches && !provider_matches {
                return false;
            }
        }

        // 请求搜索
        if let Some(ref search) = self.request_search {
            let request_text = Self::get_request_text(flow);
            if !request_text.to_lowercase().contains(&search.to_lowercase()) {
                return false;
            }
        }

        // Token 范围过滤
        if let Some(ref token_range) = self.token_range {
            let total_tokens = flow.response.as_ref().map_or(0, |r| r.usage.total_tokens);
            if !token_range.contains(total_tokens) {
                return false;
            }
        }

        // 延迟范围过滤
        if let Some(ref latency_range) = self.latency_range {
            if !latency_range.contains(flow.timestamps.duration_ms) {
                return false;
            }
        }

        // 标签过滤
        if let Some(ref tags) = self.tags {
            let has_any_tag = tags.iter().any(|t| flow.annotations.tags.contains(t));
            if !has_any_tag {
                return false;
            }
        }

        // 收藏过滤
        if self.starred_only && !flow.annotations.starred {
            return false;
        }

        // 凭证 ID 过滤
        if let Some(ref credential_id) = self.credential_id {
            if flow.metadata.credential_id.as_ref() != Some(credential_id) {
                return false;
            }
        }

        // Flow 类型过滤
        if let Some(ref flow_types) = self.flow_types {
            if !flow_types.contains(&flow.flow_type) {
                return false;
            }
        }

        true
    }

    /// 模式匹配（支持 * 通配符）
    fn match_pattern(pattern: &str, text: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.contains('*') {
            // 简单的通配符匹配
            let parts: Vec<&str> = pattern.split('*').collect();
            let mut pos = 0;
            let text_lower = text.to_lowercase();

            for (i, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }

                let part_lower = part.to_lowercase();
                if let Some(found_pos) = text_lower[pos..].find(&part_lower) {
                    // 第一个部分必须从开头匹配
                    if i == 0 && found_pos != 0 {
                        return false;
                    }
                    pos += found_pos + part.len();
                } else {
                    return false;
                }
            }

            // 最后一个部分必须匹配到结尾
            if !pattern.ends_with('*') && pos != text.len() {
                return false;
            }

            true
        } else {
            text.to_lowercase() == pattern.to_lowercase()
        }
    }

    /// 获取请求文本（用于搜索）
    fn get_request_text(flow: &LLMFlow) -> String {
        let mut text = String::new();

        // 添加系统提示词
        if let Some(ref system) = flow.request.system_prompt {
            text.push_str(system);
            text.push('\n');
        }

        // 添加消息内容
        for msg in &flow.request.messages {
            text.push_str(&msg.content.get_all_text());
            text.push('\n');
        }

        text
    }
}

// ============================================================================
// 内存存储
// ============================================================================

/// Flow 内存存储
///
/// 使用 LRU 策略管理内存中的 Flow 缓存。
/// 线程安全，支持并发读写。
pub struct FlowMemoryStore {
    /// Flow 存储（ID -> Flow）
    flows: HashMap<String, Arc<RwLock<LLMFlow>>>,
    /// 有序 ID 列表（用于 LRU 驱逐）
    ordered_ids: VecDeque<String>,
    /// 最大缓存大小
    max_size: usize,
}

impl FlowMemoryStore {
    /// 创建新的内存存储
    ///
    /// # 参数
    /// - `max_size`: 最大缓存 Flow 数量
    pub fn new(max_size: usize) -> Self {
        Self {
            flows: HashMap::with_capacity(max_size),
            ordered_ids: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// 获取当前缓存大小
    pub fn len(&self) -> usize {
        self.flows.len()
    }

    /// 检查缓存是否为空
    pub fn is_empty(&self) -> bool {
        self.flows.is_empty()
    }

    /// 获取最大缓存大小
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// 添加 Flow 到缓存
    ///
    /// 如果缓存已满，会驱逐最旧的 Flow。
    pub fn add(&mut self, flow: LLMFlow) {
        let id = flow.id.clone();

        // 如果已存在，先移除旧的
        if self.flows.contains_key(&id) {
            self.ordered_ids.retain(|i| i != &id);
        }

        // 检查是否需要驱逐
        while self.flows.len() >= self.max_size {
            self.evict_oldest();
        }

        // 添加新 Flow
        self.flows.insert(id.clone(), Arc::new(RwLock::new(flow)));
        self.ordered_ids.push_back(id);
    }

    /// 获取 Flow
    ///
    /// 返回 Flow 的共享引用，可用于读取或更新。
    pub fn get(&self, id: &str) -> Option<Arc<RwLock<LLMFlow>>> {
        self.flows.get(id).cloned()
    }

    /// 更新 Flow
    ///
    /// 使用提供的更新函数修改 Flow。
    ///
    /// # 参数
    /// - `id`: Flow ID
    /// - `updater`: 更新函数
    ///
    /// # 返回
    /// - `true`: 更新成功
    /// - `false`: Flow 不存在
    pub fn update<F>(&self, id: &str, updater: F) -> bool
    where
        F: FnOnce(&mut LLMFlow),
    {
        if let Some(flow_lock) = self.flows.get(id) {
            if let Ok(mut flow) = flow_lock.write() {
                updater(&mut flow);
                return true;
            }
        }
        false
    }

    /// 获取最近的 Flow 列表
    ///
    /// # 参数
    /// - `limit`: 最大返回数量
    ///
    /// # 返回
    /// 按时间倒序排列的 Flow 列表
    pub fn get_recent(&self, limit: usize) -> Vec<LLMFlow> {
        let mut flows: Vec<LLMFlow> = Vec::with_capacity(limit.min(self.flows.len()));

        // 从最新到最旧遍历
        for id in self.ordered_ids.iter().rev().take(limit) {
            if let Some(flow_lock) = self.flows.get(id) {
                if let Ok(flow) = flow_lock.read() {
                    flows.push(flow.clone());
                }
            }
        }

        flows
    }

    /// 查询 Flow
    ///
    /// # 参数
    /// - `filter`: 过滤条件
    ///
    /// # 返回
    /// 匹配过滤条件的 Flow 列表（按时间倒序）
    pub fn query(&self, filter: &FlowFilter) -> Vec<LLMFlow> {
        let mut results: Vec<LLMFlow> = Vec::new();

        // 从最新到最旧遍历
        for id in self.ordered_ids.iter().rev() {
            if let Some(flow_lock) = self.flows.get(id) {
                if let Ok(flow) = flow_lock.read() {
                    if filter.matches(&flow) {
                        results.push(flow.clone());
                    }
                }
            }
        }

        results
    }

    /// 删除 Flow
    ///
    /// # 返回
    /// - `true`: 删除成功
    /// - `false`: Flow 不存在
    pub fn remove(&mut self, id: &str) -> bool {
        if self.flows.remove(id).is_some() {
            self.ordered_ids.retain(|i| i != id);
            true
        } else {
            false
        }
    }

    /// 清空所有 Flow
    pub fn clear(&mut self) {
        self.flows.clear();
        self.ordered_ids.clear();
    }

    /// 按时间清理 Flow
    ///
    /// 删除指定时间之前创建的所有 Flow。
    ///
    /// # 参数
    /// - `before`: 截止时间，早于此时间的 Flow 将被删除
    ///
    /// # 返回
    /// 删除的 Flow 数量
    pub fn cleanup_before(&mut self, before: chrono::DateTime<chrono::Utc>) -> usize {
        let mut to_remove = Vec::new();

        for id in &self.ordered_ids {
            if let Some(flow_lock) = self.flows.get(id) {
                if let Ok(flow) = flow_lock.read() {
                    if flow.timestamps.created < before {
                        to_remove.push(id.clone());
                    }
                }
            }
        }

        let count = to_remove.len();
        for id in to_remove {
            self.flows.remove(&id);
            self.ordered_ids.retain(|i| i != &id);
        }

        count
    }

    /// 驱逐最旧的 Flow
    fn evict_oldest(&mut self) {
        if let Some(oldest_id) = self.ordered_ids.pop_front() {
            self.flows.remove(&oldest_id);
        }
    }

    /// 获取所有 Flow ID
    pub fn get_all_ids(&self) -> Vec<String> {
        self.ordered_ids.iter().cloned().collect()
    }

    /// 检查 Flow 是否存在
    pub fn contains(&self, id: &str) -> bool {
        self.flows.contains_key(id)
    }
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_monitor::models::{
        FlowMetadata, FlowTimestamps, LLMRequest, RequestParameters,
    };

    /// 创建测试用的 Flow
    fn create_test_flow(id: &str, model: &str, provider: ProviderType) -> LLMFlow {
        let request = LLMRequest {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            model: model.to_string(),
            parameters: RequestParameters {
                stream: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let metadata = FlowMetadata {
            provider,
            ..Default::default()
        };

        LLMFlow::new(id.to_string(), FlowType::ChatCompletions, request, metadata)
    }

    #[test]
    fn test_memory_store_add_and_get() {
        let mut store = FlowMemoryStore::new(10);
        let flow = create_test_flow("test-1", "gpt-4", ProviderType::OpenAI);

        store.add(flow.clone());

        assert_eq!(store.len(), 1);
        assert!(store.contains("test-1"));

        let retrieved = store.get("test-1").unwrap();
        let retrieved_flow = retrieved.read().unwrap();
        assert_eq!(retrieved_flow.id, "test-1");
        assert_eq!(retrieved_flow.request.model, "gpt-4");
    }

    #[test]
    fn test_memory_store_lru_eviction() {
        let mut store = FlowMemoryStore::new(3);

        // 添加 3 个 Flow
        store.add(create_test_flow("flow-1", "gpt-4", ProviderType::OpenAI));
        store.add(create_test_flow("flow-2", "gpt-4", ProviderType::OpenAI));
        store.add(create_test_flow("flow-3", "gpt-4", ProviderType::OpenAI));

        assert_eq!(store.len(), 3);

        // 添加第 4 个，应该驱逐最旧的
        store.add(create_test_flow("flow-4", "gpt-4", ProviderType::OpenAI));

        assert_eq!(store.len(), 3);
        assert!(!store.contains("flow-1")); // 最旧的被驱逐
        assert!(store.contains("flow-2"));
        assert!(store.contains("flow-3"));
        assert!(store.contains("flow-4"));
    }

    #[test]
    fn test_memory_store_update() {
        let mut store = FlowMemoryStore::new(10);
        let flow = create_test_flow("test-1", "gpt-4", ProviderType::OpenAI);

        store.add(flow);

        // 更新 Flow
        let updated = store.update("test-1", |f| {
            f.state = FlowState::Completed;
        });

        assert!(updated);

        // 验证更新
        let retrieved = store.get("test-1").unwrap();
        let retrieved_flow = retrieved.read().unwrap();
        assert_eq!(retrieved_flow.state, FlowState::Completed);
    }

    #[test]
    fn test_memory_store_get_recent() {
        let mut store = FlowMemoryStore::new(10);

        store.add(create_test_flow("flow-1", "gpt-4", ProviderType::OpenAI));
        store.add(create_test_flow("flow-2", "gpt-4", ProviderType::OpenAI));
        store.add(create_test_flow("flow-3", "gpt-4", ProviderType::OpenAI));

        let recent = store.get_recent(2);

        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].id, "flow-3"); // 最新的在前
        assert_eq!(recent[1].id, "flow-2");
    }

    #[test]
    fn test_memory_store_remove() {
        let mut store = FlowMemoryStore::new(10);

        store.add(create_test_flow("flow-1", "gpt-4", ProviderType::OpenAI));
        store.add(create_test_flow("flow-2", "gpt-4", ProviderType::OpenAI));

        assert!(store.remove("flow-1"));
        assert_eq!(store.len(), 1);
        assert!(!store.contains("flow-1"));
        assert!(store.contains("flow-2"));

        // 删除不存在的
        assert!(!store.remove("flow-999"));
    }

    #[test]
    fn test_memory_store_clear() {
        let mut store = FlowMemoryStore::new(10);

        store.add(create_test_flow("flow-1", "gpt-4", ProviderType::OpenAI));
        store.add(create_test_flow("flow-2", "gpt-4", ProviderType::OpenAI));

        store.clear();

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_flow_filter_provider() {
        let flow = create_test_flow("test-1", "gpt-4", ProviderType::OpenAI);

        let filter = FlowFilter {
            providers: Some(vec![ProviderType::OpenAI]),
            ..Default::default()
        };
        assert!(filter.matches(&flow));

        let filter = FlowFilter {
            providers: Some(vec![ProviderType::Claude]),
            ..Default::default()
        };
        assert!(!filter.matches(&flow));
    }

    #[test]
    fn test_flow_filter_model_wildcard() {
        let flow = create_test_flow("test-1", "gpt-4-turbo", ProviderType::OpenAI);

        // 精确匹配
        let filter = FlowFilter {
            models: Some(vec!["gpt-4-turbo".to_string()]),
            ..Default::default()
        };
        assert!(filter.matches(&flow));

        // 通配符匹配
        let filter = FlowFilter {
            models: Some(vec!["gpt-4*".to_string()]),
            ..Default::default()
        };
        assert!(filter.matches(&flow));

        // 通配符不匹配
        let filter = FlowFilter {
            models: Some(vec!["claude*".to_string()]),
            ..Default::default()
        };
        assert!(!filter.matches(&flow));
    }

    #[test]
    fn test_flow_filter_state() {
        let mut flow = create_test_flow("test-1", "gpt-4", ProviderType::OpenAI);
        flow.state = FlowState::Completed;

        let filter = FlowFilter {
            states: Some(vec![FlowState::Completed]),
            ..Default::default()
        };
        assert!(filter.matches(&flow));

        let filter = FlowFilter {
            states: Some(vec![FlowState::Pending]),
            ..Default::default()
        };
        assert!(!filter.matches(&flow));
    }

    #[test]
    fn test_flow_filter_starred() {
        let mut flow = create_test_flow("test-1", "gpt-4", ProviderType::OpenAI);

        let filter = FlowFilter {
            starred_only: true,
            ..Default::default()
        };
        assert!(!filter.matches(&flow));

        flow.annotations.starred = true;
        assert!(filter.matches(&flow));
    }

    #[test]
    fn test_memory_store_query() {
        let mut store = FlowMemoryStore::new(10);

        store.add(create_test_flow("flow-1", "gpt-4", ProviderType::OpenAI));
        store.add(create_test_flow("flow-2", "claude-3", ProviderType::Claude));
        store.add(create_test_flow(
            "flow-3",
            "gpt-4-turbo",
            ProviderType::OpenAI,
        ));

        // 按提供商过滤
        let filter = FlowFilter {
            providers: Some(vec![ProviderType::OpenAI]),
            ..Default::default()
        };
        let results = store.query(&filter);
        assert_eq!(results.len(), 2);

        // 按模型通配符过滤
        let filter = FlowFilter {
            models: Some(vec!["gpt-4*".to_string()]),
            ..Default::default()
        };
        let results = store.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_time_range() {
        let now = Utc::now();
        let past = now - chrono::Duration::hours(1);
        let future = now + chrono::Duration::hours(1);

        let range = TimeRange::new(Some(past), Some(future));
        assert!(range.contains(&now));

        let range = TimeRange::new(Some(future), None);
        assert!(!range.contains(&now));

        let range = TimeRange::new(None, Some(past));
        assert!(!range.contains(&now));
    }

    #[test]
    fn test_token_range() {
        let range = TokenRange {
            min: Some(100),
            max: Some(1000),
        };

        assert!(range.contains(500));
        assert!(range.contains(100));
        assert!(range.contains(1000));
        assert!(!range.contains(50));
        assert!(!range.contains(1500));
    }

    #[test]
    fn test_latency_range() {
        let range = LatencyRange {
            min_ms: Some(100),
            max_ms: Some(1000),
        };

        assert!(range.contains(500));
        assert!(range.contains(100));
        assert!(range.contains(1000));
        assert!(!range.contains(50));
        assert!(!range.contains(1500));
    }
}

// ============================================================================
// 属性测试模块
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::flow_monitor::models::{
        FlowAnnotations, FlowError, FlowErrorType, FlowMetadata, FlowTimestamps, LLMRequest,
        LLMResponse, RequestParameters, ThinkingContent, TokenUsage,
    };
    use proptest::prelude::*;

    // ========================================================================
    // 生成器
    // ========================================================================

    /// 生成随机的 ProviderType
    fn arb_provider_type() -> impl Strategy<Value = ProviderType> {
        prop_oneof![
            Just(ProviderType::Kiro),
            Just(ProviderType::Gemini),
            Just(ProviderType::Qwen),
            Just(ProviderType::OpenAI),
            Just(ProviderType::Claude),
            Just(ProviderType::Antigravity),
            Just(ProviderType::Vertex),
            Just(ProviderType::GeminiApiKey),
            Just(ProviderType::Codex),
            Just(ProviderType::ClaudeOAuth),
            Just(ProviderType::IFlow),
        ]
    }

    /// 生成随机的 FlowType
    fn arb_flow_type() -> impl Strategy<Value = FlowType> {
        prop_oneof![
            Just(FlowType::ChatCompletions),
            Just(FlowType::AnthropicMessages),
            Just(FlowType::GeminiGenerateContent),
            Just(FlowType::Embeddings),
            "[a-z]{3,10}".prop_map(FlowType::Other),
        ]
    }

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
            "[a-z]{3,10}-[0-9]{1,2}".prop_map(|s| s),
        ]
    }

    /// 生成随机的 LLMRequest
    fn arb_llm_request() -> impl Strategy<Value = LLMRequest> {
        (arb_model_name(), any::<bool>()).prop_map(|(model, stream)| LLMRequest {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            model,
            parameters: RequestParameters {
                stream,
                ..Default::default()
            },
            ..Default::default()
        })
    }

    /// 生成随机的 FlowMetadata
    fn arb_flow_metadata() -> impl Strategy<Value = FlowMetadata> {
        arb_provider_type().prop_map(|provider| FlowMetadata {
            provider,
            ..Default::default()
        })
    }

    /// 生成随机的 LLMFlow
    fn arb_llm_flow() -> impl Strategy<Value = LLMFlow> {
        (
            arb_flow_id(),
            arb_flow_type(),
            arb_llm_request(),
            arb_flow_metadata(),
        )
            .prop_map(|(id, flow_type, request, metadata)| {
                LLMFlow::new(id, flow_type, request, metadata)
            })
    }

    /// 生成随机的缓存大小（1-100）
    fn arb_cache_size() -> impl Strategy<Value = usize> {
        1usize..=100usize
    }

    /// 生成随机的 Flow 数量（用于测试）
    fn arb_flow_count() -> impl Strategy<Value = usize> {
        1usize..=200usize
    }

    // ========================================================================
    // 属性测试
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: llm-flow-monitor, Property 4: 内存缓存大小不变量**
        /// **Validates: Requirements 3.1, 3.2**
        ///
        /// *对于任意* 数量的 Flow 添加操作，内存缓存中的 Flow 数量应该永远不超过配置的最大值。
        #[test]
        fn prop_memory_cache_size_invariant(
            max_size in arb_cache_size(),
            flow_count in arb_flow_count(),
        ) {
            let mut store = FlowMemoryStore::new(max_size);

            // 添加多个 Flow
            for i in 0..flow_count {
                let id = format!("flow-{}", i);
                let request = LLMRequest {
                    method: "POST".to_string(),
                    path: "/v1/chat/completions".to_string(),
                    model: "gpt-4".to_string(),
                    ..Default::default()
                };
                let metadata = FlowMetadata::default();
                let flow = LLMFlow::new(id, FlowType::ChatCompletions, request, metadata);

                store.add(flow);

                // 验证不变量：缓存大小永远不超过 max_size
                prop_assert!(
                    store.len() <= max_size,
                    "缓存大小 {} 超过了最大值 {}",
                    store.len(),
                    max_size
                );
            }

            // 最终验证
            let expected_size = flow_count.min(max_size);
            prop_assert_eq!(
                store.len(),
                expected_size,
                "最终缓存大小应该是 min(flow_count, max_size)"
            );
        }

        /// **Feature: llm-flow-monitor, Property 4b: LRU 驱逐正确性**
        /// **Validates: Requirements 3.2**
        ///
        /// *对于任意* 缓存大小和 Flow 序列，当缓存满时应该驱逐最旧的 Flow。
        #[test]
        fn prop_lru_eviction_correctness(
            max_size in 2usize..=10usize,
        ) {
            let mut store = FlowMemoryStore::new(max_size);

            // 添加 max_size + 1 个 Flow
            let total_flows = max_size + 1;
            for i in 0..total_flows {
                let id = format!("flow-{}", i);
                let request = LLMRequest {
                    method: "POST".to_string(),
                    path: "/v1/chat/completions".to_string(),
                    model: "gpt-4".to_string(),
                    ..Default::default()
                };
                let metadata = FlowMetadata::default();
                let flow = LLMFlow::new(id, FlowType::ChatCompletions, request, metadata);
                store.add(flow);
            }

            // 验证最旧的 Flow 被驱逐
            prop_assert!(
                !store.contains("flow-0"),
                "最旧的 Flow (flow-0) 应该被驱逐"
            );

            // 验证最新的 Flow 仍然存在
            for i in 1..total_flows {
                prop_assert!(
                    store.contains(&format!("flow-{}", i)),
                    "Flow flow-{} 应该仍然存在",
                    i
                );
            }
        }

        /// **Feature: llm-flow-monitor, Property 4c: 存储 Round-Trip**
        /// **Validates: Requirements 3.1**
        ///
        /// *对于任意* 有效的 LLMFlow，添加到缓存后再读取，读取的 Flow 应该与原始 Flow 等价。
        #[test]
        fn prop_memory_store_roundtrip(
            id in arb_flow_id(),
            flow_type in arb_flow_type(),
            request in arb_llm_request(),
            metadata in arb_flow_metadata(),
        ) {
            let mut store = FlowMemoryStore::new(100);

            let original_flow = LLMFlow::new(id.clone(), flow_type, request, metadata);

            // 添加到缓存
            store.add(original_flow.clone());

            // 读取
            let retrieved = store.get(&id).expect("Flow 应该存在");
            let retrieved_flow = retrieved.read().unwrap();

            // 验证关键字段一致
            prop_assert_eq!(&retrieved_flow.id, &original_flow.id, "ID 应该一致");
            prop_assert_eq!(&retrieved_flow.state, &original_flow.state, "状态应该一致");
            prop_assert_eq!(
                &retrieved_flow.request.model,
                &original_flow.request.model,
                "模型应该一致"
            );
            prop_assert_eq!(
                &retrieved_flow.metadata.provider,
                &original_flow.metadata.provider,
                "Provider 应该一致"
            );
        }

        /// **Feature: llm-flow-monitor, Property 4d: 过滤正确性**
        /// **Validates: Requirements 4.1-4.9**
        ///
        /// *对于任意* 过滤条件和 Flow 集合，查询返回的所有 Flow 都应该满足该过滤条件。
        #[test]
        fn prop_filter_correctness(
            provider in arb_provider_type(),
        ) {
            let mut store = FlowMemoryStore::new(100);

            // 添加不同 Provider 的 Flow
            let providers = vec![
                ProviderType::OpenAI,
                ProviderType::Claude,
                ProviderType::Gemini,
                ProviderType::Kiro,
            ];

            for (i, p) in providers.iter().enumerate() {
                let id = format!("flow-{}", i);
                let request = LLMRequest {
                    method: "POST".to_string(),
                    path: "/v1/chat/completions".to_string(),
                    model: "gpt-4".to_string(),
                    ..Default::default()
                };
                let metadata = FlowMetadata {
                    provider: p.clone(),
                    ..Default::default()
                };
                let flow = LLMFlow::new(id, FlowType::ChatCompletions, request, metadata);
                store.add(flow);
            }

            // 按 Provider 过滤
            let filter = FlowFilter {
                providers: Some(vec![provider.clone()]),
                ..Default::default()
            };

            let results = store.query(&filter);

            // 验证所有结果都匹配过滤条件
            for flow in &results {
                prop_assert_eq!(
                    flow.metadata.provider,
                    provider,
                    "查询结果的 Provider 应该匹配过滤条件"
                );
            }
        }

        /// **Feature: llm-flow-monitor, Property 4e: 模型通配符过滤正确性**
        /// **Validates: Requirements 4.3**
        ///
        /// *对于任意* 模型通配符模式，查询返回的所有 Flow 的模型名称都应该匹配该模式。
        #[test]
        fn prop_model_wildcard_filter_correctness(
            prefix in "[a-z]{2,5}",
        ) {
            let mut store = FlowMemoryStore::new(100);

            // 添加不同模型的 Flow
            // 使用数字前缀的模型名称，确保不会与随机生成的字母 prefix 冲突
            let models = vec![
                format!("{}-model-1", prefix),
                format!("{}-model-2", prefix),
                "123-non-matching-model".to_string(),
                "456-another-non-matching".to_string(),
            ];

            for (i, model) in models.iter().enumerate() {
                let id = format!("flow-{}", i);
                let request = LLMRequest {
                    method: "POST".to_string(),
                    path: "/v1/chat/completions".to_string(),
                    model: model.clone(),
                    ..Default::default()
                };
                let metadata = FlowMetadata::default();
                let flow = LLMFlow::new(id, FlowType::ChatCompletions, request, metadata);
                store.add(flow);
            }

            // 使用通配符过滤
            let pattern = format!("{}*", prefix);
            let filter = FlowFilter {
                models: Some(vec![pattern.clone()]),
                ..Default::default()
            };

            let results = store.query(&filter);

            // 验证所有结果都匹配通配符模式
            for flow in &results {
                prop_assert!(
                    flow.request.model.to_lowercase().starts_with(&prefix.to_lowercase()),
                    "模型 {} 应该以 {} 开头",
                    flow.request.model,
                    prefix
                );
            }

            // 验证匹配数量正确（应该是 2 个以 prefix 开头的模型）
            prop_assert_eq!(results.len(), 2, "应该有 2 个匹配的 Flow");
        }

        /// **Feature: llm-flow-monitor, Property 4f: 更新操作正确性**
        /// **Validates: Requirements 3.1**
        ///
        /// *对于任意* Flow 和更新操作，更新后的 Flow 应该反映更新内容。
        #[test]
        fn prop_update_correctness(
            id in arb_flow_id(),
            new_state in prop_oneof![
                Just(FlowState::Streaming),
                Just(FlowState::Completed),
                Just(FlowState::Failed),
            ],
        ) {
            let mut store = FlowMemoryStore::new(100);

            let request = LLMRequest::default();
            let metadata = FlowMetadata::default();
            let flow = LLMFlow::new(id.clone(), FlowType::ChatCompletions, request, metadata);

            store.add(flow);

            // 更新状态
            let updated = store.update(&id, |f| {
                f.state = new_state.clone();
            });

            prop_assert!(updated, "更新应该成功");

            // 验证更新生效
            let retrieved = store.get(&id).unwrap();
            let retrieved_flow = retrieved.read().unwrap();
            prop_assert_eq!(
                &retrieved_flow.state,
                &new_state,
                "状态应该被更新"
            );
        }

        /// **Feature: llm-flow-monitor, Property 4g: get_recent 顺序正确性**
        /// **Validates: Requirements 3.1**
        ///
        /// *对于任意* Flow 序列，get_recent 返回的 Flow 应该按添加顺序倒序排列。
        #[test]
        fn prop_get_recent_order(
            count in 5usize..=20usize,
        ) {
            let mut store = FlowMemoryStore::new(100);

            // 添加多个 Flow
            for i in 0..count {
                let id = format!("flow-{:03}", i);
                let request = LLMRequest::default();
                let metadata = FlowMetadata::default();
                let flow = LLMFlow::new(id, FlowType::ChatCompletions, request, metadata);
                store.add(flow);
            }

            // 获取最近的 Flow
            let recent = store.get_recent(count);

            // 验证顺序（最新的在前）
            for (i, flow) in recent.iter().enumerate() {
                let expected_id = format!("flow-{:03}", count - 1 - i);
                prop_assert_eq!(
                    &flow.id,
                    &expected_id,
                    "第 {} 个 Flow 应该是 {}",
                    i,
                    expected_id
                );
            }
        }
    }
}
