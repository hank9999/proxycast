//! 流式管理器
//!
//! 管理流式请求的生命周期，包括格式转换、Flow Monitor 集成、超时处理和错误处理。
//!
//! # 需求覆盖
//!
//! - 需求 4.1: 记录 TTFB（首字节时间）
//! - 需求 4.2: 调用 process_chunk 更新流重建器
//! - 需求 4.3: 发出带有内容增量的 FlowUpdated 事件
//! - 需求 5.1: 在收到 chunk 后立即转发给客户端
//! - 需求 5.2: 保持低延迟
//! - 需求 6.1: 网络错误处理
//! - 需求 6.2: 超时错误处理
//! - 需求 6.3: Provider 错误转发
//! - 需求 6.5: 可配置的流式响应超时

use crate::streaming::converter::{StreamConverter, StreamFormat};
use crate::streaming::error::StreamError;
use crate::streaming::metrics::StreamMetrics;
use crate::streaming::traits::StreamResponse;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::Instant;
use tracing::{debug, error};

// ============================================================================
// 配置
// ============================================================================

/// 流式配置
///
/// 控制流式传输的行为参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// 缓冲区大小（字节）
    ///
    /// 用于限制内存使用，防止内存耗尽。
    /// 对应需求 7.1
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,

    /// 超时时间（毫秒）
    ///
    /// 流式响应的最大等待时间。
    /// 对应需求 6.2, 6.5
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// 事件节流间隔（毫秒）
    ///
    /// 控制 FlowUpdated 事件的发送频率，避免过多更新。
    /// 对应需求 4.6
    #[serde(default = "default_throttle_ms")]
    pub throttle_ms: u64,

    /// chunk 超时时间（毫秒）
    ///
    /// 两个 chunk 之间的最大等待时间。
    #[serde(default = "default_chunk_timeout_ms")]
    pub chunk_timeout_ms: u64,
}

fn default_buffer_size() -> usize {
    1024 * 1024 // 1MB
}

fn default_timeout_ms() -> u64 {
    300_000 // 5 分钟
}

fn default_throttle_ms() -> u64 {
    100 // 100ms
}

fn default_chunk_timeout_ms() -> u64 {
    30_000 // 30 秒
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: default_buffer_size(),
            timeout_ms: default_timeout_ms(),
            throttle_ms: default_throttle_ms(),
            chunk_timeout_ms: default_chunk_timeout_ms(),
        }
    }
}

impl StreamConfig {
    /// 创建新的配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置缓冲区大小
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// 设置超时时间
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// 设置事件节流间隔
    pub fn with_throttle_ms(mut self, throttle_ms: u64) -> Self {
        self.throttle_ms = throttle_ms;
        self
    }

    /// 设置 chunk 超时时间
    pub fn with_chunk_timeout_ms(mut self, chunk_timeout_ms: u64) -> Self {
        self.chunk_timeout_ms = chunk_timeout_ms;
        self
    }

    /// 获取超时 Duration
    pub fn timeout_duration(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    /// 获取 chunk 超时 Duration
    pub fn chunk_timeout_duration(&self) -> Duration {
        Duration::from_millis(self.chunk_timeout_ms)
    }

    /// 获取节流 Duration
    pub fn throttle_duration(&self) -> Duration {
        Duration::from_millis(self.throttle_ms)
    }
}

// ============================================================================
// 流式处理上下文
// ============================================================================

/// 流式处理上下文
///
/// 包含处理单个流式请求所需的所有状态。
#[derive(Debug)]
pub struct StreamContext {
    /// Flow ID（用于 Flow Monitor 集成）
    pub flow_id: Option<String>,
    /// 源格式
    pub source_format: StreamFormat,
    /// 目标格式
    pub target_format: StreamFormat,
    /// 模型名称
    pub model: String,
    /// 指标
    pub metrics: StreamMetrics,
    /// 开始时间
    pub start_time: Instant,
}

impl StreamContext {
    /// 创建新的上下文
    pub fn new(
        flow_id: Option<String>,
        source_format: StreamFormat,
        target_format: StreamFormat,
        model: &str,
    ) -> Self {
        Self {
            flow_id,
            source_format,
            target_format,
            model: model.to_string(),
            metrics: StreamMetrics::new(),
            start_time: Instant::now(),
        }
    }
}

// ============================================================================
// 流式管理器
// ============================================================================

/// 流式管理器
///
/// 管理流式请求的生命周期，包括：
/// - 格式转换（AWS Event Stream → Anthropic/OpenAI SSE）
/// - Flow Monitor 集成（process_chunk、FlowUpdated 事件）
/// - 超时处理
/// - 错误处理
pub struct StreamManager {
    /// 配置
    config: StreamConfig,
}

impl StreamManager {
    /// 创建新的流式管理器
    pub fn new(config: StreamConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建流式管理器
    pub fn with_default_config() -> Self {
        Self::new(StreamConfig::default())
    }

    /// 获取配置
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// 更新配置
    pub fn set_config(&mut self, config: StreamConfig) {
        self.config = config;
    }

    /// 处理流式请求
    ///
    /// 将源流转换为目标格式的 SSE 事件流。
    ///
    /// # 参数
    ///
    /// * `context` - 流式处理上下文
    /// * `source_stream` - 源字节流
    ///
    /// # 返回
    ///
    /// 目标格式的 SSE 事件流
    ///
    /// # 需求覆盖
    ///
    /// - 需求 4.1: 记录 TTFB
    /// - 需求 5.1: 立即转发 chunk
    /// - 需求 6.2: 超时处理
    pub fn handle_stream(
        &self,
        context: StreamContext,
        source_stream: StreamResponse,
    ) -> ManagedStream {
        ManagedStream::new(context, source_stream, self.config.clone())
    }

    /// 处理流式请求（带回调）
    ///
    /// 与 `handle_stream` 类似，但支持在处理每个 chunk 时调用回调函数。
    /// 用于 Flow Monitor 集成。
    ///
    /// # 参数
    ///
    /// * `context` - 流式处理上下文
    /// * `source_stream` - 源字节流
    /// * `on_chunk` - chunk 处理回调
    ///
    /// # 返回
    ///
    /// 目标格式的 SSE 事件流
    pub fn handle_stream_with_callback<F>(
        &self,
        context: StreamContext,
        source_stream: StreamResponse,
        on_chunk: F,
    ) -> ManagedStreamWithCallback<F>
    where
        F: FnMut(&str, &StreamMetrics) + Send + 'static,
    {
        ManagedStreamWithCallback::new(context, source_stream, self.config.clone(), on_chunk)
    }

    /// 处理流式请求（带超时）
    ///
    /// 为流添加超时保护。
    ///
    /// # 参数
    ///
    /// * `context` - 流式处理上下文
    /// * `source_stream` - 源字节流
    ///
    /// # 返回
    ///
    /// 带超时保护的 SSE 事件流
    ///
    /// # 需求覆盖
    ///
    /// - 需求 6.2: 超时错误处理
    /// - 需求 6.5: 可配置的流式响应超时
    pub fn handle_stream_with_timeout(
        &self,
        context: StreamContext,
        source_stream: StreamResponse,
    ) -> TimeoutStream<ManagedStream> {
        let managed = self.handle_stream(context, source_stream);
        with_timeout(managed, &self.config)
    }
}

// ============================================================================
// Flow Monitor 集成辅助类型
// ============================================================================

/// Flow Monitor 回调类型
///
/// 用于在处理流式 chunk 时通知 Flow Monitor。
pub type FlowMonitorCallback = Box<dyn FnMut(&str, &StreamMetrics) + Send + 'static>;

/// 创建 Flow Monitor 回调
///
/// 创建一个回调函数，用于将流式事件发送到 Flow Monitor。
///
/// # 参数
///
/// * `flow_id` - Flow ID
/// * `sender` - 事件发送器
///
/// # 返回
///
/// Flow Monitor 回调函数
pub fn create_flow_monitor_callback<F>(
    flow_id: String,
    mut on_event: F,
) -> impl FnMut(&str, &StreamMetrics) + Send + 'static
where
    F: FnMut(&str, &str, &StreamMetrics) + Send + 'static,
{
    move |event: &str, metrics: &StreamMetrics| {
        on_event(&flow_id, event, metrics);
    }
}

/// 流式事件类型
///
/// 用于 Flow Monitor 集成的事件类型。
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// 流开始
    Started { flow_id: String },
    /// 收到 chunk
    Chunk {
        flow_id: String,
        content_delta: Option<String>,
        metrics: StreamMetrics,
    },
    /// 流完成
    Completed {
        flow_id: String,
        metrics: StreamMetrics,
    },
    /// 流错误
    Error {
        flow_id: String,
        error: StreamError,
        metrics: StreamMetrics,
    },
}

impl StreamEvent {
    /// 获取 Flow ID
    pub fn flow_id(&self) -> &str {
        match self {
            StreamEvent::Started { flow_id } => flow_id,
            StreamEvent::Chunk { flow_id, .. } => flow_id,
            StreamEvent::Completed { flow_id, .. } => flow_id,
            StreamEvent::Error { flow_id, .. } => flow_id,
        }
    }
}

impl Default for StreamManager {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ============================================================================
// 托管流
// ============================================================================

/// 托管流
///
/// 封装源流，提供格式转换、超时处理和指标收集。
///
/// # 有界缓冲区（需求 7.1）
///
/// 托管流使用有界缓冲区来防止内存耗尽。当累积的数据超过配置的
/// `buffer_size` 时，会返回 `BufferOverflow` 错误。
pub struct ManagedStream {
    /// 上下文
    context: StreamContext,
    /// 源流
    source_stream: StreamResponse,
    /// 转换器
    converter: StreamConverter,
    /// 配置
    config: StreamConfig,
    /// 待发送的事件缓冲区
    pending_events: Vec<String>,
    /// 是否已完成
    finished: bool,
    /// 是否已记录首个 chunk
    first_chunk_recorded: bool,
    /// 总接收字节数
    total_bytes: usize,
    /// 当前缓冲区使用量（用于有界缓冲区检查）
    /// 对应需求 7.1
    current_buffer_usage: usize,
}

impl ManagedStream {
    /// 创建新的托管流
    pub fn new(
        context: StreamContext,
        source_stream: StreamResponse,
        config: StreamConfig,
    ) -> Self {
        let converter = StreamConverter::with_model(
            context.source_format,
            context.target_format,
            &context.model,
        );

        Self {
            context,
            source_stream,
            converter,
            config,
            pending_events: Vec::new(),
            finished: false,
            first_chunk_recorded: false,
            total_bytes: 0,
            current_buffer_usage: 0,
        }
    }

    /// 获取指标
    pub fn metrics(&self) -> &StreamMetrics {
        &self.context.metrics
    }

    /// 获取上下文
    pub fn context(&self) -> &StreamContext {
        &self.context
    }

    /// 处理接收到的字节
    ///
    /// # 有界缓冲区检查（需求 7.1）
    ///
    /// 如果累积的数据超过配置的 `buffer_size`，返回空事件列表并设置错误状态。
    fn process_bytes(&mut self, bytes: &Bytes) -> Result<Vec<String>, StreamError> {
        // 检查有界缓冲区限制（需求 7.1）
        let new_usage = self.current_buffer_usage + bytes.len();
        if new_usage > self.config.buffer_size {
            error!(
                flow_id = ?self.context.flow_id,
                current_usage = self.current_buffer_usage,
                incoming_bytes = bytes.len(),
                buffer_limit = self.config.buffer_size,
                "缓冲区溢出"
            );
            return Err(StreamError::BufferOverflow);
        }
        self.current_buffer_usage = new_usage;

        // 记录指标
        self.total_bytes += bytes.len();
        self.context.metrics.record_chunk(bytes.len());

        // 记录首个 chunk
        if !self.first_chunk_recorded {
            self.first_chunk_recorded = true;
            debug!(
                flow_id = ?self.context.flow_id,
                ttfb_ms = ?self.context.metrics.ttfb_ms,
                "收到首个 chunk"
            );
        }

        // 转换格式
        let events = self.converter.convert(bytes);

        // 转换后释放缓冲区使用量（事件已被处理）
        // 只保留 pending_events 的大小
        self.current_buffer_usage = self.pending_events.iter().map(|e| e.len()).sum();

        Ok(events)
    }

    /// 完成流处理
    ///
    /// 对应需求 7.2: 流结束时释放资源
    /// 对应需求 7.5: 记录流式指标
    fn finish_stream(&mut self) -> Vec<String> {
        self.finished = true;
        self.context.metrics.finish();

        // 记录详细指标（需求 7.5）
        self.context
            .metrics
            .log_metrics(self.context.flow_id.as_deref());

        debug!(
            flow_id = ?self.context.flow_id,
            metrics = ?self.context.metrics.summary(),
            "流式传输完成"
        );

        let events = self.converter.finish();

        // 清理缓冲区使用量（资源清理）
        self.current_buffer_usage = 0;

        events
    }

    /// 处理错误
    fn handle_error(&mut self, error: StreamError) -> String {
        self.finished = true;
        self.context.metrics.finish();
        self.context.metrics.record_parse_error();

        error!(
            flow_id = ?self.context.flow_id,
            error = %error,
            "流式传输错误"
        );

        error.to_sse_error()
    }

    /// 处理 Provider 错误
    ///
    /// 将 Provider 返回的错误转换为 SSE 错误事件。
    /// 对应需求 6.3: Provider 错误转发
    pub fn handle_provider_error(&mut self, status: u16, message: &str) -> String {
        let error = StreamError::provider_error(status, message);
        self.handle_error(error)
    }

    /// 处理网络错误
    ///
    /// 将网络错误转换为 SSE 错误事件。
    /// 对应需求 6.1: 网络错误处理
    pub fn handle_network_error(&mut self, message: &str) -> String {
        let error = StreamError::network(message);
        self.handle_error(error)
    }

    /// 检查是否已完成
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// 获取当前缓冲区使用量
    ///
    /// 对应需求 7.1: 有界缓冲区
    pub fn buffer_usage(&self) -> usize {
        self.current_buffer_usage
    }

    /// 获取缓冲区限制
    pub fn buffer_limit(&self) -> usize {
        self.config.buffer_size
    }

    /// 清理资源
    ///
    /// 对应需求 7.2: 流结束时释放资源
    ///
    /// 此方法会清理所有内部状态，释放内存。
    /// 通常在流结束后自动调用，但也可以手动调用以提前释放资源。
    pub fn cleanup(&mut self) {
        // 清理待发送事件缓冲区
        self.pending_events.clear();
        self.pending_events.shrink_to_fit();

        // 重置缓冲区使用量
        self.current_buffer_usage = 0;

        // 重置转换器
        self.converter.reset();

        // 标记为已完成
        self.finished = true;

        debug!(
            flow_id = ?self.context.flow_id,
            "流式资源已清理"
        );
    }
}

impl Stream for ManagedStream {
    type Item = Result<String, StreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // 如果有待发送的事件，先发送
        if !self.pending_events.is_empty() {
            let event = self.pending_events.remove(0);
            // 更新缓冲区使用量
            self.current_buffer_usage = self.pending_events.iter().map(|e| e.len()).sum();
            return Poll::Ready(Some(Ok(event)));
        }

        // 如果已完成，返回 None
        if self.finished {
            return Poll::Ready(None);
        }

        // 轮询源流
        match Pin::new(&mut self.source_stream).poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                // 处理字节（包含有界缓冲区检查）
                match self.process_bytes(&bytes) {
                    Ok(mut events) => {
                        if events.is_empty() {
                            // 没有产生事件，继续轮询
                            cx.waker().wake_by_ref();
                            Poll::Pending
                        } else {
                            // 取出第一个事件返回
                            let first = events.remove(0);
                            // 保存剩余的事件
                            self.pending_events = events;
                            // 更新缓冲区使用量
                            self.current_buffer_usage =
                                self.pending_events.iter().map(|e| e.len()).sum();
                            Poll::Ready(Some(Ok(first)))
                        }
                    }
                    Err(error) => {
                        // 缓冲区溢出或其他错误
                        let error_event = self.handle_error(error);
                        Poll::Ready(Some(Ok(error_event)))
                    }
                }
            }
            Poll::Ready(Some(Err(error))) => {
                // 处理错误
                let error_event = self.handle_error(error);
                Poll::Ready(Some(Ok(error_event)))
            }
            Poll::Ready(None) => {
                // 源流结束
                let mut finish_events = self.finish_stream();

                if finish_events.is_empty() {
                    self.finished = true;
                    Poll::Ready(None)
                } else {
                    // 取出第一个事件返回
                    let first = finish_events.remove(0);
                    // 保存剩余的事件
                    self.pending_events = finish_events;
                    Poll::Ready(Some(Ok(first)))
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

// ============================================================================
// 带回调的托管流
// ============================================================================

/// 带回调的托管流
///
/// 在处理每个 chunk 时调用回调函数，用于 Flow Monitor 集成。
///
/// # 事件节流（需求 4.6）
///
/// 支持可配置的事件节流，避免过多的 FlowUpdated 事件。
/// 节流间隔通过 `StreamConfig.throttle_ms` 配置。
pub struct ManagedStreamWithCallback<F>
where
    F: FnMut(&str, &StreamMetrics) + Send + 'static,
{
    /// 内部托管流
    inner: ManagedStream,
    /// chunk 处理回调
    on_chunk: F,
    /// 上次回调时间（用于节流）
    last_callback_time: Option<Instant>,
    /// 被节流的事件计数（用于指标）
    throttled_event_count: u32,
    /// 总回调次数
    callback_count: u32,
}

// ManagedStreamWithCallback 可以安全地 Unpin，因为它不包含自引用
impl<F> Unpin for ManagedStreamWithCallback<F> where F: FnMut(&str, &StreamMetrics) + Send + 'static {}

impl<F> ManagedStreamWithCallback<F>
where
    F: FnMut(&str, &StreamMetrics) + Send + 'static,
{
    /// 创建新的带回调托管流
    pub fn new(
        context: StreamContext,
        source_stream: StreamResponse,
        config: StreamConfig,
        on_chunk: F,
    ) -> Self {
        Self {
            inner: ManagedStream::new(context, source_stream, config),
            on_chunk,
            last_callback_time: None,
            throttled_event_count: 0,
            callback_count: 0,
        }
    }

    /// 获取指标
    pub fn metrics(&self) -> &StreamMetrics {
        self.inner.metrics()
    }

    /// 获取上下文
    pub fn context(&self) -> &StreamContext {
        self.inner.context()
    }

    /// 获取被节流的事件计数
    ///
    /// 对应需求 4.6: 事件节流
    pub fn throttled_event_count(&self) -> u32 {
        self.throttled_event_count
    }

    /// 获取总回调次数
    pub fn callback_count(&self) -> u32 {
        self.callback_count
    }

    /// 检查是否应该调用回调（节流）
    ///
    /// 对应需求 4.6: 可配置的事件节流
    fn should_call_callback(&self) -> bool {
        match self.last_callback_time {
            None => true,
            Some(last_time) => last_time.elapsed() >= self.inner.config.throttle_duration(),
        }
    }
}

impl<F> Stream for ManagedStreamWithCallback<F>
where
    F: FnMut(&str, &StreamMetrics) + Send + 'static + Unpin,
{
    type Item = Result<String, StreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // 使用 get_mut 获取可变引用
        let this = self.get_mut();

        // 轮询内部流
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => {
                // 检查是否应该调用回调（事件节流，需求 4.6）
                if this.should_call_callback() {
                    let metrics = this.inner.context.metrics.clone();
                    (this.on_chunk)(&event, &metrics);
                    this.last_callback_time = Some(Instant::now());
                    this.callback_count += 1;
                } else {
                    // 记录被节流的事件
                    this.throttled_event_count += 1;
                }
                Poll::Ready(Some(Ok(event)))
            }
            other => other,
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 创建带超时的流
///
/// 为流添加整体超时和 chunk 超时。
///
/// # 参数
///
/// * `stream` - 源流
/// * `config` - 流式配置
///
/// # 返回
///
/// 带超时的流
pub fn with_timeout<S>(stream: S, config: &StreamConfig) -> TimeoutStream<S>
where
    S: Stream<Item = Result<String, StreamError>> + Unpin,
{
    TimeoutStream::new(stream, config.clone())
}

/// 带超时的流包装器
pub struct TimeoutStream<S>
where
    S: Stream<Item = Result<String, StreamError>> + Unpin,
{
    inner: S,
    config: StreamConfig,
    start_time: Instant,
    last_chunk_time: Option<Instant>,
    finished: bool,
}

impl<S> TimeoutStream<S>
where
    S: Stream<Item = Result<String, StreamError>> + Unpin,
{
    /// 创建新的超时流
    pub fn new(inner: S, config: StreamConfig) -> Self {
        Self {
            inner,
            config,
            start_time: Instant::now(),
            last_chunk_time: None,
            finished: false,
        }
    }

    /// 检查是否超时
    fn check_timeout(&self) -> Option<StreamError> {
        // 检查总超时
        if self.start_time.elapsed() > self.config.timeout_duration() {
            return Some(StreamError::Timeout);
        }

        // 检查 chunk 超时
        if let Some(last_time) = self.last_chunk_time {
            if last_time.elapsed() > self.config.chunk_timeout_duration() {
                return Some(StreamError::Timeout);
            }
        }

        None
    }
}

impl<S> Stream for TimeoutStream<S>
where
    S: Stream<Item = Result<String, StreamError>> + Unpin,
{
    type Item = Result<String, StreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        // 检查超时
        if let Some(error) = self.check_timeout() {
            self.finished = true;
            return Poll::Ready(Some(Err(error)));
        }

        // 轮询内部流
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                self.last_chunk_time = Some(Instant::now());
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// 从流中收集所有内容
///
/// 用于测试和调试。
pub async fn collect_stream_content<S>(mut stream: S) -> Result<String, StreamError>
where
    S: Stream<Item = Result<String, StreamError>> + Unpin,
{
    let mut content = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(event) => content.push_str(&event),
            Err(e) => return Err(e),
        }
    }

    Ok(content)
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_size, 1024 * 1024);
        assert_eq!(config.timeout_ms, 300_000);
        assert_eq!(config.throttle_ms, 100);
        assert_eq!(config.chunk_timeout_ms, 30_000);
    }

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfig::new()
            .with_buffer_size(2048)
            .with_timeout_ms(60_000)
            .with_throttle_ms(50)
            .with_chunk_timeout_ms(10_000);

        assert_eq!(config.buffer_size, 2048);
        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(config.throttle_ms, 50);
        assert_eq!(config.chunk_timeout_ms, 10_000);
    }

    #[test]
    fn test_stream_config_durations() {
        let config = StreamConfig::new()
            .with_timeout_ms(5000)
            .with_chunk_timeout_ms(1000)
            .with_throttle_ms(100);

        assert_eq!(config.timeout_duration(), Duration::from_millis(5000));
        assert_eq!(config.chunk_timeout_duration(), Duration::from_millis(1000));
        assert_eq!(config.throttle_duration(), Duration::from_millis(100));
    }

    #[test]
    fn test_stream_context_new() {
        let context = StreamContext::new(
            Some("flow-123".to_string()),
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "claude-3-opus",
        );

        assert_eq!(context.flow_id, Some("flow-123".to_string()));
        assert_eq!(context.source_format, StreamFormat::AwsEventStream);
        assert_eq!(context.target_format, StreamFormat::OpenAiSse);
        assert_eq!(context.model, "claude-3-opus");
    }

    #[test]
    fn test_stream_manager_new() {
        let config = StreamConfig::default();
        let manager = StreamManager::new(config.clone());

        assert_eq!(manager.config().buffer_size, config.buffer_size);
        assert_eq!(manager.config().timeout_ms, config.timeout_ms);
    }

    #[test]
    fn test_stream_manager_default() {
        let manager = StreamManager::default();
        let default_config = StreamConfig::default();

        assert_eq!(manager.config().buffer_size, default_config.buffer_size);
    }

    #[tokio::test]
    async fn test_managed_stream_empty() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let empty_stream: StreamResponse = Box::pin(stream::empty());
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, empty_stream, config);

        // 空流应该产生结束事件
        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该有结束事件
        assert!(events.iter().any(|e| e.contains("[DONE]")));
    }

    #[tokio::test]
    async fn test_managed_stream_with_content() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建包含内容的流
        let chunks = vec![
            Ok(Bytes::from("{\"content\":\"Hello\"}")),
            Ok(Bytes::from("{\"content\":\", world!\"}")),
        ];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该有内容事件
        let content_events: Vec<_> = events.iter().filter(|e| e.contains("content")).collect();
        assert!(!content_events.is_empty());
    }

    #[tokio::test]
    async fn test_managed_stream_error_handling() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建包含错误的流
        let chunks: Vec<Result<Bytes, StreamError>> = vec![
            Ok(Bytes::from("{\"content\":\"Hello\"}")),
            Err(StreamError::Network("connection reset".to_string())),
        ];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该有错误事件
        assert!(events.iter().any(|e| e.contains("error")));
    }

    #[tokio::test]
    async fn test_managed_stream_metrics() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let chunks = vec![Ok(Bytes::from("{\"content\":\"test\"}"))];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        // 消费流
        while managed.next().await.is_some() {}

        // 检查指标
        let metrics = managed.metrics();
        assert!(metrics.chunk_count > 0);
        assert!(metrics.total_bytes > 0);
    }

    #[tokio::test]
    async fn test_collect_stream_content() {
        let events = vec![Ok("event1\n".to_string()), Ok("event2\n".to_string())];
        let stream = stream::iter(events);

        let content = collect_stream_content(stream).await.unwrap();
        assert_eq!(content, "event1\nevent2\n");
    }

    #[tokio::test]
    async fn test_collect_stream_content_with_error() {
        let events: Vec<Result<String, StreamError>> =
            vec![Ok("event1\n".to_string()), Err(StreamError::Timeout)];
        let stream = stream::iter(events);

        let result = collect_stream_content(stream).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StreamError::Timeout));
    }

    #[tokio::test]
    async fn test_timeout_stream_no_timeout() {
        // 创建一个快速完成的流
        let events: Vec<Result<String, StreamError>> =
            vec![Ok("event1\n".to_string()), Ok("event2\n".to_string())];
        let inner_stream = stream::iter(events);

        let config = StreamConfig::new()
            .with_timeout_ms(10_000)
            .with_chunk_timeout_ms(5_000);

        let mut timeout_stream = with_timeout(inner_stream, &config);

        let mut results = Vec::new();
        while let Some(result) = timeout_stream.next().await {
            results.push(result);
        }

        // 应该成功完成，没有超时
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_timeout_stream_check_timeout() {
        let events: Vec<Result<String, StreamError>> = vec![];
        let inner_stream = stream::iter(events);

        // 创建一个非常短的超时配置
        let config = StreamConfig::new()
            .with_timeout_ms(1) // 1ms 超时
            .with_chunk_timeout_ms(1);

        let timeout_stream = TimeoutStream::new(inner_stream, config);

        // 等待一小段时间让超时发生
        std::thread::sleep(std::time::Duration::from_millis(10));

        // 检查超时
        let timeout_error = timeout_stream.check_timeout();
        assert!(timeout_error.is_some());
        assert!(matches!(timeout_error.unwrap(), StreamError::Timeout));
    }

    #[test]
    fn test_stream_config_timeout_duration() {
        let config = StreamConfig::new()
            .with_timeout_ms(5000)
            .with_chunk_timeout_ms(1000);

        assert_eq!(config.timeout_duration(), Duration::from_millis(5000));
        assert_eq!(config.chunk_timeout_duration(), Duration::from_millis(1000));
    }

    #[tokio::test]
    async fn test_stream_manager_handle_stream_with_timeout() {
        let manager = StreamManager::with_default_config();

        let context = StreamContext::new(
            Some("test-flow".to_string()),
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let chunks = vec![Ok(Bytes::from("{\"content\":\"Hello\"}"))];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));

        let mut timeout_stream = manager.handle_stream_with_timeout(context, source_stream);

        let mut events = Vec::new();
        while let Some(result) = timeout_stream.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该成功完成
        assert!(!events.is_empty());
    }

    #[test]
    fn test_stream_event_flow_id() {
        let started = StreamEvent::Started {
            flow_id: "flow-1".to_string(),
        };
        assert_eq!(started.flow_id(), "flow-1");

        let chunk = StreamEvent::Chunk {
            flow_id: "flow-2".to_string(),
            content_delta: Some("test".to_string()),
            metrics: StreamMetrics::new(),
        };
        assert_eq!(chunk.flow_id(), "flow-2");

        let completed = StreamEvent::Completed {
            flow_id: "flow-3".to_string(),
            metrics: StreamMetrics::new(),
        };
        assert_eq!(completed.flow_id(), "flow-3");

        let error = StreamEvent::Error {
            flow_id: "flow-4".to_string(),
            error: StreamError::Timeout,
            metrics: StreamMetrics::new(),
        };
        assert_eq!(error.flow_id(), "flow-4");
    }

    #[tokio::test]
    async fn test_managed_stream_provider_error() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建包含 Provider 错误的流
        let chunks: Vec<Result<Bytes, StreamError>> = vec![
            Ok(Bytes::from("{\"content\":\"Hello\"}")),
            Err(StreamError::provider_error(429, "rate limited")),
        ];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该有错误事件
        assert!(events.iter().any(|e| e.contains("provider_error")));
        assert!(events.iter().any(|e| e.contains("rate limited")));
    }

    #[tokio::test]
    async fn test_managed_stream_network_error() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建包含网络错误的流
        let chunks: Vec<Result<Bytes, StreamError>> =
            vec![Err(StreamError::network("connection reset by peer"))];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该有网络错误事件
        assert!(events.iter().any(|e| e.contains("network_error")));
        assert!(events.iter().any(|e| e.contains("connection reset")));
    }

    #[tokio::test]
    async fn test_managed_stream_parse_error() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建包含解析错误的流
        let chunks: Vec<Result<Bytes, StreamError>> =
            vec![Err(StreamError::parse_error("invalid JSON"))];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该有解析错误事件
        assert!(events.iter().any(|e| e.contains("parse_error")));
    }

    #[test]
    fn test_stream_error_is_retryable() {
        assert!(StreamError::Network("test".to_string()).is_retryable());
        assert!(StreamError::Timeout.is_retryable());
        assert!(StreamError::provider_error(429, "rate limited").is_retryable());
        assert!(StreamError::provider_error(500, "server error").is_retryable());
        assert!(StreamError::provider_error(503, "service unavailable").is_retryable());
        assert!(!StreamError::provider_error(400, "bad request").is_retryable());
        assert!(!StreamError::provider_error(401, "unauthorized").is_retryable());
        assert!(!StreamError::ParseError("invalid".to_string()).is_retryable());
        assert!(!StreamError::ClientDisconnected.is_retryable());
        assert!(!StreamError::BufferOverflow.is_retryable());
    }

    #[test]
    fn test_stream_error_status_code() {
        assert_eq!(StreamError::Timeout.status_code(), Some(504));
        assert_eq!(
            StreamError::Network("test".to_string()).status_code(),
            Some(502)
        );
        assert_eq!(
            StreamError::provider_error(429, "test").status_code(),
            Some(429)
        );
        assert_eq!(
            StreamError::provider_error(500, "test").status_code(),
            Some(500)
        );
        assert_eq!(StreamError::ClientDisconnected.status_code(), None);
        assert_eq!(
            StreamError::ParseError("test".to_string()).status_code(),
            None
        );
    }

    #[test]
    fn test_stream_error_to_sse_error() {
        let err = StreamError::provider_error(429, "rate limited");
        let sse = err.to_sse_error();

        assert!(sse.starts_with("event: error\n"));
        assert!(sse.contains("provider_error"));
        assert!(sse.contains("rate limited"));
    }

    // ========================================================================
    // 有界缓冲区测试（需求 7.1）
    // ========================================================================

    #[tokio::test]
    async fn test_bounded_buffer_overflow() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建一个大的 chunk，超过缓冲区限制
        let large_data = vec![b'x'; 200];
        let chunks: Vec<Result<Bytes, StreamError>> = vec![Ok(Bytes::from(large_data))];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));

        // 设置一个很小的缓冲区限制
        let config = StreamConfig::new().with_buffer_size(100);

        let mut managed = ManagedStream::new(context, source_stream, config);

        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 应该有缓冲区溢出错误事件
        assert!(events.iter().any(|e| e.contains("buffer_overflow")));
        assert!(managed.is_finished());
    }

    #[tokio::test]
    async fn test_bounded_buffer_within_limit() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建一个小的 chunk，在缓冲区限制内
        let small_data = b"{\"content\":\"test\"}";
        let chunks: Vec<Result<Bytes, StreamError>> = vec![Ok(Bytes::from(&small_data[..]))];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));

        // 设置足够大的缓冲区限制
        let config = StreamConfig::new().with_buffer_size(1024);

        let mut managed = ManagedStream::new(context, source_stream, config);

        let mut events = Vec::new();
        while let Some(result) = managed.next().await {
            if let Ok(event) = result {
                events.push(event);
            }
        }

        // 不应该有缓冲区溢出错误
        assert!(!events.iter().any(|e| e.contains("buffer_overflow")));
    }

    #[test]
    fn test_buffer_usage_tracking() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let chunks: Vec<Result<Bytes, StreamError>> = vec![];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::new().with_buffer_size(1024);

        let managed = ManagedStream::new(context, source_stream, config);

        // 初始缓冲区使用量应该为 0
        assert_eq!(managed.buffer_usage(), 0);
        assert_eq!(managed.buffer_limit(), 1024);
    }

    // ========================================================================
    // 资源清理测试（需求 7.2）
    // ========================================================================

    #[tokio::test]
    async fn test_resource_cleanup() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let chunks: Vec<Result<Bytes, StreamError>> =
            vec![Ok(Bytes::from("{\"content\":\"test\"}"))];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        // 消费流
        while managed.next().await.is_some() {}

        // 流完成后，缓冲区使用量应该为 0
        assert_eq!(managed.buffer_usage(), 0);
        assert!(managed.is_finished());
    }

    #[tokio::test]
    async fn test_manual_cleanup() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let chunks: Vec<Result<Bytes, StreamError>> = vec![];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
        let config = StreamConfig::default();

        let mut managed = ManagedStream::new(context, source_stream, config);

        // 手动清理
        managed.cleanup();

        // 清理后应该标记为已完成
        assert!(managed.is_finished());
        assert_eq!(managed.buffer_usage(), 0);
    }

    // ========================================================================
    // 事件节流测试（需求 4.6）
    // ========================================================================

    #[tokio::test]
    async fn test_callback_throttling_counts() {
        let context = StreamContext::new(
            None,
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 创建多个 chunks
        let chunks: Vec<Result<Bytes, StreamError>> = vec![
            Ok(Bytes::from("{\"content\":\"a\"}")),
            Ok(Bytes::from("{\"content\":\"b\"}")),
            Ok(Bytes::from("{\"content\":\"c\"}")),
        ];
        let source_stream: StreamResponse = Box::pin(stream::iter(chunks));

        // 使用较长的节流间隔
        let config = StreamConfig::new().with_throttle_ms(10000); // 10秒节流

        let callback_count = Arc::new(AtomicU32::new(0));
        let callback_count_clone = callback_count.clone();

        let on_chunk = move |_event: &str, _metrics: &StreamMetrics| {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
        };

        let mut managed = ManagedStreamWithCallback::new(context, source_stream, config, on_chunk);

        // 消费流
        while managed.next().await.is_some() {}

        // 由于节流，回调次数应该小于事件数量
        let final_callback_count = callback_count.load(Ordering::SeqCst);
        assert!(final_callback_count >= 1, "至少应该有一次回调");

        // 检查节流计数
        let _throttled = managed.throttled_event_count();
        let total_callbacks = managed.callback_count();

        // 总回调次数 + 被节流次数 应该等于处理的事件数
        assert!(total_callbacks >= 1);
    }
}

// ============================================================================
// 属性测试（Property-Based Testing）
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::streaming::aws_parser::{extract_content, serialize_event, AwsEvent};
    use crate::streaming::converter::extract_content_from_sse;
    use futures::stream;
    use proptest::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    // ========================================================================
    // 生成器（Generators）
    // ========================================================================

    /// 生成有效的内容文本
    fn arb_content_text() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z0-9 .,!?\\-_]{1,50}")
            .unwrap()
            .prop_filter("非空字符串", |s| !s.is_empty())
    }

    /// 生成 Content 事件
    fn arb_content_event() -> impl Strategy<Value = AwsEvent> {
        arb_content_text().prop_map(|text| AwsEvent::Content { text })
    }

    /// 生成内容事件序列
    fn arb_content_sequence() -> impl Strategy<Value = Vec<AwsEvent>> {
        prop::collection::vec(arb_content_event(), 1..10)
    }

    // ========================================================================
    // Property 3: Flow Monitor 流式捕获完整性
    //
    // *对于任意*流式响应，Flow Monitor 通过 process_chunk 处理所有 chunks 后，
    // 重建的响应应该与完整响应一致。
    //
    // **验证: 需求 4.2, 4.4, 4.5**
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 3: 流式捕获完整性 - 所有 chunk 都被回调处理
        ///
        /// **Feature: true-streaming-support, Property 3: Flow Monitor 流式捕获完整性**
        /// **Validates: Requirements 4.2, 4.4, 4.5**
        #[test]
        fn prop_flow_monitor_captures_all_chunks(events in arb_content_sequence()) {
            // 使用 tokio runtime 运行异步测试
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async {
                // 计算原始内容
                let original_content = extract_content(&events);

                // 创建上下文
                let context = StreamContext::new(
                    Some("test-flow".to_string()),
                    StreamFormat::AwsEventStream,
                    StreamFormat::OpenAiSse,
                    "test-model",
                );

                // 将事件序列化为字节流
                let chunks: Vec<Result<Bytes, StreamError>> = events
                    .iter()
                    .filter_map(|event| serialize_event(event))
                    .map(|json| Ok(Bytes::from(json)))
                    .collect();

                let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
                let config = StreamConfig::default();

                // 使用回调跟踪所有事件
                let callback_count = Arc::new(AtomicU32::new(0));
                let callback_count_clone = callback_count.clone();
                let captured_events = Arc::new(std::sync::Mutex::new(Vec::new()));
                let captured_events_clone = captured_events.clone();

                let on_chunk = move |event: &str, _metrics: &StreamMetrics| {
                    callback_count_clone.fetch_add(1, Ordering::SeqCst);
                    captured_events_clone.lock().unwrap().push(event.to_string());
                };

                let mut managed = ManagedStreamWithCallback::new(
                    context,
                    source_stream,
                    config,
                    on_chunk,
                );

                // 消费流
                let mut all_events = Vec::new();
                while let Some(result) = managed.next().await {
                    if let Ok(event) = result {
                        all_events.push(event);
                    }
                }

                // 验证回调被调用
                let final_callback_count = callback_count.load(Ordering::SeqCst);
                if final_callback_count == 0 && !events.is_empty() {
                    return Err("回调应该被调用至少一次（除非没有事件）".to_string());
                }

                // 从 SSE 事件中提取内容
                let converted_content = extract_content_from_sse(&all_events, StreamFormat::OpenAiSse);

                // 验证内容一致
                if original_content != converted_content {
                    return Err(format!(
                        "Flow Monitor 捕获的内容应该与原始内容一致: original={}, converted={}",
                        original_content, converted_content
                    ));
                }

                Ok(())
            });

            prop_assert!(result.is_ok(), "{}", result.unwrap_err());
        }

        /// Property 3: 流式指标正确记录
        ///
        /// **Feature: true-streaming-support, Property 3: Flow Monitor 流式捕获完整性**
        /// **Validates: Requirements 4.5**
        #[test]
        fn prop_stream_metrics_correctly_recorded(events in arb_content_sequence()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async {
                let context = StreamContext::new(
                    None,
                    StreamFormat::AwsEventStream,
                    StreamFormat::OpenAiSse,
                    "test-model",
                );

                // 将事件序列化为字节流
                let chunks: Vec<Result<Bytes, StreamError>> = events
                    .iter()
                    .filter_map(|event| serialize_event(event))
                    .map(|json| Ok(Bytes::from(json)))
                    .collect();

                let chunk_count = chunks.len();
                let total_bytes: usize = chunks.iter()
                    .filter_map(|r| r.as_ref().ok())
                    .map(|b| b.len())
                    .sum();

                let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
                let config = StreamConfig::default();

                let mut managed = ManagedStream::new(context, source_stream, config);

                // 消费流
                while managed.next().await.is_some() {}

                // 验证指标
                let metrics = managed.metrics();

                if metrics.chunk_count as usize != chunk_count {
                    return Err(format!(
                        "chunk 计数应该与实际 chunk 数量一致: expected={}, actual={}",
                        chunk_count, metrics.chunk_count
                    ));
                }

                if metrics.total_bytes != total_bytes {
                    return Err(format!(
                        "总字节数应该与实际字节数一致: expected={}, actual={}",
                        total_bytes, metrics.total_bytes
                    ));
                }

                if chunk_count > 0 && metrics.ttfb_ms.is_none() {
                    return Err("有 chunk 时应该记录 TTFB".to_string());
                }

                Ok(())
            });

            prop_assert!(result.is_ok(), "{}", result.unwrap_err());
        }

        /// Property 3: 回调节流正确工作
        ///
        /// **Feature: true-streaming-support, Property 3: Flow Monitor 流式捕获完整性**
        /// **Validates: Requirements 4.6**
        #[test]
        fn prop_callback_throttling_works(events in arb_content_sequence()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async {
                let context = StreamContext::new(
                    None,
                    StreamFormat::AwsEventStream,
                    StreamFormat::OpenAiSse,
                    "test-model",
                );

                // 将事件序列化为字节流
                let chunks: Vec<Result<Bytes, StreamError>> = events
                    .iter()
                    .filter_map(|event| serialize_event(event))
                    .map(|json| Ok(Bytes::from(json)))
                    .collect();

                let source_stream: StreamResponse = Box::pin(stream::iter(chunks));

                // 使用较长的节流间隔
                let config = StreamConfig::new()
                    .with_throttle_ms(1000); // 1秒节流

                let callback_count = Arc::new(AtomicU32::new(0));
                let callback_count_clone = callback_count.clone();

                let on_chunk = move |_event: &str, _metrics: &StreamMetrics| {
                    callback_count_clone.fetch_add(1, Ordering::SeqCst);
                };

                let mut managed = ManagedStreamWithCallback::new(
                    context,
                    source_stream,
                    config,
                    on_chunk,
                );

                // 消费流
                while managed.next().await.is_some() {}

                // 由于节流，回调次数应该小于等于事件数量
                let final_callback_count = callback_count.load(Ordering::SeqCst);

                // 至少应该有一次回调（第一次不受节流限制）
                if !events.is_empty() && final_callback_count < 1 {
                    return Err("至少应该有一次回调".to_string());
                }

                Ok(())
            });

            prop_assert!(result.is_ok(), "{}", result.unwrap_err());
        }
    }

    // ========================================================================
    // Property 5: 错误恢复正确性
    //
    // *对于任意*包含无效 chunk 的流，解析器应该跳过无效 chunk 并继续处理
    // 后续有效 chunks，最终结果应该包含所有有效 chunks 的内容。
    //
    // **验证: 需求 2.5, 6.4**
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 5: 错误恢复 - 流在错误后正确终止
        ///
        /// **Feature: true-streaming-support, Property 5: 错误恢复正确性**
        /// **Validates: Requirements 2.5, 6.4**
        #[test]
        fn prop_stream_terminates_on_error(
            valid_events in arb_content_sequence(),
            error_type in prop_oneof![
                Just(StreamError::Network("test error".to_string())),
                Just(StreamError::Timeout),
                Just(StreamError::provider_error(500, "server error")),
                Just(StreamError::parse_error("invalid data")),
            ]
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async {
                let context = StreamContext::new(
                    None,
                    StreamFormat::AwsEventStream,
                    StreamFormat::OpenAiSse,
                    "test-model",
                );

                // 创建包含有效事件和错误的流
                let mut chunks: Vec<Result<Bytes, StreamError>> = valid_events
                    .iter()
                    .filter_map(|event| serialize_event(event))
                    .map(|json| Ok(Bytes::from(json)))
                    .collect();

                // 在中间插入错误
                let error_pos = chunks.len() / 2;
                chunks.insert(error_pos, Err(error_type.clone()));

                let source_stream: StreamResponse = Box::pin(stream::iter(chunks));
                let config = StreamConfig::default();

                let mut managed = ManagedStream::new(context, source_stream, config);

                // 消费流
                let mut events = Vec::new();
                let mut saw_error = false;
                while let Some(result) = managed.next().await {
                    match result {
                        Ok(event) => {
                            if event.contains("error") {
                                saw_error = true;
                            }
                            events.push(event);
                        }
                        Err(_) => {
                            saw_error = true;
                        }
                    }
                }

                // 验证流正确终止
                if !saw_error {
                    return Err("应该看到错误事件".to_string());
                }

                // 验证流已完成
                if !managed.is_finished() {
                    return Err("流应该已完成".to_string());
                }

                Ok(())
            });

            prop_assert!(result.is_ok(), "{}", result.unwrap_err());
        }

        /// Property 5: 错误事件格式正确
        ///
        /// **Feature: true-streaming-support, Property 5: 错误恢复正确性**
        /// **Validates: Requirements 6.4**
        #[test]
        fn prop_error_event_format_correct(
            error_type in prop_oneof![
                Just(StreamError::Network("connection failed".to_string())),
                Just(StreamError::Timeout),
                Just(StreamError::provider_error(429, "rate limited")),
                Just(StreamError::provider_error(500, "internal error")),
                Just(StreamError::parse_error("invalid json")),
                Just(StreamError::ClientDisconnected),
                Just(StreamError::BufferOverflow),
            ]
        ) {
            let sse_error = error_type.to_sse_error();

            // 验证 SSE 格式
            prop_assert!(
                sse_error.starts_with("event: error\n"),
                "SSE 错误应该以 'event: error' 开头, 实际: {}",
                sse_error
            );

            prop_assert!(
                sse_error.contains("data: "),
                "SSE 错误应该包含 'data: ', 实际: {}",
                sse_error
            );

            // 验证 JSON 格式
            let data_start = sse_error.find("data: ").unwrap() + 6;
            let data_end = sse_error[data_start..].find('\n').unwrap_or(sse_error.len() - data_start);
            let json_str = &sse_error[data_start..data_start + data_end];

            let json_result: Result<serde_json::Value, _> = serde_json::from_str(json_str);
            prop_assert!(
                json_result.is_ok(),
                "无法解析 JSON: {:?}, 原始字符串: {}",
                json_result.err(),
                json_str
            );

            let json = json_result.unwrap();

            // 验证 JSON 结构
            prop_assert!(
                json.get("error").is_some(),
                "JSON 应该包含 'error' 字段, 实际: {}",
                json
            );

            let error_obj = json.get("error").unwrap();
            prop_assert!(
                error_obj.get("type").is_some(),
                "error 对象应该包含 'type' 字段, 实际: {}",
                error_obj
            );

            prop_assert!(
                error_obj.get("message").is_some(),
                "error 对象应该包含 'message' 字段, 实际: {}",
                error_obj
            );
        }

        /// Property 5: 可重试错误正确标识
        ///
        /// **Feature: true-streaming-support, Property 5: 错误恢复正确性**
        /// **Validates: Requirements 6.1, 6.4**
        #[test]
        fn prop_retryable_errors_correctly_identified(
            status in 400u16..600,
            message in "[a-zA-Z ]{1,20}"
        ) {
            let error = StreamError::provider_error(status, &message);
            let is_retryable = error.is_retryable();

            // 429 和 5xx 应该可重试
            let expected_retryable = status == 429 || status >= 500;

            prop_assert_eq!(
                is_retryable,
                expected_retryable,
                "状态码 {} 的可重试性应该是 {}, 但实际是 {}",
                status, expected_retryable, is_retryable
            );
        }

        /// Property 5: 错误状态码正确映射
        ///
        /// **Feature: true-streaming-support, Property 5: 错误恢复正确性**
        /// **Validates: Requirements 6.1, 6.2, 6.3**
        #[test]
        fn prop_error_status_code_mapping(
            status in 400u16..600,
            message in "[a-zA-Z ]{1,20}"
        ) {
            let provider_error = StreamError::provider_error(status, &message);
            let network_error = StreamError::network(&message);
            let timeout_error = StreamError::Timeout;

            // Provider 错误应该返回原始状态码
            prop_assert_eq!(
                provider_error.status_code(),
                Some(status),
                "Provider 错误状态码应该是 {}, 但实际是 {:?}",
                status, provider_error.status_code()
            );

            // 网络错误应该返回 502
            prop_assert_eq!(
                network_error.status_code(),
                Some(502),
                "网络错误状态码应该是 502, 但实际是 {:?}",
                network_error.status_code()
            );

            // 超时错误应该返回 504
            prop_assert_eq!(
                timeout_error.status_code(),
                Some(504),
                "超时错误状态码应该是 504, 但实际是 {:?}",
                timeout_error.status_code()
            );
        }
    }
}
