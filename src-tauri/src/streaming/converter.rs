//! 流式格式转换器
//!
//! 在不同流式格式之间转换，支持 AWS Event Stream、Anthropic SSE 和 OpenAI SSE。
//!
//! # 需求覆盖
//!
//! - 需求 3.1: AWS Event Stream 到 Anthropic SSE 转换
//! - 需求 3.2: AWS Event Stream 到 OpenAI SSE 转换
//! - 需求 3.3: Anthropic SSE 到 OpenAI SSE 转换
//! - 需求 3.5: 处理工具调用参数中的部分 JSON

use crate::streaming::aws_parser::{AwsEvent, AwsEventStreamParser};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KiroThinkingTagParseState {
    /// 尚不确定是否需要解析 `<thinking>` 标签
    Armed,
    /// 已确定这是 Kiro thinking 标签输出，开始解析并拆分 thinking/content
    Enabled,
    /// 明确不解析（当作普通文本）
    Disabled,
}

impl Default for KiroThinkingTagParseState {
    fn default() -> Self {
        Self::Armed
    }
}

/// Kiro `<thinking>...</thinking>` 标签增量解析器
///
/// 背景：Kiro 在开启 thinking 模式时，思考内容会混入 `assistantResponseEvent` 的 `content`
/// 字段中，以 `<thinking>...</thinking>` 标签包裹。
///
/// 本解析器用于把标签内文本拆分为 thinking 段落，从而让上层可以回调到"思考块"里。
///
/// 设计目标：
/// - 增量解析：标签可能跨多个 chunk/事件分段
/// - 低误判：仅当响应以（可选空白后）`<thinking>` 开头时才启用解析
/// - 最小兼容：默认只解析第一个 thinking 标签块，避免后续内容里出现字面量标签导致误拆
#[derive(Debug, Clone)]
struct KiroThinkingTagParser {
    state: KiroThinkingTagParseState,
    buffer: String,
    in_thinking: bool,
    closed_once: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KiroTextSegmentKind {
    Content,
    Thinking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KiroTextSegment {
    kind: KiroTextSegmentKind,
    text: String,
}

impl KiroThinkingTagParser {
    const OPEN_TAG: &'static str = "<thinking>";
    const CLOSE_TAG: &'static str = "</thinking>";

    fn new() -> Self {
        Self {
            state: KiroThinkingTagParseState::Armed,
            buffer: String::new(),
            in_thinking: false,
            closed_once: false,
        }
    }

    fn reset(&mut self) {
        self.state = KiroThinkingTagParseState::Armed;
        self.buffer.clear();
        self.in_thinking = false;
        self.closed_once = false;
    }

    fn disable(&mut self) {
        self.state = KiroThinkingTagParseState::Disabled;
        self.in_thinking = false;
        self.closed_once = false;
        self.buffer.clear();
    }

    fn flush_finish(&mut self) -> Vec<KiroTextSegment> {
        match self.state {
            KiroThinkingTagParseState::Armed | KiroThinkingTagParseState::Disabled => {
                let text = std::mem::take(&mut self.buffer);
                if text.is_empty() {
                    vec![]
                } else {
                    vec![KiroTextSegment {
                        kind: KiroTextSegmentKind::Content,
                        text,
                    }]
                }
            }
            KiroThinkingTagParseState::Enabled => self.parse_enabled(false),
        }
    }

    fn push_and_parse(&mut self, incoming: &str) -> Vec<KiroTextSegment> {
        if incoming.is_empty() && self.buffer.is_empty() {
            return vec![];
        }

        match self.state {
            KiroThinkingTagParseState::Disabled => {
                if !self.buffer.is_empty() {
                    let mut out = vec![KiroTextSegment {
                        kind: KiroTextSegmentKind::Content,
                        text: std::mem::take(&mut self.buffer),
                    }];
                    if !incoming.is_empty() {
                        out.push(KiroTextSegment {
                            kind: KiroTextSegmentKind::Content,
                            text: incoming.to_string(),
                        });
                    }
                    out
                } else if incoming.is_empty() {
                    vec![]
                } else {
                    vec![KiroTextSegment {
                        kind: KiroTextSegmentKind::Content,
                        text: incoming.to_string(),
                    }]
                }
            }
            KiroThinkingTagParseState::Armed => {
                self.buffer.push_str(incoming);
                self.try_arm_and_parse()
            }
            KiroThinkingTagParseState::Enabled => {
                self.buffer.push_str(incoming);
                self.parse_enabled(true)
            }
        }
    }

    fn try_arm_and_parse(&mut self) -> Vec<KiroTextSegment> {
        // 仅当响应以（可选空白后）`<thinking>` 开头时，才启用标签解析。
        // 否则一律当作普通文本，避免误把用户/模型的字面量 `<thinking>` 当成思考块。
        let first_non_ws = self
            .buffer
            .char_indices()
            .find(|(_, ch)| !ch.is_whitespace())
            .map(|(i, _)| i);

        let Some(idx) = first_non_ws else {
            // 全是空白：直接输出，避免缓冲无限增长
            let text = std::mem::take(&mut self.buffer);
            if text.is_empty() {
                return vec![];
            }
            return vec![KiroTextSegment {
                kind: KiroTextSegmentKind::Content,
                text,
            }];
        };

        let tail = &self.buffer[idx..];
        if tail.starts_with(Self::OPEN_TAG) {
            self.state = KiroThinkingTagParseState::Enabled;
            self.parse_enabled(true)
        } else if Self::OPEN_TAG.starts_with(tail) {
            // `<thinking>` 标签可能被切成多个 chunk（例如 `<think` + `ing>`），继续等待
            vec![]
        } else {
            // 不是以 `<thinking>` 开头：禁用解析并全部按 content 输出
            self.state = KiroThinkingTagParseState::Disabled;
            let text = std::mem::take(&mut self.buffer);
            vec![KiroTextSegment {
                kind: KiroTextSegmentKind::Content,
                text,
            }]
        }
    }

    fn parse_enabled(&mut self, keep_partial_tag_suffix: bool) -> Vec<KiroTextSegment> {
        if self.closed_once {
            // 默认只解析第一个 `<thinking>...</thinking>`，后续全部当作普通内容
            let text = std::mem::take(&mut self.buffer);
            if text.is_empty() {
                return vec![];
            }
            return vec![KiroTextSegment {
                kind: KiroTextSegmentKind::Content,
                text,
            }];
        }

        let mut out = Vec::new();
        loop {
            if self.buffer.is_empty() {
                break;
            }

            if self.in_thinking {
                if let Some(pos) = self.buffer.find(Self::CLOSE_TAG) {
                    if pos > 0 {
                        out.push(KiroTextSegment {
                            kind: KiroTextSegmentKind::Thinking,
                            text: self.buffer[..pos].to_string(),
                        });
                    }
                    self.buffer.drain(..pos + Self::CLOSE_TAG.len());
                    self.in_thinking = false;
                    self.closed_once = true;
                    // 关闭后立即降级为普通文本，避免后续误判
                    self.state = KiroThinkingTagParseState::Disabled;
                    continue;
                }

                if keep_partial_tag_suffix {
                    let keep = longest_suffix_prefix(&self.buffer, Self::CLOSE_TAG);
                    if self.buffer.len() > keep {
                        let split_at = self.buffer.len() - keep;
                        out.push(KiroTextSegment {
                            kind: KiroTextSegmentKind::Thinking,
                            text: self.buffer[..split_at].to_string(),
                        });
                        self.buffer.drain(..split_at);
                    }
                } else {
                    out.push(KiroTextSegment {
                        kind: KiroTextSegmentKind::Thinking,
                        text: std::mem::take(&mut self.buffer),
                    });
                }
                break;
            } else if let Some(pos) = self.buffer.find(Self::OPEN_TAG) {
                if pos > 0 {
                    out.push(KiroTextSegment {
                        kind: KiroTextSegmentKind::Content,
                        text: self.buffer[..pos].to_string(),
                    });
                }
                self.buffer.drain(..pos + Self::OPEN_TAG.len());
                self.in_thinking = true;
                continue;
            } else {
                if keep_partial_tag_suffix {
                    let keep = longest_suffix_prefix(&self.buffer, Self::OPEN_TAG);
                    if self.buffer.len() > keep {
                        let split_at = self.buffer.len() - keep;
                        out.push(KiroTextSegment {
                            kind: KiroTextSegmentKind::Content,
                            text: self.buffer[..split_at].to_string(),
                        });
                        self.buffer.drain(..split_at);
                    }
                } else {
                    out.push(KiroTextSegment {
                        kind: KiroTextSegmentKind::Content,
                        text: std::mem::take(&mut self.buffer),
                    });
                }
                break;
            }
        }

        out.into_iter().filter(|s| !s.text.is_empty()).collect()
    }
}

fn longest_suffix_prefix(haystack: &str, needle: &str) -> usize {
    if haystack.is_empty() || needle.is_empty() {
        return 0;
    }
    let max = haystack.len().min(needle.len().saturating_sub(1));
    if max == 0 {
        return 0;
    }
    for len in (1..=max).rev() {
        let start = haystack.len() - len;
        if haystack.is_char_boundary(start) && needle.starts_with(&haystack[start..]) {
            return len;
        }
    }
    0
}

/// 流式格式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamFormat {
    /// AWS Event Stream (Kiro/CodeWhisperer)
    AwsEventStream,
    /// Anthropic SSE 格式
    AnthropicSse,
    /// OpenAI SSE 格式
    OpenAiSse,
}

/// 转换器状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConverterState {
    /// 初始状态
    Idle,
    /// 正在转换
    Converting,
    /// 已完成
    Completed,
    /// 错误状态
    Error(String),
}

impl Default for ConverterState {
    fn default() -> Self {
        Self::Idle
    }
}

/// 工具调用累积器
///
/// 用于跟踪正在进行的工具调用，累积部分 JSON 输入
#[derive(Debug, Clone, Default)]
struct ToolCallAccumulator {
    /// 工具调用 ID
    id: String,
    /// 工具名称
    name: String,
    /// 累积的输入 JSON（部分）
    input: String,
    /// 是否已发送开始事件
    started: bool,
    /// 内容块索引（用于 Anthropic 格式）
    index: u32,
}

/// 部分 JSON 累积器
///
/// 用于处理工具调用参数中的部分 JSON
/// 对应需求 3.5
#[derive(Debug, Clone, Default)]
pub struct PartialJsonAccumulator {
    /// 累积的 JSON 字符串
    buffer: String,
    /// 括号深度（用于检测 JSON 完整性）
    brace_depth: i32,
    /// 是否在字符串内
    in_string: bool,
    /// 是否转义下一个字符
    escape_next: bool,
}

impl PartialJsonAccumulator {
    /// 创建新的累积器
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加部分 JSON
    ///
    /// 返回 true 如果 JSON 已完整
    pub fn append(&mut self, partial: &str) -> bool {
        for ch in partial.chars() {
            self.buffer.push(ch);

            if self.escape_next {
                self.escape_next = false;
                continue;
            }

            match ch {
                '\\' if self.in_string => self.escape_next = true,
                '"' => self.in_string = !self.in_string,
                '{' | '[' if !self.in_string => self.brace_depth += 1,
                '}' | ']' if !self.in_string => self.brace_depth -= 1,
                _ => {}
            }
        }

        self.is_complete()
    }

    /// 检查 JSON 是否完整
    pub fn is_complete(&self) -> bool {
        !self.buffer.is_empty() && self.brace_depth == 0 && !self.in_string
    }

    /// 获取累积的 JSON
    pub fn get_json(&self) -> &str {
        &self.buffer
    }

    /// 重置累积器
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.brace_depth = 0;
        self.in_string = false;
        self.escape_next = false;
    }

    /// 获取缓冲区长度
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// 检查缓冲区是否为空
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// 流式格式转换器
///
/// 支持在不同流式格式之间转换。
#[derive(Debug)]
pub struct StreamConverter {
    /// 源格式
    source_format: StreamFormat,
    /// 目标格式
    target_format: StreamFormat,
    /// AWS 解析器（如果源是 AWS Event Stream）
    aws_parser: Option<AwsEventStreamParser>,
    /// 状态
    state: ConverterState,
    /// 响应 ID
    response_id: String,
    /// 模型名称
    model: String,
    /// 工具调用累积器
    tool_accumulators: HashMap<String, ToolCallAccumulator>,
    /// 下一个内容块索引（用于 Anthropic 格式）
    next_content_block_index: u32,
    /// 当前打开的文本内容块索引（Anthropic）
    text_block_index: Option<u32>,
    /// 当前打开的 thinking 内容块索引（Anthropic）
    thinking_block_index: Option<u32>,
    /// 是否已发送 message_start（用于 Anthropic 格式）
    message_started: bool,
    /// 累积的内容（用于重建完整响应）
    accumulated_content: String,
    /// 累积的 thinking 内容（用于 token 估算）
    accumulated_thinking: String,
    /// 累积的工具调用参数长度（用于 token 估算）
    accumulated_tool_args_len: usize,
    /// 上下文使用百分比（用于估算 input tokens）
    context_usage_percentage: f64,
    /// Kiro `<thinking>` 标签解析器（用于 assistantResponseEvent 混入 thinking 的场景）
    kiro_thinking_tag_parser: KiroThinkingTagParser,
}

impl StreamConverter {
    /// 创建新的转换器
    pub fn new(source: StreamFormat, target: StreamFormat) -> Self {
        let aws_parser = if source == StreamFormat::AwsEventStream {
            Some(AwsEventStreamParser::new())
        } else {
            None
        };

        Self {
            source_format: source,
            target_format: target,
            aws_parser,
            state: ConverterState::Idle,
            response_id: format!("chatcmpl-{}", Uuid::new_v4()),
            model: String::new(),
            tool_accumulators: HashMap::new(),
            next_content_block_index: 0,
            text_block_index: None,
            thinking_block_index: None,
            message_started: false,
            accumulated_content: String::new(),
            accumulated_thinking: String::new(),
            accumulated_tool_args_len: 0,
            context_usage_percentage: 0.0,
            kiro_thinking_tag_parser: KiroThinkingTagParser::new(),
        }
    }

    /// 创建带模型名称的转换器
    pub fn with_model(source: StreamFormat, target: StreamFormat, model: &str) -> Self {
        let mut converter = Self::new(source, target);
        converter.model = model.to_string();
        converter
    }

    /// 获取当前状态
    pub fn state(&self) -> &ConverterState {
        &self.state
    }

    /// 获取响应 ID
    pub fn response_id(&self) -> &str {
        &self.response_id
    }

    /// 获取累积的内容
    pub fn accumulated_content(&self) -> &str {
        &self.accumulated_content
    }

    /// 估算 output tokens
    ///
    /// 基于累积的响应内容长度估算 Token 数量（约 4 字符 = 1 token）：
    /// - content: 文本内容
    /// - thinking: 思考内容
    /// - tool_args: 工具调用参数
    pub fn estimate_output_tokens(&self) -> u32 {
        let content_tokens = (self.accumulated_content.len() / 3) as u32;
        let thinking_tokens = (self.accumulated_thinking.len() / 3) as u32;
        let tool_args_tokens = (self.accumulated_tool_args_len / 3) as u32;
        content_tokens + thinking_tokens + tool_args_tokens
    }

    /// 估算 input tokens
    ///
    /// 基于 context_usage_percentage 估算 input tokens
    /// 假设 100% = 200k tokens (Claude 的上下文窗口)
    pub fn estimate_input_tokens(&self) -> u32 {
        (((self.context_usage_percentage / 100.0) * 200000.0)/0.59) as u32
    }

    /// 估算 Token 使用量
    ///
    /// # 返回
    /// (input_tokens, output_tokens) 元组
    pub fn estimate_tokens(&self) -> (u32, u32) {
        (self.estimate_input_tokens(), self.estimate_output_tokens())
    }

    /// 重置转换器
    pub fn reset(&mut self) {
        if let Some(parser) = &mut self.aws_parser {
            parser.reset();
        }
        self.state = ConverterState::Idle;
        self.response_id = format!("chatcmpl-{}", Uuid::new_v4());
        self.tool_accumulators.clear();
        self.next_content_block_index = 0;
        self.text_block_index = None;
        self.thinking_block_index = None;
        self.message_started = false;
        self.accumulated_content.clear();
        self.accumulated_thinking.clear();
        self.accumulated_tool_args_len = 0;
        self.context_usage_percentage = 0.0;
        self.kiro_thinking_tag_parser.reset();
    }

    // 转换 chunk
    // 将源格式的 chunk 转换为目标格式的 SSE 事件列表。
    //
    // # 参数
    // * `chunk` - 源格式的字节数据
    //
    // # 返回
    // 目标格式的 SSE 事件字符串列表
    pub fn convert(&mut self, chunk: &[u8]) -> Vec<String> {
        if self.state == ConverterState::Idle {
            self.state = ConverterState::Converting;
        }

        match self.source_format {
            StreamFormat::AwsEventStream => self.convert_aws_event_stream(chunk),
            StreamFormat::AnthropicSse => self.convert_anthropic_sse(chunk),
            StreamFormat::OpenAiSse => self.convert_openai_sse(chunk),
        }
    }

    /// 完成转换
    ///
    /// 处理剩余数据并生成结束事件。
    pub fn finish(&mut self) -> Vec<String> {
        let mut events = Vec::new();

        // 处理 AWS 解析器中的剩余数据
        if let Some(parser) = &mut self.aws_parser {
            let aws_events = parser.finish();
            for aws_event in aws_events {
                events.extend(self.convert_aws_event(&aws_event));
            }
        }

        // Kiro `<thinking>` 标签解析可能会缓存部分内容（例如标签被切块）
        // 在流结束时尽量把剩余缓冲区输出，避免内容丢失。
        events.extend(self.flush_kiro_thinking_tag_buffer());

        // 生成结束事件
        events.extend(self.generate_end_events());

        self.state = ConverterState::Completed;
        events
    }

    /// 转换 AWS Event Stream
    fn convert_aws_event_stream(&mut self, chunk: &[u8]) -> Vec<String> {
        let parser = self.aws_parser.as_mut().expect("AWS parser should exist");
        let aws_events = parser.process(chunk);

        let mut sse_events = Vec::new();
        for aws_event in aws_events {
            sse_events.extend(self.convert_aws_event(&aws_event));
        }
        sse_events
    }

    /// 转换单个 AWS 事件
    fn convert_aws_event(&mut self, event: &AwsEvent) -> Vec<String> {
        match self.target_format {
            StreamFormat::AnthropicSse => self.aws_to_anthropic(event),
            StreamFormat::OpenAiSse => self.aws_to_openai(event),
            StreamFormat::AwsEventStream => {
                // 源和目标相同，直接序列化
                if let Some(json) = crate::streaming::aws_parser::serialize_event(event) {
                    vec![json]
                } else {
                    vec![]
                }
            }
        }
    }

    /// AWS Event Stream 到 Anthropic SSE 转换
    ///
    /// 对应需求 3.1
    fn aws_to_anthropic(&mut self, event: &AwsEvent) -> Vec<String> {
        let mut sse_events = Vec::new();

        // 确保发送 message_start
        if !self.message_started {
            sse_events.push(self.create_anthropic_message_start());
            self.message_started = true;
        }

        match event {
            AwsEvent::Content { text } => {
                let segments = self.parse_kiro_thinking_tags_if_needed(text);
                for seg in segments {
                    match seg.kind {
                        KiroTextSegmentKind::Content => {
                            self.emit_aws_content_as_anthropic(&mut sse_events, &seg.text);
                        }
                        KiroTextSegmentKind::Thinking => {
                            self.emit_aws_thinking_as_anthropic(&mut sse_events, &seg.text);
                        }
                    }
                }
            }
            AwsEvent::ToolUseStart { id, name } => {
                // 工具调用前先清空可能缓存的 `<thinking>` 标签内容，避免顺序错乱
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());

                // 工具调用前，关闭可能打开的 thinking/text 内容块
                self.close_anthropic_thinking_block_if_open(&mut sse_events);
                if let Some(text_idx) = self.text_block_index.take() {
                    sse_events.push(self.create_anthropic_content_block_stop(text_idx));
                }

                let index = self.next_content_block_index;
                self.next_content_block_index += 1;

                // 创建工具调用累积器
                self.tool_accumulators.insert(
                    id.clone(),
                    ToolCallAccumulator {
                        id: id.clone(),
                        name: name.clone(),
                        input: String::new(),
                        started: true,
                        index,
                    },
                );

                // 发送 content_block_start (tool_use)
                sse_events.push(self.create_anthropic_content_block_start_tool(index, id, name));
            }
            AwsEvent::ToolUseInput { id, input } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                if let Some(acc) = self.tool_accumulators.get_mut(id) {
                    acc.input.push_str(input);
                    // 累积工具调用参数长度（用于 token 估算）
                    self.accumulated_tool_args_len += input.len();
                }
                // 发送 input_json_delta
                if let Some(acc) = self.tool_accumulators.get(id) {
                    sse_events.push(self.create_anthropic_input_json_delta(acc.index, input));
                }
            }
            AwsEvent::ToolUseStop { id } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                if let Some(acc) = self.tool_accumulators.remove(id) {
                    // 发送 content_block_stop
                    sse_events.push(self.create_anthropic_content_block_stop(acc.index));
                }
            }
            AwsEvent::Stop => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                // 关闭所有未关闭的内容块
                if self.next_content_block_index > 0 && !self.accumulated_content.is_empty() {
                    sse_events.push(self.create_anthropic_content_block_stop(0));
                }
                // message_delta 和 message_stop 在 finish() 中处理
            }
            AwsEvent::Usage {
                credits,
                context_percentage,
            } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                // 保存 context_usage_percentage 用于估算 input tokens
                self.context_usage_percentage = *context_percentage;
                let _ = credits;
            }
            AwsEvent::FollowupPrompt { .. } | AwsEvent::ParseError { .. } => {
                // 忽略这些事件
            }
        }

        sse_events
    }

    /// AWS Event Stream 到 OpenAI SSE 转换
    ///
    /// 对应需求 3.2
    fn aws_to_openai(&mut self, event: &AwsEvent) -> Vec<String> {
        let mut sse_events = Vec::new();

        match event {
            AwsEvent::Content { text } => {
                let segments = self.parse_kiro_thinking_tags_if_needed(text);
                for seg in segments {
                    match seg.kind {
                        KiroTextSegmentKind::Content => {
                            self.accumulated_content.push_str(&seg.text);
                            sse_events.push(self.create_openai_content_chunk(&seg.text, false));
                        }
                        KiroTextSegmentKind::Thinking => {
                            // 累积 thinking 内容（用于 token 估算）
                            self.accumulated_thinking.push_str(&seg.text);
                            // OpenAI 协议扩展：使用 reasoning_content 输出思考内容
                            sse_events.push(self.create_openai_reasoning_chunk(&seg.text, false));
                        }
                    }
                }
            }
            AwsEvent::ToolUseStart { id, name } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                let index = self.tool_accumulators.len() as u32;
                self.tool_accumulators.insert(
                    id.clone(),
                    ToolCallAccumulator {
                        id: id.clone(),
                        name: name.clone(),
                        input: String::new(),
                        started: true,
                        index,
                    },
                );
                // 发送工具调用开始 chunk
                sse_events.push(self.create_openai_tool_call_chunk(index, id, name, "", true));
            }
            AwsEvent::ToolUseInput { id, input } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                let (index, tool_id, tool_name) =
                    if let Some(acc) = self.tool_accumulators.get_mut(id) {
                        acc.input.push_str(input);
                        // 累积工具调用参数长度（用于 token 估算）
                        self.accumulated_tool_args_len += input.len();
                        (acc.index, acc.id.clone(), acc.name.clone())
                    } else {
                        return sse_events;
                    };
                // 发送工具调用参数增量
                sse_events.push(
                    self.create_openai_tool_call_chunk(index, &tool_id, &tool_name, input, false),
                );
            }
            AwsEvent::ToolUseStop { id } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                // OpenAI 格式不需要显式的工具调用结束事件
                self.tool_accumulators.remove(id);
            }
            AwsEvent::Stop => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                // 结束事件在 finish() 中处理
            }
            AwsEvent::Usage {
                context_percentage, ..
            } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                // 保存 context_usage_percentage 用于估算 input tokens
                self.context_usage_percentage = *context_percentage;
            }
            AwsEvent::FollowupPrompt { .. } | AwsEvent::ParseError { .. } => {
                sse_events.extend(self.flush_kiro_thinking_tag_buffer());
                // 忽略这些事件
            }
        }

        sse_events
    }

    fn parse_kiro_thinking_tags_if_needed(&mut self, text: &str) -> Vec<KiroTextSegment> {
        self.kiro_thinking_tag_parser.push_and_parse(text)
    }

    fn flush_kiro_thinking_tag_buffer(&mut self) -> Vec<String> {
        let segments = self.kiro_thinking_tag_parser.flush_finish();
        if segments.is_empty() {
            return vec![];
        }

        let mut out = Vec::new();
        match self.target_format {
            StreamFormat::OpenAiSse => {
                for seg in segments {
                    match seg.kind {
                        KiroTextSegmentKind::Content => {
                            self.accumulated_content.push_str(&seg.text);
                            out.push(self.create_openai_content_chunk(&seg.text, false));
                        }
                        KiroTextSegmentKind::Thinking => {
                            // 累积 thinking 内容（用于 token 估算）
                            self.accumulated_thinking.push_str(&seg.text);
                            out.push(self.create_openai_reasoning_chunk(&seg.text, false));
                        }
                    }
                }
            }
            StreamFormat::AnthropicSse => {
                // Anthropic SSE 需要 message_start 才能输出内容块
                if !self.message_started {
                    out.push(self.create_anthropic_message_start());
                    self.message_started = true;
                }

                for seg in segments {
                    match seg.kind {
                        KiroTextSegmentKind::Content => {
                            self.emit_aws_content_as_anthropic(&mut out, &seg.text);
                        }
                        KiroTextSegmentKind::Thinking => {
                            self.emit_aws_thinking_as_anthropic(&mut out, &seg.text);
                        }
                    }
                }
            }
            StreamFormat::AwsEventStream => {}
        }

        out
    }

    fn emit_aws_content_as_anthropic(&mut self, sse_events: &mut Vec<String>, text: &str) {
        if text.is_empty() {
            return;
        }

        // 累积内容（用于重建完整响应）
        self.accumulated_content.push_str(text);

        // 如果 thinking 块还开着，先关闭它再开始输出文本
        self.close_anthropic_thinking_block_if_open(sse_events);

        // 确保文本内容块已开始
        let text_idx = if let Some(idx) = self.text_block_index {
            idx
        } else {
            let idx = self.next_content_block_index;
            self.next_content_block_index += 1;
            self.text_block_index = Some(idx);
            sse_events.push(self.create_anthropic_content_block_start_text(idx));
            idx
        };

        // 发送 content_block_delta
        sse_events.push(self.create_anthropic_text_delta(text_idx, text));
    }

    fn emit_aws_thinking_as_anthropic(
        &mut self,
        sse_events: &mut Vec<String>,
        text: &str,
    ) {
        if text.is_empty() {
            return;
        }

        // 累积 thinking 内容（用于 token 估算）
        self.accumulated_thinking.push_str(text);

        // 如果已经开始输出文本，则不要再插入 thinking（避免块交错导致协议不一致）
        if self.text_block_index.is_some() {
            // 最小兼容：忽略后续 thinking
            return;
        }

        // 启动 thinking 内容块（若尚未启动）
        let idx = if let Some(idx) = self.thinking_block_index {
            idx
        } else {
            let idx = self.next_content_block_index;
            self.next_content_block_index += 1;
            self.thinking_block_index = Some(idx);
            sse_events.push(self.create_anthropic_content_block_start_thinking(idx));
            idx
        };

        // thinking delta
        sse_events.push(self.create_anthropic_thinking_delta(idx, text));
    }

    /// 转换 Anthropic SSE（直通或转换为 OpenAI）
    fn convert_anthropic_sse(&mut self, chunk: &[u8]) -> Vec<String> {
        // 解析 SSE 数据
        let data = match String::from_utf8(chunk.to_vec()) {
            Ok(s) => s,
            Err(_) => return vec![],
        };

        match self.target_format {
            StreamFormat::AnthropicSse => {
                // 直通
                vec![data]
            }
            StreamFormat::OpenAiSse => {
                // 转换为 OpenAI 格式
                self.anthropic_to_openai(&data)
            }
            StreamFormat::AwsEventStream => {
                // 不支持反向转换
                vec![]
            }
        }
    }

    /// Anthropic SSE 到 OpenAI SSE 转换
    ///
    /// 对应需求 3.3
    fn anthropic_to_openai(&mut self, data: &str) -> Vec<String> {
        let mut sse_events = Vec::new();

        // 解析 SSE 事件
        for line in data.lines() {
            if let Some(json_str) = line.strip_prefix("data: ") {
                if json_str == "[DONE]" {
                    sse_events.push("data: [DONE]\n\n".to_string());
                    continue;
                }

                if let Ok(event) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(event_type) = event.get("type").and_then(|t| t.as_str()) {
                        match event_type {
                            "content_block_delta" => {
                                if let Some(delta) = event.get("delta") {
                                    if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                        self.accumulated_content.push_str(text);
                                        sse_events
                                            .push(self.create_openai_content_chunk(text, false));
                                    } else if let Some(thinking) =
                                        delta.get("thinking").and_then(|t| t.as_str())
                                    {
                                        // Anthropic thinking_delta -> OpenAI reasoning_content
                                        sse_events.push(
                                            self.create_openai_reasoning_chunk(thinking, false),
                                        );
                                    } else if let Some(partial_json) =
                                        delta.get("partial_json").and_then(|t| t.as_str())
                                    {
                                        // 工具调用参数增量
                                        let index = event
                                            .get("index")
                                            .and_then(|i| i.as_u64())
                                            .unwrap_or(0)
                                            as u32;
                                        let tool_info = self
                                            .tool_accumulators
                                            .values_mut()
                                            .find(|a| a.index == index)
                                            .map(|acc| {
                                                acc.input.push_str(partial_json);
                                                (acc.index, acc.id.clone(), acc.name.clone())
                                            });
                                        if let Some((idx, tool_id, tool_name)) = tool_info {
                                            sse_events.push(self.create_openai_tool_call_chunk(
                                                idx,
                                                &tool_id,
                                                &tool_name,
                                                partial_json,
                                                false,
                                            ));
                                        }
                                    }
                                }
                            }
                            "content_block_start" => {
                                if let Some(content_block) = event.get("content_block") {
                                    if content_block.get("type").and_then(|t| t.as_str())
                                        == Some("tool_use")
                                    {
                                        let id = content_block
                                            .get("id")
                                            .and_then(|i| i.as_str())
                                            .unwrap_or("");
                                        let name = content_block
                                            .get("name")
                                            .and_then(|n| n.as_str())
                                            .unwrap_or("");
                                        let index = event
                                            .get("index")
                                            .and_then(|i| i.as_u64())
                                            .unwrap_or(0)
                                            as u32;
                                        self.tool_accumulators.insert(
                                            id.to_string(),
                                            ToolCallAccumulator {
                                                id: id.to_string(),
                                                name: name.to_string(),
                                                input: String::new(),
                                                started: true,
                                                index,
                                            },
                                        );
                                        sse_events.push(self.create_openai_tool_call_chunk(
                                            index, id, name, "", true,
                                        ));
                                    }
                                }
                            }
                            "message_stop" => {
                                sse_events.push(self.create_openai_finish_chunk("stop"));
                                sse_events.push("data: [DONE]\n\n".to_string());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        sse_events
    }

    /// 转换 OpenAI SSE（直通）
    fn convert_openai_sse(&mut self, chunk: &[u8]) -> Vec<String> {
        match String::from_utf8(chunk.to_vec()) {
            Ok(s) => vec![s],
            Err(_) => vec![],
        }
    }

    /// 生成结束事件
    fn generate_end_events(&mut self) -> Vec<String> {
        match self.target_format {
            StreamFormat::AnthropicSse => {
                let mut events = Vec::new();
                // 关闭尚未结束的内容块（thinking / text）
                self.close_anthropic_thinking_block_if_open(&mut events);
                if let Some(text_idx) = self.text_block_index.take() {
                    events.push(self.create_anthropic_content_block_stop(text_idx));
                }
                // message_delta
                events.push(self.create_anthropic_message_delta());
                // message_stop
                events.push(self.create_anthropic_message_stop());
                events
            }
            StreamFormat::OpenAiSse => {
                let finish_reason = if self.tool_accumulators.is_empty() {
                    "stop"
                } else {
                    "tool_calls"
                };
                vec![
                    self.create_openai_finish_chunk(finish_reason),
                    "data: [DONE]\n\n".to_string(),
                ]
            }
            StreamFormat::AwsEventStream => {
                vec![]
            }
        }
    }

    // ========================================================================
    // Anthropic SSE 事件创建辅助方法
    // ========================================================================

    fn create_anthropic_message_start(&self) -> String {
        let input_tokens = self.estimate_input_tokens();
        let event = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": self.response_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0
                }
            }
        });
        format!("event: message_start\ndata: {}\n\n", event)
    }

    fn create_anthropic_content_block_start_text(&self, index: u32) -> String {
        let event = serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "text",
                "text": ""
            }
        });
        format!("event: content_block_start\ndata: {}\n\n", event)
    }

    fn create_anthropic_content_block_start_tool(
        &self,
        index: u32,
        id: &str,
        name: &str,
    ) -> String {
        let event = serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": {}
            }
        });
        format!("event: content_block_start\ndata: {}\n\n", event)
    }

    fn create_anthropic_text_delta(&self, index: u32, text: &str) -> String {
        let event = serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "text_delta",
                "text": text
            }
        });
        format!("event: content_block_delta\ndata: {}\n\n", event)
    }

    fn create_anthropic_content_block_start_thinking(
        &self,
        index: u32,
    ) -> String {
        let event = serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "thinking",
                "thinking": ""
            }
        });
        format!("event: content_block_start\ndata: {}\n\n", event)
    }

    fn create_anthropic_thinking_delta(&self, index: u32, thinking: &str) -> String {
        let event = serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "thinking_delta",
                "thinking": thinking
            }
        });
        format!("event: content_block_delta\ndata: {}\n\n", event)
    }

    fn create_anthropic_input_json_delta(&self, index: u32, partial_json: &str) -> String {
        let event = serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": partial_json
            }
        });
        format!("event: content_block_delta\ndata: {}\n\n", event)
    }

    fn create_anthropic_content_block_stop(&self, index: u32) -> String {
        let event = serde_json::json!({
            "type": "content_block_stop",
            "index": index
        });
        format!("event: content_block_stop\ndata: {}\n\n", event)
    }

    fn close_anthropic_thinking_block_if_open(&mut self, events: &mut Vec<String>) {
        let Some(thinking_idx) = self.thinking_block_index.take() else {
            return;
        };

        events.push(self.create_anthropic_content_block_stop(thinking_idx));
    }

    fn create_anthropic_message_delta(&self) -> String {
        let (input_tokens, output_tokens) = self.estimate_tokens();
        let event = serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": null
            },
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        });
        format!("event: message_delta\ndata: {}\n\n", event)
    }

    fn create_anthropic_message_stop(&self) -> String {
        let event = serde_json::json!({
            "type": "message_stop"
        });
        format!("event: message_stop\ndata: {}\n\n", event)
    }

    // ========================================================================
    // OpenAI SSE 事件创建辅助方法
    // ========================================================================

    fn get_created_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn create_openai_content_chunk(&self, content: &str, is_first: bool) -> String {
        let chunk = serde_json::json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.get_created_timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": if is_first { Some("assistant") } else { None::<&str> },
                    "content": content
                },
                "finish_reason": null
            }]
        });
        format!("data: {}\n\n", chunk)
    }

    fn create_openai_reasoning_chunk(&self, reasoning: &str, is_first: bool) -> String {
        let chunk = serde_json::json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.get_created_timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": if is_first { Some("assistant") } else { None::<&str> },
                    "reasoning_content": reasoning
                },
                "finish_reason": null
            }]
        });
        format!("data: {}\n\n", chunk)
    }

    fn create_openai_tool_call_chunk(
        &self,
        index: u32,
        id: &str,
        name: &str,
        arguments: &str,
        is_first: bool,
    ) -> String {
        let tool_call = if is_first {
            serde_json::json!({
                "index": index,
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            })
        } else {
            serde_json::json!({
                "index": index,
                "function": {
                    "arguments": arguments
                }
            })
        };

        let chunk = serde_json::json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.get_created_timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [tool_call]
                },
                "finish_reason": null
            }]
        });
        format!("data: {}\n\n", chunk)
    }

    fn create_openai_finish_chunk(&self, finish_reason: &str) -> String {
        let chunk = serde_json::json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.get_created_timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        format!("data: {}\n\n", chunk)
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 从 SSE 事件列表中提取所有文本内容
pub fn extract_content_from_sse(events: &[String], format: StreamFormat) -> String {
    let mut content = String::new();

    for event in events {
        match format {
            StreamFormat::OpenAiSse => {
                for line in event.lines() {
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if json_str == "[DONE]" {
                            continue;
                        }
                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(json_str) {
                            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                                for choice in choices {
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(text) =
                                            delta.get("content").and_then(|c| c.as_str())
                                        {
                                            content.push_str(text);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            StreamFormat::AnthropicSse => {
                for line in event.lines() {
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if let Ok(evt) = serde_json::from_str::<serde_json::Value>(json_str) {
                            if evt.get("type").and_then(|t| t.as_str())
                                == Some("content_block_delta")
                            {
                                if let Some(delta) = evt.get("delta") {
                                    if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                        content.push_str(text);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            StreamFormat::AwsEventStream => {
                // AWS Event Stream 不是 SSE 格式
            }
        }
    }

    content
}

/// 从 SSE 事件列表中提取所有工具调用
pub fn extract_tool_calls_from_sse(
    events: &[String],
    format: StreamFormat,
) -> Vec<(String, String, String)> {
    let mut tool_calls: HashMap<String, (String, String)> = HashMap::new();

    for event in events {
        match format {
            StreamFormat::OpenAiSse => {
                for line in event.lines() {
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if json_str == "[DONE]" {
                            continue;
                        }
                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(json_str) {
                            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                                for choice in choices {
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(tcs) =
                                            delta.get("tool_calls").and_then(|t| t.as_array())
                                        {
                                            for tc in tcs {
                                                let id = tc
                                                    .get("id")
                                                    .and_then(|i| i.as_str())
                                                    .unwrap_or("");
                                                let name = tc
                                                    .get("function")
                                                    .and_then(|f| f.get("name"))
                                                    .and_then(|n| n.as_str())
                                                    .unwrap_or("");
                                                let args = tc
                                                    .get("function")
                                                    .and_then(|f| f.get("arguments"))
                                                    .and_then(|a| a.as_str())
                                                    .unwrap_or("");

                                                if !id.is_empty() {
                                                    tool_calls
                                                        .entry(id.to_string())
                                                        .or_insert((String::new(), String::new()))
                                                        .0 = name.to_string();
                                                }
                                                if !args.is_empty() {
                                                    if let Some(entry) =
                                                        tool_calls.values_mut().last()
                                                    {
                                                        entry.1.push_str(args);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            StreamFormat::AnthropicSse | StreamFormat::AwsEventStream => {
                // 简化处理
            }
        }
    }

    tool_calls
        .into_iter()
        .map(|(id, (name, args))| (id, name, args))
        .collect()
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_new() {
        let converter = StreamConverter::new(StreamFormat::AwsEventStream, StreamFormat::OpenAiSse);
        assert_eq!(converter.state(), &ConverterState::Idle);
        assert!(converter.aws_parser.is_some());
    }

    #[test]
    fn test_converter_with_model() {
        let converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "claude-3-opus",
        );
        assert_eq!(converter.model, "claude-3-opus");
    }

    #[test]
    fn test_converter_reset() {
        let mut converter =
            StreamConverter::new(StreamFormat::AwsEventStream, StreamFormat::OpenAiSse);
        converter.convert(b"{\"content\":\"test\"}");
        assert_eq!(converter.state(), &ConverterState::Converting);

        converter.reset();
        assert_eq!(converter.state(), &ConverterState::Idle);
        assert!(converter.accumulated_content.is_empty());
    }

    #[test]
    fn test_aws_to_openai_content() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let events = converter.convert(b"{\"content\":\"Hello, world!\"}");
        assert!(!events.is_empty());

        let content = extract_content_from_sse(&events, StreamFormat::OpenAiSse);
        assert_eq!(content, "Hello, world!");
        assert_eq!(converter.accumulated_content(), "Hello, world!");
    }

    #[test]
    fn test_aws_to_openai_multiple_content() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let events1 = converter.convert(b"{\"content\":\"Hello\"}");
        let events2 = converter.convert(b"{\"content\":\", world!\"}");

        let all_events: Vec<_> = events1.into_iter().chain(events2).collect();
        let content = extract_content_from_sse(&all_events, StreamFormat::OpenAiSse);
        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn test_aws_to_openai_thinking_tag_in_content_as_reasoning_content() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let events = converter.convert(b"{\"content\":\"<thinking>think</thinking>Hello\"}");

        assert!(
            events.iter().any(|e| e.contains("\"reasoning_content\"")),
            "当 content 中包含 <thinking>...</thinking> 时，应回调 reasoning_content"
        );

        let content = extract_content_from_sse(&events, StreamFormat::OpenAiSse);
        assert_eq!(content, "Hello");
        assert_eq!(converter.accumulated_content(), "Hello");
    }

    #[test]
    fn test_aws_to_openai_thinking_tag_split_across_events() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        let events1 = converter.convert(b"{\"content\":\"<think\"}");
        let events2 = converter.convert(b"{\"content\":\"ing>t</thinking>Hi\"}");

        let all_events: Vec<_> = events1.into_iter().chain(events2).collect();

        assert!(
            all_events
                .iter()
                .any(|e| e.contains("\"reasoning_content\"")),
            "标签被切块时仍应能解析出 reasoning_content"
        );

        let content = extract_content_from_sse(&all_events, StreamFormat::OpenAiSse);
        assert_eq!(content, "Hi");
    }

    #[test]
    fn test_aws_to_openai_tool_call() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 工具调用开始
        let events1 = converter.convert(b"{\"toolUseId\":\"tool_1\",\"name\":\"read_file\"}");
        assert!(!events1.is_empty());

        // 工具调用输入
        let events2 = converter
            .convert(b"{\"toolUseId\":\"tool_1\",\"input\":\"{\\\"path\\\":\\\"/tmp\\\"}\"}");
        assert!(!events2.is_empty());

        // 工具调用结束
        let events3 = converter.convert(b"{\"toolUseId\":\"tool_1\",\"stop\":true}");

        let all_events: Vec<_> = events1.into_iter().chain(events2).chain(events3).collect();

        // 验证工具调用存在
        let has_tool_call = all_events.iter().any(|e| e.contains("tool_calls"));
        assert!(has_tool_call);
    }

    #[test]
    fn test_aws_to_anthropic_content() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::AnthropicSse,
            "test-model",
        );

        let events = converter.convert(b"{\"content\":\"Hello!\"}");

        // 应该有 message_start, content_block_start, content_block_delta
        assert!(events.iter().any(|e| e.contains("message_start")));
        assert!(events.iter().any(|e| e.contains("content_block_start")));
        assert!(events.iter().any(|e| e.contains("content_block_delta")));

        let content = extract_content_from_sse(&events, StreamFormat::AnthropicSse);
        assert_eq!(content, "Hello!");
    }

    #[test]
    fn test_aws_to_anthropic_thinking_tag_in_content_as_thinking_block() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::AnthropicSse,
            "test-model",
        );

        let events = converter.convert(b"{\"content\":\"<thinking>think</thinking>Hello\"}");

        assert!(
            events.iter().any(|e| e.contains("thinking_delta")),
            "当 content 中包含 <thinking>...</thinking> 时，应回调 thinking_delta"
        );

        let content = extract_content_from_sse(&events, StreamFormat::AnthropicSse);
        assert_eq!(content, "Hello");
    }

    #[test]
    fn test_aws_to_anthropic_tool_call() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::AnthropicSse,
            "test-model",
        );

        // 工具调用开始
        let events1 = converter.convert(b"{\"toolUseId\":\"tool_1\",\"name\":\"test_tool\"}");
        assert!(events1.iter().any(|e| e.contains("tool_use")));

        // 工具调用输入
        let events2 = converter
            .convert(b"{\"toolUseId\":\"tool_1\",\"input\":\"{\\\"key\\\":\\\"value\\\"}\"}");
        assert!(events2.iter().any(|e| e.contains("input_json_delta")));

        // 工具调用结束
        let events3 = converter.convert(b"{\"toolUseId\":\"tool_1\",\"stop\":true}");
        assert!(events3.iter().any(|e| e.contains("content_block_stop")));
    }

    #[test]
    fn test_converter_finish() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        converter.convert(b"{\"content\":\"test\"}");
        let finish_events = converter.finish();

        // 应该有结束事件
        assert!(finish_events.iter().any(|e| e.contains("finish_reason")));
        assert!(finish_events.iter().any(|e| e.contains("[DONE]")));
        assert_eq!(converter.state(), &ConverterState::Completed);
    }

    #[test]
    fn test_partial_json_accumulator() {
        let mut acc = PartialJsonAccumulator::new();

        // 追加部分 JSON
        assert!(!acc.append("{\"key\":"));
        assert!(!acc.is_complete());

        assert!(acc.append("\"value\"}"));
        assert!(acc.is_complete());
        assert_eq!(acc.get_json(), "{\"key\":\"value\"}");
    }

    #[test]
    fn test_partial_json_accumulator_nested() {
        let mut acc = PartialJsonAccumulator::new();

        assert!(!acc.append("{\"outer\":{\"inner\":"));
        assert!(!acc.append("\"value\""));
        assert!(acc.append("}}"));
        assert!(acc.is_complete());
    }

    #[test]
    fn test_partial_json_accumulator_with_string_braces() {
        let mut acc = PartialJsonAccumulator::new();

        // JSON 字符串中包含括号
        assert!(acc.append("{\"text\":\"hello {world}\"}"));
        assert!(acc.is_complete());
    }

    #[test]
    fn test_partial_json_accumulator_reset() {
        let mut acc = PartialJsonAccumulator::new();
        acc.append("{\"key\":\"value\"}");

        acc.reset();
        assert!(acc.is_empty());
        assert!(!acc.is_complete());
    }

    #[test]
    fn test_extract_content_from_openai_sse() {
        let events = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n".to_string(),
            "data: {\"choices\":[{\"delta\":{\"content\":\", world!\"}}]}\n\n".to_string(),
            "data: [DONE]\n\n".to_string(),
        ];

        let content = extract_content_from_sse(&events, StreamFormat::OpenAiSse);
        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn test_extract_content_from_anthropic_sse() {
        let events = vec![
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\", world!\"}}\n\n".to_string(),
        ];

        let content = extract_content_from_sse(&events, StreamFormat::AnthropicSse);
        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn test_incremental_conversion() {
        let mut converter = StreamConverter::with_model(
            StreamFormat::AwsEventStream,
            StreamFormat::OpenAiSse,
            "test-model",
        );

        // 分片发送 JSON
        let events1 = converter.convert(b"{\"content\":");
        assert!(events1.is_empty()); // 不完整的 JSON 不产生事件

        let events2 = converter.convert(b"\"test\"}");
        assert!(!events2.is_empty()); // 完整的 JSON 产生事件

        let content = extract_content_from_sse(&events2, StreamFormat::OpenAiSse);
        assert_eq!(content, "test");
    }

    #[test]
    fn test_converter_state_transitions() {
        let mut converter =
            StreamConverter::new(StreamFormat::AwsEventStream, StreamFormat::OpenAiSse);

        assert_eq!(converter.state(), &ConverterState::Idle);

        converter.convert(b"{\"content\":\"test\"}");
        assert_eq!(converter.state(), &ConverterState::Converting);

        converter.finish();
        assert_eq!(converter.state(), &ConverterState::Completed);

        converter.reset();
        assert_eq!(converter.state(), &ConverterState::Idle);
    }
}

// ============================================================================
// 属性测试（Property-Based Testing）
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::streaming::aws_parser::{
        extract_content, extract_tool_calls, serialize_event, AwsEvent,
    };
    use proptest::prelude::*;

    // ========================================================================
    // 生成器（Generators）
    // ========================================================================

    /// 生成有效的内容文本
    fn arb_content_text() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z0-9\\u4e00-\\u9fff .,!?\\-_]{1,100}")
            .unwrap()
            .prop_filter("非空字符串", |s| !s.is_empty())
    }

    /// 生成有效的工具 ID
    fn arb_tool_id() -> impl Strategy<Value = String> {
        prop::string::string_regex("tool_[a-zA-Z0-9]{4,12}").unwrap()
    }

    /// 生成有效的工具名称
    fn arb_tool_name() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-z_][a-z0-9_]{2,20}").unwrap()
    }

    /// 生成有效的工具输入（简单 JSON）
    fn arb_tool_input() -> impl Strategy<Value = String> {
        prop::string::string_regex(r#"\{"[a-z]+":"[a-zA-Z0-9]+"\}"#).unwrap()
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
    // Property 2: 流式格式转换内容保留
    //
    // *对于任意*流式响应内容，从 AWS Event Stream 转换为 Anthropic SSE 或
    // OpenAI SSE 后，最终重建的内容应该与原始内容一致。
    //
    // **验证: 需求 3.1, 3.2, 3.3**
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 2: AWS 到 OpenAI 转换保留内容
        ///
        /// **Feature: true-streaming-support, Property 2: 流式格式转换内容保留**
        /// **Validates: Requirements 3.2**
        #[test]
        fn prop_aws_to_openai_content_preservation(events in arb_content_sequence()) {
            // 计算原始内容
            let original_content = extract_content(&events);

            // 创建转换器
            let mut converter = StreamConverter::with_model(
                StreamFormat::AwsEventStream,
                StreamFormat::OpenAiSse,
                "test-model",
            );

            // 将所有事件序列化并转换
            let mut all_sse_events = Vec::new();
            for event in &events {
                if let Some(json) = serialize_event(event) {
                    let sse_events = converter.convert(json.as_bytes());
                    all_sse_events.extend(sse_events);
                }
            }
            all_sse_events.extend(converter.finish());

            // 从 SSE 事件中提取内容
            let converted_content = extract_content_from_sse(&all_sse_events, StreamFormat::OpenAiSse);

            // 验证内容一致
            prop_assert_eq!(
                original_content,
                converted_content,
                "AWS 到 OpenAI 转换应该保留内容"
            );
        }

        /// Property 2: AWS 到 Anthropic 转换保留内容
        ///
        /// **Feature: true-streaming-support, Property 2: 流式格式转换内容保留**
        /// **Validates: Requirements 3.1**
        #[test]
        fn prop_aws_to_anthropic_content_preservation(events in arb_content_sequence()) {
            // 计算原始内容
            let original_content = extract_content(&events);

            // 创建转换器
            let mut converter = StreamConverter::with_model(
                StreamFormat::AwsEventStream,
                StreamFormat::AnthropicSse,
                "test-model",
            );

            // 将所有事件序列化并转换
            let mut all_sse_events = Vec::new();
            for event in &events {
                if let Some(json) = serialize_event(event) {
                    let sse_events = converter.convert(json.as_bytes());
                    all_sse_events.extend(sse_events);
                }
            }
            all_sse_events.extend(converter.finish());

            // 从 SSE 事件中提取内容
            let converted_content = extract_content_from_sse(&all_sse_events, StreamFormat::AnthropicSse);

            // 验证内容一致
            prop_assert_eq!(
                original_content,
                converted_content,
                "AWS 到 Anthropic 转换应该保留内容"
            );
        }

        /// Property 2: 转换器累积内容与原始内容一致
        ///
        /// **Feature: true-streaming-support, Property 2: 流式格式转换内容保留**
        /// **Validates: Requirements 3.1, 3.2**
        #[test]
        fn prop_converter_accumulated_content_matches(
            events in arb_content_sequence(),
            target in prop_oneof![Just(StreamFormat::OpenAiSse), Just(StreamFormat::AnthropicSse)]
        ) {
            // 计算原始内容
            let original_content = extract_content(&events);

            // 创建转换器
            let mut converter = StreamConverter::with_model(
                StreamFormat::AwsEventStream,
                target,
                "test-model",
            );

            // 将所有事件序列化并转换
            for event in &events {
                if let Some(json) = serialize_event(event) {
                    converter.convert(json.as_bytes());
                }
            }
            converter.finish();

            // 验证累积内容一致
            prop_assert_eq!(
                original_content,
                converter.accumulated_content(),
                "转换器累积的内容应该与原始内容一致"
            );
        }
    }

    // ========================================================================
    // Property 4: 部分 JSON 处理正确性
    //
    // *对于任意*有效的 JSON 字符串，将其分割成任意多个部分后，
    // 解析器应该能正确累积并最终产生完整的 JSON。
    //
    // **验证: 需求 3.5**
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 4: 部分 JSON 累积正确性
        ///
        /// **Feature: true-streaming-support, Property 4: 部分 JSON 处理正确性**
        /// **Validates: Requirements 3.5**
        #[test]
        fn prop_partial_json_accumulation(
            key in "[a-z]{1,10}",
            value in "[a-zA-Z0-9]{1,20}",
            split_points in prop::collection::vec(1usize..50, 1..5)
        ) {
            let json = format!("{{\"{}\":\"{}\"}}", key, value);
            let bytes = json.as_bytes();

            let mut acc = PartialJsonAccumulator::new();

            // 根据分割点将 JSON 分成多个部分
            let mut last_pos = 0;
            for &split_point in &split_points {
                let split_pos = split_point.min(bytes.len());
                if split_pos > last_pos && split_pos < bytes.len() {
                    let part = std::str::from_utf8(&bytes[last_pos..split_pos]).unwrap_or("");
                    acc.append(part);
                    last_pos = split_pos;
                }
            }

            // 追加剩余部分
            if last_pos < bytes.len() {
                let remaining = std::str::from_utf8(&bytes[last_pos..]).unwrap_or("");
                acc.append(remaining);
            }

            // 验证累积的 JSON 与原始 JSON 一致
            prop_assert_eq!(
                acc.get_json(),
                &json,
                "累积的 JSON 应该与原始 JSON 一致"
            );

            // 验证 JSON 完整性
            prop_assert!(
                acc.is_complete(),
                "累积完成后 JSON 应该是完整的"
            );
        }

        /// Property 4: 嵌套 JSON 部分累积
        ///
        /// **Feature: true-streaming-support, Property 4: 部分 JSON 处理正确性**
        /// **Validates: Requirements 3.5**
        #[test]
        fn prop_nested_json_partial_accumulation(
            outer_key in "[a-z]{1,5}",
            inner_key in "[a-z]{1,5}",
            value in "[a-zA-Z0-9]{1,10}",
            chunk_size in 1usize..10
        ) {
            let json = format!("{{\"{}\":{{\"{}\":\"{}\"}}}}", outer_key, inner_key, value);
            let bytes = json.as_bytes();

            let mut acc = PartialJsonAccumulator::new();

            // 按固定大小分块
            for chunk in bytes.chunks(chunk_size) {
                let part = std::str::from_utf8(chunk).unwrap_or("");
                acc.append(part);
            }

            // 验证累积的 JSON 与原始 JSON 一致
            prop_assert_eq!(
                acc.get_json(),
                &json,
                "嵌套 JSON 累积应该与原始一致"
            );

            prop_assert!(
                acc.is_complete(),
                "嵌套 JSON 累积完成后应该是完整的"
            );
        }

        /// Property 4: 包含字符串括号的 JSON 部分累积
        ///
        /// **Feature: true-streaming-support, Property 4: 部分 JSON 处理正确性**
        /// **Validates: Requirements 3.5**
        #[test]
        fn prop_json_with_braces_in_string(
            key in "[a-z]{1,5}",
            prefix in "[a-zA-Z0-9]{0,5}",
            suffix in "[a-zA-Z0-9]{0,5}",
            chunk_size in 1usize..15
        ) {
            // 创建包含括号的字符串值
            let json = format!("{{\"{}\":\"{}{{}}[]{}\"}}", key, prefix, suffix);
            let bytes = json.as_bytes();

            let mut acc = PartialJsonAccumulator::new();

            // 按固定大小分块
            for chunk in bytes.chunks(chunk_size) {
                let part = std::str::from_utf8(chunk).unwrap_or("");
                acc.append(part);
            }

            // 验证累积的 JSON 与原始 JSON 一致
            prop_assert_eq!(
                acc.get_json(),
                &json,
                "包含括号的字符串 JSON 累积应该与原始一致"
            );

            prop_assert!(
                acc.is_complete(),
                "包含括号的字符串 JSON 累积完成后应该是完整的"
            );
        }

        /// Property 4: 重置后重新累积
        ///
        /// **Feature: true-streaming-support, Property 4: 部分 JSON 处理正确性**
        /// **Validates: Requirements 3.5**
        #[test]
        fn prop_reset_and_reaccumulate(
            key1 in "[a-z]{1,5}",
            value1 in "[a-zA-Z0-9]{1,10}",
            key2 in "[a-z]{1,5}",
            value2 in "[a-zA-Z0-9]{1,10}"
        ) {
            let json1 = format!("{{\"{}\":\"{}\"}}", key1, value1);
            let json2 = format!("{{\"{}\":\"{}\"}}", key2, value2);

            let mut acc = PartialJsonAccumulator::new();

            // 累积第一个 JSON
            acc.append(&json1);
            prop_assert_eq!(acc.get_json(), &json1);
            prop_assert!(acc.is_complete());

            // 重置
            acc.reset();
            prop_assert!(acc.is_empty());
            prop_assert!(!acc.is_complete());

            // 累积第二个 JSON
            acc.append(&json2);
            prop_assert_eq!(acc.get_json(), &json2);
            prop_assert!(acc.is_complete());
        }
    }
}
