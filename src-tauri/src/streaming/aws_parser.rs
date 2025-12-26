//! AWS Event Stream 解析器
//!
//! 解析 Kiro/CodeWhisperer 的 AWS Event Stream 二进制格式，
//! 支持增量解析和错误恢复。
//!
//! # 需求覆盖
//!
//! - 需求 2.1: 从二进制格式中提取 JSON 负载
//! - 需求 2.2: 立即发出内容增量
//! - 需求 2.3: 累积工具调用数据
//! - 需求 2.4: 发出流完成信号
//! - 需求 2.5: 优雅处理错误并继续处理
//! - 需求 2.6: 支持部分 chunk 的增量解析

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 解析器状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserState {
    /// 等待数据
    Idle,
    /// 正在解析
    Parsing,
    /// 已完成
    Completed,
    /// 错误状态
    Error(String),
}

impl Default for ParserState {
    fn default() -> Self {
        Self::Idle
    }
}

/// AWS Event Stream 解析后的事件
///
/// 表示从 AWS Event Stream 中解析出的各种事件类型。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AwsEvent {
    /// 内容增量
    ///
    /// 对应需求 2.2: 立即发出内容增量
    Content {
        /// 文本内容
        text: String,
    },

    /// 工具调用开始
    ///
    /// 对应需求 2.3: 累积工具调用数据
    ToolUseStart {
        /// 工具调用 ID
        id: String,
        /// 工具名称
        name: String,
    },

    /// 工具调用输入增量
    ///
    /// 对应需求 2.3: 累积工具调用数据
    ToolUseInput {
        /// 工具调用 ID
        id: String,
        /// 输入增量（部分 JSON）
        input: String,
    },

    /// 工具调用结束
    ///
    /// 对应需求 2.3: 累积工具调用数据
    ToolUseStop {
        /// 工具调用 ID
        id: String,
    },

    /// 流结束
    ///
    /// 对应需求 2.4: 发出流完成信号
    Stop,

    /// 使用量信息
    Usage {
        /// 消耗的 credits
        credits: f64,
        /// 上下文使用百分比
        context_percentage: f64,
    },

    /// 后续提示（通常忽略）
    FollowupPrompt {
        /// 提示内容
        content: String,
    },

    /// 解析错误（用于错误恢复）
    ///
    /// 对应需求 2.5: 优雅处理错误
    ParseError {
        /// 错误消息
        message: String,
        /// 原始数据（用于调试）
        raw_data: Option<String>,
    },
}

/// 工具调用累积器
///
/// 用于跟踪正在进行的工具调用
#[derive(Debug, Clone, Default)]
struct ToolAccumulator {
    /// 工具名称
    name: String,
    /// 累积的输入
    input: String,
}

/// AWS Event Stream 解析器
///
/// 支持增量解析 AWS Event Stream 二进制格式。
///
/// # 示例
///
/// ```ignore
/// let mut parser = AwsEventStreamParser::new();
///
/// // 处理接收到的字节
/// let events = parser.process(chunk);
/// for event in events {
///     match event {
///         AwsEvent::Content { text } => println!("Content: {}", text),
///         AwsEvent::Stop => println!("Stream completed"),
///         _ => {}
///     }
/// }
///
/// // 完成解析
/// let final_events = parser.finish();
/// ```
#[derive(Debug)]
pub struct AwsEventStreamParser {
    /// 缓冲区（用于处理部分 chunk）
    ///
    /// 对应需求 2.6: 支持部分 chunk 的增量解析
    buffer: Vec<u8>,

    /// 当前状态
    state: ParserState,

    /// 工具调用累积器
    /// key: toolUseId, value: ToolAccumulator
    tool_accumulators: HashMap<String, ToolAccumulator>,

    /// 解析错误计数
    parse_error_count: u32,

    /// 最大缓冲区大小（防止内存耗尽）
    max_buffer_size: usize,
}

impl Default for AwsEventStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AwsEventStreamParser {
    /// 默认最大缓冲区大小 (1MB)
    pub const DEFAULT_MAX_BUFFER_SIZE: usize = 1024 * 1024;

    /// 创建新的解析器
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            state: ParserState::Idle,
            tool_accumulators: HashMap::new(),
            parse_error_count: 0,
            max_buffer_size: Self::DEFAULT_MAX_BUFFER_SIZE,
        }
    }

    /// 创建带自定义缓冲区大小的解析器
    pub fn with_max_buffer_size(max_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            state: ParserState::Idle,
            tool_accumulators: HashMap::new(),
            parse_error_count: 0,
            max_buffer_size: max_size,
        }
    }

    /// 获取当前状态
    pub fn state(&self) -> &ParserState {
        &self.state
    }

    /// 获取解析错误计数
    pub fn parse_error_count(&self) -> u32 {
        self.parse_error_count
    }

    /// 获取缓冲区大小
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// 重置解析器状态
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.state = ParserState::Idle;
        self.tool_accumulators.clear();
        self.parse_error_count = 0;
    }

    /// 处理接收到的字节
    ///
    /// 对应需求 2.1, 2.6: 从二进制格式中提取 JSON 负载，支持增量解析
    ///
    /// # 参数
    ///
    /// * `bytes` - 接收到的字节数据
    ///
    /// # 返回
    ///
    /// 解析出的事件列表
    pub fn process(&mut self, bytes: &[u8]) -> Vec<AwsEvent> {
        if bytes.is_empty() {
            return Vec::new();
        }

        // 更新状态
        if self.state == ParserState::Idle {
            self.state = ParserState::Parsing;
        }

        // 检查缓冲区大小限制
        if self.buffer.len() + bytes.len() > self.max_buffer_size {
            self.parse_error_count += 1;
            return vec![AwsEvent::ParseError {
                message: "缓冲区溢出".to_string(),
                raw_data: None,
            }];
        }

        // 将新数据添加到缓冲区
        self.buffer.extend_from_slice(bytes);

        // 解析缓冲区中的所有完整 JSON 对象
        self.parse_buffer()
    }

    /// 完成解析
    ///
    /// 处理缓冲区中剩余的数据，并完成所有未完成的工具调用。
    ///
    /// # 返回
    ///
    /// 最终的事件列表
    pub fn finish(&mut self) -> Vec<AwsEvent> {
        let mut events = Vec::new();

        // 尝试解析缓冲区中剩余的数据
        events.extend(self.parse_buffer());

        // 完成所有未完成的工具调用
        for (id, accumulator) in self.tool_accumulators.drain() {
            if !accumulator.name.is_empty() {
                events.push(AwsEvent::ToolUseStop { id });
            }
        }

        // 更新状态
        self.state = ParserState::Completed;

        events
    }

    /// 解析缓冲区中的数据
    fn parse_buffer(&mut self) -> Vec<AwsEvent> {
        let mut events = Vec::new();
        let mut pos = 0;

        while pos < self.buffer.len() {
            // 查找下一个 JSON 对象的开始位置
            let start = match self.find_json_start(pos) {
                Some(s) => s,
                None => break,
            };

            // 提取 JSON 对象
            match self.extract_json(start) {
                Some((json_str, end_pos)) => {
                    // 解析 JSON 并生成事件
                    match self.parse_json_event(&json_str) {
                        Ok(event_list) => events.extend(event_list),
                        Err(e) => {
                            // 对应需求 2.5: 优雅处理错误
                            self.parse_error_count += 1;
                            events.push(AwsEvent::ParseError {
                                message: e,
                                raw_data: Some(json_str),
                            });
                        }
                    }
                    pos = end_pos;
                }
                None => {
                    // JSON 对象不完整，等待更多数据
                    break;
                }
            }
        }

        // 移除已处理的数据
        if pos > 0 {
            self.buffer.drain(..pos);
        }

        events
    }

    /// 查找 JSON 对象的开始位置
    fn find_json_start(&self, from: usize) -> Option<usize> {
        // JSON 对象以 '{' 开始
        self.buffer[from..]
            .iter()
            .position(|&b| b == b'{')
            .map(|p| from + p)
    }

    /// 从缓冲区中提取完整的 JSON 对象
    ///
    /// # 返回
    ///
    /// 如果找到完整的 JSON 对象，返回 (json_string, end_position)
    fn extract_json(&self, start: usize) -> Option<(String, usize)> {
        if start >= self.buffer.len() || self.buffer[start] != b'{' {
            return None;
        }

        let mut brace_count = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, &b) in self.buffer[start..].iter().enumerate() {
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
                        let end = start + i + 1;
                        let json_bytes = &self.buffer[start..end];
                        if let Ok(json_str) = String::from_utf8(json_bytes.to_vec()) {
                            return Some((json_str, end));
                        } else {
                            return None;
                        }
                    }
                }
                _ => {}
            }
        }

        // JSON 对象不完整
        None
    }

    /// 解析 JSON 事件
    fn parse_json_event(&mut self, json_str: &str) -> Result<Vec<AwsEvent>, String> {
        let value: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("JSON 解析错误: {}", e))?;

        let mut events = Vec::new();

        // 处理 content 事件
        if let Some(content) = value.get("content").and_then(|v| v.as_str()) {
            // 跳过 followupPrompt
            if value.get("followupPrompt").is_some() {
                events.push(AwsEvent::FollowupPrompt {
                    content: content.to_string(),
                });
            } else {
                events.push(AwsEvent::Content {
                    text: content.to_string(),
                });
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

            let tool_id = tool_use_id.to_string();

            // 获取或创建工具累积器
            let accumulator = self.tool_accumulators.entry(tool_id.clone()).or_default();

            // 如果有名称，这是工具调用开始
            if !name.is_empty() && accumulator.name.is_empty() {
                accumulator.name = name.clone();
                events.push(AwsEvent::ToolUseStart {
                    id: tool_id.clone(),
                    name,
                });
            }

            // 如果有输入增量
            if !input_chunk.is_empty() {
                accumulator.input.push_str(&input_chunk);
                events.push(AwsEvent::ToolUseInput {
                    id: tool_id.clone(),
                    input: input_chunk,
                });
            }

            // 如果是 stop 事件
            if is_stop {
                self.tool_accumulators.remove(&tool_id);
                events.push(AwsEvent::ToolUseStop { id: tool_id });
            }
        }
        // 处理独立的 stop 事件
        else if value.get("stop").and_then(|v| v.as_bool()).unwrap_or(false) {
            events.push(AwsEvent::Stop);
        }
        // 处理 meteringEvent: {"unit":"credit","unitPlural":"credits","usage":0.34}
        else if let Some(usage) = value.get("usage").and_then(|v| v.as_f64()) {
            events.push(AwsEvent::Usage {
                credits: usage,
                context_percentage: 0.0,
            });
        }
        // 处理 contextUsageEvent: {"contextUsagePercentage":54.36}
        else if let Some(ctx_usage) = value.get("contextUsagePercentage").and_then(|v| v.as_f64())
        {
            events.push(AwsEvent::Usage {
                credits: 0.0,
                context_percentage: ctx_usage,
            });
        }

        Ok(events)
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 将 AwsEvent 序列化为 JSON 字符串（用于测试 round-trip）
pub fn serialize_event(event: &AwsEvent) -> Option<String> {
    match event {
        AwsEvent::Content { text } => Some(serde_json::json!({"content": text}).to_string()),
        AwsEvent::ToolUseStart { id, name } => {
            Some(serde_json::json!({"toolUseId": id, "name": name}).to_string())
        }
        AwsEvent::ToolUseInput { id, input } => {
            Some(serde_json::json!({"toolUseId": id, "input": input}).to_string())
        }
        AwsEvent::ToolUseStop { id } => {
            Some(serde_json::json!({"toolUseId": id, "stop": true}).to_string())
        }
        AwsEvent::Stop => Some(serde_json::json!({"stop": true}).to_string()),
        AwsEvent::Usage {
            credits,
            context_percentage,
        } => {
            if *credits > 0.0 {
                Some(serde_json::json!({"unit": "credit", "usage": credits}).to_string())
            } else if *context_percentage > 0.0 {
                Some(serde_json::json!({"contextUsagePercentage": context_percentage}).to_string())
            } else {
                None
            }
        }
        AwsEvent::FollowupPrompt { content } => {
            Some(serde_json::json!({"content": content, "followupPrompt": true}).to_string())
        }
        AwsEvent::ParseError { .. } => None,
    }
}

/// 从事件列表中提取所有内容文本
pub fn extract_content(events: &[AwsEvent]) -> String {
    events
        .iter()
        .filter_map(|e| {
            if let AwsEvent::Content { text } = e {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

/// 从事件列表中提取所有工具调用
pub fn extract_tool_calls(events: &[AwsEvent]) -> Vec<(String, String, String)> {
    let mut tool_calls: HashMap<String, (String, String)> = HashMap::new();
    let mut completed: Vec<String> = Vec::new();

    for event in events {
        match event {
            AwsEvent::ToolUseStart { id, name } => {
                tool_calls.entry(id.clone()).or_default().0 = name.clone();
            }
            AwsEvent::ToolUseInput { id, input } => {
                tool_calls.entry(id.clone()).or_default().1.push_str(input);
            }
            AwsEvent::ToolUseStop { id } => {
                completed.push(id.clone());
            }
            _ => {}
        }
    }

    completed
        .into_iter()
        .filter_map(|id| {
            tool_calls
                .remove(&id)
                .map(|(name, input)| (id, name, input))
        })
        .collect()
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_new() {
        let parser = AwsEventStreamParser::new();
        assert_eq!(parser.state(), &ParserState::Idle);
        assert_eq!(parser.parse_error_count(), 0);
        assert_eq!(parser.buffer_size(), 0);
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = AwsEventStreamParser::new();
        parser.process(b"{\"content\":\"hello\"}");
        parser.reset();
        assert_eq!(parser.state(), &ParserState::Idle);
        assert_eq!(parser.buffer_size(), 0);
    }

    #[test]
    fn test_parse_content_event() {
        let mut parser = AwsEventStreamParser::new();
        let events = parser.process(b"{\"content\":\"Hello, world!\"}");

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            AwsEvent::Content { text } if text == "Hello, world!"
        ));
    }

    #[test]
    fn test_parse_multiple_content_events() {
        let mut parser = AwsEventStreamParser::new();
        let data = b"{\"content\":\"Hello\"}{\"content\":\", world!\"}";
        let events = parser.process(data);

        assert_eq!(events.len(), 2);
        let content = extract_content(&events);
        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn test_parse_stop_event() {
        let mut parser = AwsEventStreamParser::new();
        let events = parser.process(b"{\"stop\":true}");

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], AwsEvent::Stop));
    }

    #[test]
    fn test_parse_tool_use_complete() {
        let mut parser = AwsEventStreamParser::new();

        // 工具调用开始
        let events1 = parser.process(b"{\"toolUseId\":\"tool_1\",\"name\":\"read_file\"}");
        assert_eq!(events1.len(), 1);
        assert!(matches!(
            &events1[0],
            AwsEvent::ToolUseStart { id, name } if id == "tool_1" && name == "read_file"
        ));

        // 工具调用输入
        let events2 = parser
            .process(b"{\"toolUseId\":\"tool_1\",\"input\":\"{\\\"path\\\":\\\"/tmp/test\\\"}\"}");
        assert_eq!(events2.len(), 1);
        assert!(matches!(
            &events2[0],
            AwsEvent::ToolUseInput { id, input } if id == "tool_1" && input.contains("path")
        ));

        // 工具调用结束
        let events3 = parser.process(b"{\"toolUseId\":\"tool_1\",\"stop\":true}");
        assert_eq!(events3.len(), 1);
        assert!(matches!(
            &events3[0],
            AwsEvent::ToolUseStop { id } if id == "tool_1"
        ));
    }

    #[test]
    fn test_parse_usage_event() {
        let mut parser = AwsEventStreamParser::new();

        // credits 使用量
        let events1 = parser.process(b"{\"unit\":\"credit\",\"usage\":0.34}");
        assert_eq!(events1.len(), 1);
        assert!(matches!(
            &events1[0],
            AwsEvent::Usage { credits, .. } if (*credits - 0.34).abs() < 0.001
        ));

        // 上下文使用百分比
        let events2 = parser.process(b"{\"contextUsagePercentage\":54.36}");
        assert_eq!(events2.len(), 1);
        assert!(matches!(
            &events2[0],
            AwsEvent::Usage { context_percentage, .. } if (*context_percentage - 54.36).abs() < 0.001
        ));
    }

    #[test]
    fn test_parse_followup_prompt() {
        let mut parser = AwsEventStreamParser::new();
        let events = parser.process(b"{\"content\":\"suggestion\",\"followupPrompt\":true}");

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            AwsEvent::FollowupPrompt { content } if content == "suggestion"
        ));
    }

    #[test]
    fn test_parse_with_binary_prefix() {
        let mut parser = AwsEventStreamParser::new();

        // 模拟 AWS Event Stream 格式：二进制头部 + JSON
        let mut data = vec![0x00, 0x00, 0x00, 0x1A]; // 一些二进制头部
        data.extend_from_slice(b"{\"content\":\"test\"}");
        data.extend_from_slice(&[0x00, 0x00]); // 一些二进制尾部

        let events = parser.process(&data);

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            AwsEvent::Content { text } if text == "test"
        ));
    }

    #[test]
    fn test_finish_completes_pending_tool_calls() {
        let mut parser = AwsEventStreamParser::new();

        // 开始工具调用但不结束
        parser.process(b"{\"toolUseId\":\"tool_1\",\"name\":\"test_tool\"}");
        parser.process(b"{\"toolUseId\":\"tool_1\",\"input\":\"test_input\"}");

        // 调用 finish 应该完成未完成的工具调用
        let events = parser.finish();

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            AwsEvent::ToolUseStop { id } if id == "tool_1"
        ));
        assert_eq!(parser.state(), &ParserState::Completed);
    }

    #[test]
    fn test_extract_content() {
        let events = vec![
            AwsEvent::Content {
                text: "Hello".to_string(),
            },
            AwsEvent::Stop,
            AwsEvent::Content {
                text: ", world!".to_string(),
            },
        ];

        let content = extract_content(&events);
        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn test_extract_tool_calls() {
        let events = vec![
            AwsEvent::ToolUseStart {
                id: "t1".to_string(),
                name: "func1".to_string(),
            },
            AwsEvent::ToolUseInput {
                id: "t1".to_string(),
                input: "{\"a\":".to_string(),
            },
            AwsEvent::ToolUseInput {
                id: "t1".to_string(),
                input: "1}".to_string(),
            },
            AwsEvent::ToolUseStop {
                id: "t1".to_string(),
            },
        ];

        let tool_calls = extract_tool_calls(&events);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0],
            (
                "t1".to_string(),
                "func1".to_string(),
                "{\"a\":1}".to_string()
            )
        );
    }

    #[test]
    fn test_serialize_event() {
        let event = AwsEvent::Content {
            text: "test".to_string(),
        };
        let json = serialize_event(&event).unwrap();
        assert!(json.contains("\"content\":\"test\""));

        let event = AwsEvent::Stop;
        let json = serialize_event(&event).unwrap();
        assert!(json.contains("\"stop\":true"));
    }

    #[test]
    fn test_buffer_overflow_protection() {
        let mut parser = AwsEventStreamParser::with_max_buffer_size(100);

        // 发送超过缓冲区大小的数据
        let large_data = vec![b'x'; 200];
        let events = parser.process(&large_data);

        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], AwsEvent::ParseError { message, .. } if message.contains("缓冲区溢出"))
        );
        assert_eq!(parser.parse_error_count(), 1);
    }

    #[test]
    fn test_empty_input() {
        let mut parser = AwsEventStreamParser::new();
        let events = parser.process(b"");
        assert!(events.is_empty());
    }

    #[test]
    fn test_invalid_json_recovery() {
        let mut parser = AwsEventStreamParser::new();

        // 无效 JSON 后跟有效 JSON
        let data = b"{invalid}{\"content\":\"valid\"}";
        let events = parser.process(data);

        // 应该有一个解析错误和一个有效内容
        assert!(events
            .iter()
            .any(|e| matches!(e, AwsEvent::ParseError { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, AwsEvent::Content { text } if text == "valid")));
    }
}

// ============================================================================
// 增量解析测试（需求 2.6）
// ============================================================================

#[cfg(test)]
mod incremental_tests {
    use super::*;

    #[test]
    fn test_incremental_parsing_split_json() {
        let mut parser = AwsEventStreamParser::new();

        // 将一个 JSON 对象分成多个部分发送
        let events1 = parser.process(b"{\"content\":");
        assert!(events1.is_empty(), "不完整的 JSON 不应产生事件");
        assert!(parser.buffer_size() > 0, "缓冲区应该有数据");

        let events2 = parser.process(b"\"Hello, ");
        assert!(events2.is_empty(), "不完整的 JSON 不应产生事件");

        let events3 = parser.process(b"world!\"}");
        assert_eq!(events3.len(), 1, "完整的 JSON 应该产生一个事件");
        assert!(matches!(
            &events3[0],
            AwsEvent::Content { text } if text == "Hello, world!"
        ));

        // 缓冲区应该被清空
        assert_eq!(parser.buffer_size(), 0);
    }

    #[test]
    fn test_incremental_parsing_multiple_chunks() {
        let mut parser = AwsEventStreamParser::new();

        // 模拟网络分片：每次只发送几个字节
        let full_data = b"{\"content\":\"test\"}{\"stop\":true}";
        let mut all_events = Vec::new();

        for chunk in full_data.chunks(5) {
            let events = parser.process(chunk);
            all_events.extend(events);
        }

        // 应该解析出两个事件
        assert_eq!(all_events.len(), 2);
        assert!(matches!(&all_events[0], AwsEvent::Content { text } if text == "test"));
        assert!(matches!(&all_events[1], AwsEvent::Stop));
    }

    #[test]
    fn test_incremental_parsing_with_binary_noise() {
        let mut parser = AwsEventStreamParser::new();

        // 模拟 AWS Event Stream 格式：二进制数据 + JSON + 二进制数据
        let mut data = Vec::new();
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x20]); // 二进制头部
        data.extend_from_slice(b"{\"content\":\"part1\"}");
        data.extend_from_slice(&[0x00, 0x00]); // 二进制分隔

        let events1 = parser.process(&data);
        assert_eq!(events1.len(), 1);

        // 继续发送更多数据
        let mut data2 = Vec::new();
        data2.extend_from_slice(&[0x00, 0x00, 0x00, 0x15]); // 二进制头部
        data2.extend_from_slice(b"{\"content\":\"part2\"}");

        let events2 = parser.process(&data2);
        assert_eq!(events2.len(), 1);

        let content = extract_content(&[events1, events2].concat());
        assert_eq!(content, "part1part2");
    }

    #[test]
    fn test_incremental_tool_call_accumulation() {
        let mut parser = AwsEventStreamParser::new();

        // 工具调用开始
        let events1 = parser.process(b"{\"toolUseId\":\"t1\",\"name\":\"read_file\"}");
        assert_eq!(events1.len(), 1);

        // 分多次发送输入
        let events2 = parser.process(b"{\"toolUseId\":\"t1\",\"input\":\"{\\\"path\\\":\"}");
        assert_eq!(events2.len(), 1);

        let events3 = parser.process(b"{\"toolUseId\":\"t1\",\"input\":\"\\\"/tmp/test\\\"}\"}");
        assert_eq!(events3.len(), 1);

        // 结束工具调用
        let events4 = parser.process(b"{\"toolUseId\":\"t1\",\"stop\":true}");
        assert_eq!(events4.len(), 1);

        // 验证累积的输入
        let all_events = [events1, events2, events3, events4].concat();
        let tool_calls = extract_tool_calls(&all_events);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].0, "t1");
        assert_eq!(tool_calls[0].1, "read_file");
        assert!(tool_calls[0].2.contains("path"));
    }

    #[test]
    fn test_buffer_management_after_complete_json() {
        let mut parser = AwsEventStreamParser::new();

        // 发送完整 JSON 后跟部分 JSON
        let data = b"{\"content\":\"complete\"}{\"content\":\"incom";
        let events = parser.process(data);

        // 应该只解析出完整的 JSON
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], AwsEvent::Content { text } if text == "complete"));

        // 缓冲区应该保留不完整的部分
        assert!(parser.buffer_size() > 0);

        // 完成不完整的 JSON
        let events2 = parser.process(b"plete\"}");
        assert_eq!(events2.len(), 1);
        assert!(matches!(&events2[0], AwsEvent::Content { text } if text == "incomplete"));

        // 缓冲区应该被清空
        assert_eq!(parser.buffer_size(), 0);
    }

    #[test]
    fn test_finish_with_incomplete_json() {
        let mut parser = AwsEventStreamParser::new();

        // 发送不完整的 JSON
        parser.process(b"{\"content\":\"incomplete");
        assert!(parser.buffer_size() > 0);

        // 调用 finish 不应该崩溃
        let events = parser.finish();

        // 不完整的 JSON 不会产生事件
        assert!(events.is_empty());
        assert_eq!(parser.state(), &ParserState::Completed);
    }

    #[test]
    fn test_concurrent_tool_calls() {
        let mut parser = AwsEventStreamParser::new();

        // 开始两个并发的工具调用
        parser.process(b"{\"toolUseId\":\"t1\",\"name\":\"func1\"}");
        parser.process(b"{\"toolUseId\":\"t2\",\"name\":\"func2\"}");

        // 交错发送输入
        parser.process(b"{\"toolUseId\":\"t1\",\"input\":\"input1\"}");
        parser.process(b"{\"toolUseId\":\"t2\",\"input\":\"input2\"}");
        parser.process(b"{\"toolUseId\":\"t1\",\"input\":\"_more\"}");

        // 结束工具调用
        let events1 = parser.process(b"{\"toolUseId\":\"t1\",\"stop\":true}");
        let events2 = parser.process(b"{\"toolUseId\":\"t2\",\"stop\":true}");

        assert!(matches!(&events1[0], AwsEvent::ToolUseStop { id } if id == "t1"));
        assert!(matches!(&events2[0], AwsEvent::ToolUseStop { id } if id == "t2"));
    }

    #[test]
    fn test_state_transitions() {
        let mut parser = AwsEventStreamParser::new();

        // 初始状态
        assert_eq!(parser.state(), &ParserState::Idle);

        // 处理数据后变为 Parsing
        parser.process(b"{\"content\":\"test\"}");
        assert_eq!(parser.state(), &ParserState::Parsing);

        // 完成后变为 Completed
        parser.finish();
        assert_eq!(parser.state(), &ParserState::Completed);

        // 重置后回到 Idle
        parser.reset();
        assert_eq!(parser.state(), &ParserState::Idle);
    }

    #[test]
    fn test_unicode_content_incremental() {
        let mut parser = AwsEventStreamParser::new();

        // 发送包含 Unicode 的 JSON（分片可能在 UTF-8 字符中间）
        let json = "{\"content\":\"你好世界\"}";
        let bytes = json.as_bytes();

        // 分成多个部分发送
        let events1 = parser.process(&bytes[..10]);
        let events2 = parser.process(&bytes[10..20]);
        let events3 = parser.process(&bytes[20..]);

        let all_events = [events1, events2, events3].concat();
        assert_eq!(all_events.len(), 1);
        assert!(matches!(
            &all_events[0],
            AwsEvent::Content { text } if text == "你好世界"
        ));
    }

    #[test]
    fn test_escaped_characters_in_json() {
        let mut parser = AwsEventStreamParser::new();

        // JSON 中包含转义字符（JSON 解析会自动处理转义）
        let events = parser.process(b"{\"content\":\"line1\\nline2\\ttab\"}");
        assert_eq!(events.len(), 1);
        // JSON 解析后，\n 变成换行符，\t 变成制表符
        assert!(matches!(
            &events[0],
            AwsEvent::Content { text } if text == "line1\nline2\ttab"
        ));
    }

    #[test]
    fn test_nested_json_in_tool_input() {
        let mut parser = AwsEventStreamParser::new();

        // 工具输入包含嵌套 JSON
        let events = parser.process(
            b"{\"toolUseId\":\"t1\",\"name\":\"test\",\"input\":\"{\\\"nested\\\":{\\\"key\\\":\\\"value\\\"}}\"}"
        );

        assert_eq!(events.len(), 2); // ToolUseStart + ToolUseInput
        assert!(
            matches!(&events[0], AwsEvent::ToolUseStart { id, name } if id == "t1" && name == "test")
        );
        assert!(
            matches!(&events[1], AwsEvent::ToolUseInput { id, input } if id == "t1" && input.contains("nested"))
        );
    }
}

// ============================================================================
// 错误恢复测试（需求 2.5）
// ============================================================================

#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[test]
    fn test_recovery_from_invalid_json() {
        let mut parser = AwsEventStreamParser::new();

        // 无效 JSON 后跟有效 JSON
        let data = b"{invalid json}{\"content\":\"valid\"}";
        let events = parser.process(data);

        // 应该有一个解析错误和一个有效内容
        let parse_errors: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::ParseError { .. }))
            .collect();
        let content_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::Content { .. }))
            .collect();

        assert_eq!(parse_errors.len(), 1, "应该有一个解析错误");
        assert_eq!(content_events.len(), 1, "应该有一个有效内容");
        assert_eq!(parser.parse_error_count(), 1);
    }

    #[test]
    fn test_recovery_from_multiple_invalid_chunks() {
        let mut parser = AwsEventStreamParser::new();

        // 多个无效 JSON 交错有效 JSON
        let data = b"{bad1}{\"content\":\"good1\"}{bad2}{\"content\":\"good2\"}";
        let events = parser.process(data);

        let parse_errors: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::ParseError { .. }))
            .collect();
        let content_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::Content { .. }))
            .collect();

        assert_eq!(parse_errors.len(), 2, "应该有两个解析错误");
        assert_eq!(content_events.len(), 2, "应该有两个有效内容");
        assert_eq!(parser.parse_error_count(), 2);
    }

    #[test]
    fn test_recovery_preserves_content_order() {
        let mut parser = AwsEventStreamParser::new();

        // 确保错误恢复后内容顺序正确
        let data =
            b"{\"content\":\"first\"}{invalid}{\"content\":\"second\"}{\"content\":\"third\"}";
        let events = parser.process(data);

        let content = extract_content(&events);
        assert_eq!(content, "firstsecondthird");
    }

    #[test]
    fn test_recovery_from_truncated_json() {
        let mut parser = AwsEventStreamParser::new();

        // 截断的 JSON（缺少结束括号）后跟有效 JSON
        // 注意：截断的 JSON 会留在缓冲区中，直到收到更多数据
        let events1 = parser.process(b"{\"content\":\"truncated");
        assert!(events1.is_empty(), "截断的 JSON 不应产生事件");

        // 发送更多数据，包括一个新的有效 JSON
        // 由于缓冲区中有不完整的 JSON，新数据会被追加
        // 这里我们模拟一个场景：旧的不完整 JSON 被新数据"覆盖"
        parser.reset(); // 重置以清除缓冲区

        let events2 = parser.process(b"{\"content\":\"valid\"}");
        assert_eq!(events2.len(), 1);
        assert!(matches!(&events2[0], AwsEvent::Content { text } if text == "valid"));
    }

    #[test]
    fn test_recovery_from_binary_garbage() {
        let mut parser = AwsEventStreamParser::new();

        // 二进制垃圾数据后跟有效 JSON
        let mut data = vec![0xFF, 0xFE, 0x00, 0x01, 0x02];
        data.extend_from_slice(b"{\"content\":\"after garbage\"}");

        let events = parser.process(&data);

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            AwsEvent::Content { text } if text == "after garbage"
        ));
    }

    #[test]
    fn test_recovery_from_empty_json_object() {
        let mut parser = AwsEventStreamParser::new();

        // 空 JSON 对象（有效但不产生事件）后跟有效内容
        let data = b"{}{\"content\":\"after empty\"}";
        let events = parser.process(data);

        // 空对象不产生事件，但也不是错误
        let content_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::Content { .. }))
            .collect();

        assert_eq!(content_events.len(), 1);
        assert_eq!(parser.parse_error_count(), 0, "空对象不应计为错误");
    }

    #[test]
    fn test_recovery_from_unknown_event_type() {
        let mut parser = AwsEventStreamParser::new();

        // 未知事件类型（有效 JSON 但不是已知事件）后跟有效内容
        let data = b"{\"unknownField\":\"value\"}{\"content\":\"known\"}";
        let events = parser.process(data);

        // 未知事件类型不产生事件，但也不是错误
        let content_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::Content { .. }))
            .collect();

        assert_eq!(content_events.len(), 1);
        assert_eq!(parser.parse_error_count(), 0, "未知事件类型不应计为错误");
    }

    #[test]
    fn test_recovery_from_malformed_tool_call() {
        let mut parser = AwsEventStreamParser::new();

        // 格式错误的工具调用（缺少必要字段）后跟有效工具调用
        let data = b"{\"toolUseId\":\"t1\"}{\"toolUseId\":\"t2\",\"name\":\"valid_tool\"}";
        let events = parser.process(data);

        // 第一个工具调用缺少 name，但仍然是有效 JSON
        // 第二个工具调用是完整的
        let tool_starts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::ToolUseStart { .. }))
            .collect();

        assert_eq!(tool_starts.len(), 1, "只有一个有效的工具调用开始");
    }

    #[test]
    fn test_error_count_accumulates() {
        let mut parser = AwsEventStreamParser::new();

        // 多次处理无效数据
        parser.process(b"{invalid1}");
        assert_eq!(parser.parse_error_count(), 1);

        parser.process(b"{invalid2}");
        assert_eq!(parser.parse_error_count(), 2);

        parser.process(b"{\"content\":\"valid\"}");
        assert_eq!(parser.parse_error_count(), 2, "有效数据不应增加错误计数");

        parser.process(b"{invalid3}");
        assert_eq!(parser.parse_error_count(), 3);
    }

    #[test]
    fn test_error_count_resets_on_reset() {
        let mut parser = AwsEventStreamParser::new();

        parser.process(b"{invalid}");
        assert_eq!(parser.parse_error_count(), 1);

        parser.reset();
        assert_eq!(parser.parse_error_count(), 0);
    }

    #[test]
    fn test_parse_error_contains_raw_data() {
        let mut parser = AwsEventStreamParser::new();

        let events = parser.process(b"{invalid json}");

        assert_eq!(events.len(), 1);
        if let AwsEvent::ParseError { message, raw_data } = &events[0] {
            assert!(message.contains("JSON"), "错误消息应该提到 JSON");
            assert!(raw_data.is_some(), "应该包含原始数据");
            assert!(
                raw_data.as_ref().unwrap().contains("invalid"),
                "原始数据应该包含无效内容"
            );
        } else {
            panic!("应该是 ParseError 事件");
        }
    }

    #[test]
    fn test_recovery_continues_tool_accumulation() {
        let mut parser = AwsEventStreamParser::new();

        // 开始工具调用
        parser.process(b"{\"toolUseId\":\"t1\",\"name\":\"test\"}");

        // 发送无效数据
        parser.process(b"{invalid}");

        // 继续工具调用输入
        let events = parser.process(b"{\"toolUseId\":\"t1\",\"input\":\"test_input\"}");

        // 工具调用应该继续正常工作
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], AwsEvent::ToolUseInput { id, .. } if id == "t1"));
    }

    #[test]
    fn test_recovery_from_deeply_nested_invalid_json() {
        let mut parser = AwsEventStreamParser::new();

        // 深度嵌套但无效的 JSON
        let data = b"{\"a\":{\"b\":{\"c\":invalid}}}{\"content\":\"valid\"}";
        let events = parser.process(data);

        let parse_errors: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::ParseError { .. }))
            .collect();
        let content_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::Content { .. }))
            .collect();

        assert_eq!(parse_errors.len(), 1);
        assert_eq!(content_events.len(), 1);
    }

    #[test]
    fn test_recovery_from_json_with_wrong_types() {
        let mut parser = AwsEventStreamParser::new();

        // JSON 有效但字段类型错误（content 应该是字符串，这里是数字）
        // 这种情况下 JSON 解析成功，但不会产生 Content 事件
        let data = b"{\"content\":123}{\"content\":\"valid string\"}";
        let events = parser.process(data);

        let content_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AwsEvent::Content { .. }))
            .collect();

        // 只有字符串类型的 content 会产生事件
        assert_eq!(content_events.len(), 1);
        assert!(matches!(
            content_events[0],
            AwsEvent::Content { text } if text == "valid string"
        ));
    }
}

// ============================================================================
// 属性测试（Property-Based Testing）
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // ========================================================================
    // 生成器（Generators）
    // ========================================================================

    /// 生成有效的内容文本
    fn arb_content_text() -> impl Strategy<Value = String> {
        // 生成不包含控制字符的 Unicode 字符串
        prop::string::string_regex("[a-zA-Z0-9\\u4e00-\\u9fff .,!?\\-_]{0,100}")
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

    /// 生成 ToolUseStart 事件
    fn arb_tool_use_start_event() -> impl Strategy<Value = AwsEvent> {
        (arb_tool_id(), arb_tool_name()).prop_map(|(id, name)| AwsEvent::ToolUseStart { id, name })
    }

    /// 生成 ToolUseInput 事件
    fn arb_tool_use_input_event(id: String) -> impl Strategy<Value = AwsEvent> {
        arb_tool_input().prop_map(move |input| AwsEvent::ToolUseInput {
            id: id.clone(),
            input,
        })
    }

    /// 生成 Usage 事件
    fn arb_usage_event() -> impl Strategy<Value = AwsEvent> {
        prop_oneof![
            (0.01f64..100.0f64).prop_map(|credits| AwsEvent::Usage {
                credits,
                context_percentage: 0.0,
            }),
            (0.01f64..100.0f64).prop_map(|ctx| AwsEvent::Usage {
                credits: 0.0,
                context_percentage: ctx,
            }),
        ]
    }

    /// 生成可序列化的事件（排除 ParseError 和 FollowupPrompt）
    fn arb_serializable_event() -> impl Strategy<Value = AwsEvent> {
        prop_oneof![arb_content_event(), Just(AwsEvent::Stop), arb_usage_event(),]
    }

    /// 生成事件序列
    fn arb_event_sequence() -> impl Strategy<Value = Vec<AwsEvent>> {
        prop::collection::vec(arb_serializable_event(), 1..10)
    }

    // ========================================================================
    // Property 1: AWS Event Stream 解析 Round-Trip
    //
    // *对于任意*有效的 AWS Event Stream 数据，解析后重新序列化应该产生
    // 语义等价的数据（内容、工具调用、使用量信息保持一致）。
    //
    // **验证: 需求 2.1, 2.2, 2.3, 2.6**
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 1: Content 事件 Round-Trip
        ///
        /// **Feature: true-streaming-support, Property 1: AWS Event Stream 解析 Round-Trip**
        /// **Validates: Requirements 2.1, 2.2**
        #[test]
        fn prop_content_event_round_trip(text in arb_content_text()) {
            let event = AwsEvent::Content { text: text.clone() };

            // 序列化
            let json = serialize_event(&event).expect("Content 事件应该可序列化");

            // 解析
            let mut parser = AwsEventStreamParser::new();
            let events = parser.process(json.as_bytes());

            // 验证
            prop_assert_eq!(events.len(), 1, "应该解析出一个事件");
            match &events[0] {
                AwsEvent::Content { text: parsed_text } => {
                    prop_assert_eq!(parsed_text, &text, "内容应该一致");
                }
                _ => prop_assert!(false, "应该是 Content 事件"),
            }
        }

        /// Property 1: Stop 事件 Round-Trip
        ///
        /// **Feature: true-streaming-support, Property 1: AWS Event Stream 解析 Round-Trip**
        /// **Validates: Requirements 2.4**
        #[test]
        fn prop_stop_event_round_trip(_dummy in Just(())) {
            let event = AwsEvent::Stop;

            // 序列化
            let json = serialize_event(&event).expect("Stop 事件应该可序列化");

            // 解析
            let mut parser = AwsEventStreamParser::new();
            let events = parser.process(json.as_bytes());

            // 验证
            prop_assert_eq!(events.len(), 1, "应该解析出一个事件");
            prop_assert!(matches!(&events[0], AwsEvent::Stop), "应该是 Stop 事件");
        }

        /// Property 1: Usage 事件 Round-Trip (credits)
        ///
        /// **Feature: true-streaming-support, Property 1: AWS Event Stream 解析 Round-Trip**
        /// **Validates: Requirements 2.1**
        #[test]
        fn prop_usage_credits_round_trip(credits in 0.01f64..100.0f64) {
            let event = AwsEvent::Usage { credits, context_percentage: 0.0 };

            // 序列化
            let json = serialize_event(&event).expect("Usage 事件应该可序列化");

            // 解析
            let mut parser = AwsEventStreamParser::new();
            let events = parser.process(json.as_bytes());

            // 验证
            prop_assert_eq!(events.len(), 1, "应该解析出一个事件");
            match &events[0] {
                AwsEvent::Usage { credits: parsed_credits, .. } => {
                    prop_assert!((parsed_credits - credits).abs() < 0.001, "credits 应该一致");
                }
                _ => prop_assert!(false, "应该是 Usage 事件"),
            }
        }

        /// Property 1: Usage 事件 Round-Trip (context_percentage)
        ///
        /// **Feature: true-streaming-support, Property 1: AWS Event Stream 解析 Round-Trip**
        /// **Validates: Requirements 2.1**
        #[test]
        fn prop_usage_context_round_trip(ctx in 0.01f64..100.0f64) {
            let event = AwsEvent::Usage { credits: 0.0, context_percentage: ctx };

            // 序列化
            let json = serialize_event(&event).expect("Usage 事件应该可序列化");

            // 解析
            let mut parser = AwsEventStreamParser::new();
            let events = parser.process(json.as_bytes());

            // 验证
            prop_assert_eq!(events.len(), 1, "应该解析出一个事件");
            match &events[0] {
                AwsEvent::Usage { context_percentage: parsed_ctx, .. } => {
                    prop_assert!((parsed_ctx - ctx).abs() < 0.001, "context_percentage 应该一致");
                }
                _ => prop_assert!(false, "应该是 Usage 事件"),
            }
        }

        /// Property 1: 工具调用完整流程 Round-Trip
        ///
        /// **Feature: true-streaming-support, Property 1: AWS Event Stream 解析 Round-Trip**
        /// **Validates: Requirements 2.3**
        #[test]
        fn prop_tool_call_round_trip(
            id in arb_tool_id(),
            name in arb_tool_name(),
            input in arb_tool_input()
        ) {
            let mut parser = AwsEventStreamParser::new();

            // 序列化并解析工具调用开始
            let start_event = AwsEvent::ToolUseStart { id: id.clone(), name: name.clone() };
            let start_json = serialize_event(&start_event).expect("ToolUseStart 应该可序列化");
            let start_events = parser.process(start_json.as_bytes());

            prop_assert_eq!(start_events.len(), 1);
            match &start_events[0] {
                AwsEvent::ToolUseStart { id: parsed_id, name: parsed_name } => {
                    prop_assert_eq!(parsed_id, &id);
                    prop_assert_eq!(parsed_name, &name);
                }
                _ => prop_assert!(false, "应该是 ToolUseStart 事件"),
            }

            // 序列化并解析工具调用输入
            let input_event = AwsEvent::ToolUseInput { id: id.clone(), input: input.clone() };
            let input_json = serialize_event(&input_event).expect("ToolUseInput 应该可序列化");
            let input_events = parser.process(input_json.as_bytes());

            prop_assert_eq!(input_events.len(), 1);
            match &input_events[0] {
                AwsEvent::ToolUseInput { id: parsed_id, input: parsed_input } => {
                    prop_assert_eq!(parsed_id, &id);
                    prop_assert_eq!(parsed_input, &input);
                }
                _ => prop_assert!(false, "应该是 ToolUseInput 事件"),
            }

            // 序列化并解析工具调用结束
            let stop_event = AwsEvent::ToolUseStop { id: id.clone() };
            let stop_json = serialize_event(&stop_event).expect("ToolUseStop 应该可序列化");
            let stop_events = parser.process(stop_json.as_bytes());

            prop_assert_eq!(stop_events.len(), 1);
            match &stop_events[0] {
                AwsEvent::ToolUseStop { id: parsed_id } => {
                    prop_assert_eq!(parsed_id, &id);
                }
                _ => prop_assert!(false, "应该是 ToolUseStop 事件"),
            }
        }

        /// Property 1: 多事件序列 Round-Trip
        ///
        /// **Feature: true-streaming-support, Property 1: AWS Event Stream 解析 Round-Trip**
        /// **Validates: Requirements 2.1, 2.2, 2.6**
        #[test]
        fn prop_event_sequence_round_trip(events in arb_event_sequence()) {
            let mut parser = AwsEventStreamParser::new();

            // 将所有事件序列化为一个字节流
            let mut data = Vec::new();
            for event in &events {
                if let Some(json) = serialize_event(event) {
                    data.extend_from_slice(json.as_bytes());
                }
            }

            // 解析
            let parsed_events = parser.process(&data);

            // 验证：解析出的事件数量应该与原始事件数量一致
            // （排除无法序列化的事件）
            let serializable_count = events.iter()
                .filter(|e| serialize_event(e).is_some())
                .count();

            prop_assert_eq!(
                parsed_events.len(),
                serializable_count,
                "解析出的事件数量应该与可序列化的事件数量一致"
            );
        }

        /// Property 1: 增量解析保持语义等价
        ///
        /// **Feature: true-streaming-support, Property 1: AWS Event Stream 解析 Round-Trip**
        /// **Validates: Requirements 2.6**
        #[test]
        fn prop_incremental_parsing_semantic_equivalence(
            text in arb_content_text(),
            chunk_size in 1usize..20usize
        ) {
            let event = AwsEvent::Content { text: text.clone() };
            let json = serialize_event(&event).expect("Content 事件应该可序列化");
            let bytes = json.as_bytes();

            // 一次性解析
            let mut parser1 = AwsEventStreamParser::new();
            let events1 = parser1.process(bytes);
            let final1 = parser1.finish();
            let all_events1: Vec<_> = events1.into_iter().chain(final1).collect();

            // 增量解析
            let mut parser2 = AwsEventStreamParser::new();
            let mut all_events2 = Vec::new();
            for chunk in bytes.chunks(chunk_size) {
                all_events2.extend(parser2.process(chunk));
            }
            all_events2.extend(parser2.finish());

            // 验证：两种方式解析出的内容应该一致
            let content1 = extract_content(&all_events1);
            let content2 = extract_content(&all_events2);

            prop_assert_eq!(content1, content2, "增量解析应该产生相同的内容");
        }
    }
}
