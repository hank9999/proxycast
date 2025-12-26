//! 服务器工具函数
//!
//! 包含响应解析、字符串处理、响应构建等公共工具函数。

use crate::models::openai::{ContentPart, FunctionCall, MessageContent, ToolCall};
use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use futures::stream;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// CodeWhisperer(Kiro) 响应解析选项
#[derive(Debug, Clone, Copy)]
pub struct CWParseOptions {
    /// 是否从响应中提取思考内容（thinking）
    ///
    /// - 当客户端（如 Claude/Anthropic）显式启用 thinking 时，建议打开
    /// - 默认关闭以避免误解析（例如模型解释 `<thinking>` 标签时）
    pub extract_thinking: bool,
}

impl Default for CWParseOptions {
    fn default() -> Self {
        Self {
            extract_thinking: false,
        }
    }
}

/// 解析到的思考内容（thinking）
#[derive(Debug, Clone)]
pub struct CWThinkingContent {
    pub text: String,
    pub signature: Option<String>,
}

/// CodeWhisperer 响应解析结果
#[derive(Debug, Default)]
pub struct CWParsedResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub thinking: Option<CWThinkingContent>,
    pub usage_credits: f64,
    pub context_usage_percentage: f64,
}

impl CWParsedResponse {
    /// 估算 Token 使用量
    ///
    /// 基于响应内容长度和上下文使用百分比估算 Token 数量：
    /// - output_tokens: 基于响应内容长度（约 4 字符 = 1 token）
    /// - input_tokens: 基于 context_usage_percentage（假设 100% = 200k tokens）
    ///
    /// # 返回
    /// (input_tokens, output_tokens) 元组
    pub fn estimate_tokens(&self) -> (u32, u32) {
        // 估算 output tokens: 基于响应内容长度 (约 4 字符 = 1 token)
        let mut output_tokens: u32 = (self.content.len() / 3) as u32;
        if let Some(thinking) = &self.thinking {
            output_tokens += (thinking.text.len() / 3) as u32;
        }
        for tc in &self.tool_calls {
            output_tokens += (tc.function.arguments.len() / 3) as u32;
        }

        // 从 context_usage_percentage 估算 input tokens
        // 假设 100% = 200k tokens (Claude 的上下文窗口)
        let input_tokens = (((self.context_usage_percentage / 100.0) * 200000.0)/0.59) as u32;

        (input_tokens, output_tokens)
    }
}

/// 安全截断字符串到指定字符数，避免 UTF-8 边界问题
pub fn safe_truncate(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        chars[..max_chars].iter().collect()
    }
}

/// 计算 MessageContent 的字符长度
pub fn message_content_len(content: &MessageContent) -> usize {
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

/// 解析 CodeWhisperer AWS Event Stream 响应
///
/// AWS Event Stream 是二进制格式，JSON payload 嵌入在二进制头部之间
pub fn parse_cw_response(body: &str) -> CWParsedResponse {
    parse_cw_response_with_options(body, CWParseOptions::default())
}

/// 解析 CodeWhisperer/Kiro 响应（可选提取 thinking）
pub fn parse_cw_response_with_options(body: &str, options: CWParseOptions) -> CWParsedResponse {
    let mut result = CWParsedResponse::default();
    // 使用 HashMap 来跟踪多个并发的 tool calls
    // key: toolUseId, value: (name, input_accumulated)
    let mut tool_map: HashMap<String, (String, String)> = HashMap::new();

    // thinking 内容（可选）
    let mut thinking_text = String::new();
    let mut thinking_signature: Option<String> = None;

    // 将字符串转换为字节，因为 AWS Event Stream 包含二进制数据
    let bytes = body.as_bytes();

    // 搜索所有 JSON 对象的模式
    // AWS Event Stream 格式: [binary headers]{"content":"..."}[binary trailer]
    let json_patterns: Vec<&[u8]> = vec![
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

        for pattern in &json_patterns {
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
                // 处理独立的 stop 事件（没有 toolUseId）- 这种情况不应该发生，但以防万一
                else if value.get("stop").and_then(|v| v.as_bool()).unwrap_or(false) {
                    // no-op
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

    // 如果启用了 thinking，尝试从 content 中提取 <thinking>...</thinking> 标签作为回退
    if options.extract_thinking {
        let (cleaned, extracted) = extract_thinking_from_content(&result.content);
        result.content = cleaned;
        if !extracted.is_empty() {
            thinking_text.push_str(&extracted);
        }

        let final_thinking = thinking_text;
        if !final_thinking.trim().is_empty() {
            let signature = thinking_signature.or(None);
            result.thinking = Some(CWThinkingContent {
                text: final_thinking,
                signature,
            });
        }
    }

    // 解析 bracket 格式的 tool calls: [Called xxx with args: {...}]
    parse_bracket_tool_calls(&mut result);

    result
}

/// 从 content 中提取 `<thinking>...</thinking>` 标签内容
///
/// 返回：(cleaned_content, extracted_thinking)
fn extract_thinking_from_content(content: &str) -> (String, String) {
    let start_tag = "<thinking>";
    let end_tag = "</thinking>";

    let mut cleaned = String::with_capacity(content.len());
    let mut thinking = String::new();

    let mut remaining = content;
    loop {
        let Some(start) = remaining.find(start_tag) else {
            cleaned.push_str(remaining);
            break;
        };

        let (before, after_start_with_tag) = remaining.split_at(start);
        cleaned.push_str(before);

        let after_start = &after_start_with_tag[start_tag.len()..];
        let Some(end) = after_start.find(end_tag) else {
            // 没有闭合标签：认为不是可解析的 thinking，保留原文
            cleaned.push_str(start_tag);
            cleaned.push_str(after_start);
            break;
        };

        let (think_part, after_end_with_tag) = after_start.split_at(end);
        thinking.push_str(think_part);

        remaining = &after_end_with_tag[end_tag.len()..];
    }

    (cleaned, thinking)
}

/// 在字节数组中查找子序列
pub fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// 从字节数组中提取 JSON 对象字符串
pub fn extract_json_from_bytes(bytes: &[u8]) -> Option<String> {
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
pub fn extract_json_object(s: &str) -> Option<&str> {
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
///
/// 格式: [Called xxx with args: {...}]
pub fn parse_bracket_tool_calls(result: &mut CWParsedResponse) {
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

/// 构建 Anthropic 非流式响应
pub fn build_anthropic_response(model: &str, parsed: &CWParsedResponse) -> Response {
    let has_tool_calls = !parsed.tool_calls.is_empty();
    let mut content_array: Vec<serde_json::Value> = Vec::new();

    // thinking 内容（如果启用并成功解析）
    if let Some(thinking) = &parsed.thinking {
        let mut block = serde_json::json!({
            "type": "thinking",
            "thinking": thinking.text,
        });
        if let Some(sig) = &thinking.signature {
            block["signature"] = serde_json::json!(sig);
        }
        content_array.push(block);
    }

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
    if let Some(thinking) = &parsed.thinking {
        output_tokens += (thinking.text.len() / 4) as u32;
    }
    for tc in &parsed.tool_calls {
        output_tokens += (tc.function.arguments.len() / 4) as u32;
    }
    // 从 context_usage_percentage 估算 input tokens
    // 假设 100% = 200k tokens (Claude 的上下文窗口)
    let input_tokens = (((parsed.context_usage_percentage / 100.0) * 200000.0)/0.59) as u32;

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
pub fn build_anthropic_stream_response(model: &str, parsed: &CWParsedResponse) -> Response {
    let has_tool_calls = !parsed.tool_calls.is_empty();
    let message_id = format!("msg_{}", uuid::Uuid::new_v4());
    let model = model.to_string();
    let events = build_anthropic_stream_events(
        &model,
        &message_id,
        parsed,
        if has_tool_calls { "tool_use" } else { "end_turn" },
    );

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

fn build_anthropic_stream_events(
    model: &str,
    message_id: &str,
    parsed: &CWParsedResponse,
    stop_reason: &str,
) -> Vec<String> {
    let content = parsed.content.clone();
    let tool_calls = parsed.tool_calls.clone();
    let thinking = parsed.thinking.clone();

    let (input_tokens, output_tokens) = parsed.estimate_tokens();

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

    // 1.5 thinking 内容块（可选）
    //
    // Anthropic 官方格式要求：
    // - content_block_start 的 content_block 不包含 signature
    // - signature 通过 content_block_delta 的 signature_delta 发送
    if let Some(thinking) = &thinking {
        // content_block_start
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "thinking", "thinking": ""}
        });
        events.push(format!(
            "event: content_block_start\ndata: {block_start}\n\n"
        ));

        if !thinking.text.is_empty() {
            // content_block_delta: thinking_delta
            let block_delta = serde_json::json!({
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": thinking.text}
            });
            events.push(format!(
                "event: content_block_delta\ndata: {block_delta}\n\n"
            ));
        }

        // 如果有 signature，则按官方格式发送 signature_delta
        if let Some(sig) = &thinking.signature {
            // 官方流里经常在 signature_delta 前额外出现一次空的 thinking_delta
            let empty_thinking_delta = serde_json::json!({
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": ""}
            });
            events.push(format!(
                "event: content_block_delta\ndata: {empty_thinking_delta}\n\n"
            ));

            let signature_delta = serde_json::json!({
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "signature_delta", "signature": sig}
            });
            events.push(format!(
                "event: content_block_delta\ndata: {signature_delta}\n\n"
            ));
        }

        // content_block_stop
        let block_stop = serde_json::json!({
            "type": "content_block_stop",
            "index": block_index
        });
        events.push(format!("event: content_block_stop\ndata: {block_stop}\n\n"));

        block_index += 1;
    }

    // 2. 文本内容块 - 即使为空也要发送，Claude Code 需要至少一个 content block
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {"type": "text", "text": ""}
    });
    events.push(format!(
        "event: content_block_start\ndata: {block_start}\n\n"
    ));

    if !content.is_empty() {
        let block_delta = serde_json::json!({
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "text_delta", "text": content}
        });
        events.push(format!(
            "event: content_block_delta\ndata: {block_delta}\n\n"
        ));
    }

    let block_stop = serde_json::json!({
        "type": "content_block_stop",
        "index": block_index
    });
    events.push(format!("event: content_block_stop\ndata: {block_stop}\n\n"));

    block_index += 1;

    // 3. Tool use 块
    for tc in &tool_calls {
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
            "stop_reason": stop_reason,
            "stop_sequence": null
        },
        "usage": {"output_tokens": output_tokens}
    });
    events.push(format!("event: message_delta\ndata: {message_delta}\n\n"));

    // 5. message_stop
    let message_stop = serde_json::json!({"type": "message_stop"});
    events.push(format!("event: message_stop\ndata: {message_stop}\n\n"));

    events
}

/// 构建 Gemini 原生请求体
///
/// 将用户传入的 Gemini 格式请求转换为 Antigravity 请求格式
pub fn build_gemini_native_request(
    request: &serde_json::Value,
    model: &str,
    project_id: &str,
) -> serde_json::Value {
    // 模型名称映射
    let actual_model = match model {
        "gemini-2.5-computer-use-preview-10-2025" => "rev19-uic3-1p",
        "gemini-3-pro-image-preview" => "gemini-3-pro-image",
        "gemini-3-pro-preview" => "gemini-3-pro-high",
        "gemini-claude-sonnet-4-5" => "claude-sonnet-4-5",
        "gemini-claude-sonnet-4-5-thinking" => "claude-sonnet-4-5-thinking",
        _ => model,
    };

    // 是否启用思维链
    let enable_thinking = model.ends_with("-thinking")
        || model == "gemini-2.5-pro"
        || model.starts_with("gemini-3-pro-")
        || model == "rev19-uic3-1p"
        || model == "gpt-oss-120b-medium";

    // 生成请求 ID 和会话 ID
    let request_id = format!("agent-{}", uuid::Uuid::new_v4());
    let session_id = {
        let uuid = uuid::Uuid::new_v4();
        let bytes = uuid.as_bytes();
        let n: u64 = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) % 9_000_000_000_000_000_000;
        format!("-{}", n)
    };

    // 构建内部请求
    let mut inner_request = request.clone();

    // 添加会话 ID
    inner_request["sessionId"] = serde_json::json!(session_id);

    // 确保有 generationConfig
    if inner_request.get("generationConfig").is_none() {
        inner_request["generationConfig"] = serde_json::json!({
            "temperature": 1.0,
            "maxOutputTokens": 8096,
            "topP": 0.85,
            "topK": 50,
            "candidateCount": 1,
            "stopSequences": [
                "<|user|>",
                "<|bot|>",
                "<|context_request|>",
                "<|endoftext|>",
                "<|end_of_turn|>"
            ],
            "thinkingConfig": {
                "includeThoughts": enable_thinking,
                "thinkingBudget": if enable_thinking { 1024 } else { 0 }
            }
        });
    } else {
        // 确保有 thinkingConfig
        if inner_request["generationConfig"]
            .get("thinkingConfig")
            .is_none()
        {
            inner_request["generationConfig"]["thinkingConfig"] = serde_json::json!({
                "includeThoughts": enable_thinking,
                "thinkingBudget": if enable_thinking { 1024 } else { 0 }
            });
        }
    }

    // 删除安全设置（Antigravity 不支持）
    if let Some(obj) = inner_request.as_object_mut() {
        obj.remove("safetySettings");
    }

    // 构建完整的 Antigravity 请求体
    serde_json::json!({
        "project": project_id,
        "requestId": request_id,
        "request": inner_request,
        "model": actual_model,
        "userAgent": "antigravity"
    })
}

/// 健康检查端点响应
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// 模型列表端点响应
pub async fn models() -> impl IntoResponse {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_truncate() {
        assert_eq!(safe_truncate("hello", 10), "hello");
        assert_eq!(safe_truncate("hello world", 5), "hello");
        assert_eq!(safe_truncate("你好世界", 2), "你好");
    }

    #[test]
    fn test_find_subsequence() {
        let haystack = b"hello world";
        assert_eq!(find_subsequence(haystack, b"world"), Some(6));
        assert_eq!(find_subsequence(haystack, b"foo"), None);
    }

    #[test]
    fn test_extract_json_from_bytes() {
        let json = b"{\"key\":\"value\"}";
        assert_eq!(
            extract_json_from_bytes(json),
            Some("{\"key\":\"value\"}".to_string())
        );

        let nested = b"{\"outer\":{\"inner\":\"value\"}}";
        assert_eq!(
            extract_json_from_bytes(nested),
            Some("{\"outer\":{\"inner\":\"value\"}}".to_string())
        );

        assert_eq!(extract_json_from_bytes(b"not json"), None);
    }
}
