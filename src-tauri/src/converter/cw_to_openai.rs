//! CodeWhisperer 响应转换为 OpenAI 格式
#![allow(dead_code)]

use crate::models::codewhisperer::*;
use crate::models::openai::*;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// 将 CodeWhisperer 流式事件转换为 OpenAI 格式
pub fn convert_cw_event_to_openai_chunk(
    event: &CWStreamEvent,
    model: &str,
    response_id: &str,
) -> Option<ChatCompletionChunk> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if let Some(resp_event) = &event.assistant_response_event {
        // 文本内容
        if let Some(content) = &resp_event.content {
            return Some(ChatCompletionChunk {
                id: response_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        role: Some("assistant".to_string()),
                        content: Some(content.clone()),
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            });
        }

        // Tool use
        if let Some(tool_use) = &resp_event.tool_use {
            return Some(ChatCompletionChunk {
                id: response_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: tool_use.tool_use_id.clone(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: tool_use.name.clone(),
                                arguments: serde_json::to_string(&tool_use.input)
                                    .unwrap_or_default(),
                            },
                        }]),
                    },
                    finish_reason: None,
                }],
            });
        }
    }

    None
}

/// 创建完成的 OpenAI 响应
pub fn create_openai_response(
    content: &str,
    tool_calls: Option<Vec<ToolCall>>,
    model: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> ChatCompletionResponse {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let finish_reason = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };

    ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: if content.is_empty() {
                    None
                } else {
                    Some(content.to_string())
                },
                tool_calls,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

/// 创建流式结束 chunk
pub fn create_stream_end_chunk(model: &str, response_id: &str) -> ChatCompletionChunk {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    ChatCompletionChunk {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    }
}
