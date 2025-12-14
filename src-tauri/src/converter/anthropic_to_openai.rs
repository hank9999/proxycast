//! Anthropic 格式转换为 OpenAI 格式 (支持 Claude Code)
use crate::models::anthropic::*;
use crate::models::openai::*;
use uuid::Uuid;

/// 将 Anthropic MessagesRequest 转换为 OpenAI ChatCompletionRequest
pub fn convert_anthropic_to_openai(request: &AnthropicMessagesRequest) -> ChatCompletionRequest {
    let mut openai_messages: Vec<ChatMessage> = Vec::new();

    // 处理 system prompt
    if let Some(system) = &request.system {
        let system_text = extract_system_text(system);
        if !system_text.is_empty() {
            openai_messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text(system_text)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    // 转换消息
    for msg in &request.messages {
        let converted = convert_anthropic_message(msg);
        openai_messages.extend(converted);
    }

    // 转换 tools
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| Tool {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect()
    });

    ChatCompletionRequest {
        model: request.model.clone(),
        messages: openai_messages,
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        stream: request.stream,
        tools,
        tool_choice: request.tool_choice.clone(),
    }
}

fn extract_system_text(system: &serde_json::Value) -> String {
    match system {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|item| {
                if item.get("type") == Some(&serde_json::Value::String("text".to_string())) {
                    item.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn convert_anthropic_message(msg: &AnthropicMessage) -> Vec<ChatMessage> {
    let mut result: Vec<ChatMessage> = Vec::new();

    match &msg.content {
        serde_json::Value::String(s) => {
            result.push(ChatMessage {
                role: msg.role.clone(),
                content: Some(MessageContent::Text(s.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        serde_json::Value::Array(parts) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut tool_results: Vec<(String, String)> = Vec::new(); // (tool_use_id, content)

            for part in parts {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");

                match part_type {
                    "text" => {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    "tool_use" => {
                        let default_id = format!("call_{}", &Uuid::new_v4().to_string()[..8]);
                        let id = part
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or(&default_id);
                        let name = part.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let input = part.get("input").cloned().unwrap_or(serde_json::json!({}));

                        tool_calls.push(ToolCall {
                            id: id.to_string(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: name.to_string(),
                                arguments: serde_json::to_string(&input).unwrap_or_default(),
                            },
                        });
                    }
                    "tool_result" => {
                        let tool_use_id = part
                            .get("tool_use_id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("");
                        let content = extract_tool_result_content(part.get("content"));
                        tool_results.push((tool_use_id.to_string(), content));
                    }
                    _ => {}
                }
            }

            // 处理 assistant 消息
            if msg.role == "assistant" {
                let content = if text_parts.is_empty() {
                    None
                } else {
                    Some(MessageContent::Text(text_parts.join("")))
                };
                let tc = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                result.push(ChatMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls: tc,
                    tool_call_id: None,
                });
            }
            // 处理 user 消息
            else if msg.role == "user" {
                // 先添加 tool results 作为 tool 角色消息
                for (tool_use_id, content) in tool_results {
                    result.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(MessageContent::Text(content)),
                        tool_calls: None,
                        tool_call_id: Some(tool_use_id),
                    });
                }

                // 添加文本内容
                if !text_parts.is_empty() {
                    result.push(ChatMessage {
                        role: "user".to_string(),
                        content: Some(MessageContent::Text(text_parts.join(""))),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
        }
        _ => {}
    }

    result
}

fn extract_tool_result_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|item| {
                if item.get("type") == Some(&serde_json::Value::String("text".to_string())) {
                    item.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
