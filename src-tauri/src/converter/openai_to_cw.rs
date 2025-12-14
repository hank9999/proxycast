//! OpenAI 格式转换为 CodeWhisperer 格式
use crate::models::codewhisperer::*;
use crate::models::openai::*;
use std::collections::HashMap;
use uuid::Uuid;

/// 模型映射表
pub fn get_model_map() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    map.insert("claude-opus-4-5", "claude-opus-4.5");
    map.insert("claude-haiku-4-5", "claude-haiku-4.5");
    map.insert("claude-sonnet-4-5", "CLAUDE_SONNET_4_5_20250929_V1_0");
    map.insert(
        "claude-sonnet-4-5-20250929",
        "CLAUDE_SONNET_4_5_20250929_V1_0",
    );
    map.insert("claude-sonnet-4-20250514", "CLAUDE_SONNET_4_20250514_V1_0");
    map.insert(
        "claude-3-7-sonnet-20250219",
        "CLAUDE_3_7_SONNET_20250219_V1_0",
    );
    map.insert(
        "claude-3-5-sonnet-20241022",
        "CLAUDE_3_7_SONNET_20250219_V1_0",
    );
    map.insert(
        "claude-3-5-sonnet-latest",
        "CLAUDE_3_7_SONNET_20250219_V1_0",
    );
    map
}

pub const DEFAULT_MODEL: &str = "CLAUDE_SONNET_4_5_20250929_V1_0";

/// 将 OpenAI ChatCompletionRequest 转换为 CodeWhisperer 请求
pub fn convert_openai_to_codewhisperer(
    request: &ChatCompletionRequest,
    profile_arn: Option<String>,
) -> CodeWhispererRequest {
    let model_map = get_model_map();
    let cw_model = model_map
        .get(request.model.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let conversation_id = Uuid::new_v4().to_string();

    // 提取 system prompt 和消息
    let mut system_prompt = String::new();
    let mut messages: Vec<&ChatMessage> = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            system_prompt = msg.get_content_text();
        } else {
            messages.push(msg);
        }
    }

    // 构建历史记录
    let mut history: Vec<HistoryItem> = Vec::new();
    let mut start_idx = 0;

    // 处理 system prompt - 合并到第一条用户消息
    if !system_prompt.is_empty() && !messages.is_empty() && messages[0].role == "user" {
        let first_content = messages[0].get_content_text();
        let combined = format!("{system_prompt}\n\n{first_content}");
        history.push(HistoryItem::User(UserHistoryItem {
            user_input_message: UserInputMessage {
                content: combined,
                model_id: cw_model.clone(),
                origin: "AI_EDITOR".to_string(),
                images: None,
                user_input_message_context: None,
            },
        }));
        start_idx = 1;
    }

    // 处理历史消息（除最后一条）
    for msg in messages
        .iter()
        .take(messages.len().saturating_sub(1))
        .skip(start_idx)
    {
        match msg.role.as_str() {
            "user" => {
                let content = msg.get_content_text();
                let tool_results = extract_tool_results(msg);

                let mut user_msg = UserInputMessage {
                    content: if content.is_empty() {
                        if tool_results.is_some() {
                            "Tool results provided.".to_string()
                        } else {
                            "Continue".to_string()
                        }
                    } else {
                        content
                    },
                    model_id: cw_model.clone(),
                    origin: "AI_EDITOR".to_string(),
                    images: None,
                    user_input_message_context: None,
                };

                if tool_results.is_some() {
                    user_msg.user_input_message_context = Some(UserInputMessageContext {
                        tools: None,
                        tool_results,
                    });
                }

                history.push(HistoryItem::User(UserHistoryItem {
                    user_input_message: user_msg,
                }));
            }
            "assistant" => {
                let content = msg.get_content_text();
                let tool_uses = extract_tool_uses(msg);

                history.push(HistoryItem::Assistant(AssistantHistoryItem {
                    assistant_response_message: AssistantResponseMessage {
                        content: if content.is_empty() {
                            "I understand.".to_string()
                        } else {
                            content
                        },
                        tool_uses,
                    },
                }));
            }
            "tool" => {
                let tool_content = msg.get_content_text();
                let tool_id = msg.tool_call_id.clone().unwrap_or_default();

                history.push(HistoryItem::User(UserHistoryItem {
                    user_input_message: UserInputMessage {
                        content: format!(
                            "Tool result: {}",
                            &tool_content[..tool_content.len().min(200)]
                        ),
                        model_id: cw_model.clone(),
                        origin: "AI_EDITOR".to_string(),
                        images: None,
                        user_input_message_context: Some(UserInputMessageContext {
                            tools: None,
                            tool_results: Some(vec![CWToolResult {
                                content: vec![CWTextContent { text: tool_content }],
                                status: "success".to_string(),
                                tool_use_id: tool_id,
                            }]),
                        }),
                    },
                }));
            }
            _ => {}
        }
    }

    // 修复历史记录交替顺序
    let history = fix_history_alternation(history, &cw_model);

    // 构建当前消息
    let current_content = if messages.is_empty() {
        "Continue".to_string()
    } else {
        let last_msg = messages.last().unwrap();
        if last_msg.role == "assistant" {
            "Continue".to_string()
        } else {
            let content = last_msg.get_content_text();
            if content.is_empty() {
                "Continue".to_string()
            } else {
                content
            }
        }
    };

    // 构建 tools
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .take(50)
            .map(|t| {
                let params = t
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));

                let desc = t
                    .function
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("Tool: {}", t.function.name));

                CWTool {
                    tool_specification: ToolSpecification {
                        name: t.function.name.clone(),
                        description: if desc.len() > 500 {
                            format!("{}...", &desc[..497])
                        } else {
                            desc
                        },
                        input_schema: InputSchema { json: params },
                    },
                }
            })
            .collect()
    });

    let current_tool_results = if !messages.is_empty() {
        let last_msg = messages.last().unwrap();
        extract_tool_results(last_msg)
    } else {
        None
    };

    let user_input_message_context = if tools.is_some() || current_tool_results.is_some() {
        Some(UserInputMessageContext {
            tools,
            tool_results: current_tool_results,
        })
    } else {
        None
    };

    CodeWhispererRequest {
        conversation_state: ConversationState {
            chat_trigger_type: "MANUAL".to_string(),
            conversation_id,
            current_message: CurrentMessage {
                user_input_message: UserInputMessage {
                    content: current_content,
                    model_id: cw_model,
                    origin: "AI_EDITOR".to_string(),
                    images: None,
                    user_input_message_context,
                },
            },
            history: if history.is_empty() {
                None
            } else {
                Some(history)
            },
        },
        profile_arn,
    }
}

fn extract_tool_results(msg: &ChatMessage) -> Option<Vec<CWToolResult>> {
    if msg.role == "tool" {
        let content = msg.get_content_text();
        let tool_id = msg.tool_call_id.clone().unwrap_or_default();
        return Some(vec![CWToolResult {
            content: vec![CWTextContent { text: content }],
            status: "success".to_string(),
            tool_use_id: tool_id,
        }]);
    }
    None
}

fn extract_tool_uses(msg: &ChatMessage) -> Option<Vec<CWToolUse>> {
    msg.tool_calls.as_ref().map(|calls| {
        calls
            .iter()
            .map(|tc| {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
                CWToolUse {
                    input,
                    name: tc.function.name.clone(),
                    tool_use_id: tc.id.clone(),
                }
            })
            .collect()
    })
}

/// 修复历史记录，确保 user/assistant 严格交替
fn fix_history_alternation(history: Vec<HistoryItem>, model_id: &str) -> Vec<HistoryItem> {
    if history.is_empty() {
        return history;
    }

    let mut fixed: Vec<HistoryItem> = Vec::new();

    for item in history {
        match &item {
            HistoryItem::User(_) => {
                // 如果上一条也是 user，插入占位 assistant
                if let Some(HistoryItem::User(_)) = fixed.last() {
                    fixed.push(HistoryItem::Assistant(AssistantHistoryItem {
                        assistant_response_message: AssistantResponseMessage {
                            content: "I understand.".to_string(),
                            tool_uses: None,
                        },
                    }));
                }
                fixed.push(item);
            }
            HistoryItem::Assistant(_) => {
                // 如果上一条也是 assistant，插入占位 user
                if let Some(HistoryItem::Assistant(_)) = fixed.last() {
                    fixed.push(HistoryItem::User(UserHistoryItem {
                        user_input_message: UserInputMessage {
                            content: "Continue".to_string(),
                            model_id: model_id.to_string(),
                            origin: "AI_EDITOR".to_string(),
                            images: None,
                            user_input_message_context: None,
                        },
                    }));
                }
                // 如果历史为空，先插入 user
                if fixed.is_empty() {
                    fixed.push(HistoryItem::User(UserHistoryItem {
                        user_input_message: UserInputMessage {
                            content: "Continue".to_string(),
                            model_id: model_id.to_string(),
                            origin: "AI_EDITOR".to_string(),
                            images: None,
                            user_input_message_context: None,
                        },
                    }));
                }
                fixed.push(item);
            }
        }
    }

    // 确保以 assistant 结尾
    if let Some(HistoryItem::User(_)) = fixed.last() {
        fixed.push(HistoryItem::Assistant(AssistantHistoryItem {
            assistant_response_message: AssistantResponseMessage {
                content: "I understand.".to_string(),
                tool_uses: None,
            },
        }));
    }

    fixed
}
