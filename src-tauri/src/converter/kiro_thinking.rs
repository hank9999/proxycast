//! Kiro 思考（thinking）模式辅助
//!
//! Kiro/CodeWhisperer 的「思考」能力需要通过在 system prompt 注入特殊标签启用：
//! - `<thinking_mode>enabled</thinking_mode>`
//! - `<max_thinking_length>...</max_thinking_length>`
//!
//! 这里提供对 OpenAI ChatCompletionRequest 的原地修改函数，供 Kiro Provider 调用前注入。

use crate::models::openai::{ChatCompletionRequest, ChatMessage, MessageContent};

/// 默认最大思考长度（字符数）
///
/// 参考社区实现：通常设置到 200000 以允许更长的推理链。
pub const DEFAULT_MAX_THINKING_LENGTH: usize = 200_000;

/// 判断 OpenAI 请求是否“显式启用”思考/推理模式。
///
/// 设计原则：
/// - **只在明确请求时开启**，避免误把用户对 `<thinking>` 标签的讨论当成启用信号
/// - 优先支持 OpenAI 侧常见扩展参数 `reasoning_effort`
/// - 兼容 system prompt 中已存在的 `<thinking_mode>` / `<max_thinking_length>` 配置
pub fn is_openai_thinking_enabled(request: &ChatCompletionRequest) -> bool {
    // 1) OpenAI 扩展：reasoning_effort（low/medium/high/auto/none）
    if let Some(effort) = request.reasoning_effort.as_deref() {
        let effort = effort.trim().to_lowercase();
        if !effort.is_empty() && effort != "none" {
            return true;
        }
    }

    // 2) 兼容：system prompt 已经包含 thinking 配置标签
    request
        .messages
        .iter()
        .filter(|m| m.role == "system")
        .any(|m| {
            let text = m.get_content_text();
            text.contains("<thinking_mode>") || text.contains("<max_thinking_length>")
        })
}

/// 在 OpenAI 请求的 system 消息中注入 Kiro thinking 标签（如果尚未存在）。
///
/// - 如果请求已经包含 `<thinking_mode>` 或 `<max_thinking_length>`，则不会重复注入。
/// - 如果不存在 system 消息，会在 messages 开头插入一条新的 system 消息。
pub fn ensure_kiro_thinking_tags(request: &mut ChatCompletionRequest, max_thinking_length: usize) {
    let thinking_hint = format!(
        "<thinking_mode>enabled</thinking_mode>\n<max_thinking_length>{}</max_thinking_length>",
        max_thinking_length
    );

    // 避免重复注入：仅检查 system 消息（降低误判概率）
    let already_has_thinking_tags = request
        .messages
        .iter()
        .filter(|m| m.role == "system")
        .any(|m| {
            let text = m.get_content_text();
            text.contains("<thinking_mode>") || text.contains("<max_thinking_length>")
        });
    if already_has_thinking_tags {
        return;
    }

    // 找到第一条 system 消息并注入；否则插入新的 system 消息
    if let Some(system_msg) = request.messages.iter_mut().find(|m| m.role == "system") {
        let existing = system_msg.get_content_text();
        let merged = if existing.trim().is_empty() {
            thinking_hint
        } else {
            format!("{thinking_hint}\n\n{existing}")
        };
        system_msg.content = Some(MessageContent::Text(merged));
        return;
    }

    // 没有 system 消息，插入到开头
    request.messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(thinking_hint)),
            tool_calls: None,
            tool_call_id: None,
        },
    );
}
