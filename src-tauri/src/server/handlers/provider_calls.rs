//! Provider 调用处理器
//!
//! 根据凭证类型调用不同的 Provider API
//!
//! # 流式传输支持
//!
//! 本模块支持真正的端到端流式传输，通过以下组件实现：
//! - `StreamManager`: 管理流式请求的生命周期
//! - `StreamingProvider`: Provider 的流式 API 接口
//! - `FlowMonitor`: 实时捕获流式响应
//!
//! # 需求覆盖
//!
//! - 需求 4.2: 调用 process_chunk 更新流重建器
//! - 需求 5.1: 在收到 chunk 后立即转发给客户端

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;

use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::converter::kiro_thinking::{ensure_kiro_thinking_tags, DEFAULT_MAX_THINKING_LENGTH};
use crate::converter::openai_to_antigravity::{
    convert_antigravity_to_openai_response, convert_openai_to_antigravity_with_context,
};
use crate::flow_monitor::stream_rebuilder::StreamFormat;
use crate::models::anthropic::AnthropicMessagesRequest;
use crate::models::openai::ChatCompletionRequest;
use crate::models::provider_pool_model::{CredentialData, ProviderCredential};
use crate::providers::{
    AntigravityProvider, ClaudeCustomProvider, KiroProvider, OpenAICustomProvider, VertexProvider,
};
use crate::server::AppState;
use crate::server_utils::{
    build_anthropic_response, build_anthropic_stream_response, parse_cw_response,
    parse_cw_response_with_options, safe_truncate, CWParseOptions, CWParsedResponse,
};
use crate::streaming::{
    StreamConfig, StreamContext, StreamError, StreamFormat as StreamingFormat, StreamManager,
    StreamResponse,
};

/// 根据凭证调用 Provider (Anthropic 格式)
///
/// # 参数
/// - `state`: 应用状态
/// - `credential`: 凭证信息
/// - `request`: Anthropic 格式请求
/// - `flow_id`: Flow ID（可选，用于流式响应处理）
pub async fn call_provider_anthropic(
    state: &AppState,
    credential: &ProviderCredential,
    request: &AnthropicMessagesRequest,
    flow_id: Option<&str>,
) -> Response {
    // 如果是流式请求且有 flow_id，设置流式状态
    if request.stream {
        if let Some(fid) = flow_id {
            // 根据凭证类型确定流格式
            let format = match &credential.credential {
                CredentialData::KiroOAuth { .. } => StreamFormat::OpenAI,
                CredentialData::ClaudeKey { .. } => StreamFormat::Anthropic,
                CredentialData::AntigravityOAuth { .. } => StreamFormat::Gemini,
                _ => StreamFormat::Unknown,
            };
            state.flow_monitor.set_streaming(fid, format).await;
        }
    }

    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            // 是否启用 Claude Thinking（用于 Kiro 思考模式注入）
            let thinking_enabled = request
                .thinking
                .as_ref()
                .map(|t| t.thinking_type == "enabled" && t.budget_tokens.unwrap_or(1) > 0)
                .unwrap_or(false);

            // 使用 TokenCacheService 获取有效 token
            let db = match &state.db {
                Some(db) => db,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": "Database not available"}})),
                    )
                        .into_response();
                }
            };
            // 获取缓存的 token
            let token = match state
                .token_cache
                .get_valid_token(db, &credential.uuid)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("[POOL] Token cache miss, loading from source: {}", e);
                    // 回退到从源文件加载
                    let mut kiro = KiroProvider::new();
                    if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                        // 记录凭证加载失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Failed to load credentials: {}", e)),
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                        )
                            .into_response();
                    }
                    if let Err(e) = kiro.refresh_token().await {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                    kiro.credentials.access_token.unwrap_or_default()
                }
            };
            // 使用获取到的 token 创建 KiroProvider
            let mut kiro = KiroProvider::new();
            kiro.credentials.access_token = Some(token);
            // 从源文件加载其他配置（region, profile_arn 等）
            let _ = kiro.load_credentials_from_path(creds_file_path).await;
            let mut openai_request = convert_anthropic_to_openai(request);
            if thinking_enabled {
                ensure_kiro_thinking_tags(&mut openai_request, DEFAULT_MAX_THINKING_LENGTH);
            }
            let resp = match kiro.call_api(&openai_request).await {
                Ok(r) => r,
                Err(e) => {
                    // 记录 API 调用失败
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&e.to_string()),
                    );
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response();
                }
            };
            let status = resp.status();
            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        let body = String::from_utf8_lossy(&bytes).to_string();
                        let parsed = if thinking_enabled {
                            parse_cw_response_with_options(
                                &body,
                                CWParseOptions {
                                    extract_thinking: true,
                                },
                            )
                        } else {
                            parse_cw_response(&body)
                        };
                        // 记录成功
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                        if request.stream {
                            build_anthropic_stream_response(&request.model, &parsed)
                        } else {
                            build_anthropic_response(&request.model, &parsed)
                        }
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                // Token 过期，强制刷新并重试
                tracing::info!(
                    "[POOL] Got {}, forcing token refresh for {}",
                    status,
                    &credential.uuid[..8]
                );
                let new_token = match state
                    .token_cache
                    .refresh_and_cache(db, &credential.uuid, true)
                    .await
                {
                    Ok(t) => t,
                    Err(e) => {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                };
                // 使用新 token 重试
                kiro.credentials.access_token = Some(new_token);
                match kiro.call_api(&openai_request).await {
                    Ok(retry_resp) => {
                        if retry_resp.status().is_success() {
                            match retry_resp.bytes().await {
                                Ok(bytes) => {
                                    let body = String::from_utf8_lossy(&bytes).to_string();
                                    let parsed = if thinking_enabled {
                                        parse_cw_response_with_options(
                                            &body,
                                            CWParseOptions {
                                                extract_thinking: true,
                                            },
                                        )
                                    } else {
                                        parse_cw_response(&body)
                                    };
                                    // 记录重试成功
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                    if request.stream {
                                        build_anthropic_stream_response(&request.model, &parsed)
                                    } else {
                                        build_anthropic_response(&request.model, &parsed)
                                    }
                                }
                                Err(e) => {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                    )
                                        .into_response()
                                }
                            }
                        } else {
                            let body = retry_resp.text().await.unwrap_or_default();
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&format!("Retry failed: {}", body)),
                            );
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                            )
                                .into_response()
                        }
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                let _ = state
                    .pool_service
                    .mark_unhealthy(db, &credential.uuid, Some(&body));
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": body}})),
                )
                    .into_response()
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            // Gemini OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented. Use /v1/messages with Gemini models instead."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            // Qwen OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented. Use /v1/messages with Qwen models instead."}})),
            )
                .into_response()
        }
        CredentialData::AntigravityOAuth {
            creds_file_path,
            project_id,
        } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity
                .load_credentials_from_path(creds_file_path)
                .await
            {
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }
            // 检查并刷新 token
            if antigravity.is_token_expiring_soon() {
                if let Err(e) = antigravity.refresh_token().await {
                    // 记录 Token 刷新失败
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                    }
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                    )
                        .into_response();
                }
            }
            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }
            // 获取 project_id 用于请求
            let proj_id = antigravity.project_id.clone().unwrap_or_default();
            // 先转换为 OpenAI 格式，再转换为 Antigravity 格式
            let openai_request = convert_anthropic_to_openai(request);
            let antigravity_request = convert_openai_to_antigravity_with_context(&openai_request, &proj_id);
            match antigravity
                .generate_content(&request.model, &antigravity_request)
                .await
            {
                Ok(resp) => {
                    // 转换为 OpenAI 格式，再构建 Anthropic 响应
                    let content = resp["candidates"][0]["content"]["parts"][0]["text"]
                        .as_str()
                        .unwrap_or("");
                    let parsed = CWParsedResponse {
                        content: content.to_string(),
                        tool_calls: Vec::new(),
                        thinking: None,
                        usage_credits: 0.0,
                        context_usage_percentage: 0.0,
                    };
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    if request.stream {
                        build_anthropic_stream_response(&request.model, &parsed)
                    } else {
                        build_anthropic_response(&request.model, &parsed)
                    }
                }
                Err(e) => {
                    // 记录 API 调用失败
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            let openai_request = convert_anthropic_to_openai(request);
            match openai.call_api(&openai_request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(openai_resp) =
                                    serde_json::from_str::<serde_json::Value>(&body)
                                {
                                    let content = openai_resp["choices"][0]["message"]["content"]
                                        .as_str()
                                        .unwrap_or("");
                                    let parsed = CWParsedResponse {
                                        content: content.to_string(),
                                        tool_calls: Vec::new(),
                                        thinking: None,
                                        usage_credits: 0.0,
                                        context_usage_percentage: 0.0,
                                    };
                                    // 记录成功
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_healthy(
                                            db,
                                            &credential.uuid,
                                            Some(&request.model),
                                        );
                                        let _ =
                                            state.pool_service.record_usage(db, &credential.uuid);
                                    }
                                    if request.stream {
                                        build_anthropic_stream_response(&request.model, &parsed)
                                    } else {
                                        build_anthropic_response(&request.model, &parsed)
                                    }
                                } else {
                                    // 记录解析失败
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_unhealthy(
                                            db,
                                            &credential.uuid,
                                            Some("Failed to parse OpenAI response"),
                                        );
                                    }
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Failed to parse OpenAI response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&body),
                            );
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            let request_url = claude.get_base_url();
            state.logs.write().await.add(
                "info",
                &format!(
                    "[CLAUDE] 使用 Claude API 代理: base_url={} -> {}/v1/messages credential_uuid={}",
                    actual_base_url,
                    request_url,
                    &credential.uuid[..8]
                ),
            );
            // 打印请求参数
            let request_json = serde_json::to_string(request).unwrap_or_default();
            state.logs.write().await.add(
                "debug",
                &format!(
                    "[CLAUDE] 请求参数: {}",
                    &request_json.chars().take(500).collect::<String>()
                ),
            );
            match claude.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    // 打印响应状态
                    state.logs.write().await.add(
                        "info",
                        &format!(
                            "[CLAUDE] 响应状态: status={} model={}",
                            status,
                            request.model
                        ),
                    );
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                // 打印响应内容预览
                                state.logs.write().await.add(
                                    "debug",
                                    &format!(
                                        "[CLAUDE] 响应内容: {}",
                                        &body.chars().take(500).collect::<String>()
                                    ),
                                );
                                // 记录成功
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({"error": {"message": "Failed to build response"}})),
                                        )
                                            .into_response()
                                    })
                            } else {
                                state.logs.write().await.add(
                                    "error",
                                    &format!(
                                        "[CLAUDE] 请求失败: status={} body={}",
                                        status,
                                        &body.chars().take(200).collect::<String>()
                                    ),
                                );
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&body),
                                    );
                                }
                                (
                                    StatusCode::from_u16(status.as_u16())
                                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                                    Json(serde_json::json!({"error": {"message": body}})),
                                )
                                    .into_response()
                            }
                        }
                        Err(e) => {
                            state.logs.write().await.add(
                                "error",
                                &format!("[CLAUDE] 读取响应失败: {}", e),
                            );
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(
                                    db,
                                    &credential.uuid,
                                    Some(&e.to_string()),
                                );
                            }
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response()
                        }
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::VertexKey { api_key, base_url, .. } => {
            // Vertex AI uses Gemini-compatible API, convert Anthropic to OpenAI format first
            let openai_request = convert_anthropic_to_openai(request);
            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&openai_request).unwrap_or_default()).await {
                Ok(resp) => {
                    let status = resp.status();
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Failed to build response"}}))).into_response()
                                    })
                            } else {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&body));
                                }
                                (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(serde_json::json!({"error": {"message": body}}))).into_response()
                            }
                        }
                        Err(e) => {
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                            }
                            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                        }
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                }
            }
        }
        // Gemini API Key credentials - not supported for Anthropic format
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support Anthropic format"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 Anthropic 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support Anthropic format yet"}})),
            )
                .into_response()
        }
    }
}

/// 根据凭证调用 Provider (OpenAI 格式)
///
/// # 参数
/// - `state`: 应用状态
/// - `credential`: 凭证信息
/// - `request`: OpenAI 格式请求
/// - `flow_id`: Flow ID（可选，用于流式响应处理）
pub async fn call_provider_openai(
    state: &AppState,
    credential: &ProviderCredential,
    request: &ChatCompletionRequest,
    flow_id: Option<&str>,
) -> Response {
    let _start_time = std::time::Instant::now();
    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            let mut kiro = KiroProvider::new();
            if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                )
                    .into_response();
            }
            if let Err(e) = kiro.refresh_token().await {
                // 记录 Token 刷新失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&format!("Token refresh failed: {}", e)));
                }
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                )
                    .into_response();
            }
            match kiro.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // 记录成功
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                            let _ = state.pool_service.record_usage(db, &credential.uuid);
                        }
                        match resp.text().await {
                            Ok(body) => {
                                let parsed = parse_cw_response(&body);
                                let has_tool_calls = !parsed.tool_calls.is_empty();
                                let message = if has_tool_calls {
                                    serde_json::json!({
                                        "role": "assistant",
                                        "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                        "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                            serde_json::json!({
                                                "id": tc.id,
                                                "type": "function",
                                                "function": {
                                                    "name": tc.function.name,
                                                    "arguments": tc.function.arguments
                                                }
                                            })
                                        }).collect::<Vec<_>>()
                                    })
                                } else {
                                    serde_json::json!({
                                        "role": "assistant",
                                        "content": parsed.content
                                    })
                                };
                                Json(serde_json::json!({
                                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                    "object": "chat.completion",
                                    "created": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs(),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "message": message,
                                        "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                                    }],
                                    "usage": {
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "total_tokens": 0
                                    }
                                }))
                                .into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        // 记录 API 调用失败
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&format!("HTTP {}: {}", status, safe_truncate(&body, 100))));
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    // 记录请求错误
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::AntigravityOAuth { creds_file_path, project_id } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity.load_credentials_from_path(creds_file_path).await {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }
            // 检查并刷新 token
            if antigravity.is_token_expiring_soon() {
                if let Err(e) = antigravity.refresh_token().await {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                    )
                        .into_response();
                }
            }
            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }
            // 获取 project_id 用于请求
            let proj_id = antigravity.project_id.clone().unwrap_or_default();
            // 转换请求格式
            let antigravity_request = convert_openai_to_antigravity_with_context(request, &proj_id);
            match antigravity.generate_content(&request.model, &antigravity_request).await {
                Ok(resp) => {
                    let openai_response = convert_antigravity_to_openai_response(&resp, &request.model);
                    Json(openai_response).into_response()
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            match openai.call_api(request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Invalid JSON response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={}",
                actual_base_url,
                &credential.uuid[..8]
            );
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            match claude.call_openai_api(request).await {
                Ok(resp) => Json(resp).into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::VertexKey { api_key, base_url, model_aliases } => {
            // Resolve model alias if present
            let resolved_model = model_aliases.get(&request.model).cloned().unwrap_or_else(|| request.model.clone());
            let mut modified_request = request.clone();
            modified_request.model = resolved_model;
            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&modified_request).unwrap_or_default()).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Invalid JSON response"}}))).into_response()
                                }
                            }
                            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": body}}))).into_response()
                    }
                }
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
            }
        }
        // Gemini API Key credentials - not supported for OpenAI format yet
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support OpenAI format yet"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 OpenAI 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support OpenAI format yet"}})),
            )
                .into_response()
        }
    }
}

// ============================================================================
// 流式传输支持
// ============================================================================

/// 获取凭证对应的流式格式
///
/// 根据凭证类型返回对应的流式响应格式。
///
/// # 参数
/// - `credential`: 凭证信息
///
/// # 返回
/// 流式格式枚举
pub fn get_stream_format_for_credential(credential: &ProviderCredential) -> StreamingFormat {
    match &credential.credential {
        CredentialData::KiroOAuth { .. } => StreamingFormat::AwsEventStream,
        CredentialData::ClaudeKey { .. } => StreamingFormat::AnthropicSse,
        CredentialData::OpenAIKey { .. } => StreamingFormat::OpenAiSse,
        // TODO: 任务 6 完成后，将这些改为 GeminiStream
        CredentialData::AntigravityOAuth { .. } => StreamingFormat::OpenAiSse,
        CredentialData::GeminiOAuth { .. } => StreamingFormat::OpenAiSse,
        CredentialData::GeminiApiKey { .. } => StreamingFormat::OpenAiSse,
        CredentialData::VertexKey { .. } => StreamingFormat::OpenAiSse,
        _ => StreamingFormat::OpenAiSse,
    }
}

/// 处理流式响应
///
/// 使用 StreamManager 处理流式响应，集成 Flow Monitor。
///
/// # 参数
/// - `state`: 应用状态
/// - `flow_id`: Flow ID（用于 Flow Monitor 集成）
/// - `source_stream`: 源字节流
/// - `source_format`: 源流格式
/// - `target_format`: 目标流格式
/// - `model`: 模型名称
///
/// # 返回
/// SSE 格式的 HTTP 响应
///
/// # 需求覆盖
/// - 需求 4.2: 调用 process_chunk 更新流重建器
/// - 需求 5.1: 在收到 chunk 后立即转发给客户端
pub async fn handle_streaming_response(
    state: &AppState,
    flow_id: Option<&str>,
    source_stream: StreamResponse,
    source_format: StreamingFormat,
    target_format: StreamingFormat,
    model: &str,
) -> Response {
    // 创建流式管理器
    let manager = StreamManager::with_default_config();

    // 创建流式上下文
    let context = StreamContext::new(
        flow_id.map(|s| s.to_string()),
        source_format,
        target_format,
        model,
    );

    // 获取 flow_id 的克隆用于回调
    let flow_id_for_callback = flow_id.map(|s| s.to_string());
    let flow_monitor = state.flow_monitor.clone();

    // 创建带回调的流式处理
    let managed_stream = if let Some(fid) = flow_id_for_callback {
        // 使用带回调的流式处理，集成 Flow Monitor
        let on_chunk = move |event: &str, _metrics: &crate::streaming::StreamMetrics| {
            // 解析 SSE 事件并调用 process_chunk
            // SSE 格式: "event: xxx\ndata: {...}\n\n"
            let lines: Vec<&str> = event.lines().collect();
            let mut event_type: Option<&str> = None;
            let mut data: Option<&str> = None;

            for line in lines {
                if line.starts_with("event: ") {
                    event_type = Some(&line[7..]);
                } else if line.starts_with("data: ") {
                    data = Some(&line[6..]);
                }
            }

            if let Some(d) = data {
                // 使用 tokio::spawn 异步调用 process_chunk
                let flow_monitor_clone = flow_monitor.clone();
                let fid_clone = fid.clone();
                let event_type_owned = event_type.map(|s| s.to_string());
                let data_owned = d.to_string();

                tokio::spawn(async move {
                    flow_monitor_clone
                        .process_chunk(&fid_clone, event_type_owned.as_deref(), &data_owned)
                        .await;
                });
            }
        };

        let stream = manager.handle_stream_with_callback(context, source_stream, on_chunk);

        // 转换为 Body 流
        let body_stream = stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
            match result {
                Ok(event) => Ok(axum::body::Bytes::from(event)),
                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
            }
        });

        Body::from_stream(body_stream)
    } else {
        // 没有 flow_id，使用普通流式处理
        let stream = manager.handle_stream(context, source_stream);

        let body_stream = stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
            match result {
                Ok(event) => Ok(axum::body::Bytes::from(event)),
                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
            }
        });

        Body::from_stream(body_stream)
    };

    // 构建 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(managed_stream)
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                ),
            )
                .into_response()
        })
}

/// 处理流式响应（带超时）
///
/// 与 `handle_streaming_response` 类似，但添加了超时保护。
///
/// # 参数
/// - `state`: 应用状态
/// - `flow_id`: Flow ID
/// - `source_stream`: 源字节流
/// - `source_format`: 源流格式
/// - `target_format`: 目标流格式
/// - `model`: 模型名称
/// - `timeout_ms`: 超时时间（毫秒）
///
/// # 返回
/// SSE 格式的 HTTP 响应
///
/// # 需求覆盖
/// - 需求 6.2: 超时错误处理
/// - 需求 6.5: 可配置的流式响应超时
pub async fn handle_streaming_response_with_timeout(
    state: &AppState,
    flow_id: Option<&str>,
    source_stream: StreamResponse,
    source_format: StreamingFormat,
    target_format: StreamingFormat,
    model: &str,
    timeout_ms: u64,
) -> Response {
    use futures::stream::BoxStream;

    // 创建带超时配置的流式管理器
    let config = StreamConfig::new()
        .with_timeout_ms(timeout_ms)
        .with_chunk_timeout_ms(30_000); // 30 秒 chunk 超时

    let manager = StreamManager::new(config.clone());

    // 创建流式上下文
    let context = StreamContext::new(
        flow_id.map(|s| s.to_string()),
        source_format,
        target_format,
        model,
    );

    // 获取 flow_id 的克隆用于回调
    let flow_id_for_callback = flow_id.map(|s| s.to_string());
    let flow_monitor = state.flow_monitor.clone();

    // 创建带超时的流式处理，使用 BoxStream 统一类型
    let timeout_stream: BoxStream<'static, Result<String, crate::streaming::StreamError>> =
        if let Some(fid) = flow_id_for_callback {
            let on_chunk = move |event: &str, _metrics: &crate::streaming::StreamMetrics| {
                let lines: Vec<&str> = event.lines().collect();
                let mut event_type: Option<&str> = None;
                let mut data: Option<&str> = None;

                for line in lines {
                    if line.starts_with("event: ") {
                        event_type = Some(&line[7..]);
                    } else if line.starts_with("data: ") {
                        data = Some(&line[6..]);
                    }
                }

                if let Some(d) = data {
                    let flow_monitor_clone = flow_monitor.clone();
                    let fid_clone = fid.clone();
                    let event_type_owned = event_type.map(|s| s.to_string());
                    let data_owned = d.to_string();

                    tokio::spawn(async move {
                        flow_monitor_clone
                            .process_chunk(&fid_clone, event_type_owned.as_deref(), &data_owned)
                            .await;
                    });
                }
            };

            let stream = manager.handle_stream_with_callback(context, source_stream, on_chunk);
            Box::pin(crate::streaming::with_timeout(stream, &config))
        } else {
            let stream = manager.handle_stream(context, source_stream);
            Box::pin(crate::streaming::with_timeout(stream, &config))
        };

    // 转换为 Body 流
    let body_stream = timeout_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
        match result {
            Ok(event) => Ok(axum::body::Bytes::from(event)),
            Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
        }
    });

    // 构建 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                ),
            )
                .into_response()
        })
}

/// 将 reqwest 响应转换为 StreamResponse
///
/// 用于将 Provider 的 HTTP 响应转换为统一的流式响应类型。
///
/// # 参数
/// - `response`: reqwest HTTP 响应
///
/// # 返回
/// 统一的流式响应类型
pub fn response_to_stream(response: reqwest::Response) -> StreamResponse {
    crate::streaming::reqwest_stream_to_stream_response(response)
}

// ============================================================================
// 客户端断开检测
// ============================================================================

/// 带客户端断开检测的流式响应处理
///
/// 在流式传输过程中检测客户端是否断开连接，并在断开时：
/// 1. 停止处理上游数据
/// 2. 标记 Flow 为取消状态
/// 3. 清理资源
///
/// # 参数
/// - `state`: 应用状态
/// - `flow_id`: Flow ID
/// - `source_stream`: 源字节流
/// - `source_format`: 源流格式
/// - `target_format`: 目标流格式
/// - `model`: 模型名称
/// - `cancel_token`: 取消令牌（用于取消上游请求）
///
/// # 返回
/// SSE 格式的 HTTP 响应
///
/// # 需求覆盖
/// - 需求 5.4: 客户端断开时取消上游请求
pub async fn handle_streaming_with_disconnect_detection(
    state: &AppState,
    flow_id: Option<&str>,
    source_stream: StreamResponse,
    source_format: StreamingFormat,
    target_format: StreamingFormat,
    model: &str,
    cancel_token: Option<tokio_util::sync::CancellationToken>,
) -> Response {
    use futures::StreamExt;

    // 创建流式管理器
    let manager = StreamManager::with_default_config();

    // 创建流式上下文
    let context = StreamContext::new(
        flow_id.map(|s| s.to_string()),
        source_format,
        target_format,
        model,
    );

    // 获取 flow_id 的克隆
    let flow_id_for_callback = flow_id.map(|s| s.to_string());
    let flow_id_for_cancel = flow_id.map(|s| s.to_string());
    let flow_monitor = state.flow_monitor.clone();
    let flow_monitor_for_cancel = state.flow_monitor.clone();

    // 创建带回调的流式处理
    // 使用 BoxStream 统一类型
    let managed_stream: futures::stream::BoxStream<
        'static,
        Result<String, crate::streaming::StreamError>,
    > = if let Some(fid) = flow_id_for_callback {
        let on_chunk = move |event: &str, _metrics: &crate::streaming::StreamMetrics| {
            let lines: Vec<&str> = event.lines().collect();
            let mut event_type: Option<&str> = None;
            let mut data: Option<&str> = None;

            for line in lines {
                if line.starts_with("event: ") {
                    event_type = Some(&line[7..]);
                } else if line.starts_with("data: ") {
                    data = Some(&line[6..]);
                }
            }

            if let Some(d) = data {
                let flow_monitor_clone = flow_monitor.clone();
                let fid_clone = fid.clone();
                let event_type_owned = event_type.map(|s| s.to_string());
                let data_owned = d.to_string();

                tokio::spawn(async move {
                    flow_monitor_clone
                        .process_chunk(&fid_clone, event_type_owned.as_deref(), &data_owned)
                        .await;
                });
            }
        };

        Box::pin(manager.handle_stream_with_callback(context, source_stream, on_chunk))
    } else {
        // 没有 flow_id，使用普通流式处理
        Box::pin(manager.handle_stream(context, source_stream))
    };

    // 如果有取消令牌，创建一个可取消的流
    let body_stream = if let Some(token) = cancel_token {
        // 创建一个可取消的流
        let cancellable_stream = CancellableStream::new(managed_stream, token.clone());

        // 当流被取消时，标记 Flow 为取消状态
        let cancel_handler = {
            let token = token.clone();
            let flow_id = flow_id_for_cancel.clone();
            async move {
                token.cancelled().await;
                if let Some(fid) = flow_id {
                    flow_monitor_for_cancel.cancel_flow(&fid).await;
                    tracing::info!("[STREAM] 客户端断开，已取消 Flow: {}", fid);
                }
            }
        };

        // 在后台运行取消处理器
        tokio::spawn(cancel_handler);

        // 转换为 Body 流
        let stream =
            cancellable_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
                match result {
                    Ok(event) => Ok(axum::body::Bytes::from(event)),
                    Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
                }
            });

        Body::from_stream(stream)
    } else {
        // 没有取消令牌，使用普通流
        let stream = managed_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
            match result {
                Ok(event) => Ok(axum::body::Bytes::from(event)),
                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
            }
        });

        Body::from_stream(stream)
    };

    // 构建 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(body_stream)
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                ),
            )
                .into_response()
        })
}

/// 可取消的流包装器
///
/// 包装一个流，使其可以通过取消令牌取消。
/// 当取消令牌被触发时，流将返回 ClientDisconnected 错误。
pub struct CancellableStream<S> {
    inner: S,
    cancel_token: tokio_util::sync::CancellationToken,
    cancelled: bool,
}

impl<S> CancellableStream<S> {
    /// 创建新的可取消流
    pub fn new(inner: S, cancel_token: tokio_util::sync::CancellationToken) -> Self {
        Self {
            inner,
            cancel_token,
            cancelled: false,
        }
    }
}

impl<S> futures::Stream for CancellableStream<S>
where
    S: futures::Stream<Item = Result<String, StreamError>> + Unpin,
{
    type Item = Result<String, StreamError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;

        // 检查是否已取消
        if self.cancelled {
            return Poll::Ready(None);
        }

        // 检查取消令牌
        if self.cancel_token.is_cancelled() {
            self.cancelled = true;
            return Poll::Ready(Some(Err(StreamError::ClientDisconnected)));
        }

        // 轮询内部流
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// 创建取消令牌
///
/// 创建一个可用于取消流式请求的令牌。
///
/// # 返回
/// 取消令牌
pub fn create_cancel_token() -> tokio_util::sync::CancellationToken {
    tokio_util::sync::CancellationToken::new()
}

/// 检测客户端断开并触发取消
///
/// 监控客户端连接状态，当检测到断开时触发取消令牌。
///
/// # 参数
/// - `cancel_token`: 取消令牌
///
/// # 注意
/// 此函数应该在单独的任务中运行，与流式响应并行。
/// 实际的断开检测依赖于 axum 的连接管理。
pub async fn monitor_client_disconnect(cancel_token: tokio_util::sync::CancellationToken) {
    // 在实际应用中，这里会监控客户端连接状态
    // 当检测到断开时，调用 cancel_token.cancel()
    //
    // 由于 axum 的 SSE 响应会自动处理客户端断开，
    // 这个函数主要用于需要主动检测断开的场景

    // 等待取消令牌被触发（由其他地方触发）
    cancel_token.cancelled().await;
}
