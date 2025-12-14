//! HTTP API 服务器
use crate::config::Config;
use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::logger::LogStore;
use crate::models::anthropic::*;
use crate::models::openai::*;
use crate::providers::claude_custom::ClaudeCustomProvider;
use crate::providers::gemini::GeminiProvider;
use crate::providers::kiro::KiroProvider;
use crate::providers::openai_custom::OpenAICustomProvider;
use crate::providers::qwen::QwenProvider;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{oneshot, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub running: bool,
    pub host: String,
    pub port: u16,
    pub requests: u64,
    pub uptime_secs: u64,
}

pub struct ServerState {
    pub config: Config,
    pub running: bool,
    pub requests: u64,
    pub start_time: Option<std::time::Instant>,
    pub kiro_provider: KiroProvider,
    pub gemini_provider: GeminiProvider,
    pub qwen_provider: QwenProvider,
    pub openai_custom_provider: OpenAICustomProvider,
    pub claude_custom_provider: ClaudeCustomProvider,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl ServerState {
    pub fn new(config: Config) -> Self {
        let mut kiro = KiroProvider::new();
        let _ = kiro.load_credentials();

        let mut gemini = GeminiProvider::new();
        let _ = gemini.load_credentials();

        let mut qwen = QwenProvider::new();
        let _ = qwen.load_credentials();

        let openai_custom = OpenAICustomProvider::new();
        let claude_custom = ClaudeCustomProvider::new();

        Self {
            config,
            running: false,
            requests: 0,
            start_time: None,
            kiro_provider: kiro,
            gemini_provider: gemini,
            qwen_provider: qwen,
            openai_custom_provider: openai_custom,
            claude_custom_provider: claude_custom,
            shutdown_tx: None,
        }
    }

    pub fn status(&self) -> ServerStatus {
        ServerStatus {
            running: self.running,
            host: self.config.server.host.clone(),
            port: self.config.server.port,
            requests: self.requests,
            uptime_secs: self.start_time.map(|t| t.elapsed().as_secs()).unwrap_or(0),
        }
    }

    pub async fn start(
        &mut self,
        logs: Arc<RwLock<LogStore>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.running {
            return Ok(());
        }

        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        let host = self.config.server.host.clone();
        let port = self.config.server.port;
        let api_key = self.config.server.api_key.clone();

        // 重新加载凭证
        let _ = self.kiro_provider.load_credentials();
        let kiro = self.kiro_provider.clone();

        tokio::spawn(async move {
            if let Err(e) = run_server(&host, port, &api_key, kiro, logs, rx).await {
                tracing::error!("Server error: {}", e);
            }
        });

        self.running = true;
        self.start_time = Some(std::time::Instant::now());
        Ok(())
    }

    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        self.running = false;
        self.start_time = None;
    }
}

impl Clone for KiroProvider {
    fn clone(&self) -> Self {
        Self {
            credentials: self.credentials.clone(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Clone)]
struct AppState {
    api_key: String,
    kiro: Arc<RwLock<KiroProvider>>,
    logs: Arc<RwLock<LogStore>>,
}

async fn run_server(
    host: &str,
    port: u16,
    api_key: &str,
    kiro: KiroProvider,
    logs: Arc<RwLock<LogStore>>,
    shutdown: oneshot::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let state = AppState {
        api_key: api_key.to_string(),
        kiro: Arc::new(RwLock::new(kiro)),
        logs,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("Server listening on {}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown.await;
        })
        .await?;

    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": "0.1.0"
    }))
}

async fn models() -> impl IntoResponse {
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

async fn verify_api_key(
    headers: &HeaderMap,
    expected_key: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let auth = headers
        .get("authorization")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": {"message": "No API key provided"}})),
            ))
        }
    };

    if key != expected_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "Invalid API key"}})),
        ));
    }

    Ok(())
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/chat/completions");
        return e.into_response();
    }

    state.logs.write().await.add(
        "info",
        &format!("POST /v1/chat/completions model={}", request.model),
    );

    let kiro = state.kiro.read().await;

    match kiro.call_api(&request).await {
        Ok(resp) => {
            if resp.status().is_success() {
                // 解析 CodeWhisperer 响应并转换
                match resp.text().await {
                    Ok(body) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("info", "Request completed successfully");
                        let response = serde_json::json!({
                            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                            "object": "chat.completion",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": extract_content_from_cw_response(&body)
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        });
                        Json(response).into_response()
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response(),
                }
            } else {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}))
                ).into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
            .into_response(),
    }
}

async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<AnthropicMessagesRequest>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/messages");
        return e.into_response();
    }

    state.logs.write().await.add(
        "info",
        &format!("POST /v1/messages (Anthropic) model={}", request.model),
    );

    // 转换为 OpenAI 格式
    let openai_request = convert_anthropic_to_openai(&request);
    let kiro = state.kiro.read().await;

    match kiro.call_api(&openai_request).await {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.text().await {
                    Ok(body) => {
                        let content = extract_content_from_cw_response(&body);
                        state
                            .logs
                            .write()
                            .await
                            .add("info", "Anthropic request completed successfully");
                        // 返回 Anthropic 格式响应
                        let response = serde_json::json!({
                            "id": format!("msg_{}", uuid::Uuid::new_v4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": content}],
                            "model": request.model,
                            "stop_reason": "end_turn",
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0
                            }
                        });
                        Json(response).into_response()
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("Response parse error: {e}"));
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                state.logs.write().await.add(
                    "error",
                    &format!(
                        "Upstream error {}: {}",
                        status,
                        &body[..body.len().min(200)]
                    ),
                );
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}))
                ).into_response()
            }
        }
        Err(e) => {
            state
                .logs
                .write()
                .await
                .add("error", &format!("API call failed: {e}"));
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response()
        }
    }
}

async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(_request): Json<serde_json::Value>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        return e.into_response();
    }

    // Claude Code 需要这个端点，返回估算值
    Json(serde_json::json!({
        "input_tokens": 100
    }))
    .into_response()
}

fn extract_content_from_cw_response(body: &str) -> String {
    // CodeWhisperer 返回 AWS Event Stream 格式
    // 使用正则提取 JSON 内容
    let mut content = String::new();

    // 查找所有 {"content":"..."} 模式
    let re = regex::Regex::new(r#"\{"content":"([^"\\]*(\\.[^"\\]*)*)"\}"#).ok();

    if let Some(re) = re {
        for cap in re.captures_iter(body) {
            if let Some(text) = cap.get(1) {
                // 处理转义字符
                let unescaped = text
                    .as_str()
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\");
                content.push_str(&unescaped);
            }
        }
    }

    if content.is_empty() {
        // 备用方案：查找 assistantResponseEvent
        if let Some(start) = body.find(r#""content":""#) {
            let rest = &body[start + 11..];
            if let Some(end) = rest.find('"') {
                content = rest[..end].to_string();
            }
        }
    }

    if content.is_empty() {
        "Response received but could not parse content".to_string()
    } else {
        content
    }
}
