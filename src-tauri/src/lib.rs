mod config;
mod converter;
mod logger;
mod models;
mod providers;
mod server;

use std::sync::Arc;
use tokio::sync::RwLock;

pub type AppState = Arc<RwLock<server::ServerState>>;
pub type LogState = Arc<RwLock<logger::LogStore>>;

#[tauri::command]
async fn start_server(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "Starting server...");
    s.start(logs.inner().clone())
        .await
        .map_err(|e| e.to_string())?;
    logs.write().await.add(
        "info",
        &format!(
            "Server started on {}:{}",
            s.config.server.host, s.config.server.port
        ),
    );
    Ok("Server started".to_string())
}

#[tauri::command]
async fn stop_server(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    s.stop().await;
    logs.write().await.add("info", "Server stopped");
    Ok("Server stopped".to_string())
}

#[tauri::command]
async fn get_server_status(
    state: tauri::State<'_, AppState>,
) -> Result<server::ServerStatus, String> {
    let s = state.read().await;
    Ok(s.status())
}

#[tauri::command]
async fn get_config(state: tauri::State<'_, AppState>) -> Result<config::Config, String> {
    let s = state.read().await;
    Ok(s.config.clone())
}

#[tauri::command]
async fn save_config(
    state: tauri::State<'_, AppState>,
    config: config::Config,
) -> Result<(), String> {
    let mut s = state.write().await;
    s.config = config.clone();
    config::save_config(&config).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_default_provider(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let s = state.read().await;
    Ok(s.config.default_provider.clone())
}

#[tauri::command]
async fn set_default_provider(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    provider: String,
) -> Result<String, String> {
    let valid_providers = ["kiro", "gemini", "qwen", "openai", "claude"];
    if !valid_providers.contains(&provider.as_str()) {
        return Err(format!("Invalid provider: {provider}"));
    }

    let mut s = state.write().await;
    s.config.default_provider = provider.clone();
    config::save_config(&s.config).map_err(|e| e.to_string())?;
    logs.write()
        .await
        .add("info", &format!("默认 Provider 已切换为: {provider}"));
    Ok(provider)
}

#[tauri::command]
async fn refresh_kiro_token(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "Refreshing Kiro token...");
    let result = s
        .kiro_provider
        .refresh_token()
        .await
        .map_err(|e| e.to_string());
    match &result {
        Ok(_) => logs
            .write()
            .await
            .add("info", "Token refreshed successfully"),
        Err(e) => logs
            .write()
            .await
            .add("error", &format!("Token refresh failed: {e}")),
    }
    result
}

#[tauri::command]
async fn reload_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "Reloading credentials...");
    s.kiro_provider
        .load_credentials()
        .map_err(|e| e.to_string())?;
    logs.write().await.add("info", "Credentials reloaded");
    Ok("Credentials reloaded".to_string())
}

#[derive(serde::Serialize)]
struct KiroCredentialStatus {
    loaded: bool,
    has_access_token: bool,
    has_refresh_token: bool,
    region: Option<String>,
    auth_method: Option<String>,
    expires_at: Option<String>,
    creds_path: String,
}

#[tauri::command]
async fn get_kiro_credentials(
    state: tauri::State<'_, AppState>,
) -> Result<KiroCredentialStatus, String> {
    let s = state.read().await;
    let creds = &s.kiro_provider.credentials;
    let path = providers::kiro::KiroProvider::default_creds_path();

    Ok(KiroCredentialStatus {
        loaded: creds.access_token.is_some() || creds.refresh_token.is_some(),
        has_access_token: creds.access_token.is_some(),
        has_refresh_token: creds.refresh_token.is_some(),
        region: creds.region.clone(),
        auth_method: creds.auth_method.clone(),
        expires_at: creds.expires_at.clone(),
        creds_path: path.to_string_lossy().to_string(),
    })
}

#[derive(serde::Serialize)]
struct EnvVariable {
    key: String,
    value: String,
    masked: String,
}

#[tauri::command]
async fn get_env_variables(state: tauri::State<'_, AppState>) -> Result<Vec<EnvVariable>, String> {
    let s = state.read().await;
    let creds = &s.kiro_provider.credentials;
    let mut vars = Vec::new();

    if let Some(token) = &creds.access_token {
        vars.push(EnvVariable {
            key: "KIRO_ACCESS_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(token) = &creds.refresh_token {
        vars.push(EnvVariable {
            key: "KIRO_REFRESH_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(id) = &creds.client_id {
        vars.push(EnvVariable {
            key: "KIRO_CLIENT_ID".to_string(),
            value: id.clone(),
            masked: mask_token(id),
        });
    }
    if let Some(secret) = &creds.client_secret {
        vars.push(EnvVariable {
            key: "KIRO_CLIENT_SECRET".to_string(),
            value: secret.clone(),
            masked: mask_token(secret),
        });
    }
    if let Some(arn) = &creds.profile_arn {
        vars.push(EnvVariable {
            key: "KIRO_PROFILE_ARN".to_string(),
            value: arn.clone(),
            masked: arn.clone(),
        });
    }
    if let Some(region) = &creds.region {
        vars.push(EnvVariable {
            key: "KIRO_REGION".to_string(),
            value: region.clone(),
            masked: region.clone(),
        });
    }
    if let Some(method) = &creds.auth_method {
        vars.push(EnvVariable {
            key: "KIRO_AUTH_METHOD".to_string(),
            value: method.clone(),
            masked: method.clone(),
        });
    }

    Ok(vars)
}

fn mask_token(token: &str) -> String {
    if token.len() <= 12 {
        "****".to_string()
    } else {
        format!("{}****{}", &token[..6], &token[token.len() - 4..])
    }
}

#[tauri::command]
async fn get_token_file_hash() -> Result<String, String> {
    let path = providers::kiro::KiroProvider::default_creds_path();
    if !path.exists() {
        return Ok("".to_string());
    }

    let content = std::fs::read(&path).map_err(|e| e.to_string())?;
    let hash = format!("{:x}", md5::compute(&content));
    Ok(hash)
}

/// 检查凭证文件变化并自动重新加载（带日志记录）
#[tauri::command]
async fn check_and_reload_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    last_hash: String,
) -> Result<CheckResult, String> {
    let path = providers::kiro::KiroProvider::default_creds_path();

    if !path.exists() {
        return Ok(CheckResult {
            changed: false,
            new_hash: "".to_string(),
            reloaded: false,
        });
    }

    let content = std::fs::read(&path).map_err(|e| e.to_string())?;
    let new_hash = format!("{:x}", md5::compute(&content));

    if !last_hash.is_empty() && new_hash != last_hash {
        logs.write()
            .await
            .add("info", "[自动检测] 凭证文件已变化，正在重新加载...");

        let mut s = state.write().await;
        match s.kiro_provider.load_credentials() {
            Ok(_) => {
                logs.write()
                    .await
                    .add("info", "[自动检测] 凭证重新加载成功");
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: true,
                })
            }
            Err(e) => {
                logs.write()
                    .await
                    .add("error", &format!("[自动检测] 凭证重新加载失败: {e}"));
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: false,
                })
            }
        }
    } else {
        Ok(CheckResult {
            changed: false,
            new_hash,
            reloaded: false,
        })
    }
}

#[derive(serde::Serialize)]
struct CheckResult {
    changed: bool,
    new_hash: String,
    reloaded: bool,
}

// ============ Gemini Provider Commands ============

#[derive(serde::Serialize)]
struct GeminiCredentialStatus {
    loaded: bool,
    has_access_token: bool,
    has_refresh_token: bool,
    expiry_date: Option<i64>,
    is_valid: bool,
    creds_path: String,
}

#[tauri::command]
async fn get_gemini_credentials(
    state: tauri::State<'_, AppState>,
) -> Result<GeminiCredentialStatus, String> {
    let s = state.read().await;
    let creds = &s.gemini_provider.credentials;
    let path = providers::gemini::GeminiProvider::default_creds_path();

    Ok(GeminiCredentialStatus {
        loaded: creds.access_token.is_some() || creds.refresh_token.is_some(),
        has_access_token: creds.access_token.is_some(),
        has_refresh_token: creds.refresh_token.is_some(),
        expiry_date: creds.expiry_date,
        is_valid: s.gemini_provider.is_token_valid(),
        creds_path: path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
async fn reload_gemini_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Gemini] 正在加载凭证...");
    s.gemini_provider
        .load_credentials()
        .map_err(|e| e.to_string())?;
    logs.write().await.add("info", "[Gemini] 凭证加载成功");
    Ok("Gemini credentials reloaded".to_string())
}

#[tauri::command]
async fn refresh_gemini_token(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Gemini] 正在刷新 Token...");
    let result = s
        .gemini_provider
        .refresh_token()
        .await
        .map_err(|e| e.to_string());
    match &result {
        Ok(_) => logs.write().await.add("info", "[Gemini] Token 刷新成功"),
        Err(e) => logs
            .write()
            .await
            .add("error", &format!("[Gemini] Token 刷新失败: {e}")),
    }
    result
}

#[tauri::command]
async fn get_gemini_env_variables(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<EnvVariable>, String> {
    let s = state.read().await;
    let creds = &s.gemini_provider.credentials;
    let mut vars = Vec::new();

    if let Some(token) = &creds.access_token {
        vars.push(EnvVariable {
            key: "GEMINI_ACCESS_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(token) = &creds.refresh_token {
        vars.push(EnvVariable {
            key: "GEMINI_REFRESH_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(expiry) = creds.expiry_date {
        let expiry_str = expiry.to_string();
        vars.push(EnvVariable {
            key: "GEMINI_EXPIRY_DATE".to_string(),
            value: expiry_str.clone(),
            masked: expiry_str,
        });
    }

    Ok(vars)
}

#[tauri::command]
async fn get_gemini_token_file_hash() -> Result<String, String> {
    let path = providers::gemini::GeminiProvider::default_creds_path();
    if !path.exists() {
        return Ok("".to_string());
    }

    let content = std::fs::read(&path).map_err(|e| e.to_string())?;
    let hash = format!("{:x}", md5::compute(&content));
    Ok(hash)
}

#[tauri::command]
async fn check_and_reload_gemini_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    last_hash: String,
) -> Result<CheckResult, String> {
    let path = providers::gemini::GeminiProvider::default_creds_path();

    if !path.exists() {
        return Ok(CheckResult {
            changed: false,
            new_hash: "".to_string(),
            reloaded: false,
        });
    }

    let content = std::fs::read(&path).map_err(|e| e.to_string())?;
    let new_hash = format!("{:x}", md5::compute(&content));

    if !last_hash.is_empty() && new_hash != last_hash {
        logs.write()
            .await
            .add("info", "[Gemini][自动检测] 凭证文件已变化，正在重新加载...");

        let mut s = state.write().await;
        match s.gemini_provider.load_credentials() {
            Ok(_) => {
                logs.write()
                    .await
                    .add("info", "[Gemini][自动检测] 凭证重新加载成功");
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: true,
                })
            }
            Err(e) => {
                logs.write().await.add(
                    "error",
                    &format!("[Gemini][自动检测] 凭证重新加载失败: {e}"),
                );
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: false,
                })
            }
        }
    } else {
        Ok(CheckResult {
            changed: false,
            new_hash,
            reloaded: false,
        })
    }
}

// ============ Qwen Provider Commands ============

#[derive(serde::Serialize)]
struct QwenCredentialStatus {
    loaded: bool,
    has_access_token: bool,
    has_refresh_token: bool,
    expiry_date: Option<i64>,
    is_valid: bool,
    creds_path: String,
}

#[tauri::command]
async fn get_qwen_credentials(
    state: tauri::State<'_, AppState>,
) -> Result<QwenCredentialStatus, String> {
    let s = state.read().await;
    let creds = &s.qwen_provider.credentials;
    let path = providers::qwen::QwenProvider::default_creds_path();

    Ok(QwenCredentialStatus {
        loaded: creds.access_token.is_some() || creds.refresh_token.is_some(),
        has_access_token: creds.access_token.is_some(),
        has_refresh_token: creds.refresh_token.is_some(),
        expiry_date: creds.expiry_date,
        is_valid: s.qwen_provider.is_token_valid(),
        creds_path: path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
async fn reload_qwen_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Qwen] 正在加载凭证...");
    s.qwen_provider
        .load_credentials()
        .map_err(|e| e.to_string())?;
    logs.write().await.add("info", "[Qwen] 凭证加载成功");
    Ok("Qwen credentials reloaded".to_string())
}

#[tauri::command]
async fn refresh_qwen_token(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Qwen] 正在刷新 Token...");
    let result = s
        .qwen_provider
        .refresh_token()
        .await
        .map_err(|e| e.to_string());
    match &result {
        Ok(_) => logs.write().await.add("info", "[Qwen] Token 刷新成功"),
        Err(e) => logs
            .write()
            .await
            .add("error", &format!("[Qwen] Token 刷新失败: {e}")),
    }
    result
}

#[tauri::command]
async fn get_qwen_env_variables(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<EnvVariable>, String> {
    let s = state.read().await;
    let creds = &s.qwen_provider.credentials;
    let mut vars = Vec::new();

    if let Some(token) = &creds.access_token {
        vars.push(EnvVariable {
            key: "QWEN_ACCESS_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(token) = &creds.refresh_token {
        vars.push(EnvVariable {
            key: "QWEN_REFRESH_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(url) = &creds.resource_url {
        vars.push(EnvVariable {
            key: "QWEN_RESOURCE_URL".to_string(),
            value: url.clone(),
            masked: url.clone(),
        });
    }
    if let Some(expiry) = creds.expiry_date {
        let expiry_str = expiry.to_string();
        vars.push(EnvVariable {
            key: "QWEN_EXPIRY_DATE".to_string(),
            value: expiry_str.clone(),
            masked: expiry_str,
        });
    }

    Ok(vars)
}

#[tauri::command]
async fn get_qwen_token_file_hash() -> Result<String, String> {
    let path = providers::qwen::QwenProvider::default_creds_path();
    if !path.exists() {
        return Ok("".to_string());
    }

    let content = std::fs::read(&path).map_err(|e| e.to_string())?;
    let hash = format!("{:x}", md5::compute(&content));
    Ok(hash)
}

#[tauri::command]
async fn check_and_reload_qwen_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    last_hash: String,
) -> Result<CheckResult, String> {
    let path = providers::qwen::QwenProvider::default_creds_path();

    if !path.exists() {
        return Ok(CheckResult {
            changed: false,
            new_hash: "".to_string(),
            reloaded: false,
        });
    }

    let content = std::fs::read(&path).map_err(|e| e.to_string())?;
    let new_hash = format!("{:x}", md5::compute(&content));

    if !last_hash.is_empty() && new_hash != last_hash {
        logs.write()
            .await
            .add("info", "[Qwen][自动检测] 凭证文件已变化，正在重新加载...");

        let mut s = state.write().await;
        match s.qwen_provider.load_credentials() {
            Ok(_) => {
                logs.write()
                    .await
                    .add("info", "[Qwen][自动检测] 凭证重新加载成功");
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: true,
                })
            }
            Err(e) => {
                logs.write()
                    .await
                    .add("error", &format!("[Qwen][自动检测] 凭证重新加载失败: {e}"));
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: false,
                })
            }
        }
    } else {
        Ok(CheckResult {
            changed: false,
            new_hash,
            reloaded: false,
        })
    }
}

// ============ OpenAI Custom Provider Commands ============

#[derive(serde::Serialize, serde::Deserialize)]
struct OpenAICustomStatus {
    enabled: bool,
    has_api_key: bool,
    base_url: String,
}

#[tauri::command]
async fn get_openai_custom_status(
    state: tauri::State<'_, AppState>,
) -> Result<OpenAICustomStatus, String> {
    let s = state.read().await;
    let config = &s.openai_custom_provider.config;
    Ok(OpenAICustomStatus {
        enabled: config.enabled,
        has_api_key: config.api_key.is_some(),
        base_url: s.openai_custom_provider.get_base_url(),
    })
}

#[tauri::command]
async fn set_openai_custom_config(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    api_key: Option<String>,
    base_url: Option<String>,
    enabled: bool,
) -> Result<String, String> {
    let mut s = state.write().await;
    s.openai_custom_provider.config.api_key = api_key;
    s.openai_custom_provider.config.base_url = base_url;
    s.openai_custom_provider.config.enabled = enabled;
    logs.write().await.add(
        "info",
        &format!("[OpenAI Custom] 配置已更新, enabled={enabled}"),
    );
    Ok("OpenAI Custom config updated".to_string())
}

// ============ Claude Custom Provider Commands ============

#[derive(serde::Serialize, serde::Deserialize)]
struct ClaudeCustomStatus {
    enabled: bool,
    has_api_key: bool,
    base_url: String,
}

#[tauri::command]
async fn get_claude_custom_status(
    state: tauri::State<'_, AppState>,
) -> Result<ClaudeCustomStatus, String> {
    let s = state.read().await;
    let config = &s.claude_custom_provider.config;
    Ok(ClaudeCustomStatus {
        enabled: config.enabled,
        has_api_key: config.api_key.is_some(),
        base_url: s.claude_custom_provider.get_base_url(),
    })
}

#[tauri::command]
async fn set_claude_custom_config(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    api_key: Option<String>,
    base_url: Option<String>,
    enabled: bool,
) -> Result<String, String> {
    let mut s = state.write().await;
    s.claude_custom_provider.config.api_key = api_key;
    s.claude_custom_provider.config.base_url = base_url;
    s.claude_custom_provider.config.enabled = enabled;
    logs.write().await.add(
        "info",
        &format!("[Claude Custom] 配置已更新, enabled={enabled}"),
    );
    Ok("Claude Custom config updated".to_string())
}

#[tauri::command]
async fn get_logs(logs: tauri::State<'_, LogState>) -> Result<Vec<logger::LogEntry>, String> {
    Ok(logs.read().await.get_logs())
}

#[tauri::command]
async fn clear_logs(logs: tauri::State<'_, LogState>) -> Result<(), String> {
    logs.write().await.clear();
    Ok(())
}

#[derive(serde::Serialize)]
struct TestResult {
    success: bool,
    status: u16,
    body: String,
    time_ms: u64,
}

#[derive(serde::Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[tauri::command]
async fn get_available_models() -> Result<Vec<ModelInfo>, String> {
    Ok(vec![
        // Kiro/Claude models
        ModelInfo {
            id: "claude-sonnet-4-5".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-sonnet-4-5-20250514".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-sonnet-4-5-20250929".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-3-7-sonnet-20250219".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-3-5-sonnet-latest".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-opus-4-5-20250514".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-haiku-4-5-20250514".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        // Gemini models
        ModelInfo {
            id: "gemini-2.5-flash".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-2.5-flash-lite".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-2.5-pro".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-2.5-pro-preview-06-05".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-3-pro-preview".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        // Qwen models
        ModelInfo {
            id: "qwen3-coder-plus".to_string(),
            object: "model".to_string(),
            owned_by: "alibaba".to_string(),
        },
        ModelInfo {
            id: "qwen3-coder-flash".to_string(),
            object: "model".to_string(),
            owned_by: "alibaba".to_string(),
        },
    ])
}

#[tauri::command]
async fn test_api(
    state: tauri::State<'_, AppState>,
    method: String,
    path: String,
    body: Option<String>,
    auth: bool,
) -> Result<TestResult, String> {
    let s = state.read().await;
    let base_url = format!("http://{}:{}", s.config.server.host, s.config.server.port);
    let api_key = &s.config.server.api_key;

    // 创建一个禁用代理的客户端
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("{base_url}{path}");

    tracing::info!("Testing API: {} {}", method, url);

    let start = std::time::Instant::now();

    let mut req = match method.as_str() {
        "GET" => client.get(&url),
        "POST" => client.post(&url),
        _ => return Err("Unsupported method".to_string()),
    };

    req = req.header("Content-Type", "application/json");

    if auth {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    if let Some(b) = body {
        req = req.body(b);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            let time_ms = start.elapsed().as_millis() as u64;

            tracing::info!(
                "API test result: status={}, body_len={}",
                status,
                body.len()
            );

            Ok(TestResult {
                success: (200..300).contains(&status),
                status,
                body,
                time_ms,
            })
        }
        Err(e) => {
            tracing::error!("API test error: {}", e);
            Err(e.to_string())
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let config = config::load_config().unwrap_or_default();
    let state: AppState = Arc::new(RwLock::new(server::ServerState::new(config)));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .manage(Arc::new(RwLock::new(logger::LogStore::new())))
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            get_server_status,
            get_config,
            save_config,
            get_default_provider,
            set_default_provider,
            // Kiro commands
            refresh_kiro_token,
            reload_credentials,
            get_kiro_credentials,
            get_env_variables,
            get_token_file_hash,
            check_and_reload_credentials,
            // Gemini commands
            get_gemini_credentials,
            reload_gemini_credentials,
            refresh_gemini_token,
            get_gemini_env_variables,
            get_gemini_token_file_hash,
            check_and_reload_gemini_credentials,
            // Qwen commands
            get_qwen_credentials,
            reload_qwen_credentials,
            refresh_qwen_token,
            get_qwen_env_variables,
            get_qwen_token_file_hash,
            check_and_reload_qwen_credentials,
            // OpenAI Custom commands
            get_openai_custom_status,
            set_openai_custom_config,
            // Claude Custom commands
            get_claude_custom_status,
            set_claude_custom_config,
            // Common
            get_logs,
            clear_logs,
            test_api,
            get_available_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
