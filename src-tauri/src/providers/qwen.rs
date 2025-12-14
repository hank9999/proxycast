//! Qwen (通义千问) OAuth Provider
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;

// Constants
const QWEN_DIR: &str = ".qwen";
const CREDENTIALS_FILE: &str = "oauth_creds.json";
const QWEN_BASE_URL: &str = "https://portal.qwen.ai/v1";

pub const QWEN_MODELS: &[&str] = &["qwen3-coder-plus", "qwen3-coder-flash"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenCredentials {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub token_type: Option<String>,
    pub resource_url: Option<String>,
    pub expiry_date: Option<i64>,
}

impl Default for QwenCredentials {
    fn default() -> Self {
        Self {
            access_token: None,
            refresh_token: None,
            token_type: Some("Bearer".to_string()),
            resource_url: None,
            expiry_date: None,
        }
    }
}

pub struct QwenProvider {
    pub credentials: QwenCredentials,
    pub client: Client,
}

impl Default for QwenProvider {
    fn default() -> Self {
        Self {
            credentials: QwenCredentials::default(),
            client: Client::new(),
        }
    }
}

impl QwenProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_creds_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(QWEN_DIR)
            .join(CREDENTIALS_FILE)
    }

    pub fn load_credentials(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();

        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let creds: QwenCredentials = serde_json::from_str(&content)?;
            self.credentials = creds;
        }

        Ok(())
    }

    pub fn save_credentials(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(&self.credentials)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    pub fn is_token_valid(&self) -> bool {
        if self.credentials.access_token.is_none() {
            return false;
        }
        if let Some(expiry) = self.credentials.expiry_date {
            let now = chrono::Utc::now().timestamp_millis();
            // Token valid if more than 30 seconds until expiry
            return expiry > now + 30_000;
        }
        true
    }

    pub fn get_base_url(&self) -> String {
        self.credentials
            .resource_url
            .as_ref()
            .map(|url| {
                let normalized = if url.starts_with("http") {
                    url.clone()
                } else {
                    format!("https://{url}")
                };
                if normalized.ends_with("/v1") {
                    normalized
                } else {
                    format!("{normalized}/v1")
                }
            })
            .unwrap_or_else(|| QWEN_BASE_URL.to_string())
    }

    pub async fn refresh_token(&mut self) -> Result<String, Box<dyn Error + Send + Sync>> {
        let refresh_token = self
            .credentials
            .refresh_token
            .as_ref()
            .ok_or("No refresh token available")?;

        let client_id = std::env::var("QWEN_OAUTH_CLIENT_ID")
            .ok()
            .or_else(|| Some("f0304373b74a44d2b584a3fb70ca9e56".to_string()))
            .unwrap();

        let body = serde_json::json!({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id
        });

        let resp = self
            .client
            .post("https://chat.qwen.ai/api/v1/oauth2/token")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Token refresh failed: {status} - {body}").into());
        }

        let data: serde_json::Value = resp.json().await?;

        let new_token = data["access_token"]
            .as_str()
            .ok_or("No access token in response")?;

        self.credentials.access_token = Some(new_token.to_string());

        if let Some(rt) = data["refresh_token"].as_str() {
            self.credentials.refresh_token = Some(rt.to_string());
        }

        if let Some(resource_url) = data["resource_url"].as_str() {
            self.credentials.resource_url = Some(resource_url.to_string());
        }

        if let Some(expires_in) = data["expires_in"].as_i64() {
            self.credentials.expiry_date =
                Some(chrono::Utc::now().timestamp_millis() + expires_in * 1000);
        }

        // Save refreshed credentials
        self.save_credentials()?;

        Ok(new_token.to_string())
    }

    pub async fn chat_completions(
        &self,
        request: &serde_json::Value,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
        let token = self
            .credentials
            .access_token
            .as_ref()
            .ok_or("No access token")?;

        let base_url = self.get_base_url();
        let url = format!("{base_url}/chat/completions");

        // Ensure model is valid
        let mut req_body = request.clone();
        if let Some(model) = req_body.get("model").and_then(|m| m.as_str()) {
            if !QWEN_MODELS.contains(&model) {
                req_body["model"] = serde_json::json!(QWEN_MODELS[0]);
            }
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("X-DashScope-AuthType", "qwen-oauth")
            .json(&req_body)
            .send()
            .await?;

        Ok(resp)
    }
}
