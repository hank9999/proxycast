//! Gemini CLI OAuth Provider
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;

// Constants
const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const CODE_ASSIST_API_VERSION: &str = "v1internal";
const CREDENTIALS_DIR: &str = ".gemini";
const CREDENTIALS_FILE: &str = "oauth_creds.json";

// OAuth credentials - loaded from environment variables
// Set GEMINI_OAUTH_CLIENT_ID and GEMINI_OAUTH_CLIENT_SECRET
// These are the same as Gemini CLI uses (public OAuth app credentials)
fn get_oauth_client_id() -> Option<String> {
    std::env::var("GEMINI_OAUTH_CLIENT_ID").ok()
}

fn get_oauth_client_secret() -> Option<String> {
    std::env::var("GEMINI_OAUTH_CLIENT_SECRET").ok()
}

#[allow(dead_code)]
pub const GEMINI_MODELS: &[&str] = &[
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-3-pro-preview",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiCredentials {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub token_type: Option<String>,
    pub expiry_date: Option<i64>,
    pub scope: Option<String>,
}

impl Default for GeminiCredentials {
    fn default() -> Self {
        Self {
            access_token: None,
            refresh_token: None,
            token_type: Some("Bearer".to_string()),
            expiry_date: None,
            scope: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub model: String,
    pub project: String,
    pub request: GeminiRequestBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequestBody {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCandidate {
    pub content: Option<GeminiContent>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiUsageMetadata {
    pub prompt_token_count: Option<i32>,
    pub candidates_token_count: Option<i32>,
    pub total_token_count: Option<i32>,
}

pub struct GeminiProvider {
    pub credentials: GeminiCredentials,
    pub project_id: Option<String>,
    pub client: Client,
}

impl Default for GeminiProvider {
    fn default() -> Self {
        Self {
            credentials: GeminiCredentials::default(),
            project_id: None,
            client: Client::new(),
        }
    }
}

impl GeminiProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_creds_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CREDENTIALS_DIR)
            .join(CREDENTIALS_FILE)
    }

    pub fn load_credentials(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();

        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let creds: GeminiCredentials = serde_json::from_str(&content)?;
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
            // Token valid if more than 5 minutes until expiry
            return expiry > now + 300_000;
        }
        true
    }

    pub async fn refresh_token(&mut self) -> Result<String, Box<dyn Error + Send + Sync>> {
        let refresh_token = self
            .credentials
            .refresh_token
            .as_ref()
            .ok_or("No refresh token available")?;

        let client_id = get_oauth_client_id().ok_or("GEMINI_OAUTH_CLIENT_ID not set")?;
        let client_secret =
            get_oauth_client_secret().ok_or("GEMINI_OAUTH_CLIENT_SECRET not set")?;

        let params = [
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("refresh_token", refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];

        let resp = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&params)
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

        if let Some(expires_in) = data["expires_in"].as_i64() {
            self.credentials.expiry_date =
                Some(chrono::Utc::now().timestamp_millis() + expires_in * 1000);
        }

        // Save refreshed credentials
        self.save_credentials()?;

        Ok(new_token.to_string())
    }

    pub fn get_api_url(&self, action: &str) -> String {
        format!("{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}:{action}")
    }

    pub async fn call_api(
        &self,
        action: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let token = self
            .credentials
            .access_token
            .as_ref()
            .ok_or("No access token")?;

        let url = self.get_api_url(action);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("API call failed: {status} - {body}").into());
        }

        let data: serde_json::Value = resp.json().await?;
        Ok(data)
    }

    pub async fn discover_project(&mut self) -> Result<String, Box<dyn Error + Send + Sync>> {
        if let Some(ref project_id) = self.project_id {
            return Ok(project_id.clone());
        }

        let body = serde_json::json!({
            "cloudaicompanionProject": "",
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
                "duetProject": ""
            }
        });

        let resp = self.call_api("loadCodeAssist", &body).await?;

        if let Some(project) = resp["cloudaicompanionProject"].as_str() {
            if !project.is_empty() {
                self.project_id = Some(project.to_string());
                return Ok(project.to_string());
            }
        }

        // Need to onboard
        let onboard_body = serde_json::json!({
            "tierId": "free-tier",
            "cloudaicompanionProject": "",
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
                "duetProject": ""
            }
        });

        let mut lro_resp = self.call_api("onboardUser", &onboard_body).await?;

        // Poll until done
        for _ in 0..30 {
            if lro_resp["done"].as_bool().unwrap_or(false) {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            lro_resp = self.call_api("onboardUser", &onboard_body).await?;
        }

        let project_id = lro_resp["response"]["cloudaicompanionProject"]["id"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if project_id.is_empty() {
            return Err("Failed to discover project ID".into());
        }

        self.project_id = Some(project_id.clone());
        Ok(project_id)
    }
}
