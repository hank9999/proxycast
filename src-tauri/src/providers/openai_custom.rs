//! OpenAI Custom Provider (自定义 OpenAI 兼容 API)
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAICustomConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub enabled: bool,
}

pub struct OpenAICustomProvider {
    pub config: OpenAICustomConfig,
    pub client: Client,
}

impl Default for OpenAICustomProvider {
    fn default() -> Self {
        Self {
            config: OpenAICustomConfig::default(),
            client: Client::new(),
        }
    }
}

impl OpenAICustomProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_base_url(&self) -> String {
        self.config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
    }

    pub fn is_configured(&self) -> bool {
        self.config.api_key.is_some() && self.config.enabled
    }

    pub async fn chat_completions(
        &self,
        request: &serde_json::Value,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or("OpenAI API key not configured")?;

        let base_url = self.get_base_url();
        let url = format!("{base_url}/chat/completions");

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;

        Ok(resp)
    }

    pub async fn list_models(&self) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or("OpenAI API key not configured")?;

        let base_url = self.get_base_url();
        let url = format!("{base_url}/models");

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Failed to list models: {status} - {body}").into());
        }

        let data: serde_json::Value = resp.json().await?;
        Ok(data)
    }
}
