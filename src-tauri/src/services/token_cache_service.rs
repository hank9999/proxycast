//! Token 缓存管理服务
//!
//! 负责管理凭证池中 OAuth Token 的生命周期：
//! - 从源文件加载初始 Token
//! - 缓存刷新后的 Token 到数据库
//! - 按需刷新即将过期的 Token
//! - 处理 401/403 错误时的强制刷新

use crate::database::dao::provider_pool::ProviderPoolDao;
use crate::database::DbConnection;
use crate::models::provider_pool_model::{
    CachedTokenInfo, CredentialData, PoolProviderType, ProviderCredential,
};
use crate::providers::gemini::GeminiProvider;
use crate::providers::kiro::KiroProvider;
use crate::providers::qwen::QwenProvider;
use chrono::Utc;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Token 缓存服务
pub struct TokenCacheService {
    /// 每凭证一把锁，防止并发刷新
    locks: DashMap<String, Arc<Mutex<()>>>,
}

impl Default for TokenCacheService {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCacheService {
    pub fn new() -> Self {
        Self {
            locks: DashMap::new(),
        }
    }

    /// 获取有效的 Token（核心方法）
    ///
    /// 1. 检查数据库缓存是否有效
    /// 2. 如果缓存有效且未过期，直接返回
    /// 3. 如果缓存无效或即将过期，执行刷新
    /// 4. 如果刷新失败（如 refreshToken 被截断），尝试使用源文件中的 accessToken
    pub async fn get_valid_token(&self, db: &DbConnection, uuid: &str) -> Result<String, String> {
        // 首先检查缓存
        let cached = {
            let conn = db.lock().map_err(|e| e.to_string())?;
            ProviderPoolDao::get_token_cache(&conn, uuid).map_err(|e| e.to_string())?
        };

        // 缓存有效且未即将过期，直接返回
        if let Some(ref cache) = cached {
            if cache.is_valid() && !cache.is_expiring_soon() {
                if let Some(token) = &cache.access_token {
                    tracing::debug!(
                        "[TOKEN_CACHE] Using cached token for {}, expires at {:?}",
                        &uuid[..8],
                        cache.expiry_time
                    );
                    return Ok(token.clone());
                }
            }
        }

        // 需要刷新（无缓存、已过期或即将过期）
        match self.refresh_and_cache(db, uuid, false).await {
            Ok(token) => Ok(token),
            Err(refresh_error) => {
                // 刷新失败时，检查是否是因为 refreshToken 被截断
                // 如果是，尝试直接使用源文件中的 accessToken（可能仍然有效）
                if refresh_error.contains("截断") || refresh_error.contains("truncated") {
                    tracing::warn!(
                        "[TOKEN_CACHE] refreshToken 被截断，尝试使用源文件中的 accessToken: {}",
                        &uuid[..8]
                    );

                    // 获取凭证信息
                    let credential = {
                        let conn = db.lock().map_err(|e| e.to_string())?;
                        ProviderPoolDao::get_by_uuid(&conn, uuid)
                            .map_err(|e| e.to_string())?
                            .ok_or_else(|| format!("Credential not found: {}", uuid))?
                    };

                    // 尝试从源文件读取 accessToken
                    match self.read_token_from_source(&credential).await {
                        Ok(token_info) => {
                            if let Some(token) = token_info.access_token {
                                tracing::info!(
                                    "[TOKEN_CACHE] 使用源文件中的 accessToken（可能已过期）: {}",
                                    &uuid[..8]
                                );
                                // 注意：这个 token 可能已过期，但至少可以尝试使用
                                // 缓存这个 token（但不设置过期时间，因为我们不知道它何时过期）
                                let cache_info = CachedTokenInfo {
                                    access_token: Some(token.clone()),
                                    refresh_token: token_info.refresh_token,
                                    expiry_time: None, // 不知道过期时间
                                    last_refresh: Some(Utc::now()),
                                    refresh_error_count: 1,
                                    last_refresh_error: Some(format!(
                                        "refreshToken 被截断，使用源文件 accessToken: {}",
                                        refresh_error
                                    )),
                                };

                                // 缓存到数据库
                                if let Ok(conn) = db.lock() {
                                    let _ = ProviderPoolDao::update_token_cache(
                                        &conn,
                                        uuid,
                                        &cache_info,
                                    );
                                }

                                return Ok(token);
                            }
                        }
                        Err(e) => {
                            tracing::error!("[TOKEN_CACHE] 无法从源文件读取 accessToken: {}", e);
                        }
                    }
                }

                // 返回原始刷新错误
                Err(refresh_error)
            }
        }
    }

    /// 刷新 Token 并缓存到数据库
    ///
    /// - force: 是否强制刷新（忽略缓存状态）
    pub async fn refresh_and_cache(
        &self,
        db: &DbConnection,
        uuid: &str,
        force: bool,
    ) -> Result<String, String> {
        // 获取该凭证的锁
        let lock = self
            .locks
            .entry(uuid.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        let _guard = lock.lock().await;

        // 双重检查：可能其他线程已完成刷新
        if !force {
            let cached = {
                let conn = db.lock().map_err(|e| e.to_string())?;
                ProviderPoolDao::get_token_cache(&conn, uuid).map_err(|e| e.to_string())?
            };

            if let Some(cache) = cached {
                if cache.is_valid() && !cache.is_expiring_soon() {
                    if let Some(token) = cache.access_token {
                        tracing::debug!(
                            "[TOKEN_CACHE] Double-check: another thread refreshed for {}",
                            &uuid[..8]
                        );
                        return Ok(token);
                    }
                }
            }
        }

        // 获取凭证信息
        let credential = {
            let conn = db.lock().map_err(|e| e.to_string())?;
            ProviderPoolDao::get_by_uuid(&conn, uuid)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("Credential not found: {}", uuid))?
        };

        tracing::info!(
            "[TOKEN_CACHE] Refreshing token for {} ({})",
            &uuid[..8],
            credential.provider_type
        );

        // 执行刷新
        match self.do_refresh(&credential).await {
            Ok(token_info) => {
                // 缓存到数据库
                {
                    let conn = db.lock().map_err(|e| e.to_string())?;
                    ProviderPoolDao::update_token_cache(&conn, uuid, &token_info)
                        .map_err(|e| e.to_string())?;
                }

                let token = token_info
                    .access_token
                    .ok_or_else(|| "Refresh succeeded but no access_token".to_string())?;

                tracing::info!(
                    "[TOKEN_CACHE] Token refreshed and cached for {}, expires at {:?}",
                    &uuid[..8],
                    token_info.expiry_time
                );

                Ok(token)
            }
            Err(e) => {
                // 记录刷新错误
                {
                    let conn = db.lock().map_err(|e| e.to_string())?;
                    let _ = ProviderPoolDao::record_token_refresh_error(&conn, uuid, &e);
                }

                tracing::error!(
                    "[TOKEN_CACHE] Token refresh failed for {}: {}",
                    &uuid[..8],
                    e
                );

                Err(e)
            }
        }
    }

    /// 执行实际的 Token 刷新
    async fn do_refresh(&self, credential: &ProviderCredential) -> Result<CachedTokenInfo, String> {
        match &credential.credential {
            CredentialData::KiroOAuth { creds_file_path } => {
                self.refresh_kiro(creds_file_path).await
            }
            CredentialData::GeminiOAuth {
                creds_file_path, ..
            } => self.refresh_gemini(creds_file_path).await,
            CredentialData::QwenOAuth { creds_file_path } => {
                self.refresh_qwen(creds_file_path).await
            }
            CredentialData::AntigravityOAuth {
                creds_file_path, ..
            } => self.refresh_antigravity(creds_file_path).await,
            CredentialData::OpenAIKey { api_key, .. } => {
                // API Key 不需要刷新，直接返回
                Ok(CachedTokenInfo {
                    access_token: Some(api_key.clone()),
                    refresh_token: None,
                    expiry_time: None, // 永不过期
                    last_refresh: Some(Utc::now()),
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::ClaudeKey { api_key, .. } => {
                // API Key 不需要刷新，直接返回
                Ok(CachedTokenInfo {
                    access_token: Some(api_key.clone()),
                    refresh_token: None,
                    expiry_time: None, // 永不过期
                    last_refresh: Some(Utc::now()),
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::VertexKey { api_key, .. } => {
                // API Key 不需要刷新，直接返回
                Ok(CachedTokenInfo {
                    access_token: Some(api_key.clone()),
                    refresh_token: None,
                    expiry_time: None, // 永不过期
                    last_refresh: Some(Utc::now()),
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::GeminiApiKey { api_key, .. } => {
                // API Key 不需要刷新，直接返回
                Ok(CachedTokenInfo {
                    access_token: Some(api_key.clone()),
                    refresh_token: None,
                    expiry_time: None, // 永不过期
                    last_refresh: Some(Utc::now()),
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::CodexOAuth { creds_file_path } => {
                self.refresh_codex(creds_file_path).await
            }
            CredentialData::ClaudeOAuth { creds_file_path } => {
                self.refresh_claude_oauth(creds_file_path).await
            }
            CredentialData::IFlowOAuth { creds_file_path } => {
                self.refresh_iflow_oauth(creds_file_path).await
            }
            CredentialData::IFlowCookie { creds_file_path } => {
                self.refresh_iflow_cookie(creds_file_path).await
            }
        }
    }

    /// 刷新 Kiro Token
    async fn refresh_kiro(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        let mut provider = KiroProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 Kiro 凭证失败: {}", e))?;

        let token = provider
            .refresh_token()
            .await
            .map_err(|e| format!("刷新 Kiro Token 失败: {}", e))?;

        // Kiro token 通常 1 小时过期，我们假设 50 分钟
        let expiry_time = Utc::now() + chrono::Duration::minutes(50);

        Ok(CachedTokenInfo {
            access_token: Some(token),
            refresh_token: provider.credentials.refresh_token.clone(),
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 刷新 Gemini Token
    async fn refresh_gemini(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        let mut provider = GeminiProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 Gemini 凭证失败: {}", e))?;

        let token = provider
            .refresh_token()
            .await
            .map_err(|e| format!("刷新 Gemini Token 失败: {}", e))?;

        // Gemini token 通常 1 小时过期
        let expiry_time = provider
            .credentials
            .expiry_date
            .map(|ts| chrono::DateTime::from_timestamp(ts, 0).unwrap_or_default())
            .unwrap_or_else(|| Utc::now() + chrono::Duration::minutes(50));

        Ok(CachedTokenInfo {
            access_token: Some(token),
            refresh_token: provider.credentials.refresh_token.clone(),
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 刷新 Qwen Token
    async fn refresh_qwen(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        let mut provider = QwenProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 Qwen 凭证失败: {}", e))?;

        let token = provider
            .refresh_token()
            .await
            .map_err(|e| format!("刷新 Qwen Token 失败: {}", e))?;

        // Qwen token 通常 1 小时过期
        let expiry_time = provider
            .credentials
            .expiry_date
            .map(|ts| chrono::DateTime::from_timestamp(ts, 0).unwrap_or_default())
            .unwrap_or_else(|| Utc::now() + chrono::Duration::minutes(50));

        Ok(CachedTokenInfo {
            access_token: Some(token),
            refresh_token: provider.credentials.refresh_token.clone(),
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 刷新 Antigravity Token
    async fn refresh_antigravity(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        use crate::providers::antigravity::AntigravityProvider;

        let mut provider = AntigravityProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 Antigravity 凭证失败: {}", e))?;

        let token = provider
            .refresh_token()
            .await
            .map_err(|e| format!("刷新 Antigravity Token 失败: {}", e))?;

        // Antigravity token 通常 1 小时过期
        let expiry_time = provider
            .credentials
            .expiry_date
            .map(|ts| chrono::DateTime::from_timestamp_millis(ts).unwrap_or_default())
            .unwrap_or_else(|| Utc::now() + chrono::Duration::minutes(50));

        Ok(CachedTokenInfo {
            access_token: Some(token),
            refresh_token: provider.credentials.refresh_token.clone(),
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 刷新 Codex Token
    async fn refresh_codex(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        use crate::providers::codex::CodexProvider;

        let mut provider = CodexProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 Codex 凭证失败: {}", e))?;

        let token = provider
            .refresh_token_with_retry(3)
            .await
            .map_err(|e| format!("刷新 Codex Token 失败: {}", e))?;

        // 解析过期时间
        let expiry_time = provider
            .credentials
            .expires_at
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|| Utc::now() + chrono::Duration::minutes(50));

        Ok(CachedTokenInfo {
            access_token: Some(token),
            refresh_token: provider.credentials.refresh_token.clone(),
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 刷新 Claude OAuth Token
    async fn refresh_claude_oauth(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        use crate::providers::claude_oauth::ClaudeOAuthProvider;

        let mut provider = ClaudeOAuthProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 Claude OAuth 凭证失败: {}", e))?;

        let token = provider
            .refresh_token_with_retry(3)
            .await
            .map_err(|e| format!("刷新 Claude OAuth Token 失败: {}", e))?;

        // 解析过期时间
        let expiry_time = provider
            .credentials
            .expire
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|| Utc::now() + chrono::Duration::minutes(50));

        Ok(CachedTokenInfo {
            access_token: Some(token),
            refresh_token: provider.credentials.refresh_token.clone(),
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 刷新 iFlow OAuth Token
    async fn refresh_iflow_oauth(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        use crate::providers::iflow::IFlowProvider;

        let mut provider = IFlowProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 iFlow OAuth 凭证失败: {}", e))?;

        let token = provider
            .refresh_token_with_retry(3)
            .await
            .map_err(|e| format!("刷新 iFlow OAuth Token 失败: {}", e))?;

        // 解析过期时间
        let expiry_time = provider
            .credentials
            .expire
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|| Utc::now() + chrono::Duration::minutes(50));

        Ok(CachedTokenInfo {
            access_token: Some(token),
            refresh_token: provider.credentials.refresh_token.clone(),
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 刷新 iFlow Cookie Token
    async fn refresh_iflow_cookie(&self, creds_path: &str) -> Result<CachedTokenInfo, String> {
        use crate::providers::iflow::IFlowProvider;

        let mut provider = IFlowProvider::new();
        provider
            .load_credentials_from_path(creds_path)
            .await
            .map_err(|e| format!("加载 iFlow Cookie 凭证失败: {}", e))?;

        // iFlow Cookie 凭证使用 API Key，不需要刷新 OAuth Token
        // 直接从凭证中获取 API Key
        let api_key = provider
            .credentials
            .api_key
            .clone()
            .ok_or_else(|| "iFlow Cookie 凭证中没有 API Key".to_string())?;

        // 解析过期时间
        let expiry_time = provider
            .credentials
            .expire
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|| Utc::now() + chrono::Duration::days(30)); // Cookie 通常有效期较长

        Ok(CachedTokenInfo {
            access_token: Some(api_key),
            refresh_token: None,
            expiry_time: Some(expiry_time),
            last_refresh: Some(Utc::now()),
            refresh_error_count: 0,
            last_refresh_error: None,
        })
    }

    /// 从源文件加载初始 Token（首次使用时）
    pub async fn load_initial_token(
        &self,
        db: &DbConnection,
        uuid: &str,
    ) -> Result<String, String> {
        let credential = {
            let conn = db.lock().map_err(|e| e.to_string())?;
            ProviderPoolDao::get_by_uuid(&conn, uuid)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("Credential not found: {}", uuid))?
        };

        // 尝试从源文件读取 token
        let token_info = self.read_token_from_source(&credential).await?;

        // 缓存到数据库
        {
            let conn = db.lock().map_err(|e| e.to_string())?;
            ProviderPoolDao::update_token_cache(&conn, uuid, &token_info)
                .map_err(|e| e.to_string())?;
        }

        token_info
            .access_token
            .ok_or_else(|| "源文件中没有 access_token".to_string())
    }

    /// 从源文件读取 Token（不刷新）
    async fn read_token_from_source(
        &self,
        credential: &ProviderCredential,
    ) -> Result<CachedTokenInfo, String> {
        match &credential.credential {
            CredentialData::KiroOAuth { creds_file_path } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 Kiro 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let access_token = creds["accessToken"]
                    .as_str()
                    .or_else(|| creds["access_token"].as_str())
                    .map(|s| s.to_string());
                let refresh_token = creds["refreshToken"]
                    .as_str()
                    .or_else(|| creds["refresh_token"].as_str())
                    .map(|s| s.to_string());

                Ok(CachedTokenInfo {
                    access_token,
                    refresh_token,
                    expiry_time: None, // Kiro 源文件通常没有过期时间
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::GeminiOAuth {
                creds_file_path, ..
            } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 Gemini 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let access_token = creds["access_token"].as_str().map(|s| s.to_string());
                let refresh_token = creds["refresh_token"].as_str().map(|s| s.to_string());
                let expiry_time = creds["expiry_date"]
                    .as_i64()
                    .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0));

                Ok(CachedTokenInfo {
                    access_token,
                    refresh_token,
                    expiry_time,
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::QwenOAuth { creds_file_path } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 Qwen 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let access_token = creds["access_token"].as_str().map(|s| s.to_string());
                let refresh_token = creds["refresh_token"].as_str().map(|s| s.to_string());
                let expiry_time = creds["expiry_date"]
                    .as_i64()
                    .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0));

                Ok(CachedTokenInfo {
                    access_token,
                    refresh_token,
                    expiry_time,
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::AntigravityOAuth {
                creds_file_path, ..
            } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 Antigravity 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let access_token = creds["access_token"].as_str().map(|s| s.to_string());
                let refresh_token = creds["refresh_token"].as_str().map(|s| s.to_string());
                let expiry_time = creds["expiry_date"]
                    .as_i64()
                    .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0));

                Ok(CachedTokenInfo {
                    access_token,
                    refresh_token,
                    expiry_time,
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::OpenAIKey { api_key, .. } => Ok(CachedTokenInfo {
                access_token: Some(api_key.clone()),
                refresh_token: None,
                expiry_time: None,
                last_refresh: None,
                refresh_error_count: 0,
                last_refresh_error: None,
            }),
            CredentialData::ClaudeKey { api_key, .. } => Ok(CachedTokenInfo {
                access_token: Some(api_key.clone()),
                refresh_token: None,
                expiry_time: None,
                last_refresh: None,
                refresh_error_count: 0,
                last_refresh_error: None,
            }),
            CredentialData::VertexKey { api_key, .. } => Ok(CachedTokenInfo {
                access_token: Some(api_key.clone()),
                refresh_token: None,
                expiry_time: None,
                last_refresh: None,
                refresh_error_count: 0,
                last_refresh_error: None,
            }),
            CredentialData::GeminiApiKey { api_key, .. } => Ok(CachedTokenInfo {
                access_token: Some(api_key.clone()),
                refresh_token: None,
                expiry_time: None,
                last_refresh: None,
                refresh_error_count: 0,
                last_refresh_error: None,
            }),
            CredentialData::CodexOAuth { creds_file_path } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 Codex 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let access_token = creds["access_token"].as_str().map(|s| s.to_string());
                let refresh_token = creds["refresh_token"].as_str().map(|s| s.to_string());
                let expiry_time = creds["expired"]
                    .as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));

                Ok(CachedTokenInfo {
                    access_token,
                    refresh_token,
                    expiry_time,
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::ClaudeOAuth { creds_file_path } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 Claude OAuth 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let access_token = creds["access_token"].as_str().map(|s| s.to_string());
                let refresh_token = creds["refresh_token"].as_str().map(|s| s.to_string());
                let expiry_time = creds["expire"]
                    .as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));

                Ok(CachedTokenInfo {
                    access_token,
                    refresh_token,
                    expiry_time,
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::IFlowOAuth { creds_file_path } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 iFlow OAuth 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let access_token = creds["access_token"].as_str().map(|s| s.to_string());
                let refresh_token = creds["refresh_token"].as_str().map(|s| s.to_string());
                let expiry_time = creds["expire"]
                    .as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));

                Ok(CachedTokenInfo {
                    access_token,
                    refresh_token,
                    expiry_time,
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
            CredentialData::IFlowCookie { creds_file_path } => {
                let content = tokio::fs::read_to_string(creds_file_path)
                    .await
                    .map_err(|e| format!("读取 iFlow Cookie 凭证文件失败: {}", e))?;
                let creds: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| format!("解析凭证失败: {}", e))?;

                let api_key = creds["api_key"].as_str().map(|s| s.to_string());
                let expiry_time = creds["expire"]
                    .as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));

                Ok(CachedTokenInfo {
                    access_token: api_key,
                    refresh_token: None,
                    expiry_time,
                    last_refresh: None,
                    refresh_error_count: 0,
                    last_refresh_error: None,
                })
            }
        }
    }

    /// 清除凭证的 Token 缓存
    pub fn clear_cache(&self, db: &DbConnection, uuid: &str) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        ProviderPoolDao::clear_token_cache(&conn, uuid).map_err(|e| e.to_string())
    }

    /// 检查凭证类型是否支持 Token 刷新
    pub fn supports_refresh(provider_type: PoolProviderType) -> bool {
        matches!(
            provider_type,
            PoolProviderType::Kiro | PoolProviderType::Gemini | PoolProviderType::Qwen
        )
    }

    /// 获取凭证的缓存状态
    pub fn get_cache_status(
        &self,
        db: &DbConnection,
        uuid: &str,
    ) -> Result<Option<CachedTokenInfo>, String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        ProviderPoolDao::get_token_cache(&conn, uuid).map_err(|e| e.to_string())
    }
}
