//! Kiro/CodeWhisperer Provider
use crate::converter::openai_to_cw::convert_openai_to_codewhisperer;
use crate::models::openai::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;

/// 生成设备指纹 (Machine ID 的 SHA256)
///
/// 与 Kiro IDE 保持一致的指纹生成方式（参考 Kir-Manager）：
/// - macOS: 使用 IOPlatformUUID（硬件级别唯一标识）
/// - Linux: 使用 /etc/machine-id
/// - Windows: 使用 WMI 获取系统 UUID
///
/// 最终返回 SHA256 哈希后的 64 字符十六进制字符串
fn get_device_fingerprint() -> String {
    use sha2::{Digest, Sha256};

    let raw_id =
        get_raw_machine_id().unwrap_or_else(|| "00000000-0000-0000-0000-000000000000".to_string());

    // 使用 SHA256 生成 64 字符的十六进制指纹
    let mut hasher = Sha256::new();
    hasher.update(raw_id.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// 获取原始 Machine ID（未哈希）
fn get_raw_machine_id() -> Option<String> {
    use std::process::Command;

    if cfg!(target_os = "macos") {
        // macOS: 使用 ioreg 获取 IOPlatformUUID
        Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .find(|l| l.contains("IOPlatformUUID"))
                    .and_then(|l| l.split('=').nth(1))
                    .map(|s| s.trim().trim_matches('"').to_lowercase())
            })
    } else if cfg!(target_os = "linux") {
        // Linux: 读取 /etc/machine-id 或 /var/lib/dbus/machine-id
        std::fs::read_to_string("/etc/machine-id")
            .or_else(|_| std::fs::read_to_string("/var/lib/dbus/machine-id"))
            .ok()
            .map(|s| s.trim().to_lowercase())
    } else if cfg!(target_os = "windows") {
        // Windows: 使用 wmic 获取系统 UUID
        Command::new("wmic")
            .args(["csproduct", "get", "UUID"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .skip(1) // 跳过表头
                    .find(|l| !l.trim().is_empty())
                    .map(|s| s.trim().to_lowercase())
            })
    } else {
        None
    }
}

/// 获取 Kiro IDE 版本号
///
/// 尝试从 Kiro.app 的 Info.plist 读取实际版本，失败时使用默认值
fn get_kiro_version() -> String {
    use std::process::Command;

    if cfg!(target_os = "macos") {
        // 尝试从 Kiro.app 读取版本
        let kiro_paths = [
            "/Applications/Kiro.app/Contents/Info.plist",
            // 用户目录下的安装
            &format!(
                "{}/Applications/Kiro.app/Contents/Info.plist",
                dirs::home_dir()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default()
            ),
        ];

        for plist_path in &kiro_paths {
            if let Ok(output) = Command::new("defaults")
                .args(["read", plist_path, "CFBundleShortVersionString"])
                .output()
            {
                if let Ok(version) = String::from_utf8(output.stdout) {
                    let version = version.trim();
                    if !version.is_empty() {
                        return version.to_string();
                    }
                }
            }
        }
    }

    // 默认版本号
    "0.1.25".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KiroCredentials {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub profile_arn: Option<String>,
    /// 过期时间（支持 RFC3339 格式和时间戳格式）
    pub expires_at: Option<String>,
    /// 过期时间（RFC3339 格式）- 与 CLIProxyAPI 兼容
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expire: Option<String>,
    pub region: Option<String>,
    pub auth_method: Option<String>,
    pub client_id_hash: Option<String>,
    /// 最后刷新时间（RFC3339 格式）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_refresh: Option<String>,
    /// 凭证类型标识
    #[serde(default = "default_kiro_type", rename = "type")]
    pub cred_type: String,
}

fn default_kiro_type() -> String {
    "kiro".to_string()
}

impl Default for KiroCredentials {
    fn default() -> Self {
        Self {
            access_token: None,
            refresh_token: None,
            client_id: None,
            client_secret: None,
            profile_arn: None,
            expires_at: None,
            expire: None,
            region: Some("us-east-1".to_string()),
            auth_method: Some("social".to_string()),
            client_id_hash: None,
            last_refresh: None,
            cred_type: default_kiro_type(),
        }
    }
}

pub struct KiroProvider {
    pub credentials: KiroCredentials,
    pub client: Client,
    /// 当前加载的凭证文件路径
    pub creds_path: Option<PathBuf>,
}

impl Default for KiroProvider {
    fn default() -> Self {
        Self {
            credentials: KiroCredentials::default(),
            client: Client::new(),
            creds_path: None,
        }
    }
}

impl KiroProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_creds_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".aws")
            .join("sso")
            .join("cache")
            .join("kiro-auth-token.json")
    }

    pub async fn load_credentials(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();
        let dir = path.parent().ok_or("Invalid path: no parent directory")?;

        let mut merged = KiroCredentials::default();

        // 读取主凭证文件
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            let content = tokio::fs::read_to_string(&path).await?;
            let creds: KiroCredentials = serde_json::from_str(&content)?;
            tracing::info!(
                "[KIRO] Main file loaded: has_access={}, has_refresh={}, has_client_id={}, auth_method={:?}",
                creds.access_token.is_some(),
                creds.refresh_token.is_some(),
                creds.client_id.is_some(),
                creds.auth_method
            );
            merge_credentials(&mut merged, &creds);
        }

        // 如果有 clientIdHash，尝试加载对应的 client_id 和 client_secret
        if let Some(hash) = &merged.client_id_hash {
            let hash_file_path = dir.join(format!("{}.json", hash));
            tracing::info!(
                "[KIRO] 检查 clientIdHash 文件: {}",
                hash_file_path.display()
            );
            if tokio::fs::try_exists(&hash_file_path)
                .await
                .unwrap_or(false)
            {
                if let Ok(content) = tokio::fs::read_to_string(&hash_file_path).await {
                    if let Ok(creds) = serde_json::from_str::<KiroCredentials>(&content) {
                        tracing::info!(
                            "[KIRO] Hash file {:?}: has_client_id={}, has_client_secret={}",
                            hash_file_path.file_name(),
                            creds.client_id.is_some(),
                            creds.client_secret.is_some()
                        );
                        merge_credentials(&mut merged, &creds);
                    } else {
                        tracing::error!(
                            "[KIRO] 无法解析 clientIdHash 文件: {}",
                            hash_file_path.display()
                        );
                    }
                } else {
                    tracing::error!(
                        "[KIRO] 无法读取 clientIdHash 文件: {}",
                        hash_file_path.display()
                    );
                }
            } else {
                tracing::warn!(
                    "[KIRO] clientIdHash {} 指向的文件不存在: {}",
                    hash,
                    hash_file_path.display()
                );
            }
        } else {
            tracing::info!("[KIRO] 没有 clientIdHash 字段");
        }

        // 读取目录中其他 JSON 文件
        if tokio::fs::try_exists(dir).await.unwrap_or(false) {
            let mut entries = tokio::fs::read_dir(dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let file_path = entry.path();
                if file_path.extension().map(|e| e == "json").unwrap_or(false) && file_path != path
                {
                    if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                        if let Ok(creds) = serde_json::from_str::<KiroCredentials>(&content) {
                            tracing::info!(
                                "[KIRO] Extra file {:?}: has_client_id={}, has_client_secret={}",
                                file_path.file_name(),
                                creds.client_id.is_some(),
                                creds.client_secret.is_some()
                            );
                            merge_credentials(&mut merged, &creds);
                        }
                    }
                }
            }
        }

        tracing::info!(
            "[KIRO] Final merged: has_access={}, has_refresh={}, has_client_id={}, has_client_secret={}, auth_method={:?}",
            merged.access_token.is_some(),
            merged.refresh_token.is_some(),
            merged.client_id.is_some(),
            merged.client_secret.is_some(),
            merged.auth_method
        );

        self.credentials = merged;
        self.creds_path = Some(path);

        // 加载完成后，智能检测并更新认证方式（如果需要）
        let detected_auth_method = self.detect_auth_method();
        if self.credentials.auth_method.as_deref().unwrap_or("social") != detected_auth_method {
            tracing::info!(
                "[KIRO] 加载后检测到需要调整认证方式为: {}",
                detected_auth_method
            );
            self.set_auth_method(&detected_auth_method);
        }

        Ok(())
    }

    /// 从指定路径加载凭证
    ///
    /// 副本文件应包含完整的 client_id/client_secret（在复制时已合并）。
    /// 如果副本文件中没有，会尝试从 clientIdHash 文件读取作为回退。
    pub async fn load_credentials_from_path(
        &mut self,
        path: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = std::path::PathBuf::from(path);

        let mut merged = KiroCredentials::default();

        // 读取主凭证文件
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            let content = tokio::fs::read_to_string(&path).await?;
            let creds: KiroCredentials = serde_json::from_str(&content)?;
            tracing::info!(
                "[KIRO] 加载凭证文件 {:?}: has_access={}, has_refresh={}, has_client_id={}, has_client_secret={}, auth_method={:?}",
                path,
                creds.access_token.is_some(),
                creds.refresh_token.is_some(),
                creds.client_id.is_some(),
                creds.client_secret.is_some(),
                creds.auth_method
            );
            merge_credentials(&mut merged, &creds);
        } else {
            return Err(format!("凭证文件不存在: {:?}", path).into());
        }

        // 如果副本文件中已有 client_id/client_secret，直接使用（方案B：完全独立）
        if merged.client_id.is_some() && merged.client_secret.is_some() {
            tracing::info!("[KIRO] 副本文件包含完整的 client_id/client_secret，无需读取外部文件");
        } else {
            // 回退：尝试从外部文件读取（兼容旧的副本文件）
            let aws_sso_cache_dir = dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".aws")
                .join("sso")
                .join("cache");

            let mut found_credentials = false;

            // 方式1：如果有 clientIdHash，尝试从对应文件读取
            if let Some(hash) = &merged.client_id_hash.clone() {
                tracing::info!(
                    "[KIRO] 副本文件缺少 client_id/client_secret，尝试从 clientIdHash 文件读取"
                );
                let hash_file_path = aws_sso_cache_dir.join(format!("{}.json", hash));

                if tokio::fs::try_exists(&hash_file_path)
                    .await
                    .unwrap_or(false)
                {
                    if let Ok(content) = tokio::fs::read_to_string(&hash_file_path).await {
                        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&content)
                        {
                            if merged.client_id.is_none() {
                                merged.client_id = json_value
                                    .get("clientId")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                            }
                            if merged.client_secret.is_none() {
                                merged.client_secret = json_value
                                    .get("clientSecret")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                            }
                            if merged.client_id.is_some() && merged.client_secret.is_some() {
                                found_credentials = true;
                                tracing::info!(
                                    "[KIRO] 从 clientIdHash 文件补充: has_client_id={}, has_client_secret={}",
                                    merged.client_id.is_some(),
                                    merged.client_secret.is_some()
                                );
                            }
                        }
                    }
                }
            }

            // 方式2：如果没有 clientIdHash 或未找到，扫描目录中的其他 JSON 文件
            if !found_credentials
                && tokio::fs::try_exists(&aws_sso_cache_dir)
                    .await
                    .unwrap_or(false)
            {
                tracing::info!("[KIRO] 扫描 .aws/sso/cache 目录查找 client_id/client_secret");
                if let Ok(mut entries) = tokio::fs::read_dir(&aws_sso_cache_dir).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let file_path = entry.path();
                        if file_path.extension().map(|e| e == "json").unwrap_or(false) {
                            let file_name =
                                file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                            // 跳过主凭证文件和备份文件
                            if file_name.starts_with("kiro-auth-token") {
                                continue;
                            }
                            if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                                if let Ok(json_value) =
                                    serde_json::from_str::<serde_json::Value>(&content)
                                {
                                    let has_client_id = json_value
                                        .get("clientId")
                                        .and_then(|v| v.as_str())
                                        .is_some();
                                    let has_client_secret = json_value
                                        .get("clientSecret")
                                        .and_then(|v| v.as_str())
                                        .is_some();
                                    if has_client_id && has_client_secret {
                                        merged.client_id = json_value
                                            .get("clientId")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        merged.client_secret = json_value
                                            .get("clientSecret")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        found_credentials = true;
                                        tracing::info!(
                                            "[KIRO] 从 {} 补充 client_id/client_secret",
                                            file_name
                                        );
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !found_credentials {
                tracing::warn!("[KIRO] 未找到 client_id/client_secret，将使用 social 认证");
            }
        }

        tracing::info!(
            "[KIRO] 最终凭证状态: has_access={}, has_refresh={}, has_client_id={}, has_client_secret={}, auth_method={:?}",
            merged.access_token.is_some(),
            merged.refresh_token.is_some(),
            merged.client_id.is_some(),
            merged.client_secret.is_some(),
            merged.auth_method
        );

        self.credentials = merged;
        self.creds_path = Some(path);

        // 加载完成后，智能检测并更新认证方式（如果需要）
        let detected_auth_method = self.detect_auth_method();
        if self.credentials.auth_method.as_deref().unwrap_or("social") != detected_auth_method {
            tracing::info!(
                "[KIRO] 从路径加载后检测到需要调整认证方式为: {}",
                detected_auth_method
            );
            self.set_auth_method(&detected_auth_method);
        }

        Ok(())
    }

    pub fn get_base_url(&self) -> String {
        let region = self.credentials.region.as_deref().unwrap_or("us-east-1");
        format!("https://codewhisperer.{region}.amazonaws.com/generateAssistantResponse")
    }

    pub fn get_refresh_url(&self) -> String {
        let region = self.credentials.region.as_deref().unwrap_or("us-east-1");
        let auth_method = self
            .credentials
            .auth_method
            .as_deref()
            .unwrap_or("social")
            .to_lowercase();

        if auth_method == "idc" {
            format!("https://oidc.{region}.amazonaws.com/token")
        } else {
            format!("https://prod.{region}.auth.desktop.kiro.dev/refreshToken")
        }
    }

    /// 构建健康检查使用的端点，与实际API调用保持一致
    pub fn get_health_check_url(&self) -> String {
        // 重用基础URL逻辑，确保健康检查与实际API调用使用相同端点
        self.get_base_url()
    }

    /// 从凭证文件中提取 region 信息的静态方法，供健康检查服务使用
    pub fn extract_region_from_creds(creds_content: &str) -> Result<String, String> {
        let creds: serde_json::Value =
            serde_json::from_str(creds_content).map_err(|e| format!("解析凭证失败: {}", e))?;

        let region = creds["region"].as_str().unwrap_or("us-east-1").to_string();

        Ok(region)
    }

    /// 构建健康检查端点的静态方法，供外部服务使用
    pub fn build_health_check_url(region: &str) -> String {
        format!("https://codewhisperer.{region}.amazonaws.com/generateAssistantResponse")
    }

    /// 检查 Token 是否已过期
    ///
    /// 支持两种格式：
    /// - RFC3339 格式（新格式，与 CLIProxyAPI 兼容）
    /// - 时间戳格式（旧格式）
    pub fn is_token_expired(&self) -> bool {
        // 优先检查 RFC3339 格式的过期时间（新格式）
        if let Some(expire_str) = &self.credentials.expire {
            if let Ok(expires) = chrono::DateTime::parse_from_rfc3339(expire_str) {
                let now = chrono::Utc::now();
                // 提前5分钟判断为过期，避免边界情况
                return expires <= now + chrono::Duration::minutes(5);
            }
        }

        // 兼容旧的时间戳格式
        if let Some(expires_str) = &self.credentials.expires_at {
            if let Ok(expires_timestamp) = expires_str.parse::<i64>() {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;

                // 提前5分钟判断为过期，避免边界情况
                return now >= (expires_timestamp - 300);
            }
        }

        // 如果没有过期时间信息，保守地认为可能需要刷新
        true
    }

    /// 验证 refresh_token 的基本有效性
    pub fn validate_refresh_token(&self) -> Result<(), String> {
        let refresh_token = self.credentials.refresh_token.as_ref()
            .ok_or("缺少 refresh_token。\n💡 解决方案：\n1. 重新添加 OAuth 凭证\n2. 确保凭证文件包含完整的认证信息")?;

        // 基本格式验证
        if refresh_token.trim().is_empty() {
            return Err("refresh_token 为空。\n💡 解决方案：\n1. 检查凭证文件是否损坏\n2. 重新生成 OAuth 凭证".to_string());
        }

        let token_len = refresh_token.len();

        // 检测 refreshToken 是否被截断
        // 正常的 refreshToken 长度应该在 500+ 字符
        let is_truncated =
            token_len < 100 || refresh_token.ends_with("...") || refresh_token.contains("...");

        if is_truncated {
            tracing::error!(
                "[KIRO] 检测到 refreshToken 被截断！长度: {}, 内容: {}...",
                token_len,
                &refresh_token[..std::cmp::min(30, token_len)]
            );
            return Err(format!(
                "refreshToken 已被截断（长度: {} 字符）。\n\n⚠️ 这通常是 Kiro IDE 为了防止凭证被第三方工具使用而故意截断的。\n\n💡 解决方案：\n1. 使用 Kir-Manager 工具获取完整的凭证\n2. 或者使用其他方式获取未截断的凭证文件\n3. 正常的 refreshToken 长度应该在 500+ 字符",
                token_len
            ));
        }

        // 检查是否看起来像有效的 token（简单的长度和格式检查）
        if refresh_token.len() < 10 {
            return Err("refresh_token 格式异常（长度过短）。\n💡 解决方案：\n1. 凭证文件可能已损坏\n2. 重新获取 OAuth 凭证".to_string());
        }

        Ok(())
    }

    /// 检测认证方式
    ///
    /// 注意：不再自动降级！IdC 和 Social 的 refreshToken 不兼容，
    /// 不能将 IdC 的 refreshToken 用于 Social 端点。
    pub fn detect_auth_method(&self) -> String {
        // 直接返回配置中的认证方式，不做降级
        let auth_method = self.credentials.auth_method.as_deref().unwrap_or("social");
        tracing::debug!("[KIRO] 使用配置的认证方式: {}", auth_method);
        auth_method.to_lowercase()
    }

    /// 检查 IdC 认证配置是否完整
    pub fn is_idc_config_complete(&self) -> bool {
        self.credentials.client_id.is_some() && self.credentials.client_secret.is_some()
    }

    /// 更新认证方式到凭证中（仅在内存中，需要调用 save_credentials 持久化）
    pub fn set_auth_method(&mut self, method: &str) {
        let old_method = self.credentials.auth_method.as_deref().unwrap_or("social");
        if old_method != method {
            tracing::info!("[KIRO] 认证方式从 {} 切换到 {}", old_method, method);
            self.credentials.auth_method = Some(method.to_string());
        }
    }

    pub async fn refresh_token(&mut self) -> Result<String, Box<dyn Error + Send + Sync>> {
        // 首先验证 refresh_token 的有效性
        self.validate_refresh_token()?;

        tracing::info!("[KIRO] 开始 Token 刷新流程");
        tracing::info!(
            "[KIRO] 当前凭证状态: has_client_id={}, has_client_secret={}, auth_method={:?}",
            self.credentials.client_id.is_some(),
            self.credentials.client_secret.is_some(),
            self.credentials.auth_method
        );

        // 先克隆必要的值，避免借用冲突
        let refresh_token = self
            .credentials
            .refresh_token
            .as_ref()
            .ok_or("No refresh token")?
            .clone();

        // 获取认证方式
        let auth_method = self.detect_auth_method();
        tracing::info!("[KIRO] 使用认证方式: {}", auth_method);

        // 检查 IdC 认证是否有完整配置
        if auth_method == "idc" && !self.is_idc_config_complete() {
            let has_client_id = self.credentials.client_id.is_some();
            let has_client_secret = self.credentials.client_secret.is_some();

            // IdC 认证缺少必要凭证，返回明确错误（不能降级到 social，因为 refreshToken 不兼容）
            let missing = match (has_client_id, has_client_secret) {
                (false, false) => "clientId 和 clientSecret",
                (false, true) => "clientId",
                (true, false) => "clientSecret",
                _ => unreachable!(),
            };

            return Err(format!(
                "IdC 认证配置不完整：缺少 {}。\n\n⚠️ 注意：IdC 凭证的 refreshToken 无法用于 Social 认证，必须提供完整的 IdC 配置。\n\n💡 解决方案：\n1. 删除当前凭证\n2. 重新从 Kiro IDE 获取最新的凭证文件（确保完成完整的 SSO 登录流程）\n3. 确保 ~/.aws/sso/cache/ 目录下有对应的 clientIdHash 文件\n4. 重新添加凭证到 ProxyCast",
                missing
            ).into());
        }
        let refresh_url = self.get_refresh_url();

        tracing::debug!(
            "[KIRO] refresh_token: auth_method={}, refresh_url={}",
            auth_method,
            refresh_url
        );
        tracing::debug!(
            "[KIRO] has_client_id={}, has_client_secret={}",
            self.credentials.client_id.is_some(),
            self.credentials.client_secret.is_some()
        );

        // 获取设备指纹和版本号（用于 Social 认证的 User-Agent）
        let device_fp = get_device_fingerprint();
        let kiro_version = get_kiro_version();

        let resp = if auth_method == "idc" {
            // IdC 认证使用 JSON 格式（参考 Kir-Manager 实现）
            let client_id = self
                .credentials
                .client_id
                .as_ref()
                .ok_or("IdC 认证配置错误：缺少 client_id。建议删除后重新添加 OAuth 凭证")?;
            let client_secret = self
                .credentials
                .client_secret
                .as_ref()
                .ok_or("IdC 认证配置错误：缺少 client_secret。建议删除后重新添加 OAuth 凭证")?;

            // 使用 JSON 格式发送请求（与 Kir-Manager 保持一致）
            let body = serde_json::json!({
                "refreshToken": &refresh_token,
                "clientId": client_id,
                "clientSecret": client_secret,
                "grantType": "refresh_token"
            });

            tracing::debug!("[KIRO] IdC 刷新请求体已构建");

            // IdC 认证的 Headers（参考 Kir-Manager）
            self.client
                .post(&refresh_url)
                .header("Content-Type", "application/json")
                .header("Host", "oidc.us-east-1.amazonaws.com")
                .header(
                    "x-amz-user-agent",
                    "aws-sdk-js/3.738.0 ua/2.1 os/other lang/js api/sso-oidc#3.738.0 m/E KiroIDE",
                )
                .header("User-Agent", "node")
                .header("Accept", "*/*")
                .header("Connection", "keep-alive")
                .json(&body)
                .send()
                .await?
        } else {
            // Social 认证使用简单的 JSON 格式（参考 Kir-Manager）
            let body = serde_json::json!({ "refreshToken": &refresh_token });

            // Social 认证的 Headers（参考 Kir-Manager）
            self.client
                .post(&refresh_url)
                .header(
                    "User-Agent",
                    format!("KiroIDE-{}-{}", kiro_version, device_fp),
                )
                .header("Accept", "application/json, text/plain, */*")
                .header("Accept-Encoding", "br, gzip, deflate")
                .header("Content-Type", "application/json")
                .header("Accept-Language", "*")
                .header("Sec-Fetch-Mode", "cors")
                .json(&body)
                .send()
                .await?
        };

        tracing::info!("[KIRO] Token 刷新响应状态: {}", resp.status());

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();

            tracing::warn!("[KIRO] Token 刷新失败: {} - {}", status, body_text);

            // 根据具体的HTTP状态码提供更友好的错误信息
            let error_msg = match status.as_u16() {
                401 => {
                    if body_text.contains("Bad credentials") || body_text.contains("invalid") {
                        format!("OAuth 凭证已过期或无效，需要重新认证。\n💡 解决方案：\n1. 删除当前 OAuth 凭证\n2. 重新添加 OAuth 凭证\n3. 确保使用最新的凭证文件\n\n技术详情：{} {}", status, body_text)
                    } else {
                        format!("认证失败，Token 可能已过期。\n💡 解决方案：\n1. 检查 AWS 账户状态\n2. 重新生成 OAuth 凭证\n3. 确保凭证文件格式正确\n\n技术详情：{} {}", status, body_text)
                    }
                }
                403 => format!("权限不足，无法刷新 Token。\n💡 解决方案：\n1. 检查 AWS 账户权限\n2. 确保 OAuth 应用配置正确\n3. 联系管理员检查权限设置\n\n技术详情：{} {}", status, body_text),
                429 => format!("请求过于频繁，已被限流。\n💡 解决方案：\n1. 等待 5-10 分钟后重试\n2. 减少 Token 刷新频率\n3. 检查是否有其他程序在同时使用\n\n技术详情：{} {}", status, body_text),
                500..=599 => format!("服务器错误，AWS OAuth 服务暂时不可用。\n💡 解决方案：\n1. 稍后重试（通常几分钟后恢复）\n2. 检查 AWS 服务状态页面\n3. 如持续失败，联系 AWS 支持\n\n技术详情：{} {}", status, body_text),
                _ => format!("Token 刷新失败。\n💡 解决方案：\n1. 检查网络连接\n2. 确认凭证文件完整性\n3. 尝试重新添加凭证\n\n技术详情：{} {}", status, body_text)
            };

            return Err(error_msg.into());
        }

        let data: serde_json::Value = resp.json().await?;

        // AWS OIDC returns snake_case, social endpoint returns camelCase
        let new_token = data["accessToken"]
            .as_str()
            .or_else(|| data["access_token"].as_str())
            .ok_or("No access token in response")?;

        self.credentials.access_token = Some(new_token.to_string());

        // Handle both camelCase and snake_case response formats
        if let Some(rt) = data["refreshToken"]
            .as_str()
            .or_else(|| data["refresh_token"].as_str())
        {
            self.credentials.refresh_token = Some(rt.to_string());
        }
        if let Some(arn) = data["profileArn"].as_str() {
            self.credentials.profile_arn = Some(arn.to_string());
        }

        // 更新过期时间（如果响应中包含）
        if let Some(expires_in) = data["expiresIn"]
            .as_i64()
            .or_else(|| data["expires_in"].as_i64())
        {
            let expires_at = chrono::Utc::now() + chrono::Duration::seconds(expires_in);
            self.credentials.expire = Some(expires_at.to_rfc3339());
            // 同时更新旧格式以保持兼容
            self.credentials.expires_at = Some(expires_at.timestamp().to_string());
        }

        // 更新最后刷新时间（RFC3339 格式）
        self.credentials.last_refresh = Some(chrono::Utc::now().to_rfc3339());

        // 保存更新后的凭证到文件
        self.save_credentials().await?;

        Ok(new_token.to_string())
    }

    pub async fn save_credentials(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // 使用加载时的路径或默认路径
        let path = self
            .creds_path
            .clone()
            .unwrap_or_else(Self::default_creds_path);

        // 读取现有文件内容
        let mut existing: serde_json::Value = if tokio::fs::try_exists(&path).await.unwrap_or(false)
        {
            let content = tokio::fs::read_to_string(&path).await?;
            serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
        } else {
            serde_json::json!({})
        };

        // 更新字段
        if let Some(token) = &self.credentials.access_token {
            existing["accessToken"] = serde_json::json!(token);
        }
        if let Some(token) = &self.credentials.refresh_token {
            existing["refreshToken"] = serde_json::json!(token);
        }
        if let Some(arn) = &self.credentials.profile_arn {
            existing["profileArn"] = serde_json::json!(arn);
        }

        // 添加统一凭证格式字段（与 CLIProxyAPI 兼容）
        existing["type"] = serde_json::json!(self.credentials.cred_type);
        if let Some(expire) = &self.credentials.expire {
            existing["expire"] = serde_json::json!(expire);
        }
        if let Some(last_refresh) = &self.credentials.last_refresh {
            existing["lastRefresh"] = serde_json::json!(last_refresh);
        }

        // 写回文件
        let content = serde_json::to_string_pretty(&existing)?;
        tokio::fs::write(&path, content).await?;

        Ok(())
    }

    /// 检查 token 是否即将过期（10 分钟内）
    ///
    /// 支持两种格式：
    /// - RFC3339 格式（新格式，与 CLIProxyAPI 兼容）
    /// - 时间戳格式（旧格式）
    pub fn is_token_expiring_soon(&self) -> bool {
        // 优先检查 RFC3339 格式的过期时间（新格式）
        if let Some(expire_str) = &self.credentials.expire {
            if let Ok(expiry) = chrono::DateTime::parse_from_rfc3339(expire_str) {
                let now = chrono::Utc::now();
                let threshold = now + chrono::Duration::minutes(10);
                return expiry < threshold;
            }
        }

        // 兼容旧格式（expires_at 可能是 RFC3339 或时间戳）
        if let Some(expires_at) = &self.credentials.expires_at {
            // 尝试解析为 RFC3339
            if let Ok(expiry) = chrono::DateTime::parse_from_rfc3339(expires_at) {
                let now = chrono::Utc::now();
                let threshold = now + chrono::Duration::minutes(10);
                return expiry < threshold;
            }
            // 尝试解析为时间戳
            if let Ok(expires_timestamp) = expires_at.parse::<i64>() {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                return now >= (expires_timestamp - 600); // 10 分钟 = 600 秒
            }
        }
        // 如果没有过期时间，假设不需要刷新
        false
    }

    pub async fn call_api(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
        let token = self
            .credentials
            .access_token
            .as_ref()
            .ok_or("No access token")?;

        let profile_arn = if self.credentials.auth_method.as_deref() == Some("social") {
            self.credentials.profile_arn.clone()
        } else {
            None
        };

        let cw_request = convert_openai_to_codewhisperer(request, profile_arn);
        let url = self.get_base_url();

        // Debug: 记录转换后的请求
        if let Ok(json_str) = serde_json::to_string_pretty(&cw_request) {
            // 保存到文件用于调试
            let uuid_prefix = uuid::Uuid::new_v4()
                .to_string()
                .split('-')
                .next()
                .unwrap_or("unknown")
                .to_string();
            let debug_path = dirs::home_dir()
                .unwrap_or_default()
                .join(".proxycast")
                .join("logs")
                .join(format!("cw_request_{uuid_prefix}.json"));
            let _ = tokio::fs::write(&debug_path, &json_str).await;
            tracing::debug!("[CW_REQ] Request saved to {:?}", debug_path);

            // 记录历史消息数量和 tool_results 情况
            let history_len = cw_request
                .conversation_state
                .history
                .as_ref()
                .map(|h| h.len())
                .unwrap_or(0);
            let current_has_tools = cw_request
                .conversation_state
                .current_message
                .user_input_message
                .user_input_message_context
                .as_ref()
                .map(|ctx| ctx.tool_results.as_ref().map(|tr| tr.len()).unwrap_or(0))
                .unwrap_or(0);
            tracing::info!(
                "[CW_REQ] history={} current_tool_results={}",
                history_len,
                current_has_tools
            );
        }

        // 生成设备指纹用于伪装 Kiro IDE
        let device_fp = get_device_fingerprint();
        let kiro_version = get_kiro_version();

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("amz-sdk-invocation-id", uuid::Uuid::new_v4().to_string())
            .header("amz-sdk-request", "attempt=1; max=1")
            .header(
                "x-amz-user-agent",
                format!("aws-sdk-js/1.0.7 KiroIDE-{kiro_version}-{device_fp}"),
            )
            .header(
                "user-agent",
                format!(
                    "aws-sdk-js/1.0.7 ua/2.1 os/macos#14.0 lang/js md/nodejs#20.16.0 api/codewhispererstreaming#1.0.7 m/E KiroIDE-{kiro_version}-{device_fp}"
                ),
            )
            .header("x-amzn-kiro-agent-mode", "vibe")
            .json(&cw_request)
            .send()
            .await?;

        Ok(resp)
    }
}

fn merge_credentials(target: &mut KiroCredentials, source: &KiroCredentials) {
    if source.access_token.is_some() {
        target.access_token = source.access_token.clone();
    }
    if source.refresh_token.is_some() {
        target.refresh_token = source.refresh_token.clone();
    }
    if source.client_id.is_some() {
        target.client_id = source.client_id.clone();
    }
    if source.client_secret.is_some() {
        target.client_secret = source.client_secret.clone();
    }
    if source.profile_arn.is_some() {
        target.profile_arn = source.profile_arn.clone();
    }
    if source.expires_at.is_some() {
        target.expires_at = source.expires_at.clone();
    }
    if source.expire.is_some() {
        target.expire = source.expire.clone();
    }
    if source.region.is_some() {
        target.region = source.region.clone();
    }
    if source.auth_method.is_some() {
        target.auth_method = source.auth_method.clone();
    }
    if source.client_id_hash.is_some() {
        target.client_id_hash = source.client_id_hash.clone();
    }
    if source.last_refresh.is_some() {
        target.last_refresh = source.last_refresh.clone();
    }
    // cred_type 使用默认值，不需要合并
}
