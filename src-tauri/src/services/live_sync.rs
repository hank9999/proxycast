use crate::models::{AppType, Provider};
use serde_json::{json, Value};
use std::path::PathBuf;

/// 原子写入 JSON 文件，防止配置损坏
/// 参考 cc-switch 的实现：使用临时文件 + 重命名的原子操作
fn write_json_file_atomic(
    path: &std::path::Path,
    value: &Value,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::fs;
    use std::io::Write;

    // 确保目录存在
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // 创建临时文件
    let temp_path = path.with_extension("tmp");

    // 写入临时文件
    let content = serde_json::to_string_pretty(value)?;
    let mut temp_file = fs::File::create(&temp_path)?;
    temp_file.write_all(content.as_bytes())?;
    temp_file.flush()?;
    drop(temp_file); // 确保文件句柄被释放

    // 验证 JSON 格式正确性
    let verify_content = fs::read_to_string(&temp_path)?;
    let _: Value = serde_json::from_str(&verify_content)?; // 验证解析

    // 原子性重命名
    fs::rename(&temp_path, path)?;

    tracing::info!("Successfully wrote config file: {}", path.display());
    Ok(())
}

/// 创建配置文件的备份
fn create_backup(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if path.exists() {
        let backup_path = path.with_extension("bak");
        std::fs::copy(path, &backup_path)?;
        tracing::info!("Created backup: {}", backup_path.display());
    }
    Ok(())
}

/// Get the configuration file path for an app type
#[allow(dead_code)]
pub fn get_app_config_path(app_type: &AppType) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    match app_type {
        AppType::Claude => Some(home.join(".claude").join("settings.json")),
        AppType::Codex => Some(home.join(".codex")),
        AppType::Gemini => Some(home.join(".gemini")),
        AppType::ProxyCast => None,
    }
}

/// Sync provider configuration to live config files
pub fn sync_to_live(
    app_type: &AppType,
    provider: &Provider,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match app_type {
        AppType::Claude => sync_claude_settings(provider),
        AppType::Codex => sync_codex_config(provider),
        AppType::Gemini => sync_gemini_config(provider),
        AppType::ProxyCast => Ok(()),
    }
}

/// 清理 Claude 配置中冲突的认证环境变量
///
/// Claude Code 同时检测到 ANTHROPIC_AUTH_TOKEN 和 ANTHROPIC_API_KEY 时会报警告。
/// 此函数确保只保留一个认证变量：
/// - 优先保留 ANTHROPIC_AUTH_TOKEN（OAuth token）
/// - 如果只有 ANTHROPIC_API_KEY，则保留它
fn clean_claude_auth_conflict(settings: &mut Value) {
    if let Some(env) = settings.get_mut("env").and_then(|v| v.as_object_mut()) {
        let has_auth_token = env
            .get("ANTHROPIC_AUTH_TOKEN")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        let has_api_key = env
            .get("ANTHROPIC_API_KEY")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);

        // 如果两者都存在，移除 ANTHROPIC_API_KEY（优先使用 AUTH_TOKEN）
        if has_auth_token && has_api_key {
            tracing::info!(
                "检测到 Claude 认证冲突：同时存在 ANTHROPIC_AUTH_TOKEN 和 ANTHROPIC_API_KEY，移除 ANTHROPIC_API_KEY"
            );
            env.remove("ANTHROPIC_API_KEY");
        }
    }
}

/// Sync Claude settings to ~/.claude/settings.json
fn sync_claude_settings(
    provider: &Provider,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let home = dirs::home_dir().ok_or("Cannot find home directory")?;
    let claude_dir = home.join(".claude");
    let config_path = claude_dir.join("settings.json");

    tracing::info!("开始同步 Claude 配置: {}", provider.name);

    // 创建备份（如果文件存在）
    create_backup(&config_path)?;

    // Ensure .claude directory exists
    if !claude_dir.exists() {
        std::fs::create_dir_all(&claude_dir)?;
    }

    // Read existing settings to preserve other fields
    let mut settings: Value = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("配置文件格式有误，使用默认配置: {}", e);
                json!({})
            }
        }
    } else {
        json!({})
    };

    // Merge env variables into settings
    if let Some(env_obj) = provider
        .settings_config
        .get("env")
        .and_then(|v| v.as_object())
    {
        let settings_obj = settings.as_object_mut().ok_or("Invalid settings format")?;

        // Ensure env object exists
        if !settings_obj.contains_key("env") {
            settings_obj.insert("env".to_string(), json!({}));
        }

        if let Some(target_env) = settings_obj.get_mut("env").and_then(|v| v.as_object_mut()) {
            for (key, value) in env_obj {
                target_env.insert(key.clone(), value.clone());
                tracing::debug!("设置环境变量: {} = [MASKED]", key);
            }
        }
    } else {
        // If settings_config is the full settings object, use it directly
        settings = provider.settings_config.clone();
        tracing::debug!("使用完整配置对象");
    }

    // 清理冲突的认证环境变量
    clean_claude_auth_conflict(&mut settings);

    // 使用原子写入
    write_json_file_atomic(&config_path, &settings)?;

    tracing::info!("Claude 配置同步完成: {}", config_path.display());
    Ok(())
}

/// Sync Codex config to ~/.codex/auth.json and ~/.codex/config.toml
fn sync_codex_config(provider: &Provider) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let home = dirs::home_dir().ok_or("Cannot find home directory")?;
    let codex_dir = home.join(".codex");

    // Create directory if not exists
    std::fs::create_dir_all(&codex_dir)?;

    if let Some(obj) = provider.settings_config.as_object() {
        // Write auth.json
        if let Some(auth) = obj.get("auth") {
            let auth_path = codex_dir.join("auth.json");
            let content = serde_json::to_string_pretty(auth)?;
            std::fs::write(&auth_path, content)?;
        }

        // Write config.toml
        if let Some(config) = obj.get("config").and_then(|v| v.as_str()) {
            let config_path = codex_dir.join("config.toml");
            std::fs::write(&config_path, config)?;
        }
    }

    Ok(())
}

/// Sync Gemini config to ~/.gemini/.env and ~/.gemini/settings.json
fn sync_gemini_config(provider: &Provider) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let home = dirs::home_dir().ok_or("Cannot find home directory")?;
    let gemini_dir = home.join(".gemini");

    // Create directory if not exists
    std::fs::create_dir_all(&gemini_dir)?;

    // Write .env file
    if let Some(env_obj) = provider
        .settings_config
        .get("env")
        .and_then(|v| v.as_object())
    {
        let env_path = gemini_dir.join(".env");
        let mut content = String::new();

        for (key, value) in env_obj {
            if let Some(val) = value.as_str() {
                // Only write non-empty values
                if !val.is_empty() {
                    content.push_str(&format!("{key}={val}\n"));
                }
            }
        }

        std::fs::write(&env_path, content)?;
    }

    // Write settings.json (for MCP servers and other config)
    if let Some(config) = provider.settings_config.get("config") {
        if config.is_object() {
            let settings_path = gemini_dir.join("settings.json");

            // Read existing settings to preserve mcpServers
            let mut settings: Value = if settings_path.exists() {
                let content = std::fs::read_to_string(&settings_path)?;
                serde_json::from_str(&content).unwrap_or_else(|_| json!({}))
            } else {
                json!({})
            };

            // Merge config into settings
            if let (Some(settings_obj), Some(config_obj)) =
                (settings.as_object_mut(), config.as_object())
            {
                for (key, value) in config_obj {
                    settings_obj.insert(key.clone(), value.clone());
                }
            }

            let content = serde_json::to_string_pretty(&settings)?;
            std::fs::write(&settings_path, content)?;
        }
    }

    Ok(())
}

/// Read current live settings for an app type
pub fn read_live_settings(
    app_type: &AppType,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let home = dirs::home_dir().ok_or("Cannot find home directory")?;

    match app_type {
        AppType::Claude => {
            let path = home.join(".claude").join("settings.json");
            if !path.exists() {
                return Err("Claude settings file not found".into());
            }
            let content = std::fs::read_to_string(&path)?;
            Ok(serde_json::from_str(&content)?)
        }
        AppType::Codex => {
            let codex_dir = home.join(".codex");
            let auth_path = codex_dir.join("auth.json");
            let config_path = codex_dir.join("config.toml");

            let auth: Value = if auth_path.exists() {
                let content = std::fs::read_to_string(&auth_path)?;
                serde_json::from_str(&content)?
            } else {
                json!({})
            };

            let config = if config_path.exists() {
                std::fs::read_to_string(&config_path)?
            } else {
                String::new()
            };

            Ok(json!({
                "auth": auth,
                "config": config
            }))
        }
        AppType::Gemini => {
            let gemini_dir = home.join(".gemini");
            let env_path = gemini_dir.join(".env");
            let settings_path = gemini_dir.join("settings.json");

            // Read .env file
            let mut env_map: serde_json::Map<String, Value> = serde_json::Map::new();
            if env_path.exists() {
                let content = std::fs::read_to_string(&env_path)?;
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some((key, value)) = line.split_once('=') {
                        env_map.insert(key.trim().to_string(), json!(value.trim()));
                    }
                }
            }

            // Read settings.json
            let config: Value = if settings_path.exists() {
                let content = std::fs::read_to_string(&settings_path)?;
                serde_json::from_str(&content)?
            } else {
                json!({})
            };

            Ok(json!({
                "env": env_map,
                "config": config
            }))
        }
        AppType::ProxyCast => Ok(json!({})),
    }
}

/// 同步状态枚举
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SyncStatus {
    InSync,    // 完全同步
    OutOfSync, // 有差异但无冲突
    Conflict,  // 有冲突需要用户选择
}

/// 配置冲突信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConfigConflict {
    pub field: String,
    pub local_value: String,
    pub external_value: String,
}

/// 同步检查结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncCheckResult {
    pub status: SyncStatus,
    pub current_provider: String,
    pub external_provider: String,
    pub last_modified: Option<String>,
    pub conflicts: Vec<ConfigConflict>,
}

/// 从外部配置文件解析当前生效的 provider
pub fn parse_current_provider_from_live(
    app_type: &AppType,
    live_settings: &Value,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    match app_type {
        AppType::Claude => {
            // 检查 Claude 配置中的认证信息来判断当前 provider
            if let Some(env) = live_settings.get("env").and_then(|v| v.as_object()) {
                // 优先检查 ANTHROPIC_AUTH_TOKEN (OAuth)
                if let Some(token) = env.get("ANTHROPIC_AUTH_TOKEN").and_then(|v| v.as_str()) {
                    if !token.is_empty() {
                        return Ok("claude_oauth".to_string());
                    }
                }

                // 检查 ANTHROPIC_API_KEY (API Key)
                if let Some(api_key) = env.get("ANTHROPIC_API_KEY").and_then(|v| v.as_str()) {
                    if !api_key.is_empty() {
                        return Ok("claude".to_string());
                    }
                }
            }

            Ok("unknown".to_string())
        }
        AppType::Codex => {
            // 检查 Codex 认证信息
            if let Some(auth) = live_settings.get("auth").and_then(|v| v.as_object()) {
                if auth
                    .get("access_token")
                    .and_then(|v| v.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false)
                {
                    return Ok("codex".to_string());
                }
            }

            Ok("unknown".to_string())
        }
        AppType::Gemini => {
            // 检查 Gemini 环境变量
            if let Some(env) = live_settings.get("env").and_then(|v| v.as_object()) {
                if let Some(api_key) = env.get("GOOGLE_API_KEY").and_then(|v| v.as_str()) {
                    if !api_key.is_empty() {
                        return Ok("gemini".to_string());
                    }
                }
            }

            Ok("unknown".to_string())
        }
        AppType::ProxyCast => Ok("proxycast".to_string()),
    }
}

/// 检查配置同步状态
pub fn check_config_sync(
    app_type: &AppType,
    current_provider: &str,
) -> Result<SyncCheckResult, Box<dyn std::error::Error + Send + Sync>> {
    // 读取外部配置文件
    let live_settings = read_live_settings(app_type)?;

    // 解析外部配置中的当前 provider
    let external_provider = parse_current_provider_from_live(app_type, &live_settings)?;

    // 获取配置文件的修改时间
    let last_modified = get_config_last_modified(app_type);

    // 比较配置 - 重要：需要更智能的比对逻辑
    let status = if current_provider == external_provider {
        SyncStatus::InSync
    } else if external_provider == "unknown" {
        // 外部配置无法识别，可能是配置文件不存在或损坏
        SyncStatus::OutOfSync
    } else {
        // 这里需要更智能的判断：
        // 如果外部配置是通过ProxyCast设置的，应该检查是否已有匹配的provider
        // 只有确实是来自其他外部软件的配置才标记为冲突

        // 对于简化，先标记为冲突，让前端的更详细的比对逻辑来处理
        // 实际上前端的 configsMatch 函数会做更精确的比对
        SyncStatus::Conflict
    };

    // 检测具体的冲突字段
    let conflicts = if matches!(status, SyncStatus::Conflict) {
        vec![ConfigConflict {
            field: "provider".to_string(),
            local_value: current_provider.to_string(),
            external_value: external_provider.clone(),
        }]
    } else {
        vec![]
    };

    Ok(SyncCheckResult {
        status,
        current_provider: current_provider.to_string(),
        external_provider,
        last_modified,
        conflicts,
    })
}

/// 获取配置文件的最后修改时间
fn get_config_last_modified(app_type: &AppType) -> Option<String> {
    let home = dirs::home_dir()?;
    let path = match app_type {
        AppType::Claude => home.join(".claude").join("settings.json"),
        AppType::Codex => home.join(".codex").join("auth.json"),
        AppType::Gemini => home.join(".gemini").join(".env"),
        AppType::ProxyCast => return None,
    };

    if let Ok(metadata) = std::fs::metadata(&path) {
        if let Ok(modified) = metadata.modified() {
            if let Ok(datetime) = modified.duration_since(std::time::UNIX_EPOCH) {
                return Some(datetime.as_secs().to_string());
            }
        }
    }

    None
}

/// 从外部配置同步到 ProxyCast 数据库
/// 这个函数需要与 switch service 集成来更新数据库中的 provider 记录
pub fn sync_from_external(
    app_type: &AppType,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // 读取外部配置
    let live_settings = read_live_settings(app_type)?;

    // 解析当前生效的 provider
    let external_provider = parse_current_provider_from_live(app_type, &live_settings)?;

    if external_provider == "unknown" {
        return Err("无法识别外部配置中的 provider".into());
    }

    // 返回检测到的 provider，由调用方负责更新数据库
    Ok(external_provider)
}
