//! Usage Tauri 命令
//!
//! 提供 Kiro 用量查询的 Tauri 命令接口。

use crate::database::dao::provider_pool::ProviderPoolDao;
use crate::database::DbConnection;
use crate::models::provider_pool_model::{CredentialData, PoolProviderType};
use crate::services::usage_service::{self, UsageInfo};
use crate::TokenCacheServiceState;
use tauri::State;

/// 默认 Kiro 版本号
const DEFAULT_KIRO_VERSION: &str = "1.0.0";

/// 获取 Kiro 用量信息
///
/// **Validates: Requirements 1.1**
///
/// # Arguments
/// * `credential_uuid` - 凭证的 UUID
/// * `db` - 数据库连接
/// * `token_cache` - Token 缓存服务
///
/// # Returns
/// * `Ok(UsageInfo)` - 成功时返回用量信息
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_kiro_usage(
    credential_uuid: String,
    db: State<'_, DbConnection>,
    token_cache: State<'_, TokenCacheServiceState>,
) -> Result<UsageInfo, String> {
    // 1. 获取凭证信息
    let credential = {
        let conn = db.lock().map_err(|e| e.to_string())?;
        ProviderPoolDao::get_by_uuid(&conn, &credential_uuid)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("凭证不存在: {}", credential_uuid))?
    };

    // 2. 验证是否为 Kiro 凭证
    if credential.provider_type != PoolProviderType::Kiro {
        return Err(format!(
            "不支持的凭证类型: {:?}，仅支持 Kiro 凭证",
            credential.provider_type
        ));
    }

    // 3. 获取凭证文件路径
    let creds_file_path = match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => creds_file_path.clone(),
        _ => return Err("凭证数据类型不匹配".to_string()),
    };

    // 4. 获取有效的 access_token
    let access_token = token_cache
        .0
        .get_valid_token(&db, &credential_uuid)
        .await
        .map_err(|e| {
            // 提供更友好的错误信息
            if e.contains("401") || e.contains("Bad credentials") || e.contains("过期") || e.contains("无效") {
                format!("刷新 Kiro Token 失败: OAuth 凭证已过期或无效，需要重新认证。\n💡 解决方案：\n1. 删除当前 OAuth 凭证\n2. 重新添加 OAuth 凭证\n3. 确保使用最新的凭证文件\n\n技术详情：{}", e)
            } else {
                e
            }
        })?;

    // 5. 从凭证文件读取 auth_method 和 profile_arn
    let (auth_method, profile_arn) = read_kiro_credential_info(&creds_file_path)?;

    // 6. 获取 machine_id
    let machine_id = get_machine_id()?;

    // 7. 调用 Usage API
    let usage_info = usage_service::get_usage_limits_safe(
        &access_token,
        &auth_method,
        profile_arn.as_deref(),
        &machine_id,
        DEFAULT_KIRO_VERSION,
    )
    .await;

    Ok(usage_info)
}

/// 从 Kiro 凭证文件读取 auth_method 和 profile_arn
fn read_kiro_credential_info(creds_file_path: &str) -> Result<(String, Option<String>), String> {
    // 展开 ~ 路径
    let expanded_path = expand_tilde(creds_file_path);

    // 读取文件
    let content =
        std::fs::read_to_string(&expanded_path).map_err(|e| format!("读取凭证文件失败: {}", e))?;

    // 解析 JSON
    let json: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("解析凭证文件失败: {}", e))?;

    // 获取 auth_method，默认为 "social"
    let auth_method = json
        .get("authMethod")
        .and_then(|v| v.as_str())
        .unwrap_or("social")
        .to_string();

    // 获取 profile_arn（可选）
    let profile_arn = json
        .get("profileArn")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok((auth_method, profile_arn))
}

/// 展开路径中的 ~ 为用户主目录
fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path[2..]).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

/// 获取设备 ID（SHA256 哈希）
fn get_machine_id() -> Result<String, String> {
    // 尝试获取系统 machine-id
    let raw_id = get_raw_machine_id()?;

    // 计算 SHA256 哈希
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(raw_id.as_bytes());
    let result = hasher.finalize();

    Ok(format!("{:x}", result))
}

/// 获取原始设备 ID
fn get_raw_machine_id() -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        // macOS: 使用 IOPlatformUUID
        use std::process::Command;
        let output = Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .map_err(|e| format!("执行 ioreg 失败: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("IOPlatformUUID") {
                if let Some(uuid) = line.split('"').nth(3) {
                    return Ok(uuid.to_string());
                }
            }
        }
        Err("无法获取 IOPlatformUUID".to_string())
    }

    #[cfg(target_os = "linux")]
    {
        // Linux: 读取 /etc/machine-id
        std::fs::read_to_string("/etc/machine-id")
            .map(|s| s.trim().to_string())
            .map_err(|e| format!("读取 /etc/machine-id 失败: {}", e))
    }

    #[cfg(target_os = "windows")]
    {
        // Windows: 使用注册表中的 MachineGuid
        use std::process::Command;
        let output = Command::new("reg")
            .args([
                "query",
                "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .output()
            .map_err(|e| format!("执行 reg query 失败: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("MachineGuid") {
                if let Some(guid) = line.split_whitespace().last() {
                    return Ok(guid.to_string());
                }
            }
        }
        Err("无法获取 MachineGuid".to_string())
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Err("不支持的操作系统".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde() {
        let path = "~/test/path";
        let expanded = expand_tilde(path);
        assert!(!expanded.starts_with("~/"));
        assert!(expanded.ends_with("test/path"));
    }

    #[test]
    fn test_expand_tilde_no_tilde() {
        let path = "/absolute/path";
        let expanded = expand_tilde(path);
        assert_eq!(expanded, path);
    }

    #[test]
    fn test_get_machine_id() {
        // 这个测试在不同平台上行为不同
        let result = get_machine_id();
        // 应该能成功获取 machine_id
        assert!(result.is_ok(), "Failed to get machine_id: {:?}", result);
        // machine_id 应该是 64 字符的十六进制字符串（SHA256）
        let id = result.unwrap();
        assert_eq!(id.len(), 64, "Machine ID should be 64 hex chars");
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "Machine ID should be hex"
        );
    }
}

// ============================================================================
// 集成测试
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// 测试 read_kiro_credential_info 函数
    /// 验证能正确解析 Kiro 凭证文件中的 auth_method 和 profile_arn
    #[test]
    fn test_read_kiro_credential_info_social() {
        // 创建临时文件
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_kiro_creds_social.json");

        let creds_json = serde_json::json!({
            "accessToken": "test_access_token",
            "refreshToken": "test_refresh_token",
            "authMethod": "social",
            "profileArn": "arn:aws:iam::123456789:profile/test"
        });

        std::fs::write(&temp_file, serde_json::to_string(&creds_json).unwrap()).unwrap();

        let result = read_kiro_credential_info(temp_file.to_str().unwrap());
        assert!(result.is_ok());

        let (auth_method, profile_arn) = result.unwrap();
        assert_eq!(auth_method, "social");
        assert_eq!(
            profile_arn,
            Some("arn:aws:iam::123456789:profile/test".to_string())
        );

        // 清理
        let _ = std::fs::remove_file(&temp_file);
    }

    /// 测试 read_kiro_credential_info 函数 - IdC 认证
    #[test]
    fn test_read_kiro_credential_info_idc() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_kiro_creds_idc.json");

        let creds_json = serde_json::json!({
            "accessToken": "test_access_token",
            "refreshToken": "test_refresh_token",
            "authMethod": "idc"
        });

        std::fs::write(&temp_file, serde_json::to_string(&creds_json).unwrap()).unwrap();

        let result = read_kiro_credential_info(temp_file.to_str().unwrap());
        assert!(result.is_ok());

        let (auth_method, profile_arn) = result.unwrap();
        assert_eq!(auth_method, "idc");
        assert_eq!(profile_arn, None);

        // 清理
        let _ = std::fs::remove_file(&temp_file);
    }

    /// 测试 read_kiro_credential_info 函数 - 默认 auth_method
    #[test]
    fn test_read_kiro_credential_info_default_auth_method() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_kiro_creds_default.json");

        // 没有 authMethod 字段，应该默认为 "social"
        let creds_json = serde_json::json!({
            "accessToken": "test_access_token",
            "refreshToken": "test_refresh_token"
        });

        std::fs::write(&temp_file, serde_json::to_string(&creds_json).unwrap()).unwrap();

        let result = read_kiro_credential_info(temp_file.to_str().unwrap());
        assert!(result.is_ok());

        let (auth_method, profile_arn) = result.unwrap();
        assert_eq!(auth_method, "social");
        assert_eq!(profile_arn, None);

        // 清理
        let _ = std::fs::remove_file(&temp_file);
    }

    /// 测试 read_kiro_credential_info 函数 - 文件不存在
    #[test]
    fn test_read_kiro_credential_info_file_not_found() {
        let result = read_kiro_credential_info("/nonexistent/path/to/creds.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("读取凭证文件失败"));
    }

    /// 测试 read_kiro_credential_info 函数 - 无效 JSON
    #[test]
    fn test_read_kiro_credential_info_invalid_json() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_kiro_creds_invalid.json");

        std::fs::write(&temp_file, "not valid json").unwrap();

        let result = read_kiro_credential_info(temp_file.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("解析凭证文件失败"));

        // 清理
        let _ = std::fs::remove_file(&temp_file);
    }
}
