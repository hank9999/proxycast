use crate::models::machine_id::*;
use crate::services::machine_id_service::MachineIdService;
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

/// 机器码管理服务状态
pub type MachineIdState = Arc<RwLock<MachineIdService>>;

/// 获取当前机器码信息
#[tauri::command]
pub async fn get_current_machine_id(
    service: State<'_, MachineIdState>,
) -> Result<MachineIdInfo, String> {
    let service = service.read().await;
    service.get_current_machine_id().await
}

/// 设置新的机器码
#[tauri::command]
pub async fn set_machine_id(
    new_id: String,
    service: State<'_, MachineIdState>,
) -> Result<MachineIdResult, String> {
    let service = service.read().await;
    service.set_machine_id(&new_id).await
}

/// 生成随机机器码
#[tauri::command]
pub async fn generate_random_machine_id(
    service: State<'_, MachineIdState>,
) -> Result<String, String> {
    let service = service.read().await;
    Ok(service.generate_random_machine_id())
}

/// 验证机器码格式
#[tauri::command]
pub async fn validate_machine_id(
    machine_id: String,
    service: State<'_, MachineIdState>,
) -> Result<MachineIdValidation, String> {
    let service = service.read().await;
    service.validate_machine_id(&machine_id)
}

/// 检查管理员权限
#[tauri::command]
pub async fn check_admin_privileges(
    service: State<'_, MachineIdState>,
) -> Result<AdminStatus, String> {
    let service = service.read().await;
    service.check_admin_privileges().await
}

/// 获取操作系统类型
#[tauri::command]
pub async fn get_os_type() -> Result<String, String> {
    Ok(MachineIdService::get_os_type())
}

/// 备份机器码到文件
#[tauri::command]
pub async fn backup_machine_id_to_file(
    file_path: String,
    service: State<'_, MachineIdState>,
) -> Result<bool, String> {
    let service = service.read().await;
    service.backup_machine_id(&file_path).await
}

/// 从文件恢复机器码
#[tauri::command]
pub async fn restore_machine_id_from_file(
    file_path: String,
    service: State<'_, MachineIdState>,
) -> Result<MachineIdResult, String> {
    let service = service.read().await;
    service.restore_machine_id(&file_path).await
}

/// 格式化机器码
#[tauri::command]
pub async fn format_machine_id(
    machine_id: String,
    format_type: String, // "uuid" 或 "hex32"
) -> Result<String, String> {
    let format = match format_type.to_lowercase().as_str() {
        "uuid" => MachineIdFormat::Uuid,
        "hex32" => MachineIdFormat::Hex32,
        _ => return Err("Unsupported format type. Use 'uuid' or 'hex32'".to_string()),
    };

    format.format_machine_id(&machine_id)
}

/// 检测机器码格式
#[tauri::command]
pub async fn detect_machine_id_format(machine_id: String) -> Result<String, String> {
    let format = MachineIdFormat::detect(&machine_id);
    Ok(format.to_string())
}

/// 转换机器码格式
#[tauri::command]
pub async fn convert_machine_id_format(
    machine_id: String,
    target_format: String,
) -> Result<String, String> {
    // 首先清理输入
    let cleaned = machine_id.replace("-", "").replace(" ", "").to_lowercase();

    // 验证输入是否为有效的十六进制
    if cleaned.len() != 32 || !cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("Invalid machine ID: must be 32 hex characters".to_string());
    }

    match target_format.to_lowercase().as_str() {
        "uuid" => {
            let format = MachineIdFormat::Uuid;
            format.format_machine_id(&cleaned)
        }
        "hex32" => {
            let format = MachineIdFormat::Hex32;
            format.format_machine_id(&cleaned)
        }
        _ => Err("Unsupported target format. Use 'uuid' or 'hex32'".to_string()),
    }
}

/// 获取机器码历史记录
#[tauri::command]
pub async fn get_machine_id_history(
    service: State<'_, MachineIdState>,
) -> Result<Vec<MachineIdHistory>, String> {
    let service = service.read().await;
    service.get_history()
}

/// 清除机器码覆盖（仅限 macOS）
#[tauri::command]
pub async fn clear_machine_id_override() -> Result<MachineIdResult, String> {
    #[cfg(target_os = "macos")]
    {
        use dirs;
        use std::fs;

        let override_file = dirs::data_dir()
            .ok_or("Failed to get app data directory")?
            .join("proxycast")
            .join("machine-id-override");

        match fs::remove_file(override_file) {
            Ok(_) => Ok(MachineIdResult {
                success: true,
                message: "Machine ID override removed successfully".to_string(),
                requires_restart: false,
                requires_admin: false,
                new_machine_id: None,
            }),
            Err(e) => Ok(MachineIdResult {
                success: false,
                message: format!("Failed to remove override: {}", e),
                requires_restart: false,
                requires_admin: false,
                new_machine_id: None,
            }),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(MachineIdResult {
            success: false,
            message: "Machine ID override is only supported on macOS".to_string(),
            requires_restart: false,
            requires_admin: false,
            new_machine_id: None,
        })
    }
}

/// 复制机器码到剪贴板
#[tauri::command]
pub async fn copy_machine_id_to_clipboard(machine_id: String) -> Result<bool, String> {
    use arboard::Clipboard;
    let mut clipboard =
        Clipboard::new().map_err(|e| format!("Failed to access clipboard: {}", e))?;
    clipboard
        .set_text(machine_id)
        .map_err(|e| format!("Failed to copy to clipboard: {}", e))?;

    Ok(true)
}

/// 从剪贴板粘贴机器码
#[tauri::command]
pub async fn paste_machine_id_from_clipboard() -> Result<String, String> {
    use arboard::Clipboard;
    let mut clipboard =
        Clipboard::new().map_err(|e| format!("Failed to access clipboard: {}", e))?;
    let text = clipboard
        .get_text()
        .map_err(|e| format!("Failed to read from clipboard: {}", e))?;

    // 基本验证
    let cleaned = text.replace("-", "").replace(" ", "").trim().to_lowercase();
    if cleaned.len() == 32 && cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(text.trim().to_string())
    } else {
        Err("Clipboard does not contain a valid machine ID".to_string())
    }
}

/// 获取系统信息（用于调试和支持）
#[tauri::command]
pub async fn get_system_info() -> Result<SystemInfo, String> {
    let os = MachineIdService::get_os_type();
    let arch = std::env::consts::ARCH.to_string();
    let family = std::env::consts::FAMILY.to_string();

    Ok(SystemInfo {
        os,
        arch,
        family,
        machine_id_support: get_machine_id_platform_support(),
        requires_admin: match MachineIdService::get_os_type().as_str() {
            "windows" | "linux" => true,
            "macos" => false,
            _ => true,
        },
    })
}

/// 系统信息结构
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub family: String,
    pub machine_id_support: PlatformSupport,
    pub requires_admin: bool,
}

/// 平台支持信息
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct PlatformSupport {
    pub can_read: bool,
    pub can_write: bool,
    pub format: String,
    pub method: String,
    pub limitations: Vec<String>,
}

/// 获取当前平台的机器码支持信息
fn get_machine_id_platform_support() -> PlatformSupport {
    match MachineIdService::get_os_type().as_str() {
        "windows" => PlatformSupport {
            can_read: true,
            can_write: true,
            format: "UUID".to_string(),
            method: "Windows Registry (HKLM\\SOFTWARE\\Microsoft\\Cryptography\\MachineGuid)"
                .to_string(),
            limitations: vec![
                "Requires administrator privileges".to_string(),
                "May require restart for some applications".to_string(),
            ],
        },
        "macos" => PlatformSupport {
            can_read: true,
            can_write: true,
            format: "UUID".to_string(),
            method: "Application-level override (ioreg IOPlatformUUID + override file)".to_string(),
            limitations: vec![
                "Original system UUID cannot be modified".to_string(),
                "Only affects applications that use the override".to_string(),
            ],
        },
        "linux" => PlatformSupport {
            can_read: true,
            can_write: true,
            format: "32-bit Hex".to_string(),
            method: "File system (/etc/machine-id)".to_string(),
            limitations: vec![
                "Requires root privileges".to_string(),
                "May affect system services".to_string(),
                "Some services may require restart".to_string(),
            ],
        },
        _ => PlatformSupport {
            can_read: false,
            can_write: false,
            format: "Unknown".to_string(),
            method: "Not supported".to_string(),
            limitations: vec!["Platform not supported".to_string()],
        },
    }
}
