use crate::models::machine_id::*;
use dirs;
use serde_json;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing;
use uuid::Uuid;

#[cfg(target_os = "windows")]
use std::ptr;
#[cfg(target_os = "windows")]
use winapi::um::winnt::KEY_READ;
#[cfg(target_os = "windows")]
use winreg::{enums::*, RegKey};

pub struct MachineIdService {
    backup_dir: PathBuf,
    history_file: PathBuf,
}

impl MachineIdService {
    pub fn new() -> Result<Self, String> {
        let app_data_dir = dirs::data_dir()
            .ok_or("Failed to get app data directory")?
            .join("proxycast");

        let backup_dir = app_data_dir.join("machine_id_backups");
        let history_file = app_data_dir.join("machine_id_history.json");

        // 确保应用数据目录存在
        if let Err(e) = fs::create_dir_all(&app_data_dir) {
            return Err(format!("Failed to create app data directory: {}", e));
        }

        // 确保备份目录存在
        if let Err(e) = fs::create_dir_all(&backup_dir) {
            return Err(format!("Failed to create backup directory: {}", e));
        }

        Ok(MachineIdService {
            backup_dir,
            history_file,
        })
    }

    /// 获取当前操作系统类型
    pub fn get_os_type() -> String {
        std::env::consts::OS.to_string()
    }

    /// 获取当前机器码信息
    pub async fn get_current_machine_id(&self) -> Result<MachineIdInfo, String> {
        let os = Self::get_os_type();
        let requires_admin = self.check_requires_admin().await?;
        let backup_exists = self.check_backup_exists();

        match os.as_str() {
            "windows" => {
                self.get_windows_machine_id(requires_admin, backup_exists)
                    .await
            }
            "macos" => {
                self.get_macos_machine_id(requires_admin, backup_exists)
                    .await
            }
            "linux" => {
                self.get_linux_machine_id(requires_admin, backup_exists)
                    .await
            }
            _ => Err(format!("Unsupported operating system: {}", os)),
        }
    }

    /// 设置新的机器码
    pub async fn set_machine_id(&self, new_id: &str) -> Result<MachineIdResult, String> {
        let validation = self.validate_machine_id(new_id)?;
        if !validation.is_valid {
            return Ok(MachineIdResult {
                success: false,
                message: validation
                    .error_message
                    .unwrap_or("Invalid machine ID format".to_string()),
                requires_restart: false,
                requires_admin: false,
                new_machine_id: None,
            });
        }

        let formatted_id = validation.formatted_id.unwrap();
        let os = Self::get_os_type();

        // 在设置前先备份当前机器码
        if let Err(e) = self.create_auto_backup().await {
            tracing::warn!("Failed to create auto backup: {}", e);
        }

        let result = match os.as_str() {
            "windows" => self.set_windows_machine_id(&formatted_id).await,
            "macos" => self.set_macos_machine_id(&formatted_id).await,
            "linux" => self.set_linux_machine_id(&formatted_id).await,
            _ => Ok(MachineIdResult {
                success: false,
                message: format!("Unsupported operating system: {}", os),
                requires_restart: false,
                requires_admin: false,
                new_machine_id: None,
            }),
        };

        // 添加到历史记录
        if let Ok(ref res) = result {
            if res.success {
                if let Err(e) = self.add_history_record(formatted_id.clone(), None) {
                    tracing::warn!("Failed to add history record: {}", e);
                }
            }
        }

        result
    }

    /// 生成随机机器码
    pub fn generate_random_machine_id(&self) -> String {
        let uuid = Uuid::new_v4();
        uuid.to_string()
    }

    /// 检查管理员权限
    pub async fn check_admin_privileges(&self) -> Result<AdminStatus, String> {
        let os = Self::get_os_type();

        match os.as_str() {
            "windows" => self.check_windows_admin().await,
            "macos" => self.check_macos_admin().await,
            "linux" => self.check_linux_admin().await,
            _ => Ok(AdminStatus {
                is_admin: false,
                platform: os,
                elevation_method: None,
                check_success: false,
            }),
        }
    }

    /// 验证机器码格式
    pub fn validate_machine_id(&self, machine_id: &str) -> Result<MachineIdValidation, String> {
        let detected_format = MachineIdFormat::detect(machine_id);

        match detected_format.format_machine_id(machine_id) {
            Ok(formatted) => Ok(MachineIdValidation {
                is_valid: true,
                detected_format,
                error_message: None,
                formatted_id: Some(formatted),
            }),
            Err(error) => Ok(MachineIdValidation {
                is_valid: false,
                detected_format,
                error_message: Some(error),
                formatted_id: None,
            }),
        }
    }

    /// 备份机器码到文件
    pub async fn backup_machine_id(&self, file_path: &str) -> Result<bool, String> {
        let current_info = self.get_current_machine_id().await?;

        let backup = MachineIdBackup {
            machine_id: current_info.current_id.clone(),
            timestamp: chrono::Utc::now().timestamp(),
            platform: current_info.platform,
            format: current_info.format_type,
            description: Some("Manual backup".to_string()),
        };

        let backup_json = serde_json::to_string_pretty(&backup)
            .map_err(|e| format!("Failed to serialize backup: {}", e))?;

        let write_result = fs::write(file_path, backup_json)
            .map_err(|e| format!("Failed to write backup file: {}", e));

        // 添加到历史记录
        if write_result.is_ok() {
            if let Err(e) =
                self.add_history_record(current_info.current_id, Some(file_path.to_string()))
            {
                tracing::warn!("Failed to add backup history record: {}", e);
            }
        }

        write_result.map(|_| true)
    }

    /// 从文件恢复机器码
    pub async fn restore_machine_id(&self, file_path: &str) -> Result<MachineIdResult, String> {
        let backup_content = fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read backup file: {}", e))?;

        let backup: MachineIdBackup = serde_json::from_str(&backup_content)
            .map_err(|e| format!("Failed to parse backup file: {}", e))?;

        self.set_machine_id(&backup.machine_id).await
    }

    // === Windows 特定实现 ===

    #[cfg(target_os = "windows")]
    async fn get_windows_machine_id(
        &self,
        requires_admin: bool,
        backup_exists: bool,
    ) -> Result<MachineIdInfo, String> {
        let machine_id = self.read_windows_registry_machine_id()?;

        Ok(MachineIdInfo {
            current_id: machine_id,
            original_id: self.get_original_backup(),
            platform: "Windows".to_string(),
            can_modify: true,
            requires_admin,
            backup_exists,
            format_type: MachineIdFormat::Uuid,
        })
    }

    #[cfg(not(target_os = "windows"))]
    async fn get_windows_machine_id(
        &self,
        _requires_admin: bool,
        _backup_exists: bool,
    ) -> Result<MachineIdInfo, String> {
        Err("Windows machine ID not supported on this platform".to_string())
    }

    #[cfg(target_os = "windows")]
    fn read_windows_registry_machine_id(&self) -> Result<String, String> {
        use winreg::{enums::*, RegKey};

        let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
        let key = hklm.open_subkey_with_flags("SOFTWARE\\Microsoft\\Cryptography", KEY_READ)
            .map_err(|e| {
                tracing::error!("Failed to open Windows registry key: {}", e);
                format!("Failed to access Windows machine ID registry key: {}. This may indicate a system configuration issue.", e)
            })?;

        let machine_guid: String = key.get_value("MachineGuid")
            .map_err(|e| {
                tracing::error!("Failed to read MachineGuid from registry: {}", e);
                format!("Failed to read Windows MachineGuid: {}. The registry value may be missing or corrupted.", e)
            })?;

        // 验证读取到的GUID格式
        let cleaned_guid = machine_guid.trim();
        if cleaned_guid.is_empty() {
            return Err(
                "Windows MachineGuid is empty. This indicates a system configuration problem."
                    .to_string(),
            );
        }

        // 基本格式验证
        let format = MachineIdFormat::detect(cleaned_guid);
        if matches!(format, MachineIdFormat::Unknown) {
            tracing::warn!(
                "Windows MachineGuid has unexpected format: {}",
                cleaned_guid
            );
            // 不返回错误，因为某些Windows系统可能有非标准格式
        }

        Ok(cleaned_guid.to_string())
    }

    #[cfg(target_os = "windows")]
    async fn set_windows_machine_id(&self, new_id: &str) -> Result<MachineIdResult, String> {
        // Windows 需要管理员权限来修改注册表
        let admin_status = self.check_windows_admin().await?;
        if !admin_status.is_admin {
            return Ok(MachineIdResult {
                success: false,
                message: "Administrator privileges required to modify Windows machine ID"
                    .to_string(),
                requires_restart: false,
                requires_admin: true,
                new_machine_id: None,
            });
        }

        match self.write_windows_registry_machine_id(new_id) {
            Ok(_) => Ok(MachineIdResult {
                success: true,
                message: "Machine ID updated successfully. Restart may be required for some applications.".to_string(),
                requires_restart: true,
                requires_admin: false,
                new_machine_id: Some(new_id.to_string()),
            }),
            Err(e) => Ok(MachineIdResult {
                success: false,
                message: e,
                requires_restart: false,
                requires_admin: true,
                new_machine_id: None,
            }),
        }
    }

    #[cfg(target_os = "windows")]
    fn write_windows_registry_machine_id(&self, new_id: &str) -> Result<(), String> {
        let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
        let (key, _) = hklm
            .create_subkey("SOFTWARE\\Microsoft\\Cryptography")
            .map_err(|e| format!("Failed to create/open registry key: {}", e))?;

        key.set_value("MachineGuid", &new_id)
            .map_err(|e| format!("Failed to set MachineGuid: {}", e))?;

        Ok(())
    }

    #[cfg(not(target_os = "windows"))]
    fn write_windows_registry_machine_id(&self, _new_id: &str) -> Result<(), String> {
        Err("Windows registry modification not supported on this platform".to_string())
    }

    #[cfg(not(target_os = "windows"))]
    fn read_windows_registry_machine_id(&self) -> Result<String, String> {
        Err("Windows registry reading not supported on this platform".to_string())
    }

    #[cfg(target_os = "windows")]
    async fn check_windows_admin(&self) -> Result<AdminStatus, String> {
        // 尝试多种方法检查管理员权限

        // 方法1：尝试打开需要管理员权限的注册表项
        let registry_check = {
            use winreg::{enums::*, RegKey};
            let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
            hklm.open_subkey_with_flags("SOFTWARE\\Microsoft\\Cryptography", KEY_WRITE)
                .is_ok()
        };

        // 方法2：运行 net session 命令作为后备
        let net_session_check = Command::new("net")
            .args(&["session"])
            .output()
            .map(|result| result.status.success())
            .unwrap_or(false);

        // 方法3：使用 Windows API 检查 (需要额外实现)
        // 这里可以添加更复杂的 Windows API 调用

        let is_admin = registry_check || net_session_check;

        Ok(AdminStatus {
            is_admin,
            platform: "Windows".to_string(),
            elevation_method: Some("Right-click and select 'Run as administrator'".to_string()),
            check_success: true,
        })
    }

    #[cfg(not(target_os = "windows"))]
    async fn check_windows_admin(&self) -> Result<AdminStatus, String> {
        Err("Windows admin check not supported on this platform".to_string())
    }

    #[cfg(not(target_os = "windows"))]
    async fn set_windows_machine_id(&self, _new_id: &str) -> Result<MachineIdResult, String> {
        Ok(MachineIdResult {
            success: false,
            message: "Windows machine ID modification not supported on this platform".to_string(),
            requires_restart: false,
            requires_admin: false,
            new_machine_id: None,
        })
    }

    // === macOS 特定实现 ===

    #[cfg(target_os = "macos")]
    async fn get_macos_machine_id(
        &self,
        requires_admin: bool,
        backup_exists: bool,
    ) -> Result<MachineIdInfo, String> {
        let machine_id = self.read_macos_machine_id().await?;

        Ok(MachineIdInfo {
            current_id: machine_id,
            original_id: self.get_original_backup(),
            platform: "macOS".to_string(),
            can_modify: true, // macOS 使用应用层覆盖
            requires_admin,
            backup_exists,
            format_type: MachineIdFormat::Uuid,
        })
    }

    #[cfg(not(target_os = "macos"))]
    async fn get_macos_machine_id(
        &self,
        _requires_admin: bool,
        _backup_exists: bool,
    ) -> Result<MachineIdInfo, String> {
        Err("macOS machine ID not supported on this platform".to_string())
    }

    #[cfg(target_os = "macos")]
    async fn read_macos_machine_id(&self) -> Result<String, String> {
        // 首先检查是否有应用层覆盖
        if let Ok(override_id) = self.read_macos_override() {
            return Ok(override_id);
        }

        // 读取系统原始 UUID
        let output = Command::new("ioreg")
            .args(&["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .map_err(|e| format!("Failed to execute ioreg: {}", e))?;

        let output_str = String::from_utf8_lossy(&output.stdout);

        for line in output_str.lines() {
            if line.contains("IOPlatformUUID") {
                if let Some(uuid_part) = line.split('"').nth(3) {
                    return Ok(uuid_part.to_string());
                }
            }
        }

        Err("Failed to find IOPlatformUUID in ioreg output".to_string())
    }

    #[cfg(target_os = "macos")]
    async fn set_macos_machine_id(&self, new_id: &str) -> Result<MachineIdResult, String> {
        // macOS 使用应用层覆盖文件
        match self.write_macos_override(new_id) {
            Ok(_) => Ok(MachineIdResult {
                success: true,
                message: "Machine ID override created successfully. Applications using this override will see the new ID.".to_string(),
                requires_restart: false,
                requires_admin: false,
                new_machine_id: Some(new_id.to_string()),
            }),
            Err(e) => Ok(MachineIdResult {
                success: false,
                message: e,
                requires_restart: false,
                requires_admin: false,
                new_machine_id: None,
            }),
        }
    }

    #[cfg(target_os = "macos")]
    fn read_macos_override(&self) -> Result<String, String> {
        let override_file = dirs::data_dir()
            .ok_or("Failed to get app data directory")?
            .join("proxycast")
            .join("machine-id-override");

        fs::read_to_string(override_file).map_err(|e| format!("No override file found: {}", e))
    }

    #[cfg(target_os = "macos")]
    fn write_macos_override(&self, new_id: &str) -> Result<(), String> {
        let override_dir = dirs::data_dir()
            .ok_or("Failed to get app data directory")?
            .join("proxycast");

        fs::create_dir_all(&override_dir)
            .map_err(|e| format!("Failed to create override directory: {}", e))?;

        let override_file = override_dir.join("machine-id-override");
        fs::write(override_file, new_id)
            .map_err(|e| format!("Failed to write override file: {}", e))?;

        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn check_macos_admin(&self) -> Result<AdminStatus, String> {
        // macOS 不需要管理员权限（使用应用层覆盖）
        Ok(AdminStatus {
            is_admin: true, // 应用层覆盖不需要管理员权限
            platform: "macOS".to_string(),
            elevation_method: None,
            check_success: true,
        })
    }

    // === macOS 存根实现 (非 macOS 平台) ===

    #[cfg(not(target_os = "macos"))]
    async fn set_macos_machine_id(&self, _new_id: &str) -> Result<MachineIdResult, String> {
        Ok(MachineIdResult {
            success: false,
            message: "macOS machine ID modification not supported on this platform".to_string(),
            requires_restart: false,
            requires_admin: false,
            new_machine_id: None,
        })
    }

    #[cfg(not(target_os = "macos"))]
    async fn check_macos_admin(&self) -> Result<AdminStatus, String> {
        Ok(AdminStatus {
            is_admin: false,
            platform: "non-macOS".to_string(),
            elevation_method: None,
            check_success: false,
        })
    }

    // === Linux 特定实现 ===

    #[cfg(target_os = "linux")]
    async fn get_linux_machine_id(
        &self,
        requires_admin: bool,
        backup_exists: bool,
    ) -> Result<MachineIdInfo, String> {
        let machine_id = self.read_linux_machine_id()?;

        Ok(MachineIdInfo {
            current_id: machine_id,
            original_id: self.get_original_backup(),
            platform: "Linux".to_string(),
            can_modify: true,
            requires_admin,
            backup_exists,
            format_type: MachineIdFormat::Hex32,
        })
    }

    #[cfg(not(target_os = "linux"))]
    async fn get_linux_machine_id(
        &self,
        _requires_admin: bool,
        _backup_exists: bool,
    ) -> Result<MachineIdInfo, String> {
        Err("Linux machine ID not supported on this platform".to_string())
    }

    #[cfg(target_os = "linux")]
    fn read_linux_machine_id(&self) -> Result<String, String> {
        // 尝试读取 /etc/machine-id
        if let Ok(content) = fs::read_to_string("/etc/machine-id") {
            return Ok(content.trim().to_string());
        }

        // 尝试读取 /var/lib/dbus/machine-id
        if let Ok(content) = fs::read_to_string("/var/lib/dbus/machine-id") {
            return Ok(content.trim().to_string());
        }

        Err(
            "Failed to read Linux machine ID from /etc/machine-id or /var/lib/dbus/machine-id"
                .to_string(),
        )
    }

    #[cfg(target_os = "linux")]
    async fn set_linux_machine_id(&self, new_id: &str) -> Result<MachineIdResult, String> {
        let admin_status = self.check_linux_admin().await?;
        if !admin_status.is_admin {
            return Ok(MachineIdResult {
                success: false,
                message: "Root privileges required to modify Linux machine ID".to_string(),
                requires_restart: false,
                requires_admin: true,
                new_machine_id: None,
            });
        }

        // 转换为 32 位十六进制格式（去除连字符）
        let hex_id = new_id.replace("-", "").to_lowercase();

        // 尝试写入 /etc/machine-id
        match fs::write("/etc/machine-id", &hex_id) {
            Ok(_) => {
                // 同时更新 /var/lib/dbus/machine-id（如果存在）
                let _ = fs::write("/var/lib/dbus/machine-id", &hex_id);

                Ok(MachineIdResult {
                    success: true,
                    message: "Machine ID updated successfully. Some services may require restart."
                        .to_string(),
                    requires_restart: true,
                    requires_admin: false,
                    new_machine_id: Some(hex_id),
                })
            }
            Err(e) => Ok(MachineIdResult {
                success: false,
                message: format!("Failed to write machine ID: {}", e),
                requires_restart: false,
                requires_admin: true,
                new_machine_id: None,
            }),
        }
    }

    #[cfg(target_os = "linux")]
    async fn check_linux_admin(&self) -> Result<AdminStatus, String> {
        let output = Command::new("id")
            .args(&["-u"])
            .output()
            .map_err(|e| format!("Failed to check user ID: {}", e))?;

        let uid_str = String::from_utf8_lossy(&output.stdout);
        let uid: u32 = uid_str
            .trim()
            .parse()
            .map_err(|e| format!("Failed to parse user ID: {}", e))?;

        Ok(AdminStatus {
            is_admin: uid == 0,
            platform: "Linux".to_string(),
            elevation_method: Some("sudo or run as root".to_string()),
            check_success: true,
        })
    }

    #[cfg(not(target_os = "linux"))]
    async fn check_linux_admin(&self) -> Result<AdminStatus, String> {
        Err("Linux admin check not supported on this platform".to_string())
    }

    #[cfg(not(target_os = "linux"))]
    async fn set_linux_machine_id(&self, _new_id: &str) -> Result<MachineIdResult, String> {
        Ok(MachineIdResult {
            success: false,
            message: "Linux machine ID modification not supported on this platform".to_string(),
            requires_restart: false,
            requires_admin: false,
            new_machine_id: None,
        })
    }

    // === 通用辅助方法 ===

    async fn check_requires_admin(&self) -> Result<bool, String> {
        let admin_status = self.check_admin_privileges().await?;
        Ok(!admin_status.is_admin && admin_status.check_success)
    }

    fn check_backup_exists(&self) -> bool {
        let backup_file = self.backup_dir.join("original_machine_id.json");
        backup_file.exists()
    }

    fn get_original_backup(&self) -> Option<String> {
        let backup_file = self.backup_dir.join("original_machine_id.json");
        if let Ok(content) = fs::read_to_string(backup_file) {
            if let Ok(backup) = serde_json::from_str::<MachineIdBackup>(&content) {
                return Some(backup.machine_id);
            }
        }
        None
    }

    async fn create_auto_backup(&self) -> Result<(), String> {
        // 如果已有备份则跳过
        if self.check_backup_exists() {
            return Ok(());
        }

        let current_info = self.get_current_machine_id().await?;
        let backup_file = self.backup_dir.join("original_machine_id.json");

        let backup = MachineIdBackup {
            machine_id: current_info.current_id,
            timestamp: chrono::Utc::now().timestamp(),
            platform: current_info.platform,
            format: current_info.format_type,
            description: Some("Auto backup before first modification".to_string()),
        };

        let backup_json = serde_json::to_string_pretty(&backup)
            .map_err(|e| format!("Failed to serialize backup: {}", e))?;

        fs::write(backup_file, backup_json)
            .map_err(|e| format!("Failed to create auto backup: {}", e))?;

        Ok(())
    }

    // === 历史记录管理 ===

    /// 加载历史记录
    fn load_history(&self) -> Result<Vec<MachineIdHistory>, String> {
        if !self.history_file.exists() {
            return Ok(vec![]);
        }

        let content = fs::read_to_string(&self.history_file)
            .map_err(|e| format!("Failed to read history file: {}", e))?;

        let history: Vec<MachineIdHistory> =
            serde_json::from_str(&content).unwrap_or_else(|_| vec![]);

        Ok(history)
    }

    /// 保存历史记录
    fn save_history(&self, history: &[MachineIdHistory]) -> Result<(), String> {
        let history_json = serde_json::to_string_pretty(history)
            .map_err(|e| format!("Failed to serialize history: {}", e))?;

        fs::write(&self.history_file, history_json)
            .map_err(|e| format!("Failed to save history: {}", e))?;

        Ok(())
    }

    /// 添加历史记录
    pub fn add_history_record(
        &self,
        machine_id: String,
        backup_path: Option<String>,
    ) -> Result<(), String> {
        let mut history = self.load_history().unwrap_or_else(|_| vec![]);

        let record = MachineIdHistory {
            id: uuid::Uuid::new_v4().to_string(),
            machine_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            platform: Self::get_os_type(),
            backup_path,
        };

        history.push(record);

        // 保留最近100条记录
        if history.len() > 100 {
            history.drain(0..history.len() - 100);
        }

        self.save_history(&history)?;
        Ok(())
    }

    /// 获取历史记录
    pub fn get_history(&self) -> Result<Vec<MachineIdHistory>, String> {
        self.load_history()
    }
}
