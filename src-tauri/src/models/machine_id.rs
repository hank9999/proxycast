use serde::{Deserialize, Serialize};

/// 机器码信息结构 - v0.20.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineIdInfo {
    /// 当前机器码
    pub current_id: String,
    /// 原始机器码（如果有备份）
    pub original_id: Option<String>,
    /// 操作系统平台
    pub platform: String,
    /// 是否可以修改
    pub can_modify: bool,
    /// 是否需要管理员权限
    pub requires_admin: bool,
    /// 是否存在备份
    pub backup_exists: bool,
    /// 机器码格式类型
    pub format_type: MachineIdFormat,
}

/// 机器码操作结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineIdResult {
    /// 操作是否成功
    pub success: bool,
    /// 结果消息
    pub message: String,
    /// 是否需要重启
    pub requires_restart: bool,
    /// 是否需要管理员权限
    pub requires_admin: bool,
    /// 新的机器码（如果操作成功）
    pub new_machine_id: Option<String>,
}

/// 管理员权限状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminStatus {
    /// 是否具有管理员权限
    pub is_admin: bool,
    /// 操作系统平台
    pub platform: String,
    /// 权限提升方法说明
    pub elevation_method: Option<String>,
    /// 权限检查是否成功
    pub check_success: bool,
}

/// 机器码格式类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MachineIdFormat {
    /// UUID 格式 (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    Uuid,
    /// 32位十六进制格式 (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)
    #[serde(rename = "hex32")]
    Hex32,
    /// 其他格式
    #[serde(rename = "unknown")]
    Unknown,
}

/// 机器码备份信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineIdBackup {
    /// 备份的机器码
    pub machine_id: String,
    /// 备份时间戳
    pub timestamp: i64,
    /// 操作系统平台
    pub platform: String,
    /// 机器码格式
    pub format: MachineIdFormat,
    /// 备份描述
    pub description: Option<String>,
}

/// 机器码历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineIdHistory {
    /// 记录ID
    pub id: String,
    /// 机器码
    pub machine_id: String,
    /// 操作时间戳
    pub timestamp: String,
    /// 操作系统平台
    pub platform: String,
    /// 备份路径（可选）
    pub backup_path: Option<String>,
}

/// 机器码操作类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MachineIdOperation {
    /// 获取当前机器码
    Get,
    /// 设置新机器码
    Set,
    /// 生成随机机器码
    Generate,
    /// 备份机器码
    Backup,
    /// 恢复机器码
    Restore,
    /// 重置为原始机器码
    Reset,
}

/// 机器码验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineIdValidation {
    /// 是否有效
    pub is_valid: bool,
    /// 检测到的格式
    pub detected_format: MachineIdFormat,
    /// 验证错误信息
    pub error_message: Option<String>,
    /// 格式化后的机器码（如果有效）
    pub formatted_id: Option<String>,
}

impl MachineIdFormat {
    /// 从字符串检测机器码格式
    pub fn detect(machine_id: &str) -> Self {
        let cleaned = machine_id.replace("-", "").replace(" ", "").to_lowercase();

        // 检查UUID格式：8-4-4-4-12个十六进制字符
        if machine_id.contains("-") && machine_id.len() == 36 {
            let parts: Vec<&str> = machine_id.split('-').collect();
            if parts.len() == 5
                && parts[0].len() == 8
                && parts[1].len() == 4
                && parts[2].len() == 4
                && parts[3].len() == 4
                && parts[4].len() == 12
                && cleaned.chars().all(|c| c.is_ascii_hexdigit())
            {
                return MachineIdFormat::Uuid;
            }
        }

        // 检查32位十六进制格式
        if cleaned.len() == 32 && cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
            return MachineIdFormat::Hex32;
        }

        MachineIdFormat::Unknown
    }

    /// 格式化机器码为标准格式
    pub fn format_machine_id(&self, machine_id: &str) -> Result<String, String> {
        let cleaned = machine_id.replace("-", "").replace(" ", "").to_lowercase();

        match self {
            MachineIdFormat::Uuid => {
                if cleaned.len() != 32 {
                    return Err("UUID format requires 32 hex characters".to_string());
                }
                if !cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err("UUID format requires hex characters only".to_string());
                }
                Ok(format!(
                    "{}-{}-{}-{}-{}",
                    &cleaned[0..8],
                    &cleaned[8..12],
                    &cleaned[12..16],
                    &cleaned[16..20],
                    &cleaned[20..32]
                ))
            }
            MachineIdFormat::Hex32 => {
                if cleaned.len() != 32 {
                    return Err("Hex32 format requires 32 hex characters".to_string());
                }
                if !cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err("Hex32 format requires hex characters only".to_string());
                }
                Ok(cleaned)
            }
            MachineIdFormat::Unknown => Err("Cannot format unknown machine ID format".to_string()),
        }
    }
}

impl std::fmt::Display for MachineIdFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MachineIdFormat::Uuid => write!(f, "uuid"),
            MachineIdFormat::Hex32 => write!(f, "hex32"),
            MachineIdFormat::Unknown => write!(f, "unknown"),
        }
    }
}

impl std::fmt::Display for MachineIdOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MachineIdOperation::Get => write!(f, "Get"),
            MachineIdOperation::Set => write!(f, "Set"),
            MachineIdOperation::Generate => write!(f, "Generate"),
            MachineIdOperation::Backup => write!(f, "Backup"),
            MachineIdOperation::Restore => write!(f, "Restore"),
            MachineIdOperation::Reset => write!(f, "Reset"),
        }
    }
}
