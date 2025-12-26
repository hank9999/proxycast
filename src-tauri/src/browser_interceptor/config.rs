use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 拦截器状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterceptorState {
    pub enabled: bool,
    pub active_hooks: Vec<String>,
    pub intercepted_count: u32,
    pub last_activity: Option<DateTime<Utc>>,
    pub can_restore: bool, // 是否可以恢复正常状态
}

impl Default for InterceptorState {
    fn default() -> Self {
        Self {
            enabled: false,
            active_hooks: Vec::new(),
            intercepted_count: 0,
            last_activity: None,
            can_restore: false,
        }
    }
}

/// 被拦截的 URL 信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterceptedUrl {
    pub id: String,
    pub url: String,
    pub source_process: String,
    pub timestamp: DateTime<Utc>,
    pub copied: bool,
    pub opened_in_browser: bool,
    pub dismissed: bool,
}

impl InterceptedUrl {
    pub fn new(url: String, source_process: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            url,
            source_process,
            timestamp: Utc::now(),
            copied: false,
            opened_in_browser: false,
            dismissed: false,
        }
    }
}

/// 指纹浏览器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintBrowserConfig {
    pub enabled: bool,
    pub executable_path: String,
    pub profile_path: String,
    pub additional_args: Vec<String>,
}

impl Default for FingerprintBrowserConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            executable_path: String::new(),
            profile_path: String::new(),
            additional_args: Vec::new(),
        }
    }
}

/// 恢复机制配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    pub backup_system_state: bool,
    pub emergency_recovery_hotkey: String,
    pub auto_recovery_on_crash: bool,
    pub recovery_timeout: u64, // 恢复操作超时时间（秒）
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            backup_system_state: true,
            emergency_recovery_hotkey: "Ctrl+Alt+Shift+R".to_string(),
            auto_recovery_on_crash: true,
            recovery_timeout: 30,
        }
    }
}

/// 浏览器拦截器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserInterceptorConfig {
    pub enabled: bool,
    pub target_processes: Vec<String>,
    pub url_patterns: Vec<String>,
    pub excluded_processes: Vec<String>,
    pub notification_enabled: bool,
    pub auto_copy_to_clipboard: bool,
    pub auto_launch_browser: bool,
    pub restore_on_exit: bool,
    pub temporary_disable_timeout: Option<u64>, // 临时禁用超时（秒）
    pub fingerprint_browser: FingerprintBrowserConfig,
    pub recovery: RecoveryConfig,
}

impl Default for BrowserInterceptorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_processes: vec![
                "kiro".to_string(),
                "kiro.exe".to_string(),
                "cursor".to_string(),
                "cursor.exe".to_string(),
                "code".to_string(),
                "code.exe".to_string(),
                "Kiro".to_string(), // macOS 应用名称通常首字母大写
                "Cursor".to_string(),
                "Visual Studio Code".to_string(),
            ],
            url_patterns: vec![
                "https://auth.*".to_string(),
                "https://*/oauth/*".to_string(),
                "https://accounts.google.com/*".to_string(),
                "https://github.com/login/*".to_string(),
                "https://login.microsoftonline.com/*".to_string(),
            ],
            excluded_processes: vec![
                "explorer.exe".to_string(),
                "winlogon.exe".to_string(),
                "system".to_string(),
                "chrome.exe".to_string(),
                "firefox.exe".to_string(),
                "safari".to_string(),
                "Safari".to_string(),        // macOS Safari
                "Google Chrome".to_string(), // macOS Chrome
                "Firefox".to_string(),       // macOS Firefox
            ],
            notification_enabled: true,
            auto_copy_to_clipboard: true,
            auto_launch_browser: false,
            restore_on_exit: true,
            temporary_disable_timeout: Some(300), // 5分钟
            fingerprint_browser: FingerprintBrowserConfig::default(),
            recovery: RecoveryConfig::default(),
        }
    }
}

impl BrowserInterceptorConfig {
    /// 检查进程是否在目标列表中
    pub fn is_target_process(&self, process_name: &str) -> bool {
        self.target_processes.iter().any(|pattern| {
            // 支持简单的通配符匹配
            if pattern.contains('*') {
                // TODO: 实现更复杂的模式匹配
                process_name.contains(&pattern.replace('*', ""))
            } else {
                process_name.eq_ignore_ascii_case(pattern)
            }
        })
    }

    /// 检查进程是否在排除列表中
    pub fn is_excluded_process(&self, process_name: &str) -> bool {
        self.excluded_processes.iter().any(|pattern| {
            if pattern.contains('*') {
                process_name.contains(&pattern.replace('*', ""))
            } else {
                process_name.eq_ignore_ascii_case(pattern)
            }
        })
    }

    /// 检查 URL 是否匹配拦截模式
    pub fn matches_url_pattern(&self, url: &str) -> bool {
        self.url_patterns.iter().any(|pattern| {
            // 简单的模式匹配实现
            if pattern.contains('*') {
                let parts: Vec<&str> = pattern.split('*').collect();
                if parts.len() == 2 {
                    url.starts_with(parts[0]) && url.ends_with(parts[1])
                } else {
                    // 更复杂的模式匹配
                    url.contains(&pattern.replace('*', ""))
                }
            } else {
                url.starts_with(pattern)
            }
        })
    }

    /// 验证配置的有效性
    pub fn validate(&self) -> Result<(), String> {
        if self.target_processes.is_empty() {
            return Err("目标进程列表不能为空".to_string());
        }

        if self.url_patterns.is_empty() {
            return Err("URL 模式列表不能为空".to_string());
        }

        if self.fingerprint_browser.enabled && self.fingerprint_browser.executable_path.is_empty() {
            return Err("启用指纹浏览器时必须指定可执行文件路径".to_string());
        }

        if let Some(timeout) = self.temporary_disable_timeout {
            if timeout == 0 {
                return Err("临时禁用超时时间必须大于 0".to_string());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_target_process() {
        let config = BrowserInterceptorConfig::default();

        assert!(config.is_target_process("kiro"));
        assert!(config.is_target_process("KIRO.EXE"));
        assert!(config.is_target_process("cursor"));
        assert!(!config.is_target_process("notepad"));
    }

    #[test]
    fn test_is_excluded_process() {
        let config = BrowserInterceptorConfig::default();

        assert!(config.is_excluded_process("chrome.exe"));
        assert!(config.is_excluded_process("FIREFOX.EXE"));
        assert!(!config.is_excluded_process("kiro"));
    }

    #[test]
    fn test_matches_url_pattern() {
        let config = BrowserInterceptorConfig::default();

        assert!(config.matches_url_pattern("https://accounts.google.com/oauth/authorize"));
        assert!(config.matches_url_pattern("https://github.com/login/oauth"));
        assert!(config.matches_url_pattern("https://auth.example.com/login"));
        assert!(!config.matches_url_pattern("https://example.com/normal-page"));
    }

    #[test]
    fn test_config_validation() {
        let mut config = BrowserInterceptorConfig::default();
        assert!(config.validate().is_ok());

        config.target_processes.clear();
        assert!(config.validate().is_err());

        config = BrowserInterceptorConfig::default();
        config.fingerprint_browser.enabled = true;
        config.fingerprint_browser.executable_path = String::new();
        assert!(config.validate().is_err());
    }
}
