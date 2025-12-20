//! 配置导入服务
//!
//! 提供配置和凭证的统一导入功能，支持：
//! - YAML 配置导入
//! - 完整导入包导入（配置 + 凭证 + OAuth Token 文件）
//! - 导入验证（格式、版本、脱敏状态）
//! - 合并和替换模式

use super::export::{base64_decode, ExportBundle, REDACTED_PLACEHOLDER};
use super::path_utils::expand_tilde;
use super::types::{ApiKeyEntry, Config, CredentialEntry, CredentialPoolConfig};
use super::yaml::{ConfigError, ConfigManager, YamlService};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 导入选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportOptions {
    /// 是否合并（false 则替换）
    pub merge: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self { merge: true }
    }
}

impl ImportOptions {
    /// 创建合并模式选项
    pub fn merge() -> Self {
        Self { merge: true }
    }

    /// 创建替换模式选项
    pub fn replace() -> Self {
        Self { merge: false }
    }
}

/// 验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// 是否有效
    pub valid: bool,
    /// 格式版本
    pub version: Option<String>,
    /// 是否已脱敏
    pub redacted: bool,
    /// 是否包含配置
    pub has_config: bool,
    /// 是否包含凭证
    pub has_credentials: bool,
    /// 错误信息列表
    pub errors: Vec<String>,
    /// 警告信息列表
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// 创建有效的验证结果
    pub fn valid() -> Self {
        Self {
            valid: true,
            version: None,
            redacted: false,
            has_config: false,
            has_credentials: false,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// 创建无效的验证结果
    pub fn invalid(error: impl Into<String>) -> Self {
        Self {
            valid: false,
            version: None,
            redacted: false,
            has_config: false,
            has_credentials: false,
            errors: vec![error.into()],
            warnings: Vec::new(),
        }
    }

    /// 添加错误
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
        self.valid = false;
    }

    /// 添加警告
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
}

/// 导入结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// 是否成功
    pub success: bool,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 导入的配置
    pub config: Config,
}

impl ImportResult {
    /// 创建成功的导入结果
    pub fn success(config: Config) -> Self {
        Self {
            success: true,
            warnings: Vec::new(),
            config,
        }
    }

    /// 创建带警告的成功导入结果
    pub fn success_with_warnings(config: Config, warnings: Vec<String>) -> Self {
        Self {
            success: true,
            warnings,
            config,
        }
    }
}

/// 导入错误类型
#[derive(Debug, Clone)]
pub enum ImportError {
    /// 格式错误
    FormatError(String),
    /// 版本不兼容
    VersionError(String),
    /// 配置错误
    ConfigError(String),
    /// IO 错误
    IoError(String),
    /// 验证错误
    ValidationError(String),
    /// 脱敏数据无法导入
    RedactedDataError(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::FormatError(msg) => write!(f, "格式错误: {}", msg),
            ImportError::VersionError(msg) => write!(f, "版本不兼容: {}", msg),
            ImportError::ConfigError(msg) => write!(f, "配置错误: {}", msg),
            ImportError::IoError(msg) => write!(f, "IO 错误: {}", msg),
            ImportError::ValidationError(msg) => write!(f, "验证错误: {}", msg),
            ImportError::RedactedDataError(msg) => write!(f, "脱敏数据无法导入: {}", msg),
        }
    }
}

impl std::error::Error for ImportError {}

impl From<ConfigError> for ImportError {
    fn from(err: ConfigError) -> Self {
        ImportError::ConfigError(err.to_string())
    }
}

impl From<std::io::Error> for ImportError {
    fn from(err: std::io::Error) -> Self {
        ImportError::IoError(err.to_string())
    }
}

/// 导入服务
///
/// 提供配置和凭证的统一导入功能
pub struct ImportService;

impl ImportService {
    /// 支持的导入格式版本
    pub const SUPPORTED_VERSIONS: &'static [&'static str] = &["1.0"];

    /// 验证导入内容
    ///
    /// # Arguments
    /// * `content` - 导入内容（JSON 格式的 ExportBundle 或 YAML 配置）
    ///
    /// # Returns
    /// * `ValidationResult` - 验证结果
    pub fn validate(content: &str) -> ValidationResult {
        // 首先尝试解析为 ExportBundle (JSON)
        if let Ok(bundle) = ExportBundle::from_json(content) {
            return Self::validate_bundle(&bundle);
        }

        // 尝试解析为 YAML 配置
        if let Ok(_config) = ConfigManager::parse_yaml(content) {
            let mut result = ValidationResult::valid();
            result.has_config = true;
            result.has_credentials = false;
            result.version = Some("yaml".to_string());
            return result;
        }

        ValidationResult::invalid(
            "无法解析导入内容：既不是有效的 JSON 导出包，也不是有效的 YAML 配置",
        )
    }

    /// 验证导出包
    fn validate_bundle(bundle: &ExportBundle) -> ValidationResult {
        let mut result = ValidationResult::valid();
        result.version = Some(bundle.version.clone());
        result.redacted = bundle.redacted;
        result.has_config = bundle.has_config();
        result.has_credentials = bundle.has_credentials();

        // 检查版本兼容性
        if !Self::SUPPORTED_VERSIONS.contains(&bundle.version.as_str()) {
            result.add_warning(format!(
                "导出包版本 {} 可能不完全兼容，支持的版本: {:?}",
                bundle.version,
                Self::SUPPORTED_VERSIONS
            ));
        }

        // 检查脱敏状态
        if bundle.redacted {
            result.add_warning("导出包已脱敏，凭证数据无法恢复");
        }

        // 验证配置内容（如果存在）
        if let Some(ref yaml) = bundle.config_yaml {
            if let Err(e) = ConfigManager::parse_yaml(yaml) {
                result.add_error(format!("配置 YAML 解析失败: {}", e));
            }
        }

        result
    }

    /// 导入 YAML 配置
    ///
    /// # Arguments
    /// * `yaml` - YAML 配置字符串
    /// * `current_config` - 当前配置（用于合并模式）
    /// * `options` - 导入选项
    ///
    /// # Returns
    /// * `Ok(ImportResult)` - 导入成功
    /// * `Err(ImportError)` - 导入失败
    pub fn import_yaml(
        yaml: &str,
        current_config: &Config,
        options: &ImportOptions,
    ) -> Result<ImportResult, ImportError> {
        // 解析 YAML
        let imported_config = ConfigManager::parse_yaml(yaml)?;

        // 根据选项合并或替换
        let final_config = if options.merge {
            Self::merge_configs(current_config, &imported_config)
        } else {
            imported_config
        };

        Ok(ImportResult::success(final_config))
    }

    /// 导入完整的导出包
    ///
    /// # Arguments
    /// * `bundle` - 导出包
    /// * `current_config` - 当前配置（用于合并模式）
    /// * `options` - 导入选项
    /// * `auth_dir` - 认证目录路径（用于恢复 OAuth token 文件）
    ///
    /// # Returns
    /// * `Ok(ImportResult)` - 导入成功
    /// * `Err(ImportError)` - 导入失败
    pub fn import(
        bundle: &ExportBundle,
        current_config: &Config,
        options: &ImportOptions,
        auth_dir: &str,
    ) -> Result<ImportResult, ImportError> {
        let mut warnings = Vec::new();

        // 检查脱敏状态
        if bundle.redacted {
            warnings.push("导出包已脱敏，凭证数据将使用占位符".to_string());
        }

        // 导入配置
        let mut config = if let Some(ref yaml) = bundle.config_yaml {
            let imported = ConfigManager::parse_yaml(yaml)?;
            if options.merge {
                Self::merge_configs(current_config, &imported)
            } else {
                imported
            }
        } else if options.merge {
            current_config.clone()
        } else {
            Config::default()
        };

        // 恢复 OAuth token 文件
        if !bundle.token_files.is_empty() {
            let token_warnings = Self::restore_token_files(&bundle.token_files, auth_dir)?;
            warnings.extend(token_warnings);
        }

        // 如果是脱敏数据，清理凭证池中的占位符
        if bundle.redacted {
            let server_key_cleared = Self::clean_redacted_credentials(&mut config);
            if server_key_cleared {
                warnings.push("检测到脱敏的服务器 API Key，已清空，需要手动设置".to_string());
            }
        }

        Ok(ImportResult::success_with_warnings(config, warnings))
    }

    /// 合并配置
    ///
    /// 将导入的配置合并到当前配置中
    fn merge_configs(current: &Config, imported: &Config) -> Config {
        let mut merged = current.clone();

        // 合并服务器配置（导入的覆盖当前的）
        merged.server = imported.server.clone();

        // 合并 Provider 配置
        merged.providers = imported.providers.clone();

        // 合并路由配置
        merged.routing = imported.routing.clone();
        merged.default_provider = imported.default_provider.clone();

        // 合并重试配置
        merged.retry = imported.retry.clone();

        // 合并日志配置
        merged.logging = imported.logging.clone();

        // 合并注入配置
        merged.injection = imported.injection.clone();

        // 合并 auth_dir
        merged.auth_dir = imported.auth_dir.clone();

        // 合并凭证池（添加新的，保留现有的）
        merged.credential_pool =
            Self::merge_credential_pools(&current.credential_pool, &imported.credential_pool);

        merged
    }

    /// 合并凭证池
    ///
    /// 将导入的凭证添加到现有凭证池中（按 ID 去重）
    fn merge_credential_pools(
        current: &CredentialPoolConfig,
        imported: &CredentialPoolConfig,
    ) -> CredentialPoolConfig {
        CredentialPoolConfig {
            kiro: Self::merge_credential_entries(&current.kiro, &imported.kiro),
            gemini: Self::merge_credential_entries(&current.gemini, &imported.gemini),
            qwen: Self::merge_credential_entries(&current.qwen, &imported.qwen),
            openai: Self::merge_api_key_entries(&current.openai, &imported.openai),
            claude: Self::merge_api_key_entries(&current.claude, &imported.claude),
            gemini_api_keys: imported.gemini_api_keys.clone(),
            vertex_api_keys: imported.vertex_api_keys.clone(),
            codex: Self::merge_credential_entries(&current.codex, &imported.codex),
            iflow: imported.iflow.clone(),
        }
    }

    /// 合并 OAuth 凭证条目
    fn merge_credential_entries(
        current: &[CredentialEntry],
        imported: &[CredentialEntry],
    ) -> Vec<CredentialEntry> {
        let mut result: Vec<CredentialEntry> = current.to_vec();

        for entry in imported {
            // 如果 ID 已存在，更新；否则添加
            if let Some(existing) = result.iter_mut().find(|e| e.id == entry.id) {
                *existing = entry.clone();
            } else {
                result.push(entry.clone());
            }
        }

        result
    }

    /// 合并 API Key 凭证条目
    fn merge_api_key_entries(
        current: &[ApiKeyEntry],
        imported: &[ApiKeyEntry],
    ) -> Vec<ApiKeyEntry> {
        let mut result: Vec<ApiKeyEntry> = current.to_vec();

        for entry in imported {
            // 跳过脱敏的条目
            if entry.api_key == REDACTED_PLACEHOLDER {
                continue;
            }

            // 如果 ID 已存在，更新；否则添加
            if let Some(existing) = result.iter_mut().find(|e| e.id == entry.id) {
                *existing = entry.clone();
            } else {
                result.push(entry.clone());
            }
        }

        result
    }

    /// 恢复 OAuth token 文件到 auth_dir
    ///
    /// # Arguments
    /// * `token_files` - token 文件映射（相对路径 -> base64 编码内容）
    /// * `auth_dir` - 认证目录路径
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - 警告信息列表
    fn restore_token_files(
        token_files: &std::collections::HashMap<String, String>,
        auth_dir: &str,
    ) -> Result<Vec<String>, ImportError> {
        let mut warnings = Vec::new();
        let auth_path = expand_tilde(auth_dir);

        // 确保 auth_dir 存在
        std::fs::create_dir_all(&auth_path)?;

        for (relative_path, base64_content) in token_files {
            let token_path = auth_path.join(relative_path);

            // 确保父目录存在
            if let Some(parent) = token_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // 解码 base64 内容
            match base64_decode(base64_content) {
                Ok(content) => {
                    // 检查是否是脱敏内容
                    if content == REDACTED_PLACEHOLDER.as_bytes() {
                        warnings.push(format!("Token 文件 {} 已脱敏，无法恢复", relative_path));
                        continue;
                    }

                    // 写入文件
                    if let Err(e) = std::fs::write(&token_path, &content) {
                        warnings.push(format!("写入 token 文件 {} 失败: {}", relative_path, e));
                    }
                }
                Err(e) => {
                    warnings.push(format!("解码 token 文件 {} 失败: {}", relative_path, e));
                }
            }
        }

        Ok(warnings)
    }

    /// 清理脱敏的凭证数据
    ///
    /// 移除凭证池中使用占位符的条目
    fn clean_redacted_credentials(config: &mut Config) -> bool {
        let mut server_key_cleared = false;

        // 清理 OpenAI 凭证池中的脱敏条目
        config
            .credential_pool
            .openai
            .retain(|e| e.api_key != REDACTED_PLACEHOLDER);

        // 清理 Claude 凭证池中的脱敏条目
        config
            .credential_pool
            .claude
            .retain(|e| e.api_key != REDACTED_PLACEHOLDER);

        // 清理 Provider 配置中的脱敏 API 密钥
        if config.providers.openai.api_key.as_deref() == Some(REDACTED_PLACEHOLDER) {
            config.providers.openai.api_key = None;
        }
        if config.providers.claude.api_key.as_deref() == Some(REDACTED_PLACEHOLDER) {
            config.providers.claude.api_key = None;
        }

        // 清理服务器 API 密钥（如果是脱敏的，清空并提示手动设置）
        if config.server.api_key == REDACTED_PLACEHOLDER {
            config.server.api_key = String::new();
            server_key_cleared = true;
        }

        server_key_cleared
    }

    /// 从文件导入配置
    ///
    /// # Arguments
    /// * `path` - 文件路径
    /// * `current_config` - 当前配置
    /// * `options` - 导入选项
    ///
    /// # Returns
    /// * `Ok(ImportResult)` - 导入成功
    /// * `Err(ImportError)` - 导入失败
    pub fn import_from_file(
        path: &Path,
        current_config: &Config,
        options: &ImportOptions,
    ) -> Result<ImportResult, ImportError> {
        let content = std::fs::read_to_string(path)?;

        // 首先尝试解析为 ExportBundle
        if let Ok(bundle) = ExportBundle::from_json(&content) {
            return Self::import(&bundle, current_config, options, &current_config.auth_dir);
        }

        // 尝试解析为 YAML
        Self::import_yaml(&content, current_config, options)
    }

    /// 保存导入的配置到文件
    ///
    /// # Arguments
    /// * `config` - 要保存的配置
    /// * `path` - 配置文件路径
    ///
    /// # Returns
    /// * `Ok(())` - 保存成功
    /// * `Err(ImportError)` - 保存失败
    pub fn save_config(config: &Config, path: &Path) -> Result<(), ImportError> {
        YamlService::save_preserve_comments(path, config)?;
        Ok(())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_import_options_default() {
        let options = ImportOptions::default();
        assert!(options.merge);
    }

    #[test]
    fn test_import_options_merge() {
        let options = ImportOptions::merge();
        assert!(options.merge);
    }

    #[test]
    fn test_import_options_replace() {
        let options = ImportOptions::replace();
        assert!(!options.merge);
    }

    #[test]
    fn test_validation_result_valid() {
        let result = ValidationResult::valid();
        assert!(result.valid);
        assert!(result.errors.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_validation_result_invalid() {
        let result = ValidationResult::invalid("test error");
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("test error"));
    }

    #[test]
    fn test_validation_result_add_error() {
        let mut result = ValidationResult::valid();
        result.add_error("error 1");
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_validation_result_add_warning() {
        let mut result = ValidationResult::valid();
        result.add_warning("warning 1");
        assert!(result.valid); // 警告不影响有效性
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_validate_valid_yaml() {
        let yaml = r#"
server:
  host: 127.0.0.1
  port: 8999
  api_key: test_key
"#;
        let result = ImportService::validate(yaml);
        assert!(result.valid);
        assert!(result.has_config);
        assert!(!result.has_credentials);
    }

    #[test]
    fn test_validate_invalid_content() {
        let content = "this is not valid yaml or json {{{";
        let result = ImportService::validate(content);
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validate_export_bundle() {
        let bundle = ExportBundle::new("1.0.0");
        let json = bundle.to_json().expect("序列化应成功");
        let result = ImportService::validate(&json);
        assert!(result.valid);
        assert_eq!(result.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_validate_redacted_bundle() {
        let mut bundle = ExportBundle::new("1.0.0");
        bundle.redacted = true;
        let json = bundle.to_json().expect("序列化应成功");
        let result = ImportService::validate(&json);
        assert!(result.valid);
        assert!(result.redacted);
        assert!(!result.warnings.is_empty()); // 应有脱敏警告
    }

    #[test]
    fn test_import_yaml_replace_mode() {
        let current = Config::default();
        let yaml = r#"
server:
  host: 127.0.0.1
  port: 9000
  api_key: new_key
"#;
        let options = ImportOptions::replace();
        let result = ImportService::import_yaml(yaml, &current, &options).expect("导入应成功");

        assert!(result.success);
        assert_eq!(result.config.server.host, "127.0.0.1");
        assert_eq!(result.config.server.port, 9000);
        assert_eq!(result.config.server.api_key, "new_key");
    }

    #[test]
    fn test_import_yaml_merge_mode() {
        let mut current = Config::default();
        current.credential_pool.openai.push(ApiKeyEntry {
            id: "existing".to_string(),
            api_key: "sk-existing".to_string(),
            base_url: None,
            disabled: false,
            proxy_url: None,
        });

        let yaml = r#"
server:
  host: 127.0.0.1
  port: 9000
  api_key: new_key
credential_pool:
  openai:
    - id: new
      api_key: sk-new
"#;
        let options = ImportOptions::merge();
        let result = ImportService::import_yaml(yaml, &current, &options).expect("导入应成功");

        assert!(result.success);
        // 服务器配置应被更新
        assert_eq!(result.config.server.host, "127.0.0.1");
        // 凭证池应合并
        assert_eq!(result.config.credential_pool.openai.len(), 2);
    }

    #[test]
    fn test_merge_credential_entries() {
        let current = vec![CredentialEntry {
            id: "id1".to_string(),
            token_file: "old.json".to_string(),
            disabled: false,
            proxy_url: None,
        }];
        let imported = vec![
            CredentialEntry {
                id: "id1".to_string(),
                token_file: "new.json".to_string(),
                disabled: true,
                proxy_url: None,
            },
            CredentialEntry {
                id: "id2".to_string(),
                token_file: "id2.json".to_string(),
                disabled: false,
                proxy_url: None,
            },
        ];

        let merged = ImportService::merge_credential_entries(&current, &imported);
        assert_eq!(merged.len(), 2);
        // id1 应被更新
        assert_eq!(merged[0].token_file, "new.json");
        assert!(merged[0].disabled);
        // id2 应被添加
        assert_eq!(merged[1].id, "id2");
    }

    #[test]
    fn test_merge_api_key_entries_skips_redacted() {
        let current = vec![ApiKeyEntry {
            id: "id1".to_string(),
            api_key: "sk-real".to_string(),
            base_url: None,
            disabled: false,
            proxy_url: None,
        }];
        let imported = vec![ApiKeyEntry {
            id: "id1".to_string(),
            api_key: REDACTED_PLACEHOLDER.to_string(),
            base_url: None,
            disabled: false,
            proxy_url: None,
        }];

        let merged = ImportService::merge_api_key_entries(&current, &imported);
        assert_eq!(merged.len(), 1);
        // 脱敏的条目不应覆盖现有的
        assert_eq!(merged[0].api_key, "sk-real");
    }

    #[test]
    fn test_clean_redacted_credentials() {
        let mut config = Config::default();
        config.server.api_key = REDACTED_PLACEHOLDER.to_string();
        config.providers.openai.api_key = Some(REDACTED_PLACEHOLDER.to_string());
        config.credential_pool.openai.push(ApiKeyEntry {
            id: "redacted".to_string(),
            api_key: REDACTED_PLACEHOLDER.to_string(),
            base_url: None,
            disabled: false,
            proxy_url: None,
        });
        config.credential_pool.openai.push(ApiKeyEntry {
            id: "real".to_string(),
            api_key: "sk-real".to_string(),
            base_url: None,
            disabled: false,
            proxy_url: None,
        });

        let server_key_cleared = ImportService::clean_redacted_credentials(&mut config);

        // 服务器 API 密钥应被清空并提示手动设置
        assert!(server_key_cleared);
        assert_eq!(config.server.api_key, "");
        // Provider API 密钥应被清除
        assert!(config.providers.openai.api_key.is_none());
        // 凭证池中脱敏的条目应被移除
        assert_eq!(config.credential_pool.openai.len(), 1);
        assert_eq!(config.credential_pool.openai[0].id, "real");
    }

    #[test]
    fn test_import_error_display() {
        let err = ImportError::FormatError("test".to_string());
        assert!(err.to_string().contains("格式错误"));

        let err = ImportError::VersionError("test".to_string());
        assert!(err.to_string().contains("版本不兼容"));

        let err = ImportError::RedactedDataError("test".to_string());
        assert!(err.to_string().contains("脱敏数据"));
    }
}
