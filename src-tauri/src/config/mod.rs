//! 配置管理模块
//!
//! 提供 YAML 配置文件支持、热重载和配置导入导出功能
//! 同时保持与旧版 JSON 配置的向后兼容性

mod export;
mod hot_reload;
mod import;
mod path_utils;
mod types;
mod yaml;

pub use export::{
    base64_decode, base64_encode, ExportBundle, ExportError, ExportOptions, ExportService,
    REDACTED_PLACEHOLDER,
};
pub use hot_reload::{
    ConfigChangeEvent, ConfigChangeKind, FileWatcher, HotReloadError, HotReloadManager,
    HotReloadStatus, ReloadResult,
};
pub use import::{ImportError, ImportOptions, ImportResult, ImportService, ValidationResult};
pub use path_utils::{collapse_tilde, contains_tilde, expand_tilde};
pub use types::{
    is_default_api_key, AmpConfig, AmpModelMapping, ApiKeyEntry, Config, CredentialEntry,
    CredentialPoolConfig, CustomProviderConfig, GeminiApiKeyEntry, IFlowCredentialEntry,
    InjectionRuleConfig, InjectionSettings, LoggingConfig, ProviderConfig, ProvidersConfig,
    QuotaExceededConfig, RemoteManagementConfig, RetrySettings, RoutingConfig, RoutingRuleConfig,
    ServerConfig, TlsConfig, VertexApiKeyEntry, VertexModelAlias, DEFAULT_API_KEY,
};
pub use yaml::{
    load_config, save_config, save_config_yaml, ConfigError, ConfigManager, YamlService,
};

#[cfg(test)]
mod tests;
