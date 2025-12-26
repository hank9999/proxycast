import { invoke } from "@tauri-apps/api/core";

// 类型定义
export interface InterceptorState {
  enabled: boolean;
  active_hooks: string[];
  intercepted_count: number;
  last_activity?: string;
  can_restore: boolean;
}

export interface InterceptedUrl {
  id: string;
  url: string;
  source_process: string;
  timestamp: string;
  copied: boolean;
  opened_in_browser: boolean;
  dismissed: boolean;
}

export interface FingerprintBrowserConfig {
  enabled: boolean;
  executable_path: string;
  profile_path: string;
  additional_args: string[];
}

export interface RecoveryConfig {
  backup_system_state: boolean;
  emergency_recovery_hotkey: string;
  auto_recovery_on_crash: boolean;
  recovery_timeout: number;
}

export interface BrowserInterceptorConfig {
  enabled: boolean;
  target_processes: string[];
  url_patterns: string[];
  excluded_processes: string[];
  notification_enabled: boolean;
  auto_copy_to_clipboard: boolean;
  restore_on_exit: boolean;
  temporary_disable_timeout?: number | null;
  auto_launch_browser: boolean;
  fingerprint_browser: FingerprintBrowserConfig;
  recovery: RecoveryConfig;
}

export interface InterceptorStatistics {
  total_intercepted: number;
  current_intercepted: number;
  copied_count: number;
  opened_count: number;
  dismissed_count: number;
}

// API 函数
export const browserInterceptorApi = {
  // 获取拦截器状态
  async getState(): Promise<InterceptorState | null> {
    return await invoke("get_browser_interceptor_state");
  },

  // 启动拦截器
  async start(config: BrowserInterceptorConfig): Promise<string> {
    return await invoke("start_browser_interceptor", { config });
  },

  // 停止拦截器
  async stop(): Promise<string> {
    return await invoke("stop_browser_interceptor");
  },

  // 恢复正常浏览器行为
  async restoreNormalBehavior(): Promise<string> {
    return await invoke("restore_normal_browser_behavior");
  },

  // 临时禁用拦截器
  async temporaryDisable(durationSeconds: number): Promise<string> {
    return await invoke("temporary_disable_interceptor", { durationSeconds });
  },

  // 获取拦截的 URL 列表
  async getInterceptedUrls(): Promise<InterceptedUrl[]> {
    return await invoke("get_intercepted_urls");
  },

  // 获取历史记录
  async getHistory(limit?: number): Promise<InterceptedUrl[]> {
    return await invoke("get_interceptor_history", { limit });
  },

  // 复制 URL 到剪贴板
  async copyUrlToClipboard(urlId: string): Promise<string> {
    return await invoke("copy_intercepted_url_to_clipboard", { urlId });
  },

  // 在指纹浏览器中打开 URL
  async openInFingerprintBrowser(urlId: string): Promise<string> {
    return await invoke("open_url_in_fingerprint_browser", { urlId });
  },

  // 忽略 URL
  async dismissUrl(urlId: string): Promise<string> {
    return await invoke("dismiss_intercepted_url", { urlId });
  },

  // 更新配置
  async updateConfig(config: BrowserInterceptorConfig): Promise<string> {
    return await invoke("update_browser_interceptor_config", { config });
  },

  // 获取默认配置
  async getDefaultConfig(): Promise<BrowserInterceptorConfig> {
    return await invoke("get_default_browser_interceptor_config");
  },

  // 验证配置
  async validateConfig(config: BrowserInterceptorConfig): Promise<string> {
    return await invoke("validate_browser_interceptor_config", { config });
  },

  // 检查是否正在运行
  async isRunning(): Promise<boolean> {
    return await invoke("is_browser_interceptor_running");
  },

  // 获取统计信息
  async getStatistics(): Promise<InterceptorStatistics> {
    return await invoke("get_browser_interceptor_statistics");
  },

  // 通知相关函数
  async showNotification(
    title: string,
    body: string,
    icon?: string,
  ): Promise<string> {
    return await invoke("show_notification", { title, body, icon });
  },

  async showUrlInterceptNotification(
    url: string,
    sourceProcess: string,
  ): Promise<string> {
    return await invoke("show_url_intercept_notification", {
      url,
      sourceProcess,
    });
  },

  async showStatusNotification(
    message: string,
    notificationType: string,
  ): Promise<string> {
    return await invoke("show_status_notification", {
      message,
      notificationType,
    });
  },
};

// 导出便捷函数
export const getBrowserInterceptorState = () =>
  browserInterceptorApi.getState();
export const startBrowserInterceptor = (config: BrowserInterceptorConfig) =>
  browserInterceptorApi.start(config);
export const stopBrowserInterceptor = () => browserInterceptorApi.stop();
export const restoreNormalBrowserBehavior = () =>
  browserInterceptorApi.restoreNormalBehavior();
export const temporaryDisableInterceptor = (durationSeconds: number) =>
  browserInterceptorApi.temporaryDisable(durationSeconds);
export const getInterceptedUrls = () =>
  browserInterceptorApi.getInterceptedUrls();
export const getInterceptorHistory = (limit?: number) =>
  browserInterceptorApi.getHistory(limit);
export const copyInterceptedUrlToClipboard = (urlId: string) =>
  browserInterceptorApi.copyUrlToClipboard(urlId);
export const openUrlInFingerprintBrowser = (urlId: string) =>
  browserInterceptorApi.openInFingerprintBrowser(urlId);
export const dismissInterceptedUrl = (urlId: string) =>
  browserInterceptorApi.dismissUrl(urlId);
export const updateBrowserInterceptorConfig = (
  config: BrowserInterceptorConfig,
) => browserInterceptorApi.updateConfig(config);
export const getDefaultBrowserInterceptorConfig = () =>
  browserInterceptorApi.getDefaultConfig();
export const validateBrowserInterceptorConfig = (
  config: BrowserInterceptorConfig,
) => browserInterceptorApi.validateConfig(config);
export const isBrowserInterceptorRunning = () =>
  browserInterceptorApi.isRunning();
export const getBrowserInterceptorStatistics = () =>
  browserInterceptorApi.getStatistics();

// 通知函数导出
export const showBrowserInterceptorNotification = (
  title: string,
  body: string,
  icon?: string,
) => browserInterceptorApi.showNotification(title, body, icon);
export const showUrlInterceptNotification = (
  url: string,
  sourceProcess: string,
) => browserInterceptorApi.showUrlInterceptNotification(url, sourceProcess);
export const showBrowserInterceptorStatusNotification = (
  message: string,
  type: string,
) => browserInterceptorApi.showStatusNotification(message, type);
