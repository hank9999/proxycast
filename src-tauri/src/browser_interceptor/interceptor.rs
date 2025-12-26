use crate::browser_interceptor::{
    BrowserInterceptorConfig, BrowserInterceptorError, InterceptedUrl, NotificationService, Result,
    StateManager, UrlManager,
};
use std::sync::Arc;
use tokio::sync::RwLock;

/// 浏览器拦截器主结构
pub struct BrowserInterceptor {
    config: Arc<RwLock<BrowserInterceptorConfig>>,
    state_manager: Arc<StateManager>,
    url_manager: Arc<UrlManager>,
    notification_service: Arc<RwLock<NotificationService>>,
    #[cfg(target_os = "windows")]
    windows_interceptor: Option<crate::browser_interceptor::platform::windows::WindowsInterceptor>,
    #[cfg(target_os = "macos")]
    macos_interceptor: Option<crate::browser_interceptor::platform::macos::MacOSInterceptor>,
    #[cfg(target_os = "linux")]
    linux_interceptor: Option<crate::browser_interceptor::platform::linux::LinuxInterceptor>,
}

impl BrowserInterceptor {
    /// 创建新的浏览器拦截器实例
    pub fn new(config: BrowserInterceptorConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            state_manager: Arc::new(StateManager::new()),
            url_manager: Arc::new(UrlManager::new()),
            notification_service: Arc::new(RwLock::new(NotificationService::new())),
            #[cfg(target_os = "windows")]
            windows_interceptor: None,
            #[cfg(target_os = "macos")]
            macos_interceptor: None,
            #[cfg(target_os = "linux")]
            linux_interceptor: None,
        }
    }

    /// 启动拦截器
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("开始启动浏览器拦截器...");

        let config = self.config.read().await;

        if !config.enabled {
            tracing::error!("拦截器配置中 enabled 为 false");
            return Err(BrowserInterceptorError::InterceptorError(
                "拦截器未启用".to_string(),
            ));
        }

        // 验证配置
        tracing::info!("验证配置...");
        config.validate().map_err(|e| {
            tracing::error!("配置验证失败: {}", e);
            BrowserInterceptorError::ConfigError(format!("配置验证失败: {}", e))
        })?;

        drop(config); // 释放读锁

        // 启用状态管理器
        tracing::info!("启用状态管理器...");
        self.state_manager.enable_interceptor().await.map_err(|e| {
            tracing::error!("启用状态管理器失败: {}", e);
            e
        })?;

        // 启动平台特定的拦截器
        tracing::info!("启动平台拦截器...");
        self.start_platform_interceptor().await.map_err(|e| {
            tracing::error!("启动平台拦截器失败: {}", e);
            e
        })?;

        // 发送启用通知
        tracing::info!("发送启用通知...");
        let notification_service = self.notification_service.read().await;
        notification_service.notify_interceptor_enabled().await?;

        tracing::info!("浏览器拦截器已成功启动");
        Ok(())
    }

    /// 停止拦截器
    pub async fn stop(&mut self) -> Result<()> {
        // 停止平台特定的拦截器
        self.stop_platform_interceptor().await?;

        // 禁用状态管理器
        self.state_manager.disable_interceptor().await?;

        // 发送禁用通知
        let notification_service = self.notification_service.read().await;
        notification_service.notify_interceptor_disabled().await?;

        tracing::info!("浏览器拦截器已停止");
        Ok(())
    }

    /// 恢复正常浏览器行为
    pub async fn restore_normal_behavior(&mut self) -> Result<()> {
        // 停止拦截
        self.stop_platform_interceptor().await?;

        // 恢复系统状态
        self.state_manager.restore_normal_behavior().await?;

        // 发送恢复通知
        let notification_service = self.notification_service.read().await;
        notification_service.notify_system_restored().await?;

        tracing::info!("已恢复正常浏览器行为");
        Ok(())
    }

    /// 临时禁用拦截器
    pub async fn temporary_disable(&mut self, duration_seconds: u64) -> Result<()> {
        self.state_manager
            .temporary_disable(duration_seconds)
            .await?;

        // 临时停止平台拦截器
        self.stop_platform_interceptor().await?;

        tracing::info!("拦截器已临时禁用 {} 秒", duration_seconds);
        Ok(())
    }

    /// 处理拦截到的浏览器启动请求
    pub async fn handle_browser_launch(&self, url: String, source_process: String) -> Result<bool> {
        let config = self.config.read().await;

        // 检查是否应该拦截这个进程
        if !self.should_intercept(&config, &source_process, &url) {
            return Ok(false); // 不拦截，让浏览器正常启动
        }

        drop(config); // 释放读锁

        // 添加到拦截列表
        let url_id = self
            .url_manager
            .add_intercepted_url(url.clone(), source_process.clone())?;

        // 更新拦截计数
        self.state_manager.increment_intercept_count()?;

        // 创建拦截的 URL 对象用于通知
        let intercepted_url = InterceptedUrl::new(url, source_process);

        // 发送通知
        let notification_service = self.notification_service.read().await;
        notification_service
            .notify_url_intercepted(&intercepted_url)
            .await?;

        // 如果启用了自动复制到剪贴板
        let config = self.config.read().await;
        if config.auto_copy_to_clipboard {
            self.copy_to_clipboard(&intercepted_url.url).await?;
            self.url_manager.mark_as_copied(&url_id)?;
        }

        tracing::info!(
            "已拦截浏览器启动: {} (来源: {})",
            intercepted_url.url,
            intercepted_url.source_process
        );
        Ok(true) // 已拦截
    }

    /// 获取当前状态
    pub async fn get_state(&self) -> Result<crate::browser_interceptor::InterceptorState> {
        self.state_manager.get_state()
    }

    /// 获取拦截的 URL 列表
    pub async fn get_intercepted_urls(&self) -> Result<Vec<InterceptedUrl>> {
        self.url_manager.get_intercepted_urls()
    }

    /// 获取历史记录
    pub async fn get_history(&self, limit: Option<usize>) -> Result<Vec<InterceptedUrl>> {
        self.url_manager.get_history(limit)
    }

    /// 复制 URL 到剪贴板
    pub async fn copy_url_to_clipboard(&self, url_id: &str) -> Result<()> {
        if let Some(intercepted_url) = self.url_manager.get_intercepted_url(url_id)? {
            self.copy_to_clipboard(&intercepted_url.url).await?;
            self.url_manager.mark_as_copied(url_id)?;
            tracing::info!("URL {} 已复制到剪贴板", url_id);
        }
        Ok(())
    }

    /// 在指纹浏览器中打开 URL
    pub async fn open_in_fingerprint_browser(&self, url_id: &str) -> Result<()> {
        let config = self.config.read().await;

        if !config.fingerprint_browser.enabled {
            return Err(BrowserInterceptorError::InterceptorError(
                "指纹浏览器未启用".to_string(),
            ));
        }

        if let Some(intercepted_url) = self.url_manager.get_intercepted_url(url_id)? {
            self.launch_fingerprint_browser(&config.fingerprint_browser, &intercepted_url.url)
                .await?;
            self.url_manager.mark_as_opened(url_id)?;
            tracing::info!("URL {} 已在指纹浏览器中打开", url_id);
        }

        Ok(())
    }

    /// 忽略指定的 URL
    pub async fn dismiss_url(&self, url_id: &str) -> Result<()> {
        self.url_manager.dismiss_url(url_id)?;
        tracing::info!("URL {} 已被忽略", url_id);
        Ok(())
    }

    /// 更新配置
    pub async fn update_config(&self, new_config: BrowserInterceptorConfig) -> Result<()> {
        // 验证新配置
        new_config
            .validate()
            .map_err(|e| BrowserInterceptorError::ConfigError(format!("配置验证失败: {}", e)))?;

        let mut config = self.config.write().await;
        *config = new_config;

        tracing::info!("浏览器拦截器配置已更新");
        Ok(())
    }

    /// 检查是否应该拦截指定的进程和 URL
    fn should_intercept(
        &self,
        config: &BrowserInterceptorConfig,
        process_name: &str,
        url: &str,
    ) -> bool {
        // 检查是否在排除列表中
        if config.is_excluded_process(process_name) {
            return false;
        }

        // 检查是否是目标进程
        if !config.is_target_process(process_name) {
            return false;
        }

        // 检查 URL 是否匹配模式
        config.matches_url_pattern(url)
    }

    /// 启动平台特定的拦截器
    async fn start_platform_interceptor(&mut self) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            // 创建 URL 处理器闭包
            let url_manager = Arc::clone(&self.url_manager);
            let state_manager = Arc::clone(&self.state_manager);

            let url_handler = move |intercepted_url: InterceptedUrl| {
                let url_manager = Arc::clone(&url_manager);
                let state_manager = Arc::clone(&state_manager);

                tokio::spawn(async move {
                    if let Err(e) = url_manager
                        .add_intercepted_url(intercepted_url.url, intercepted_url.source_process)
                    {
                        tracing::error!("添加拦截 URL 失败: {}", e);
                    }
                    if let Err(e) = state_manager.increment_intercept_count() {
                        tracing::error!("更新拦截计数失败: {}", e);
                    }
                });
            };

            let mut interceptor =
                crate::browser_interceptor::platform::windows::WindowsInterceptor::new(url_handler);
            interceptor.start().await?;
            self.windows_interceptor = Some(interceptor);
        }

        #[cfg(target_os = "macos")]
        {
            let url_manager = Arc::clone(&self.url_manager);
            let state_manager = Arc::clone(&self.state_manager);

            let url_handler = move |intercepted_url: InterceptedUrl| {
                let url_manager = Arc::clone(&url_manager);
                let state_manager = Arc::clone(&state_manager);

                tokio::spawn(async move {
                    if let Err(e) = url_manager
                        .add_intercepted_url(intercepted_url.url, intercepted_url.source_process)
                    {
                        tracing::error!("添加拦截 URL 失败: {}", e);
                    }
                    if let Err(e) = state_manager.increment_intercept_count() {
                        tracing::error!("更新拦截计数失败: {}", e);
                    }
                });
            };

            let mut interceptor =
                crate::browser_interceptor::platform::macos::MacOSInterceptor::new(url_handler);
            interceptor.start().await?;
            self.macos_interceptor = Some(interceptor);
        }

        #[cfg(target_os = "linux")]
        {
            let url_manager = Arc::clone(&self.url_manager);
            let state_manager = Arc::clone(&self.state_manager);

            let url_handler = move |intercepted_url: InterceptedUrl| {
                let url_manager = Arc::clone(&url_manager);
                let state_manager = Arc::clone(&state_manager);

                tokio::spawn(async move {
                    if let Err(e) = url_manager
                        .add_intercepted_url(intercepted_url.url, intercepted_url.source_process)
                    {
                        tracing::error!("添加拦截 URL 失败: {}", e);
                    }
                    if let Err(e) = state_manager.increment_intercept_count() {
                        tracing::error!("更新拦截计数失败: {}", e);
                    }
                });
            };

            let mut interceptor =
                crate::browser_interceptor::platform::linux::LinuxInterceptor::new(url_handler);
            interceptor.start().await?;
            self.linux_interceptor = Some(interceptor);
        }

        Ok(())
    }

    /// 停止平台特定的拦截器
    async fn stop_platform_interceptor(&mut self) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            if let Some(ref mut interceptor) = self.windows_interceptor {
                interceptor.stop().await?;
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Some(ref mut interceptor) = self.macos_interceptor {
                interceptor.stop().await?;
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Some(ref mut interceptor) = self.linux_interceptor {
                interceptor.stop().await?;
            }
        }

        Ok(())
    }

    /// 复制文本到剪贴板
    async fn copy_to_clipboard(&self, text: &str) -> Result<()> {
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => {
                if let Err(e) = clipboard.set_text(text) {
                    return Err(BrowserInterceptorError::InterceptorError(format!(
                        "复制到剪贴板失败: {}",
                        e
                    )));
                }
                tracing::info!("已复制到剪贴板: {}", text);
                Ok(())
            }
            Err(e) => Err(BrowserInterceptorError::InterceptorError(format!(
                "创建剪贴板实例失败: {}",
                e
            ))),
        }
    }

    /// 启动指纹浏览器
    async fn launch_fingerprint_browser(
        &self,
        browser_config: &crate::browser_interceptor::config::FingerprintBrowserConfig,
        url: &str,
    ) -> Result<()> {
        if browser_config.executable_path.is_empty() {
            return Err(BrowserInterceptorError::ConfigError(
                "指纹浏览器可执行文件路径未配置".to_string(),
            ));
        }

        // 构建启动命令
        let mut command = std::process::Command::new(&browser_config.executable_path);

        // 添加 URL 参数
        command.arg(url);

        // 添加额外的参数
        for arg in &browser_config.additional_args {
            command.arg(arg);
        }

        // 如果配置了配置文件路径
        if !browser_config.profile_path.is_empty() {
            command
                .arg("--user-data-dir")
                .arg(&browser_config.profile_path);
        }

        // 异步启动进程
        match command.spawn() {
            Ok(mut child) => {
                // 在后台等待进程完成
                tokio::spawn(async move {
                    match child.wait() {
                        Ok(status) => {
                            if status.success() {
                                tracing::info!("指纹浏览器启动成功");
                            } else {
                                tracing::error!("指纹浏览器退出异常: {}", status);
                            }
                        }
                        Err(e) => {
                            tracing::error!("等待指纹浏览器进程失败: {}", e);
                        }
                    }
                });

                tracing::info!(
                    "已启动指纹浏览器: {} -> {}",
                    browser_config.executable_path,
                    url
                );
                Ok(())
            }
            Err(e) => Err(BrowserInterceptorError::InterceptorError(format!(
                "启动指纹浏览器失败: {}",
                e
            ))),
        }
    }
}

impl Default for BrowserInterceptor {
    fn default() -> Self {
        Self::new(BrowserInterceptorConfig::default())
    }
}
