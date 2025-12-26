use crate::browser_interceptor::{BrowserInterceptorError, InterceptedUrl, Result};

/// Linux 平台的浏览器拦截器
pub struct LinuxInterceptor {
    running: bool,
}

impl LinuxInterceptor {
    pub fn new<F>(_url_handler: F) -> Self
    where
        F: Fn(InterceptedUrl) + Send + Sync + 'static,
    {
        Self { running: false }
    }

    /// 启动拦截
    pub async fn start(&mut self) -> Result<()> {
        // TODO: 实现 Linux 平台的浏览器拦截
        // 可以使用 xdg-open 拦截或 D-Bus 监听
        self.running = true;
        tracing::info!("Linux 浏览器拦截器已启动 (占位符)");
        Ok(())
    }

    /// 停止拦截
    pub async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        self.running = false;
        tracing::info!("Linux 浏览器拦截器已停止");
        Ok(())
    }

    /// 检查是否正在拦截
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// 恢复系统默认设置
    pub async fn restore_system_defaults(&self) -> Result<()> {
        tracing::info!("Linux 系统默认设置已恢复");
        Ok(())
    }

    /// 临时禁用拦截
    pub async fn temporarily_disable(&mut self) -> Result<()> {
        tracing::info!("Linux 拦截器已临时禁用");
        Ok(())
    }

    /// 重新启用拦截
    pub async fn re_enable(&mut self) -> Result<()> {
        tracing::info!("Linux 拦截器已重新启用");
        Ok(())
    }
}

impl Drop for LinuxInterceptor {
    fn drop(&mut self) {
        if self.running {
            tokio::runtime::Handle::try_current().map(|handle| {
                handle.block_on(async {
                    let _ = self.stop().await;
                    let _ = self.restore_system_defaults().await;
                })
            });
        }
    }
}
