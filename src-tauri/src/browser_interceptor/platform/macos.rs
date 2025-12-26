use crate::browser_interceptor::{BrowserInterceptorError, InterceptedUrl, Result};
use once_cell::sync::Lazy;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// 全局 URL sender，用于从 deep-link 事件接收 URL（使用 Mutex 保证线程安全）
static GLOBAL_URL_SENDER: Lazy<Mutex<Option<mpsc::UnboundedSender<InterceptedUrl>>>> =
    Lazy::new(|| Mutex::new(None));

/// macOS 平台的浏览器拦截器（基于设置默认浏览器 + Deep Link）
pub struct MacOSInterceptor {
    running: bool,
    url_sender: Option<mpsc::UnboundedSender<InterceptedUrl>>,
    original_default_browser: Option<String>,
    url_handler: Option<Arc<dyn Fn(InterceptedUrl) + Send + Sync + 'static>>,
}

impl MacOSInterceptor {
    pub fn new<F>(url_handler: F) -> Self
    where
        F: Fn(InterceptedUrl) + Send + Sync + 'static,
    {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handler = Arc::new(url_handler);
        let handler_clone = handler.clone();

        // 启动后台任务处理拦截的 URL
        tokio::spawn(async move {
            while let Some(intercepted_url) = rx.recv().await {
                handler_clone(intercepted_url);
            }
        });

        Self {
            running: false,
            url_sender: Some(tx),
            original_default_browser: None,
            url_handler: Some(handler),
        }
    }

    /// 启动拦截
    pub async fn start(&mut self) -> Result<()> {
        if self.running {
            tracing::info!("macOS 拦截器已在运行中，跳过启动");
            return Ok(());
        }

        tracing::info!("正在启动 macOS 浏览器拦截器...");

        // 1. 保存当前默认浏览器
        self.original_default_browser = self.get_default_browser().await;
        tracing::info!("当前默认浏览器: {:?}", self.original_default_browser);

        // 2. 设置全局 URL sender
        if let Some(ref sender) = self.url_sender {
            if let Ok(mut global_sender) = GLOBAL_URL_SENDER.lock() {
                *global_sender = Some(sender.clone());
            }
        }

        // 3. 将 ProxyCast 设置为默认浏览器
        self.set_as_default_browser().await?;

        self.running = true;
        tracing::info!("macOS 浏览器拦截器已启动");
        tracing::info!("提示：现在所有 http/https URL 打开请求都会被 ProxyCast 拦截");

        Ok(())
    }

    /// 获取当前默认浏览器的 Bundle ID
    async fn get_default_browser(&self) -> Option<String> {
        // 使用 Swift 调用 LSCopyDefaultHandlerForURLScheme 获取真实的默认浏览器
        let swift_code = r#"
import Foundation
import CoreServices

if let handler = LSCopyDefaultHandlerForURLScheme("https" as CFString) {
    print(handler.takeRetainedValue() as String)
} else if let handler = LSCopyDefaultHandlerForURLScheme("http" as CFString) {
    print(handler.takeRetainedValue() as String)
} else {
    print("")
}
"#;

        let output = Command::new("swift")
            .args(["-e", swift_code])
            .output()
            .ok()?;

        if output.status.success() {
            let bundle_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !bundle_id.is_empty() && !bundle_id.contains("proxycast") {
                tracing::info!("检测到当前默认浏览器: {}", bundle_id);
                return Some(bundle_id);
            }
            // 如果当前是 ProxyCast，说明之前已设置过，需要检测用户常用的浏览器
            if bundle_id.contains("proxycast") {
                tracing::info!("当前默认浏览器是 ProxyCast，尝试检测用户常用浏览器");
                return self.detect_installed_browser().await;
            }
        }

        // 备选方法：使用 duti 查询
        let duti_output = Command::new("duti").args(["-x", "https"]).output().ok();

        if let Some(output) = duti_output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // duti -x 输出格式：第三行是 bundle id
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    if (trimmed.starts_with("com.") || trimmed.starts_with("org."))
                        && !trimmed.contains("proxycast")
                    {
                        tracing::info!("通过 duti 检测到默认浏览器: {}", trimmed);
                        return Some(trimmed.to_string());
                    }
                }
            }
        }

        // 最后尝试检测已安装的浏览器，优先返回 Chrome
        self.detect_installed_browser().await
    }

    /// 将 ProxyCast 设置为默认浏览器
    async fn set_as_default_browser(&self) -> Result<()> {
        tracing::info!("正在将 ProxyCast 设置为默认浏览器...");

        // 使用 Swift 代码通过 Launch Services API 设置默认处理程序
        // 由于 Rust 直接调用 Launch Services 较复杂，这里使用 osascript 辅助

        // 方法 1: 使用 duti 工具（如果已安装）
        let duti_result = Command::new("duti")
            .args(["-s", "com.proxycast.app", "http", "all"])
            .output();

        if let Ok(output) = duti_result {
            if output.status.success() {
                let _ = Command::new("duti")
                    .args(["-s", "com.proxycast.app", "https", "all"])
                    .output();
                tracing::info!("已通过 duti 设置默认浏览器");
                return Ok(());
            }
        }

        // 方法 2: 使用 Swift 脚本
        let swift_code = r#"
import Foundation
import CoreServices

let bundleId = "com.proxycast.app" as CFString

// 设置 HTTP handler
LSSetDefaultHandlerForURLScheme("http" as CFString, bundleId)

// 设置 HTTPS handler
LSSetDefaultHandlerForURLScheme("https" as CFString, bundleId)

print("OK")
"#;

        let output = Command::new("swift")
            .args(["-e", swift_code])
            .output()
            .map_err(|e| {
                BrowserInterceptorError::PlatformError(format!("执行 Swift 脚本失败: {}", e))
            })?;

        if output.status.success() {
            tracing::info!("已通过 Launch Services API 设置默认浏览器");
            return Ok(());
        }

        // 方法 3: 提示用户手动设置
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::warn!("自动设置默认浏览器失败: {}", stderr);
        tracing::info!("请手动在系统设置中将 ProxyCast 设置为默认浏览器");

        // 打开系统设置
        let _ = Command::new("open")
            .args(["x-apple.systempreferences:com.apple.preference.general"])
            .output();

        Ok(())
    }

    /// 恢复原来的默认浏览器
    async fn restore_default_browser(&self) -> Result<()> {
        // 获取要恢复的浏览器
        let browser_id = match &self.original_default_browser {
            Some(id) if !id.contains("proxycast") => id.clone(),
            _ => {
                // 如果没有记录原始浏览器，尝试检测已安装的浏览器
                self.detect_installed_browser()
                    .await
                    .unwrap_or_else(|| "com.google.Chrome".to_string())
            }
        };

        tracing::info!("正在恢复默认浏览器为: {}", browser_id);

        // 使用 duti
        let duti_result = Command::new("duti")
            .args(["-s", &browser_id, "http", "all"])
            .output();

        if let Ok(output) = duti_result {
            if output.status.success() {
                let _ = Command::new("duti")
                    .args(["-s", &browser_id, "https", "all"])
                    .output();
                tracing::info!("已通过 duti 恢复默认浏览器为: {}", browser_id);
                return Ok(());
            }
        }

        // 使用 Swift
        let swift_code = format!(
            r#"
import Foundation
import CoreServices

let bundleId = "{}" as CFString
LSSetDefaultHandlerForURLScheme("http" as CFString, bundleId)
LSSetDefaultHandlerForURLScheme("https" as CFString, bundleId)
print("OK")
"#,
            browser_id
        );

        let output = Command::new("swift").args(["-e", &swift_code]).output();

        if let Ok(out) = output {
            if out.status.success() {
                tracing::info!("已通过 Swift 恢复默认浏览器为: {}", browser_id);
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tracing::warn!("Swift 恢复默认浏览器失败: {}", stderr);
            }
        }

        Ok(())
    }

    /// 检测已安装的浏览器（用于恢复时选择）
    async fn detect_installed_browser(&self) -> Option<String> {
        // 按优先级排序的浏览器列表
        let browsers = [
            ("com.google.Chrome", "/Applications/Google Chrome.app"),
            ("org.mozilla.firefox", "/Applications/Firefox.app"),
            ("com.microsoft.edgemac", "/Applications/Microsoft Edge.app"),
            ("com.brave.Browser", "/Applications/Brave Browser.app"),
            ("com.apple.Safari", "/Applications/Safari.app"),
        ];

        for (bundle_id, app_path) in &browsers {
            // 直接检查应用是否存在
            if std::path::Path::new(app_path).exists() {
                tracing::info!("检测到已安装的浏览器: {} ({})", bundle_id, app_path);
                return Some(bundle_id.to_string());
            }
        }

        // Safari 总是存在
        Some("com.apple.Safari".to_string())
    }

    /// 停止拦截
    pub async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        tracing::info!("正在停止 macOS 浏览器拦截器...");

        // 清除全局 URL sender
        if let Ok(mut global_sender) = GLOBAL_URL_SENDER.lock() {
            *global_sender = None;
        }

        // 恢复默认浏览器
        self.restore_default_browser().await?;

        self.running = false;
        tracing::info!("macOS 浏览器拦截器已停止");
        Ok(())
    }

    /// 检查是否正在拦截
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// 获取当前监听的端口（不再使用，保留接口兼容性）
    pub fn get_port(&self) -> Option<u16> {
        None
    }

    /// 恢复系统默认设置
    pub async fn restore_system_defaults(&self) -> Result<()> {
        self.restore_default_browser().await
    }

    /// 临时禁用拦截
    pub async fn temporarily_disable(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        tracing::info!("正在临时禁用 macOS 浏览器拦截器...");

        // 临时恢复默认浏览器
        self.restore_default_browser().await?;

        tracing::info!("macOS 浏览器拦截器已临时禁用");
        Ok(())
    }

    /// 重新启用拦截
    pub async fn re_enable(&mut self) -> Result<()> {
        if !self.running {
            return Err(BrowserInterceptorError::InterceptorError(
                "拦截器未运行，无法重新启用".to_string(),
            ));
        }

        tracing::info!("正在重新启用 macOS 浏览器拦截器...");

        // 重新设置为默认浏览器
        self.set_as_default_browser().await?;

        tracing::info!("macOS 浏览器拦截器已重新启用");
        Ok(())
    }
}

impl Drop for MacOSInterceptor {
    fn drop(&mut self) {
        if self.running {
            tracing::info!("macOS 浏览器拦截器资源已清理");
            // 注意：在 drop 中无法使用 async，恢复操作在 stop() 中完成
        }
    }
}

/// 处理从 deep-link 接收到的 URL（由 lib.rs 中的事件监听器调用）
pub fn handle_deep_link_url(url: String) {
    tracing::info!("收到 deep-link URL: {}", url);

    // 检查是否是 http/https URL
    if !url.starts_with("http://") && !url.starts_with("https://") {
        tracing::debug!("忽略非 HTTP URL: {}", url);
        return;
    }

    // 尝试识别来源进程（macOS 上较难获取，使用默认值）
    let source_process = detect_source_process(&url);

    let intercepted_url = InterceptedUrl::new(url, source_process);

    // 发送到处理器
    if let Ok(global_sender) = GLOBAL_URL_SENDER.lock() {
        if let Some(ref sender) = *global_sender {
            match sender.send(intercepted_url) {
                Ok(_) => tracing::debug!("URL 已发送到处理器"),
                Err(e) => tracing::error!("发送 URL 到处理器失败: {:?}", e),
            }
        } else {
            tracing::warn!("拦截器未运行，忽略 URL");
        }
    } else {
        tracing::error!("无法获取 URL sender 锁");
    }
}

/// 尝试检测 URL 的来源进程
fn detect_source_process(url: &str) -> String {
    // 基于 URL 特征推测来源
    if url.contains("kiro") || url.contains("amazon") || url.contains("aws") {
        return "Kiro".to_string();
    }
    if url.contains("cursor") || url.contains("anysphere") {
        return "Cursor".to_string();
    }
    if url.contains("vscode") || url.contains("microsoft") || url.contains("visualstudio") {
        return "VSCode".to_string();
    }
    if url.contains("claude") || url.contains("anthropic") {
        return "Claude App".to_string();
    }
    if url.contains("github") {
        return "GitHub App".to_string();
    }
    if url.contains("google") || url.contains("accounts.google") {
        return "OAuth Request".to_string();
    }

    // 尝试获取前台应用
    if let Ok(output) = Command::new("osascript")
        .args([
            "-e",
            "tell application \"System Events\" to get the name of first process whose frontmost is true",
        ])
        .output()
    {
        if output.status.success() {
            let app_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !app_name.is_empty() && app_name != "ProxyCast" {
                return app_name;
            }
        }
    }

    "Unknown App".to_string()
}

/// 检查拦截器是否有活跃的 URL sender
pub fn is_interceptor_active() -> bool {
    GLOBAL_URL_SENDER
        .lock()
        .map(|sender| sender.is_some())
        .unwrap_or(false)
}
