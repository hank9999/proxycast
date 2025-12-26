use crate::browser_interceptor::{
    BrowserInterceptor, BrowserInterceptorConfig, InterceptedUrl, InterceptorState,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{AppHandle, State};
use tokio::sync::RwLock;

/// 浏览器拦截器状态封装
pub struct BrowserInterceptorState(pub Arc<RwLock<Option<BrowserInterceptor>>>);

impl Default for BrowserInterceptorState {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(None)))
    }
}

/// 获取拦截器状态
#[tauri::command]
pub async fn get_browser_interceptor_state(
    state: State<'_, BrowserInterceptorState>,
) -> Result<Option<InterceptorState>, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        match interceptor.get_state().await {
            Ok(state) => Ok(Some(state)),
            Err(e) => Err(format!("获取拦截器状态失败: {}", e)),
        }
    } else {
        Ok(None)
    }
}

/// 启动浏览器拦截器
#[tauri::command]
pub async fn start_browser_interceptor(
    state: State<'_, BrowserInterceptorState>,
    config: BrowserInterceptorConfig,
) -> Result<String, String> {
    let mut interceptor_guard = state.0.write().await;

    // 如果已经有拦截器在运行，先停止它
    if let Some(existing_interceptor) = interceptor_guard.as_mut() {
        if let Err(e) = existing_interceptor.stop().await {
            tracing::warn!("停止现有拦截器时出错: {}", e);
        }
    }

    // 创建新的拦截器
    let mut interceptor = BrowserInterceptor::new(config);

    match interceptor.start().await {
        Ok(_) => {
            *interceptor_guard = Some(interceptor);
            Ok("浏览器拦截器已启动".to_string())
        }
        Err(e) => Err(format!("启动浏览器拦截器失败: {}", e)),
    }
}

/// 停止浏览器拦截器
#[tauri::command]
pub async fn stop_browser_interceptor(
    state: State<'_, BrowserInterceptorState>,
) -> Result<String, String> {
    let mut interceptor_guard = state.0.write().await;

    if let Some(interceptor) = interceptor_guard.as_mut() {
        match interceptor.stop().await {
            Ok(_) => {
                *interceptor_guard = None;
                Ok("浏览器拦截器已停止".to_string())
            }
            Err(e) => Err(format!("停止浏览器拦截器失败: {}", e)),
        }
    } else {
        Err("拦截器未运行".to_string())
    }
}

/// 恢复正常浏览器行为
#[tauri::command]
pub async fn restore_normal_browser_behavior(
    state: State<'_, BrowserInterceptorState>,
) -> Result<String, String> {
    let mut interceptor_guard = state.0.write().await;

    if let Some(interceptor) = interceptor_guard.as_mut() {
        match interceptor.restore_normal_behavior().await {
            Ok(_) => Ok("已恢复正常浏览器行为".to_string()),
            Err(e) => Err(format!("恢复正常浏览器行为失败: {}", e)),
        }
    } else {
        Err("拦截器未运行".to_string())
    }
}

/// 临时禁用拦截器
#[tauri::command]
pub async fn temporary_disable_interceptor(
    state: State<'_, BrowserInterceptorState>,
    duration_seconds: u64,
) -> Result<String, String> {
    let mut interceptor_guard = state.0.write().await;

    if let Some(interceptor) = interceptor_guard.as_mut() {
        match interceptor.temporary_disable(duration_seconds).await {
            Ok(_) => Ok(format!("拦截器已临时禁用 {} 秒", duration_seconds)),
            Err(e) => Err(format!("临时禁用拦截器失败: {}", e)),
        }
    } else {
        Err("拦截器未运行".to_string())
    }
}

/// 获取拦截的 URL 列表
#[tauri::command]
pub async fn get_intercepted_urls(
    state: State<'_, BrowserInterceptorState>,
) -> Result<Vec<InterceptedUrl>, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        match interceptor.get_intercepted_urls().await {
            Ok(urls) => Ok(urls),
            Err(e) => Err(format!("获取拦截 URL 失败: {}", e)),
        }
    } else {
        Ok(Vec::new())
    }
}

/// 获取历史记录
#[tauri::command]
pub async fn get_interceptor_history(
    state: State<'_, BrowserInterceptorState>,
    limit: Option<usize>,
) -> Result<Vec<InterceptedUrl>, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        match interceptor.get_history(limit).await {
            Ok(history) => Ok(history),
            Err(e) => Err(format!("获取历史记录失败: {}", e)),
        }
    } else {
        Ok(Vec::new())
    }
}

/// 复制 URL 到剪贴板
#[tauri::command]
pub async fn copy_intercepted_url_to_clipboard(
    state: State<'_, BrowserInterceptorState>,
    url_id: String,
) -> Result<String, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        match interceptor.copy_url_to_clipboard(&url_id).await {
            Ok(_) => Ok("URL 已复制到剪贴板".to_string()),
            Err(e) => Err(format!("复制 URL 失败: {}", e)),
        }
    } else {
        Err("拦截器未运行".to_string())
    }
}

/// 在指纹浏览器中打开 URL
#[tauri::command]
pub async fn open_url_in_fingerprint_browser(
    state: State<'_, BrowserInterceptorState>,
    url_id: String,
) -> Result<String, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        match interceptor.open_in_fingerprint_browser(&url_id).await {
            Ok(_) => Ok("URL 已在指纹浏览器中打开".to_string()),
            Err(e) => Err(format!("在指纹浏览器中打开 URL 失败: {}", e)),
        }
    } else {
        Err("拦截器未运行".to_string())
    }
}

/// 忽略指定的 URL
#[tauri::command]
pub async fn dismiss_intercepted_url(
    state: State<'_, BrowserInterceptorState>,
    url_id: String,
) -> Result<String, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        match interceptor.dismiss_url(&url_id).await {
            Ok(_) => Ok("URL 已忽略".to_string()),
            Err(e) => Err(format!("忽略 URL 失败: {}", e)),
        }
    } else {
        Err("拦截器未运行".to_string())
    }
}

/// 更新拦截器配置
#[tauri::command]
pub async fn update_browser_interceptor_config(
    state: State<'_, BrowserInterceptorState>,
    config: BrowserInterceptorConfig,
) -> Result<String, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        match interceptor.update_config(config).await {
            Ok(_) => Ok("拦截器配置已更新".to_string()),
            Err(e) => Err(format!("更新拦截器配置失败: {}", e)),
        }
    } else {
        Err("拦截器未运行".to_string())
    }
}

/// 获取默认配置
#[tauri::command]
pub async fn get_default_browser_interceptor_config() -> Result<BrowserInterceptorConfig, String> {
    Ok(BrowserInterceptorConfig::default())
}

/// 验证配置
#[tauri::command]
pub async fn validate_browser_interceptor_config(
    config: BrowserInterceptorConfig,
) -> Result<String, String> {
    match config.validate() {
        Ok(_) => Ok("配置验证通过".to_string()),
        Err(e) => Err(format!("配置验证失败: {}", e)),
    }
}

/// 检查拦截器是否正在运行
#[tauri::command]
pub async fn is_browser_interceptor_running(
    state: State<'_, BrowserInterceptorState>,
) -> Result<bool, String> {
    let interceptor_guard = state.0.read().await;
    Ok(interceptor_guard.is_some())
}

/// 获取拦截器统计信息
#[derive(Debug, Serialize, Deserialize)]
pub struct InterceptorStatistics {
    pub total_intercepted: usize,
    pub current_intercepted: usize,
    pub copied_count: usize,
    pub opened_count: usize,
    pub dismissed_count: usize,
}

#[tauri::command]
pub async fn get_browser_interceptor_statistics(
    state: State<'_, BrowserInterceptorState>,
) -> Result<InterceptorStatistics, String> {
    let interceptor_guard = state.0.read().await;

    if let Some(interceptor) = interceptor_guard.as_ref() {
        // 获取当前拦截的 URL
        let current_urls = interceptor
            .get_intercepted_urls()
            .await
            .map_err(|e| format!("获取当前 URL 失败: {}", e))?;

        // 获取历史记录
        let history = interceptor
            .get_history(None)
            .await
            .map_err(|e| format!("获取历史记录失败: {}", e))?;

        let copied_count = history.iter().filter(|u| u.copied).count();
        let opened_count = history.iter().filter(|u| u.opened_in_browser).count();
        let dismissed_count = history.iter().filter(|u| u.dismissed).count();

        Ok(InterceptorStatistics {
            total_intercepted: history.len(),
            current_intercepted: current_urls.len(),
            copied_count,
            opened_count,
            dismissed_count,
        })
    } else {
        Ok(InterceptorStatistics {
            total_intercepted: 0,
            current_intercepted: 0,
            copied_count: 0,
            opened_count: 0,
            dismissed_count: 0,
        })
    }
}

/// 通知相关结构体
#[derive(Debug, Serialize, Deserialize)]
pub struct NotificationPayload {
    pub title: String,
    pub body: String,
    pub icon: Option<String>,
}

/// 显示系统通知
#[tauri::command]
pub async fn show_notification(
    _app: AppHandle,
    title: String,
    body: String,
    _icon: Option<String>,
) -> Result<String, String> {
    // 简化实现，返回成功但实际功能待实现
    tracing::info!("显示通知: {} - {}", title, body);
    Ok("通知已记录到日志".to_string())
}

/// 显示URL拦截通知
#[tauri::command]
pub async fn show_url_intercept_notification(
    app: AppHandle,
    url: String,
    source_process: String,
) -> Result<String, String> {
    let title = "🔐 拦截到新的URL".to_string();
    let body = format!("来自 {}: {}", source_process, truncate_url(&url, 60));

    show_notification(app, title, body, Some("icon".to_string())).await
}

/// 显示状态变更通知
#[tauri::command]
pub async fn show_status_notification(
    app: AppHandle,
    message: String,
    notification_type: String,
) -> Result<String, String> {
    let (icon, title) = match notification_type.as_str() {
        "success" => ("✅", "操作成功"),
        "warning" => ("⚠️", "警告"),
        "error" => ("❌", "错误"),
        _ => ("ℹ️", "信息"),
    };

    let title = format!("{} {}", icon, title);
    show_notification(app, title, message, None).await
}

/// 截断URL用于通知显示
fn truncate_url(url: &str, max_length: usize) -> String {
    if url.len() <= max_length {
        url.to_string()
    } else {
        format!("{}...", &url[0..max_length])
    }
}
