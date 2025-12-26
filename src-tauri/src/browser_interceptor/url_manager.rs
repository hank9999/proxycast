use crate::browser_interceptor::{BrowserInterceptorError, InterceptedUrl, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// URL 管理器，负责管理被拦截的 URL
pub struct UrlManager {
    intercepted_urls: Arc<RwLock<HashMap<String, InterceptedUrl>>>,
    history: Arc<RwLock<Vec<InterceptedUrl>>>,
    max_history_size: usize,
    storage_path: Option<String>,
}

impl UrlManager {
    pub fn new() -> Self {
        Self {
            intercepted_urls: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            max_history_size: 1000, // 最多保存 1000 条历史记录
            storage_path: None,
        }
    }

    /// 创建带持久化存储的 URL 管理器
    pub fn with_storage<P: AsRef<Path>>(storage_path: P) -> Result<Self> {
        let storage_path = storage_path.as_ref().to_string_lossy().to_string();
        let mut manager = Self {
            intercepted_urls: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            max_history_size: 1000,
            storage_path: Some(storage_path.clone()),
        };

        // 从文件加载历史记录
        manager.load_from_storage()?;

        Ok(manager)
    }

    /// 添加被拦截的 URL
    pub fn add_intercepted_url(&self, url: String, source_process: String) -> Result<String> {
        let intercepted_url = InterceptedUrl::new(url, source_process);
        let id = intercepted_url.id.clone();
        let url_for_log = intercepted_url.url.clone();
        let process_for_log = intercepted_url.source_process.clone();

        // 添加到当前拦截列表
        {
            let mut urls = self.intercepted_urls.write().map_err(|e| {
                BrowserInterceptorError::StateError(format!("添加拦截 URL 失败: {}", e))
            })?;
            urls.insert(id.clone(), intercepted_url.clone());
        }

        // 添加到历史记录
        {
            let mut history = self.history.write().map_err(|e| {
                BrowserInterceptorError::StateError(format!("添加历史记录失败: {}", e))
            })?;

            history.push(intercepted_url);

            // 限制历史记录大小
            if history.len() > self.max_history_size {
                history.remove(0);
            }
        }

        // 自动保存
        let _ = self.auto_save();

        tracing::info!(
            "已添加拦截 URL: {} (来源: {})",
            url_for_log,
            process_for_log
        );
        Ok(id)
    }

    /// 获取所有当前拦截的 URL
    pub fn get_intercepted_urls(&self) -> Result<Vec<InterceptedUrl>> {
        let urls = self.intercepted_urls.read().map_err(|e| {
            BrowserInterceptorError::StateError(format!("读取拦截 URL 失败: {}", e))
        })?;

        let mut result: Vec<InterceptedUrl> = urls.values().cloned().collect();
        result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp)); // 按时间倒序排列

        Ok(result)
    }

    /// 获取指定 ID 的拦截 URL
    pub fn get_intercepted_url(&self, id: &str) -> Result<Option<InterceptedUrl>> {
        let urls = self.intercepted_urls.read().map_err(|e| {
            BrowserInterceptorError::StateError(format!("读取拦截 URL 失败: {}", e))
        })?;

        Ok(urls.get(id).cloned())
    }

    /// 标记 URL 为已复制
    pub fn mark_as_copied(&self, id: &str) -> Result<()> {
        let mut urls = self.intercepted_urls.write().map_err(|e| {
            BrowserInterceptorError::StateError(format!("更新拦截 URL 失败: {}", e))
        })?;

        if let Some(url) = urls.get_mut(id) {
            url.copied = true;
            tracing::info!("URL {} 已标记为已复制", id);
        }

        Ok(())
    }

    /// 标记 URL 为已在浏览器中打开
    pub fn mark_as_opened(&self, id: &str) -> Result<()> {
        let mut urls = self.intercepted_urls.write().map_err(|e| {
            BrowserInterceptorError::StateError(format!("更新拦截 URL 失败: {}", e))
        })?;

        if let Some(url) = urls.get_mut(id) {
            url.opened_in_browser = true;
            tracing::info!("URL {} 已标记为已在浏览器中打开", id);
        }

        Ok(())
    }

    /// 忽略（移除）指定的 URL
    pub fn dismiss_url(&self, id: &str) -> Result<()> {
        let mut urls = self.intercepted_urls.write().map_err(|e| {
            BrowserInterceptorError::StateError(format!("移除拦截 URL 失败: {}", e))
        })?;

        if let Some(mut url) = urls.remove(id) {
            url.dismissed = true;

            // 更新历史记录中的状态
            let mut history = self.history.write().map_err(|e| {
                BrowserInterceptorError::StateError(format!("更新历史记录失败: {}", e))
            })?;

            if let Some(history_url) = history.iter_mut().find(|u| u.id == id) {
                history_url.dismissed = true;
            }

            tracing::info!("URL {} 已被忽略", id);
        }

        Ok(())
    }

    /// 清除所有当前拦截的 URL
    pub fn clear_intercepted_urls(&self) -> Result<()> {
        let mut urls = self.intercepted_urls.write().map_err(|e| {
            BrowserInterceptorError::StateError(format!("清除拦截 URL 失败: {}", e))
        })?;

        let count = urls.len();
        urls.clear();

        tracing::info!("已清除 {} 个拦截的 URL", count);
        Ok(())
    }

    /// 获取历史记录
    pub fn get_history(&self, limit: Option<usize>) -> Result<Vec<InterceptedUrl>> {
        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e)))?;

        let mut result = history.clone();
        result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp)); // 按时间倒序排列

        if let Some(limit) = limit {
            result.truncate(limit);
        }

        Ok(result)
    }

    /// 搜索历史记录
    pub fn search_history(&self, query: &str, limit: Option<usize>) -> Result<Vec<InterceptedUrl>> {
        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("搜索历史记录失败: {}", e)))?;

        let query_lower = query.to_lowercase();
        let mut result: Vec<InterceptedUrl> = history
            .iter()
            .filter(|url| {
                url.url.to_lowercase().contains(&query_lower)
                    || url.source_process.to_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect();

        result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        if let Some(limit) = limit {
            result.truncate(limit);
        }

        Ok(result)
    }

    /// 按日期范围获取历史记录
    pub fn get_history_by_date_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<InterceptedUrl>> {
        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e)))?;

        let mut result: Vec<InterceptedUrl> = history
            .iter()
            .filter(|url| url.timestamp >= start && url.timestamp <= end)
            .cloned()
            .collect();

        result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(result)
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> Result<UrlStatistics> {
        let urls = self.intercepted_urls.read().map_err(|e| {
            BrowserInterceptorError::StateError(format!("读取拦截 URL 失败: {}", e))
        })?;

        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e)))?;

        let current_count = urls.len();
        let total_intercepted = history.len();
        let copied_count = history.iter().filter(|u| u.copied).count();
        let opened_count = history.iter().filter(|u| u.opened_in_browser).count();
        let dismissed_count = history.iter().filter(|u| u.dismissed).count();

        // 统计来源进程
        let mut process_stats = HashMap::new();
        for url in history.iter() {
            *process_stats.entry(url.source_process.clone()).or_insert(0) += 1;
        }

        Ok(UrlStatistics {
            current_intercepted: current_count,
            total_intercepted,
            copied_count,
            opened_count,
            dismissed_count,
            process_stats,
        })
    }

    /// 清理过期的历史记录
    pub fn cleanup_old_history(&self, days: u32) -> Result<usize> {
        let cutoff_date = Utc::now() - chrono::Duration::days(days as i64);

        let mut history = self
            .history
            .write()
            .map_err(|e| BrowserInterceptorError::StateError(format!("清理历史记录失败: {}", e)))?;

        let original_len = history.len();
        history.retain(|url| url.timestamp > cutoff_date);
        let removed_count = original_len - history.len();

        if removed_count > 0 {
            tracing::info!("已清理 {} 条过期的历史记录", removed_count);
        }

        Ok(removed_count)
    }

    /// 保存到存储文件
    pub fn save_to_storage(&self) -> Result<()> {
        if let Some(storage_path) = &self.storage_path {
            let history = self.history.read().map_err(|e| {
                BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e))
            })?;

            let storage_data = UrlStorageData {
                history: history.clone(),
                max_history_size: self.max_history_size,
                saved_at: Utc::now(),
            };

            let json_data = serde_json::to_string_pretty(&storage_data).map_err(|e| {
                BrowserInterceptorError::StateError(format!("序列化数据失败: {}", e))
            })?;

            // 确保目录存在
            if let Some(parent) = Path::new(storage_path).parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    BrowserInterceptorError::StateError(format!("创建目录失败: {}", e))
                })?;
            }

            fs::write(storage_path, json_data)
                .map_err(|e| BrowserInterceptorError::StateError(format!("写入文件失败: {}", e)))?;

            tracing::info!("已保存历史记录到: {}", storage_path);
        }

        Ok(())
    }

    /// 从存储文件加载
    pub fn load_from_storage(&mut self) -> Result<()> {
        if let Some(storage_path) = &self.storage_path {
            if Path::new(storage_path).exists() {
                let json_data = fs::read_to_string(storage_path).map_err(|e| {
                    BrowserInterceptorError::StateError(format!("读取文件失败: {}", e))
                })?;

                let storage_data: UrlStorageData =
                    serde_json::from_str(&json_data).map_err(|e| {
                        BrowserInterceptorError::StateError(format!("反序列化数据失败: {}", e))
                    })?;

                {
                    let mut history = self.history.write().map_err(|e| {
                        BrowserInterceptorError::StateError(format!("写入历史记录失败: {}", e))
                    })?;
                    *history = storage_data.history;
                }

                self.max_history_size = storage_data.max_history_size;

                tracing::info!("已从 {} 加载历史记录", storage_path);
            }
        }

        Ok(())
    }

    /// 导出历史记录为 JSON
    pub fn export_history_json(&self, export_path: &str) -> Result<()> {
        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e)))?;

        let export_data = UrlExportData {
            urls: history.clone(),
            exported_at: Utc::now(),
            total_count: history.len(),
        };

        let json_data = serde_json::to_string_pretty(&export_data).map_err(|e| {
            BrowserInterceptorError::StateError(format!("序列化导出数据失败: {}", e))
        })?;

        // 确保目录存在
        if let Some(parent) = Path::new(export_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| BrowserInterceptorError::StateError(format!("创建目录失败: {}", e)))?;
        }

        fs::write(export_path, json_data)
            .map_err(|e| BrowserInterceptorError::StateError(format!("写入导出文件失败: {}", e)))?;

        tracing::info!("已导出历史记录到: {}", export_path);
        Ok(())
    }

    /// 导出历史记录为 CSV
    pub fn export_history_csv(&self, export_path: &str) -> Result<()> {
        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e)))?;

        let mut csv_content =
            String::from("ID,URL,Source Process,Timestamp,Copied,Opened in Browser,Dismissed\n");

        for url in history.iter() {
            csv_content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                url.id,
                url.url.replace(",", "%2C"), // 转义逗号
                url.source_process,
                url.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                url.copied,
                url.opened_in_browser,
                url.dismissed
            ));
        }

        // 确保目录存在
        if let Some(parent) = Path::new(export_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| BrowserInterceptorError::StateError(format!("创建目录失败: {}", e)))?;
        }

        fs::write(export_path, csv_content).map_err(|e| {
            BrowserInterceptorError::StateError(format!("写入 CSV 文件失败: {}", e))
        })?;

        tracing::info!("已导出历史记录到 CSV: {}", export_path);
        Ok(())
    }

    /// 验证 URL 是否有效
    pub fn validate_url(url: &str) -> bool {
        // 基本的 URL 验证
        if url.is_empty() {
            return false;
        }

        // 检查是否包含必要的协议
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return false;
        }

        // 使用 url crate 进行更严格的验证
        if let Ok(_) = url::Url::parse(url) {
            return true;
        }

        false
    }

    /// 按来源进程过滤历史记录
    pub fn get_history_by_process(
        &self,
        process: &str,
        limit: Option<usize>,
    ) -> Result<Vec<InterceptedUrl>> {
        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e)))?;

        let mut result: Vec<InterceptedUrl> = history
            .iter()
            .filter(|url| url.source_process.eq_ignore_ascii_case(process))
            .cloned()
            .collect();

        result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        if let Some(limit) = limit {
            result.truncate(limit);
        }

        Ok(result)
    }

    /// 获取所有唯一的来源进程列表
    pub fn get_unique_processes(&self) -> Result<Vec<String>> {
        let history = self
            .history
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取历史记录失败: {}", e)))?;

        let mut processes: Vec<String> = history
            .iter()
            .map(|url| url.source_process.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        processes.sort();
        Ok(processes)
    }

    /// 自动保存（如果已配置存储路径）
    fn auto_save(&self) -> Result<()> {
        if self.storage_path.is_some() {
            self.save_to_storage()?;
        }
        Ok(())
    }
}

/// 存储数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UrlStorageData {
    history: Vec<InterceptedUrl>,
    max_history_size: usize,
    saved_at: DateTime<Utc>,
}

/// 导出数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UrlExportData {
    urls: Vec<InterceptedUrl>,
    exported_at: DateTime<Utc>,
    total_count: usize,
}

/// URL 统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlStatistics {
    pub current_intercepted: usize,
    pub total_intercepted: usize,
    pub copied_count: usize,
    pub opened_count: usize,
    pub dismissed_count: usize,
    pub process_stats: HashMap<String, usize>,
}

impl Default for UrlManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_intercepted_url() {
        let manager = UrlManager::new();

        let id = manager
            .add_intercepted_url(
                "https://accounts.google.com/oauth/authorize".to_string(),
                "kiro".to_string(),
            )
            .unwrap();

        let urls = manager.get_intercepted_urls().unwrap();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].id, id);
        assert_eq!(urls[0].url, "https://accounts.google.com/oauth/authorize");
        assert_eq!(urls[0].source_process, "kiro");
    }

    #[test]
    fn test_mark_as_copied() {
        let manager = UrlManager::new();

        let id = manager
            .add_intercepted_url("https://test.com".to_string(), "test".to_string())
            .unwrap();

        manager.mark_as_copied(&id).unwrap();

        let url = manager.get_intercepted_url(&id).unwrap().unwrap();
        assert!(url.copied);
    }

    #[test]
    fn test_dismiss_url() {
        let manager = UrlManager::new();

        let id = manager
            .add_intercepted_url("https://test.com".to_string(), "test".to_string())
            .unwrap();

        manager.dismiss_url(&id).unwrap();

        let urls = manager.get_intercepted_urls().unwrap();
        assert_eq!(urls.len(), 0);

        // 但历史记录中应该还存在
        let history = manager.get_history(None).unwrap();
        assert_eq!(history.len(), 1);
        assert!(history[0].dismissed);
    }

    #[test]
    fn test_search_history() {
        let manager = UrlManager::new();

        manager
            .add_intercepted_url(
                "https://accounts.google.com/oauth".to_string(),
                "kiro".to_string(),
            )
            .unwrap();

        manager
            .add_intercepted_url("https://github.com/login".to_string(), "cursor".to_string())
            .unwrap();

        let results = manager.search_history("google", None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].url.contains("google"));

        let results = manager.search_history("cursor", None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_process, "cursor");
    }
}
