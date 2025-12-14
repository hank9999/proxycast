//! 日志管理模块
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
}

pub struct LogStore {
    logs: Vec<LogEntry>,
    max_logs: usize,
}

impl Default for LogStore {
    fn default() -> Self {
        Self {
            logs: Vec::new(),
            max_logs: 1000,
        }
    }
}

impl LogStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, level: &str, message: &str) {
        let entry = LogEntry {
            timestamp: Utc::now().to_rfc3339(),
            level: level.to_string(),
            message: message.to_string(),
        };

        self.logs.push(entry);

        // 保持日志数量在限制内
        if self.logs.len() > self.max_logs {
            self.logs.remove(0);
        }
    }

    pub fn get_logs(&self) -> Vec<LogEntry> {
        self.logs.clone()
    }

    pub fn clear(&mut self) {
        self.logs.clear();
    }
}

#[allow(dead_code)]
pub type SharedLogStore = Arc<RwLock<LogStore>>;
