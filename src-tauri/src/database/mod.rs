pub mod dao;
pub mod migration;
pub mod schema;

use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub type DbConnection = Arc<Mutex<Connection>>;

/// 获取数据库文件路径
pub fn get_db_path() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or_else(|| "无法获取主目录".to_string())?;
    let db_dir = home.join(".proxycast");
    std::fs::create_dir_all(&db_dir)
        .map_err(|e| format!("无法创建数据库目录 {:?}: {}", db_dir, e))?;
    Ok(db_dir.join("proxycast.db"))
}

/// 初始化数据库连接
pub fn init_database() -> Result<DbConnection, String> {
    let db_path = get_db_path()?;
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // 创建表结构
    schema::create_tables(&conn).map_err(|e| e.to_string())?;
    migration::migrate_from_json(&conn)?;

    Ok(Arc::new(Mutex::new(conn)))
}
