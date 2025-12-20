fn main() {
    // tauri::generate_context! 在编译期会校验 `frontendDist` 路径是否存在。
    // 开发/CI 场景下可能只跑 `cargo check/test` 而未先构建前端，从而导致宏 panic。
    // 这里提前创建配置中的 `../dist` 目录，避免无关的编译阻塞。
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let dist_dir = std::path::PathBuf::from(manifest_dir).join("../dist");
        let _ = std::fs::create_dir_all(dist_dir);
    }
    tauri_build::build()
}
