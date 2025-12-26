import { useState, useEffect } from "react";
import {
  ExternalLink,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { ProviderIcon } from "@/icons/providers";

interface VersionInfo {
  current: string;
  latest?: string;
  hasUpdate: boolean;
  downloadUrl?: string;
  error?: string;
}

interface DownloadResult {
  success: boolean;
  message: string;
  filePath?: string;
}

interface ToolVersion {
  name: string;
  version: string | null;
  installed: boolean;
}

export function AboutSection() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    current: "",
    latest: undefined,
    hasUpdate: false,
    downloadUrl: undefined,
    error: undefined,
  });
  const [checking, setChecking] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadResult, setDownloadResult] = useState<DownloadResult | null>(
    null,
  );
  const [toolVersions, setToolVersions] = useState<ToolVersion[]>([]);
  const [loadingTools, setLoadingTools] = useState(true);

  // 加载当前版本号（从后端获取，确保与 Cargo.toml 同步）
  useEffect(() => {
    const loadCurrentVersion = async () => {
      try {
        // check_for_updates 会返回当前版本号
        const result = await invoke<VersionInfo>("check_for_updates");
        setVersionInfo((prev) => ({
          ...prev,
          current: result.current,
        }));
      } catch (error) {
        console.error("Failed to load version:", error);
      }
    };
    loadCurrentVersion();
  }, []);

  // 加载本地工具版本
  useEffect(() => {
    const loadToolVersions = async () => {
      try {
        const versions = await invoke<ToolVersion[]>("get_tool_versions");
        setToolVersions(versions);
      } catch (error) {
        console.error("Failed to load tool versions:", error);
      } finally {
        setLoadingTools(false);
      }
    };
    loadToolVersions();
  }, []);

  const handleCheckUpdate = async () => {
    setChecking(true);
    setDownloadResult(null);
    try {
      const result = await invoke<VersionInfo>("check_for_updates");
      setVersionInfo(result);
    } catch (error) {
      console.error("Failed to check for updates:", error);
      setVersionInfo((prev) => ({
        ...prev,
        error: "检查更新失败",
      }));
    } finally {
      setChecking(false);
    }
  };

  const handleDownloadUpdate = async () => {
    setDownloading(true);
    setDownloadResult(null);
    try {
      const result = await invoke<DownloadResult>("download_update");
      setDownloadResult(result);

      if (result.success) {
        // 下载成功，显示安装提示
        setTimeout(() => {
          setDownloadResult({
            ...result,
            message: "安装程序已启动，应用将自动关闭以完成更新",
          });
        }, 1000);
      } else {
        // 下载失败，显示错误信息
        console.error("Download failed:", result.message);
      }
    } catch (error) {
      console.error("Failed to download update:", error);
      setDownloadResult({
        success: false,
        message: "下载失败，请手动下载",
        filePath: undefined,
      });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      {/* 应用信息 */}
      <div className="p-6 rounded-lg border text-center space-y-4">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
          <ProviderIcon providerType="proxycast" size={40} />
        </div>

        <div>
          <h2 className="text-xl font-bold">ProxyCast</h2>
          <p className="text-sm text-muted-foreground">AI API 代理服务</p>
        </div>

        <div className="flex items-center justify-center gap-2">
          <span className="text-sm">版本 {versionInfo.current}</span>
          {versionInfo.hasUpdate ? (
            <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-700 text-xs">
              有新版本 {versionInfo.latest}
            </span>
          ) : versionInfo.error ? (
            <span className="flex items-center gap-1 text-xs text-red-500">
              <AlertCircle className="h-3 w-3" />
              {versionInfo.error}
            </span>
          ) : versionInfo.latest ? (
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <CheckCircle2 className="h-3 w-3" />
              已是最新
            </span>
          ) : null}
        </div>

        <div className="flex items-center justify-center gap-2">
          <button
            onClick={handleCheckUpdate}
            disabled={checking || downloading}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm hover:bg-muted disabled:opacity-50"
          >
            <RefreshCw
              className={`h-4 w-4 ${checking ? "animate-spin" : ""}`}
            />
            检查更新
          </button>

          {versionInfo.hasUpdate && (
            <>
              <button
                onClick={handleDownloadUpdate}
                disabled={downloading}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 text-white text-sm hover:bg-green-700 disabled:opacity-50"
              >
                <RefreshCw
                  className={`h-4 w-4 ${downloading ? "animate-spin" : ""}`}
                />
                {downloading ? "下载中..." : "下载更新"}
              </button>

              {versionInfo.downloadUrl && (
                <a
                  href={versionInfo.downloadUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm hover:bg-muted"
                >
                  <ExternalLink className="h-4 w-4" />
                  网页下载
                </a>
              )}
            </>
          )}
        </div>

        {/* 下载结果提示 */}
        {downloadResult && (
          <div
            className={`mt-2 p-3 rounded-lg text-sm ${
              downloadResult.success
                ? "bg-green-50 text-green-700 border border-green-200"
                : "bg-red-50 text-red-700 border border-red-200"
            }`}
          >
            <div className="flex items-start gap-2">
              {downloadResult.success ? (
                <CheckCircle2 className="h-4 w-4 mt-0.5 flex-shrink-0" />
              ) : (
                <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              )}
              <div className="flex-1">
                <p>{downloadResult.message}</p>
                {!downloadResult.success && versionInfo.downloadUrl && (
                  <a
                    href={versionInfo.downloadUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-2 underline hover:no-underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    前往网页下载
                  </a>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 链接 */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">相关链接</h3>
        <div className="space-y-2">
          <a
            href="https://github.com/aiclientproxy/proxycast"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
          >
            <span className="text-sm">GitHub 仓库</span>
            <ExternalLink className="h-4 w-4 text-muted-foreground" />
          </a>
          <a
            href="https://aiclientproxy.github.io/proxycast/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
          >
            <span className="text-sm">文档</span>
            <ExternalLink className="h-4 w-4 text-muted-foreground" />
          </a>
          <a
            href="https://github.com/aiclientproxy/proxycast/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
          >
            <span className="text-sm">问题反馈</span>
            <ExternalLink className="h-4 w-4 text-muted-foreground" />
          </a>
        </div>
      </div>

      {/* 使用说明 Q&A */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">使用说明</h3>
        <div className="space-y-2">
          <QAItem
            question="ProxyCast 是什么？"
            answer="ProxyCast 是一个本地 AI API 代理服务，可以将 Kiro、Gemini CLI 等工具的凭证转换为标准的 OpenAI/Anthropic API，供 Claude Code、Cherry Studio、Cursor 等工具使用。"
          />
          <QAItem
            question="如何开始使用？"
            answer="1. 在「凭证池」添加你的凭证（如 Kiro 凭证文件或 Claude API Key）；2. 在「API Server」启动服务并选择默认 Provider；3. 在你的 AI 工具中配置 API 地址为 http://localhost:8999"
          />
          <QAItem
            question="什么是配置切换？"
            answer="配置切换可以一键修改 Claude Code、Codex、Gemini CLI 的配置文件，快速在不同 Provider 间切换。添加 ProxyCast 配置后，这些工具就会使用本地代理服务。"
          />
          <QAItem
            question="凭证文件在哪里？"
            answer="Kiro 凭证：~/.kiro/kiro_creds.json；Gemini CLI 凭证：~/.gemini/oauth_creds.json；Qwen 凭证：~/.qwen-coder/auth.json"
          />
          <QAItem
            question="支持哪些 AI 工具？"
            answer="支持所有兼容 OpenAI API 或 Anthropic API 的工具，如 Claude Code、Cursor、Cherry Studio、Continue、Cline 等。"
          />
        </div>
      </div>

      {/* 本地工具版本 */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">本地工具版本</h3>
        <div className="p-4 rounded-lg border space-y-3">
          {loadingTools ? (
            <>
              <ToolVersionItem name="Claude Code" version="检测中..." />
              <ToolVersionItem name="Codex" version="检测中..." />
              <ToolVersionItem name="Gemini CLI" version="检测中..." />
            </>
          ) : (
            toolVersions.map((tool) => (
              <ToolVersionItem
                key={tool.name}
                name={tool.name}
                version={tool.installed ? tool.version || "已安装" : "未安装"}
              />
            ))
          )}
        </div>
      </div>

      {/* 版权信息 */}
      <div className="text-center text-xs text-muted-foreground pt-4 border-t">
        <p>Made with love for AI developers</p>
        <p className="mt-1">2025-2026 ProxyCast</p>
      </div>
    </div>
  );
}

function ToolVersionItem({ name, version }: { name: string; version: string }) {
  const isInstalled = version !== "未安装" && !version.includes("检测");

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm">{name}</span>
      <div className="flex items-center gap-2">
        {isInstalled ? (
          <CheckCircle2 className="h-4 w-4 text-green-500" />
        ) : (
          <AlertCircle className="h-4 w-4 text-muted-foreground" />
        )}
        <span className="text-sm text-muted-foreground font-mono">
          {version}
        </span>
      </div>
    </div>
  );
}

function QAItem({ question, answer }: { question: string; answer: string }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="rounded-lg border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between p-3 text-left hover:bg-muted/50"
      >
        <span className="text-sm font-medium">{question}</span>
        <span className="text-muted-foreground">{isOpen ? "−" : "+"}</span>
      </button>
      {isOpen && (
        <div className="px-3 pb-3 text-sm text-muted-foreground">{answer}</div>
      )}
    </div>
  );
}
