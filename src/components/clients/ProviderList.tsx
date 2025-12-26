import { useState, useEffect, useMemo } from "react";
import {
  Plus,
  RefreshCw,
  Eye,
  GitCompare,
  Download,
  AlertTriangle,
} from "lucide-react";
import {
  AppType,
  SyncCheckResult,
  switchApi,
  Provider,
} from "@/lib/api/switch";
import { useSwitch } from "@/hooks/useSwitch";
import { ProviderCard } from "./ProviderCard";
import { ProviderForm } from "./ProviderForm";
import { LiveConfigModal } from "./LiveConfigModal";
import { ConfigSyncDialog } from "./ConfigSyncDialog";
import { ConfirmDialog } from "@/components/ConfirmDialog";

// 敏感字段关键词
const SENSITIVE_KEYS = [
  "key",
  "token",
  "secret",
  "password",
  "auth",
  "credential",
  "api_key",
  "apikey",
  "access_token",
  "refresh_token",
];

// 脱敏函数：对敏感值进行遮盖
function maskSensitiveValue(value: string): string {
  if (value.length <= 8) return "****";
  return value.slice(0, 4) + "****" + value.slice(-4);
}

// 递归脱敏对象中的敏感字段
function maskSensitiveData(
  data: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(data)) {
    const lowerKey = key.toLowerCase();
    const isSensitive = SENSITIVE_KEYS.some((k) => lowerKey.includes(k));

    if (typeof value === "string" && isSensitive && value.length > 0) {
      result[key] = maskSensitiveValue(value);
    } else if (
      typeof value === "object" &&
      value !== null &&
      !Array.isArray(value)
    ) {
      result[key] = maskSensitiveData(value as Record<string, unknown>);
    } else {
      result[key] = value;
    }
  }

  return result;
}

// 比较两个配置是否匹配（忽略非关键字段）
function configsMatch(
  liveConfig: Record<string, unknown>,
  providerConfig: Record<string, unknown>,
): boolean {
  // 提取关键字段进行比较
  // Claude: { env: { ANTHROPIC_AUTH_TOKEN: ..., ANTHROPIC_API_KEY: ... } }
  // Gemini: { env: { GEMINI_API_KEY: ..., GOOGLE_API_KEY: ... } }
  // Codex: { auth: { OPENAI_API_KEY: ... }, config: ... }
  const getLiveEnv = (config: Record<string, unknown>) =>
    (config.env as Record<string, unknown>) || {};
  const getProviderEnv = (config: Record<string, unknown>) =>
    (config.env as Record<string, unknown>) || {};

  // Codex 特殊处理：从 auth 对象中提取
  const getLiveAuth = (config: Record<string, unknown>) =>
    (config.auth as Record<string, unknown>) || {};
  const getProviderAuth = (config: Record<string, unknown>) =>
    (config.auth as Record<string, unknown>) || {};

  const liveEnv = getLiveEnv(liveConfig);
  const providerEnv = getProviderEnv(providerConfig);
  const liveAuth = getLiveAuth(liveConfig);
  const providerAuth = getProviderAuth(providerConfig);

  // 比较关键认证字段（Claude/Gemini）
  const envKeysToCompare = [
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
  ];

  // 检查是否有任何关键字段匹配
  let hasAnyMatch = false;
  let hasConflict = false;

  // 比较 env 字段 - 修复：更宽松的匹配逻辑
  for (const key of envKeysToCompare) {
    const liveVal = liveEnv[key] as string | undefined;
    const providerVal = providerEnv[key] as string | undefined;

    // 如果两个值都存在且不为空，它们必须相等
    if (liveVal && liveVal.trim() && providerVal && providerVal.trim()) {
      if (liveVal !== providerVal) {
        hasConflict = true;
      } else {
        hasAnyMatch = true;
      }
    }
    // 如果只有一方有值，不算冲突，但也不算匹配
    // 这允许配置中可能存在额外的字段
  }

  // 比较 Codex auth.OPENAI_API_KEY
  const liveApiKey = liveAuth.OPENAI_API_KEY as string | undefined;
  const providerApiKey = providerAuth.OPENAI_API_KEY as string | undefined;

  if (
    liveApiKey &&
    liveApiKey.trim() &&
    providerApiKey &&
    providerApiKey.trim()
  ) {
    if (liveApiKey !== providerApiKey) {
      hasConflict = true;
    } else {
      hasAnyMatch = true;
    }
  }

  // 如果有冲突，返回 false
  // 如果有任何匹配且无冲突，返回 true
  // 如果没有匹配也没有冲突（比如配置为空），返回 false
  return hasAnyMatch && !hasConflict;
}

// 从实际配置中提取简短描述
function getLiveConfigSummary(config: Record<string, unknown>): string {
  const env = (config.env as Record<string, unknown>) || {};
  const auth = (config.auth as Record<string, unknown>) || {};

  if (env.ANTHROPIC_AUTH_TOKEN) {
    const baseUrl = env.ANTHROPIC_BASE_URL as string;
    if (baseUrl) {
      try {
        const url = new URL(baseUrl);
        return `OAuth · ${url.host}`;
      } catch {
        return `OAuth · ${baseUrl}`;
      }
    }
    return "Claude OAuth";
  }

  if (env.ANTHROPIC_API_KEY) {
    const baseUrl = env.ANTHROPIC_BASE_URL as string;
    if (baseUrl) {
      try {
        const url = new URL(baseUrl);
        return `API Key · ${url.host}`;
      } catch {
        return `API Key · ${baseUrl}`;
      }
    }
    return "Claude API Key";
  }

  // Gemini: GEMINI_API_KEY 或 GOOGLE_API_KEY
  if (env.GEMINI_API_KEY || env.GOOGLE_API_KEY) {
    return "Gemini API Key";
  }

  // Codex: auth.OPENAI_API_KEY
  if (auth.OPENAI_API_KEY) {
    return "Codex API Key";
  }

  return "未知配置";
}

interface ProviderListProps {
  appType: AppType;
}

export function ProviderList({ appType }: ProviderListProps) {
  const {
    providers,
    currentProvider,
    loading,
    error,
    addProvider,
    updateProvider,
    deleteProvider,
    switchToProvider,
    refresh,
    checkConfigSync,
    syncFromExternal,
  } = useSwitch(appType);

  const [showForm, setShowForm] = useState(false);
  const [editingProvider, setEditingProvider] = useState<
    (typeof providers)[0] | null
  >(null);
  const [showLiveConfig, setShowLiveConfig] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [showSyncDialog, setShowSyncDialog] = useState(false);
  const [syncResult, setSyncResult] = useState<SyncCheckResult | null>(null);
  const [checkingSync, setCheckingSync] = useState(false);
  const [switchingId, setSwitchingId] = useState<string | null>(null);

  // 实际生效的配置
  const [liveConfig, setLiveConfig] = useState<Record<string, unknown> | null>(
    null,
  );
  const [loadingLiveConfig, setLoadingLiveConfig] = useState(false);
  const [importingConfig, setImportingConfig] = useState(false);

  // 始终读取当前生效的配置（用于检测外部变更）
  useEffect(() => {
    const loadConfig = async () => {
      if (appType === "proxycast") return;
      setLoadingLiveConfig(true);
      try {
        const config = await switchApi.readLiveSettings(appType);
        setLiveConfig(config);
      } catch {
        setLiveConfig(null);
      } finally {
        setLoadingLiveConfig(false);
      }
    };

    if (!loading) {
      loadConfig();
    }
  }, [loading, appType]);

  // 检测实际配置是否与当前选中的 provider 匹配
  const configMismatch = useMemo(() => {
    if (!liveConfig || !currentProvider || loadingLiveConfig) return null;
    if (Object.keys(liveConfig).length === 0) return null;

    const matches = configsMatch(liveConfig, currentProvider.settings_config);
    if (matches) return null;

    // 检查是否有其他 provider 匹配实际配置
    const matchingProvider = providers.find((p) =>
      configsMatch(liveConfig, p.settings_config),
    );

    return {
      liveConfig,
      liveSummary: getLiveConfigSummary(liveConfig),
      matchingProvider,
    };
  }, [liveConfig, currentProvider, providers, loadingLiveConfig]);

  const handleImportCurrentConfig = async () => {
    if (!liveConfig) return;
    setImportingConfig(true);
    try {
      // 直接使用读取到的配置创建新的 provider
      const providerName = `导入配置 ${new Date().toLocaleDateString()}`;
      await addProvider({
        name: providerName,
        app_type: appType,
        settings_config: liveConfig,
        category: "custom",
      });
      // 重新加载配置
      const config = await switchApi.readLiveSettings(appType);
      setLiveConfig(config);
      await refresh();
    } catch (e) {
      alert("导入失败: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setImportingConfig(false);
    }
  };

  // 切换到匹配的 provider
  const handleSwitchToMatching = async (provider: Provider) => {
    try {
      setSwitchingId(provider.id);
      await switchToProvider(provider.id);
      // 切换后重新读取实际配置，更新 UI 状态
      const config = await switchApi.readLiveSettings(appType);
      setLiveConfig(config);
    } catch (e) {
      console.error("切换失败:", e);
    } finally {
      setSwitchingId(null);
    }
  };

  const handleAdd = () => {
    setEditingProvider(null);
    setShowForm(true);
  };

  const handleEdit = (provider: (typeof providers)[0]) => {
    setEditingProvider(provider);
    setShowForm(true);
  };

  const handleSave = async (data: Parameters<typeof addProvider>[0]) => {
    if (editingProvider) {
      await updateProvider({ ...editingProvider, ...data });
    } else {
      await addProvider(data);
    }
    setShowForm(false);
    setEditingProvider(null);
  };

  const handleDeleteClick = (id: string) => {
    // 不能删除当前使用中的 provider
    if (currentProvider?.id === id) {
      alert("无法删除当前使用中的配置");
      return;
    }
    setDeleteConfirm(id);
  };

  const handleDeleteConfirm = async () => {
    if (!deleteConfirm) return;
    try {
      await deleteProvider(deleteConfirm);
    } catch (e) {
      alert("删除失败: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setDeleteConfirm(null);
    }
  };

  const handleCheckSync = async () => {
    setCheckingSync(true);
    try {
      const result = await checkConfigSync();
      setSyncResult(result);
      setShowSyncDialog(true);
    } catch (_e) {
      // Error is handled in the hook
    } finally {
      setCheckingSync(false);
    }
  };

  const handleSyncFromExternal = async () => {
    await syncFromExternal();
    // 重新检查同步状态
    const result = await checkConfigSync();
    setSyncResult(result);
  };

  const handleRefreshSyncCheck = async () => {
    const result = await checkConfigSync();
    setSyncResult(result);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-destructive bg-destructive/10 p-3 text-sm">
        <p className="text-destructive">{error}</p>
        <button
          onClick={refresh}
          className="mt-1 text-xs text-muted-foreground hover:underline"
        >
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* 工具栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          {loadingLiveConfig ? (
            <span className="flex items-center gap-1">
              <RefreshCw className="h-3 w-3 animate-spin" />
              检测中...
            </span>
          ) : configMismatch ? (
            <span className="flex items-center gap-1 text-amber-600">
              <AlertTriangle className="h-3.5 w-3.5" />
              实际: {configMismatch.liveSummary}
            </span>
          ) : (
            <span>当前: {currentProvider?.name || "未设置"}</span>
          )}
          <button
            onClick={() => setShowLiveConfig(true)}
            className="p-1 rounded hover:bg-muted"
            title="查看当前生效的配置"
          >
            <Eye className="h-3.5 w-3.5" />
          </button>
        </div>
        <div className="flex gap-1.5">
          <button
            onClick={handleCheckSync}
            disabled={checkingSync}
            className="p-1.5 rounded-lg hover:bg-muted"
            title="检查外部配置同步状态"
          >
            <GitCompare
              className={`h-4 w-4 ${checkingSync ? "animate-pulse" : ""}`}
            />
          </button>
          <button
            onClick={refresh}
            className="p-1.5 rounded-lg hover:bg-muted"
            title="刷新"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
          <button
            onClick={handleAdd}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm"
          >
            <Plus className="h-3.5 w-3.5" />
            添加
          </button>
        </div>
      </div>

      {/* 配置不匹配警告 */}
      {configMismatch && (
        <div className="border border-amber-500/50 bg-amber-500/10 rounded-lg p-3 space-y-2">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-amber-700 dark:text-amber-400">
                检测到外部配置变更
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                实际生效的配置与当前选中的 "{currentProvider?.name}" 不一致
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {configMismatch.matchingProvider ? (
              <button
                onClick={() =>
                  handleSwitchToMatching(configMismatch.matchingProvider!)
                }
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-600 text-white text-sm"
              >
                切换到 "{configMismatch.matchingProvider.name}"
              </button>
            ) : (
              <button
                onClick={handleImportCurrentConfig}
                disabled={importingConfig}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-600 text-white text-sm disabled:opacity-50"
              >
                <Download className="h-3.5 w-3.5" />
                {importingConfig ? "导入中..." : "导入为新配置"}
              </button>
            )}
            <button
              onClick={() => setShowLiveConfig(true)}
              className="px-3 py-1.5 rounded-lg border text-sm hover:bg-muted"
            >
              查看详情
            </button>
          </div>
        </div>
      )}

      {/* Provider 列表 */}
      {providers.length === 0 ? (
        <div className="border border-dashed rounded-lg p-4">
          {loadingLiveConfig ? (
            <div className="flex items-center justify-center py-4">
              <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          ) : liveConfig && Object.keys(liveConfig).length > 0 ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  检测到当前配置，可一键导入
                </p>
                <button
                  onClick={handleImportCurrentConfig}
                  disabled={importingConfig}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm disabled:opacity-50"
                >
                  <Download className="h-3.5 w-3.5" />
                  {importingConfig ? "导入中..." : "导入配置"}
                </button>
              </div>
              <pre className="p-3 rounded-lg bg-muted/50 font-mono text-xs overflow-auto max-h-40">
                {JSON.stringify(maskSensitiveData(liveConfig), null, 2)}
              </pre>
            </div>
          ) : (
            <div className="text-center py-4 text-muted-foreground">
              <p>暂无配置</p>
              <button
                onClick={handleAdd}
                className="mt-2 text-sm text-primary hover:underline"
              >
                添加第一个配置
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2">
          {providers.map((provider) => (
            <ProviderCard
              key={provider.id}
              provider={provider}
              isCurrent={provider.id === currentProvider?.id}
              switching={switchingId === provider.id}
              onSwitch={async () => {
                try {
                  setSwitchingId(provider.id);
                  await switchToProvider(provider.id);
                  // 切换后重新读取实际配置
                  const config = await switchApi.readLiveSettings(appType);
                  setLiveConfig(config);
                } catch (e) {
                  console.error("切换失败:", e);
                } finally {
                  setSwitchingId(null);
                }
              }}
              onEdit={() => handleEdit(provider)}
              onDelete={() => handleDeleteClick(provider.id)}
            />
          ))}
        </div>
      )}

      {showForm && (
        <ProviderForm
          appType={appType}
          provider={editingProvider}
          onSave={handleSave}
          onCancel={() => {
            setShowForm(false);
            setEditingProvider(null);
          }}
        />
      )}

      {showLiveConfig && (
        <LiveConfigModal
          appType={appType}
          onClose={() => setShowLiveConfig(false)}
        />
      )}

      <ConfirmDialog
        isOpen={!!deleteConfirm}
        title="删除确认"
        message="确定要删除这个 Provider 吗？"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteConfirm(null)}
      />

      <ConfigSyncDialog
        isOpen={showSyncDialog}
        syncResult={syncResult}
        onClose={() => setShowSyncDialog(false)}
        onSyncFromExternal={handleSyncFromExternal}
        onRefreshCheck={handleRefreshSyncCheck}
      />
    </div>
  );
}
