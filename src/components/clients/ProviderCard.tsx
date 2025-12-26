import { Check, Edit2, Trash2, Loader2 } from "lucide-react";
import { Provider } from "@/lib/api/switch";
import { cn } from "@/lib/utils";
import { ProviderIcon } from "@/icons/providers";
import { useState } from "react";

// 从供应商名称和分类推断图标类型
function getProviderTypeFromName(name: string, category: string): string {
  const lowerName = name.toLowerCase();

  // 精确匹配
  if (lowerName.includes("智谱") || lowerName.includes("glm")) return "zhipu";
  if (lowerName.includes("proxycast")) return "proxycast";
  if (lowerName.includes("deepseek")) return "deepseek";
  if (lowerName.includes("kimi")) return "kimi";
  if (lowerName.includes("minimax")) return "minimax";
  if (lowerName.includes("doubao") || lowerName.includes("豆包"))
    return "doubao";
  if (lowerName.includes("qwen") || lowerName.includes("通义")) return "qwen";
  if (lowerName.includes("claude")) return "claude";
  if (lowerName.includes("anthropic")) return "anthropic";
  if (lowerName.includes("openai")) return "openai";
  if (lowerName.includes("gemini")) return "gemini";
  if (lowerName.includes("google")) return "google";
  if (lowerName.includes("kiro")) return "kiro";
  if (lowerName.includes("azure")) return "azure";
  if (lowerName.includes("alibaba") || lowerName.includes("阿里"))
    return "alibaba";
  if (lowerName.includes("copilot")) return "copilot";

  // 根据分类推断
  if (category === "custom") return "custom";
  if (category === "proxy") return "amp";

  // 默认回退
  return lowerName.replace(/\s+/g, "");
}

interface ProviderCardProps {
  provider: Provider;
  isCurrent: boolean;
  onSwitch: () => void;
  onEdit: () => void;
  onDelete: () => void;
  switching?: boolean;
}

export function ProviderCard({
  provider,
  isCurrent,
  onSwitch,
  onEdit,
  onDelete,
  switching = false,
}: ProviderCardProps) {
  const [showConfirm, setShowConfirm] = useState(false);
  const isProxyCast =
    provider.category === "custom" && provider.name === "ProxyCast";

  const handleClick = () => {
    if (isCurrent || switching) return;

    // 对于关键配置，显示确认对话框
    if (provider.category === "official" || isProxyCast) {
      setShowConfirm(true);
    } else {
      onSwitch();
    }
  };

  const handleConfirmSwitch = () => {
    setShowConfirm(false);
    onSwitch();
  };

  return (
    <>
      <div
        onClick={handleClick}
        className={cn(
          "group relative rounded-xl border p-3 transition-all",
          switching
            ? "cursor-wait opacity-75"
            : isCurrent
              ? "cursor-default border-primary bg-gradient-to-r from-primary/10 to-transparent shadow-sm"
              : "cursor-pointer hover:border-primary/50 hover:shadow-md",
        )}
      >
        {/* 选中标记或切换状态 */}
        {switching ? (
          <div className="absolute -top-1.5 -right-1.5 rounded-full bg-blue-500 p-1 shadow-sm">
            <Loader2 className="h-3 w-3 text-white animate-spin" />
          </div>
        ) : isCurrent ? (
          <div className="absolute -top-1.5 -right-1.5 rounded-full bg-primary p-1 shadow-sm">
            <Check className="h-3 w-3 text-primary-foreground" />
          </div>
        ) : null}

        <div className="flex items-center gap-3">
          {/* 图标 */}
          <div
            className={cn(
              "shrink-0 h-10 w-10 rounded-lg flex items-center justify-center",
              isCurrent ? "bg-primary/20" : "bg-muted",
            )}
          >
            <ProviderIcon
              providerType={getProviderTypeFromName(
                provider.name,
                provider.category || "",
              )}
              size={22}
              showFallback={true}
            />
          </div>

          {/* 名称和分类 */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-medium truncate">{provider.name}</h3>
              {provider.category && (
                <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                  {provider.category}
                </span>
              )}
            </div>
            {isProxyCast ? (
              <p className="text-xs text-blue-600 dark:text-blue-400 truncate">
                凭证池 → 标准 API
              </p>
            ) : provider.notes ? (
              <p className="text-xs text-muted-foreground truncate">
                {provider.notes}
              </p>
            ) : null}
          </div>

          {/* 操作按钮 */}
          <div
            className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={onEdit}
              className="p-1.5 rounded hover:bg-muted"
              title="编辑"
            >
              <Edit2 className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => !isCurrent && onDelete()}
              disabled={isCurrent}
              className={cn(
                "p-1.5 rounded text-destructive",
                isCurrent
                  ? "opacity-30 cursor-not-allowed"
                  : "hover:bg-destructive/10",
              )}
              title={isCurrent ? "无法删除当前使用中的配置" : "删除"}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* 切换确认对话框 */}
      {showConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-sm mx-4 shadow-xl">
            <h3 className="text-lg font-medium mb-2">确认切换配置</h3>
            <p className="text-sm text-muted-foreground mb-4">
              您正在切换到 "{provider.name}" 配置，这将更改当前的 API
              设置。确定要继续吗？
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowConfirm(false)}
                className="px-3 py-1.5 text-sm rounded border hover:bg-muted"
              >
                取消
              </button>
              <button
                onClick={handleConfirmSwitch}
                className="px-3 py-1.5 text-sm rounded bg-primary text-primary-foreground hover:bg-primary/90"
              >
                确认切换
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
