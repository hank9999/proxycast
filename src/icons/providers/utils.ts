// Provider 图标辅助函数

// 可用的图标名称列表
export const availableIcons = [
  "aws",
  "gemini",
  "anthropic",
  "claude",
  "qwen",
  "google",
  "openai",
  "alibaba",
  "copilot",
  "amp",
  "kiro",
  "deepseek",
  "zhipu",
  "kimi",
  "minimax",
  "doubao",
  "azure",
  "antigravity",
  "proxycast",
] as const;

// Provider 类型到图标名称的映射
export const providerTypeToIcon: Record<string, string> = {
  kiro: "kiro",
  gemini: "gemini",
  qwen: "qwen",
  antigravity: "antigravity",
  openai: "openai",
  claude: "claude",
  anthropic: "anthropic",
  codex: "openai",
  claude_oauth: "claude",
  iflow: "alibaba",
  amp: "amp",
  google: "google",
  alibaba: "alibaba",
  copilot: "copilot",
  deepseek: "deepseek",
  zhipu: "zhipu",
  kimi: "kimi",
  minimax: "minimax",
  doubao: "doubao",
  azure: "azure",
  aws: "aws",
  proxycast: "proxycast",
};

// 获取规范化的图标名称
export const getIconName = (providerType: string): string => {
  return providerTypeToIcon[providerType] || providerType;
};

// 检查是否有对应的图标
export const hasProviderIcon = (providerType: string): boolean => {
  const iconName = getIconName(providerType);
  return (availableIcons as readonly string[]).includes(iconName);
};
