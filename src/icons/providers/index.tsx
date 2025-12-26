import React, { useMemo, ComponentType, SVGProps } from "react";
import { cn } from "@/lib/utils";
import { providerTypeToIcon } from "./utils";

// 使用 SVGR 导入 SVG 组件
import AwsIcon from "./aws.svg?react";
import GeminiIcon from "./gemini.svg?react";
import AnthropicIcon from "./anthropic.svg?react";
import ClaudeIcon from "./claude.svg?react";
import QwenIcon from "./qwen.svg?react";
import GoogleIcon from "./google.svg?react";
import OpenaiIcon from "./openai.svg?react";
import AlibabaIcon from "./alibaba.svg?react";
import CopilotIcon from "./copilot.svg?react";
import AmpIcon from "./amp.svg?react";
import KiroIcon from "./kiro.svg?react";
import DeepseekIcon from "./deepseek.svg?react";
import ZhipuIcon from "./zhipu.svg?react";
import KimiIcon from "./kimi.svg?react";
import MinimaxIcon from "./minimax.svg?react";
import DoubaoIcon from "./doubao.svg?react";
import AzureIcon from "./azure.svg?react";
import AntigravityIcon from "./antigravity.svg?react";
import ProxycastIcon from "./proxycast.svg?react";

// 图标组件映射
const iconComponents: Record<string, ComponentType<SVGProps<SVGSVGElement>>> = {
  aws: AwsIcon,
  gemini: GeminiIcon,
  anthropic: AnthropicIcon,
  claude: ClaudeIcon,
  qwen: QwenIcon,
  google: GoogleIcon,
  openai: OpenaiIcon,
  alibaba: AlibabaIcon,
  copilot: CopilotIcon,
  amp: AmpIcon,
  kiro: KiroIcon,
  deepseek: DeepseekIcon,
  zhipu: ZhipuIcon,
  kimi: KimiIcon,
  minimax: MinimaxIcon,
  doubao: DoubaoIcon,
  azure: AzureIcon,
  antigravity: AntigravityIcon,
  proxycast: ProxycastIcon,
};

interface ProviderIconProps {
  providerType: string;
  size?: number | string;
  className?: string;
  showFallback?: boolean;
}

export const ProviderIcon: React.FC<ProviderIconProps> = ({
  providerType,
  size = 24,
  className,
  showFallback = true,
}) => {
  const iconName = providerTypeToIcon[providerType] || providerType;
  const IconComponent = iconComponents[iconName];

  const sizeStyle = useMemo(() => {
    const sizeValue = typeof size === "number" ? `${size}px` : size;
    return {
      width: sizeValue,
      height: sizeValue,
      fontSize: sizeValue,
      lineHeight: 1,
    };
  }, [size]);

  if (IconComponent) {
    return (
      <span
        className={cn(
          "inline-flex items-center justify-center flex-shrink-0",
          className,
        )}
        style={sizeStyle}
      >
        <IconComponent width="1em" height="1em" />
      </span>
    );
  }

  // Fallback：显示首字母
  if (showFallback) {
    const initials = providerType
      .split("_")
      .map((word) => word[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
    const fallbackFontSize =
      typeof size === "number" ? `${Math.max(size * 0.5, 12)}px` : "0.5em";
    return (
      <span
        className={cn(
          "inline-flex items-center justify-center flex-shrink-0 rounded-lg",
          "bg-muted text-muted-foreground font-semibold",
          className,
        )}
        style={sizeStyle}
      >
        <span style={{ fontSize: fallbackFontSize }}>{initials}</span>
      </span>
    );
  }

  return null;
};
