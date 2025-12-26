import React, { useState } from "react";
import {
  RotateCcw,
  Pause,
  Shield,
  AlertCircle,
  CheckCircle,
  Clock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { formatDistanceToNow } from "date-fns";
import { zhCN } from "date-fns/locale";
import { InterceptorState } from "@/lib/api/browserInterceptor";
import * as browserInterceptorApi from "@/lib/api/browserInterceptor";

interface StatusControlPanelProps {
  state: InterceptorState | null;
  onStateChange: () => void;
}

export function StatusControlPanel({
  state,
  onStateChange,
}: StatusControlPanelProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleToggleInterceptor = async (enabled: boolean) => {
    setIsLoading(true);
    setError(null);
    try {
      if (enabled) {
        // 使用默认配置启动拦截器，确保 enabled 为 true
        const defaultConfig =
          await browserInterceptorApi.getDefaultBrowserInterceptorConfig();
        await browserInterceptorApi.startBrowserInterceptor({
          ...defaultConfig,
          enabled: true, // 覆盖默认值，确保启动
        });
      } else {
        await browserInterceptorApi.stopBrowserInterceptor();
      }
      onStateChange();
    } catch (err) {
      const errorMessage = String(err);
      console.error("切换拦截器状态失败:", errorMessage);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestoreNormal = async () => {
    setIsLoading(true);
    try {
      await browserInterceptorApi.restoreNormalBrowserBehavior();
      onStateChange();
    } catch (error) {
      console.error("恢复正常浏览器行为失败:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTemporaryDisable = async (minutes: number) => {
    setIsLoading(true);
    try {
      await browserInterceptorApi.temporaryDisableInterceptor(minutes * 60);
      onStateChange();
    } catch (error) {
      console.error("临时禁用拦截器失败:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = () => {
    if (!state) return "bg-gray-400";
    return state.enabled ? "bg-green-500" : "bg-gray-400";
  };

  const getStatusText = () => {
    if (!state) return "已停止";
    return state.enabled ? "拦截中" : "已停止";
  };

  const getStatusIcon = () => {
    if (!state) return <Pause className="w-4 h-4" />;
    return state.enabled ? (
      <CheckCircle className="w-4 h-4" />
    ) : (
      <Pause className="w-4 h-4" />
    );
  };

  return (
    <Card className="w-80">
      <CardContent className="p-4">
        {/* 状态显示 */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${getStatusColor()}`} />
            <div className="flex items-center space-x-2">
              {getStatusIcon()}
              <span className="font-medium">{getStatusText()}</span>
            </div>
          </div>

          <Switch
            checked={state?.enabled || false}
            onCheckedChange={handleToggleInterceptor}
            disabled={isLoading}
          />
        </div>

        {/* 统计信息 */}
        <div className="space-y-2 text-sm text-gray-600 mb-4">
          <div className="flex justify-between">
            <span>已拦截:</span>
            <Badge variant="outline">
              {state?.intercepted_count || 0} 个 URL
            </Badge>
          </div>
          <div className="flex justify-between">
            <span>活跃钩子:</span>
            <Badge variant="outline">
              {state?.active_hooks?.length || 0} 个
            </Badge>
          </div>
          {state?.last_activity && (
            <div className="flex justify-between">
              <span>最后活动:</span>
              <span className="text-xs">
                {formatDistanceToNow(new Date(state.last_activity), {
                  addSuffix: true,
                  locale: zhCN,
                })}
              </span>
            </div>
          )}
        </div>

        {/* 控制按钮 */}
        <div className="flex space-x-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleRestoreNormal}
            disabled={!state?.can_restore || isLoading}
            className="flex-1"
          >
            <RotateCcw className="w-4 h-4 mr-1" />
            恢复正常
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="outline" disabled={isLoading}>
                <Pause className="w-4 h-4 mr-1" />
                临时禁用
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => handleTemporaryDisable(5)}>
                <Clock className="w-4 h-4 mr-2" />
                禁用 5 分钟
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleTemporaryDisable(15)}>
                <Clock className="w-4 h-4 mr-2" />
                禁用 15 分钟
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleTemporaryDisable(30)}>
                <Clock className="w-4 h-4 mr-2" />
                禁用 30 分钟
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleTemporaryDisable(60)}>
                <Clock className="w-4 h-4 mr-2" />
                禁用 1 小时
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* 状态提示 */}
        {state?.enabled && (
          <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-green-600" />
              <span className="text-sm text-green-700 font-medium">
                拦截器正在运行
              </span>
            </div>
            <p className="text-xs text-green-600 mt-1">
              桌面应用的浏览器启动请求将被拦截
            </p>
          </div>
        )}

        {!state?.enabled && state?.can_restore && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-yellow-600" />
              <span className="text-sm text-yellow-700 font-medium">
                可以恢复正常
              </span>
            </div>
            <p className="text-xs text-yellow-600 mt-1">
              点击"恢复正常"按钮完全恢复浏览器行为
            </p>
          </div>
        )}

        {isLoading && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-blue-700 font-medium">
                操作中...
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-sm text-red-700 font-medium">
                  启动失败
                </span>
                <p className="text-xs text-red-600 mt-1 break-all">{error}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
