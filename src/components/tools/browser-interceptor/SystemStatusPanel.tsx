import React, { useState, useEffect, useCallback } from "react";
import {
  Activity,
  Shield,
  Database,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Monitor,
  Settings,
  Link,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatDistanceToNow } from "date-fns";
import { zhCN } from "date-fns/locale";
import { InterceptorState } from "@/lib/api/browserInterceptor";

interface SystemStatusPanelProps {
  state: InterceptorState | null;
}

interface SystemDiagnostics {
  registry_backup_exists: boolean;
  registry_backup_size: number;
  active_hooks_count: number;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  uptime_seconds: number;
  last_error?: string;
  last_error_time?: string;
  total_intercepted_urls: number;
  successful_operations: number;
  failed_operations: number;
}

export function SystemStatusPanel({ state }: SystemStatusPanelProps) {
  const [diagnostics, setDiagnostics] = useState<SystemDiagnostics | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadDiagnostics = useCallback(async () => {
    try {
      // 模拟诊断信息加载（实际项目中这些信息应该来自后端）
      const mockDiagnostics: SystemDiagnostics = {
        registry_backup_exists: true,
        registry_backup_size: 2048,
        active_hooks_count: state?.active_hooks?.length || 0,
        memory_usage_mb: Math.random() * 50 + 20, // 模拟20-70MB
        cpu_usage_percent: Math.random() * 5 + 1, // 模拟1-6% CPU使用率
        uptime_seconds: Date.now() / 1000 - (Date.now() / 1000 - 3600), // 模拟运行1小时
        total_intercepted_urls: state?.intercepted_count || 0,
        successful_operations: Math.floor(
          (state?.intercepted_count || 0) * 0.95,
        ),
        failed_operations: Math.floor((state?.intercepted_count || 0) * 0.05),
      };

      setDiagnostics(mockDiagnostics);
    } catch (error) {
      console.error("加载诊断信息失败:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [state?.active_hooks?.length, state?.intercepted_count]);

  useEffect(() => {
    loadDiagnostics();

    // 每30秒自动刷新诊断信息
    const interval = setInterval(loadDiagnostics, 30000);

    return () => clearInterval(interval);
  }, [loadDiagnostics]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadDiagnostics();
  };

  const getStatusColor = (status: "good" | "warning" | "error") => {
    switch (status) {
      case "good":
        return "text-green-600 bg-green-50 border-green-200";
      case "warning":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "error":
        return "text-red-600 bg-red-50 border-red-200";
    }
  };

  const getOverallStatus = () => {
    if (!state) return { status: "error", text: "无法获取状态" };
    if (!state.enabled) return { status: "warning", text: "拦截器已停用" };
    if (!diagnostics?.registry_backup_exists)
      return { status: "error", text: "注册表备份丢失" };
    if (diagnostics?.failed_operations > 0)
      return { status: "warning", text: "存在失败操作" };
    return { status: "good", text: "系统运行正常" };
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-8">
          <div className="flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mr-2" />
            <span>加载系统状态中...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const overallStatus = getOverallStatus();

  return (
    <div className="space-y-6">
      {/* 系统概览 */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          系统状态监控
        </h2>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <RefreshCw
            className={`w-4 h-4 mr-1 ${refreshing ? "animate-spin" : ""}`}
          />
          刷新
        </Button>
      </div>

      {/* 整体状态卡片 */}
      <Card
        className={`border ${getStatusColor(overallStatus.status as "good" | "warning" | "error")}`}
      >
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {overallStatus.status === "good" && (
                <CheckCircle className="w-6 h-6 text-green-600" />
              )}
              {overallStatus.status === "warning" && (
                <AlertTriangle className="w-6 h-6 text-yellow-600" />
              )}
              {overallStatus.status === "error" && (
                <XCircle className="w-6 h-6 text-red-600" />
              )}
              <div>
                <h3 className="font-semibold">
                  系统状态：{overallStatus.text}
                </h3>
                <p className="text-sm opacity-75">
                  {state?.last_activity && (
                    <>
                      最后活动：
                      {formatDistanceToNow(new Date(state.last_activity), {
                        addSuffix: true,
                        locale: zhCN,
                      })}
                    </>
                  )}
                </p>
              </div>
            </div>
            <Badge
              variant={
                overallStatus.status === "good" ? "default" : "destructive"
              }
            >
              {state?.enabled ? "运行中" : "已停止"}
            </Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 拦截器状态 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Shield className="w-4 h-4 mr-2" />
              拦截器状态
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">启用状态</span>
              <Badge variant={state?.enabled ? "default" : "secondary"}>
                {state?.enabled ? "已启用" : "已禁用"}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">活跃钩子</span>
              <span className="text-sm font-medium">
                {state?.active_hooks?.length || 0} 个
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">拦截计数</span>
              <span className="text-sm font-medium">
                {state?.intercepted_count || 0} 个
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">恢复能力</span>
              <Badge variant={state?.can_restore ? "default" : "destructive"}>
                {state?.can_restore ? "可恢复" : "无法恢复"}
              </Badge>
            </div>

            {diagnostics?.last_error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
                  <div>
                    <div className="text-sm font-medium text-red-800">
                      最近错误
                    </div>
                    <div className="text-xs text-red-700 mt-1">
                      {diagnostics.last_error}
                    </div>
                    {diagnostics.last_error_time && (
                      <div className="text-xs text-red-600 mt-1">
                        {formatDistanceToNow(
                          new Date(diagnostics.last_error_time),
                          { addSuffix: true, locale: zhCN },
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 系统资源 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Monitor className="w-4 h-4 mr-2" />
              系统资源
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>内存使用</span>
                <span>
                  {diagnostics?.memory_usage_mb?.toFixed(1) || "0"} MB
                </span>
              </div>
              <Progress
                value={((diagnostics?.memory_usage_mb || 0) / 100) * 100}
                className="h-2"
              />
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>CPU 使用率</span>
                <span>
                  {diagnostics?.cpu_usage_percent?.toFixed(1) || "0"}%
                </span>
              </div>
              <Progress
                value={diagnostics?.cpu_usage_percent || 0}
                className="h-2"
              />
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">运行时间</span>
              <span className="text-sm font-medium">
                {diagnostics?.uptime_seconds
                  ? formatDistanceToNow(
                      new Date(Date.now() - diagnostics.uptime_seconds * 1000),
                      { locale: zhCN },
                    )
                  : "未知"}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* 注册表状态 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Database className="w-4 h-4 mr-2" />
              注册表状态
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">备份状态</span>
              <Badge
                variant={
                  diagnostics?.registry_backup_exists
                    ? "default"
                    : "destructive"
                }
              >
                {diagnostics?.registry_backup_exists ? "已备份" : "无备份"}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">备份大小</span>
              <span className="text-sm font-medium">
                {diagnostics?.registry_backup_size
                  ? `${(diagnostics.registry_backup_size / 1024).toFixed(1)} KB`
                  : "0 KB"}
              </span>
            </div>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex justify-between items-center cursor-help">
                    <span className="text-sm">修改位置</span>
                    <Badge variant="outline">HKEY_CLASSES_ROOT</Badge>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>浏览器注册表项修改位置</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </CardContent>
        </Card>

        {/* 操作统计 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              操作统计
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">总拦截数</span>
              <span className="text-sm font-medium">
                {diagnostics?.total_intercepted_urls || 0}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">成功操作</span>
              <span className="text-sm font-medium text-green-600">
                {diagnostics?.successful_operations || 0}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">失败操作</span>
              <span className="text-sm font-medium text-red-600">
                {diagnostics?.failed_operations || 0}
              </span>
            </div>

            {diagnostics && (
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>成功率</span>
                  <span>
                    {diagnostics.total_intercepted_urls > 0
                      ? (
                          (diagnostics.successful_operations /
                            diagnostics.total_intercepted_urls) *
                          100
                        ).toFixed(1)
                      : "100"}
                    %
                  </span>
                </div>
                <Progress
                  value={
                    diagnostics.total_intercepted_urls > 0
                      ? (diagnostics.successful_operations /
                          diagnostics.total_intercepted_urls) *
                        100
                      : 100
                  }
                  className="h-2"
                />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 活跃钩子详情 */}
      {state?.active_hooks && state.active_hooks.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Link className="w-4 h-4 mr-2" />
              活跃钩子详情
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-48">
              <div className="space-y-2">
                {state.active_hooks.map((hook, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded border"
                  >
                    <div className="flex items-center space-x-2">
                      <Settings className="w-4 h-4 text-gray-600" />
                      <span className="font-mono text-sm">{hook}</span>
                    </div>
                    <Badge variant="outline">活跃</Badge>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {/* 系统建议 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <AlertTriangle className="w-4 h-4 mr-2" />
            系统建议
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {!diagnostics?.registry_backup_exists && (
              <div className="p-3 bg-red-50 border border-red-200 rounded">
                <div className="flex items-start space-x-2">
                  <XCircle className="w-4 h-4 text-red-600 mt-0.5" />
                  <div>
                    <div className="text-sm font-medium text-red-800">
                      注册表备份丢失
                    </div>
                    <div className="text-xs text-red-700 mt-1">
                      建议重新启动拦截器以创建备份，确保系统可以正常恢复。
                    </div>
                  </div>
                </div>
              </div>
            )}

            {diagnostics && diagnostics.failed_operations > 0 && (
              <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5" />
                  <div>
                    <div className="text-sm font-medium text-yellow-800">
                      存在失败操作
                    </div>
                    <div className="text-xs text-yellow-700 mt-1">
                      有 {diagnostics.failed_operations}{" "}
                      个操作失败，建议检查日志或重启拦截器。
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!state?.enabled && (
              <div className="p-3 bg-blue-50 border border-blue-200 rounded">
                <div className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5" />
                  <div>
                    <div className="text-sm font-medium text-blue-800">
                      拦截器未启用
                    </div>
                    <div className="text-xs text-blue-700 mt-1">
                      拦截器当前未运行，不会拦截任何浏览器启动请求。
                    </div>
                  </div>
                </div>
              </div>
            )}

            {state?.enabled &&
              diagnostics?.registry_backup_exists &&
              (diagnostics?.failed_operations || 0) === 0 && (
                <div className="p-3 bg-green-50 border border-green-200 rounded">
                  <div className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                    <div>
                      <div className="text-sm font-medium text-green-800">
                        系统运行正常
                      </div>
                      <div className="text-xs text-green-700 mt-1">
                        所有组件正常工作，拦截器已准备就绪。
                      </div>
                    </div>
                  </div>
                </div>
              )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
