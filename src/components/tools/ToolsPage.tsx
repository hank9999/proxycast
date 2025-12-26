import React, { useState } from "react";
import { Globe, Plus, Settings, Activity, Cpu } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface ToolsPageProps {
  onNavigate: (
    page:
      | "provider-pool"
      | "config-management"
      | "api-server"
      | "flow-monitor"
      | "tools"
      | "browser-interceptor"
      | "machine-id"
      | "settings",
  ) => void;
}

interface ToolCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  status?: string;
  disabled?: boolean;
  onClick?: () => void;
}

function ToolCard({
  title,
  description,
  icon,
  status,
  disabled = false,
  onClick,
}: ToolCardProps) {
  return (
    <Card
      className={`cursor-pointer transition-colors hover:bg-muted/50 ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/10 rounded-lg">{icon}</div>
            <div>
              <CardTitle className="text-lg">{title}</CardTitle>
              {status && (
                <Badge
                  variant={status === "运行中" ? "default" : "secondary"}
                  className="mt-1"
                >
                  {status}
                </Badge>
              )}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <CardDescription className="text-sm text-muted-foreground mb-4">
          {description}
        </CardDescription>
        <Button
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={onClick}
          className="w-full"
        >
          {disabled ? "敬请期待" : "打开工具"}
        </Button>
      </CardContent>
    </Card>
  );
}

export function ToolsPage({ onNavigate }: ToolsPageProps) {
  const [interceptorEnabled] = useState(false); // TODO: 从状态管理中获取

  const handleBrowserInterceptorClick = () => {
    onNavigate("browser-interceptor");
  };

  const handleMachineIdClick = () => {
    onNavigate("machine-id");
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">工具箱</h1>
          <p className="text-muted-foreground mt-1">
            ProxyCast 提供的实用工具集合
          </p>
        </div>
        <Badge variant="outline">2 个工具</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <ToolCard
          title="浏览器拦截器"
          description="拦截桌面应用的浏览器启动，支持手动复制 URL 到指纹浏览器"
          icon={<Globe className="w-6 h-6 text-primary" />}
          status={interceptorEnabled ? "运行中" : "已停止"}
          onClick={handleBrowserInterceptorClick}
        />

        <ToolCard
          title="机器码管理工具"
          description="查看、修改和管理系统机器码，支持跨平台操作和备份恢复"
          icon={<Cpu className="w-6 h-6 text-primary" />}
          onClick={handleMachineIdClick}
        />

        {/* 未来可以添加更多工具 */}
        <ToolCard
          title="网络监控工具"
          description="监控和分析网络请求，提供详细的流量分析"
          icon={<Activity className="w-6 h-6 text-muted-foreground" />}
          disabled
        />

        <ToolCard
          title="配置同步工具"
          description="在多个设备间同步 ProxyCast 配置"
          icon={<Settings className="w-6 h-6 text-muted-foreground" />}
          disabled
        />

        <ToolCard
          title="更多工具"
          description="更多实用工具正在开发中..."
          icon={<Plus className="w-6 h-6 text-muted-foreground" />}
          disabled
        />
      </div>

      <div className="mt-8 p-6 bg-muted/30 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">关于工具箱</h3>
        <p className="text-sm text-muted-foreground">
          工具箱是 ProxyCast
          的扩展功能模块，提供各种实用工具来增强您的使用体验。
          每个工具都经过精心设计，旨在解决特定的使用场景和需求。
        </p>
      </div>
    </div>
  );
}
