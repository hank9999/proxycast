import { useState } from "react";
import {
  Copy,
  Shield,
  ShieldCheck,
  AlertTriangle,
  Monitor,
  HardDrive,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { MachineIdInfo, AdminStatus } from "@/lib/api/machineId";
import { machineIdApi, machineIdUtils } from "@/lib/api/machineId";

interface MachineIdInfoPanelProps {
  machineIdInfo: MachineIdInfo | null;
  adminStatus: AdminStatus | null;
  onRefresh: () => void;
}

export function MachineIdInfoPanel({
  machineIdInfo,
  adminStatus,
  onRefresh,
}: MachineIdInfoPanelProps) {
  const { toast } = useToast();
  const [copying, setCopying] = useState(false);

  const handleCopyMachineId = async () => {
    if (!machineIdInfo) return;

    try {
      setCopying(true);
      await machineIdApi.copyMachineIdToClipboard(machineIdInfo.current_id);
      toast({
        title: "复制成功",
        description: "机器码已复制到剪贴板",
      });
    } catch (error) {
      console.error("复制失败:", error);
      toast({
        variant: "destructive",
        title: "复制失败",
        description: "无法复制机器码到剪贴板",
      });
    } finally {
      setCopying(false);
    }
  };

  if (!machineIdInfo || !adminStatus) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span>无法获取机器码信息</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            无法加载机器码信息，请检查系统状态或权限设置。
          </p>
          <Button onClick={onRefresh} className="mt-4">
            重新加载
          </Button>
        </CardContent>
      </Card>
    );
  }

  const getStatusBadge = () => {
    if (!machineIdInfo.can_modify) {
      return (
        <Badge variant="secondary" className="space-x-1">
          <Shield className="w-3 h-3" />
          <span>只读</span>
        </Badge>
      );
    }

    if (machineIdInfo.requires_admin && !adminStatus.is_admin) {
      return (
        <Badge variant="destructive" className="space-x-1">
          <AlertTriangle className="w-3 h-3" />
          <span>需要管理员权限</span>
        </Badge>
      );
    }

    return (
      <Badge variant="default" className="space-x-1">
        <ShieldCheck className="w-3 h-3" />
        <span>可修改</span>
      </Badge>
    );
  };

  const getFormatBadge = () => {
    const color =
      machineIdInfo.format_type === "uuid"
        ? "default"
        : machineIdInfo.format_type === "hex32"
          ? "secondary"
          : "outline";

    return (
      <Badge variant={color}>
        {machineIdUtils.getFormatDisplayName(machineIdInfo.format_type)}
      </Badge>
    );
  };

  return (
    <div className="space-y-6">
      {/* 主要信息卡片 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <HardDrive className="w-5 h-5 text-blue-500" />
                <span>当前机器码</span>
              </CardTitle>
              <CardDescription className="mt-2">
                系统唯一标识符，用于设备识别和授权验证
              </CardDescription>
            </div>
            {getStatusBadge()}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">机器码</p>
              <p className="font-mono text-lg font-semibold">
                {machineIdInfo.current_id}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              {getFormatBadge()}
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopyMachineId}
                disabled={copying}
                className="space-x-1"
              >
                <Copy className="w-4 h-4" />
                <span>{copying ? "复制中..." : "复制"}</span>
              </Button>
            </div>
          </div>

          {machineIdInfo.original_id && (
            <>
              <Separator />
              <div className="p-4 bg-amber-50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-800">
                <div className="flex items-center space-x-2 mb-2">
                  <AlertTriangle className="w-4 h-4 text-amber-600" />
                  <p className="text-sm font-medium text-amber-800 dark:text-amber-300">
                    检测到机器码覆盖
                  </p>
                </div>
                <p className="text-sm text-amber-700 dark:text-amber-400 mb-2">
                  原始机器码：
                </p>
                <p className="font-mono text-sm text-amber-900 dark:text-amber-200 bg-amber-100 dark:bg-amber-900/30 p-2 rounded">
                  {machineIdInfo.original_id}
                </p>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* 系统信息卡片 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Monitor className="w-5 h-5 text-green-500" />
            <span>系统状态</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">操作系统</p>
              <p className="font-semibold">
                {machineIdUtils.getPlatformDisplayName(machineIdInfo.platform)}
              </p>
            </div>

            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">管理员权限</p>
              <div className="flex items-center space-x-2">
                {adminStatus.is_admin ? (
                  <Badge variant="default" className="space-x-1">
                    <ShieldCheck className="w-3 h-3" />
                    <span>已获取</span>
                  </Badge>
                ) : (
                  <Badge variant="secondary" className="space-x-1">
                    <Shield className="w-3 h-3" />
                    <span>未获取</span>
                  </Badge>
                )}
              </div>
            </div>

            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">备份状态</p>
              <div className="flex items-center space-x-2">
                {machineIdInfo.backup_exists ? (
                  <Badge variant="default">已备份</Badge>
                ) : (
                  <Badge variant="outline">未备份</Badge>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 权限提醒 */}
      {machineIdInfo.requires_admin && !adminStatus.is_admin && (
        <Card className="border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/20">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-amber-800 dark:text-amber-300">
              <AlertTriangle className="w-5 h-5" />
              <span>权限提醒</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-amber-700 dark:text-amber-400">
            <p className="mb-2">
              在当前平台（{machineIdInfo.platform}）上修改机器码需要管理员权限。
            </p>
            {adminStatus.elevation_method && (
              <p className="text-sm">
                提升权限方法：{adminStatus.elevation_method}
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
