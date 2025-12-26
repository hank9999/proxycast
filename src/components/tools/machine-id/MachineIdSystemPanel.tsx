import {
  Cpu,
  Shield,
  Server,
  AlertTriangle,
  CheckCircle,
  Info,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { SystemInfo, AdminStatus } from "@/lib/api/machineId";
import { machineIdUtils } from "@/lib/api/machineId";

interface MachineIdSystemPanelProps {
  systemInfo: SystemInfo | null;
  adminStatus: AdminStatus | null;
  onRefresh: () => void;
}

export function MachineIdSystemPanel({
  systemInfo,
  adminStatus,
}: MachineIdSystemPanelProps) {
  if (!systemInfo || !adminStatus) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span>无法获取系统信息</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            无法加载系统信息，请检查系统状态。
          </p>
        </CardContent>
      </Card>
    );
  }

  const getPlatformIcon = (os: string) => {
    switch (os.toLowerCase()) {
      case "windows":
        return "🪟";
      case "macos":
        return "🍎";
      case "linux":
        return "🐧";
      default:
        return "💻";
    }
  };

  const getArchIcon = (arch: string) => {
    switch (arch.toLowerCase()) {
      case "x86_64":
      case "amd64":
        return "🖥️";
      case "aarch64":
      case "arm64":
        return "📱";
      case "x86":
      case "i386":
        return "🖧";
      default:
        return "🔧";
    }
  };

  const getSupportBadge = (canRead: boolean, canWrite: boolean) => {
    if (canRead && canWrite) {
      return (
        <Badge variant="default" className="space-x-1">
          <CheckCircle className="w-3 h-3" />
          <span>完全支持</span>
        </Badge>
      );
    } else if (canRead) {
      return (
        <Badge variant="secondary" className="space-x-1">
          <Shield className="w-3 h-3" />
          <span>只读支持</span>
        </Badge>
      );
    } else {
      return (
        <Badge variant="outline" className="space-x-1">
          <AlertTriangle className="w-3 h-3" />
          <span>不支持</span>
        </Badge>
      );
    }
  };

  return (
    <div className="space-y-6">
      {/* 系统基本信息 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Server className="w-5 h-5 text-blue-500" />
            <span>系统信息</span>
          </CardTitle>
          <CardDescription>当前系统的基本信息和配置</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">
                  {getPlatformIcon(systemInfo.os)}
                </span>
                <div>
                  <p className="text-sm text-muted-foreground">操作系统</p>
                  <p className="font-semibold">
                    {machineIdUtils.getPlatformDisplayName(systemInfo.os)}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    系列: {systemInfo.family}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <span className="text-2xl">{getArchIcon(systemInfo.arch)}</span>
                <div>
                  <p className="text-sm text-muted-foreground">系统架构</p>
                  <p className="font-semibold">{systemInfo.arch}</p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Shield className="w-6 h-6 text-green-500" />
                <div>
                  <p className="text-sm text-muted-foreground">权限状态</p>
                  <div className="flex items-center space-x-2">
                    {adminStatus.is_admin ? (
                      <Badge variant="default" className="space-x-1">
                        <CheckCircle className="w-3 h-3" />
                        <span>管理员权限</span>
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="space-x-1">
                        <Shield className="w-3 h-3" />
                        <span>普通用户</span>
                      </Badge>
                    )}
                  </div>
                  {adminStatus.elevation_method && (
                    <p className="text-xs text-muted-foreground mt-1">
                      提升方法：{adminStatus.elevation_method}
                    </p>
                  )}
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <Cpu className="w-6 h-6 text-purple-500" />
                <div>
                  <p className="text-sm text-muted-foreground">机器码支持</p>
                  {getSupportBadge(
                    systemInfo.machine_id_support.can_read,
                    systemInfo.machine_id_support.can_write,
                  )}
                  {systemInfo.requires_admin && (
                    <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                      ⚠️ 修改需要管理员权限
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 平台支持详情 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-purple-500" />
            <span>平台支持详情</span>
          </CardTitle>
          <CardDescription>当前平台对机器码操作的支持情况</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div>
                <p className="text-sm text-muted-foreground">支持格式</p>
                <p className="font-semibold">
                  {systemInfo.machine_id_support.format}
                </p>
              </div>

              <div>
                <p className="text-sm text-muted-foreground">实现方法</p>
                <p className="text-sm bg-muted/50 p-2 rounded">
                  {systemInfo.machine_id_support.method}
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <p className="text-sm text-muted-foreground">操作权限</p>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    {systemInfo.machine_id_support.can_read ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    )}
                    <span className="text-sm">
                      读取机器码{" "}
                      {systemInfo.machine_id_support.can_read ? "✓" : "✗"}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    {systemInfo.machine_id_support.can_write ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    )}
                    <span className="text-sm">
                      修改机器码{" "}
                      {systemInfo.machine_id_support.can_write ? "✓" : "✗"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {systemInfo.machine_id_support.limitations.length > 0 && (
            <>
              <Separator />
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="w-4 h-4 text-amber-500" />
                  <p className="text-sm font-medium">平台限制</p>
                </div>
                <div className="space-y-2">
                  {systemInfo.machine_id_support.limitations.map(
                    (limitation, index) => (
                      <div
                        key={index}
                        className="flex items-start space-x-2 text-sm text-muted-foreground"
                      >
                        <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                        <span>{limitation}</span>
                      </div>
                    ),
                  )}
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* 平台特定信息 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Info className="w-5 h-5 text-indigo-500" />
            <span>平台特定说明</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {systemInfo.os.toLowerCase() === "windows" && (
            <div className="space-y-3">
              <h4 className="font-medium">Windows 平台说明</h4>
              <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
                <li>
                  机器码存储在注册表中：
                  <code className="bg-muted px-1 rounded">
                    HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Cryptography\MachineGuid
                  </code>
                </li>
                <li>修改机器码需要管理员权限</li>
                <li>某些应用程序可能需要重启才能识别新的机器码</li>
                <li>支持标准 UUID 格式</li>
              </ul>
            </div>
          )}

          {systemInfo.os.toLowerCase() === "macos" && (
            <div className="space-y-3">
              <h4 className="font-medium">macOS 平台说明</h4>
              <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
                <li>使用应用层覆盖机制，不修改系统原始 UUID</li>
                <li>
                  原始机器码通过{" "}
                  <code className="bg-muted px-1 rounded">ioreg</code> 命令获取
                </li>
                <li>覆盖文件存储在用户数据目录</li>
                <li>不需要管理员权限，但只影响使用覆盖的应用</li>
                <li>支持清除覆盖恢复原始状态</li>
              </ul>
            </div>
          )}

          {systemInfo.os.toLowerCase() === "linux" && (
            <div className="space-y-3">
              <h4 className="font-medium">Linux 平台说明</h4>
              <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
                <li>
                  机器码存储在{" "}
                  <code className="bg-muted px-1 rounded">/etc/machine-id</code>{" "}
                  文件中
                </li>
                <li>修改需要 root 权限</li>
                <li>使用 32 位十六进制格式</li>
                <li>某些系统服务可能需要重启</li>
                <li>修改可能影响系统服务的正常运行</li>
              </ul>
            </div>
          )}

          {!["windows", "macos", "linux"].includes(
            systemInfo.os.toLowerCase(),
          ) && (
            <div className="text-center py-6">
              <AlertTriangle className="w-12 h-12 mx-auto mb-3 text-muted-foreground" />
              <h4 className="font-medium mb-2">不支持的平台</h4>
              <p className="text-sm text-muted-foreground">
                当前平台（{systemInfo.os}）暂不支持机器码管理功能
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
