import { useState, useEffect } from "react";
import { ArrowLeft, Cpu, Settings, History, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MachineIdInfoPanel } from "./MachineIdInfoPanel";
import { MachineIdManagePanel } from "./MachineIdManagePanel";
import { MachineIdHistoryPanel } from "./MachineIdHistoryPanel";
import { MachineIdSystemPanel } from "./MachineIdSystemPanel";
import { MachineIdInfo, SystemInfo, AdminStatus } from "@/lib/api/machineId";
import { machineIdApi } from "@/lib/api/machineId";

interface MachineIdToolProps {
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

export function MachineIdTool({ onNavigate }: MachineIdToolProps) {
  const [machineIdInfo, setMachineIdInfo] = useState<MachineIdInfo | null>(
    null,
  );
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [adminStatus, setAdminStatus] = useState<AdminStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    loadAllInfo();
  }, [refreshKey]);

  const loadAllInfo = async () => {
    try {
      setLoading(true);
      const [machineId, system, admin] = await Promise.all([
        machineIdApi.getCurrentMachineId(),
        machineIdApi.getSystemInfo(),
        machineIdApi.checkAdminPrivileges(),
      ]);

      setMachineIdInfo(machineId);
      setSystemInfo(system);
      setAdminStatus(admin);
    } catch (error) {
      console.error("加载机器码信息失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Cpu className="w-16 h-16 mx-auto mb-4 animate-spin text-blue-500" />
          <p className="text-gray-600">加载中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 页面头部 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Button
            variant="ghost"
            onClick={() => onNavigate("tools")}
            className="p-2"
          >
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <div className="flex items-center space-x-3">
            <Cpu className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-3xl font-bold">机器码管理工具</h1>
              <p className="text-gray-600 text-sm mt-1">
                查看、修改和管理系统机器码，支持跨平台操作
              </p>
            </div>
          </div>
        </div>

        {/* 刷新按钮 */}
        <Button variant="outline" onClick={handleRefresh} className="space-x-2">
          <Settings className="w-4 h-4" />
          <span>刷新信息</span>
        </Button>
      </div>

      {/* 主要功能标签页 */}
      <Tabs defaultValue="info" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="info" className="space-x-2">
            <Info className="w-4 h-4" />
            <span>机器码信息</span>
          </TabsTrigger>
          <TabsTrigger value="manage" className="space-x-2">
            <Settings className="w-4 h-4" />
            <span>管理操作</span>
          </TabsTrigger>
          <TabsTrigger value="history" className="space-x-2">
            <History className="w-4 h-4" />
            <span>操作历史</span>
          </TabsTrigger>
          <TabsTrigger value="system" className="space-x-2">
            <Cpu className="w-4 h-4" />
            <span>系统信息</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="info" className="space-y-6">
          <MachineIdInfoPanel
            machineIdInfo={machineIdInfo}
            adminStatus={adminStatus}
            onRefresh={handleRefresh}
          />
        </TabsContent>

        <TabsContent value="manage" className="space-y-6">
          <MachineIdManagePanel
            machineIdInfo={machineIdInfo}
            adminStatus={adminStatus}
            onRefresh={handleRefresh}
          />
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <MachineIdHistoryPanel _onRefresh={handleRefresh} />
        </TabsContent>

        <TabsContent value="system" className="space-y-6">
          <MachineIdSystemPanel
            systemInfo={systemInfo}
            adminStatus={adminStatus}
            onRefresh={handleRefresh}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
