import { useState, useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { SettingsPage } from "./components/settings";
import { ApiServerPage } from "./components/api-server/ApiServerPage";
import { ProviderPoolPage } from "./components/provider-pool";
import { ConfigManagementPage } from "./components/config/ConfigManagementPage";
import { FlowMonitorPage } from "./pages";
import { ToolsPage } from "./components/tools/ToolsPage";
import { BrowserInterceptorTool } from "./components/tools/browser-interceptor/BrowserInterceptorTool";
import { MachineIdTool } from "./components/tools/machine-id/MachineIdTool";
import { Toaster } from "./components/ui/sonner";
import { flowEventManager } from "./lib/flowEventManager";

type Page =
  | "provider-pool"
  | "config-management"
  | "api-server"
  | "flow-monitor"
  | "tools"
  | "browser-interceptor"
  | "machine-id"
  | "settings";

function App() {
  const [currentPage, setCurrentPage] = useState<Page>("api-server");

  // 在应用启动时初始化 Flow 事件订阅
  useEffect(() => {
    flowEventManager.subscribe();
    // 应用卸载时不取消订阅，因为这是全局订阅
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case "provider-pool":
        return <ProviderPoolPage />;
      case "config-management":
        return <ConfigManagementPage />;
      case "api-server":
        return <ApiServerPage />;
      case "flow-monitor":
        return <FlowMonitorPage />;
      case "tools":
        return <ToolsPage onNavigate={setCurrentPage} />;
      case "browser-interceptor":
        return <BrowserInterceptorTool onNavigate={setCurrentPage} />;
      case "machine-id":
        return <MachineIdTool onNavigate={setCurrentPage} />;
      case "settings":
        return <SettingsPage />;
      default:
        return <ApiServerPage />;
    }
  };

  return (
    <div className="flex h-screen bg-background">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
      <main className="flex-1 overflow-auto p-6">{renderPage()}</main>
      <Toaster />
    </div>
  );
}

export default App;
