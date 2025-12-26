import {
  Settings,
  Globe,
  Database,
  FileCode,
  Activity,
  Wrench,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Page =
  | "provider-pool"
  | "config-management"
  | "api-server"
  | "flow-monitor"
  | "tools"
  | "browser-interceptor"
  | "machine-id"
  | "settings";

interface SidebarProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

const navItems = [
  { id: "api-server" as Page, label: "API Server", icon: Globe },
  { id: "provider-pool" as Page, label: "凭证池", icon: Database },
  { id: "config-management" as Page, label: "配置管理", icon: FileCode },
  { id: "flow-monitor" as Page, label: "Flow Monitor", icon: Activity },
  { id: "tools" as Page, label: "工具", icon: Wrench },
  { id: "settings" as Page, label: "设置", icon: Settings },
];

export function Sidebar({ currentPage, onNavigate }: SidebarProps) {
  return (
    <div className="w-56 border-r bg-card p-4">
      <div className="mb-8">
        <h1 className="text-xl font-bold">ProxyCast</h1>
        <p className="text-xs text-muted-foreground">AI API Proxy</p>
      </div>
      <nav className="space-y-1">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={cn(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
              currentPage === item.id
                ? "bg-primary text-primary-foreground"
                : "hover:bg-muted",
            )}
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </button>
        ))}
      </nav>
    </div>
  );
}
