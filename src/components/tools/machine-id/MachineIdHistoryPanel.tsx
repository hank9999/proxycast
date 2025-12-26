import { useState, useEffect, useCallback } from "react";
import { History, Clock, FileText, AlertTriangle } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MachineIdHistory } from "@/lib/api/machineId";
import { machineIdApi } from "@/lib/api/machineId";
import { useToast } from "@/hooks/use-toast";

interface MachineIdHistoryPanelProps {
  _onRefresh?: () => void;
}

export function MachineIdHistoryPanel({
  _onRefresh,
}: MachineIdHistoryPanelProps) {
  const { toast } = useToast();
  const [history, setHistory] = useState<MachineIdHistory[]>([]);
  const [loading, setLoading] = useState(true);

  const loadHistory = useCallback(async () => {
    try {
      setLoading(true);
      const historyData = await machineIdApi.getMachineIdHistory();
      setHistory(historyData);
    } catch (error) {
      console.error("加载历史记录失败:", error);
      toast({
        variant: "destructive",
        title: "加载失败",
        description: "无法加载机器码历史记录",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return timestamp;
    }
  };

  const getPlatformIcon = (platform: string) => {
    switch (platform.toLowerCase()) {
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

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <History className="w-5 h-5 text-blue-500" />
            <span>操作历史</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <History className="w-8 h-8 mx-auto mb-2 animate-spin text-blue-500" />
              <p className="text-muted-foreground">加载历史记录中...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <History className="w-5 h-5 text-blue-500" />
                <span>操作历史</span>
              </CardTitle>
              <CardDescription>查看机器码的历史修改记录</CardDescription>
            </div>
            <Button
              variant="outline"
              onClick={loadHistory}
              className="space-x-2"
            >
              <History className="w-4 h-4" />
              <span>刷新</span>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-24 h-24 mx-auto mb-4 bg-muted rounded-full flex items-center justify-center">
                <History className="w-12 h-12 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">暂无历史记录</h3>
              <p className="text-muted-foreground mb-4">
                当前还没有机器码操作历史记录
              </p>
              <div className="text-sm text-muted-foreground bg-muted/50 p-4 rounded-lg">
                <p>💡 提示：历史记录功能正在开发中</p>
                <p>完成后将记录所有机器码修改操作，包括：</p>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>机器码修改时间</li>
                  <li>修改前后的值</li>
                  <li>操作平台信息</li>
                  <li>备份文件路径</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {history.map((record, index) => (
                <Card
                  key={record.id || index}
                  className="border-l-4 border-l-blue-500"
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
                          <span className="text-lg">
                            {getPlatformIcon(record.platform)}
                          </span>
                        </div>
                        <div>
                          <h4 className="font-semibold">机器码操作</h4>
                          <div className="flex items-center space-x-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              {record.platform}
                            </Badge>
                            <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                              <Clock className="w-3 h-3" />
                              <span>{formatTimestamp(record.timestamp)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0 space-y-3">
                    <div className="grid gap-3">
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">
                          机器码
                        </p>
                        <div className="font-mono text-sm bg-muted/50 p-2 rounded border">
                          {record.machine_id}
                        </div>
                      </div>

                      {record.backup_path && (
                        <div>
                          <p className="text-sm text-muted-foreground mb-1">
                            备份路径
                          </p>
                          <div className="flex items-center space-x-2 text-sm bg-muted/50 p-2 rounded border">
                            <FileText className="w-4 h-4 text-muted-foreground" />
                            <span className="font-mono">
                              {record.backup_path}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 说明信息 */}
      <Card className="border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/20">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-amber-800 dark:text-amber-300">
            <AlertTriangle className="w-5 h-5" />
            <span>关于历史记录</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="text-amber-700 dark:text-amber-400 space-y-2">
          <p>历史记录功能目前处于开发阶段，暂时返回空记录。</p>
          <p className="text-sm">完整实现后将包含：</p>
          <ul className="list-disc list-inside text-sm space-y-1 ml-4">
            <li>所有机器码修改操作的时间记录</li>
            <li>修改前后的机器码值对比</li>
            <li>操作系统和平台信息</li>
            <li>相关备份文件的路径信息</li>
            <li>操作结果和状态信息</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
