import React, { useState, useEffect, useCallback } from "react";
import {
  History,
  Search,
  Download,
  Trash2,
  Copy,
  ExternalLink,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { formatDistanceToNow, format } from "date-fns";
import { zhCN } from "date-fns/locale";
import { InterceptedUrl } from "@/lib/api/browserInterceptor";
import * as browserInterceptorApi from "@/lib/api/browserInterceptor";

export function UrlHistoryPanel() {
  const [history, setHistory] = useState<InterceptedUrl[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<InterceptedUrl[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProcess, setSelectedProcess] = useState<string>("all");
  const [uniqueProcesses, setUniqueProcesses] = useState<string[]>([]);
  const [exporting, setExporting] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const historyData = await browserInterceptorApi.getInterceptorHistory();
      setHistory(historyData);

      // 提取所有唯一的进程名
      const processes = Array.from(
        new Set(historyData.map((item) => item.source_process)),
      ).sort();
      setUniqueProcesses(processes);
    } catch (error) {
      console.error("加载历史记录失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const filterHistory = useCallback(() => {
    let filtered = [...history];

    // 按搜索关键词过滤
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (item) =>
          item.url.toLowerCase().includes(query) ||
          item.source_process.toLowerCase().includes(query),
      );
    }

    // 按进程过滤
    if (selectedProcess !== "all") {
      filtered = filtered.filter(
        (item) => item.source_process === selectedProcess,
      );
    }

    setFilteredHistory(filtered);
  }, [history, searchQuery, selectedProcess]);

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    filterHistory();
  }, [filterHistory]);

  const handleExportHistory = async (format: "json" | "csv") => {
    setExporting(true);
    try {
      // 创建下载链接
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const filename = `proxycast-interceptor-history-${timestamp}.${format}`;

      let data: string;
      let mimeType: string;

      if (format === "json") {
        data = JSON.stringify(filteredHistory, null, 2);
        mimeType = "application/json";
      } else {
        // CSV 格式
        const headers = [
          "ID",
          "URL",
          "Source Process",
          "Timestamp",
          "Copied",
          "Opened in Browser",
          "Dismissed",
        ];
        const csvRows = [
          headers.join(","),
          ...filteredHistory.map((item) =>
            [
              `"${item.id}"`,
              `"${item.url.replace(/"/g, '""')}"`,
              `"${item.source_process}"`,
              `"${item.timestamp}"`,
              item.copied ? "true" : "false",
              item.opened_in_browser ? "true" : "false",
              item.dismissed ? "true" : "false",
            ].join(","),
          ),
        ];
        data = csvRows.join("\n");
        mimeType = "text/csv";
      }

      const blob = new Blob([data], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("导出历史记录失败:", error);
    } finally {
      setExporting(false);
    }
  };

  const handleClearHistory = async () => {
    // TODO: 实现清空历史记录功能
    console.log("清空历史记录（功能待实现）");
  };

  const getSourceProcessIcon = (processName: string) => {
    if (processName.toLowerCase().includes("kiro")) {
      return "🤖";
    } else if (processName.toLowerCase().includes("cursor")) {
      return "💻";
    } else if (processName.toLowerCase().includes("code")) {
      return "📝";
    }
    return "🔗";
  };

  const getStatusBadges = (item: InterceptedUrl) => {
    const badges = [];

    if (item.copied) {
      badges.push(
        <Badge key="copied" variant="secondary" className="text-xs">
          已复制
        </Badge>,
      );
    }

    if (item.opened_in_browser) {
      badges.push(
        <Badge key="opened" variant="secondary" className="text-xs">
          已打开
        </Badge>,
      );
    }

    if (item.dismissed) {
      badges.push(
        <Badge key="dismissed" variant="outline" className="text-xs">
          已忽略
        </Badge>,
      );
    }

    return badges;
  };

  const copyUrlToClipboard = async (url: string) => {
    try {
      await navigator.clipboard.writeText(url);
    } catch (error) {
      console.error("复制URL失败:", error);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-8">
          <div className="flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mr-2" />
            <span>加载历史记录中...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 搜索和过滤 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <History className="w-5 h-5" />
              <span>历史记录</span>
              <Badge variant="outline">
                {filteredHistory.length} / {history.length}
              </Badge>
            </CardTitle>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm" onClick={loadHistory}>
                <RefreshCw className="w-4 h-4 mr-1" />
                刷新
              </Button>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" disabled={exporting}>
                    <Download className="w-4 h-4 mr-1" />
                    导出
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => handleExportHistory("json")}>
                    导出为 JSON
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleExportHistory("csv")}>
                    导出为 CSV
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Trash2 className="w-4 h-4 mr-1" />
                    清空
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>清空历史记录</DialogTitle>
                    <DialogDescription>
                      此操作将永久删除所有历史记录，此操作不可撤销。
                    </DialogDescription>
                  </DialogHeader>
                  <DialogFooter>
                    <Button variant="outline">取消</Button>
                    <Button variant="destructive" onClick={handleClearHistory}>
                      确定清空
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <Input
                  placeholder="搜索 URL 或进程名..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Select value={selectedProcess} onValueChange={setSelectedProcess}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="筛选进程" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">所有进程</SelectItem>
                {uniqueProcesses.map((process) => (
                  <SelectItem key={process} value={process}>
                    <div className="flex items-center space-x-2">
                      <span>{getSourceProcessIcon(process)}</span>
                      <span>{process}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* 历史记录列表 */}
      <Card>
        <CardContent className="p-0">
          {filteredHistory.length === 0 ? (
            <div className="text-center py-12">
              <History className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium text-gray-700 mb-2">
                {searchQuery || selectedProcess !== "all"
                  ? "没有找到匹配的记录"
                  : "暂无历史记录"}
              </h3>
              <p className="text-gray-500 max-w-md mx-auto">
                {searchQuery || selectedProcess !== "all"
                  ? "请尝试调整搜索条件或筛选选项"
                  : "当拦截器开始工作时，拦截的URL将会显示在这里"}
              </p>
            </div>
          ) : (
            <ScrollArea className="h-96">
              <div className="p-4 space-y-3">
                {filteredHistory.map((item, index) => (
                  <div
                    key={item.id}
                    className="border rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    {/* 头部信息 */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-gray-500">
                          #{index + 1}
                        </span>
                        <span className="text-lg">
                          {getSourceProcessIcon(item.source_process)}
                        </span>
                        <Badge variant="outline" className="text-xs">
                          {item.source_process}
                        </Badge>
                        <div className="flex space-x-1">
                          {getStatusBadges(item)}
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="text-xs text-gray-500">
                          {format(
                            new Date(item.timestamp),
                            "yyyy-MM-dd HH:mm:ss",
                          )}
                        </div>
                        <div className="text-xs text-gray-400">
                          {formatDistanceToNow(new Date(item.timestamp), {
                            addSuffix: true,
                            locale: zhCN,
                          })}
                        </div>
                      </div>
                    </div>

                    {/* URL 显示 */}
                    <div className="bg-gray-100 rounded-lg p-3 mb-2 font-mono text-sm break-all">
                      {item.url}
                    </div>

                    {/* 操作按钮 */}
                    <div className="flex space-x-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => copyUrlToClipboard(item.url)}
                      >
                        <Copy className="w-3 h-3 mr-1" />
                        复制
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => window.open(item.url, "_blank")}
                      >
                        <ExternalLink className="w-3 h-3 mr-1" />
                        在浏览器中打开
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      {/* 统计信息 */}
      {history.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">统计概览</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {history.length}
                </div>
                <div className="text-sm text-gray-600">总拦截次数</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {history.filter((item) => item.copied).length}
                </div>
                <div className="text-sm text-gray-600">已复制</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {history.filter((item) => item.opened_in_browser).length}
                </div>
                <div className="text-sm text-gray-600">已打开</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-600">
                  {uniqueProcesses.length}
                </div>
                <div className="text-sm text-gray-600">涉及应用</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
