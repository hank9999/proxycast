import { useState, useEffect } from "react";
import {
  Copy,
  ExternalLink,
  X,
  Globe,
  Clock,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatDistanceToNow } from "date-fns";
import { zhCN } from "date-fns/locale";
import { InterceptedUrl } from "@/lib/api/browserInterceptor";
import * as browserInterceptorApi from "@/lib/api/browserInterceptor";

interface InterceptedUrlsPanelProps {
  onStateChange: () => void;
}

export function InterceptedUrlsPanel({
  onStateChange,
}: InterceptedUrlsPanelProps) {
  const [interceptedUrls, setInterceptedUrls] = useState<InterceptedUrl[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [operatingUrls, setOperatingUrls] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadInterceptedUrls();

    // 每10秒自动刷新
    const interval = setInterval(loadInterceptedUrls, 10000);

    return () => clearInterval(interval);
  }, []);

  const loadInterceptedUrls = async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true);
    try {
      const urls = await browserInterceptorApi.getInterceptedUrls();
      setInterceptedUrls(urls);
    } catch (error) {
      console.error("加载拦截URL失败:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    loadInterceptedUrls(true);
  };

  const setUrlOperating = (urlId: string, operating: boolean) => {
    setOperatingUrls((prev) => {
      const newSet = new Set(prev);
      if (operating) {
        newSet.add(urlId);
      } else {
        newSet.delete(urlId);
      }
      return newSet;
    });
  };

  const handleCopyUrl = async (urlData: InterceptedUrl) => {
    setUrlOperating(urlData.id, true);
    try {
      await browserInterceptorApi.copyInterceptedUrlToClipboard(urlData.id);
      // 复制成功后重新加载数据以更新状态
      await loadInterceptedUrls();
      onStateChange();
    } catch (error) {
      console.error("复制URL失败:", error);
    } finally {
      setUrlOperating(urlData.id, false);
    }
  };

  const handleOpenInBrowser = async (urlData: InterceptedUrl) => {
    setUrlOperating(urlData.id, true);
    try {
      await browserInterceptorApi.openUrlInFingerprintBrowser(urlData.id);
      await loadInterceptedUrls();
      onStateChange();
    } catch (error) {
      console.error("在指纹浏览器中打开URL失败:", error);
    } finally {
      setUrlOperating(urlData.id, false);
    }
  };

  const handleDismissUrl = async (urlData: InterceptedUrl) => {
    setUrlOperating(urlData.id, true);
    try {
      await browserInterceptorApi.dismissInterceptedUrl(urlData.id);
      await loadInterceptedUrls();
      onStateChange();
    } catch (error) {
      console.error("忽略URL失败:", error);
    } finally {
      setUrlOperating(urlData.id, false);
    }
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

  const truncateUrl = (url: string, maxLength = 60) => {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + "...";
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Globe className="w-5 h-5" />
            <span>当前拦截的 URL</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="ml-2 text-gray-600">加载中...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Globe className="w-5 h-5" />
            <span>当前拦截的 URL</span>
            <Badge variant="outline">{interceptedUrls.length}</Badge>
          </CardTitle>
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
      </CardHeader>
      <CardContent>
        {interceptedUrls.length === 0 ? (
          <div className="text-center py-12">
            <Globe className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-700 mb-2">
              暂无拦截的 URL
            </h3>
            <p className="text-gray-500 max-w-md mx-auto">
              当目标应用（如 Kiro、Cursor、VSCode）尝试打开浏览器时，URL
              将显示在这里。 请确保拦截器已启用。
            </p>
          </div>
        ) : (
          <ScrollArea className="h-96">
            <div className="space-y-4">
              {interceptedUrls.map((urlData) => (
                <div
                  key={urlData.id}
                  className="border rounded-lg p-4 hover:bg-gray-50 transition-colors"
                >
                  {/* URL 头部信息 */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">
                        {getSourceProcessIcon(urlData.source_process)}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {urlData.source_process}
                      </Badge>
                      {urlData.copied && (
                        <Badge variant="secondary" className="text-xs">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          已复制
                        </Badge>
                      )}
                      {urlData.opened_in_browser && (
                        <Badge variant="secondary" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          已打开
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center text-xs text-gray-500">
                      <Clock className="w-3 h-3 mr-1" />
                      {formatDistanceToNow(new Date(urlData.timestamp), {
                        addSuffix: true,
                        locale: zhCN,
                      })}
                    </div>
                  </div>

                  {/* URL 显示 */}
                  <div className="bg-gray-100 rounded-lg p-3 mb-3 font-mono text-sm break-all">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="cursor-help">
                            {truncateUrl(urlData.url)}
                          </div>
                        </TooltipTrigger>
                        <TooltipContent
                          side="bottom"
                          align="start"
                          className="max-w-lg"
                        >
                          <p className="break-all">{urlData.url}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>

                  {/* 操作按钮 */}
                  <div className="flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      onClick={() => handleCopyUrl(urlData)}
                      disabled={operatingUrls.has(urlData.id)}
                      className="flex items-center"
                    >
                      {operatingUrls.has(urlData.id) ? (
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-1" />
                      ) : (
                        <Copy className="w-4 h-4 mr-1" />
                      )}
                      {urlData.copied ? "重新复制" : "复制 URL"}
                    </Button>

                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleOpenInBrowser(urlData)}
                      disabled={operatingUrls.has(urlData.id)}
                    >
                      {operatingUrls.has(urlData.id) ? (
                        <div className="w-4 h-4 border-2 border-gray-500 border-t-transparent rounded-full animate-spin mr-1" />
                      ) : (
                        <ExternalLink className="w-4 h-4 mr-1" />
                      )}
                      在指纹浏览器中打开
                    </Button>

                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDismissUrl(urlData)}
                      disabled={operatingUrls.has(urlData.id)}
                      className="text-red-600 hover:text-red-700 hover:bg-red-50"
                    >
                      {operatingUrls.has(urlData.id) ? (
                        <div className="w-4 h-4 border-2 border-red-500 border-t-transparent rounded-full animate-spin mr-1" />
                      ) : (
                        <X className="w-4 h-4 mr-1" />
                      )}
                      忽略
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}

        {/* 使用提示 */}
        {interceptedUrls.length > 0 && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-700">
                <p className="font-medium mb-1">使用建议：</p>
                <ul className="text-xs space-y-1">
                  <li>• 点击"复制 URL"将链接复制到剪贴板</li>
                  <li>• 点击"在指纹浏览器中打开"自动启动配置的浏览器</li>
                  <li>• 点击"忽略"将移除此URL（会保留在历史记录中）</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
