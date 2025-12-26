import { useState, useEffect } from "react";
import { ArrowLeft, Globe, Settings, History, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatusControlPanel } from "./StatusControlPanel";
import { InterceptedUrlsPanel } from "./InterceptedUrlsPanel";
import { InterceptorConfigPanel } from "./InterceptorConfigPanel";
import { UrlHistoryPanel } from "./UrlHistoryPanel";
import { SystemStatusPanel } from "./SystemStatusPanel";
import { InterceptorState } from "@/lib/api/browserInterceptor";
import * as browserInterceptorApi from "@/lib/api/browserInterceptor";

interface BrowserInterceptorToolProps {
  onNavigate: (
    page:
      | "provider-pool"
      | "config-management"
      | "api-server"
      | "flow-monitor"
      | "tools"
      | "browser-interceptor"
      | "settings",
  ) => void;
}

export function BrowserInterceptorTool({
  onNavigate,
}: BrowserInterceptorToolProps) {
  const [interceptorState, setInterceptorState] =
    useState<InterceptorState | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadInterceptorState();

    // 每5秒刷新一次状态
    const interval = setInterval(loadInterceptorState, 5000);

    return () => clearInterval(interval);
  }, []);

  const loadInterceptorState = async () => {
    try {
      const state = await browserInterceptorApi.getBrowserInterceptorState();
      setInterceptorState(state);
    } catch (error) {
      console.error("加载拦截器状态失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleStateChange = () => {
    // 状态变化时重新加载状态
    loadInterceptorState();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Globe className="w-16 h-16 mx-auto mb-4 animate-spin text-blue-500" />
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
            <Globe className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-3xl font-bold">浏览器拦截器</h1>
              <p className="text-gray-600 text-sm mt-1">
                拦截桌面应用的浏览器启动，支持手动复制 URL 到指纹浏览器
              </p>
            </div>
          </div>
        </div>

        {/* 状态控制面板 */}
        <StatusControlPanel
          state={interceptorState}
          onStateChange={handleStateChange}
        />
      </div>

      {/* 主要功能标签页 */}
      <Tabs defaultValue="intercepted" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger
            value="intercepted"
            className="flex items-center space-x-2"
          >
            <Globe className="w-4 h-4" />
            <span>拦截的 URL</span>
          </TabsTrigger>
          <TabsTrigger value="config" className="flex items-center space-x-2">
            <Settings className="w-4 h-4" />
            <span>拦截配置</span>
          </TabsTrigger>
          <TabsTrigger value="history" className="flex items-center space-x-2">
            <History className="w-4 h-4" />
            <span>历史记录</span>
          </TabsTrigger>
          <TabsTrigger value="status" className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4" />
            <span>系统状态</span>
          </TabsTrigger>
          <TabsTrigger value="help" className="flex items-center space-x-2">
            <span>帮助</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="intercepted" className="mt-6">
          <InterceptedUrlsPanel onStateChange={handleStateChange} />
        </TabsContent>

        <TabsContent value="config" className="mt-6">
          <InterceptorConfigPanel onStateChange={handleStateChange} />
        </TabsContent>

        <TabsContent value="history" className="mt-6">
          <UrlHistoryPanel />
        </TabsContent>

        <TabsContent value="status" className="mt-6">
          <SystemStatusPanel state={interceptorState} />
        </TabsContent>

        <TabsContent value="help" className="mt-6">
          <div className="space-y-6">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-blue-900 mb-4">
                🚀 浏览器拦截器使用指南
              </h3>

              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-blue-800 mb-2">
                    什么是浏览器拦截器？
                  </h4>
                  <p className="text-blue-700 text-sm">
                    浏览器拦截器专门解决 Kiro、Cursor、VSCode 等桌面 AI 客户端的
                    OAuth 登录问题。
                    当这些应用尝试打开浏览器进行登录时，我们会拦截这些请求，让您可以手动在指纹浏览器中完成登录。
                  </p>
                </div>

                <div>
                  <h4 className="font-medium text-blue-800 mb-2">
                    🔧 工作原理
                  </h4>
                  <ul className="text-blue-700 text-sm space-y-1 ml-4">
                    <li>• 监听系统级的浏览器启动请求</li>
                    <li>• 识别来自目标应用的 URL 打开请求</li>
                    <li>• 阻止默认浏览器启动</li>
                    <li>• 捕获并存储 OAuth URL</li>
                    <li>• 提供一键复制和指纹浏览器启动功能</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-medium text-blue-800 mb-2">
                    💡 使用步骤
                  </h4>
                  <ol className="text-blue-700 text-sm space-y-1 ml-4">
                    <li>1. 启用浏览器拦截器</li>
                    <li>2. 在 Kiro 等应用中点击登录</li>
                    <li>3. 查看拦截到的 URL</li>
                    <li>4. 一键复制或在指纹浏览器中打开</li>
                    <li>5. 完成登录后可恢复正常浏览器行为</li>
                  </ol>
                </div>

                <div>
                  <h4 className="font-medium text-blue-800 mb-2">
                    ⚠️ 重要提醒
                  </h4>
                  <ul className="text-blue-700 text-sm space-y-1 ml-4">
                    <li>• 使用完成后记得点击"恢复正常"以免影响其他软件</li>
                    <li>• 支持临时禁用功能，会在指定时间后自动恢复拦截</li>
                    <li>• 所有拦截记录都会保存在历史中，可随时查看</li>
                    <li>• 目前主要支持 Windows 平台，其他平台正在开发中</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="font-medium text-yellow-800 mb-2">
                🔐 隐私和安全
              </h4>
              <p className="text-yellow-700 text-sm">
                拦截器仅捕获 OAuth 相关的 URL，不会记录任何敏感信息。
                所有数据都保存在本地，不会上传到任何服务器。
              </p>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
