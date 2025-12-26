import React, { useState, useEffect } from "react";
import {
  Settings,
  Plus,
  Trash2,
  Save,
  RotateCcw,
  AlertCircle,
  CheckCircle,
  FolderOpen,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { BrowserInterceptorConfig } from "@/lib/api/browserInterceptor";
import * as browserInterceptorApi from "@/lib/api/browserInterceptor";

interface InterceptorConfigPanelProps {
  onStateChange: () => void;
}

export function InterceptorConfigPanel({
  onStateChange,
}: InterceptorConfigPanelProps) {
  const [config, setConfig] = useState<BrowserInterceptorConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{
    valid: boolean;
    message: string;
  } | null>(null);

  const [newTargetProcess, setNewTargetProcess] = useState("");
  const [newUrlPattern, setNewUrlPattern] = useState("");
  const [newExcludedProcess, setNewExcludedProcess] = useState("");

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    setLoading(true);
    try {
      const defaultConfig =
        await browserInterceptorApi.getDefaultBrowserInterceptorConfig();
      setConfig(defaultConfig);
    } catch (error) {
      console.error("加载配置失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveConfig = async () => {
    if (!config) return;

    setSaving(true);
    try {
      // 先验证配置
      await browserInterceptorApi.validateBrowserInterceptorConfig(config);

      // 检查拦截器是否正在运行
      const isRunning =
        await browserInterceptorApi.isBrowserInterceptorRunning();

      if (isRunning) {
        // 拦截器已运行，更新配置
        await browserInterceptorApi.updateBrowserInterceptorConfig(config);
        setValidationResult({
          valid: true,
          message: "配置已更新",
        });
      } else {
        // 拦截器未运行
        if (config.enabled) {
          // 用户启用了拦截器，使用配置启动
          await browserInterceptorApi.startBrowserInterceptor(config);
          setValidationResult({
            valid: true,
            message: "拦截器已启动",
          });
        } else {
          // 用户未启用拦截器，仅保存配置到本地（配置已验证通过）
          setValidationResult({
            valid: true,
            message:
              "配置已验证通过。开启「启用拦截器」开关后点击保存即可启动。",
          });
        }
      }
      onStateChange();
    } catch (error) {
      console.error("保存配置失败:", error);
      setValidationResult({
        valid: false,
        message: "保存配置失败: " + String(error),
      });
    } finally {
      setSaving(false);
    }
  };

  const handleValidateConfig = async () => {
    if (!config) return;

    setValidating(true);
    try {
      await browserInterceptorApi.validateBrowserInterceptorConfig(config);
      setValidationResult({
        valid: true,
        message: "配置验证通过",
      });
    } catch (error) {
      console.error("验证配置失败:", error);
      setValidationResult({
        valid: false,
        message: "验证失败: " + String(error),
      });
    } finally {
      setValidating(false);
    }
  };

  const handleResetConfig = async () => {
    setLoading(true);
    try {
      const defaultConfig =
        await browserInterceptorApi.getDefaultBrowserInterceptorConfig();
      setConfig(defaultConfig);
      setValidationResult(null);
    } catch (error) {
      console.error("重置配置失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const updateConfig = (updates: Partial<BrowserInterceptorConfig>) => {
    if (!config) return;
    setConfig({ ...config, ...updates });
    setValidationResult(null); // 清除验证结果
  };

  const addTargetProcess = () => {
    if (!newTargetProcess.trim() || !config) return;
    updateConfig({
      target_processes: [...config.target_processes, newTargetProcess.trim()],
    });
    setNewTargetProcess("");
  };

  const removeTargetProcess = (index: number) => {
    if (!config) return;
    const newProcesses = [...config.target_processes];
    newProcesses.splice(index, 1);
    updateConfig({ target_processes: newProcesses });
  };

  const addUrlPattern = () => {
    if (!newUrlPattern.trim() || !config) return;
    updateConfig({
      url_patterns: [...config.url_patterns, newUrlPattern.trim()],
    });
    setNewUrlPattern("");
  };

  const removeUrlPattern = (index: number) => {
    if (!config) return;
    const newPatterns = [...config.url_patterns];
    newPatterns.splice(index, 1);
    updateConfig({ url_patterns: newPatterns });
  };

  const addExcludedProcess = () => {
    if (!newExcludedProcess.trim() || !config) return;
    updateConfig({
      excluded_processes: [
        ...config.excluded_processes,
        newExcludedProcess.trim(),
      ],
    });
    setNewExcludedProcess("");
  };

  const removeExcludedProcess = (index: number) => {
    if (!config) return;
    const newProcesses = [...config.excluded_processes];
    newProcesses.splice(index, 1);
    updateConfig({ excluded_processes: newProcesses });
  };

  if (loading || !config) {
    return (
      <Card>
        <CardContent className="p-8">
          <div className="flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mr-2" />
            <span>加载配置中...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 配置操作按钮 */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold flex items-center">
          <Settings className="w-5 h-5 mr-2" />
          拦截器配置
        </h2>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            onClick={handleValidateConfig}
            disabled={validating}
          >
            {validating ? (
              <div className="w-4 h-4 border-2 border-gray-500 border-t-transparent rounded-full animate-spin mr-1" />
            ) : (
              <CheckCircle className="w-4 h-4 mr-1" />
            )}
            验证配置
          </Button>
          <Button variant="outline" onClick={handleResetConfig}>
            <RotateCcw className="w-4 h-4 mr-1" />
            重置为默认
          </Button>
          <Button onClick={handleSaveConfig} disabled={saving}>
            {saving ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-1" />
            ) : (
              <Save className="w-4 h-4 mr-1" />
            )}
            保存配置
          </Button>
        </div>
      </div>

      {/* 验证结果显示 */}
      {validationResult && (
        <div
          className={`p-4 rounded-lg border ${
            validationResult.valid
              ? "bg-green-50 border-green-200 text-green-700"
              : "bg-red-50 border-red-200 text-red-700"
          }`}
        >
          <div className="flex items-center space-x-2">
            {validationResult.valid ? (
              <CheckCircle className="w-4 h-4" />
            ) : (
              <AlertCircle className="w-4 h-4" />
            )}
            <span className="font-medium">
              {validationResult.valid ? "配置验证通过" : "配置验证失败"}
            </span>
          </div>
          {validationResult.message && (
            <p className="mt-1 text-sm">{validationResult.message}</p>
          )}
        </div>
      )}

      {/* 配置选项卡 */}
      <Tabs defaultValue="general" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="general">基本设置</TabsTrigger>
          <TabsTrigger value="processes">进程配置</TabsTrigger>
          <TabsTrigger value="patterns">URL 模式</TabsTrigger>
          <TabsTrigger value="browser">浏览器设置</TabsTrigger>
        </TabsList>

        {/* 基本设置 */}
        <TabsContent value="general" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>基本设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-base font-medium">启用拦截器</Label>
                  <p className="text-sm text-gray-600 mt-1">
                    开启后将拦截目标应用的浏览器启动请求
                  </p>
                </div>
                <Switch
                  checked={config.enabled}
                  onCheckedChange={(enabled) => updateConfig({ enabled })}
                />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-base font-medium">启用通知</Label>
                    <p className="text-sm text-gray-600 mt-1">
                      拦截URL时显示系统通知
                    </p>
                  </div>
                  <Switch
                    checked={config.notification_enabled}
                    onCheckedChange={(notification_enabled) =>
                      updateConfig({ notification_enabled })
                    }
                  />
                </div>
                {config.notification_enabled && (
                  <div className="pl-0">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        try {
                          await browserInterceptorApi.showBrowserInterceptorStatusNotification(
                            "这是一个测试通知，如果您看到这条消息，说明通知功能正常工作！",
                            "info",
                          );
                        } catch (error) {
                          console.error("测试通知失败:", error);
                          setValidationResult({
                            valid: false,
                            message: "测试通知失败: " + String(error),
                          });
                        }
                      }}
                    >
                      <CheckCircle className="w-4 h-4 mr-1" />
                      测试通知
                    </Button>
                  </div>
                )}
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-base font-medium">
                    自动复制到剪贴板
                  </Label>
                  <p className="text-sm text-gray-600 mt-1">
                    拦截URL后自动复制到剪贴板
                  </p>
                </div>
                <Switch
                  checked={config.auto_copy_to_clipboard}
                  onCheckedChange={(auto_copy_to_clipboard) =>
                    updateConfig({ auto_copy_to_clipboard })
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-base font-medium">
                    退出时自动恢复
                  </Label>
                  <p className="text-sm text-gray-600 mt-1">
                    应用退出时自动恢复浏览器行为
                  </p>
                </div>
                <Switch
                  checked={config.restore_on_exit}
                  onCheckedChange={(restore_on_exit) =>
                    updateConfig({ restore_on_exit })
                  }
                />
              </div>

              <div className="space-y-2">
                <Label className="text-base font-medium">
                  临时禁用超时（分钟）
                </Label>
                <p className="text-sm text-gray-600">
                  临时禁用后自动重新启用的时间，0 表示不自动恢复
                </p>
                <Input
                  type="number"
                  min="0"
                  max="1440"
                  value={
                    config.temporary_disable_timeout
                      ? config.temporary_disable_timeout / 60
                      : 0
                  }
                  onChange={(e) => {
                    const minutes = parseInt(e.target.value) || 0;
                    updateConfig({
                      temporary_disable_timeout:
                        minutes > 0 ? minutes * 60 : null,
                    });
                  }}
                  className="max-w-xs"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 进程配置 */}
        <TabsContent value="processes" className="mt-6">
          <div className="space-y-6">
            {/* 目标进程 */}
            <Card>
              <CardHeader>
                <CardTitle>目标进程</CardTitle>
                <p className="text-sm text-gray-600">
                  配置需要拦截浏览器启动的应用程序名称
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex space-x-2">
                    <Input
                      placeholder="例如：kiro, cursor, code"
                      value={newTargetProcess}
                      onChange={(e) => setNewTargetProcess(e.target.value)}
                      onKeyPress={(e) =>
                        e.key === "Enter" && addTargetProcess()
                      }
                    />
                    <Button
                      onClick={addTargetProcess}
                      disabled={!newTargetProcess.trim()}
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {config.target_processes.map((process, index) => (
                      <Badge
                        key={index}
                        variant="outline"
                        className="flex items-center space-x-1"
                      >
                        <span>{process}</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-auto p-0 ml-1 hover:bg-transparent"
                          onClick={() => removeTargetProcess(index)}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 排除进程 */}
            <Card>
              <CardHeader>
                <CardTitle>排除进程</CardTitle>
                <p className="text-sm text-gray-600">
                  配置永不拦截的进程名称（如系统浏览器）
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex space-x-2">
                    <Input
                      placeholder="例如：explorer, chrome, firefox"
                      value={newExcludedProcess}
                      onChange={(e) => setNewExcludedProcess(e.target.value)}
                      onKeyPress={(e) =>
                        e.key === "Enter" && addExcludedProcess()
                      }
                    />
                    <Button
                      onClick={addExcludedProcess}
                      disabled={!newExcludedProcess.trim()}
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {config.excluded_processes.map((process, index) => (
                      <Badge
                        key={index}
                        variant="secondary"
                        className="flex items-center space-x-1"
                      >
                        <span>{process}</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-auto p-0 ml-1 hover:bg-transparent"
                          onClick={() => removeExcludedProcess(index)}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* URL 模式 */}
        <TabsContent value="patterns" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>URL 匹配模式</CardTitle>
              <p className="text-sm text-gray-600">
                配置需要拦截的URL模式，支持通配符 *
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex space-x-2">
                  <Input
                    placeholder="例如：https://auth.*, https://*/oauth/*, localhost:*/auth"
                    value={newUrlPattern}
                    onChange={(e) => setNewUrlPattern(e.target.value)}
                    onKeyPress={(e) => e.key === "Enter" && addUrlPattern()}
                  />
                  <Button
                    onClick={addUrlPattern}
                    disabled={!newUrlPattern.trim()}
                  >
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
                <div className="space-y-2">
                  {config.url_patterns.map((pattern, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-2 bg-gray-50 rounded border"
                    >
                      <code className="text-sm font-mono">{pattern}</code>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeUrlPattern(index)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 浏览器设置 */}
        <TabsContent value="browser" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>指纹浏览器设置</CardTitle>
              <p className="text-sm text-gray-600">
                配置指纹浏览器的路径和启动参数
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-base font-medium">
                    启用指纹浏览器
                  </Label>
                  <p className="text-sm text-gray-600 mt-1">
                    启用后可以一键在指纹浏览器中打开URL
                  </p>
                </div>
                <Switch
                  checked={config.fingerprint_browser.enabled}
                  onCheckedChange={(enabled) =>
                    updateConfig({
                      fingerprint_browser: {
                        ...config.fingerprint_browser,
                        enabled,
                      },
                    })
                  }
                />
              </div>

              {config.fingerprint_browser.enabled && (
                <>
                  <div className="space-y-2">
                    <Label className="text-base font-medium">
                      浏览器可执行文件路径
                    </Label>
                    <div className="flex space-x-2">
                      <Input
                        placeholder="例如：C:\Program Files\Browser\browser.exe"
                        value={config.fingerprint_browser.executable_path}
                        onChange={(e) =>
                          updateConfig({
                            fingerprint_browser: {
                              ...config.fingerprint_browser,
                              executable_path: e.target.value,
                            },
                          })
                        }
                      />
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="outline" size="sm">
                              <FolderOpen className="w-4 h-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>浏览文件夹选择浏览器</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-base font-medium">
                      配置文件路径（可选）
                    </Label>
                    <Input
                      placeholder="例如：C:\Users\user\AppData\Local\Browser\Profile"
                      value={config.fingerprint_browser.profile_path}
                      onChange={(e) =>
                        updateConfig({
                          fingerprint_browser: {
                            ...config.fingerprint_browser,
                            profile_path: e.target.value,
                          },
                        })
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-base font-medium">
                      额外启动参数（可选）
                    </Label>
                    <Textarea
                      placeholder="每行一个参数，例如：&#10;--no-first-run&#10;--disable-background-timer-throttling"
                      value={config.fingerprint_browser.additional_args.join(
                        "\n",
                      )}
                      onChange={(e) =>
                        updateConfig({
                          fingerprint_browser: {
                            ...config.fingerprint_browser,
                            additional_args: e.target.value
                              .split("\n")
                              .filter((arg) => arg.trim()),
                          },
                        })
                      }
                      rows={4}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-base font-medium">
                        自动启动浏览器
                      </Label>
                      <p className="text-sm text-gray-600 mt-1">
                        拦截到URL时自动启动指纹浏览器
                      </p>
                    </div>
                    <Switch
                      checked={config.auto_launch_browser}
                      onCheckedChange={(auto_launch_browser) =>
                        updateConfig({ auto_launch_browser })
                      }
                    />
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
