import { useState } from "react";
import {
  Save,
  Shuffle,
  ClipboardPaste,
  Settings,
  AlertTriangle,
  CheckCircle,
  FileUp,
  FileDown,
  Trash2,
  FolderOpen,
  RotateCcw,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useToast } from "@/hooks/use-toast";
import {
  MachineIdInfo,
  AdminStatus,
  MachineIdValidation,
} from "@/lib/api/machineId";
import { machineIdApi, machineIdUtils } from "@/lib/api/machineId";
import { open, save } from "@tauri-apps/plugin-dialog";

interface MachineIdManagePanelProps {
  machineIdInfo: MachineIdInfo | null;
  adminStatus: AdminStatus | null;
  onRefresh: () => void;
}

export function MachineIdManagePanel({
  machineIdInfo,
  adminStatus,
  onRefresh,
}: MachineIdManagePanelProps) {
  const { toast } = useToast();

  // 机器码修改
  const [newMachineId, setNewMachineId] = useState("");
  const [validationResult, setValidationResult] =
    useState<MachineIdValidation | null>(null);
  const [_isValidating, setIsValidating] = useState(false);
  const [isApplying, setIsApplying] = useState(false);

  // 格式转换
  const [formatInput, setFormatInput] = useState("");
  const [formatTarget, setFormatTarget] = useState<"uuid" | "hex32">("uuid");
  const [formatResult, setFormatResult] = useState("");

  // 备份恢复
  const [backupPath, setBackupPath] = useState("");
  const [restorePath, setRestorePath] = useState("");
  const [isBackingUp, setIsBackingUp] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);

  const canModify =
    machineIdInfo?.can_modify &&
    (!machineIdInfo.requires_admin || adminStatus?.is_admin);

  const handleValidateMachineId = async () => {
    if (!newMachineId.trim()) {
      setValidationResult(null);
      return;
    }

    try {
      setIsValidating(true);
      const result = await machineIdApi.validateMachineId(newMachineId.trim());
      setValidationResult(result);
    } catch (error) {
      console.error("验证失败:", error);
      toast({
        variant: "destructive",
        title: "验证失败",
        description: "无法验证机器码格式",
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleApplyMachineId = async () => {
    if (!validationResult?.is_valid) {
      toast({
        variant: "destructive",
        title: "无法应用",
        description: "机器码格式不正确",
      });
      return;
    }

    try {
      setIsApplying(true);
      const result = await machineIdApi.setMachineId(newMachineId.trim());

      if (result.success) {
        toast({
          title: "机器码修改成功",
          description: result.message,
        });
        setNewMachineId("");
        setValidationResult(null);
        onRefresh();

        if (result.requires_restart) {
          toast({
            title: "重启提醒",
            description: "部分应用程序可能需要重启才能识别新的机器码",
            duration: 8000,
          });
        }
      } else {
        toast({
          variant: "destructive",
          title: "机器码修改失败",
          description: result.message,
        });
      }
    } catch (error) {
      console.error("应用失败:", error);
      toast({
        variant: "destructive",
        title: "应用失败",
        description: "无法设置新的机器码",
      });
    } finally {
      setIsApplying(false);
    }
  };

  const handleGenerateRandom = async () => {
    try {
      const randomId = await machineIdApi.generateRandomMachineId();
      setNewMachineId(randomId);
      // 自动验证生成的机器码
      const result = await machineIdApi.validateMachineId(randomId);
      setValidationResult(result);
      toast({
        title: "随机机器码生成成功",
        description: "已生成新的随机机器码",
      });
    } catch (error) {
      console.error("生成失败:", error);
      toast({
        variant: "destructive",
        title: "生成失败",
        description: "无法生成随机机器码",
      });
    }
  };

  const handlePasteFromClipboard = async () => {
    try {
      const clipboardText = await machineIdApi.pasteMachineIdFromClipboard();
      setNewMachineId(clipboardText);
      // 自动验证粘贴的机器码
      const result = await machineIdApi.validateMachineId(clipboardText);
      setValidationResult(result);
      toast({
        title: "从剪贴板粘贴成功",
        description: "已从剪贴板粘贴机器码",
      });
    } catch (error) {
      console.error("粘贴失败:", error);
      toast({
        variant: "destructive",
        title: "粘贴失败",
        description: "剪贴板中没有有效的机器码",
      });
    }
  };

  const handleFormatConvert = async () => {
    if (!formatInput.trim()) return;

    try {
      const result = await machineIdApi.convertMachineIdFormat(
        formatInput.trim(),
        formatTarget,
      );
      setFormatResult(result);
      toast({
        title: "格式转换成功",
        description: `已转换为${formatTarget === "uuid" ? "UUID" : "32位十六进制"}格式`,
      });
    } catch (error) {
      console.error("转换失败:", error);
      toast({
        variant: "destructive",
        title: "转换失败",
        description: "无法转换机器码格式，请检查输入是否正确",
      });
    }
  };

  const handleBackup = async () => {
    if (!backupPath.trim()) return;

    try {
      setIsBackingUp(true);
      const success = await machineIdApi.backupMachineIdToFile(
        backupPath.trim(),
      );

      if (success) {
        toast({
          title: "备份成功",
          description: "机器码已备份到指定文件",
        });
        setBackupPath("");
        onRefresh();
      } else {
        toast({
          variant: "destructive",
          title: "备份失败",
          description: "无法创建备份文件",
        });
      }
    } catch (error) {
      console.error("备份失败:", error);
      toast({
        variant: "destructive",
        title: "备份失败",
        description: "备份操作失败",
      });
    } finally {
      setIsBackingUp(false);
    }
  };

  const handleSelectBackupFile = async () => {
    try {
      const selected = await save({
        title: "保存机器码备份文件",
        defaultPath: `machine_id_backup_${new Date().toISOString().slice(0, 10)}.json`,
        filters: [
          {
            name: "JSON文件",
            extensions: ["json"],
          },
        ],
      });

      if (selected && typeof selected === "string") {
        setBackupPath(selected);
        toast({
          title: "保存位置已选择",
          description: "备份文件保存路径已设置",
        });
      }
    } catch (error) {
      console.error("文件保存对话框失败:", error);
      toast({
        variant: "destructive",
        title: "文件选择失败",
        description: "无法打开文件保存对话框",
      });
    }
  };

  const handleSelectRestoreFile = async () => {
    try {
      const selected = await open({
        title: "选择机器码备份文件",
        filters: [
          {
            name: "JSON文件",
            extensions: ["json"],
          },
        ],
        multiple: false,
      });

      if (selected && typeof selected === "string") {
        setRestorePath(selected);
        toast({
          title: "文件已选择",
          description: "备份文件路径已设置",
        });
      }
    } catch (error) {
      console.error("文件选择失败:", error);
      toast({
        variant: "destructive",
        title: "文件选择失败",
        description: "无法打开文件选择对话框",
      });
    }
  };

  const handleRestore = async () => {
    if (!restorePath.trim()) return;

    try {
      setIsRestoring(true);
      const result = await machineIdApi.restoreMachineIdFromFile(
        restorePath.trim(),
      );

      if (result.success) {
        toast({
          title: "恢复成功",
          description: result.message,
        });
        setRestorePath("");
        onRefresh();

        if (result.requires_restart) {
          toast({
            title: "重启提醒",
            description: "部分应用程序可能需要重启才能识别恢复的机器码",
            duration: 8000,
          });
        }
      } else {
        toast({
          variant: "destructive",
          title: "恢复失败",
          description: result.message,
        });
      }
    } catch (error) {
      console.error("恢复失败:", error);
      toast({
        variant: "destructive",
        title: "恢复失败",
        description: "恢复操作失败",
      });
    } finally {
      setIsRestoring(false);
    }
  };

  const handleRestoreOriginal = async () => {
    if (!machineIdInfo?.original_id) return;

    try {
      setIsRestoring(true);
      const result = await machineIdApi.setMachineId(machineIdInfo.original_id);

      if (result.success) {
        toast({
          title: "恢复成功",
          description: "已恢复到原始机器码",
        });
        onRefresh();

        if (result.requires_restart) {
          toast({
            title: "重启提醒",
            description: "部分应用程序可能需要重启才能识别恢复的机器码",
            duration: 8000,
          });
        }
      } else {
        toast({
          variant: "destructive",
          title: "恢复失败",
          description: result.message,
        });
      }
    } catch (error) {
      console.error("恢复原始机器码失败:", error);
      toast({
        variant: "destructive",
        title: "恢复失败",
        description: "无法恢复到原始机器码",
      });
    } finally {
      setIsRestoring(false);
    }
  };

  const handleClearOverride = async () => {
    try {
      const result = await machineIdApi.clearMachineIdOverride();

      if (result.success) {
        toast({
          title: "清除成功",
          description: result.message,
        });
        onRefresh();
      } else {
        toast({
          variant: "destructive",
          title: "清除失败",
          description: result.message,
        });
      }
    } catch (error) {
      console.error("清除失败:", error);
      toast({
        variant: "destructive",
        title: "清除失败",
        description: "清除覆盖失败",
      });
    }
  };

  if (!machineIdInfo || !adminStatus) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span>无法加载管理功能</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            无法加载机器码管理功能，请检查系统状态。
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 权限状态提醒 */}
      {!canModify && (
        <Card className="border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/20">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-amber-800 dark:text-amber-300">
              <AlertTriangle className="w-5 h-5" />
              <span>权限不足</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-amber-700 dark:text-amber-400">
            <p>
              {!machineIdInfo.can_modify
                ? "当前平台不支持机器码修改"
                : "需要管理员权限才能修改机器码"}
            </p>
          </CardContent>
        </Card>
      )}

      {/* 机器码修改 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5 text-blue-500" />
            <span>修改机器码</span>
          </CardTitle>
          <CardDescription>
            输入新的机器码来替换当前的系统标识符
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="new-machine-id">新机器码</Label>
            <div className="flex space-x-2">
              <Input
                id="new-machine-id"
                placeholder="输入新的机器码（UUID或32位十六进制）"
                value={newMachineId}
                onChange={(e) => {
                  setNewMachineId(e.target.value);
                  setValidationResult(null);
                }}
                onBlur={handleValidateMachineId}
                disabled={!canModify}
                className="font-mono"
              />
              <Button
                variant="outline"
                onClick={handleGenerateRandom}
                disabled={!canModify}
                className="space-x-1 shrink-0"
              >
                <Shuffle className="w-4 h-4" />
                <span>随机生成</span>
              </Button>
              <Button
                variant="outline"
                onClick={handlePasteFromClipboard}
                disabled={!canModify}
                className="space-x-1 shrink-0"
              >
                <ClipboardPaste className="w-4 h-4" />
                <span>粘贴</span>
              </Button>
            </div>
          </div>

          {validationResult && (
            <div
              className={`p-3 rounded-lg border ${
                validationResult.is_valid
                  ? "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950/20"
                  : "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/20"
              }`}
            >
              <div className="flex items-center space-x-2">
                {validationResult.is_valid ? (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-red-600" />
                )}
                <span
                  className={`text-sm font-medium ${
                    validationResult.is_valid
                      ? "text-green-800 dark:text-green-300"
                      : "text-red-800 dark:text-red-300"
                  }`}
                >
                  {validationResult.is_valid ? "格式验证通过" : "格式验证失败"}
                </span>
              </div>

              {validationResult.is_valid && validationResult.formatted_id && (
                <div className="mt-2">
                  <p className="text-sm text-muted-foreground">格式化后：</p>
                  <p className="font-mono text-sm bg-background p-2 rounded border mt-1">
                    {validationResult.formatted_id}
                  </p>
                </div>
              )}

              {!validationResult.is_valid && validationResult.error_message && (
                <p className="text-sm text-red-700 dark:text-red-400 mt-1">
                  {validationResult.error_message}
                </p>
              )}

              <div className="mt-2">
                <Badge
                  variant={
                    validationResult.detected_format === "uuid"
                      ? "default"
                      : validationResult.detected_format === "hex32"
                        ? "secondary"
                        : "outline"
                  }
                >
                  检测格式：
                  {machineIdUtils.getFormatDisplayName(
                    validationResult.detected_format,
                  )}
                </Badge>
              </div>
            </div>
          )}

          <Button
            onClick={handleApplyMachineId}
            disabled={!canModify || !validationResult?.is_valid || isApplying}
            className="w-full space-x-2"
          >
            <Save className="w-4 h-4" />
            <span>{isApplying ? "应用中..." : "应用机器码"}</span>
          </Button>
        </CardContent>
      </Card>

      {/* 格式转换工具 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5 text-green-500" />
            <span>格式转换</span>
          </CardTitle>
          <CardDescription>
            在UUID格式和32位十六进制格式之间转换
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="format-input">源机器码</Label>
            <Input
              id="format-input"
              placeholder="输入要转换的机器码"
              value={formatInput}
              onChange={(e) => setFormatInput(e.target.value)}
              className="font-mono"
            />
          </div>

          <div className="space-y-2">
            <Label>目标格式</Label>
            <RadioGroup
              value={formatTarget}
              onValueChange={(value: string) =>
                setFormatTarget(value as "uuid" | "hex32")
              }
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="uuid" id="uuid" />
                <Label htmlFor="uuid">
                  UUID 格式 (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="hex32" id="hex32" />
                <Label htmlFor="hex32">
                  32位十六进制 (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)
                </Label>
              </div>
            </RadioGroup>
          </div>

          <Button
            onClick={handleFormatConvert}
            disabled={!formatInput.trim()}
            className="w-full space-x-2"
          >
            <Settings className="w-4 h-4" />
            <span>转换格式</span>
          </Button>

          {formatResult && (
            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">转换结果：</p>
              <p className="font-mono bg-background p-2 rounded border">
                {formatResult}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 备份和恢复 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <FileDown className="w-5 h-5 text-purple-500" />
            <span>备份与恢复</span>
          </CardTitle>
          <CardDescription>备份当前机器码或从备份文件恢复</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* 备份 */}
          <div className="space-y-3">
            <h4 className="font-medium">备份机器码</h4>
            <div className="flex space-x-2">
              <Input
                placeholder="选择保存位置..."
                value={backupPath}
                onChange={(e) => setBackupPath(e.target.value)}
                disabled={!canModify}
                className="flex-1"
              />
              <Button
                variant="outline"
                onClick={handleSelectBackupFile}
                disabled={!canModify}
                className="space-x-1 shrink-0"
              >
                <FolderOpen className="w-4 h-4" />
                <span>选择位置</span>
              </Button>
              <Button
                variant="outline"
                onClick={handleBackup}
                disabled={!canModify || !backupPath.trim() || isBackingUp}
                className="space-x-1 shrink-0"
              >
                <FileDown className="w-4 h-4" />
                <span>{isBackingUp ? "备份中..." : "备份"}</span>
              </Button>
            </div>
          </div>

          <Separator />

          {/* 恢复 */}
          <div className="space-y-3">
            <h4 className="font-medium">恢复机器码</h4>

            {/* 恢复到原始机器码 */}
            {machineIdInfo?.original_id && (
              <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="flex items-center justify-between">
                  <div>
                    <h5 className="font-medium text-blue-800 dark:text-blue-300">
                      恢复到原始机器码
                    </h5>
                    <p className="text-sm text-blue-700 dark:text-blue-400 mt-1">
                      将机器码恢复到首次备份的原始值
                    </p>
                    <p className="text-xs text-blue-600 dark:text-blue-500 font-mono mt-2">
                      {machineIdInfo.original_id}
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    onClick={handleRestoreOriginal}
                    disabled={!canModify || isRestoring}
                    className="space-x-1 shrink-0 border-blue-300 text-blue-700 hover:bg-blue-100"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>{isRestoring ? "恢复中..." : "恢复原始"}</span>
                  </Button>
                </div>
              </div>
            )}

            {/* 从文件恢复 */}
            <div className="space-y-2">
              <h5 className="text-sm font-medium text-muted-foreground">
                从备份文件恢复
              </h5>
              <div className="flex space-x-2">
                <Input
                  placeholder="选择备份文件..."
                  value={restorePath}
                  onChange={(e) => setRestorePath(e.target.value)}
                  disabled={!canModify}
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  onClick={handleSelectRestoreFile}
                  disabled={!canModify}
                  className="space-x-1 shrink-0"
                >
                  <FolderOpen className="w-4 h-4" />
                  <span>选择文件</span>
                </Button>
                <Button
                  variant="outline"
                  onClick={handleRestore}
                  disabled={!canModify || !restorePath.trim() || isRestoring}
                  className="space-x-1 shrink-0"
                >
                  <FileUp className="w-4 h-4" />
                  <span>{isRestoring ? "恢复中..." : "恢复"}</span>
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* macOS 特殊操作 */}
      {machineIdInfo.platform.toLowerCase() === "macos" &&
        machineIdInfo.original_id && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Trash2 className="w-5 h-5 text-red-500" />
                <span>macOS 特殊操作</span>
              </CardTitle>
              <CardDescription>
                清除机器码覆盖，恢复原始系统机器码
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="p-4 bg-amber-50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-800">
                <div className="flex items-center space-x-2 mb-3">
                  <AlertTriangle className="w-4 h-4 text-amber-600" />
                  <p className="text-sm font-medium text-amber-800 dark:text-amber-300">
                    注意：此操作将删除当前的机器码覆盖
                  </p>
                </div>
                <p className="text-sm text-amber-700 dark:text-amber-400 mb-3">
                  这将使系统恢复到原始机器码：{machineIdInfo.original_id}
                </p>
                <Button
                  variant="destructive"
                  onClick={handleClearOverride}
                  className="space-x-2"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>清除机器码覆盖</span>
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
    </div>
  );
}
