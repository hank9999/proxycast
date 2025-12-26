import { invoke } from "@tauri-apps/api/core";

// Machine ID types matching Rust backend

export type MachineIdFormat = "uuid" | "hex32" | "unknown";

export interface MachineIdInfo {
  current_id: string;
  original_id?: string;
  platform: string;
  can_modify: boolean;
  requires_admin: boolean;
  backup_exists: boolean;
  format_type: MachineIdFormat;
}

export interface MachineIdResult {
  success: boolean;
  message: string;
  requires_restart: boolean;
  requires_admin: boolean;
  new_machine_id?: string;
}

export interface AdminStatus {
  is_admin: boolean;
  platform: string;
  elevation_method?: string;
  check_success: boolean;
}

export interface MachineIdValidation {
  is_valid: boolean;
  error_message?: string;
  formatted_id?: string;
  detected_format: MachineIdFormat;
}

export interface MachineIdHistory {
  id: string;
  machine_id: string;
  timestamp: string;
  platform: string;
  backup_path?: string;
}

export interface SystemInfo {
  os: string;
  arch: string;
  family: string;
  machine_id_support: PlatformSupport;
  requires_admin: boolean;
}

export interface PlatformSupport {
  can_read: boolean;
  can_write: boolean;
  format: string;
  method: string;
  limitations: string[];
}

export const machineIdApi = {
  /**
   * 获取当前机器码信息
   */
  async getCurrentMachineId(): Promise<MachineIdInfo> {
    return invoke("get_current_machine_id");
  },

  /**
   * 设置新的机器码
   */
  async setMachineId(newId: string): Promise<MachineIdResult> {
    return invoke("set_machine_id", { newId });
  },

  /**
   * 生成随机机器码
   */
  async generateRandomMachineId(): Promise<string> {
    return invoke("generate_random_machine_id");
  },

  /**
   * 验证机器码格式
   */
  async validateMachineId(machineId: string): Promise<MachineIdValidation> {
    return invoke("validate_machine_id", { machineId });
  },

  /**
   * 检查管理员权限
   */
  async checkAdminPrivileges(): Promise<AdminStatus> {
    return invoke("check_admin_privileges");
  },

  /**
   * 获取操作系统类型
   */
  async getOsType(): Promise<string> {
    return invoke("get_os_type");
  },

  /**
   * 备份机器码到文件
   */
  async backupMachineIdToFile(filePath: string): Promise<boolean> {
    return invoke("backup_machine_id_to_file", { filePath });
  },

  /**
   * 从文件恢复机器码
   */
  async restoreMachineIdFromFile(filePath: string): Promise<MachineIdResult> {
    return invoke("restore_machine_id_from_file", { filePath });
  },

  /**
   * 格式化机器码
   * @param machineId 机器码
   * @param formatType 格式类型："uuid" 或 "hex32"
   */
  async formatMachineId(
    machineId: string,
    formatType: "uuid" | "hex32",
  ): Promise<string> {
    return invoke("format_machine_id", { machineId, formatType });
  },

  /**
   * 检测机器码格式
   */
  async detectMachineIdFormat(machineId: string): Promise<string> {
    return invoke("detect_machine_id_format", { machineId });
  },

  /**
   * 转换机器码格式
   * @param machineId 机器码
   * @param targetFormat 目标格式："uuid" 或 "hex32"
   */
  async convertMachineIdFormat(
    machineId: string,
    targetFormat: "uuid" | "hex32",
  ): Promise<string> {
    return invoke("convert_machine_id_format", { machineId, targetFormat });
  },

  /**
   * 获取机器码历史记录
   */
  async getMachineIdHistory(): Promise<MachineIdHistory[]> {
    return invoke("get_machine_id_history");
  },

  /**
   * 清除机器码覆盖（仅限 macOS）
   */
  async clearMachineIdOverride(): Promise<MachineIdResult> {
    return invoke("clear_machine_id_override");
  },

  /**
   * 复制机器码到剪贴板
   */
  async copyMachineIdToClipboard(machineId: string): Promise<boolean> {
    return invoke("copy_machine_id_to_clipboard", { machineId });
  },

  /**
   * 从剪贴板粘贴机器码
   */
  async pasteMachineIdFromClipboard(): Promise<string> {
    return invoke("paste_machine_id_from_clipboard");
  },

  /**
   * 获取系统信息
   */
  async getSystemInfo(): Promise<SystemInfo> {
    return invoke("get_system_info");
  },
};

// Helper functions for common operations
export const machineIdUtils = {
  /**
   * 检查机器码是否为有效的UUID格式
   */
  isValidUuid(machineId: string): boolean {
    const uuidRegex =
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    return uuidRegex.test(machineId);
  },

  /**
   * 检查机器码是否为有效的32位十六进制格式
   */
  isValidHex32(machineId: string): boolean {
    const hex32Regex = /^[0-9a-f]{32}$/i;
    return hex32Regex.test(machineId);
  },

  /**
   * 清理机器码字符串（移除连字符和空格）
   */
  cleanMachineId(machineId: string): string {
    return machineId.replace(/[-\s]/g, "").toLowerCase();
  },

  /**
   * 格式化机器码为UUID格式
   */
  formatAsUuid(machineId: string): string {
    const cleaned = this.cleanMachineId(machineId);
    if (cleaned.length !== 32) {
      throw new Error("Invalid machine ID length");
    }
    return `${cleaned.slice(0, 8)}-${cleaned.slice(8, 12)}-${cleaned.slice(12, 16)}-${cleaned.slice(16, 20)}-${cleaned.slice(20)}`;
  },

  /**
   * 获取机器码的显示名称
   */
  getFormatDisplayName(format: MachineIdFormat): string {
    switch (format) {
      case "uuid":
        return "UUID格式";
      case "hex32":
        return "32位十六进制";
      case "unknown":
        return "未知格式";
      default:
        return "未知格式";
    }
  },

  /**
   * 获取平台的显示名称
   */
  getPlatformDisplayName(platform: string): string {
    switch (platform.toLowerCase()) {
      case "windows":
        return "Windows";
      case "macos":
        return "macOS";
      case "linux":
        return "Linux";
      default:
        return platform;
    }
  },
};
