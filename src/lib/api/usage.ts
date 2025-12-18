import { invoke } from "@tauri-apps/api/core";

/**
 * 用量信息接口
 *
 * 与后端 UsageInfo 结构对应
 * _Requirements: 3.3_
 */
export interface UsageInfo {
  /** 订阅类型名称 */
  subscriptionTitle: string;
  /** 总额度 */
  usageLimit: number;
  /** 已使用 */
  currentUsage: number;
  /** 余额 = usageLimit - currentUsage */
  balance: number;
  /** 余额低于 20% */
  isLowBalance: boolean;
}

/**
 * Usage API
 */
export const usageApi = {
  /**
   * 获取 Kiro 凭证的用量信息
   *
   * @param credentialUuid - 凭证的 UUID
   * @returns 用量信息
   */
  getKiroUsage: (credentialUuid: string): Promise<UsageInfo> =>
    invoke("get_kiro_usage", { credentialUuid }),
};
