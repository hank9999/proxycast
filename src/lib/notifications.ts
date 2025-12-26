// 系统通知管理器
import { invoke } from "@tauri-apps/api/core";

export interface NotificationConfig {
  title: string;
  body: string;
  icon?: string;
  sound?: boolean;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  id: string;
  title: string;
}

export interface InterceptNotificationData {
  urlId: string;
  url: string;
  sourceProcess: string;
  timestamp: string;
}

class NotificationManager {
  private enabled = true;

  constructor() {
    // 检查通知权限
    this.checkPermission();
  }

  /**
   * 检查并请求通知权限
   */
  async checkPermission(): Promise<boolean> {
    // 在Tauri中，通知权限通常在应用启动时处理
    return true;
  }

  /**
   * 设置通知是否启用
   */
  setEnabled(enabled: boolean) {
    this.enabled = enabled;
  }

  /**
   * 显示URL拦截通知
   */
  async showInterceptNotification(
    data: InterceptNotificationData,
  ): Promise<void> {
    if (!this.enabled) return;

    try {
      const config: NotificationConfig = {
        title: "🔐 拦截到新的URL",
        body: `来自 ${data.sourceProcess}: ${this.truncateUrl(data.url)}`,
        icon: "icon",
        sound: true,
        actions: [
          { id: "copy", title: "复制URL" },
          { id: "open", title: "打开浏览器" },
          { id: "dismiss", title: "忽略" },
        ],
      };

      await this.showNotification(config);
    } catch (error) {
      console.error("显示拦截通知失败:", error);
    }
  }

  /**
   * 显示系统状态通知
   */
  async showStatusNotification(
    title: string,
    message: string,
    type: "info" | "success" | "warning" | "error" = "info",
  ): Promise<void> {
    if (!this.enabled) return;

    const icons = {
      info: "ℹ️",
      success: "✅",
      warning: "⚠️",
      error: "❌",
    };

    try {
      const config: NotificationConfig = {
        title: `${icons[type]} ${title}`,
        body: message,
        sound: type === "error" || type === "warning",
      };

      await this.showNotification(config);
    } catch (error) {
      console.error("显示状态通知失败:", error);
    }
  }

  /**
   * 显示通知
   */
  private async showNotification(config: NotificationConfig): Promise<void> {
    try {
      // 尝试使用Tauri的通知API
      await invoke("show_notification", {
        title: config.title,
        body: config.body,
        icon: config.icon,
      });
    } catch (error) {
      // 如果Tauri通知不可用，降级使用Web通知
      console.warn("Tauri通知不可用，使用Web通知:", error);
      await this.showWebNotification(config);
    }
  }

  /**
   * 显示Web通知（降级方案）
   */
  private async showWebNotification(config: NotificationConfig): Promise<void> {
    try {
      if (!("Notification" in window)) {
        console.warn("浏览器不支持通知");
        return;
      }

      // 检查权限
      if (Notification.permission === "default") {
        const permission = await Notification.requestPermission();
        if (permission !== "granted") {
          console.warn("用户拒绝了通知权限");
          return;
        }
      }

      if (Notification.permission === "granted") {
        const notification = new Notification(config.title, {
          body: config.body,
          icon: config.icon || "/icon.png",
          requireInteraction: true,
        });

        // 设置点击事件
        notification.onclick = () => {
          window.focus();
          notification.close();
        };

        // 自动关闭
        setTimeout(() => {
          notification.close();
        }, 5000);
      }
    } catch (error) {
      console.error("显示Web通知失败:", error);
    }
  }

  /**
   * 截断长URL用于显示
   */
  private truncateUrl(url: string, maxLength = 60): string {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + "...";
  }

  /**
   * 显示测试通知
   */
  async showTestNotification(): Promise<void> {
    await this.showStatusNotification(
      "通知测试",
      "如果您看到这条消息，说明通知功能正常工作！",
      "info",
    );
  }
}

// 创建全局通知管理器实例
export const notificationManager = new NotificationManager();

// 导出便捷方法
export const showInterceptNotification = (data: InterceptNotificationData) =>
  notificationManager.showInterceptNotification(data);

export const showStatusNotification = (
  title: string,
  message: string,
  type?: "info" | "success" | "warning" | "error",
) => notificationManager.showStatusNotification(title, message, type);

export const showTestNotification = () =>
  notificationManager.showTestNotification();

export const setNotificationsEnabled = (enabled: boolean) =>
  notificationManager.setEnabled(enabled);
