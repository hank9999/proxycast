import { toast as sonnerToast } from "sonner";

export interface ToastProps {
  title: string;
  description?: string;
  variant?: "default" | "destructive";
  duration?: number;
}

export function useToast() {
  const toast = ({
    title,
    description,
    variant = "default",
    duration,
  }: ToastProps) => {
    if (variant === "destructive") {
      sonnerToast.error(title, {
        description,
        duration,
      });
    } else {
      sonnerToast.success(title, {
        description,
        duration,
      });
    }
  };

  return { toast };
}
