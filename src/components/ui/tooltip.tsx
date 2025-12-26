import React, { createContext, useContext, useState } from "react";
import { cn } from "@/lib/utils";

interface TooltipContextType {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const TooltipContext = createContext<TooltipContextType | undefined>(undefined);

interface TooltipProviderProps {
  children: React.ReactNode;
}

const TooltipProvider: React.FC<TooltipProviderProps> = ({ children }) => {
  return <>{children}</>;
};

interface TooltipProps {
  children: React.ReactNode;
}

const Tooltip: React.FC<TooltipProps> = ({ children }) => {
  const [open, setOpen] = useState(false);

  return (
    <TooltipContext.Provider value={{ open, setOpen }}>
      <div className="relative">{children}</div>
    </TooltipContext.Provider>
  );
};

interface TooltipTriggerProps {
  asChild?: boolean;
  children: React.ReactNode;
}

const TooltipTrigger: React.FC<TooltipTriggerProps> = ({
  asChild,
  children,
}) => {
  const context = useContext(TooltipContext);
  if (!context) throw new Error("TooltipTrigger must be used within Tooltip");

  const { setOpen } = context;

  const handleMouseEnter = () => setOpen(true);
  const handleMouseLeave = () => setOpen(false);

  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, {
      onMouseEnter: handleMouseEnter,
      onMouseLeave: handleMouseLeave,
    });
  }

  return (
    <div onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
      {children}
    </div>
  );
};

interface TooltipContentProps {
  className?: string;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  children: React.ReactNode;
}

const TooltipContent: React.FC<TooltipContentProps> = ({
  className,
  side = "top",
  align = "center",
  children,
}) => {
  const context = useContext(TooltipContext);
  if (!context) throw new Error("TooltipContent must be used within Tooltip");

  const { open } = context;

  if (!open) return null;

  const sideClasses = {
    top: "bottom-full mb-2",
    right: "left-full ml-2",
    bottom: "top-full mt-2",
    left: "right-full mr-2",
  };

  const alignClasses = {
    start: side === "top" || side === "bottom" ? "left-0" : "top-0",
    center:
      side === "top" || side === "bottom"
        ? "left-1/2 transform -translate-x-1/2"
        : "top-1/2 transform -translate-y-1/2",
    end: side === "top" || side === "bottom" ? "right-0" : "bottom-0",
  };

  return (
    <div
      className={cn(
        "absolute z-50 rounded-md bg-gray-900 px-3 py-1.5 text-xs text-white shadow-md",
        sideClasses[side],
        alignClasses[align],
        className,
      )}
    >
      {children}
    </div>
  );
};

export { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent };
