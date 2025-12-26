import React, {
  createContext,
  useContext,
  useState,
  useRef,
  useEffect,
} from "react";
import { cn } from "@/lib/utils";

interface DropdownMenuContextType {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const DropdownMenuContext = createContext<DropdownMenuContextType | undefined>(
  undefined,
);

interface DropdownMenuProps {
  children: React.ReactNode;
}

const DropdownMenu: React.FC<DropdownMenuProps> = ({ children }) => {
  const [open, setOpen] = useState(false);

  return (
    <DropdownMenuContext.Provider value={{ open, setOpen }}>
      <div className="relative">{children}</div>
    </DropdownMenuContext.Provider>
  );
};

interface DropdownMenuTriggerProps {
  asChild?: boolean;
  children: React.ReactNode;
}

const DropdownMenuTrigger: React.FC<DropdownMenuTriggerProps> = ({
  asChild,
  children,
}) => {
  const context = useContext(DropdownMenuContext);
  if (!context)
    throw new Error("DropdownMenuTrigger must be used within DropdownMenu");

  const { setOpen } = context;

  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, {
      onClick: () => setOpen(true),
    });
  }

  return <button onClick={() => setOpen(true)}>{children}</button>;
};

interface DropdownMenuContentProps {
  className?: string;
  align?: "start" | "center" | "end";
  children: React.ReactNode;
}

const DropdownMenuContent: React.FC<DropdownMenuContentProps> = ({
  className,
  align = "center",
  children,
}) => {
  const context = useContext(DropdownMenuContext);
  if (!context)
    throw new Error("DropdownMenuContent must be used within DropdownMenu");

  const { open, setOpen } = context;
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [open, setOpen]);

  if (!open) return null;

  const alignmentClasses = {
    start: "left-0",
    center: "left-1/2 transform -translate-x-1/2",
    end: "right-0",
  };

  return (
    <div
      ref={ref}
      className={cn(
        "absolute top-full z-50 mt-1 min-w-32 rounded-md border bg-white shadow-md",
        alignmentClasses[align],
        className,
      )}
    >
      {children}
    </div>
  );
};

interface DropdownMenuItemProps {
  className?: string;
  children: React.ReactNode;
  onClick?: () => void;
}

const DropdownMenuItem: React.FC<DropdownMenuItemProps> = ({
  className,
  children,
  onClick,
}) => {
  const context = useContext(DropdownMenuContext);
  if (!context)
    throw new Error("DropdownMenuItem must be used within DropdownMenu");

  const { setOpen } = context;

  const handleClick = () => {
    onClick?.();
    setOpen(false);
  };

  return (
    <div
      className={cn(
        "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-gray-100",
        className,
      )}
      onClick={handleClick}
    >
      {children}
    </div>
  );
};

export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
};
