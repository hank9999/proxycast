import React, { createContext, useContext, useState } from "react";
import { Modal } from "../Modal";
import { cn } from "@/lib/utils";

interface DialogContextType {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const DialogContext = createContext<DialogContextType | undefined>(undefined);

interface DialogProps {
  children: React.ReactNode;
}

const Dialog: React.FC<DialogProps> = ({ children }) => {
  const [open, setOpen] = useState(false);

  return (
    <DialogContext.Provider value={{ open, setOpen }}>
      {children}
    </DialogContext.Provider>
  );
};

interface DialogTriggerProps {
  asChild?: boolean;
  children: React.ReactNode;
}

const DialogTrigger: React.FC<DialogTriggerProps> = ({ asChild, children }) => {
  const context = useContext(DialogContext);
  if (!context) throw new Error("DialogTrigger must be used within Dialog");

  const { setOpen } = context;

  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, {
      onClick: () => setOpen(true),
    });
  }

  return <button onClick={() => setOpen(true)}>{children}</button>;
};

interface DialogContentProps {
  className?: string;
  children: React.ReactNode;
}

const DialogContent: React.FC<DialogContentProps> = ({
  className,
  children,
}) => {
  const context = useContext(DialogContext);
  if (!context) throw new Error("DialogContent must be used within Dialog");

  const { open, setOpen } = context;

  return (
    <Modal isOpen={open} onClose={() => setOpen(false)} className={className}>
      {children}
    </Modal>
  );
};

interface DialogHeaderProps {
  className?: string;
  children: React.ReactNode;
}

const DialogHeader: React.FC<DialogHeaderProps> = ({ className, children }) => (
  <div
    className={cn(
      "flex flex-col space-y-1.5 text-center sm:text-left",
      className,
    )}
  >
    {children}
  </div>
);

interface DialogFooterProps {
  className?: string;
  children: React.ReactNode;
}

const DialogFooter: React.FC<DialogFooterProps> = ({ className, children }) => (
  <div
    className={cn(
      "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",
      className,
    )}
  >
    {children}
  </div>
);

interface DialogTitleProps {
  className?: string;
  children: React.ReactNode;
}

const DialogTitle: React.FC<DialogTitleProps> = ({ className, children }) => (
  <h2
    className={cn(
      "text-lg font-semibold leading-none tracking-tight",
      className,
    )}
  >
    {children}
  </h2>
);

interface DialogDescriptionProps {
  className?: string;
  children: React.ReactNode;
}

const DialogDescription: React.FC<DialogDescriptionProps> = ({
  className,
  children,
}) => <p className={cn("text-sm text-gray-600", className)}>{children}</p>;

export {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
};
