import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400/70 disabled:pointer-events-none disabled:opacity-45 active:scale-[0.98]",
  {
    variants: {
      variant: {
        primary: "bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-950/30 hover:-translate-y-0.5 hover:shadow-violet-700/30",
        secondary: "border border-white/10 bg-white/[0.055] text-slate-200 hover:border-violet-400/30 hover:bg-violet-400/10",
        ghost: "text-slate-400 hover:bg-white/[0.06] hover:text-white",
        danger: "border border-rose-400/25 bg-rose-400/10 text-rose-200 hover:bg-rose-400/15"
      },
      size: {
        sm: "h-9 px-3",
        md: "h-11 px-4",
        icon: "size-11 p-0 rounded-full"
      }
    },
    defaultVariants: { variant: "secondary", size: "md" }
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  )
);
Button.displayName = "Button";
