import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { cn } from "../../lib/cn";

interface CardProps {
  children: ReactNode;
  className?: string;
  /** Stagger index for entrance animation. */
  index?: number;
  interactive?: boolean;
}

export function Card({ children, className, index = 0, interactive }: CardProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: Math.min(index * 0.035, 0.3), ease: [0.22, 1, 0.36, 1] }}
      className={cn(
        "card relative min-w-0 overflow-hidden p-5",
        interactive && "transition-shadow hover:shadow-float",
        className,
      )}
    >
      {children}
    </motion.section>
  );
}

interface CardHeaderProps {
  title: string;
  hint?: ReactNode;
  right?: ReactNode;
  icon?: ReactNode;
}

export function CardHeader({ title, hint, right, icon }: CardHeaderProps) {
  return (
    <header className="mb-4 flex items-start justify-between gap-3">
      <div className="flex items-center gap-2.5 min-w-0">
        {icon && <span className="text-ink-tertiary shrink-0">{icon}</span>}
        <div className="min-w-0">
          <h3 className="text-[13px] font-semibold tracking-tight text-ink-secondary">
            {title}
          </h3>
          {hint && <p className="mt-0.5 text-xs text-ink-tertiary truncate">{hint}</p>}
        </div>
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </header>
  );
}

export function Section({
  title,
  children,
  className,
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-4", className)}>
      <h2 className="text-[15px] font-semibold tracking-tight text-ink px-0.5">{title}</h2>
      {children}
    </div>
  );
}
