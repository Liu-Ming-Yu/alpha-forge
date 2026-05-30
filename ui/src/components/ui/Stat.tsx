import type { ReactNode } from "react";
import { useSeries } from "../../lib/history";
import { cn } from "../../lib/cn";
import { Card } from "./Card";
import { Sparkline } from "./charts";
import type { Tone } from "./atoms";

interface StatProps {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  right?: ReactNode;
  /** History key to draw a sparkline from. */
  seriesKey?: string;
  tone?: Tone;
  index?: number;
  className?: string;
}

/** Big-number metric tile with an optional live sparkline. */
export function Stat({ label, value, sub, right, seriesKey, tone = "accent", index, className }: StatProps) {
  const series = useSeries(seriesKey ?? "");
  return (
    <Card index={index} className={cn("flex flex-col justify-between", className)}>
      <div className="flex items-start justify-between gap-2">
        <span className="text-[12.5px] font-medium uppercase tracking-wide text-ink-tertiary">
          {label}
        </span>
        {right}
      </div>
      <div className="mt-3 flex items-end justify-between gap-3">
        <div className="min-w-0">
          <div className="text-metric font-semibold leading-none tracking-tight text-ink">
            {value}
          </div>
          {sub && <div className="mt-2 text-[13px] text-ink-tertiary">{sub}</div>}
        </div>
        {seriesKey && series.length > 1 && (
          <Sparkline data={series} tone={tone} width={104} height={36} />
        )}
      </div>
    </Card>
  );
}
