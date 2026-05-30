import { Area, AreaChart, ResponsiveContainer, YAxis } from "recharts";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { useId } from "react";
import { cn } from "../../lib/cn";
import type { Point } from "../../lib/history";
import type { Tone } from "./atoms";

const TONE_RGB: Record<Tone, string> = {
  neutral: "var(--text-tertiary)",
  accent: "var(--accent)",
  success: "var(--success)",
  warn: "var(--warn)",
  danger: "var(--danger)",
};
const stroke = (tone: Tone) => `rgb(${TONE_RGB[tone]})`;

/** Tiny inline SVG sparkline from a rolling series. */
export function Sparkline({
  data,
  tone = "accent",
  width = 96,
  height = 28,
  strokeWidth = 1.75,
}: {
  data: Point[];
  tone?: Tone;
  width?: number;
  height?: number;
  strokeWidth?: number;
}) {
  if (data.length < 2) {
    return <div style={{ width, height }} className="opacity-30" />;
  }
  const xs = data.map((d) => d.t);
  const ys = data.map((d) => d.v);
  const minX = xs[0];
  const maxX = xs[xs.length - 1] || minX + 1;
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const pad = strokeWidth;
  const px = (x: number) => pad + ((x - minX) / spanX) * (width - 2 * pad);
  const py = (y: number) => height - pad - ((y - minY) / spanY) * (height - 2 * pad);
  const d = data.map((p, i) => `${i ? "L" : "M"}${px(p.t).toFixed(2)},${py(p.v).toFixed(2)}`).join(" ");
  const last = data[data.length - 1];
  return (
    <svg width={width} height={height} className="overflow-visible">
      <path d={d} fill="none" stroke={stroke(tone)} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={px(last.t)} cy={py(last.v)} r={2.4} fill={stroke(tone)} />
    </svg>
  );
}

/** Larger streaming area chart (recharts), themed and axis-light. */
export function LiveArea({
  data,
  tone = "accent",
  height = 160,
}: {
  data: Point[];
  tone?: Tone;
  height?: number;
}) {
  const id = useId().replace(/:/g, "");
  const color = stroke(tone);
  if (data.length < 2) {
    return (
      <div style={{ height }} className="flex items-center justify-center text-xs text-ink-tertiary">
        Collecting live data…
      </div>
    );
  }
  const ys = data.map((d) => d.v);
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.12;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 6, right: 2, bottom: 0, left: 2 }}>
        <defs>
          <linearGradient id={`g-${id}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.32} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <YAxis hide domain={[lo - pad, hi + pad]} />
        <Area
          type="monotone"
          dataKey="v"
          stroke={color}
          strokeWidth={2}
          fill={`url(#g-${id})`}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** Horizontal meter: value within [min,max] with an optional threshold tick. */
export function Meter({
  value,
  min,
  max,
  tone = "accent",
  threshold,
  height = 8,
}: {
  value: number | null;
  min: number;
  max: number;
  tone?: Tone;
  threshold?: number;
  height?: number;
}) {
  const span = max - min || 1;
  const clamp = (x: number) => Math.max(0, Math.min(1, (x - min) / span));
  const pct = value === null ? 0 : clamp(value) * 100;
  return (
    <div className="relative w-full rounded-full bg-hairline/10" style={{ height }}>
      <div
        className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700"
        style={{ width: `${pct}%`, background: stroke(tone) }}
      />
      {threshold !== undefined && (
        <div
          className="absolute inset-y-[-2px] w-px bg-ink/60"
          style={{ left: `${clamp(threshold) * 100}%` }}
          title={`target ${threshold}`}
        />
      )}
    </div>
  );
}

export function Delta({ value, alreadyPct }: { value: number | null; alreadyPct?: boolean }) {
  if (value === null || !Number.isFinite(value)) return null;
  const pct = alreadyPct ? value : value * 100;
  const up = pct >= 0;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 text-xs font-semibold tnum",
        up ? "text-success" : "text-danger",
      )}
    >
      {up ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
      {Math.abs(pct).toFixed(2)}%
    </span>
  );
}
