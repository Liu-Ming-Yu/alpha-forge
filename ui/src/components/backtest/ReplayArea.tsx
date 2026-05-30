import { Area, AreaChart, CartesianGrid, ResponsiveContainer, YAxis } from "recharts";
import { useId } from "react";
import type { Tone } from "../ui/atoms";

const TONE_RGB: Record<Tone, string> = {
  neutral: "var(--text-tertiary)",
  accent: "var(--accent)",
  success: "var(--success)",
  warn: "var(--warn)",
  danger: "var(--danger)",
};

/** Themed area chart over a (pre-sliced) series — the parent reveals a growing
 *  slice to produce the live "drawing-in" replay. */
export function ReplayArea({
  data,
  tone = "accent",
  height = 200,
  domain,
  baseline,
}: {
  data: { date: string; value: number }[];
  tone?: Tone;
  height?: number;
  domain?: [number, number];
  baseline?: number;
}) {
  const id = useId().replace(/:/g, "");
  const color = `rgb(${TONE_RGB[tone]})`;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 6, right: 4, bottom: 0, left: 4 }}>
        <defs>
          <linearGradient id={`bt-${id}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgb(var(--hairline) / 0.06)" vertical={false} />
        <YAxis hide domain={domain ?? ["auto", "auto"]} />
        <Area
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={2}
          fill={`url(#bt-${id})`}
          isAnimationActive={false}
          dot={false}
          baseValue={baseline ?? "dataMin"}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
