import {
  Activity,
  ArrowLeftRight,
  CandlestickChart,
  Cpu,
  FlaskConical,
  LineChart,
  LogOut,
  Sparkles,
  SlidersHorizontal,
  TerminalSquare,
} from "lucide-react";
import type { ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { NavLink } from "react-router-dom";
import { cn } from "../../lib/cn";
import { updateSettings } from "../../lib/settings";

interface NavItem {
  to: string;
  label: string;
  icon: ReactNode;
}

const ITEMS: NavItem[] = [
  { to: "/overview", label: "Overview", icon: <Activity size={18} /> },
  { to: "/strategy", label: "Strategy", icon: <LineChart size={18} /> },
  { to: "/alpha", label: "Alpha", icon: <Sparkles size={18} /> },
  { to: "/execution", label: "Execution", icon: <ArrowLeftRight size={18} /> },
  { to: "/research", label: "Research", icon: <FlaskConical size={18} /> },
  { to: "/backtest", label: "Backtest", icon: <CandlestickChart size={18} /> },
  { to: "/system", label: "System", icon: <Cpu size={18} /> },
  { to: "/commands", label: "Commands", icon: <TerminalSquare size={18} /> },
  { to: "/settings", label: "Settings", icon: <SlidersHorizontal size={18} /> },
];

export function NavRail() {
  const qc = useQueryClient();
  const disconnect = () => {
    updateSettings({ apiKey: "" });
    qc.clear();
  };

  return (
    <nav className="glass sticky top-0 z-30 flex h-screen w-[68px] shrink-0 flex-col border-r border-hairline/[0.07] lg:w-[228px]">
      <div className="flex items-center gap-2.5 px-4 py-5 lg:px-5">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-accent/15 text-accent">
          <Activity size={17} />
        </div>
        <div className="hidden min-w-0 lg:block">
          <p className="truncate text-[13px] font-semibold tracking-tight text-ink">Quant Console</p>
          <p className="truncate text-[11px] text-ink-tertiary">Operator</p>
        </div>
      </div>

      <div className="mt-2 flex flex-1 flex-col gap-1 px-2.5 lg:px-3">
        {ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                "group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] font-medium transition-colors",
                isActive
                  ? "bg-accent/14 text-accent"
                  : "text-ink-secondary hover:bg-hairline/[0.07] hover:text-ink",
              )
            }
            title={item.label}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-accent" />
                )}
                <span className="shrink-0">{item.icon}</span>
                <span className="hidden lg:inline">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </div>

      <button
        onClick={disconnect}
        title="Disconnect"
        className="m-2.5 flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] font-medium text-ink-tertiary transition-colors hover:bg-hairline/[0.07] hover:text-ink lg:m-3"
      >
        <LogOut size={18} className="shrink-0" />
        <span className="hidden lg:inline">Disconnect</span>
      </button>
    </nav>
  );
}
