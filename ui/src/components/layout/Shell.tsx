import { Outlet } from "react-router-dom";
import { useDashboard } from "../../lib/queries";
import { isErr } from "../../lib/types";
import { KillSwitchBanner } from "./KillSwitchBanner";
import { NavRail } from "./NavRail";
import { TopBar } from "./TopBar";

export function Shell() {
  const dash = useDashboard();
  const data = dash.data;

  const killActive =
    (data && !isErr(data.health) && data.health.kill_switch_active) ||
    (data?.kill_switch?.active ?? false);
  const killReason =
    (data && !isErr(data.health) && data.health.kill_switch_reason) ||
    data?.kill_switch?.reason ||
    "";

  return (
    <div className="relative flex h-screen overflow-hidden bg-base text-ink">
      {/* Ambient depth — fixed, subtle, behind everything */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div
          className="absolute -left-40 -top-40 h-[40rem] w-[40rem] rounded-full opacity-[0.10] blur-3xl"
          style={{ background: "radial-gradient(closest-side, rgb(var(--ambient-1)), transparent)" }}
        />
        <div
          className="absolute -bottom-52 -right-40 h-[42rem] w-[42rem] rounded-full opacity-[0.08] blur-3xl"
          style={{ background: "radial-gradient(closest-side, rgb(var(--ambient-2)), transparent)" }}
        />
      </div>

      <NavRail />

      <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
        <TopBar
          asOf={data?.as_of ?? null}
          health={data && !isErr(data.health) ? data.health : null}
          stale={dash.isError}
        />
        {killActive && <KillSwitchBanner reason={killReason} />}
        <main className="min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-8">
          <div className="mx-auto w-full max-w-[1320px]">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
