import { Navigate, Route, Routes } from "react-router-dom";
import { Connect } from "./components/Connect";
import { Shell } from "./components/layout/Shell";
import { useCapabilities } from "./lib/queries";
import Alpha from "./screens/Alpha";
import Backtest from "./screens/Backtest";
import Commands from "./screens/Commands";
import Execution from "./screens/Execution";
import Overview from "./screens/Overview";
import Research from "./screens/Research";
import Settings from "./screens/Settings";
import Strategy from "./screens/Strategy";
import System from "./screens/System";

function Splash() {
  return (
    <div className="grid min-h-screen place-items-center bg-base">
      <div className="flex flex-col items-center gap-4">
        <div className="h-9 w-9 animate-spin rounded-full border-2 border-hairline/20 border-t-accent" />
        <p className="text-sm text-ink-tertiary">Connecting…</p>
      </div>
    </div>
  );
}

export default function App() {
  const caps = useCapabilities();

  // Capabilities is the connection signal. Success → connected (works for both
  // key-auth and the unauthenticated escape hatch). Error → show Connect.
  if (caps.isSuccess) {
    return (
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Overview />} />
          <Route path="overview" element={<Overview />} />
          <Route path="strategy" element={<Strategy />} />
          <Route path="alpha" element={<Alpha />} />
          <Route path="execution" element={<Execution />} />
          <Route path="research" element={<Research />} />
          <Route path="backtest" element={<Backtest />} />
          <Route path="system" element={<System />} />
          <Route path="commands/*" element={<Commands />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Route>
      </Routes>
    );
  }
  if (caps.isLoading && caps.fetchStatus !== "idle") return <Splash />;
  return <Connect error={caps.error} loading={caps.isFetching} />;
}
