import { ShieldAlert } from "lucide-react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

export function KillSwitchBanner({ reason }: { reason: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      className="border-b border-danger/30 bg-danger/12"
    >
      <div className="mx-auto flex max-w-[1320px] items-center gap-3 px-5 py-2.5 sm:px-8">
        <ShieldAlert size={17} className="shrink-0 text-danger" />
        <p className="min-w-0 flex-1 text-[13px] text-ink">
          <span className="font-semibold text-danger">Kill switch active.</span>{" "}
          <span className="text-ink-secondary">
            Order submission is halted{reason ? ` — ${reason}` : ""}.
          </span>
        </p>
        <Link
          to="/execution"
          className="shrink-0 rounded-lg px-2.5 py-1 text-[13px] font-medium text-danger hover:bg-danger/10"
        >
          Manage →
        </Link>
      </div>
    </motion.div>
  );
}
