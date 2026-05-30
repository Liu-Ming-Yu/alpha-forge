import { cn } from "../../lib/cn";
import {
  buildCommandPayload,
  initCommandValues,
  isListCommandArg,
  isMissingRequiredCommandArg,
  preferredCommandOption,
  previewCommandArgv,
  type CommandValues,
} from "../../lib/commands";
import { titleCase } from "../../lib/format";
import type { CommandArg } from "../../lib/types";

export type Values = CommandValues;
export const initValues = initCommandValues;
export const buildPayload = buildCommandPayload;
export const previewArgv = previewCommandArgv;
export const isMissingRequired = isMissingRequiredCommandArg;
export const isListArg = isListCommandArg;

const inputCls =
  "w-full rounded-lg border bg-base/70 px-3 py-2 text-[13px] text-ink outline-none transition placeholder:text-ink-tertiary/70 focus:border-accent/60 focus:bg-base/90";

function preferredOption(arg: CommandArg): string | null {
  return preferredCommandOption(arg);
}

function fieldKindLabel(arg: CommandArg): string {
  if (arg.kind === "flag") return "toggle";
  if (arg.choices) return "menu";
  if (isListArg(arg)) return "list";
  if (arg.positional) return "positional";
  return arg.type;
}

function defaultText(arg: CommandArg): string | null {
  if (arg.default === null || arg.default === undefined || arg.default === "") return null;
  return String(arg.default);
}

export function Field({
  arg,
  value,
  onChange,
  invalid,
}: {
  arg: CommandArg;
  value: string | boolean;
  onChange: (v: string | boolean) => void;
  invalid?: boolean;
}) {
  const border = invalid ? "border-danger/60" : "border-hairline/10";
  const opt = preferredOption(arg);
  const defaultValue = defaultText(arg);
  return (
    <label className="block rounded-xl border border-hairline/[0.06] bg-base/20 p-3">
      <div className="mb-2 flex min-w-0 flex-wrap items-center gap-1.5">
        <span className="mr-auto min-w-0 truncate text-[13px] font-semibold text-ink-secondary">
          {titleCase(arg.dest)}
        </span>
        {arg.required && (
          <span className="rounded bg-danger/12 px-1.5 py-0.5 text-[10px] font-semibold text-danger">
            Required
          </span>
        )}
        <span className="rounded bg-hairline/[0.08] px-1.5 py-0.5 text-[10px] font-medium text-ink-tertiary">
          {fieldKindLabel(arg)}
        </span>
      </div>
      {arg.kind === "flag" ? (
        <button
          type="button"
          role="switch"
          aria-checked={Boolean(value)}
          onClick={() => onChange(!value)}
          className={cn(
            "flex w-full items-center justify-between rounded-lg border px-3 py-2 transition-colors",
            border,
            value ? "bg-success/10 text-success" : "bg-base/70 text-ink-tertiary hover:text-ink-secondary",
          )}
        >
          <span className="text-[13px] font-medium">{value ? "Enabled" : "Disabled"}</span>
          <span
            className={cn(
              "relative h-6 w-11 rounded-full transition-colors",
              value ? "bg-success" : "bg-hairline/20",
            )}
          >
            <span
              className={cn(
                "absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform",
                value ? "translate-x-5" : "translate-x-0.5",
              )}
            />
          </span>
        </button>
      ) : arg.choices ? (
        <select
          aria-invalid={invalid || undefined}
          className={cn(inputCls, border)}
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">{defaultValue ? `Default (${defaultValue})` : "Default"}</option>
          {arg.choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      ) : (
        <input
          aria-invalid={invalid || undefined}
          className={cn(inputCls, border, "font-mono")}
          inputMode={arg.type === "int" || arg.type === "float" || arg.type === "decimal" ? "decimal" : "text"}
          value={String(value)}
          placeholder={isListArg(arg) ? "comma, separated values" : (arg.metavar ?? arg.type)}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      {arg.help && <p className="mt-2 text-[11.5px] leading-snug text-ink-tertiary">{arg.help}</p>}
      <div className="mt-2 flex min-w-0 flex-wrap items-center gap-2 text-[10.5px] text-ink-tertiary">
        {opt && (
          <code className="rounded bg-hairline/[0.07] px-1.5 py-0.5 font-mono text-[10.5px] text-ink-secondary">
            {opt}
          </code>
        )}
        {defaultValue && <span>Default: {defaultValue}</span>}
        {isListArg(arg) && <span>Use commas between values</span>}
      </div>
    </label>
  );
}
