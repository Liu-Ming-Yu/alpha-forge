import type { CommandArg, CommandCatalog, CommandNode } from "./types";

export type CommandValues = Record<string, string | boolean>;

export interface CommandLeaf {
  group: string;
  node: CommandNode;
}

export function flattenCommandCatalog(catalog: CommandCatalog): CommandLeaf[] {
  const out: CommandLeaf[] = [];
  for (const group of catalog.groups) {
    const walk = (node: CommandNode) => {
      if (node.type === "command") out.push({ group: group.name, node });
      else node.commands?.forEach(walk);
    };
    group.commands.forEach(walk);
  }
  return out;
}

export function findCommandArg(command: CommandNode | null | undefined, dest: string): CommandArg | null {
  return command?.args?.find((arg) => arg.dest === dest) ?? null;
}

export function findCommandByArgSignature(
  catalog: CommandCatalog | null | undefined,
  signature: {
    requiredDests: readonly string[];
    preferredDests?: readonly string[];
  },
): CommandNode | null {
  if (!catalog) return null;
  const required = new Set(signature.requiredDests);
  const preferred = signature.preferredDests ?? [];
  const candidates = flattenCommandCatalog(catalog)
    .map(({ node }) => {
      const dests = new Set((node.args ?? []).map((arg) => arg.dest));
      const hasRequired = [...required].every((dest) => dests.has(dest));
      if (!hasRequired) return null;
      const score = preferred.reduce((total, dest) => total + (dests.has(dest) ? 1 : 0), 0);
      return { node, score };
    })
    .filter((candidate): candidate is { node: CommandNode; score: number } => candidate !== null)
    .sort((a, b) => b.score - a.score);
  return candidates[0]?.node ?? null;
}

export function initCommandValues(command: CommandNode): CommandValues {
  const values: CommandValues = {};
  for (const arg of command.args ?? []) {
    if (arg.kind === "flag") values[arg.dest] = Boolean(arg.default);
    else values[arg.dest] = arg.default == null ? "" : String(arg.default);
  }
  return values;
}

export function isListCommandArg(arg: CommandArg): boolean {
  return arg.kind === "append" || arg.nargs === "+" || arg.nargs === "*";
}

export function buildCommandPayload(command: CommandNode, values: CommandValues): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const arg of command.args ?? []) {
    const value = values[arg.dest];
    if (arg.kind === "flag") {
      out[arg.dest] = Boolean(value);
    } else if (value === "" || value == null) {
      continue;
    } else if (isListCommandArg(arg)) {
      out[arg.dest] = splitListValue(value);
    } else {
      out[arg.dest] = value;
    }
  }
  return out;
}

export function preferredCommandOption(arg: CommandArg): string | null {
  return arg.option_strings.find((option) => option.startsWith("--")) ?? arg.option_strings[0] ?? null;
}

export function splitListValue(value: string | boolean): string[] {
  return String(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function previewCommandArgv(command: CommandNode, values: CommandValues): string {
  const parts = [...command.path];
  for (const arg of command.args ?? []) {
    const value = values[arg.dest];
    const option = preferredCommandOption(arg);
    if (arg.kind === "flag") {
      if (Boolean(value) !== Boolean(arg.default)) parts.push(arg.option_strings[0] ?? `--${arg.dest}`);
    } else if (value === "" || value == null) {
      continue;
    } else if (arg.positional) {
      parts.push(...(isListCommandArg(arg) ? splitListValue(value) : [String(value)]));
    } else if (arg.kind === "append" && option) {
      for (const item of splitListValue(value)) parts.push(option, item);
    } else if (isListCommandArg(arg) && option) {
      parts.push(option, ...splitListValue(value));
    } else if (option) {
      parts.push(option, String(value));
    }
  }
  return `python -m quant_platform ${parts.join(" ")}`;
}

export function isMissingRequiredCommandArg(arg: CommandArg, value: string | boolean): boolean {
  if (!arg.required || arg.kind === "flag") return false;
  return (value === "" || value == null) && arg.default == null;
}
