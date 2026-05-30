"""Introspect the argparse CLI into a JSON-serializable command catalog.

The catalog mirrors the entire ``python -m quant_platform`` surface — every
group, command, and argument — so the UI can render a form for any command and
the runner can reconstruct an argv from form values. No command is executed
here; this is pure metadata extraction.
"""

from __future__ import annotations

import argparse
from typing import Any

# Substrings that mark a command as state-mutating / financial → the UI demands
# a typed confirmation. This is a UX guardrail, not the security boundary (auth
# + the execution opt-in flag are). Groups broker/engines are dangerous wholesale.
_DANGER_SUBSTRINGS = (
    "migrate",
    "migration",
    "restore",
    "kill",
    "promote",
    "ingest",
    "supervise",
    "run-cycle",
    "run-engine",
    "run-multi",
    "serve-api",
    "reprice",
    "backup",
    "delete",
    "purge",
    "drop",
)
_DANGER_GROUPS = {"broker", "engines"}
# Irreversibly destructive verbs. These are NEVER downgraded by the read-only
# override below, so a command like "delete-stale --list" can't masquerade as
# safe. Chosen to not appear as substrings of known read-only command names
# (e.g. "migrate" is not in "migrations-check"; "promote" is not in "promotion").
_STRONG_DANGER_SUBSTRINGS = (
    "delete",
    "drop",
    "purge",
    "restore",
    "kill",
    "migrate",
    "promote",
)
# Read-only verbs that override the danger heuristic (a "*-check"/"list"/
# "smoke" command is safe even if its group or name brushes a danger substring).
_SAFE_SUBSTRINGS = (
    "check",
    "verify",
    "validate",
    "list",
    "status",
    "diagnostic",
    "smoke",
    "gpu-check",
    "tearsheet",
    "health",
    "report",
)
# Commands that do not return on their own — run until cancelled.
_LONG_RUNNING = ("supervise", "serve-api", "maintain", "maintenance", "run-multi")


def _is_dangerous(group: str, path: list[str]) -> bool:
    joined = " ".join(path).lower()
    # Strong-danger verbs win over the read-only override; everything else can
    # be downgraded by a read-only verb in the name.
    if any(s in joined for s in _STRONG_DANGER_SUBSTRINGS):
        return True
    if any(s in joined for s in _SAFE_SUBSTRINGS):
        return False
    if group in _DANGER_GROUPS:
        return True
    return any(s in joined for s in _DANGER_SUBSTRINGS)


def _is_long_running(path: list[str]) -> bool:
    joined = " ".join(path).lower()
    return any(s in joined for s in _LONG_RUNNING)


def _type_label(action: argparse.Action) -> str:
    t = action.type
    if t is None:
        return "str"
    name = getattr(t, "__name__", "") or str(t)
    mapping = {"int": "int", "float": "float", "Decimal": "decimal", "str": "str"}
    return mapping.get(name, "str")


def _action_kind(action: argparse.Action) -> str:
    cls = type(action).__name__
    if cls in ("_StoreTrueAction", "_StoreFalseAction"):
        return "flag"
    if cls == "_AppendAction":
        return "append"
    if cls == "_CountAction":
        return "count"
    return "store"


def _arg_to_dict(action: argparse.Action) -> dict[str, Any] | None:
    # Skip the auto-added help action.
    if isinstance(action, argparse._HelpAction):  # noqa: SLF001 - stable argparse API
        return None
    default = action.default
    if default is argparse.SUPPRESS:
        default = None
    # A list/tuple default (nargs='+') must round-trip as the comma-separated
    # form the UI edits, not its Python repr ("[5, 10, 21]" would fail re-parse).
    if isinstance(default, (list, tuple)):
        default = ",".join(str(x) for x in default)
    kind = _action_kind(action)
    return {
        "dest": action.dest,
        "option_strings": list(action.option_strings),
        "positional": not action.option_strings,
        "kind": kind,
        "type": "bool" if kind == "flag" else _type_label(action),
        "required": bool(getattr(action, "required", False)),
        "choices": [str(c) for c in action.choices] if action.choices else None,
        # For a flag this is the dest's resting value (store_true → False,
        # store_false → True); the form toggles it and the option is emitted
        # only when the chosen value differs from this default.
        "default": None if default is None else _jsonable(default),
        "nargs": action.nargs if isinstance(action.nargs, str) else None,
        "help": action.help or "",
        "metavar": action.metavar if isinstance(action.metavar, str) else None,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:  # noqa: SLF001 - stable argparse API
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _help_map(sub: argparse._SubParsersAction) -> dict[str, str]:
    out: dict[str, str] = {}
    for choice in getattr(sub, "_choices_actions", []):  # noqa: SLF001
        out[choice.dest] = choice.help or ""
    return out


def _walk(
    name: str,
    parser: argparse.ArgumentParser,
    *,
    group: str,
    path: list[str],
    help_text: str,
) -> dict[str, Any]:
    sub = _subparsers_action(parser)
    if sub is not None:
        helps = _help_map(sub)
        children = [
            _walk(
                child_name,
                child_parser,
                group=group,
                path=[*path, child_name],
                help_text=helps.get(child_name, ""),
            )
            for child_name, child_parser in sub.choices.items()
        ]
        return {
            "name": name,
            "type": "group",
            "help": help_text or (parser.description or ""),
            "path": path,
            "commands": children,
        }
    args = [d for action in parser._actions if (d := _arg_to_dict(action)) is not None]  # noqa: SLF001
    return {
        "name": name,
        "type": "command",
        "help": help_text or (parser.description or ""),
        "path": path,
        "args": args,
        "dangerous": _is_dangerous(group, path),
        "long_running": _is_long_running(path),
    }


def build_command_catalog() -> dict[str, Any]:
    """Return the full CLI surface grouped by command group."""
    # Imported lazily so importing this module never triggers CLI registration.
    from quant_platform.cli.commands import all_command_specs

    groups: list[dict[str, Any]] = []
    for spec in all_command_specs():
        tmp = argparse.ArgumentParser(add_help=False)
        sub = tmp.add_subparsers()
        spec.register(sub)
        helps = _help_map(sub)
        commands = [
            _walk(name, parser, group=spec.name, path=[name], help_text=helps.get(name, ""))
            for name, parser in sub.choices.items()
        ]
        if commands:
            groups.append({"name": spec.name, "commands": commands})
    return {"groups": groups}


def _flatten(node: dict[str, Any], acc: dict[str, dict[str, Any]]) -> None:
    if node["type"] == "command":
        acc[" ".join(node["path"])] = node
        return
    for child in node.get("commands", []):
        _flatten(child, acc)


def find_command(path: list[str], catalog: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Locate a leaf command node by its path (e.g. ['features', 'backfill']).

    Pass a pre-built ``catalog`` to avoid re-introspecting the whole CLI.
    """
    catalog = catalog if catalog is not None else build_command_catalog()
    index: dict[str, dict[str, Any]] = {}
    for group in catalog["groups"]:
        for command in group["commands"]:
            _flatten(command, index)
    return index.get(" ".join(path))


def reconstruct_argv(command: dict[str, Any], values: dict[str, Any]) -> list[str]:
    """Build a CLI argv from a leaf command node and submitted form values."""
    argv: list[str] = list(command["path"])
    positionals: list[str] = []
    for arg in command["args"]:
        dest = arg["dest"]
        if dest not in values or values[dest] is None or values[dest] == "":
            continue
        raw = values[dest]
        if arg["positional"]:
            if isinstance(raw, list):
                positionals.extend(str(v) for v in raw)
            else:
                positionals.append(str(raw))
            continue
        opt = _preferred_option(arg["option_strings"])
        if arg["kind"] == "flag":
            # Emit the option only when the chosen value flips the dest away
            # from its default — handles both store_true and store_false.
            if bool(raw) != bool(arg.get("default")):
                argv.append(opt)
        elif arg["kind"] == "append":
            # Repeat the option once per value: --tag a --tag b.
            for v in raw if isinstance(raw, list) else [raw]:
                argv.extend([opt, str(v)])
        elif isinstance(raw, list) or arg.get("nargs") in ("+", "*"):
            # nargs multiple: one option followed by every value: --days 5 10 21.
            argv.append(opt)
            argv.extend(str(v) for v in (raw if isinstance(raw, list) else [raw]))
        else:
            argv.extend([opt, str(raw)])
    argv.extend(positionals)
    return argv


def _preferred_option(option_strings: list[str]) -> str:
    for opt in option_strings:
        if opt.startswith("--"):
            return opt
    return option_strings[0]
