"""Detect committed secrets while allowing documented placeholder values."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SCAN_ARG_CHARS = 24_000


def _git_executable() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found")
    return git


def _candidate_files() -> list[str]:
    result = subprocess.run(
        [_git_executable(), "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ignored_prefixes = (
        ".git/",
        ".venv/",
        ".venv-verify/",
        ".venv-verify-run/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".claude/",
        "data/",
    )
    ignored_suffixes = (
        ".egg-info/PKG-INFO",
        ".pyc",
    )
    files: list[str] = []
    for raw in result.stdout.splitlines():
        path = raw.strip()
        if not path or path == ".codex":
            continue
        if path.startswith(ignored_prefixes):
            continue
        if any(path.endswith(suffix) for suffix in ignored_suffixes):
            continue
        if (ROOT / path).is_file():
            files.append(path)
    return files


def _allowed_placeholder(path: str, line: str) -> bool:
    placeholders = (
        "POSTGRES_PASSWORD: quant",
        "POSTGRES_PASSWORD=change_me_before_running_compose",
        "postgresql+psycopg://quant:change_me_before_running_compose@localhost",
        "postgresql+psycopg://quant:quant@localhost",
        "postgresql+psycopg://quant:pw@db/quant_platform",
        "postgresql+psycopg://user:pass@host",
        "postgresql+psycopg://user:password@localhost",
        "postgresql://quant:pass@staging-host",
        "postgresql://user:pass@host",
        "postgresql://user:password@localhost",
        "rediss://user:pass@host",
        'monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")',
        "DEEPSEEK_API_KEY=sk-dotenv",
        '"api_key": "sk-dotenv"',
        '"api_key": "sk-test"',
        'os.environ["QP__API__OPERATOR_API_KEY"] = "test-key"',
        'settings.api.operator_api_key = "test-key"',
        'api_key = "e2e-test-key-" + uuid.uuid4().hex[:8]',
        'env["POSTGRES_PASSWORD"] = "quant"',
        'apiKey: "secret"',
        'apiKey: "bad"',
        "${{ secrets.IBKR_ACCOUNT_ID }}",
        "<paper-account>",
    )
    return any(token in line for token in placeholders)


def _file_chunks(files: list[str]) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for path in files:
        next_chars = len(path) + 1
        if current and current_chars + next_chars > MAX_SCAN_ARG_CHARS:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(path)
        current_chars += next_chars
    if current:
        chunks.append(current)
    return chunks


def _scan_files(files: list[str]) -> tuple[int, str, dict[str, list[dict[str, object]]]]:
    merged: dict[str, list[dict[str, object]]] = {}
    for chunk in _file_chunks(files):
        result = subprocess.run(
            [sys.executable, "-m", "detect_secrets", "scan", *chunk],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            return result.returncode, result.stderr, {}
        payload = json.loads(result.stdout or "{}")
        raw_results: dict[str, list[dict[str, object]]] = payload.get("results", {})
        for path, path_findings in raw_results.items():
            merged.setdefault(path, []).extend(path_findings)
    return 0, "", merged


def main() -> int:
    files = _candidate_files()
    if not files:
        print("Secret scan skipped: no candidate files.")
        return 0

    returncode, stderr, raw_results = _scan_files(files)
    if returncode != 0:
        sys.stderr.write(stderr)
        return returncode

    findings: dict[str, list[dict[str, object]]] = {}
    for path, path_findings in raw_results.items():
        lines = (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()
        for finding in path_findings:
            line_number = int(finding.get("line_number", 0))
            line = lines[line_number - 1] if 0 < line_number <= len(lines) else ""
            if _allowed_placeholder(path, line):
                continue
            findings.setdefault(path, []).append(finding)

    if findings:
        print("Potential secrets detected by detect-secrets:")
        for path, path_findings in findings.items():
            print(f"  {path}: {len(path_findings)} finding(s)")
            for finding in path_findings:
                kind = finding.get("type", "unknown")
                line = finding.get("line_number", "?")
                print(f"    - {kind} at line {line}")
        return 1

    print("Secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
