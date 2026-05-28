"""Reject generated artifacts inside source-controlled project paths."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "tests", "scripts")
GENERATED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
GENERATED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class GeneratedArtifact:
    path: Path
    kind: str

    @property
    def display_path(self) -> str:
        return self.path.relative_to(ROOT).as_posix()


def collect_generated_artifacts(root: Path = ROOT) -> list[GeneratedArtifact]:
    artifacts: list[GeneratedArtifact] = []
    for scan_name in SCAN_ROOTS:
        scan_root = root / scan_name
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*")):
            if path.name in GENERATED_DIR_NAMES and path.is_dir():
                artifacts.append(GeneratedArtifact(path=path, kind="directory"))
            elif path.is_file() and path.suffix in GENERATED_SUFFIXES:
                artifacts.append(GeneratedArtifact(path=path, kind="file"))
    return artifacts


def clean_generated_artifacts(artifacts: list[GeneratedArtifact]) -> None:
    for artifact in artifacts:
        if artifact.path.is_dir():
            shutil.rmtree(artifact.path)
        elif artifact.path.exists():
            artifact.path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete generated cache artifacts before reporting remaining issues",
    )
    args = parser.parse_args(argv)

    artifacts = collect_generated_artifacts()
    if args.clean and artifacts:
        clean_generated_artifacts(artifacts)
        artifacts = collect_generated_artifacts()

    if artifacts:
        print("Generated artifacts found in source-controlled paths:", file=sys.stderr)
        for artifact in artifacts:
            print(f"  - {artifact.display_path} ({artifact.kind})", file=sys.stderr)
        print("Run scripts/check_generated_artifacts.py --clean to remove them.", file=sys.stderr)
        return 1

    print("Generated-artifact hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
