#!/usr/bin/env python3
"""Lightweight repository hygiene checks for use from pre-commit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKED_EXTENSIONS = {
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {".git", ".venv", "data", "docs", "outputs", "notebooks", "article_images"}


def should_check(path: Path) -> bool:
    """Return whether a repository file should be checked for whitespace issues."""
    return path.suffix in CHECKED_EXTENSIONS and not any(part in SKIP_PARTS for part in path.parts)


def iter_paths(argv: list[str]) -> list[Path]:
    """Resolve the paths passed by pre-commit, or fall back to a repo scan."""
    if argv:
        return [ROOT / arg for arg in argv]
    return [path for path in ROOT.rglob("*") if path.is_file()]


def main(argv: list[str] | None = None) -> int:
    """Check for trailing whitespace and missing final newlines in text files."""
    if argv is None:
        argv = sys.argv[1:]

    failures: list[str] = []

    for path in iter_paths(argv):
        if not path.is_file() or not should_check(path):
            continue

        text = path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            failures.append(f"{path.relative_to(ROOT)}: missing trailing newline")
        if any(line.rstrip(" \t") != line for line in text.splitlines()):
            failures.append(f"{path.relative_to(ROOT)}: trailing whitespace")

    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
