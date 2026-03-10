#!/usr/bin/env python3
"""
Phase 5 inference scaffold for reconstruction experiments.

This module focuses on adapter metadata loading and smoke-safe inference plumbing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_saved_adapter(checkpoint_metadata_path: Path) -> dict[str, Any]:
    """Load adapter metadata saved by the Phase 5 training scaffold."""
    payload = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    return {
        "run_id": payload["run_id"],
        "git_sha": payload["git_sha"],
        "phase": payload["phase"],
        "model_id": payload["model_id"],
        "adapter_type": payload["adapter_type"],
        "adapter_artifact_path": payload["adapter_artifact_path"],
        "config_path": payload["config_path"],
        "tokenizer_config_path": payload["tokenizer_config_path"],
        "metrics_path": payload["metrics_path"],
        "split_counts": payload["split_counts"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the Phase 5 inference scaffold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-metadata-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load adapter metadata and print a minimal confirmation."""
    args = build_argument_parser().parse_args(argv)
    payload = load_saved_adapter(args.checkpoint_metadata_path)
    print("Loaded adapter metadata")
    print(f"  Run ID: {payload['run_id']}")
    print(f"  Model:  {payload['model_id']}")
    print(f"  Adapter: {payload['adapter_artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
