#!/usr/bin/env python3
"""
Experiment contract and filesystem policy for Part 3 reconstruction runs.

Phase 0 hardens the pipeline before any generation or training starts.
Every run gets a reproducible manifest under ``outputs/reconstruction/``,
all paths stay project-relative, and failed runs remain intact for review.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECONSTRUCTION_SEED = 42

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FILENAME = "manifest.json"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "run_id",
    "phase",
    "status",
    "created_at",
    "updated_at",
    "git_sha",
    "model_id",
    "prompt_template_id",
    "seed",
    "config_hash",
    "corpus_manifest",
    "split_manifest",
    "paths",
    "config_payload",
    "error_message",
)


class RunStatus(StrEnum):
    """Lifecycle states for a reconstruction run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SeedBundle:
    """Deterministic seeds propagated to every component."""

    python: int
    numpy: int
    torch: int
    data_splitter: int


@dataclass(frozen=True)
class ManifestPaths:
    """Project-relative manifest locations saved in the run record."""

    run_dir: str
    manifest_path: str
    indexed_manifest_path: str


@dataclass(frozen=True)
class RunManifest:
    """Serialized metadata contract for one reconstruction run."""

    schema_version: str
    run_id: str
    phase: str
    status: RunStatus
    created_at: str
    updated_at: str
    git_sha: str
    model_id: str
    prompt_template_id: str | None
    seed: SeedBundle
    config_hash: str
    corpus_manifest: str
    split_manifest: str | None
    paths: ManifestPaths
    config_payload: dict[str, Any]
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the manifest."""
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class ReconstructionPaths:
    """Centralized path builder for the reconstruction workspace."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    outputs_root: Path = field(init=False)
    reconstruction_root: Path = field(init=False)
    manifests_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    baselines_dir: Path = field(init=False)
    pilots_dir: Path = field(init=False)
    analysis_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        project_root = self.project_root.resolve()
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "outputs_root", project_root / "outputs")
        reconstruction_root = self.outputs_root / "reconstruction"
        object.__setattr__(self, "reconstruction_root", reconstruction_root)
        object.__setattr__(self, "manifests_dir", reconstruction_root / "manifests")
        object.__setattr__(self, "runs_dir", reconstruction_root / "runs")
        object.__setattr__(self, "baselines_dir", reconstruction_root / "baselines")
        object.__setattr__(self, "pilots_dir", reconstruction_root / "pilots")
        object.__setattr__(self, "analysis_dir", reconstruction_root / "analysis")

    def ensure_root_directories(self) -> None:
        """Create the immutable reconstruction directory scaffold."""
        for directory in (
            self.reconstruction_root,
            self.manifests_dir,
            self.runs_dir,
            self.baselines_dir,
            self.pilots_dir,
            self.analysis_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        """Return the per-run directory and reject unsafe identifiers."""
        validated_run_id = validate_run_id(run_id)
        return ensure_project_relative(self.runs_dir / validated_run_id, self.project_root)

    def manifest_path(self, run_id: str) -> Path:
        """Return the manifest path inside a run directory."""
        return ensure_project_relative(self.run_dir(run_id) / MANIFEST_FILENAME, self.project_root)

    def indexed_manifest_path(self, run_id: str) -> Path:
        """Return the top-level manifest index path for quick lookup."""
        filename = f"{validate_run_id(run_id)}.json"
        return ensure_project_relative(self.manifests_dir / filename, self.project_root)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def validate_run_id(run_id: str) -> str:
    """Validate the run identifier used in filesystem paths."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must match the safe pattern "
            f"{RUN_ID_PATTERN.pattern!r}; got {run_id!r}"
        )
    return run_id


def ensure_project_relative(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Reject paths that escape the repository root."""
    resolved_root = project_root.resolve()
    resolved_path = path.resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {path}") from exc

    return resolved_path


def to_project_relative(path: Path | str, project_root: Path = PROJECT_ROOT) -> str:
    """Convert an absolute or repo-relative path into a project-relative string."""
    path_obj = Path(path)
    candidate = path_obj if path_obj.is_absolute() else project_root / path_obj
    safe_path = ensure_project_relative(candidate, project_root)
    return safe_path.relative_to(project_root.resolve()).as_posix()


def coerce_seed_bundle(seed: int | SeedBundle) -> SeedBundle:
    """Normalize a single integer seed into a full component seed bundle."""
    if isinstance(seed, SeedBundle):
        return seed
    return SeedBundle(
        python=seed,
        numpy=seed,
        torch=seed,
        data_splitter=seed,
    )


def _load_torch_module() -> Any | None:
    """Load torch only when available so Phase 0 stays CPU-friendly."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def seed_everything(seed: int | SeedBundle, torch_module: Any | None = None) -> SeedBundle:
    """Seed Python, NumPy, torch, and splitter metadata deterministically."""
    bundle = coerce_seed_bundle(seed)

    os.environ["PYTHONHASHSEED"] = str(bundle.python)
    random.seed(bundle.python)
    np.random.seed(bundle.numpy)

    resolved_torch = _load_torch_module() if torch_module is None else torch_module
    if resolved_torch is not None:
        resolved_torch.manual_seed(bundle.torch)

        cuda_module = getattr(resolved_torch, "cuda", None)
        if cuda_module is not None and hasattr(cuda_module, "manual_seed_all"):
            cuda_module.manual_seed_all(bundle.torch)

        if hasattr(resolved_torch, "use_deterministic_algorithms"):
            resolved_torch.use_deterministic_algorithms(True)

        backends = getattr(resolved_torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None) if backends is not None else None
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False

    return bundle


def hash_config_payload(config_payload: dict[str, Any]) -> str:
    """Hash a config payload so run settings can be compared exactly."""
    canonical_json = json.dumps(
        config_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical_json.encode("utf-8")).hexdigest()


def prepare_run_directory(run_id: str, paths: ReconstructionPaths | None = None) -> Path:
    """Create a new immutable run directory and reject accidental reuse."""
    resolved_paths = paths or ReconstructionPaths()
    resolved_paths.ensure_root_directories()
    run_dir = resolved_paths.run_dir(run_id)

    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists and will not be overwritten: {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_run_manifest(
    run_id: str,
    phase: str,
    model_id: str,
    seed: int | SeedBundle,
    git_sha: str,
    config_payload: dict[str, Any],
    corpus_manifest: Path | str,
    prompt_template_id: str | None = None,
    split_manifest: Path | str | None = None,
    status: RunStatus = RunStatus.RUNNING,
    paths: ReconstructionPaths | None = None,
) -> RunManifest:
    """Build the canonical run manifest payload."""
    resolved_paths = paths or ReconstructionPaths()
    bundle = coerce_seed_bundle(seed)
    created_at = utc_now()

    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=validate_run_id(run_id),
        phase=phase,
        status=status,
        created_at=created_at,
        updated_at=created_at,
        git_sha=git_sha,
        model_id=model_id,
        prompt_template_id=prompt_template_id,
        seed=bundle,
        config_hash=hash_config_payload(config_payload),
        corpus_manifest=to_project_relative(corpus_manifest, resolved_paths.project_root),
        split_manifest=(
            None
            if split_manifest is None
            else to_project_relative(split_manifest, resolved_paths.project_root)
        ),
        paths=ManifestPaths(
            run_dir=to_project_relative(
                resolved_paths.run_dir(run_id),
                resolved_paths.project_root,
            ),
            manifest_path=to_project_relative(
                resolved_paths.manifest_path(run_id),
                resolved_paths.project_root,
            ),
            indexed_manifest_path=to_project_relative(
                resolved_paths.indexed_manifest_path(run_id),
                resolved_paths.project_root,
            ),
        ),
        config_payload=config_payload,
    )


def write_run_manifest(
    manifest: RunManifest,
    paths: ReconstructionPaths | None = None,
) -> Path:
    """Write a new run manifest to both the run directory and the manifest index."""
    resolved_paths = paths or ReconstructionPaths()
    resolved_paths.ensure_root_directories()

    run_manifest_path = resolved_paths.manifest_path(manifest.run_id)
    run_dir = run_manifest_path.parent
    indexed_manifest_path = resolved_paths.indexed_manifest_path(manifest.run_id)

    if not run_dir.exists():
        raise FileNotFoundError(
            f"run directory does not exist for {manifest.run_id!r}: {run_dir}. "
            "Call prepare_run_directory() before write_run_manifest()."
        )
    if run_manifest_path.exists():
        raise FileExistsError(
            f"run manifest already exists for {manifest.run_id!r}: {run_manifest_path}"
        )
    if indexed_manifest_path.exists():
        raise FileExistsError(
            f"indexed run manifest already exists for {manifest.run_id!r}: {indexed_manifest_path}"
        )

    payload = manifest.to_dict()
    _write_json(run_manifest_path, payload)
    _write_json(indexed_manifest_path, payload)
    return run_manifest_path


def finalize_run_manifest(
    run_id: str,
    status: RunStatus,
    paths: ReconstructionPaths | None = None,
    error_message: str | None = None,
) -> Path:
    """Update a manifest status without deleting any run artifacts."""
    resolved_paths = paths or ReconstructionPaths()
    manifest_path = resolved_paths.manifest_path(run_id)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = status.value
    payload["updated_at"] = utc_now()
    payload["error_message"] = error_message

    _write_json(manifest_path, payload)
    _write_json(resolved_paths.indexed_manifest_path(run_id), payload)
    return manifest_path


def detect_git_sha(project_root: Path = PROJECT_ROOT) -> str:
    """Return the current git SHA or ``unknown`` when unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON with deterministic key ordering."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_default_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Construct the minimal dry-run config payload stored in the manifest."""
    return {
        "dry_run": True,
        "phase": args.phase,
        "seed": args.seed,
        "model_id": args.model_id,
        "prompt_template_id": args.prompt_template_id,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for a non-generative dry-run manifest write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument("--phase", required=True, help="Phase label saved in the manifest.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RECONSTRUCTION_SEED,
        help="Deterministic seed propagated to Python, NumPy, torch, and splitters.",
    )
    parser.add_argument(
        "--model-id",
        default="no-generation",
        help="Model identifier recorded in the manifest.",
    )
    parser.add_argument(
        "--prompt-template-id",
        default=None,
        help="Prompt template identifier recorded in the manifest.",
    )
    parser.add_argument(
        "--corpus-manifest",
        default="outputs/corpus/corpus_metadata.json",
        help="Project-relative path to the corpus manifest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write a dry-run manifest without launching any generation step."""
    args = parse_args(argv)
    paths = ReconstructionPaths()

    seed_everything(args.seed)
    prepare_run_directory(args.run_id, paths=paths)

    manifest = build_run_manifest(
        run_id=args.run_id,
        phase=args.phase,
        model_id=args.model_id,
        seed=args.seed,
        git_sha=detect_git_sha(paths.project_root),
        config_payload=build_default_config_payload(args),
        corpus_manifest=args.corpus_manifest,
        prompt_template_id=args.prompt_template_id,
        paths=paths,
    )
    manifest_path = write_run_manifest(manifest, paths=paths)
    print(f"Dry-run manifest written to {to_project_relative(manifest_path, paths.project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
