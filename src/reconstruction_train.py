#!/usr/bin/env python3
"""
Phase 5 training scaffold for reconstruction experiments.

This module intentionally starts with a smoke-safe training path that prepares
all artifacts needed for a real fine-tuning run without pretending the model
has already been trained.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reconstruction_contract import (
    DEFAULT_RECONSTRUCTION_SEED,
    ReconstructionPaths,
    RunStatus,
    build_run_manifest,
    detect_git_sha,
    finalize_run_manifest,
    prepare_run_directory,
    seed_everything,
    to_project_relative,
    write_run_manifest,
)
from reconstruction_dataset import SplitManifest, TargetEnvelope, WindowRecord, extract_windows
from reconstruction_dataset import _load_json as _load_pilot_json
from reconstruction_metrics import ToleranceConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "google/mt5-xl"
DEFAULT_DATASET_MODE = "identity_smoke"


@dataclass(frozen=True)
class TrainingExample:
    """One training example aligned to the pilot split manifest."""

    window_id: str
    split: str
    source_text: str
    target_text: str
    target_envelope_id: str
    dataset_mode: str


@dataclass(frozen=True)
class TrainingConfig:
    """Serialized training configuration for Phase 5 scaffolding."""

    run_id: str
    model_id: str
    dataset_mode: str
    seed: int
    wandb_project: str | None
    wandb_entity: str | None
    wandb_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata recorded alongside a training checkpoint or adapter."""

    run_id: str
    git_sha: str
    phase: str
    model_id: str
    adapter_type: str
    adapter_artifact_path: str
    config_path: str
    tokenizer_config_path: str
    metrics_path: str
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class ExperimentLogger:
    """Optional experiment logger with a W&B-compatible surface."""

    def __init__(self, run: Any | None) -> None:
        self._run = run

    def log_metrics(self, payload: dict[str, float]) -> None:
        """Log metrics for the active run, if any."""
        if self._run is None:
            return
        self._run.log(payload)

    def finish(self) -> None:
        """Finalize the run, if any."""
        if self._run is None:
            return
        self._run.finish()


def _load_wandb_module() -> Any:
    """Load wandb only when requested."""
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Weights & Biases requested but `wandb` is not installed. "
            "Install it with `.venv/bin/python -m pip install wandb`."
        ) from exc


def build_experiment_logger(
    *,
    config: TrainingConfig,
    git_sha: str,
    split_counts: dict[str, int],
) -> ExperimentLogger:
    """Return a logger that optionally emits metrics to W&B."""
    if not config.wandb_project:
        return ExperimentLogger(None)

    wandb = _load_wandb_module()
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        config={
            "run_id": config.run_id,
            "model_id": config.model_id,
            "dataset_mode": config.dataset_mode,
            "seed": config.seed,
            "git_sha": git_sha,
            "split_counts": split_counts,
        },
        name=config.run_id,
    )
    return ExperimentLogger(run)


def load_split_manifest(path: Path) -> SplitManifest:
    """Load a split manifest from disk."""
    payload = _load_pilot_json(path)
    return SplitManifest(
        generated_at=str(payload["generated_at"]),
        seed=int(payload["seed"]),
        min_words=int(payload["min_words"]),
        max_words=int(payload["max_words"]),
        train_ratio=float(payload["train_ratio"]),
        val_ratio=float(payload["val_ratio"]),
        total_windows=int(payload["total_windows"]),
        split_counts={str(k): int(v) for k, v in payload["split_counts"].items()},
        assignments=tuple(payload["assignments"]),
        leakage_issues=tuple(payload.get("leakage_issues", [])),
    )


def load_target_envelopes(path: Path) -> list[TargetEnvelope]:
    """Load target envelopes from disk."""
    payload = _load_pilot_json(path)
    envelopes = []
    for record in payload["target_envelopes"]:
        envelopes.append(
            TargetEnvelope(
                envelope_id=str(record["envelope_id"]),
                work_id=str(record["work_id"]),
                author=str(record["author"]),
                title=str(record["title"]),
                aggregation_rule=str(record["aggregation_rule"]),
                provenance_window_ids=tuple(record["provenance_window_ids"]),
                provenance_segment_ids=tuple(record["provenance_segment_ids"]),
                stylometric_target={
                    str(k): float(v) for k, v in record["stylometric_target"].items()
                },
                semantic_reference={
                    str(k): float(v) for k, v in record["semantic_reference"].items()
                },
            )
        )
    return envelopes


def build_training_examples(
    windows: list[WindowRecord],
    split_manifest: SplitManifest,
    target_envelopes: list[TargetEnvelope],
    *,
    dataset_mode: str = DEFAULT_DATASET_MODE,
) -> list[TrainingExample]:
    """Build deterministic training examples aligned to the split manifest."""
    assignment_lookup = split_manifest.assignment_lookup()
    default_envelope_id = target_envelopes[0].envelope_id if target_envelopes else "target:unknown"
    examples: list[TrainingExample] = []
    for window in windows:
        split = assignment_lookup[window.window_id]
        source_text = window.text
        target_text = window.text if dataset_mode == "identity_smoke" else window.text
        examples.append(
            TrainingExample(
                window_id=window.window_id,
                split=split,
                source_text=source_text,
                target_text=target_text,
                target_envelope_id=default_envelope_id,
                dataset_mode=dataset_mode,
            )
        )
    return examples


def count_examples_by_split(examples: list[TrainingExample]) -> dict[str, int]:
    """Return counts per split for a list of training examples."""
    counts = {"train": 0, "val": 0, "test": 0}
    for example in examples:
        counts[example.split] = counts.get(example.split, 0) + 1
    return counts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist JSON with deterministic formatting."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_placeholder_adapter(adapter_dir: Path) -> Path:
    """Write a placeholder adapter artifact for scaffold-only runs."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    adapter_path.write_bytes(b"")
    return adapter_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser for Phase 5 scaffold runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-mode", default=DEFAULT_DATASET_MODE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RECONSTRUCTION_SEED)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--corpus-output-dir", type=Path, default=Path("outputs/corpus"))
    parser.add_argument("--split-manifest-path", type=Path, required=True)
    parser.add_argument("--target-envelopes-path", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="offline")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a Phase 5 scaffold pass that prepares training artifacts."""
    args = build_argument_parser().parse_args(argv)
    paths = ReconstructionPaths(project_root=PROJECT_ROOT)

    seed_everything(args.seed)
    run_dir = prepare_run_directory(args.run_id, paths=paths)

    split_manifest = load_split_manifest(args.split_manifest_path)
    target_envelopes = load_target_envelopes(args.target_envelopes_path)

    windows = extract_windows(
        corpus_dir=args.corpus_dir,
        corpus_output_dir=args.corpus_output_dir,
        min_words=split_manifest.min_words,
        max_words=split_manifest.max_words,
    )

    examples = build_training_examples(
        windows,
        split_manifest,
        target_envelopes,
        dataset_mode=args.dataset_mode,
    )
    split_counts = count_examples_by_split(examples)

    training_config = TrainingConfig(
        run_id=args.run_id,
        model_id=args.model_id,
        dataset_mode=args.dataset_mode,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )

    config_path = run_dir / "training_config.json"
    tokenizer_config_path = run_dir / "tokenizer_config.json"
    metrics_path = run_dir / "training_metrics.json"
    adapter_dir = run_dir / "adapter"
    adapter_path = _write_placeholder_adapter(adapter_dir)

    _write_json(config_path, training_config.to_dict())
    _write_json(tokenizer_config_path, {"model_id": training_config.model_id})
    _write_json(
        metrics_path,
        {
            "run_id": args.run_id,
            "status": "scaffold_only",
            "dataset_mode": args.dataset_mode,
            "split_counts": split_counts,
            "tolerance_config": ToleranceConfig().to_dict(),
        },
    )

    checkpoint_metadata = CheckpointMetadata(
        run_id=args.run_id,
        git_sha=detect_git_sha(paths.project_root),
        phase="phase-5-training-scaffold",
        model_id=args.model_id,
        adapter_type="qlora",
        adapter_artifact_path=to_project_relative(adapter_path, paths.project_root),
        config_path=to_project_relative(config_path, paths.project_root),
        tokenizer_config_path=to_project_relative(tokenizer_config_path, paths.project_root),
        metrics_path=to_project_relative(metrics_path, paths.project_root),
        split_counts=split_counts,
    )
    checkpoint_path = run_dir / "checkpoint_metadata.json"
    _write_json(checkpoint_path, checkpoint_metadata.to_dict())

    run_manifest = build_run_manifest(
        run_id=args.run_id,
        phase="phase-5-training-scaffold",
        model_id=args.model_id,
        seed=args.seed,
        git_sha=detect_git_sha(paths.project_root),
        config_payload={
            "training_config_path": to_project_relative(config_path, paths.project_root),
            "checkpoint_metadata_path": to_project_relative(checkpoint_path, paths.project_root),
            "dataset_mode": args.dataset_mode,
        },
        corpus_manifest="outputs/corpus/corpus_metadata.json",
        split_manifest=args.split_manifest_path,
        paths=paths,
    )
    write_run_manifest(run_manifest, paths=paths)

    logger = build_experiment_logger(
        config=training_config,
        git_sha=detect_git_sha(paths.project_root),
        split_counts=split_counts,
    )
    logger.log_metrics({"training_examples": float(len(examples))})
    logger.finish()

    finalize_run_manifest(
        args.run_id,
        RunStatus.COMPLETED,
        paths=paths,
        error_message=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
