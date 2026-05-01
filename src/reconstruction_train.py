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
import shutil
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
from reconstruction_metrics import ToleranceConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "google/mt5-xl"
DEFAULT_DATASET_MODE = "identity_smoke"
CONTRACT_DATASET_MODE = "contract_smoke"
SCAFFOLD_TRAINING_MODE = "scaffold"
SEQ2SEQ_SMOKE_TRAINING_MODE = "seq2seq_smoke"


def _load_pilot_json(path: Path) -> Any:
    """Load pilot JSON payloads without relying on another module's private helpers."""
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TrainingExample:
    """One training example aligned to the pilot split manifest."""

    window_id: str
    split: str
    instruction: str
    source_text: str
    target_text: str
    target_envelope_id: str
    dataset_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Serialized training configuration for Phase 5 scaffolding."""

    run_id: str
    model_id: str
    dataset_mode: str
    training_mode: str
    seed: int
    wandb_project: str | None
    wandb_entity: str | None
    wandb_mode: str
    max_steps: int
    max_train_examples: int
    max_eval_examples: int
    learning_rate: float
    per_device_train_batch_size: int
    max_source_length: int
    max_target_length: int

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
    adapter_is_placeholder: bool
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
            "training_mode": config.training_mode,
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
    if dataset_mode not in {DEFAULT_DATASET_MODE, CONTRACT_DATASET_MODE}:
        raise ValueError(
            f"Unsupported dataset_mode {dataset_mode!r}; "
            f"supported modes: {DEFAULT_DATASET_MODE!r}, {CONTRACT_DATASET_MODE!r}."
        )

    assignment_lookup = split_manifest.assignment_lookup()
    manifest_window_ids = set(assignment_lookup)
    window_ids = {window.window_id for window in windows}
    missing_window_ids = manifest_window_ids - window_ids
    extra_window_ids = window_ids - manifest_window_ids
    if missing_window_ids or extra_window_ids:
        missing_sample = ", ".join(sorted(missing_window_ids)[:5])
        extra_sample = ", ".join(sorted(extra_window_ids)[:5])
        raise ValueError(
            "Split manifest window IDs do not match extracted windows. "
            f"missing={len(missing_window_ids)}"
            f"{' [' + missing_sample + ']' if missing_sample else ''}, "
            f"extra={len(extra_window_ids)}"
            f"{' [' + extra_sample + ']' if extra_sample else ''}."
        )

    default_envelope_id = target_envelopes[0].envelope_id if target_envelopes else "target:unknown"
    examples: list[TrainingExample] = []
    for window in windows:
        split = assignment_lookup[window.window_id]
        source_text = window.text
        target_text = window.text
        if dataset_mode == CONTRACT_DATASET_MODE:
            instruction = (
                "Reescribe el pasaje en español conservando estrictamente los hechos, "
                "la escena, los personajes, el orden narrativo y una longitud cercana. "
                "Devuelve solamente el pasaje final, sin razonamiento, notas, encabezados, "
                "markdown ni explicación."
            )
        else:
            instruction = (
                "Devuelve el pasaje en español con cambios mínimos. Conserva el contenido, "
                "la longitud y el estilo. Devuelve solamente el pasaje final."
            )
        examples.append(
            TrainingExample(
                window_id=window.window_id,
                split=split,
                instruction=instruction,
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


def write_training_dataset(examples: list[TrainingExample], dataset_dir: Path) -> dict[str, str]:
    """Write split-specific JSONL training examples and return relative filenames."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_path = dataset_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                if example.split != split:
                    continue
                line = json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True)
                handle.write(line + "\n")
        paths[split] = split_path.name
    return paths


def format_seq2seq_input(example: TrainingExample) -> str:
    """Format one reconstruction example as a supervised seq2seq input."""
    return f"{example.instruction}\n\nPasaje:\n{example.source_text}"


def select_training_examples(
    examples: list[TrainingExample],
    *,
    split: str,
    limit: int,
) -> list[TrainingExample]:
    """Select a deterministic bounded subset from one split."""
    selected = [example for example in examples if example.split == split]
    if limit > 0:
        return selected[:limit]
    return selected


def _load_seq2seq_training_backend() -> dict[str, Any]:
    """Load optional training dependencies only for real training runs."""
    try:
        torch = importlib.import_module("torch")
        datasets = importlib.import_module("datasets")
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Seq2seq smoke training requires torch, datasets, transformers, and accelerate. "
            "Run inside the Rayuela analysis container or install those packages first."
        ) from exc

    return {
        "torch": torch,
        "Dataset": datasets.Dataset,
        "AutoModelForSeq2SeqLM": transformers.AutoModelForSeq2SeqLM,
        "AutoTokenizer": transformers.AutoTokenizer,
        "DataCollatorForSeq2Seq": transformers.DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": transformers.Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": transformers.Seq2SeqTrainingArguments,
    }


def run_seq2seq_smoke_training(
    *,
    examples: list[TrainingExample],
    config: TrainingConfig,
    model_output_dir: Path,
) -> dict[str, float | int | str]:
    """Run a bounded real seq2seq training smoke test and save the model artifact."""
    train_examples = select_training_examples(
        examples,
        split="train",
        limit=config.max_train_examples,
    )
    eval_examples = select_training_examples(
        examples,
        split="val",
        limit=config.max_eval_examples,
    )
    if not train_examples:
        raise ValueError("seq2seq smoke training requires at least one train example")

    backend = _load_seq2seq_training_backend()
    dataset_cls = backend["Dataset"]
    tokenizer = backend["AutoTokenizer"].from_pretrained(config.model_id)
    model = backend["AutoModelForSeq2SeqLM"].from_pretrained(config.model_id)

    def _records(selected: list[TrainingExample]) -> list[dict[str, str]]:
        return [
            {
                "input_text": format_seq2seq_input(example),
                "target_text": example.target_text,
            }
            for example in selected
        ]

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=config.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=config.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = dataset_cls.from_list(_records(train_examples)).map(
        _tokenize,
        batched=True,
        remove_columns=["input_text", "target_text"],
    )
    eval_dataset = None
    if eval_examples:
        eval_dataset = dataset_cls.from_list(_records(eval_examples)).map(
            _tokenize,
            batched=True,
            remove_columns=["input_text", "target_text"],
        )

    training_args = backend["Seq2SeqTrainingArguments"](
        output_dir=str(model_output_dir / "trainer_state"),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = backend["Seq2SeqTrainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=backend["DataCollatorForSeq2Seq"](tokenizer=tokenizer, model=model),
    )
    train_output = trainer.train()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    metrics: dict[str, float | int | str] = {
        str(key): float(value) for key, value in train_output.metrics.items()
    }
    metrics["trained_examples"] = len(train_examples)
    metrics["eval_examples"] = len(eval_examples)
    metrics["max_steps"] = config.max_steps
    metrics["artifact_type"] = "seq2seq_full_model_smoke"
    return metrics


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


def _cleanup_failed_run_initialization(run_id: str, *, paths: ReconstructionPaths) -> None:
    """Remove partially initialized manifest files so a failed setup can be retried cleanly."""
    paths.manifest_path(run_id).unlink(missing_ok=True)
    paths.indexed_manifest_path(run_id).unlink(missing_ok=True)
    run_dir = paths.run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser for Phase 5 scaffold runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-mode", default=DEFAULT_DATASET_MODE)
    parser.add_argument(
        "--training-mode",
        default=SCAFFOLD_TRAINING_MODE,
        choices=(SCAFFOLD_TRAINING_MODE, SEQ2SEQ_SMOKE_TRAINING_MODE),
        help="Use scaffold for metadata only, or seq2seq_smoke for a bounded real training run.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RECONSTRUCTION_SEED)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-train-examples", type=int, default=32)
    parser.add_argument("--max-eval-examples", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Override detected git SHA, useful in training containers without git installed.",
    )
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "corpus")
    parser.add_argument(
        "--corpus-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "corpus"
    )
    parser.add_argument("--split-manifest-path", type=Path, required=True)
    parser.add_argument("--target-envelopes-path", type=Path, required=True)
    parser.add_argument(
        "--allow-corpus-discovery",
        action="store_true",
        help="Allow non-canonical corpus discovery when using ad hoc local corpora.",
    )
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
    training_config = TrainingConfig(
        run_id=args.run_id,
        model_id=args.model_id,
        dataset_mode=args.dataset_mode,
        training_mode=args.training_mode,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        max_steps=args.max_steps,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    git_sha = args.git_sha or detect_git_sha(paths.project_root)
    config_path = run_dir / "training_config.json"
    tokenizer_config_path = run_dir / "tokenizer_config.json"
    metrics_path = run_dir / "training_metrics.json"
    checkpoint_path = run_dir / "checkpoint_metadata.json"
    adapter_dir = run_dir / "adapter"
    model_dir = run_dir / "model"
    dataset_dir = run_dir / "training_dataset"
    try:
        run_manifest = build_run_manifest(
            run_id=args.run_id,
            phase="phase-5-training-scaffold",
            model_id=args.model_id,
            seed=args.seed,
            git_sha=git_sha,
            config_payload={
                "training_config_path": to_project_relative(config_path, paths.project_root),
                "checkpoint_metadata_path": to_project_relative(
                    checkpoint_path, paths.project_root
                ),
                "dataset_mode": args.dataset_mode,
                "training_mode": args.training_mode,
            },
            corpus_manifest=to_project_relative(
                args.corpus_output_dir / "corpus_metadata.json",
                paths.project_root,
            ),
            split_manifest=args.split_manifest_path,
            paths=paths,
        )
        write_run_manifest(run_manifest, paths=paths)
    except Exception:
        _cleanup_failed_run_initialization(args.run_id, paths=paths)
        raise

    run_status = RunStatus.RUNNING
    error_message: str | None = None
    try:
        split_manifest = load_split_manifest(args.split_manifest_path)
        target_envelopes = load_target_envelopes(args.target_envelopes_path)

        windows = extract_windows(
            corpus_dir=args.corpus_dir,
            corpus_output_dir=args.corpus_output_dir,
            allow_discovery=args.allow_corpus_discovery,
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
        dataset_paths = write_training_dataset(examples, dataset_dir)
        training_metrics: dict[str, float | int | str]
        if args.training_mode == SEQ2SEQ_SMOKE_TRAINING_MODE:
            artifact_path = model_dir
            adapter_type = "seq2seq_full_model_smoke"
            adapter_is_placeholder = False
            training_status = "trained_smoke"
            training_metrics = run_seq2seq_smoke_training(
                examples=examples,
                config=training_config,
                model_output_dir=model_dir,
            )
        else:
            artifact_path = _write_placeholder_adapter(adapter_dir)
            adapter_type = "qlora"
            adapter_is_placeholder = True
            training_status = "scaffold_only"
            training_metrics = {}

        _write_json(config_path, training_config.to_dict())
        _write_json(tokenizer_config_path, {"model_id": training_config.model_id})
        _write_json(
            metrics_path,
            {
                "run_id": args.run_id,
                "status": training_status,
                "training_mode": args.training_mode,
                "dataset_mode": args.dataset_mode,
                "split_counts": split_counts,
                "dataset_paths": {
                    split: to_project_relative(dataset_dir / filename, paths.project_root)
                    for split, filename in dataset_paths.items()
                },
                "training_metrics": training_metrics,
                "tolerance_config": ToleranceConfig().to_dict(),
            },
        )

        checkpoint_metadata = CheckpointMetadata(
            run_id=args.run_id,
            git_sha=git_sha,
            phase="phase-5-training-scaffold",
            model_id=args.model_id,
            adapter_type=adapter_type,
            adapter_artifact_path=to_project_relative(artifact_path, paths.project_root),
            adapter_is_placeholder=adapter_is_placeholder,
            config_path=to_project_relative(config_path, paths.project_root),
            tokenizer_config_path=to_project_relative(tokenizer_config_path, paths.project_root),
            metrics_path=to_project_relative(metrics_path, paths.project_root),
            split_counts=split_counts,
        )
        _write_json(checkpoint_path, checkpoint_metadata.to_dict())

        logger = build_experiment_logger(
            config=training_config,
            git_sha=git_sha,
            split_counts=split_counts,
        )
        logger.log_metrics({"training_examples": float(len(examples))})
        logger.finish()
        run_status = RunStatus.COMPLETED
    except BaseException as exc:
        run_status = RunStatus.FAILED
        error_message = str(exc)
        raise
    finally:
        finalize_run_manifest(
            args.run_id,
            run_status,
            paths=paths,
            error_message=error_message,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
