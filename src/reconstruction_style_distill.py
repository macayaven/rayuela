#!/usr/bin/env python3
"""
Distill scored Phase 4 style-transfer generations into Phase 5 SFT examples.

The script reads ignored prompt-baseline artifacts and writes ignored JSONL
training data. It deliberately avoids adding generated/source passages to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reconstruction_train import TrainingExample

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reconstruction" / "style_distill"
STYLE_TRANSFER_DISTILLED_MODE = "style_transfer_distilled"


@dataclass(frozen=True)
class DistilledExampleMetadata:
    """Trace metadata for one distilled teacher example."""

    teacher_cases_path: str
    case_id: str
    target_author: str
    target_title: str
    target_envelope_id: str
    weighted_objective: float
    semantic_tolerance_pass: bool
    target_tolerance_pass: bool
    length_guardrail_pass: bool
    lexical_overlap_pass: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class DistilledTrainingExample:
    """One SFT example plus teacher/provenance metadata."""

    training_example: TrainingExample
    metadata: DistilledExampleMetadata

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable JSONL payload."""
        payload = self.training_example.to_dict()
        payload["teacher_metadata"] = self.metadata.to_dict()
        return payload


def project_relative(path: Path) -> str:
    """Return a stable path relative to the project root where possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def split_for_case_id(case_id: str, *, seed: int, train_ratio: float, val_ratio: float) -> str:
    """Assign a deterministic split from a case ID."""
    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
    bucket = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def style_summary(source_window: dict[str, Any], target_envelope: dict[str, Any]) -> str:
    """Return compact top-delta style cues for the instruction."""
    deltas: list[tuple[float, str, float, float]] = []
    source_style = source_window.get("stylometric_reference", {})
    target_style = target_envelope.get("stylometric_target", {})
    for name, target_value in target_style.items():
        if name not in source_style:
            continue
        source_value = float(source_style[name])
        target_float = float(target_value)
        deltas.append((abs(target_float - source_value), str(name), source_value, target_float))
    top_deltas = sorted(deltas, reverse=True)[:5]
    if not top_deltas:
        return "mantener una transferencia estilística controlada"
    return "; ".join(
        f"{name} fuente={source_value:.3f} objetivo={target_value:.3f}"
        for _, name, source_value, target_value in top_deltas
    )


def build_instruction(source_window: dict[str, Any], target_envelope: dict[str, Any]) -> str:
    """Build the style-transfer instruction for one distilled example."""
    cues = style_summary(source_window, target_envelope)
    return (
        "Reescribe el pasaje en español conservando los hechos, la escena, los personajes, "
        "el orden narrativo y una longitud cercana. Desplaza la textura hacia el sobre "
        f"estilístico de {target_envelope['author']} ({target_envelope['title']}). "
        f"Pistas métricas: {cues}. Devuelve solamente el pasaje final, sin razonamiento, "
        "notas, encabezados, markdown ni explicación."
    )


def best_iteration(result: dict[str, Any]) -> dict[str, Any]:
    """Return the accepted best iteration from one Phase 4 result."""
    index = int(result["best_iteration_index"])
    iterations = result["iterations"]
    return iterations[index]


def result_to_distilled_example(
    result: dict[str, Any],
    *,
    teacher_cases_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> DistilledTrainingExample:
    """Convert one scored Phase 4 result into one SFT example."""
    case = result["case"]
    source_window = case["source_window"]
    target_envelope = case["target_envelope"]
    iteration = best_iteration(result)
    score_history = iteration["score_history"]
    case_id = str(case["case_id"])
    split = split_for_case_id(
        case_id,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    example = TrainingExample(
        window_id=case_id,
        split=split,
        instruction=build_instruction(source_window, target_envelope),
        source_text=str(source_window["text"]),
        target_text=str(iteration["parsed_text"]).strip(),
        target_envelope_id=str(target_envelope["envelope_id"]),
        dataset_mode=STYLE_TRANSFER_DISTILLED_MODE,
    )
    metadata = DistilledExampleMetadata(
        teacher_cases_path=project_relative(teacher_cases_path),
        case_id=case_id,
        target_author=str(target_envelope["author"]),
        target_title=str(target_envelope["title"]),
        target_envelope_id=str(target_envelope["envelope_id"]),
        weighted_objective=float(score_history["weighted_objective"]),
        semantic_tolerance_pass=bool(score_history["semantic_tolerance_pass"]),
        target_tolerance_pass=bool(score_history["target_tolerance_pass"]),
        length_guardrail_pass=bool(score_history["length_guardrail_pass"]),
        lexical_overlap_pass=bool(score_history["lexical_overlap_pass"]),
    )
    return DistilledTrainingExample(training_example=example, metadata=metadata)


def load_distillable_results(
    path: Path,
    *,
    min_weighted_objective: float,
    require_semantic_pass: bool,
    require_target_pass: bool,
) -> list[dict[str, Any]]:
    """Load scoreable Phase 4 results that pass teacher-quality filters."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        iteration = best_iteration(result)
        text = str(iteration.get("parsed_text", "")).strip()
        if not text:
            continue
        score_history = iteration["score_history"]
        if float(score_history["weighted_objective"]) < min_weighted_objective:
            continue
        if require_semantic_pass and not bool(score_history["semantic_tolerance_pass"]):
            continue
        if require_target_pass and not bool(score_history["target_tolerance_pass"]):
            continue
        results.append(result)
    return results


def write_distilled_dataset(
    examples: list[DistilledTrainingExample],
    output_dir: Path,
) -> dict[str, Any]:
    """Write split JSONL files and a compact manifest."""
    if not examples:
        raise ValueError("no distillable teacher examples passed the configured filters")
    dataset_dir = output_dir / "training_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        split_path = dataset_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                if example.training_example.split != split:
                    continue
                split_counts[split] += 1
                handle.write(
                    json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                )
    manifest = {
        "dataset_mode": STYLE_TRANSFER_DISTILLED_MODE,
        "example_count": len(examples),
        "split_counts": split_counts,
        "dataset_paths": {
            split: project_relative(dataset_dir / f"{split}.jsonl")
            for split in ("train", "val", "test")
        },
    }
    (output_dir / "distill_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser for style-transfer distillation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cases-path", type=Path, action="append", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-weighted-objective", type=float, default=0.0)
    parser.add_argument("--require-semantic-pass", action="store_true")
    parser.add_argument("--require-target-pass", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Distill Phase 4 cases into SFT JSONL examples."""
    args = build_argument_parser().parse_args(argv)
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must leave room for test examples")

    examples: list[DistilledTrainingExample] = []
    seen_case_ids: set[str] = set()
    for path in args.teacher_cases_path:
        for result in load_distillable_results(
            path,
            min_weighted_objective=args.min_weighted_objective,
            require_semantic_pass=args.require_semantic_pass,
            require_target_pass=args.require_target_pass,
        ):
            case_id = str(result["case"]["case_id"])
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            examples.append(
                result_to_distilled_example(
                    result,
                    teacher_cases_path=path,
                    seed=args.seed,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                )
            )

    output_dir = args.output_root / args.dataset_id
    manifest = write_distilled_dataset(examples, output_dir)
    print("Distilled style-transfer dataset")
    print(f"  Dataset: {project_relative(output_dir)}")
    print(f"  Examples: {manifest['example_count']}")
    print(f"  Splits: {manifest['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
