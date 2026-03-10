#!/usr/bin/env python3
"""
Aggregate reconstruction run artifacts into analysis-ready summaries.

Phase 6 turns immutable run outputs into compact synthesis inputs: complete case
tables, labeled failure modes, coarse bias slices, close-reading queues, and
article-ready summary files.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconstruction_contract import ReconstructionPaths

DEFAULT_ANALYSIS_DIR = ReconstructionPaths().analysis_dir


@dataclass(frozen=True)
class AggregatedCaseRecord:
    """Compact analysis record for one evaluated reconstruction case."""

    run_id: str
    case_id: str
    control_mode: str
    source_work_id: str
    source_author: str
    source_title: str
    target_work_id: str
    target_author: str
    target_title: str
    weighted_objective: float
    stop_reason: str
    failure_labels: tuple[str, ...]
    manifest_path: str
    cases_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class BiasSliceRecord:
    """Aggregated objective summary for one source-side slice."""

    slice_key: str
    count: int
    mean_weighted_objective: float
    median_weighted_objective: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class CloseReadingNote:
    """Article-facing close-reading prompt for one salient case."""

    run_id: str
    case_id: str
    priority: str
    weighted_objective: float
    source_work_id: str
    target_work_id: str
    failure_labels: tuple[str, ...]
    manifest_path: str
    cases_path: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class AnalysisReport:
    """Phase 6 synthesis payload built from one or more run directories."""

    generated_at: str
    run_ids: tuple[str, ...]
    cases: tuple[AggregatedCaseRecord, ...]
    run_summaries: dict[str, dict[str, Any]]
    failure_modes: dict[str, dict[str, Any]]
    bias_slices: dict[str, tuple[BiasSliceRecord, ...]]
    close_reading_notes: tuple[CloseReadingNote, ...]

    @property
    def total_runs(self) -> int:
        """Return the number of unique runs represented in the report."""
        return len(self.run_ids)

    @property
    def total_cases(self) -> int:
        """Return the number of aggregated cases."""
        return len(self.cases)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "generated_at": self.generated_at,
            "total_runs": self.total_runs,
            "total_cases": self.total_cases,
            "run_ids": list(self.run_ids),
            "cases": [case.to_dict() for case in self.cases],
            "run_summaries": self.run_summaries,
            "failure_modes": self.failure_modes,
            "bias_slices": {
                key: [record.to_dict() for record in records]
                for key, records in self.bias_slices.items()
            },
            "close_reading_notes": [note.to_dict() for note in self.close_reading_notes],
        }


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON payload from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_wandb_module() -> Any:
    """Load wandb only when requested."""
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Weights & Biases requested but `wandb` is not installed. "
            "Install it with `.venv/bin/python -m pip install wandb`."
        ) from exc


def _case_failure_labels(result: dict[str, Any]) -> tuple[str, ...]:
    """Derive stable failure labels from one serialized baseline result."""
    best_iteration_index = int(result["best_iteration_index"])
    best_iteration = result["iterations"][best_iteration_index]
    score_history = best_iteration["score_history"]
    labels: list[str] = []
    if not bool(score_history["semantic_tolerance_pass"]):
        labels.append("semantic_drift")
    if not bool(score_history["target_tolerance_pass"]):
        labels.append("target_miss")
    if not bool(score_history["length_guardrail_pass"]):
        labels.append("length_guardrail")
    if not bool(score_history["lexical_overlap_pass"]):
        labels.append("lexical_overlap")
    if str(result["stop_reason"]) == "no_objective_improvement":
        labels.append("stalled_revision")
    return tuple(labels)


def _load_cases_from_run_dir(run_dir: Path) -> list[AggregatedCaseRecord]:
    """Load all prompt-baseline cases from one immutable run directory."""
    cases_path = run_dir / "prompt_baseline_cases.json"
    if not cases_path.exists():
        return []

    manifest_path = run_dir / "manifest.json"
    run_id = run_dir.name
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        run_id = str(manifest.get("run_id", run_id))

    payload = _load_json(cases_path)
    manifest_path = run_dir / "manifest.json"
    records: list[AggregatedCaseRecord] = []
    for result in payload["results"]:
        case = result["case"]
        source_window = case["source_window"]
        target_envelope = case["target_envelope"]
        best_iteration = result["iterations"][int(result["best_iteration_index"])]
        records.append(
            AggregatedCaseRecord(
                run_id=run_id,
                case_id=str(case["case_id"]),
                control_mode=str(case["control_mode"]),
                source_work_id=str(source_window["work_id"]),
                source_author=str(source_window["author"]),
                source_title=str(source_window["title"]),
                target_work_id=str(target_envelope["work_id"]),
                target_author=str(target_envelope["author"]),
                target_title=str(target_envelope["title"]),
                weighted_objective=float(best_iteration["score_history"]["weighted_objective"]),
                stop_reason=str(result["stop_reason"]),
                failure_labels=_case_failure_labels(result),
                manifest_path=str(manifest_path),
                cases_path=str(cases_path),
            )
        )
    return records


def _summarize_runs(
    cases: tuple[AggregatedCaseRecord, ...]
) -> dict[str, dict[str, Any]]:
    """Summarize aggregated cases at the run level."""
    grouped: dict[str, list[AggregatedCaseRecord]] = {}
    for case in cases:
        grouped.setdefault(case.run_id, []).append(case)

    return {
        run_id: {
            "case_count": len(items),
            "failure_case_count": sum(1 for item in items if item.failure_labels),
            "mean_weighted_objective": sum(item.weighted_objective for item in items) / len(items),
            "manifest_path": items[0].manifest_path,
            "cases_path": items[0].cases_path,
        }
        for run_id, items in grouped.items()
    }


def _summarize_failure_modes(
    cases: tuple[AggregatedCaseRecord, ...]
) -> dict[str, dict[str, Any]]:
    """Count cases affected by each failure label."""
    summary: dict[str, dict[str, Any]] = {}
    for label in (
        "semantic_drift",
        "target_miss",
        "length_guardrail",
        "lexical_overlap",
        "stalled_revision",
    ):
        matching = [case for case in cases if label in case.failure_labels]
        summary[label] = {
            "count": len(matching),
            "case_ids": [case.case_id for case in matching],
            "run_ids": sorted({case.run_id for case in matching}),
        }
    return summary


def _objective_median(values: list[float]) -> float:
    """Return a deterministic median without adding a NumPy dependency."""
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _build_slice_records(
    cases: tuple[AggregatedCaseRecord, ...],
    *,
    attribute: str,
) -> tuple[BiasSliceRecord, ...]:
    """Aggregate weighted objectives by one case attribute."""
    grouped: dict[str, list[float]] = {}
    for case in cases:
        grouped.setdefault(str(getattr(case, attribute)), []).append(case.weighted_objective)

    records = [
        BiasSliceRecord(
            slice_key=slice_key,
            count=len(values),
            mean_weighted_objective=sum(values) / len(values),
            median_weighted_objective=_objective_median(values),
        )
        for slice_key, values in grouped.items()
    ]
    return tuple(sorted(records, key=lambda record: (-record.count, record.slice_key)))


def _build_close_reading_notes(
    cases: tuple[AggregatedCaseRecord, ...],
) -> tuple[CloseReadingNote, ...]:
    """Create a small, deduplicated queue of salient cases for human close reading."""
    if not cases:
        return ()

    ranked = sorted(cases, key=lambda case: case.weighted_objective)
    candidates = [ranked[0], ranked[-1]]
    notes: list[CloseReadingNote] = []
    seen: set[tuple[str, str]] = set()
    for case in candidates:
        key = (case.run_id, case.case_id)
        if key in seen:
            continue
        seen.add(key)
        priority = "high" if case is ranked[-1] else "low"
        label_text = (
            ", ".join(case.failure_labels)
            if case.failure_labels
            else "no explicit failure labels"
        )
        notes.append(
            CloseReadingNote(
                run_id=case.run_id,
                case_id=case.case_id,
                priority=priority,
                weighted_objective=case.weighted_objective,
                source_work_id=case.source_work_id,
                target_work_id=case.target_work_id,
                failure_labels=case.failure_labels,
                manifest_path=case.manifest_path,
                cases_path=case.cases_path,
                prompt=(
                    f"Close-read {case.case_id} from run {case.run_id}: "
                    f"compare {case.source_title} -> {case.target_title}; "
                    f"objective={case.weighted_objective:.4f}; labels={label_text}."
                ),
            )
        )
    return tuple(notes)


def build_analysis_report(run_dirs: list[Path]) -> AnalysisReport:
    """Aggregate one or more immutable run directories into a Phase 6 report."""
    cases: list[AggregatedCaseRecord] = []
    for run_dir in run_dirs:
        cases.extend(_load_cases_from_run_dir(run_dir))

    ordered_cases = tuple(
        sorted(cases, key=lambda case: (case.run_id, case.case_id, case.weighted_objective))
    )
    run_ids = tuple(sorted({case.run_id for case in ordered_cases}))
    return AnalysisReport(
        generated_at=utc_now(),
        run_ids=run_ids,
        cases=ordered_cases,
        run_summaries=_summarize_runs(ordered_cases),
        failure_modes=_summarize_failure_modes(ordered_cases),
        bias_slices={
            "by_work": _build_slice_records(ordered_cases, attribute="source_work_id"),
            "by_author": _build_slice_records(ordered_cases, attribute="source_author"),
        },
        close_reading_notes=_build_close_reading_notes(ordered_cases),
    )


def _article_inputs_payload(report: AnalysisReport) -> dict[str, Any]:
    """Return compact article-facing inputs derived from the full report."""
    best_case = max(report.cases, key=lambda case: case.weighted_objective, default=None)
    weakest_case = min(report.cases, key=lambda case: case.weighted_objective, default=None)
    return {
        "generated_at": report.generated_at,
        "total_runs": report.total_runs,
        "total_cases": report.total_cases,
        "headline_findings": [
            (
                "Best observed reconstruction objective: "
                f"{best_case.case_id} in {best_case.run_id} "
                f"({best_case.weighted_objective:.4f})."
            )
            if best_case is not None
            else "No evaluated cases found."
        ],
        "run_summaries": report.run_summaries,
        "weakest_case": None if weakest_case is None else weakest_case.to_dict(),
        "failure_modes": report.failure_modes,
        "bias_slices": {
            key: [record.to_dict() for record in records]
            for key, records in report.bias_slices.items()
        },
        "close_reading_queue": [note.to_dict() for note in report.close_reading_notes],
    }


def _log_analysis_to_wandb(
    *,
    report: AnalysisReport,
    summary_path: Path,
    report_path: Path,
    article_inputs_path: Path,
    article_inputs: dict[str, Any],
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    wandb_group: str | None,
) -> None:
    """Log aggregate analysis outputs to an optional W&B run."""
    wandb = _load_wandb_module()
    run_name = f"analysis-{report.run_ids[0]}" if report.run_ids else "analysis-empty"
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        group=wandb_group,
        job_type="reconstruction_analysis",
        tags=["phase6", "analysis", "article-inputs"],
        name=run_name,
        config={
            "run_ids": list(report.run_ids),
            "total_runs": report.total_runs,
            "total_cases": report.total_cases,
            "failure_labels": sorted(report.failure_modes),
        },
    )
    run.log(
        {
            "analysis/total_runs": float(report.total_runs),
            "analysis/total_cases": float(report.total_cases),
            "analysis/failure_mode_semantic_drift_count": float(
                report.failure_modes["semantic_drift"]["count"]
            ),
            "analysis/failure_mode_target_miss_count": float(
                report.failure_modes["target_miss"]["count"]
            ),
            "analysis/failure_mode_length_guardrail_count": float(
                report.failure_modes["length_guardrail"]["count"]
            ),
            "analysis/failure_mode_lexical_overlap_count": float(
                report.failure_modes["lexical_overlap"]["count"]
            ),
            "analysis/failure_mode_stalled_revision_count": float(
                report.failure_modes["stalled_revision"]["count"]
            ),
            "article/close_reading_note_count": float(len(report.close_reading_notes)),
        }
    )

    run_summary_table = wandb.Table(
        columns=[
            "run_id",
            "case_count",
            "failure_case_count",
            "mean_weighted_objective",
            "manifest_path",
            "cases_path",
        ],
        data=[
            [
                run_id,
                payload["case_count"],
                payload["failure_case_count"],
                payload["mean_weighted_objective"],
                payload["manifest_path"],
                payload["cases_path"],
            ]
            for run_id, payload in sorted(report.run_summaries.items())
        ],
    )
    run.log({"analysis/run_summary_table": run_summary_table})

    close_reading_table = wandb.Table(
        columns=[
            "run_id",
            "case_id",
            "priority",
            "weighted_objective",
            "source_work_id",
            "target_work_id",
            "failure_labels",
            "prompt",
        ],
        data=[
            [
                note.run_id,
                note.case_id,
                note.priority,
                note.weighted_objective,
                note.source_work_id,
                note.target_work_id,
                list(note.failure_labels),
                note.prompt,
            ]
            for note in report.close_reading_notes
        ],
    )
    run.log({"article/close_reading_queue": close_reading_table})

    run.summary.update(
        {
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "article_inputs_path": str(article_inputs_path),
            "headline_findings": article_inputs["headline_findings"],
        }
    )

    artifact = wandb.Artifact(
        name=f"{run_name}-artifacts",
        type="reconstruction-analysis",
    )
    artifact.add_file(str(summary_path), name=summary_path.name)
    artifact.add_file(str(report_path), name=report_path.name)
    artifact.add_file(str(article_inputs_path), name=article_inputs_path.name)
    run.log_artifact(artifact)
    run.finish()


def write_analysis_artifacts(
    report: AnalysisReport,
    *,
    output_dir: Path,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str = "offline",
    wandb_group: str | None = None,
) -> dict[str, Path]:
    """Persist the Phase 6 summary, report, and article-input artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "reconstruction_analysis_summary.json"
    report_path = output_dir / "reconstruction_analysis_report.md"
    article_inputs_path = output_dir / "reconstruction_article_inputs.json"

    summary_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    article_inputs = _article_inputs_payload(report)
    article_inputs_path.write_text(
        json.dumps(article_inputs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Phase 6 Reconstruction Analysis",
        "",
        f"Generated at: {report.generated_at}",
        f"Runs analyzed: {report.total_runs}",
        f"Cases analyzed: {report.total_cases}",
        "",
        "## Failure Modes",
        "",
    ]
    for label, payload in report.failure_modes.items():
        report_lines.append(
            f"- `{label}`: count={payload['count']}, runs={', '.join(payload['run_ids']) or 'none'}"
        )
    report_lines.extend(
        [
            "",
            "## Close Reading Queue",
            "",
        ]
    )
    for note in report.close_reading_notes:
        report_lines.append(
            f"- `{note.case_id}` ({note.run_id}, {note.priority}): "
            f"{note.weighted_objective:.4f}"
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if wandb_project:
        _log_analysis_to_wandb(
            report=report,
            summary_path=summary_path,
            report_path=report_path,
            article_inputs_path=article_inputs_path,
            article_inputs=article_inputs,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            wandb_group=wandb_group,
        )
    return {
        "summary": summary_path,
        "report": report_path,
        "article_inputs": article_inputs_path,
    }


def _default_run_dirs(paths: ReconstructionPaths) -> list[Path]:
    """Return immutable run directories that contain Phase 4 case histories."""
    if not paths.runs_dir.exists():
        return []
    return sorted(
        run_dir
        for run_dir in paths.runs_dir.iterdir()
        if run_dir.is_dir() and (run_dir / "prompt_baseline_cases.json").exists()
    )


def run_dirs_from_schedule_summary(
    schedule_summary_path: Path,
    *,
    project_root: Path | None = None,
) -> list[Path]:
    """Resolve kept run directories from one scheduler summary payload."""
    payload = _load_json(schedule_summary_path)
    run_ids = [str(run_id) for run_id in payload.get("kept_run_ids", [])]
    resolved_project_root = (
        ReconstructionPaths().project_root if project_root is None else project_root
    )
    return [
        (resolved_project_root / "outputs" / "reconstruction" / "runs" / run_id).resolve()
        for run_id in run_ids
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Phase 6 artifact aggregation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        type=Path,
        help="Immutable run directory to include. Defaults to all Phase 4 runs.",
    )
    parser.add_argument(
        "--schedule-summary-path",
        action="append",
        dest="schedule_summary_paths",
        type=Path,
        help="Schedule summary whose kept run_ids should be aggregated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Optional W&B group, e.g. a schedule_id for aggregation runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Aggregate the selected run directories and write analysis artifacts."""
    args = build_argument_parser().parse_args(argv)
    paths = ReconstructionPaths()
    run_dirs: list[Path] = []
    if args.run_dirs:
        run_dirs.extend(args.run_dirs)
    if args.schedule_summary_paths:
        for schedule_summary_path in args.schedule_summary_paths:
            run_dirs.extend(
                run_dirs_from_schedule_summary(
                    schedule_summary_path,
                    project_root=paths.project_root,
                )
            )
    if not run_dirs:
        run_dirs = _default_run_dirs(paths)
    report = build_analysis_report(run_dirs=run_dirs)
    write_analysis_artifacts(
        report,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
    )
    print(f"Analyzed {report.total_cases} cases across {report.total_runs} runs.")
    print(f"Summary path: {(args.output_dir / 'reconstruction_analysis_summary.json')}")
    print(f"Article inputs: {(args.output_dir / 'reconstruction_article_inputs.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
