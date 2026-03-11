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
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconstruction_contract import ReconstructionPaths

DEFAULT_ANALYSIS_DIR = ReconstructionPaths().analysis_dir
FAILURE_LABELS = (
    "semantic_drift",
    "target_miss",
    "length_guardrail",
    "lexical_overlap",
    "stalled_revision",
)
COMPARABILITY_INVARIANT_FIELDS = (
    "git_sha",
    "phase",
    "prompt_template_id",
    "model_id",
    "corpus_manifest",
    "split_manifest",
    "generation_seed",
    "api_base",
    "source_windows_path",
    "success_criteria_path",
    "target_envelopes_path",
)
DEFAULT_BOOTSTRAP_RESAMPLES = 1000


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
class RunProvenanceRecord:
    """Compact run-level provenance used to gate cross-run comparisons."""

    run_id: str
    manifest_path: str
    git_sha: str | None
    phase: str | None
    prompt_template_id: str | None
    model_id: str | None
    corpus_manifest: str | None
    split_manifest: str | None
    generation_seed: int | None
    api_base: str | None
    source_windows_path: str | None
    success_criteria_path: str | None
    target_envelopes_path: str | None
    max_cases: int | None
    max_iterations: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class FailureTransitionRecord:
    """One failure-label transition summary for a pair of compared runs."""

    label: str
    persistent_count: int
    resolved_count: int
    introduced_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class RunComparisonRecord:
    """Pairwise delta summary for two runs over overlapping cases."""

    reference_run_id: str
    candidate_run_id: str
    overlapping_case_count: int
    reference_failure_case_count: int
    candidate_failure_case_count: int
    failure_case_count_delta: int
    mean_weighted_objective_delta: float
    median_weighted_objective_delta: float
    mean_weighted_objective_delta_ci_low: float
    mean_weighted_objective_delta_ci_high: float
    mean_weighted_objective_delta_ci_excludes_zero: bool
    bootstrap_resamples: int
    improved_case_count: int
    worsened_case_count: int
    unchanged_case_count: int
    non_negative_case_share: float
    largest_gain_case_id: str | None
    largest_gain_delta: float | None
    largest_drop_case_id: str | None
    largest_drop_delta: float | None
    comparable: bool
    comparability_checks: dict[str, bool]
    comparability_issues: tuple[str, ...]
    failure_transitions: tuple[FailureTransitionRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["comparability_issues"] = list(self.comparability_issues)
        payload["failure_transitions"] = [
            transition.to_dict() for transition in self.failure_transitions
        ]
        return payload


@dataclass(frozen=True)
class PromotionCriteria:
    """Explicit research-facing criteria for promoting a candidate run."""

    min_overlapping_cases: int = 4
    min_mean_delta: float = 0.005
    min_median_delta: float = 0.0
    min_non_negative_share: float = 0.5
    max_failure_case_delta: int = 0
    require_comparable_provenance: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class PromotionRecommendation:
    """Research-facing recommendation for candidate promotion over an incumbent."""

    reference_run_id: str
    candidate_run_id: str
    recommendation: str
    overlapping_case_count: int
    mean_weighted_objective_delta: float
    median_weighted_objective_delta: float
    non_negative_case_share: float
    failure_case_count_delta: int
    criteria_results: dict[str, bool]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class AnalysisReport:
    """Phase 6 synthesis payload built from one or more run directories."""

    generated_at: str
    run_ids: tuple[str, ...]
    cases: tuple[AggregatedCaseRecord, ...]
    run_provenance: dict[str, RunProvenanceRecord]
    run_summaries: dict[str, dict[str, Any]]
    failure_modes: dict[str, dict[str, Any]]
    bias_slices: dict[str, tuple[BiasSliceRecord, ...]]
    promotion_criteria: PromotionCriteria
    run_comparisons: tuple[RunComparisonRecord, ...]
    promotion_recommendations: tuple[PromotionRecommendation, ...]
    close_reading_notes: tuple[CloseReadingNote, ...]

    @property
    def total_runs(self) -> int:
        """Return the number of unique runs represented in the report."""
        return len(self.run_ids)

    @property
    def total_cases(self) -> int:
        """Return the number of aggregated cases."""
        return len(self.cases)

    @property
    def final_incumbent_run_id(self) -> str | None:
        """Return the final incumbent after applying promotion recommendations."""
        if not self.run_ids:
            return None
        incumbent = self.run_ids[0]
        for recommendation in self.promotion_recommendations:
            if recommendation.recommendation == "promote":
                incumbent = recommendation.candidate_run_id
        return incumbent

    @property
    def comparability_summary(self) -> dict[str, Any]:
        """Summarize how many pairwise comparisons were provenance-comparable."""
        comparable_count = sum(1 for record in self.run_comparisons if record.comparable)
        field_mismatch_counts = {
            field: sum(
                1
                for record in self.run_comparisons
                if not record.comparability_checks.get(field, False)
            )
            for field in COMPARABILITY_INVARIANT_FIELDS
        }
        return {
            "comparable_comparison_count": comparable_count,
            "noncomparable_comparison_count": len(self.run_comparisons) - comparable_count,
            "field_mismatch_counts": field_mismatch_counts,
        }

    @property
    def failure_transition_summary(self) -> dict[str, dict[str, int]]:
        """Aggregate failure-label transitions across all run comparisons."""
        summary = {
            label: {
                "persistent_count": 0,
                "resolved_count": 0,
                "introduced_count": 0,
            }
            for label in FAILURE_LABELS
        }
        for comparison in self.run_comparisons:
            for transition in comparison.failure_transitions:
                summary[transition.label]["persistent_count"] += transition.persistent_count
                summary[transition.label]["resolved_count"] += transition.resolved_count
                summary[transition.label]["introduced_count"] += transition.introduced_count
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "generated_at": self.generated_at,
            "total_runs": self.total_runs,
            "total_cases": self.total_cases,
            "run_ids": list(self.run_ids),
            "cases": [case.to_dict() for case in self.cases],
            "run_provenance": {
                run_id: record.to_dict() for run_id, record in self.run_provenance.items()
            },
            "run_summaries": self.run_summaries,
            "failure_modes": self.failure_modes,
            "bias_slices": {
                key: [record.to_dict() for record in records]
                for key, records in self.bias_slices.items()
            },
            "promotion_criteria": self.promotion_criteria.to_dict(),
            "run_comparisons": [record.to_dict() for record in self.run_comparisons],
            "promotion_recommendations": [
                recommendation.to_dict() for recommendation in self.promotion_recommendations
            ],
            "final_incumbent_run_id": self.final_incumbent_run_id,
            "comparability_summary": self.comparability_summary,
            "failure_transition_summary": self.failure_transition_summary,
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


def _nested_manifest_value(payload: dict[str, Any], dotted_path: str) -> Any:
    """Return one nested manifest value, or `None` if any key is missing."""
    current: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _optional_int(value: Any) -> int | None:
    """Return an integer if the value is a non-bool int, else `None`."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _load_run_provenance(run_dir: Path) -> RunProvenanceRecord:
    """Load compact comparison provenance from one run manifest."""
    manifest_path = run_dir / "manifest.json"
    payload = _load_json(manifest_path) if manifest_path.exists() else {}
    run_id = str(payload.get("run_id", run_dir.name))
    return RunProvenanceRecord(
        run_id=run_id,
        manifest_path=str(manifest_path),
        git_sha=(
            None
            if _nested_manifest_value(payload, "git_sha") is None
            else str(_nested_manifest_value(payload, "git_sha"))
        ),
        phase=(
            None
            if _nested_manifest_value(payload, "phase") is None
            else str(_nested_manifest_value(payload, "phase"))
        ),
        prompt_template_id=(
            None
            if _nested_manifest_value(payload, "prompt_template_id") is None
            else str(_nested_manifest_value(payload, "prompt_template_id"))
        ),
        model_id=(
            None
            if _nested_manifest_value(payload, "model_id") is None
            else str(_nested_manifest_value(payload, "model_id"))
        ),
        corpus_manifest=(
            None
            if _nested_manifest_value(payload, "corpus_manifest") is None
            else str(_nested_manifest_value(payload, "corpus_manifest"))
        ),
        split_manifest=(
            None
            if _nested_manifest_value(payload, "split_manifest") is None
            else str(_nested_manifest_value(payload, "split_manifest"))
        ),
        generation_seed=_optional_int(
            _nested_manifest_value(payload, "config_payload.generation_seed")
        ),
        api_base=(
            None
            if _nested_manifest_value(payload, "config_payload.api_base") is None
            else str(_nested_manifest_value(payload, "config_payload.api_base"))
        ),
        source_windows_path=(
            None
            if _nested_manifest_value(payload, "config_payload.source_windows_path") is None
            else str(_nested_manifest_value(payload, "config_payload.source_windows_path"))
        ),
        success_criteria_path=(
            None
            if _nested_manifest_value(payload, "config_payload.success_criteria_path") is None
            else str(_nested_manifest_value(payload, "config_payload.success_criteria_path"))
        ),
        target_envelopes_path=(
            None
            if _nested_manifest_value(payload, "config_payload.target_envelopes_path") is None
            else str(_nested_manifest_value(payload, "config_payload.target_envelopes_path"))
        ),
        max_cases=_optional_int(_nested_manifest_value(payload, "config_payload.max_cases")),
        max_iterations=_optional_int(
            _nested_manifest_value(payload, "config_payload.max_iterations")
        ),
    )


def _placeholder_run_provenance(
    *,
    run_id: str,
    manifest_path: str,
) -> RunProvenanceRecord:
    """Return a minimal provenance record when case artifacts lack manifest metadata."""
    return RunProvenanceRecord(
        run_id=run_id,
        manifest_path=manifest_path,
        git_sha=None,
        phase=None,
        prompt_template_id=None,
        model_id=None,
        corpus_manifest=None,
        split_manifest=None,
        generation_seed=None,
        api_base=None,
        source_windows_path=None,
        success_criteria_path=None,
        target_envelopes_path=None,
        max_cases=None,
        max_iterations=None,
    )


def _case_failure_labels(result: dict[str, Any]) -> tuple[str, ...]:
    """Derive stable failure labels from one serialized baseline result."""
    best_iteration_index = int(result["best_iteration_index"])
    best_iteration = result["iterations"][best_iteration_index]
    score_history = best_iteration["score_history"]
    labels: list[str] = []
    if not bool(score_history["semantic_tolerance_pass"]):
        labels.append(FAILURE_LABELS[0])
    if not bool(score_history["target_tolerance_pass"]):
        labels.append(FAILURE_LABELS[1])
    if not bool(score_history["length_guardrail_pass"]):
        labels.append(FAILURE_LABELS[2])
    if not bool(score_history["lexical_overlap_pass"]):
        labels.append(FAILURE_LABELS[3])
    if str(result["stop_reason"]) == "no_objective_improvement":
        labels.append(FAILURE_LABELS[4])
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
            "median_weighted_objective": _objective_median(
                [item.weighted_objective for item in items]
            ),
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
    for label in FAILURE_LABELS:
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


def _comparison_case_key(case: AggregatedCaseRecord) -> tuple[str, str, str, str]:
    """Return a stable key for matching cases across runs."""
    return (case.case_id, case.control_mode, case.source_work_id, case.target_work_id)


def _compare_run_provenance(
    reference: RunProvenanceRecord,
    candidate: RunProvenanceRecord,
) -> tuple[dict[str, bool], tuple[str, ...]]:
    """Return comparability checks for provenance fields that should remain invariant."""
    checks: dict[str, bool] = {}
    issues: list[str] = []
    for field in COMPARABILITY_INVARIANT_FIELDS:
        reference_value = getattr(reference, field)
        candidate_value = getattr(candidate, field)
        comparable = reference_value is not None and candidate_value is not None
        comparable = comparable and reference_value == candidate_value
        checks[field] = comparable
        if comparable:
            continue
        if reference_value is None or candidate_value is None:
            issues.append(f"{field} missing")
            continue
        issues.append(f"{field} mismatch: {reference_value!r} != {candidate_value!r}")
    return checks, tuple(issues)


def _bootstrap_mean_delta_interval(
    delta_values: list[float],
    *,
    reference_run_id: str,
    candidate_run_id: str,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return a deterministic bootstrap interval for the paired mean delta."""
    if not delta_values:
        return (0.0, 0.0)
    if len(delta_values) == 1:
        return (delta_values[0], delta_values[0])

    rng = random.Random(f"{reference_run_id}->{candidate_run_id}")
    sample_means: list[float] = []
    for _ in range(resamples):
        sample = [delta_values[rng.randrange(len(delta_values))] for _ in range(len(delta_values))]
        sample_means.append(sum(sample) / len(sample))
    sample_means.sort()
    lower_index = int(0.025 * (resamples - 1))
    upper_index = int(0.975 * (resamples - 1))
    return (sample_means[lower_index], sample_means[upper_index])


def _build_failure_transitions(
    *,
    reference_cases: dict[tuple[str, str, str, str], AggregatedCaseRecord],
    candidate_cases: dict[tuple[str, str, str, str], AggregatedCaseRecord],
    overlapping_keys: list[tuple[str, str, str, str]],
) -> tuple[FailureTransitionRecord, ...]:
    """Summarize per-label failure persistence, resolution, and introduction."""
    transitions: list[FailureTransitionRecord] = []
    for label in FAILURE_LABELS:
        persistent_count = 0
        resolved_count = 0
        introduced_count = 0
        for key in overlapping_keys:
            reference_has_label = label in reference_cases[key].failure_labels
            candidate_has_label = label in candidate_cases[key].failure_labels
            if reference_has_label and candidate_has_label:
                persistent_count += 1
            elif reference_has_label:
                resolved_count += 1
            elif candidate_has_label:
                introduced_count += 1
        transitions.append(
            FailureTransitionRecord(
                label=label,
                persistent_count=persistent_count,
                resolved_count=resolved_count,
                introduced_count=introduced_count,
            )
        )
    return tuple(transitions)


def _build_run_comparisons(
    cases: tuple[AggregatedCaseRecord, ...],
    *,
    run_ids: tuple[str, ...],
    run_provenance: dict[str, RunProvenanceRecord],
) -> tuple[RunComparisonRecord, ...]:
    """Build pairwise run comparisons over overlapping case identities."""
    by_run: dict[str, dict[tuple[str, str, str, str], AggregatedCaseRecord]] = {}
    for case in cases:
        by_run.setdefault(case.run_id, {})[_comparison_case_key(case)] = case

    comparisons: list[RunComparisonRecord] = []
    for index, reference_run_id in enumerate(run_ids):
        reference_cases = by_run.get(reference_run_id, {})
        for candidate_run_id in run_ids[index + 1 :]:
            candidate_cases = by_run.get(candidate_run_id, {})
            overlapping_keys = sorted(set(reference_cases) & set(candidate_cases))
            if not overlapping_keys:
                continue

            deltas: list[tuple[str, float]] = []
            reference_failure_case_count = 0
            candidate_failure_case_count = 0
            improved_case_count = 0
            worsened_case_count = 0
            unchanged_case_count = 0
            for key in overlapping_keys:
                reference_case = reference_cases[key]
                candidate_case = candidate_cases[key]
                delta = candidate_case.weighted_objective - reference_case.weighted_objective
                deltas.append((candidate_case.case_id, delta))
                if reference_case.failure_labels:
                    reference_failure_case_count += 1
                if candidate_case.failure_labels:
                    candidate_failure_case_count += 1
                if delta > 0.0:
                    improved_case_count += 1
                elif delta < 0.0:
                    worsened_case_count += 1
                else:
                    unchanged_case_count += 1

            delta_values = [value for _, value in deltas]
            ci_low, ci_high = _bootstrap_mean_delta_interval(
                delta_values,
                reference_run_id=reference_run_id,
                candidate_run_id=candidate_run_id,
            )
            largest_gain_case_id, largest_gain_delta = max(deltas, key=lambda item: item[1])
            largest_drop_case_id, largest_drop_delta = min(deltas, key=lambda item: item[1])
            reference_metadata = run_provenance[reference_run_id]
            candidate_metadata = run_provenance[candidate_run_id]
            comparability_checks, comparability_issues = _compare_run_provenance(
                reference_metadata,
                candidate_metadata,
            )
            comparisons.append(
                RunComparisonRecord(
                    reference_run_id=reference_run_id,
                    candidate_run_id=candidate_run_id,
                    overlapping_case_count=len(overlapping_keys),
                    reference_failure_case_count=reference_failure_case_count,
                    candidate_failure_case_count=candidate_failure_case_count,
                    failure_case_count_delta=(
                        candidate_failure_case_count - reference_failure_case_count
                    ),
                    mean_weighted_objective_delta=sum(delta_values) / len(delta_values),
                    median_weighted_objective_delta=_objective_median(delta_values),
                    mean_weighted_objective_delta_ci_low=ci_low,
                    mean_weighted_objective_delta_ci_high=ci_high,
                    mean_weighted_objective_delta_ci_excludes_zero=(ci_low > 0.0 or ci_high < 0.0),
                    bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                    improved_case_count=improved_case_count,
                    worsened_case_count=worsened_case_count,
                    unchanged_case_count=unchanged_case_count,
                    non_negative_case_share=(
                        (improved_case_count + unchanged_case_count) / len(overlapping_keys)
                    ),
                    largest_gain_case_id=largest_gain_case_id,
                    largest_gain_delta=largest_gain_delta,
                    largest_drop_case_id=largest_drop_case_id,
                    largest_drop_delta=largest_drop_delta,
                    comparable=all(comparability_checks.values()),
                    comparability_checks=comparability_checks,
                    comparability_issues=comparability_issues,
                    failure_transitions=_build_failure_transitions(
                        reference_cases=reference_cases,
                        candidate_cases=candidate_cases,
                        overlapping_keys=overlapping_keys,
                    ),
                )
            )
    return tuple(comparisons)


def _recommend_promotion(
    comparison: RunComparisonRecord,
    *,
    criteria: PromotionCriteria,
) -> PromotionRecommendation:
    """Evaluate explicit promotion criteria for one candidate-vs-incumbent pair."""
    criteria_results = {
        "comparable_provenance": (
            comparison.comparable or not criteria.require_comparable_provenance
        ),
        "overlap": comparison.overlapping_case_count >= criteria.min_overlapping_cases,
        "mean_delta": comparison.mean_weighted_objective_delta >= criteria.min_mean_delta,
        "median_delta": comparison.median_weighted_objective_delta >= criteria.min_median_delta,
        "non_negative_share": comparison.non_negative_case_share
        >= criteria.min_non_negative_share,
        "failure_case_delta": comparison.failure_case_count_delta
        <= criteria.max_failure_case_delta,
    }

    if criteria.require_comparable_provenance and not comparison.comparable:
        recommendation = "hold"
        issue_text = "; ".join(comparison.comparability_issues) or "unknown provenance mismatch"
        rationale = f"run provenance mismatch prevents disciplined promotion: {issue_text}"
    elif not criteria_results["overlap"]:
        recommendation = "hold"
        rationale = "insufficient overlapping cases for a disciplined promotion decision"
    elif all(criteria_results.values()):
        recommendation = "promote"
        rationale = "candidate clears all explicit promotion criteria"
    else:
        recommendation = "reject"
        failing = [name for name, passed in criteria_results.items() if not passed]
        rationale = "candidate failed: " + ", ".join(failing)

    return PromotionRecommendation(
        reference_run_id=comparison.reference_run_id,
        candidate_run_id=comparison.candidate_run_id,
        recommendation=recommendation,
        overlapping_case_count=comparison.overlapping_case_count,
        mean_weighted_objective_delta=comparison.mean_weighted_objective_delta,
        median_weighted_objective_delta=comparison.median_weighted_objective_delta,
        non_negative_case_share=comparison.non_negative_case_share,
        failure_case_count_delta=comparison.failure_case_count_delta,
        criteria_results=criteria_results,
        rationale=rationale,
    )


def _build_promotion_recommendations(
    *,
    run_ids: tuple[str, ...],
    comparisons: tuple[RunComparisonRecord, ...],
    criteria: PromotionCriteria,
) -> tuple[PromotionRecommendation, ...]:
    """Walk runs in order and recommend whether each candidate should replace the incumbent."""
    if not run_ids:
        return ()

    comparisons_by_pair = {
        (comparison.reference_run_id, comparison.candidate_run_id): comparison
        for comparison in comparisons
    }
    recommendations: list[PromotionRecommendation] = []
    incumbent_run_id = run_ids[0]
    for candidate_run_id in run_ids[1:]:
        comparison = comparisons_by_pair.get((incumbent_run_id, candidate_run_id))
        if comparison is None:
            recommendations.append(
                PromotionRecommendation(
                    reference_run_id=incumbent_run_id,
                    candidate_run_id=candidate_run_id,
                    recommendation="hold",
                    overlapping_case_count=0,
                    mean_weighted_objective_delta=0.0,
                    median_weighted_objective_delta=0.0,
                    non_negative_case_share=0.0,
                    failure_case_count_delta=0,
                    criteria_results={
                        "comparable_provenance": False,
                        "overlap": False,
                        "mean_delta": False,
                        "median_delta": False,
                        "non_negative_share": False,
                        "failure_case_delta": False,
                    },
                    rationale="candidate and incumbent have no overlapping cases to compare",
                )
            )
            continue
        recommendation = _recommend_promotion(comparison, criteria=criteria)
        recommendations.append(recommendation)
        if recommendation.recommendation == "promote":
            incumbent_run_id = candidate_run_id
    return tuple(recommendations)


def build_analysis_report(
    run_dirs: list[Path],
    *,
    promotion_criteria: PromotionCriteria | None = None,
) -> AnalysisReport:
    """Aggregate one or more immutable run directories into a Phase 6 report."""
    cases: list[AggregatedCaseRecord] = []
    run_provenance: dict[str, RunProvenanceRecord] = {}
    ordered_run_ids: list[str] = []
    for run_dir in run_dirs:
        provenance = _load_run_provenance(run_dir)
        run_provenance[provenance.run_id] = provenance
        ordered_run_ids.append(provenance.run_id)
        for case in _load_cases_from_run_dir(run_dir):
            if case.run_id not in run_provenance:
                run_provenance[case.run_id] = _placeholder_run_provenance(
                    run_id=case.run_id,
                    manifest_path=case.manifest_path,
                )
            ordered_run_ids.append(case.run_id)
            cases.append(case)

    run_ids = tuple(dict.fromkeys(ordered_run_ids))
    run_index = {run_id: index for index, run_id in enumerate(run_ids)}
    ordered_cases = tuple(
        sorted(
            cases,
            key=lambda case: (run_index.get(case.run_id, len(run_index)), case.case_id),
        )
    )
    resolved_promotion_criteria = (
        PromotionCriteria() if promotion_criteria is None else promotion_criteria
    )
    run_comparisons = _build_run_comparisons(
        ordered_cases,
        run_ids=run_ids,
        run_provenance=run_provenance,
    )
    return AnalysisReport(
        generated_at=utc_now(),
        run_ids=run_ids,
        cases=ordered_cases,
        run_provenance=run_provenance,
        run_summaries=_summarize_runs(ordered_cases),
        failure_modes=_summarize_failure_modes(ordered_cases),
        bias_slices={
            "by_work": _build_slice_records(ordered_cases, attribute="source_work_id"),
            "by_author": _build_slice_records(ordered_cases, attribute="source_author"),
        },
        promotion_criteria=resolved_promotion_criteria,
        run_comparisons=run_comparisons,
        promotion_recommendations=_build_promotion_recommendations(
            run_ids=run_ids,
            comparisons=run_comparisons,
            criteria=resolved_promotion_criteria,
        ),
        close_reading_notes=_build_close_reading_notes(ordered_cases),
    )


def _article_inputs_payload(report: AnalysisReport) -> dict[str, Any]:
    """Return compact article-facing inputs derived from the full report."""
    best_case = max(report.cases, key=lambda case: case.weighted_objective, default=None)
    weakest_case = min(report.cases, key=lambda case: case.weighted_objective, default=None)
    promotion_summary = {
        "final_incumbent_run_id": report.final_incumbent_run_id,
        "recommendation_counts": {
            label: sum(
                1
                for recommendation in report.promotion_recommendations
                if recommendation.recommendation == label
            )
            for label in ("promote", "hold", "reject")
        },
    }
    headline_findings = [
        (
            "Best observed reconstruction objective: "
            f"{best_case.case_id} in {best_case.run_id} "
            f"({best_case.weighted_objective:.4f})."
        )
        if best_case is not None
        else "No evaluated cases found."
    ]
    if report.final_incumbent_run_id is not None:
        headline_findings.append(
            "Promotion incumbent under explicit criteria: "
            f"{report.final_incumbent_run_id}."
        )
    if report.comparability_summary["noncomparable_comparison_count"] > 0:
        headline_findings.append(
            "Non-comparable comparisons flagged: "
            f"{report.comparability_summary['noncomparable_comparison_count']}."
        )
    return {
        "generated_at": report.generated_at,
        "total_runs": report.total_runs,
        "total_cases": report.total_cases,
        "headline_findings": headline_findings,
        "run_provenance": {
            run_id: record.to_dict() for run_id, record in report.run_provenance.items()
        },
        "run_summaries": report.run_summaries,
        "weakest_case": None if weakest_case is None else weakest_case.to_dict(),
        "failure_modes": report.failure_modes,
        "bias_slices": {
            key: [record.to_dict() for record in records]
            for key, records in report.bias_slices.items()
        },
        "promotion_criteria": report.promotion_criteria.to_dict(),
        "run_comparisons": [record.to_dict() for record in report.run_comparisons],
        "promotion_recommendations": [
            recommendation.to_dict() for recommendation in report.promotion_recommendations
        ],
        "promotion_summary": promotion_summary,
        "comparability_summary": report.comparability_summary,
        "failure_transition_summary": report.failure_transition_summary,
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
            "promotion_criteria": report.promotion_criteria.to_dict(),
            "comparability_invariant_fields": list(COMPARABILITY_INVARIANT_FIELDS),
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
            "research/comparable_comparison_count": float(
                report.comparability_summary["comparable_comparison_count"]
            ),
            "research/noncomparable_comparison_count": float(
                report.comparability_summary["noncomparable_comparison_count"]
            ),
            "research/promotion_promote_count": float(
                sum(
                    1
                    for recommendation in report.promotion_recommendations
                    if recommendation.recommendation == "promote"
                )
            ),
            "research/promotion_hold_count": float(
                sum(
                    1
                    for recommendation in report.promotion_recommendations
                    if recommendation.recommendation == "hold"
                )
            ),
            "research/promotion_reject_count": float(
                sum(
                    1
                    for recommendation in report.promotion_recommendations
                    if recommendation.recommendation == "reject"
                )
            ),
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

    run_comparison_table = wandb.Table(
        columns=[
            "reference_run_id",
            "candidate_run_id",
            "overlapping_case_count",
            "comparable",
            "comparability_issues",
            "mean_weighted_objective_delta",
            "mean_weighted_objective_delta_ci_low",
            "mean_weighted_objective_delta_ci_high",
            "mean_weighted_objective_delta_ci_excludes_zero",
            "median_weighted_objective_delta",
            "improved_case_count",
            "worsened_case_count",
            "unchanged_case_count",
            "non_negative_case_share",
            "failure_case_count_delta",
            "largest_gain_case_id",
            "largest_gain_delta",
            "largest_drop_case_id",
            "largest_drop_delta",
        ],
        data=[
            [
                record.reference_run_id,
                record.candidate_run_id,
                record.overlapping_case_count,
                record.comparable,
                list(record.comparability_issues),
                record.mean_weighted_objective_delta,
                record.mean_weighted_objective_delta_ci_low,
                record.mean_weighted_objective_delta_ci_high,
                record.mean_weighted_objective_delta_ci_excludes_zero,
                record.median_weighted_objective_delta,
                record.improved_case_count,
                record.worsened_case_count,
                record.unchanged_case_count,
                record.non_negative_case_share,
                record.failure_case_count_delta,
                record.largest_gain_case_id,
                record.largest_gain_delta,
                record.largest_drop_case_id,
                record.largest_drop_delta,
            ]
            for record in report.run_comparisons
        ],
    )
    run.log({"analysis/run_comparison_table": run_comparison_table})

    failure_transition_table = wandb.Table(
        columns=[
            "reference_run_id",
            "candidate_run_id",
            "label",
            "persistent_count",
            "resolved_count",
            "introduced_count",
        ],
        data=[
            [
                record.reference_run_id,
                record.candidate_run_id,
                transition.label,
                transition.persistent_count,
                transition.resolved_count,
                transition.introduced_count,
            ]
            for record in report.run_comparisons
            for transition in record.failure_transitions
        ],
    )
    run.log({"analysis/failure_transition_table": failure_transition_table})

    promotion_table = wandb.Table(
        columns=[
            "reference_run_id",
            "candidate_run_id",
            "recommendation",
            "overlapping_case_count",
            "mean_weighted_objective_delta",
            "median_weighted_objective_delta",
            "non_negative_case_share",
            "failure_case_count_delta",
            "criteria_results",
            "rationale",
        ],
        data=[
            [
                recommendation.reference_run_id,
                recommendation.candidate_run_id,
                recommendation.recommendation,
                recommendation.overlapping_case_count,
                recommendation.mean_weighted_objective_delta,
                recommendation.median_weighted_objective_delta,
                recommendation.non_negative_case_share,
                recommendation.failure_case_count_delta,
                recommendation.criteria_results,
                recommendation.rationale,
            ]
            for recommendation in report.promotion_recommendations
        ],
    )
    run.log({"analysis/promotion_table": promotion_table})

    run_provenance_table = wandb.Table(
        columns=[
            "run_id",
            "git_sha",
            "phase",
            "prompt_template_id",
            "model_id",
            "generation_seed",
            "api_base",
            "corpus_manifest",
            "split_manifest",
            "source_windows_path",
            "success_criteria_path",
            "target_envelopes_path",
            "max_cases",
            "max_iterations",
        ],
        data=[
            [
                provenance.run_id,
                provenance.git_sha,
                provenance.phase,
                provenance.prompt_template_id,
                provenance.model_id,
                provenance.generation_seed,
                provenance.api_base,
                provenance.corpus_manifest,
                provenance.split_manifest,
                provenance.source_windows_path,
                provenance.success_criteria_path,
                provenance.target_envelopes_path,
                provenance.max_cases,
                provenance.max_iterations,
            ]
            for provenance in report.run_provenance.values()
        ],
    )
    run.log({"analysis/run_provenance_table": run_provenance_table})

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
            "promotion_summary": article_inputs["promotion_summary"],
            "comparability_summary": article_inputs["comparability_summary"],
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
        "## Comparability",
        "",
        (
            "Comparable pairwise comparisons: "
            f"{report.comparability_summary['comparable_comparison_count']}"
        ),
        (
            "Non-comparable pairwise comparisons: "
            f"{report.comparability_summary['noncomparable_comparison_count']}"
        ),
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
            "## Run Provenance",
            "",
        ]
    )
    for provenance in report.run_provenance.values():
        report_lines.append(
            f"- `{provenance.run_id}`: model={provenance.model_id or 'unknown'}, "
            f"seed={provenance.generation_seed}, "
            f"prompt={provenance.prompt_template_id or 'unknown'}, "
            f"git_sha={provenance.git_sha or 'unknown'}"
        )
    report_lines.extend(
        [
            "",
            "## Close Reading Queue",
            "",
        ]
    )
    if report.run_comparisons:
        report_lines.extend(["## Run Comparisons", ""])
        for record in report.run_comparisons:
            report_lines.append(
                f"- `{record.candidate_run_id}` vs `{record.reference_run_id}`: "
                f"overlap={record.overlapping_case_count}, "
                f"comparable={record.comparable}, "
                f"mean_delta={record.mean_weighted_objective_delta:.4f}, "
                f"mean_delta_ci=[{record.mean_weighted_objective_delta_ci_low:.4f}, "
                f"{record.mean_weighted_objective_delta_ci_high:.4f}], "
                f"ci_excludes_zero={record.mean_weighted_objective_delta_ci_excludes_zero}, "
                f"median_delta={record.median_weighted_objective_delta:.4f}, "
                f"non_negative_share={record.non_negative_case_share:.2f}, "
                f"failure_case_delta={record.failure_case_count_delta}"
            )
            if record.comparability_issues:
                report_lines.append(
                    "  comparability_issues=" + "; ".join(record.comparability_issues)
                )
            for transition in record.failure_transitions:
                report_lines.append(
                    f"  {transition.label}: persistent={transition.persistent_count}, "
                    f"resolved={transition.resolved_count}, "
                    f"introduced={transition.introduced_count}"
                )
        report_lines.extend(["", "## Promotion Recommendations", ""])
        for recommendation in report.promotion_recommendations:
            report_lines.append(
                f"- `{recommendation.candidate_run_id}` against "
                f"`{recommendation.reference_run_id}`: "
                f"{recommendation.recommendation}; {recommendation.rationale}."
            )
        report_lines.append("")
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
        "--promotion-min-overlapping-cases",
        type=int,
        default=PromotionCriteria().min_overlapping_cases,
    )
    parser.add_argument(
        "--promotion-min-mean-delta",
        type=float,
        default=PromotionCriteria().min_mean_delta,
    )
    parser.add_argument(
        "--promotion-min-median-delta",
        type=float,
        default=PromotionCriteria().min_median_delta,
    )
    parser.add_argument(
        "--promotion-min-non-negative-share",
        type=float,
        default=PromotionCriteria().min_non_negative_share,
    )
    parser.add_argument(
        "--promotion-max-failure-case-delta",
        type=int,
        default=PromotionCriteria().max_failure_case_delta,
    )
    parser.add_argument(
        "--promotion-require-comparable-provenance",
        action=argparse.BooleanOptionalAction,
        default=PromotionCriteria().require_comparable_provenance,
        help="Require invariant run provenance to match before promotion can occur.",
    )
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
    report = build_analysis_report(
        run_dirs=run_dirs,
        promotion_criteria=PromotionCriteria(
            min_overlapping_cases=args.promotion_min_overlapping_cases,
            min_mean_delta=args.promotion_min_mean_delta,
            min_median_delta=args.promotion_min_median_delta,
            min_non_negative_share=args.promotion_min_non_negative_share,
            max_failure_case_delta=args.promotion_max_failure_case_delta,
            require_comparable_provenance=args.promotion_require_comparable_provenance,
        ),
    )
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
