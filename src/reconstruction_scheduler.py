#!/usr/bin/env python3
"""
Guided experiment scheduler for reconstruction runs.

The scheduler is intentionally narrow. It executes a finite JSON-defined queue
of commands, extracts a scored metric from the resulting run artifacts, and
records keep/discard/failed decisions in append-only schedule logs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconstruction_contract import PROJECT_ROOT, ReconstructionPaths, to_project_relative

DEFAULT_SCHEDULER_ROOT = ReconstructionPaths().analysis_dir / "schedules"


@dataclass(frozen=True)
class ExperimentSpec:
    """One guided experiment entry in a scheduler plan."""

    experiment_id: str
    run_id: str
    phase: str
    command: tuple[str, ...]
    timeout_seconds: int
    metric_path_template: str
    metric_key: str
    higher_is_better: bool = True

    def render_command(self) -> tuple[str, ...]:
        """Render the command with run-aware placeholders."""
        return tuple(part.replace("{run_id}", self.run_id) for part in self.command)

    def metric_path(self, project_root: Path) -> Path:
        """Resolve the experiment metric path under the given project root."""
        rendered = self.metric_path_template.format(run_id=self.run_id)
        return (project_root / rendered).resolve()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            **asdict(self),
            "command": list(self.command),
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Recorded outcome for one scheduled experiment."""

    experiment_id: str
    run_id: str
    phase: str
    status: str
    return_code: int | None
    metric_value: float | None
    metric_path: str
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    result_path: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class SchedulePlan:
    """Finite list of experiments to execute in order."""

    schedule_id: str
    experiments: tuple[ExperimentSpec, ...]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON document from disk."""
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


def _resolve_metric_key(payload: dict[str, Any], metric_key: str) -> float:
    """Read a dotted metric key from nested JSON payloads."""
    current: Any = payload
    for key in metric_key.split("."):
        current = current[key]
    return float(current)


def _decision_for_metric(
    metric_value: float,
    *,
    incumbent_metric: float | None,
    higher_is_better: bool,
) -> str:
    """Return keep/discard based on whether the candidate improves the incumbent."""
    if incumbent_metric is None:
        return "keep"
    if higher_is_better:
        return "keep" if metric_value > incumbent_metric else "discard"
    return "keep" if metric_value < incumbent_metric else "discard"


def _artifact_paths(
    *,
    experiment_dir: Path,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Return the standard per-experiment scheduler artifact paths."""
    return (
        experiment_dir / "stdout.log",
        experiment_dir / "stderr.log",
        experiment_dir / "result.json",
    )


def _coerce_process_output(value: str | bytes | None) -> str:
    """Normalize subprocess output into text for persisted logs."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _decision_score(status: str) -> float:
    """Map keep/discard/failed into an ordered numeric signal."""
    return {
        "keep": 1.0,
        "discard": 0.0,
        "failed": -1.0,
    }[status]


def _experiment_artifact_paths(
    *,
    spec: ExperimentSpec,
    result: ExperimentResult,
    project_root: Path,
) -> dict[str, Path]:
    """Return meaningful scheduler and run artifacts for one experiment."""
    paths = {
        "result.json": (project_root / result.result_path).resolve(),
        "stdout.log": (project_root / result.stdout_path).resolve(),
        "stderr.log": (project_root / result.stderr_path).resolve(),
    }

    metric_path = spec.metric_path(project_root)
    if metric_path.exists():
        paths["metric.json"] = metric_path

    run_dir = ReconstructionPaths(project_root=project_root).run_dir(spec.run_id)
    for filename in (
        "manifest.json",
        "prompt_baseline_summary.json",
        "prompt_baseline_cases.json",
        "prompt_baseline_report.md",
    ):
        candidate = run_dir / filename
        if candidate.exists():
            paths[filename] = candidate
    return paths


def _scheduler_metric_payload(metric_path: Path) -> dict[str, Any] | None:
    """Load the metric payload if the produced artifact exists."""
    if not metric_path.exists():
        return None
    try:
        return _load_json(metric_path)
    except (OSError, json.JSONDecodeError):
        return None


def _numeric_scheduler_metrics(
    *,
    result: ExperimentResult,
    incumbent_metric_before: float | None,
    incumbent_metric_after: float | None,
    metric_payload: dict[str, Any] | None,
) -> dict[str, float]:
    """Build compact numeric metrics suitable for W&B logging."""
    metrics = {
        "scheduler/decision": _decision_score(result.status),
        "scheduler/status_code": _decision_score(result.status),
        "scheduler/duration_seconds": result.duration_seconds,
    }
    if result.return_code is not None:
        metrics["scheduler/return_code"] = float(result.return_code)
    if incumbent_metric_before is not None:
        metrics["scheduler/incumbent_metric_before"] = incumbent_metric_before
    if incumbent_metric_after is not None:
        metrics["scheduler/incumbent_metric_after"] = incumbent_metric_after
    if result.metric_value is not None:
        metrics["reconstruction/metric_value"] = result.metric_value

    if metric_payload is None:
        return metrics

    total_cases = metric_payload.get("total_cases")
    if isinstance(total_cases, int | float):
        metrics["reconstruction/total_cases"] = float(total_cases)

    controls = metric_payload.get("controls")
    if isinstance(controls, dict):
        for control_mode, stats in controls.items():
            if not isinstance(control_mode, str) or not isinstance(stats, dict):
                continue
            count = stats.get("count")
            mean_weighted_objective = stats.get("mean_weighted_objective")
            median_weighted_objective = stats.get("median_weighted_objective")
            if isinstance(count, int | float):
                metrics[f"reconstruction/control_{control_mode}_count"] = float(count)
            if isinstance(mean_weighted_objective, int | float):
                metrics[
                    f"reconstruction/control_{control_mode}_mean_weighted_objective"
                ] = float(mean_weighted_objective)
            if isinstance(median_weighted_objective, int | float):
                metrics[
                    f"reconstruction/control_{control_mode}_median_weighted_objective"
                ] = float(median_weighted_objective)
    return metrics


def _log_experiment_to_wandb(
    *,
    schedule_id: str,
    spec: ExperimentSpec,
    result: ExperimentResult,
    project_root: Path,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    incumbent_metric_before: float | None,
    incumbent_metric_after: float | None,
) -> None:
    """Log one scheduled experiment as an optional W&B run."""
    wandb = _load_wandb_module()
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        group=schedule_id,
        job_type="reconstruction_scheduler_experiment",
        tags=[spec.phase, "phase6", "guided-scheduler"],
        name=spec.experiment_id,
        config={
            "schedule_id": schedule_id,
            "experiment_id": spec.experiment_id,
            "run_id": spec.run_id,
            "phase": spec.phase,
            "command": list(spec.command),
            "timeout_seconds": spec.timeout_seconds,
            "metric_path_template": spec.metric_path_template,
            "metric_key": spec.metric_key,
            "higher_is_better": spec.higher_is_better,
            "scheduler_inspiration": "Andrej Karpathy autoresearch",
        },
    )
    metric_payload = _scheduler_metric_payload(spec.metric_path(project_root))
    run.log(
        _numeric_scheduler_metrics(
            result=result,
            incumbent_metric_before=incumbent_metric_before,
            incumbent_metric_after=incumbent_metric_after,
            metric_payload=metric_payload,
        )
    )

    artifact_paths = _experiment_artifact_paths(
        spec=spec,
        result=result,
        project_root=project_root,
    )
    run.summary.update(
        {
            "scheduler_decision": result.status,
            "metric_path": result.metric_path,
            "artifact_paths": {
                "metric_path": result.metric_path,
                "stdout_path": result.stdout_path,
                "stderr_path": result.stderr_path,
                "result_path": result.result_path,
            },
            "reconstruction_summary": metric_payload,
            "error_message": result.error_message,
        }
    )

    artifact = wandb.Artifact(
        name=f"{schedule_id}-{spec.experiment_id}",
        type="reconstruction-scheduler-experiment",
    )
    for artifact_name, artifact_path in artifact_paths.items():
        artifact.add_file(str(artifact_path), name=artifact_name)
    run.log_artifact(artifact)
    run.finish()


def run_experiment(
    spec: ExperimentSpec,
    *,
    project_root: Path,
    scheduler_dir: Path,
    incumbent_metric: float | None,
) -> ExperimentResult:
    """Execute one scheduled experiment and persist its result artifacts."""
    experiment_dir = scheduler_dir / spec.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path, result_path = _artifact_paths(
        experiment_dir=experiment_dir,
        project_root=project_root,
    )

    started_at = utc_now()
    started_dt = datetime.now(UTC)
    command = spec.render_command()
    status = "failed"
    return_code: int | None = None
    metric_value: float | None = None
    error_message: str | None = None

    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        return_code = completed.returncode
        if completed.returncode != 0:
            error_message = f"command exited with code {completed.returncode}"
        else:
            metric_payload = _load_json(spec.metric_path(project_root))
            metric_value = _resolve_metric_key(metric_payload, spec.metric_key)
            status = _decision_for_metric(
                metric_value,
                incumbent_metric=incumbent_metric,
                higher_is_better=spec.higher_is_better,
            )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_coerce_process_output(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_coerce_process_output(exc.stderr), encoding="utf-8")
        error_message = f"command timed out after {spec.timeout_seconds} seconds"
    except FileNotFoundError as exc:
        error_message = str(exc)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        error_message = f"metric extraction failed: {exc}"

    ended_at = utc_now()
    ended_dt = datetime.now(UTC)
    duration_seconds = max(0.0, (ended_dt - started_dt).total_seconds())
    if metric_value is not None and status == "failed":
        status = "discard"

    result = ExperimentResult(
        experiment_id=spec.experiment_id,
        run_id=spec.run_id,
        phase=spec.phase,
        status=status,
        return_code=return_code,
        metric_value=metric_value,
        metric_path=to_project_relative(spec.metric_path(project_root), project_root),
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        stdout_path=to_project_relative(stdout_path, project_root),
        stderr_path=to_project_relative(stderr_path, project_root),
        result_path=to_project_relative(result_path, project_root),
        error_message=error_message,
    )
    _write_json(result_path, result.to_dict())
    return result


def run_schedule(
    plan: SchedulePlan,
    *,
    project_root: Path = PROJECT_ROOT,
    scheduler_root: Path = DEFAULT_SCHEDULER_ROOT,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str = "offline",
) -> dict[str, Any]:
    """Execute the finite experiment queue and persist append-only schedule logs."""
    schedule_dir = scheduler_root / plan.schedule_id
    schedule_dir.mkdir(parents=True, exist_ok=True)
    results_path = schedule_dir / "schedule_results.jsonl"
    summary_path = schedule_dir / "schedule_summary.json"
    if results_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"schedule results already exists for {plan.schedule_id!r}: {schedule_dir}"
        )

    results: list[ExperimentResult] = []
    incumbent_metric: float | None = None
    kept_experiments: list[str] = []
    kept_run_ids: list[str] = []
    discarded_experiments: list[str] = []
    discarded_run_ids: list[str] = []
    failed_experiments: list[str] = []
    failed_run_ids: list[str] = []

    with open(results_path, "w", encoding="utf-8") as handle:
        for spec in plan.experiments:
            incumbent_metric_before = incumbent_metric
            result = run_experiment(
                spec,
                project_root=project_root,
                scheduler_dir=schedule_dir,
                incumbent_metric=incumbent_metric,
            )
            results.append(result)
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

            if result.status == "keep" and result.metric_value is not None:
                incumbent_metric = result.metric_value
                kept_experiments.append(result.experiment_id)
                kept_run_ids.append(result.run_id)
            elif result.status == "discard":
                discarded_experiments.append(result.experiment_id)
                discarded_run_ids.append(result.run_id)
            else:
                failed_experiments.append(result.experiment_id)
                failed_run_ids.append(result.run_id)

            if wandb_project:
                _log_experiment_to_wandb(
                    schedule_id=plan.schedule_id,
                    spec=spec,
                    result=result,
                    project_root=project_root,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    wandb_mode=wandb_mode,
                    incumbent_metric_before=incumbent_metric_before,
                    incumbent_metric_after=incumbent_metric,
                )

    summary = {
        "schedule_id": plan.schedule_id,
        "generated_at": utc_now(),
        "total_experiments": len(plan.experiments),
        "kept_experiments": kept_experiments,
        "kept_run_ids": kept_run_ids,
        "discarded_experiments": discarded_experiments,
        "discarded_run_ids": discarded_run_ids,
        "failed_experiments": failed_experiments,
        "failed_run_ids": failed_run_ids,
        "best_metric_value": incumbent_metric,
        "results_path": to_project_relative(results_path, project_root),
    }
    _write_json(summary_path, summary)
    return summary


def load_schedule_plan(path: Path) -> SchedulePlan:
    """Load a scheduler plan from JSON."""
    payload = _load_json(path)
    experiments = tuple(
        ExperimentSpec(
            experiment_id=str(record["experiment_id"]),
            run_id=str(record["run_id"]),
            phase=str(record["phase"]),
            command=tuple(str(part) for part in record["command"]),
            timeout_seconds=int(record["timeout_seconds"]),
            metric_path_template=str(record["metric_path_template"]),
            metric_key=str(record["metric_key"]),
            higher_is_better=bool(record.get("higher_is_better", True)),
        )
        for record in payload["experiments"]
    )
    return SchedulePlan(schedule_id=str(payload["schedule_id"]), experiments=experiments)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for guided scheduler plans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--scheduler-root", type=Path, default=DEFAULT_SCHEDULER_ROOT)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="offline")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one scheduler plan from disk."""
    args = build_argument_parser().parse_args(argv)
    plan = load_schedule_plan(args.plan_path)
    summary = run_schedule(
        plan,
        project_root=PROJECT_ROOT,
        scheduler_root=args.scheduler_root,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    print(f"Executed {summary['total_experiments']} experiments from {plan.schedule_id}.")
    print(f"Kept: {', '.join(summary['kept_experiments']) or 'none'}")
    print(f"Failed: {', '.join(summary['failed_experiments']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
