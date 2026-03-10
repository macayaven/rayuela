from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest

import reconstruction_scheduler


def _write_metric_command(metric_path: Path, value: float) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import json; "
            f"path = Path({str(metric_path)!r}); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            f"path.write_text(json.dumps({{'controls': {{'style_shift': "
            f"{{'mean_weighted_objective': {value}}}}}}}) + '\\n', encoding='utf-8')"
        ),
    ]


def test_run_experiment_extracts_metric_and_marks_first_success_keep(tmp_path: Path) -> None:
    project_root = tmp_path
    metric_path = project_root / "outputs" / "reconstruction" / "runs" / "phase4-a" / "summary.json"
    spec = reconstruction_scheduler.ExperimentSpec(
        experiment_id="baseline-a",
        run_id="phase4-a",
        phase="phase-4-prompt-baselines",
        command=tuple(_write_metric_command(metric_path, 0.61)),
        timeout_seconds=30,
        metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
        metric_key="controls.style_shift.mean_weighted_objective",
        higher_is_better=True,
    )

    result = reconstruction_scheduler.run_experiment(
        spec,
        project_root=project_root,
        scheduler_dir=tmp_path / "scheduler",
        incumbent_metric=None,
    )

    assert result.status == "keep"
    assert result.metric_value == 0.61
    assert result.metric_path == "outputs/reconstruction/runs/phase4-a/summary.json"
    assert Path(project_root / result.stdout_path).exists()
    assert Path(project_root / result.result_path).exists()


def test_run_experiment_marks_worse_result_as_discard(tmp_path: Path) -> None:
    project_root = tmp_path
    metric_path = project_root / "outputs" / "reconstruction" / "runs" / "phase4-b" / "summary.json"
    spec = reconstruction_scheduler.ExperimentSpec(
        experiment_id="baseline-b",
        run_id="phase4-b",
        phase="phase-4-prompt-baselines",
        command=tuple(_write_metric_command(metric_path, 0.41)),
        timeout_seconds=30,
        metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
        metric_key="controls.style_shift.mean_weighted_objective",
        higher_is_better=True,
    )

    result = reconstruction_scheduler.run_experiment(
        spec,
        project_root=project_root,
        scheduler_dir=tmp_path / "scheduler",
        incumbent_metric=0.55,
    )

    assert result.status == "discard"
    assert result.metric_value == 0.41


def test_run_experiment_marks_timeout_as_failed(tmp_path: Path) -> None:
    spec = reconstruction_scheduler.ExperimentSpec(
        experiment_id="timeout-case",
        run_id="phase4-timeout",
        phase="phase-4-prompt-baselines",
        command=(sys.executable, "-c", "import time; time.sleep(1.0)"),
        timeout_seconds=0,
        metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
        metric_key="controls.style_shift.mean_weighted_objective",
        higher_is_better=True,
    )

    result = reconstruction_scheduler.run_experiment(
        spec,
        project_root=tmp_path,
        scheduler_dir=tmp_path / "scheduler",
        incumbent_metric=None,
    )

    assert result.status == "failed"
    assert result.metric_value is None
    assert result.error_message is not None
    assert "timed out" in result.error_message.lower()


def test_run_schedule_writes_append_only_results_and_summary(tmp_path: Path) -> None:
    project_root = tmp_path
    first_metric = (
        project_root / "outputs" / "reconstruction" / "runs" / "phase4-a" / "summary.json"
    )
    second_metric = (
        project_root / "outputs" / "reconstruction" / "runs" / "phase4-b" / "summary.json"
    )
    plan = reconstruction_scheduler.SchedulePlan(
        schedule_id="guided-20260310a",
        experiments=(
            reconstruction_scheduler.ExperimentSpec(
                experiment_id="baseline-a",
                run_id="phase4-a",
                phase="phase-4-prompt-baselines",
                command=tuple(_write_metric_command(first_metric, 0.57)),
                timeout_seconds=30,
                metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
                metric_key="controls.style_shift.mean_weighted_objective",
                higher_is_better=True,
            ),
            reconstruction_scheduler.ExperimentSpec(
                experiment_id="baseline-b",
                run_id="phase4-b",
                phase="phase-4-prompt-baselines",
                command=tuple(_write_metric_command(second_metric, 0.49)),
                timeout_seconds=30,
                metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
                metric_key="controls.style_shift.mean_weighted_objective",
                higher_is_better=True,
            ),
        ),
    )

    summary = reconstruction_scheduler.run_schedule(
        plan,
        project_root=project_root,
        scheduler_root=tmp_path / "scheduler",
    )

    results_path = tmp_path / "scheduler" / "guided-20260310a" / "schedule_results.jsonl"
    summary_path = tmp_path / "scheduler" / "guided-20260310a" / "schedule_summary.json"
    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["total_experiments"] == 2
    assert summary["best_metric_value"] == 0.57
    assert len(lines) == 2
    assert payload["kept_experiments"] == ["baseline-a"]
    assert payload["kept_run_ids"] == ["phase4-a"]
    assert payload["discarded_experiments"] == ["baseline-b"]
    assert payload["discarded_run_ids"] == ["phase4-b"]


def test_run_schedule_rejects_reuse_of_existing_schedule_id(tmp_path: Path) -> None:
    project_root = tmp_path
    metric_path = project_root / "outputs" / "reconstruction" / "runs" / "phase4-a" / "summary.json"
    plan = reconstruction_scheduler.SchedulePlan(
        schedule_id="guided-20260310a",
        experiments=(
            reconstruction_scheduler.ExperimentSpec(
                experiment_id="baseline-a",
                run_id="phase4-a",
                phase="phase-4-prompt-baselines",
                command=tuple(_write_metric_command(metric_path, 0.57)),
                timeout_seconds=30,
                metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
                metric_key="controls.style_shift.mean_weighted_objective",
                higher_is_better=True,
            ),
        ),
    )

    reconstruction_scheduler.run_schedule(
        plan,
        project_root=project_root,
        scheduler_root=tmp_path / "scheduler",
    )

    with pytest.raises(FileExistsError, match="already exists"):
        reconstruction_scheduler.run_schedule(
            plan,
            project_root=project_root,
            scheduler_root=tmp_path / "scheduler",
        )


def test_run_schedule_optionally_logs_experiments_to_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {
        "init": [],
        "logs": [],
        "summary_updates": [],
        "artifacts": [],
        "finished": 0,
    }

    class _Artifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type
            self.files: list[tuple[str, str | None]] = []

        def add_file(self, local_path: str, name: str | None = None) -> None:
            self.files.append((local_path, name))

    class _Summary:
        def update(self, payload: dict[str, object]) -> None:
            cast_updates = calls["summary_updates"]
            assert isinstance(cast_updates, list)
            cast_updates.append(payload)

    class _Run:
        def __init__(self) -> None:
            self.summary = _Summary()

        def log(self, payload: dict[str, object]) -> None:
            cast_logs = calls["logs"]
            assert isinstance(cast_logs, list)
            cast_logs.append(payload)

        def log_artifact(self, artifact: _Artifact) -> None:
            cast_artifacts = calls["artifacts"]
            assert isinstance(cast_artifacts, list)
            cast_artifacts.append(
                {
                    "name": artifact.name,
                    "type": artifact.type,
                    "files": list(artifact.files),
                }
            )

        def finish(self) -> None:
            calls["finished"] = cast(int, calls["finished"]) + 1

    class _WandbStub:
        Artifact = _Artifact

        @staticmethod
        def init(**kwargs: object) -> _Run:
            cast_init = calls["init"]
            assert isinstance(cast_init, list)
            cast_init.append(kwargs)
            return _Run()

    monkeypatch.setattr(reconstruction_scheduler, "_load_wandb_module", lambda: _WandbStub)

    project_root = tmp_path
    kept_metric = project_root / "outputs" / "reconstruction" / "runs" / "phase4-a" / "summary.json"
    discarded_metric = (
        project_root / "outputs" / "reconstruction" / "runs" / "phase4-b" / "summary.json"
    )
    plan = reconstruction_scheduler.SchedulePlan(
        schedule_id="guided-20260310a",
        experiments=(
            reconstruction_scheduler.ExperimentSpec(
                experiment_id="baseline-a",
                run_id="phase4-a",
                phase="phase-4-prompt-baselines",
                command=tuple(_write_metric_command(kept_metric, 0.63)),
                timeout_seconds=30,
                metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
                metric_key="controls.style_shift.mean_weighted_objective",
                higher_is_better=True,
            ),
            reconstruction_scheduler.ExperimentSpec(
                experiment_id="baseline-b",
                run_id="phase4-b",
                phase="phase-4-prompt-baselines",
                command=tuple(_write_metric_command(discarded_metric, 0.41)),
                timeout_seconds=30,
                metric_path_template="outputs/reconstruction/runs/{run_id}/summary.json",
                metric_key="controls.style_shift.mean_weighted_objective",
                higher_is_better=True,
            ),
        ),
    )

    reconstruction_scheduler.run_schedule(
        plan,
        project_root=project_root,
        scheduler_root=tmp_path / "scheduler",
        wandb_project="rayuela",
        wandb_entity="macayaven",
        wandb_mode="offline",
    )

    init_calls = calls["init"]
    assert isinstance(init_calls, list)
    assert len(init_calls) == 2
    assert init_calls[0] == {
        "project": "rayuela",
        "entity": "macayaven",
        "mode": "offline",
        "group": "guided-20260310a",
        "job_type": "reconstruction_scheduler_experiment",
        "tags": ["phase-4-prompt-baselines", "phase6", "guided-scheduler"],
        "name": "baseline-a",
        "config": {
            "schedule_id": "guided-20260310a",
            "experiment_id": "baseline-a",
            "run_id": "phase4-a",
            "phase": "phase-4-prompt-baselines",
            "command": list(plan.experiments[0].command),
            "timeout_seconds": 30,
            "metric_path_template": "outputs/reconstruction/runs/{run_id}/summary.json",
            "metric_key": "controls.style_shift.mean_weighted_objective",
            "higher_is_better": True,
            "scheduler_inspiration": "Andrej Karpathy autoresearch",
        },
    }
    assert init_calls[1]["name"] == "baseline-b"

    log_calls = calls["logs"]
    assert isinstance(log_calls, list)
    assert log_calls[0]["scheduler/decision"] == 1.0
    assert log_calls[0]["scheduler/status_code"] == 1.0
    assert log_calls[0]["reconstruction/metric_value"] == 0.63
    assert log_calls[0]["reconstruction/control_style_shift_mean_weighted_objective"] == 0.63
    assert log_calls[1]["scheduler/decision"] == 0.0
    assert log_calls[1]["scheduler/status_code"] == 0.0
    assert log_calls[1]["reconstruction/metric_value"] == 0.41

    summary_updates = calls["summary_updates"]
    assert isinstance(summary_updates, list)
    assert summary_updates[0]["scheduler_decision"] == "keep"
    assert summary_updates[0]["metric_path"] == "outputs/reconstruction/runs/phase4-a/summary.json"
    assert summary_updates[0]["artifact_paths"]["result_path"].endswith(
        "guided-20260310a/baseline-a/result.json"
    )
    assert summary_updates[1]["scheduler_decision"] == "discard"

    artifact_calls = calls["artifacts"]
    assert isinstance(artifact_calls, list)
    assert artifact_calls[0]["type"] == "reconstruction-scheduler-experiment"
    assert {
        artifact_name
        for _, artifact_name in artifact_calls[0]["files"]
        if artifact_name is not None
    } >= {"result.json", "stdout.log", "stderr.log", "metric.json"}
    assert calls["finished"] == 2
