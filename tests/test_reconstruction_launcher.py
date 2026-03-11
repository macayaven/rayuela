from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import reconstruction_launcher


def _write_plan(path: Path, *, schedule_id: str = "guided-20260311a") -> Path:
    payload = {
        "schedule_id": schedule_id,
        "experiments": [
            {
                "experiment_id": "baseline-a",
                "run_id": "phase4-a",
                "phase": "phase-4-prompt-baselines",
                "command": [
                    "/tmp/venv/bin/python",
                    "src/reconstruction_baselines.py",
                    "--run-id",
                    "{run_id}",
                    "--api-base",
                    "http://localhost:8000/v1",
                ],
                "timeout_seconds": 3600,
                "metric_path_template": (
                    "outputs/reconstruction/runs/{run_id}/prompt_baseline_summary.json"
                ),
                "metric_key": "controls.style_shift.mean_weighted_objective",
                "higher_is_better": True,
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_env(path: Path, *, include_wandb: bool = True) -> Path:
    lines = []
    if include_wandb:
        lines.append("WANDB_API_KEY=test-key")
    lines.append("HF_TOKEN=test-token")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_validate_launch_prerequisites_requires_expected_env_keys(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")
    env_path = _write_env(tmp_path / ".env", include_wandb=False)
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=plan_path,
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
        wandb_project="rayuela",
        wandb_entity="entity",
        wandb_mode="online",
    )

    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        reconstruction_launcher.validate_launch_prerequisites(
            metadata,
            python_version_probe=lambda python_path: None,
            backend_probe=lambda api_base: None,
            tmux_session_exists=lambda socket_name, session_name: False,
        )


def test_validate_launch_prerequisites_rejects_existing_schedule_outputs(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")
    env_path = _write_env(tmp_path / ".env")
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=plan_path,
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
        wandb_project="rayuela",
        wandb_entity="entity",
        wandb_mode="online",
    )
    metadata.schedule_dir.mkdir(parents=True, exist_ok=True)
    (metadata.schedule_dir / "schedule_summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        reconstruction_launcher.validate_launch_prerequisites(
            metadata,
            python_version_probe=lambda python_path: None,
            backend_probe=lambda api_base: None,
            tmux_session_exists=lambda socket_name, session_name: False,
        )


def test_validate_launch_prerequisites_requires_hf_token(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")
    env_path = tmp_path / ".env"
    env_path.write_text("WANDB_API_KEY=test-key\n", encoding="utf-8")
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=plan_path,
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
        wandb_project="rayuela",
        wandb_entity="entity",
        wandb_mode="online",
    )

    with pytest.raises(ValueError, match="HF_TOKEN"):
        reconstruction_launcher.validate_launch_prerequisites(
            metadata,
            python_version_probe=lambda python_path: None,
            backend_probe=lambda api_base: None,
            tmux_session_exists=lambda socket_name, session_name: False,
        )


def test_validate_launch_prerequisites_rejects_existing_tmux_session(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")
    env_path = _write_env(tmp_path / ".env")
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=plan_path,
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
    )

    with pytest.raises(FileExistsError, match="tmux session already exists"):
        reconstruction_launcher.validate_launch_prerequisites(
            metadata,
            python_version_probe=lambda python_path: None,
            backend_probe=lambda api_base: None,
            tmux_session_exists=lambda socket_name, session_name: True,
        )


def test_validate_launch_prerequisites_rejects_missing_plan_file(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env")
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=_write_plan(tmp_path / "plan.json"),
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
    )
    metadata.plan_path.unlink()

    with pytest.raises(FileNotFoundError, match="plan file does not exist"):
        reconstruction_launcher.validate_launch_prerequisites(
            metadata,
            python_version_probe=lambda python_path: None,
            backend_probe=lambda api_base: None,
            tmux_session_exists=lambda socket_name, session_name: False,
        )


def test_validate_launch_prerequisites_rejects_missing_env_file(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env")
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=_write_plan(tmp_path / "plan.json"),
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
    )
    metadata.env_path.unlink()

    with pytest.raises(FileNotFoundError, match="env file does not exist"):
        reconstruction_launcher.validate_launch_prerequisites(
            metadata,
            python_version_probe=lambda python_path: None,
            backend_probe=lambda api_base: None,
            tmux_session_exists=lambda socket_name, session_name: False,
        )


def test_launch_schedule_writes_metadata_and_invokes_tmux(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")
    env_path = _write_env(tmp_path / ".env")
    tmux_calls: list[list[str]] = []

    def _run_tmux(command: list[str]) -> None:
        tmux_calls.append(command)

    metadata = reconstruction_launcher.launch_schedule(
        plan_path=plan_path,
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
        wandb_project="rayuela",
        wandb_entity="entity",
        wandb_mode="online",
        run_tmux_command=_run_tmux,
        python_version_probe=lambda python_path: None,
        backend_probe=lambda api_base: None,
        tmux_session_exists=lambda socket_name, session_name: False,
    )

    assert metadata.schedule_id == "guided-20260311a"
    assert metadata.scheduler_log_path.name == "scheduler.log"
    assert metadata.analysis_log_path.name == "analysis.log"
    assert metadata.launch_metadata_path.exists()
    payload = json.loads(metadata.launch_metadata_path.read_text(encoding="utf-8"))
    assert payload["schedule_id"] == "guided-20260311a"
    assert payload["tmux_session_name"] == metadata.tmux_session_name
    assert tmux_calls == [
        [
            "tmux",
            "-L",
            "rayuela",
            "new-session",
            "-d",
            "-s",
            metadata.tmux_session_name,
            metadata.launch_command,
        ]
    ]


def test_launch_command_omits_optional_wandb_flags_when_not_configured(tmp_path: Path) -> None:
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=_write_plan(tmp_path / "plan.json"),
        repo_root=tmp_path,
        env_path=_write_env(tmp_path / ".env"),
        python_path=tmp_path / ".venv" / "bin" / "python",
    )

    assert "--wandb-project" not in metadata.launch_command
    assert "--wandb-entity" not in metadata.launch_command
    assert "--wandb-mode offline" in metadata.launch_command


def test_schedule_status_reports_tmux_and_artifact_state(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")
    env_path = _write_env(tmp_path / ".env")
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=plan_path,
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
        wandb_project="rayuela",
        wandb_entity="entity",
        wandb_mode="online",
    )
    metadata.schedule_dir.mkdir(parents=True, exist_ok=True)
    metadata.analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata.launch_metadata_path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    (metadata.schedule_dir / "schedule_summary.json").write_text("{}", encoding="utf-8")
    (metadata.analysis_dir / "reconstruction_analysis_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )

    status = reconstruction_launcher.schedule_status(
        schedule_id=metadata.schedule_id,
        repo_root=tmp_path,
        tmux_session_exists=lambda socket_name, session_name: True,
    )

    assert status["schedule_id"] == metadata.schedule_id
    assert status["tmux_session_active"] is True
    assert status["schedule_summary_exists"] is True
    assert status["analysis_summary_exists"] is True


def test_schedule_status_uses_default_tmux_names_without_metadata(tmp_path: Path) -> None:
    observed: list[tuple[str, str]] = []

    def _tmux_session_exists(socket_name: str, session_name: str) -> bool:
        observed.append((socket_name, session_name))
        return False

    status = reconstruction_launcher.schedule_status(
        schedule_id="guided-20260311a",
        repo_root=tmp_path,
        tmux_session_exists=_tmux_session_exists,
    )

    assert status == {
        "schedule_id": "guided-20260311a",
        "launch_metadata_exists": False,
        "tmux_session_active": False,
        "schedule_results_exists": False,
        "schedule_summary_exists": False,
        "analysis_summary_exists": False,
    }
    assert observed == [("rayuela", "rayuela-guided-20260311a")]


def test_stop_schedule_kills_tmux_session_from_metadata(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")
    env_path = _write_env(tmp_path / ".env")
    metadata = reconstruction_launcher.build_launch_metadata(
        plan_path=plan_path,
        repo_root=tmp_path,
        env_path=env_path,
        python_path=tmp_path / ".venv" / "bin" / "python",
        wandb_project="rayuela",
        wandb_entity="entity",
        wandb_mode="online",
    )
    metadata.schedule_dir.mkdir(parents=True, exist_ok=True)
    metadata.launch_metadata_path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    tmux_calls: list[list[str]] = []

    def _run_tmux(command: list[str]) -> None:
        tmux_calls.append(command)

    reconstruction_launcher.stop_schedule(
        schedule_id=metadata.schedule_id,
        repo_root=tmp_path,
        run_tmux_command=_run_tmux,
    )

    assert tmux_calls == [
        [
            "tmux",
            "-L",
            "rayuela",
            "kill-session",
            "-t",
            metadata.tmux_session_name,
        ]
    ]


def test_stop_schedule_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="launch metadata does not exist"):
        reconstruction_launcher.stop_schedule(
            schedule_id="guided-20260311a",
            repo_root=tmp_path,
        )


def test_validate_plan_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    payload = {
        "schedule_id": "guided-20260311a",
        "experiments": [
            {
                "experiment_id": "baseline-a",
                "run_id": "phase4-a",
                "phase": "phase-4-prompt-baselines",
                "command": ["python", "run.py"],
                "timeout_seconds": 10,
                "metric_path_template": "outputs/{run_id}.json",
                "metric_key": "metric",
            },
            {
                "experiment_id": "baseline-b",
                "run_id": "phase4-a",
                "phase": "phase-4-prompt-baselines",
                "command": ["python", "run.py"],
                "timeout_seconds": 10,
                "metric_path_template": "outputs/{run_id}.json",
                "metric_key": "metric",
            },
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate run_id"):
        reconstruction_launcher.validate_plan(path)


def test_validate_plan_rejects_empty_experiment_lists(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schedule_id": "guided-20260311a", "experiments": []}) + "\n")

    with pytest.raises(ValueError, match="defines no experiments"):
        reconstruction_launcher.validate_plan(path)


def test_validate_plan_rejects_empty_commands(tmp_path: Path) -> None:
    payload = {
        "schedule_id": "guided-20260311a",
        "experiments": [
            {
                "experiment_id": "baseline-a",
                "run_id": "phase4-a",
                "phase": "phase-4-prompt-baselines",
                "command": [],
                "timeout_seconds": 10,
                "metric_path_template": "outputs/{run_id}.json",
                "metric_key": "metric",
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty command"):
        reconstruction_launcher.validate_plan(path)


def test_validate_plan_rejects_duplicate_experiment_ids(tmp_path: Path) -> None:
    payload = {
        "schedule_id": "guided-20260311a",
        "experiments": [
            {
                "experiment_id": "baseline-a",
                "run_id": "phase4-a",
                "phase": "phase-4-prompt-baselines",
                "command": ["python", "run.py"],
                "timeout_seconds": 10,
                "metric_path_template": "outputs/{run_id}.json",
                "metric_key": "metric",
            },
            {
                "experiment_id": "baseline-a",
                "run_id": "phase4-b",
                "phase": "phase-4-prompt-baselines",
                "command": ["python", "run.py"],
                "timeout_seconds": 10,
                "metric_path_template": "outputs/{run_id}.json",
                "metric_key": "metric",
            },
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate experiment_id"):
        reconstruction_launcher.validate_plan(path)


def test_parse_env_file_skips_comments_and_invalid_lines(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "INVALID_LINE",
                "HF_TOKEN = test-token",
                "WANDB_API_KEY=test-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert reconstruction_launcher._parse_env_file(env_path) == {
        "HF_TOKEN": "test-token",
        "WANDB_API_KEY": "test-key",
    }


def test_default_python_version_probe_requires_existing_interpreter(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python interpreter does not exist"):
        reconstruction_launcher._default_python_version_probe(tmp_path / "missing-python")


def test_default_python_version_probe_invokes_import_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")
    observed: list[list[str]] = []

    def _run(command: list[str], *, check: bool, capture_output: bool, text: bool) -> None:
        assert check is True
        assert capture_output is True
        assert text is True
        observed.append(command)
        return None

    monkeypatch.setattr(reconstruction_launcher.subprocess, "run", _run)

    reconstruction_launcher._default_python_version_probe(python_path)

    assert observed == [[str(python_path), "-c", "import openai; import wandb"]]


def test_default_backend_probe_requests_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[str, int]] = []

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def _urlopen(url: str, timeout: int) -> _Response:
        observed.append((url, timeout))
        return _Response()

    monkeypatch.setattr(reconstruction_launcher.urllib.request, "urlopen", _urlopen)

    reconstruction_launcher._default_backend_probe("http://localhost:8000/v1/")

    assert observed == [("http://localhost:8000/v1/models", 5)]


def test_default_tmux_helpers_use_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[str]] = []

    def _run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> SimpleNamespace:
        observed.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reconstruction_launcher.subprocess, "run", _run)

    assert reconstruction_launcher._default_tmux_session_exists("socket", "session") is True
    reconstruction_launcher._default_run_tmux_command(["tmux", "new-session"])

    assert observed == [
        ["tmux", "-L", "socket", "has-session", "-t", "session"],
        ["tmux", "new-session"],
    ]


def test_main_launch_dispatches_and_prints_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    metadata = reconstruction_launcher.LaunchMetadata(
        schedule_id="guided-20260311a",
        plan_path=tmp_path / "plan.json",
        repo_root=tmp_path,
        env_path=tmp_path / ".env",
        python_path=tmp_path / ".venv" / "bin" / "python",
        schedule_dir=tmp_path / "outputs" / "schedule",
        analysis_dir=tmp_path / "outputs" / "schedule" / "analysis",
        scheduler_log_path=tmp_path / "outputs" / "schedule" / "scheduler.log",
        analysis_log_path=tmp_path / "outputs" / "schedule" / "analysis" / "analysis.log",
        launch_metadata_path=tmp_path / "outputs" / "schedule" / "launch_metadata.json",
        tmux_socket_name="rayuela",
        tmux_session_name="rayuela-guided-20260311a",
        wandb_project="rayuela",
        wandb_entity="entity",
        wandb_mode="online",
        launched_at="2026-03-11T00:00:00Z",
    )
    called: list[tuple[Path, Path, Path, str | None, str | None, str, str, str]] = []

    def _launch_schedule(
        *,
        plan_path: Path,
        repo_root: Path,
        env_path: Path,
        python_path: Path,
        wandb_project: str | None,
        wandb_entity: str | None,
        wandb_mode: str,
        tmux_socket_name: str,
        tmux_session_prefix: str,
    ) -> reconstruction_launcher.LaunchMetadata:
        called.append(
            (
                plan_path,
                repo_root,
                env_path,
                wandb_project,
                wandb_entity,
                wandb_mode,
                tmux_socket_name,
                tmux_session_prefix,
            )
        )
        return metadata

    monkeypatch.setattr(reconstruction_launcher, "launch_schedule", _launch_schedule)

    exit_code = reconstruction_launcher.main(
        [
            "launch",
            "--plan-path",
            str(tmp_path / "plan.json"),
            "--repo-root",
            str(tmp_path),
            "--env-path",
            str(tmp_path / ".env"),
            "--python-path",
            str(tmp_path / ".venv" / "bin" / "python"),
            "--wandb-project",
            "rayuela",
            "--wandb-entity",
            "entity",
            "--wandb-mode",
            "online",
            "--tmux-socket-name",
            "socket",
            "--tmux-session-prefix",
            "prefix",
        ]
    )

    assert exit_code == 0
    assert called == [
        (
            tmp_path / "plan.json",
            tmp_path,
            tmp_path / ".env",
            "rayuela",
            "entity",
            "online",
            "socket",
            "prefix",
        )
    ]
    output = capsys.readouterr().out
    assert "Launched guided-20260311a in tmux session rayuela-guided-20260311a." in output
    assert str(metadata.scheduler_log_path) in output
    assert str(metadata.analysis_log_path) in output


def test_main_status_dispatches_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        reconstruction_launcher,
        "schedule_status",
        lambda *, schedule_id, repo_root: {
            "schedule_id": schedule_id,
            "launch_metadata_exists": True,
            "tmux_session_active": False,
            "schedule_results_exists": True,
            "schedule_summary_exists": False,
            "analysis_summary_exists": False,
        },
    )

    exit_code = reconstruction_launcher.main(
        ["status", "--schedule-id", "guided-20260311a", "--repo-root", str(tmp_path)]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schedule_id"] == "guided-20260311a"
    assert output["launch_metadata_exists"] is True


def test_main_stop_dispatches_and_prints_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    observed: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        reconstruction_launcher,
        "stop_schedule",
        lambda *, schedule_id, repo_root: observed.append((schedule_id, repo_root)),
    )

    exit_code = reconstruction_launcher.main(
        ["stop", "--schedule-id", "guided-20260311a", "--repo-root", str(tmp_path)]
    )

    assert exit_code == 0
    assert observed == [("guided-20260311a", tmp_path)]
    assert capsys.readouterr().out == "Stopped schedule guided-20260311a.\n"
