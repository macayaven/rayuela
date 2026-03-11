#!/usr/bin/env python3
"""
Detached launcher, status, and stop helpers for reconstruction schedules.

This module hardens the overnight execution path around the existing finite
guided scheduler. It validates trusted plan files, checks runtime
prerequisites, launches the scheduler inside tmux, persists non-secret launch
metadata, and exposes simple status/stop helpers keyed by schedule_id.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reconstruction_contract import PROJECT_ROOT, ReconstructionPaths, utc_now
from reconstruction_scheduler import SchedulePlan, load_schedule_plan

DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_PYTHON_PATH = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_TMUX_SOCKET_NAME = "rayuela"
DEFAULT_TMUX_SESSION_PREFIX = "rayuela"
LAUNCH_METADATA_FILENAME = "launch_metadata.json"


@dataclass(frozen=True)
class LaunchMetadata:
    """Non-secret launch metadata persisted alongside one schedule."""

    schedule_id: str
    plan_path: Path
    repo_root: Path
    env_path: Path
    python_path: Path
    schedule_dir: Path
    analysis_dir: Path
    scheduler_log_path: Path
    analysis_log_path: Path
    launch_metadata_path: Path
    tmux_socket_name: str
    tmux_session_name: str
    wandb_project: str | None
    wandb_entity: str | None
    wandb_mode: str
    launched_at: str

    @property
    def launch_command(self) -> str:
        """Return the tmux command body used for detached execution."""
        scheduler_args = [
            str(self.python_path),
            "src/reconstruction_scheduler.py",
            "--plan-path",
            str(self.plan_path),
        ]
        if self.wandb_project:
            scheduler_args.extend(["--wandb-project", self.wandb_project])
        if self.wandb_entity:
            scheduler_args.extend(["--wandb-entity", self.wandb_entity])
        scheduler_args.extend(["--wandb-mode", self.wandb_mode])

        analysis_args = [
            str(self.python_path),
            "src/reconstruction_analysis.py",
            "--schedule-summary-path",
            str(self.schedule_dir / "schedule_summary.json"),
            "--schedule-run-selection",
            "nonfailed",
            "--output-dir",
            str(self.analysis_dir),
        ]
        if self.wandb_project:
            analysis_args.extend(["--wandb-project", self.wandb_project])
        if self.wandb_entity:
            analysis_args.extend(["--wandb-entity", self.wandb_entity])
        analysis_args.extend(["--wandb-mode", self.wandb_mode, "--wandb-group", self.schedule_id])

        scheduler_command = " ".join(shlex.quote(part) for part in scheduler_args)
        analysis_command = " ".join(shlex.quote(part) for part in analysis_args)
        shell_body = " && ".join(
            [
                "set -euo pipefail",
                f"cd {shlex.quote(str(self.repo_root))}",
                "set -a",
                f"source {shlex.quote(str(self.env_path))}",
                "set +a",
                f"{scheduler_command} > {shlex.quote(str(self.scheduler_log_path))} 2>&1",
                f"{analysis_command} > {shlex.quote(str(self.analysis_log_path))} 2>&1",
            ]
        )
        return f"bash -lc {shlex.quote(shell_body)}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata payload."""
        return {
            "schedule_id": self.schedule_id,
            "plan_path": str(self.plan_path),
            "repo_root": str(self.repo_root),
            "env_path": str(self.env_path),
            "python_path": str(self.python_path),
            "schedule_dir": str(self.schedule_dir),
            "analysis_dir": str(self.analysis_dir),
            "scheduler_log_path": str(self.scheduler_log_path),
            "analysis_log_path": str(self.analysis_log_path),
            "launch_metadata_path": str(self.launch_metadata_path),
            "tmux_socket_name": self.tmux_socket_name,
            "tmux_session_name": self.tmux_session_name,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_mode": self.wandb_mode,
            "launched_at": self.launched_at,
        }


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file without executing it."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1]
        values[key.strip()] = cleaned
    return values


def _extract_api_bases(plan: SchedulePlan) -> set[str]:
    """Extract declared API bases from scheduler experiment commands."""
    api_bases: set[str] = set()
    for spec in plan.experiments:
        parts = list(spec.command)
        for index, part in enumerate(parts[:-1]):
            if part == "--api-base":
                api_bases.add(parts[index + 1])
    return api_bases


def validate_plan(plan_path: Path) -> SchedulePlan:
    """Load a trusted scheduler plan and validate uniqueness constraints."""
    plan = load_schedule_plan(plan_path)
    if not plan.experiments:
        raise ValueError(f"schedule plan {plan_path} defines no experiments")

    seen_experiment_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    for spec in plan.experiments:
        if not spec.command:
            raise ValueError(f"experiment {spec.experiment_id!r} must define a non-empty command")
        if spec.experiment_id in seen_experiment_ids:
            raise ValueError(f"duplicate experiment_id in plan: {spec.experiment_id}")
        if spec.run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id in plan: {spec.run_id}")
        seen_experiment_ids.add(spec.experiment_id)
        seen_run_ids.add(spec.run_id)
    return plan


def build_launch_metadata(
    *,
    plan_path: Path,
    repo_root: Path = PROJECT_ROOT,
    env_path: Path = DEFAULT_ENV_PATH,
    python_path: Path = DEFAULT_PYTHON_PATH,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str = "offline",
    tmux_socket_name: str = DEFAULT_TMUX_SOCKET_NAME,
    tmux_session_prefix: str = DEFAULT_TMUX_SESSION_PREFIX,
) -> LaunchMetadata:
    """Build non-secret launch metadata for one schedule plan."""
    plan = validate_plan(plan_path)
    resolved_repo_root = repo_root.resolve()
    paths = ReconstructionPaths(project_root=resolved_repo_root)
    schedule_dir = paths.analysis_dir / "schedules" / plan.schedule_id
    analysis_dir = schedule_dir / "analysis"
    session_name = f"{tmux_session_prefix}-{plan.schedule_id}"
    return LaunchMetadata(
        schedule_id=plan.schedule_id,
        plan_path=plan_path.resolve(),
        repo_root=resolved_repo_root,
        env_path=env_path.resolve(),
        python_path=python_path.absolute(),
        schedule_dir=schedule_dir,
        analysis_dir=analysis_dir,
        scheduler_log_path=schedule_dir / "scheduler.log",
        analysis_log_path=analysis_dir / "analysis.log",
        launch_metadata_path=schedule_dir / LAUNCH_METADATA_FILENAME,
        tmux_socket_name=tmux_socket_name,
        tmux_session_name=session_name,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        launched_at=utc_now(),
    )


def _default_python_version_probe(python_path: Path) -> None:
    """Verify the configured Python environment can import required modules."""
    if not python_path.exists():
        raise FileNotFoundError(f"python interpreter does not exist: {python_path}")
    subprocess.run(
        [
            str(python_path),
            "-c",
            "import openai; import wandb",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _default_backend_probe(api_base: str) -> None:
    """Verify the configured backend responds to the models endpoint."""
    with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=5):
        return


def _default_tmux_session_exists(socket_name: str, session_name: str) -> bool:
    """Return whether a tmux session currently exists."""
    result = subprocess.run(
        [
            "tmux",
            "-L",
            socket_name,
            "has-session",
            "-t",
            session_name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _default_run_tmux_command(command: list[str]) -> None:
    """Run one tmux command and raise on failure."""
    subprocess.run(command, check=True, capture_output=True, text=True)


def validate_launch_prerequisites(
    metadata: LaunchMetadata,
    *,
    python_version_probe: Callable[[Path], None] = _default_python_version_probe,
    backend_probe: Callable[[str], None] = _default_backend_probe,
    tmux_session_exists: Callable[[str, str], bool] = _default_tmux_session_exists,
) -> None:
    """Validate preflight requirements before detached launch."""
    if not metadata.plan_path.exists():
        raise FileNotFoundError(f"plan file does not exist: {metadata.plan_path}")
    if not metadata.env_path.exists():
        raise FileNotFoundError(f"env file does not exist: {metadata.env_path}")

    env_values = _parse_env_file(metadata.env_path)
    missing_env_keys: list[str] = []
    if metadata.wandb_project and metadata.wandb_mode == "online":
        if not env_values.get("WANDB_API_KEY"):
            missing_env_keys.append("WANDB_API_KEY")
    if not env_values.get("HF_TOKEN"):
        missing_env_keys.append("HF_TOKEN")
    if missing_env_keys:
        joined = ", ".join(missing_env_keys)
        raise ValueError(f"launch env file is missing required keys: {joined}")

    if metadata.schedule_dir.exists() and any(
        (metadata.schedule_dir / filename).exists()
        for filename in (
            "schedule_results.jsonl",
            "schedule_summary.json",
            LAUNCH_METADATA_FILENAME,
        )
    ):
        raise FileExistsError(
            f"schedule outputs already exists for {metadata.schedule_id!r}: {metadata.schedule_dir}"
        )

    if tmux_session_exists(metadata.tmux_socket_name, metadata.tmux_session_name):
        raise FileExistsError(
            f"tmux session already exists for {metadata.schedule_id!r}: "
            f"{metadata.tmux_session_name}"
        )

    python_version_probe(metadata.python_path)
    for api_base in sorted(_extract_api_bases(validate_plan(metadata.plan_path))):
        backend_probe(api_base)


def launch_schedule(
    *,
    plan_path: Path,
    repo_root: Path = PROJECT_ROOT,
    env_path: Path = DEFAULT_ENV_PATH,
    python_path: Path = DEFAULT_PYTHON_PATH,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str = "offline",
    tmux_socket_name: str = DEFAULT_TMUX_SOCKET_NAME,
    tmux_session_prefix: str = DEFAULT_TMUX_SESSION_PREFIX,
    run_tmux_command: Callable[[list[str]], None] = _default_run_tmux_command,
    python_version_probe: Callable[[Path], None] = _default_python_version_probe,
    backend_probe: Callable[[str], None] = _default_backend_probe,
    tmux_session_exists: Callable[[str, str], bool] = _default_tmux_session_exists,
) -> LaunchMetadata:
    """Validate and launch one schedule inside a detached tmux session."""
    metadata = build_launch_metadata(
        plan_path=plan_path,
        repo_root=repo_root,
        env_path=env_path,
        python_path=python_path,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        tmux_socket_name=tmux_socket_name,
        tmux_session_prefix=tmux_session_prefix,
    )
    validate_launch_prerequisites(
        metadata,
        python_version_probe=python_version_probe,
        backend_probe=backend_probe,
        tmux_session_exists=tmux_session_exists,
    )
    metadata.schedule_dir.mkdir(parents=True, exist_ok=True)
    metadata.analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata.launch_metadata_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_tmux_command(
        [
            "tmux",
            "-L",
            metadata.tmux_socket_name,
            "new-session",
            "-d",
            "-s",
            metadata.tmux_session_name,
            metadata.launch_command,
        ]
    )
    return metadata


def _load_launch_metadata(path: Path) -> LaunchMetadata:
    """Load persisted launch metadata from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LaunchMetadata(
        schedule_id=str(payload["schedule_id"]),
        plan_path=Path(payload["plan_path"]),
        repo_root=Path(payload["repo_root"]),
        env_path=Path(payload["env_path"]),
        python_path=Path(payload["python_path"]),
        schedule_dir=Path(payload["schedule_dir"]),
        analysis_dir=Path(payload["analysis_dir"]),
        scheduler_log_path=Path(payload["scheduler_log_path"]),
        analysis_log_path=Path(payload["analysis_log_path"]),
        launch_metadata_path=Path(payload["launch_metadata_path"]),
        tmux_socket_name=str(payload["tmux_socket_name"]),
        tmux_session_name=str(payload["tmux_session_name"]),
        wandb_project=payload.get("wandb_project"),
        wandb_entity=payload.get("wandb_entity"),
        wandb_mode=str(payload["wandb_mode"]),
        launched_at=str(payload["launched_at"]),
    )


def schedule_status(
    *,
    schedule_id: str,
    repo_root: Path = PROJECT_ROOT,
    tmux_session_exists: Callable[[str, str], bool] = _default_tmux_session_exists,
) -> dict[str, Any]:
    """Return a compact status payload for one launched schedule."""
    paths = ReconstructionPaths(project_root=repo_root.resolve())
    schedule_dir = paths.analysis_dir / "schedules" / schedule_id
    metadata_path = schedule_dir / LAUNCH_METADATA_FILENAME
    metadata = _load_launch_metadata(metadata_path) if metadata_path.exists() else None
    if metadata is None:
        tmux_socket_name = DEFAULT_TMUX_SOCKET_NAME
        tmux_session_name = f"{DEFAULT_TMUX_SESSION_PREFIX}-{schedule_id}"
        analysis_dir = schedule_dir / "analysis"
    else:
        tmux_socket_name = metadata.tmux_socket_name
        tmux_session_name = metadata.tmux_session_name
        analysis_dir = metadata.analysis_dir
    return {
        "schedule_id": schedule_id,
        "launch_metadata_exists": metadata_path.exists(),
        "tmux_session_active": tmux_session_exists(tmux_socket_name, tmux_session_name),
        "schedule_results_exists": (schedule_dir / "schedule_results.jsonl").exists(),
        "schedule_summary_exists": (schedule_dir / "schedule_summary.json").exists(),
        "analysis_summary_exists": (analysis_dir / "reconstruction_analysis_summary.json").exists(),
    }


def stop_schedule(
    *,
    schedule_id: str,
    repo_root: Path = PROJECT_ROOT,
    run_tmux_command: Callable[[list[str]], None] = _default_run_tmux_command,
    tmux_session_exists: Callable[[str, str], bool] = _default_tmux_session_exists,
) -> None:
    """Stop one launched schedule by killing its tmux session."""
    paths = ReconstructionPaths(project_root=repo_root.resolve())
    metadata_path = paths.analysis_dir / "schedules" / schedule_id / LAUNCH_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"launch metadata does not exist for schedule {schedule_id!r}")
    metadata = _load_launch_metadata(metadata_path)
    if not tmux_session_exists(metadata.tmux_socket_name, metadata.tmux_session_name):
        raise FileNotFoundError(
            f"tmux session is not active for schedule {schedule_id!r}: {metadata.tmux_session_name}"
        )
    run_tmux_command(
        [
            "tmux",
            "-L",
            metadata.tmux_socket_name,
            "kill-session",
            "-t",
            metadata.tmux_session_name,
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for reconstruction launch operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--plan-path", type=Path, required=True)
    launch.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    launch.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    launch.add_argument("--python-path", type=Path, default=DEFAULT_PYTHON_PATH)
    launch.add_argument("--wandb-project", default=None)
    launch.add_argument("--wandb-entity", default=None)
    launch.add_argument("--wandb-mode", default="offline")
    launch.add_argument("--tmux-socket-name", default=DEFAULT_TMUX_SOCKET_NAME)
    launch.add_argument("--tmux-session-prefix", default=DEFAULT_TMUX_SESSION_PREFIX)

    status = subparsers.add_parser("status")
    status.add_argument("--schedule-id", required=True)
    status.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--schedule-id", required=True)
    stop.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested launch/status/stop command."""
    args = build_argument_parser().parse_args(argv)
    if args.command == "launch":
        metadata = launch_schedule(
            plan_path=args.plan_path,
            repo_root=args.repo_root,
            env_path=args.env_path,
            python_path=args.python_path,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
            tmux_socket_name=args.tmux_socket_name,
            tmux_session_prefix=args.tmux_session_prefix,
        )
        print(f"Launched {metadata.schedule_id} in tmux session {metadata.tmux_session_name}.")
        print(f"Scheduler log: {metadata.scheduler_log_path}")
        print(f"Analysis log: {metadata.analysis_log_path}")
        return 0
    if args.command == "status":
        print(
            json.dumps(
                schedule_status(schedule_id=args.schedule_id, repo_root=args.repo_root),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    stop_schedule(schedule_id=args.schedule_id, repo_root=args.repo_root)
    print(f"Stopped schedule {args.schedule_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
