#!/usr/bin/env python3
"""
Autonomous teacher-data loop for Phase 5 reconstruction research.

The loop runs small Phase 4 teacher batches, distills scoreable outputs into
Phase 5 JSONL datasets, and records append-only events. It is deliberately
stateful so it can run unattended and resume after interruptions without
reusing case offsets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOOP_ROOT = PROJECT_ROOT / "outputs" / "reconstruction" / "analysis" / "agentic_loops"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "reconstruction" / "runs"
DEFAULT_DISTILL_ROOT = PROJECT_ROOT / "outputs" / "reconstruction" / "style_distill"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_API_BASE = "http://localhost:8000/v1"


@dataclass(frozen=True)
class BatchMetrics:
    """Compact, text-free metrics for one teacher batch."""

    run_id: str
    case_count: int
    failed_case_count: int
    scoreable_count: int
    rescue_used_count: int
    semantic_pass_count: int
    target_pass_count: int
    style_pass_count: int
    length_pass_count: int
    lexical_pass_count: int
    mean_weighted_objective: float | None
    median_weighted_objective: float | None
    distillable_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def local_stamp() -> str:
    """Return a compact timestamp for immutable run IDs."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_event(loop_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    """Append one event to the loop JSONL log."""
    loop_dir.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": utc_now(), "event_type": event_type, **payload}
    with (loop_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return parser arguments with paths converted to strings."""
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            payload[key] = value
    return payload


def load_state(path: Path) -> dict[str, Any] | None:
    """Load loop state if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_model(api_base: str, *, timeout_seconds: int = 10) -> str:
    """Return the first served model ID from an OpenAI-compatible endpoint."""
    url = f"{api_base.rstrip('/')}/models"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"no models reported by {url}")
    return str(data[0]["id"])


def unique_id(base: str, existing_root: Path) -> str:
    """Return an ID not already present under `existing_root`."""
    candidate = base
    suffix = 1
    while (existing_root / candidate).exists():
        suffix += 1
        candidate = f"{base}-r{suffix}"
    return candidate


def best_iteration(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the best iteration for a case result, if present."""
    iterations = result.get("iterations") or []
    if not iterations:
        return None
    index = int(result.get("best_iteration_index", 0))
    return iterations[index]


def summarize_teacher_cases(path: Path, *, min_weighted_objective: float) -> BatchMetrics:
    """Summarize one prompt-baseline case artifact without exposing generated text."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = path.parent.name
    summary_path = path.parent / "prompt_baseline_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    style_summary = summary.get("controls", {}).get("style_shift", {})

    scoreable_count = 0
    rescue_used_count = 0
    semantic_pass_count = 0
    target_pass_count = 0
    style_pass_count = 0
    length_pass_count = 0
    lexical_pass_count = 0
    distillable_count = 0
    weighted_values: list[float] = []
    for result in payload.get("results", []):
        iteration = best_iteration(result)
        if not iteration:
            continue
        score_history = iteration.get("score_history") or {}
        if "weighted_objective" not in score_history:
            continue
        scoreable_count += 1
        weighted = float(score_history["weighted_objective"])
        weighted_values.append(weighted)
        rescue_used_count += int(bool(iteration.get("rescue_used")))
        semantic_pass_count += int(bool(score_history.get("semantic_tolerance_pass")))
        target_pass_count += int(bool(score_history.get("target_tolerance_pass")))
        style_pass_count += int(bool(score_history.get("stylistic_tolerance_pass")))
        length_pass_count += int(bool(score_history.get("length_guardrail_pass")))
        lexical_pass_count += int(bool(score_history.get("lexical_overlap_pass")))
        parsed_text = str(iteration.get("parsed_text", "")).strip()
        if parsed_text and weighted >= min_weighted_objective:
            distillable_count += 1

    return BatchMetrics(
        run_id=run_id,
        case_count=int(style_summary.get("count", len(payload.get("results", [])))),
        failed_case_count=int(style_summary.get("failed_case_count", 0)),
        scoreable_count=scoreable_count,
        rescue_used_count=rescue_used_count,
        semantic_pass_count=semantic_pass_count,
        target_pass_count=target_pass_count,
        style_pass_count=style_pass_count,
        length_pass_count=length_pass_count,
        lexical_pass_count=lexical_pass_count,
        mean_weighted_objective=style_summary.get("mean_weighted_objective"),
        median_weighted_objective=style_summary.get("median_weighted_objective"),
        distillable_count=distillable_count,
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one command and persist stdout/stderr."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(
            (exc.stderr or "") + f"\nTimed out after {timeout_seconds}s\n",
            encoding="utf-8",
        )
        raise
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    duration = time.monotonic() - started
    duration_path = stdout_path.with_suffix(".duration_seconds.txt")
    with duration_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{duration:.3f}\n")
    return result


def build_baseline_command(args: argparse.Namespace, *, run_id: str, case_offset: int) -> list[str]:
    """Build the Phase 4 teacher command for one cycle."""
    return [
        args.python_bin,
        "src/reconstruction_baselines.py",
        "--run-id",
        run_id,
        "--api-base",
        args.api_base,
        "--model",
        args.model,
        "--case-offset",
        str(case_offset),
        "--max-cases",
        str(args.batch_size),
        "--max-iterations",
        str(args.max_iterations),
        "--generation-temperature",
        str(args.generation_temperature),
        "--generation-max-tokens",
        str(args.generation_max_tokens),
        "--rescue-generation-max-tokens",
        str(args.rescue_generation_max_tokens),
        "--semantic-generation-max-tokens",
        str(args.semantic_generation_max_tokens),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
    ]


def build_distill_command(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    teacher_cases_paths: list[str],
) -> list[str]:
    """Build the Phase 5 distillation command for accumulated teacher cases."""
    command = [
        args.python_bin,
        "src/reconstruction_style_distill.py",
        "--dataset-id",
        dataset_id,
        "--min-weighted-objective",
        str(args.min_weighted_objective),
    ]
    for path in teacher_cases_paths:
        command.extend(["--teacher-cases-path", path])
    if args.require_semantic_pass:
        command.append("--require-semantic-pass")
    if args.require_target_pass:
        command.append("--require-target-pass")
    return command


def should_continue(args: argparse.Namespace, *, started_at: float, completed_cycles: int) -> bool:
    """Return whether the outer loop should start another cycle."""
    if args.max_cycles is not None and completed_cycles >= args.max_cycles:
        return False
    if args.max_runtime_seconds is not None and time.monotonic() - started_at >= args.max_runtime_seconds:
        return False
    return args.forever or args.max_cycles is not None or completed_cycles == 0


def run_loop(args: argparse.Namespace) -> int:
    """Run the autonomous loop."""
    loop_dir = args.loop_root / args.loop_id
    state_path = loop_dir / "loop_state.json"
    stop_path = loop_dir / "STOP"
    loop_dir.mkdir(parents=True, exist_ok=True)

    state = None if args.reset else load_state(state_path)
    if state is None:
        state = {
            "loop_id": args.loop_id,
            "created_at": utc_now(),
            "completed_cycles": 0,
            "next_case_offset": args.start_offset,
            "teacher_cases_paths": [str(path) for path in args.include_teacher_cases_path],
            "latest_dataset_id": None,
            "last_metrics": None,
        }
        write_json(state_path, state)
    append_event(loop_dir, "loop_started", {"state": state, "args": jsonable_args(args)})

    started_at = time.monotonic()
    while should_continue(args, started_at=started_at, completed_cycles=int(state["completed_cycles"])):
        if stop_path.exists():
            append_event(loop_dir, "stop_file_seen", {"stop_path": str(stop_path)})
            return 0

        try:
            served_model = endpoint_model(args.api_base, timeout_seconds=args.endpoint_timeout_seconds)
        except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            append_event(loop_dir, "endpoint_unavailable", {"error_message": str(exc)})
            return 2
        if served_model != args.model:
            append_event(
                loop_dir,
                "unexpected_model",
                {"expected_model": args.model, "served_model": served_model},
            )
            return 2

        cycle_index = int(state["completed_cycles"]) + 1
        case_offset = int(state["next_case_offset"])
        run_base = f"{args.run_prefix}-offset{case_offset}-c{cycle_index:03d}-{local_stamp()}"
        run_id = unique_id(run_base, DEFAULT_RUN_ROOT)
        cycle_dir = loop_dir / f"cycle-{cycle_index:03d}"
        append_event(
            loop_dir,
            "cycle_started",
            {"cycle_index": cycle_index, "case_offset": case_offset, "run_id": run_id},
        )

        baseline_command = build_baseline_command(args, run_id=run_id, case_offset=case_offset)
        baseline_result = run_command(
            baseline_command,
            cwd=PROJECT_ROOT,
            timeout_seconds=args.batch_timeout_seconds,
            stdout_path=cycle_dir / "baseline.stdout",
            stderr_path=cycle_dir / "baseline.stderr",
        )
        if baseline_result.returncode != 0:
            append_event(
                loop_dir,
                "baseline_failed",
                {
                    "cycle_index": cycle_index,
                    "run_id": run_id,
                    "returncode": baseline_result.returncode,
                },
            )
            return baseline_result.returncode

        cases_path = DEFAULT_RUN_ROOT / run_id / "prompt_baseline_cases.json"
        metrics = summarize_teacher_cases(cases_path, min_weighted_objective=args.min_weighted_objective)
        teacher_cases_paths = list(dict.fromkeys([*state["teacher_cases_paths"], str(cases_path.relative_to(PROJECT_ROOT))]))

        if metrics.case_count == 0:
            state.update(
                {
                    "completed_cycles": cycle_index,
                    "next_case_offset": case_offset,
                    "teacher_cases_paths": teacher_cases_paths,
                    "last_metrics": metrics.to_dict(),
                    "stopped_reason": "no_cases_available",
                    "updated_at": utc_now(),
                }
            )
            write_json(state_path, state)
            write_json(
                loop_dir / "latest_summary.json",
                {
                    "loop_id": args.loop_id,
                    "cycle_index": cycle_index,
                    "run_id": run_id,
                    "dataset_id": state.get("latest_dataset_id"),
                    "metrics": metrics.to_dict(),
                    "next_case_offset": state["next_case_offset"],
                    "stopped_reason": "no_cases_available",
                },
            )
            append_event(
                loop_dir,
                "no_cases_available",
                {
                    "cycle_index": cycle_index,
                    "case_offset": case_offset,
                    "run_id": run_id,
                    "metrics": metrics.to_dict(),
                },
            )
            return 0

        dataset_base = f"{args.dataset_prefix}-{args.loop_id}-c{cycle_index:03d}"
        dataset_id = unique_id(dataset_base, DEFAULT_DISTILL_ROOT)
        distill_command = build_distill_command(
            args,
            dataset_id=dataset_id,
            teacher_cases_paths=teacher_cases_paths,
        )
        distill_result = run_command(
            distill_command,
            cwd=PROJECT_ROOT,
            timeout_seconds=args.distill_timeout_seconds,
            stdout_path=cycle_dir / "distill.stdout",
            stderr_path=cycle_dir / "distill.stderr",
        )
        if distill_result.returncode != 0:
            append_event(
                loop_dir,
                "distill_failed",
                {
                    "cycle_index": cycle_index,
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "returncode": distill_result.returncode,
                    "metrics": metrics.to_dict(),
                },
            )
            return distill_result.returncode

        manifest_path = DEFAULT_DISTILL_ROOT / dataset_id / "distill_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state.update(
            {
                "completed_cycles": cycle_index,
                "next_case_offset": case_offset + args.batch_size,
                "teacher_cases_paths": teacher_cases_paths,
                "latest_dataset_id": dataset_id,
                "last_metrics": metrics.to_dict(),
                "updated_at": utc_now(),
            }
        )
        write_json(state_path, state)
        write_json(
            loop_dir / "latest_summary.json",
            {
                "loop_id": args.loop_id,
                "cycle_index": cycle_index,
                "run_id": run_id,
                "dataset_id": dataset_id,
                "metrics": metrics.to_dict(),
                "distill_manifest": manifest,
                "next_case_offset": state["next_case_offset"],
            },
        )
        append_event(
            loop_dir,
            "cycle_completed",
            {
                "cycle_index": cycle_index,
                "run_id": run_id,
                "dataset_id": dataset_id,
                "metrics": metrics.to_dict(),
                "distill_split_counts": manifest.get("split_counts"),
                "next_case_offset": state["next_case_offset"],
            },
        )
        time.sleep(args.sleep_seconds)

    append_event(loop_dir, "loop_completed", {"state": state})
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-id", required=True)
    parser.add_argument("--loop-root", type=Path, default=DEFAULT_LOOP_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-prefix", default="phase4-qwen3-instruct-loop")
    parser.add_argument("--dataset-prefix", default="phase5-style-distill-qwen3-loop")
    parser.add_argument("--include-teacher-cases-path", type=Path, action="append", default=[])
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--generation-temperature", type=float, default=0.4)
    parser.add_argument("--generation-max-tokens", type=int, default=768)
    parser.add_argument("--rescue-generation-max-tokens", type=int, default=384)
    parser.add_argument("--semantic-generation-max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout-seconds", type=int, default=600)
    parser.add_argument("--batch-timeout-seconds", type=int, default=3600)
    parser.add_argument("--distill-timeout-seconds", type=int, default=300)
    parser.add_argument("--endpoint-timeout-seconds", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=int, default=30)
    parser.add_argument("--min-weighted-objective", type=float, default=0.14)
    parser.add_argument("--require-semantic-pass", action="store_true")
    parser.add_argument("--require-target-pass", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--max-runtime-seconds", type=int)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
