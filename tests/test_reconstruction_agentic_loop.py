from __future__ import annotations

import json
from pathlib import Path

import reconstruction_agentic_loop


def _write_cases(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    cases_path = run_dir / "prompt_baseline_cases.json"
    summary_path = run_dir / "prompt_baseline_summary.json"
    cases_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "best_iteration_index": 0,
                        "iterations": [
                            {
                                "parsed_text": "texto",
                                "rescue_used": False,
                                "score_history": {
                                    "weighted_objective": 0.2,
                                    "semantic_tolerance_pass": True,
                                    "target_tolerance_pass": True,
                                    "stylistic_tolerance_pass": False,
                                    "length_guardrail_pass": True,
                                    "lexical_overlap_pass": True,
                                },
                            }
                        ],
                    },
                    {
                        "best_iteration_index": 0,
                        "iterations": [
                            {
                                "parsed_text": "otro",
                                "rescue_used": True,
                                "score_history": {
                                    "weighted_objective": 0.1,
                                    "semantic_tolerance_pass": False,
                                    "target_tolerance_pass": False,
                                    "stylistic_tolerance_pass": True,
                                    "length_guardrail_pass": False,
                                    "lexical_overlap_pass": True,
                                },
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "controls": {
                    "style_shift": {
                        "count": 2,
                        "failed_case_count": 0,
                        "mean_weighted_objective": 0.15,
                        "median_weighted_objective": 0.15,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return cases_path


def test_summarize_teacher_cases_counts_guardrails_without_text(tmp_path: Path) -> None:
    cases_path = _write_cases(tmp_path / "phase4-a")

    metrics = reconstruction_agentic_loop.summarize_teacher_cases(
        cases_path,
        min_weighted_objective=0.14,
    )

    assert metrics.run_id == "phase4-a"
    assert metrics.case_count == 2
    assert metrics.scoreable_count == 2
    assert metrics.rescue_used_count == 1
    assert metrics.semantic_pass_count == 1
    assert metrics.target_pass_count == 1
    assert metrics.style_pass_count == 1
    assert metrics.length_pass_count == 1
    assert metrics.lexical_pass_count == 2
    assert metrics.distillable_count == 1
    assert metrics.mean_weighted_objective == 0.15


def test_unique_id_skips_existing_directories(tmp_path: Path) -> None:
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-a-r2").mkdir()

    assert reconstruction_agentic_loop.unique_id("run-a", tmp_path) == "run-a-r3"
