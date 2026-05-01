from __future__ import annotations

import json
from pathlib import Path

import pytest

import reconstruction_style_distill


def _teacher_result(
    *,
    case_id: str = "style_shift:source:target",
    parsed_text: str = "Texto reescrito.",
    weighted_objective: float = 0.42,
    semantic_pass: bool = True,
    target_pass: bool = False,
) -> dict[str, object]:
    return {
        "case": {
            "case_id": case_id,
            "control_mode": "style_shift",
            "source_window": {
                "window_id": "source",
                "text": "Texto fuente.",
                "stylometric_reference": {
                    "sent_len_mean": 8.0,
                    "mattr": 0.72,
                },
            },
            "target_envelope": {
                "envelope_id": "target:cortazar",
                "author": "Cortázar",
                "title": "Rayuela envelope",
                "stylometric_target": {
                    "sent_len_mean": 14.0,
                    "mattr": 0.61,
                },
            },
            "uses_training_examples": False,
        },
        "prompt_template_id": "style_shift_v2",
        "iterations": [
            {
                "iteration_index": 0,
                "parsed_text": parsed_text,
                "score_history": {
                    "weighted_objective": weighted_objective,
                    "semantic_tolerance_pass": semantic_pass,
                    "target_tolerance_pass": target_pass,
                    "length_guardrail_pass": True,
                    "lexical_overlap_pass": True,
                },
            }
        ],
        "best_iteration_index": 0,
        "stop_reason": "max_iterations",
        "used_training_examples": False,
        "error_message": None,
    }


def _write_teacher_cases(path: Path, results: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00+00:00",
                "results": results,
                "case_failures": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_load_distillable_results_filters_by_score_and_passes(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    _write_teacher_cases(
        cases_path,
        [
            _teacher_result(case_id="low", weighted_objective=0.1),
            _teacher_result(case_id="empty", parsed_text="  ", weighted_objective=0.9),
            _teacher_result(case_id="semantic-fail", semantic_pass=False, weighted_objective=0.9),
            _teacher_result(case_id="target-pass", target_pass=True, weighted_objective=0.9),
        ],
    )

    results = reconstruction_style_distill.load_distillable_results(
        cases_path,
        min_weighted_objective=0.2,
        require_semantic_pass=True,
        require_target_pass=True,
    )

    assert [result["case"]["case_id"] for result in results] == ["target-pass"]


def test_result_to_distilled_example_builds_training_payload(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    result = _teacher_result()

    distilled = reconstruction_style_distill.result_to_distilled_example(
        result,
        teacher_cases_path=cases_path,
        seed=7,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    payload = distilled.to_dict()

    assert payload["window_id"] == "style_shift:source:target"
    assert payload["source_text"] == "Texto fuente."
    assert payload["target_text"] == "Texto reescrito."
    assert payload["target_envelope_id"] == "target:cortazar"
    assert payload["dataset_mode"] == "style_transfer_distilled"
    assert "Cortázar" in payload["instruction"]
    assert "sent_len_mean" in payload["instruction"]
    assert payload["teacher_metadata"]["weighted_objective"] == 0.42


def test_write_distilled_dataset_writes_split_files_and_manifest(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    distilled = reconstruction_style_distill.result_to_distilled_example(
        _teacher_result(),
        teacher_cases_path=cases_path,
        seed=7,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    manifest = reconstruction_style_distill.write_distilled_dataset(
        [distilled],
        tmp_path / "style-dataset",
    )

    assert manifest["example_count"] == 1
    assert sum(manifest["split_counts"].values()) == 1
    manifest_path = tmp_path / "style-dataset" / "distill_manifest.json"
    assert manifest_path.exists()
    lines = []
    for split in ("train", "val", "test"):
        lines.extend(
            (tmp_path / "style-dataset" / "training_dataset" / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    assert len(lines) == 1
    assert json.loads(lines[0])["teacher_metadata"]["case_id"] == "style_shift:source:target"


def test_write_distilled_dataset_rejects_empty_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no distillable teacher examples"):
        reconstruction_style_distill.write_distilled_dataset([], tmp_path / "empty")


def test_main_deduplicates_teacher_cases(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    output_root = tmp_path / "distill"
    _write_teacher_cases(
        cases_path,
        [
            _teacher_result(case_id="duplicate"),
            _teacher_result(case_id="duplicate", parsed_text="Texto alternativo."),
        ],
    )

    exit_code = reconstruction_style_distill.main(
        [
            "--teacher-cases-path",
            str(cases_path),
            "--dataset-id",
            "dataset-a",
            "--output-root",
            str(output_root),
            "--min-weighted-objective",
            "0.2",
        ]
    )

    assert exit_code == 0
    manifest = json.loads(
        (output_root / "dataset-a" / "distill_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["example_count"] == 1
