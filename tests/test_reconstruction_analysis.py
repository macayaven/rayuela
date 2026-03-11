from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import reconstruction_analysis


def _make_result(
    *,
    case_id: str,
    source_work_id: str,
    source_author: str,
    target_work_id: str,
    target_author: str,
    weighted_objective: float,
    stop_reason: str = "max_iterations_reached",
    semantic_pass: bool = True,
    target_pass: bool = True,
    length_pass: bool = True,
    lexical_pass: bool = True,
) -> dict[str, Any]:
    return {
        "case": {
            "case_id": case_id,
            "control_mode": "style_shift",
            "source_window": {
                "window_id": f"{source_work_id}:ch1:w0:0-128",
                "work_id": source_work_id,
                "author": source_author,
                "title": f"{source_work_id} title",
                "chapter_number": 1,
                "segment_id": f"{source_work_id}:1",
                "chapter_word_count": 128,
                "word_start": 0,
                "word_end": 128,
                "word_count": 128,
                "text": "uno dos tres cuatro",
                "stylometric_reference": {"sent_len_mean": 1.0},
                "semantic_reference": {"metafiction": 1.0},
                "split": "test",
            },
            "target_envelope": {
                "envelope_id": f"target:{target_work_id}",
                "work_id": target_work_id,
                "author": target_author,
                "title": f"{target_work_id} title",
                "aggregation_rule": "mean",
                "provenance_window_ids": [f"{target_work_id}:ch2:w0:0-128"],
                "provenance_segment_ids": [f"{target_work_id}:2"],
                "stylometric_target": {"sent_len_mean": 2.0},
                "semantic_reference": {"metafiction": 2.0},
            },
            "uses_training_examples": False,
        },
        "prompt_template_id": "style_shift_v1",
        "iterations": [
            {
                "iteration_index": 0,
                "template_id": "style_shift_v1",
                "system_prompt": "system",
                "user_prompt": "user",
                "raw_response": "respuesta",
                "parsed_text": "respuesta",
                "score": {
                    "semantic_source_distance": 0.2,
                    "stylistic_source_distance": 0.3,
                    "stylistic_target_distance": 0.4,
                    "stylistic_target_improvement": 0.1,
                    "stylistic_target_improvement_ratio": 0.2,
                    "within_semantic_tolerance": semantic_pass,
                    "within_stylistic_tolerance": True,
                    "within_target_tolerance": target_pass,
                    "lexical_controls": {
                        "token_jaccard": 0.5,
                        "normalized_edit_similarity": 0.4,
                        "length_ratio": 1.0,
                    },
                },
                "score_history": {
                    "semantic_tolerance_pass": semantic_pass,
                    "stylistic_tolerance_pass": True,
                    "target_tolerance_pass": target_pass,
                    "length_guardrail_pass": length_pass,
                    "lexical_overlap_pass": lexical_pass,
                    "weighted_objective": weighted_objective,
                },
                "accepted_as_best": True,
            }
        ],
        "best_iteration_index": 0,
        "stop_reason": stop_reason,
        "used_training_examples": False,
    }


def _write_run(run_dir: Path, *, run_id: str, results: list[dict[str, Any]]) -> Path:
    target_dir = run_dir / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "phase": "phase-4-prompt-baselines",
        "status": "completed",
        "git_sha": "cf10ac5c86256aceb8b82e8f78483f7797c15d24",
        "model_id": "Qwen/Qwen3.5-27B-FP8",
        "prompt_template_id": "style_shift_v1",
        "corpus_manifest": "outputs/corpus/corpus_metadata.json",
        "split_manifest": "outputs/reconstruction/pilots/split_manifest.json",
        "config_payload": {
            "generation_seed": 42,
            "api_base": "http://localhost:8000/v1",
            "max_cases": 6,
            "max_iterations": 2,
            "source_windows_path": "outputs/reconstruction/pilots/source_windows.json",
            "success_criteria_path": "outputs/reconstruction/pilots/success_criteria.json",
            "target_envelopes_path": "outputs/reconstruction/pilots/target_envelopes.json",
        },
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    (target_dir / "prompt_baseline_cases.json").write_text(
        json.dumps({"generated_at": "2026-03-10T10:00:00+00:00", "results": results}) + "\n",
        encoding="utf-8",
    )
    return target_dir


def test_run_provenance_graceful_when_manifest_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "orphan-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance = reconstruction_analysis._load_run_provenance(run_dir)

    assert provenance.run_id == "orphan-run"
    assert provenance.git_sha is None
    assert provenance.prompt_template_id is None


def test_results_aggregation_is_complete(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-a",
        results=[
            _make_result(
                case_id="case-a1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.61,
            ),
            _make_result(
                case_id="case-a2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.52,
            ),
        ],
    )
    _write_run(
        run_root,
        run_id="phase4-b",
        results=[
            _make_result(
                case_id="case-b1",
                source_work_id="bolano_detectivessalvajes",
                source_author="Bolano",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.48,
            )
        ],
    )

    report = reconstruction_analysis.build_analysis_report(run_dirs=sorted(run_root.iterdir()))

    assert report.total_runs == 2
    assert report.total_cases == 3
    assert {case.run_id for case in report.cases} == {"phase4-a", "phase4-b"}
    assert {case.case_id for case in report.cases} == {"case-a1", "case-a2", "case-b1"}


def test_failure_modes_are_labeled(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-failure",
        results=[
            _make_result(
                case_id="case-failure",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.11,
                stop_reason="no_objective_improvement",
                semantic_pass=False,
                target_pass=False,
                length_pass=False,
                lexical_pass=False,
            )
        ],
    )

    report = reconstruction_analysis.build_analysis_report(run_dirs=[run_root / "phase4-failure"])

    assert report.failure_modes["semantic_drift"]["count"] == 1
    assert report.failure_modes["target_miss"]["count"] == 1
    assert report.failure_modes["length_guardrail"]["count"] == 1
    assert report.failure_modes["lexical_overlap"]["count"] == 1
    assert report.failure_modes["stalled_revision"]["count"] == 1
    assert set(report.cases[0].failure_labels) == {
        "semantic_drift",
        "target_miss",
        "length_guardrail",
        "lexical_overlap",
        "stalled_revision",
    }


def test_bias_slices_exist_by_work_and_author(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-bias",
        results=[
            _make_result(
                case_id="case-work-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.65,
            ),
            _make_result(
                case_id="case-work-2",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="bolano_detectivessalvajes",
                target_author="Bolano",
                weighted_objective=0.35,
            ),
            _make_result(
                case_id="case-work-3",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="bolano_detectivessalvajes",
                target_author="Bolano",
                weighted_objective=0.41,
            ),
        ],
    )

    report = reconstruction_analysis.build_analysis_report(run_dirs=[run_root / "phase4-bias"])

    work_ids = {slice_record.slice_key for slice_record in report.bias_slices["by_work"]}
    author_ids = {slice_record.slice_key for slice_record in report.bias_slices["by_author"]}

    assert work_ids == {"borges_elaleph", "rulfo_pedroparamo"}
    assert author_ids == {"Borges", "Rulfo"}
    assert report.bias_slices["by_work"][0].count >= 1
    assert report.bias_slices["by_author"][0].count >= 1


def test_close_reading_notes_are_linked_to_run_ids(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-notes",
        results=[
            _make_result(
                case_id="case-top",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.81,
            ),
            _make_result(
                case_id="case-bottom",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.19,
                semantic_pass=False,
            ),
        ],
    )

    report = reconstruction_analysis.build_analysis_report(run_dirs=[run_root / "phase4-notes"])

    assert len(report.close_reading_notes) == 2
    assert {note.run_id for note in report.close_reading_notes} == {"phase4-notes"}
    assert {note.case_id for note in report.close_reading_notes} == {"case-top", "case-bottom"}
    assert report.close_reading_notes[0].cases_path.endswith(
        "phase4-notes/prompt_baseline_cases.json"
    )
    assert report.close_reading_notes[0].manifest_path.endswith("phase4-notes/manifest.json")


def test_per_run_summary_and_case_paths_are_traceable(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-trace-a",
        results=[
            _make_result(
                case_id="case-trace-a1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.71,
            ),
            _make_result(
                case_id="case-trace-a2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.29,
                semantic_pass=False,
            ),
        ],
    )
    _write_run(
        run_root,
        run_id="phase4-trace-b",
        results=[
            _make_result(
                case_id="case-trace-b1",
                source_work_id="bolano_detectivessalvajes",
                source_author="Bolano",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.43,
            )
        ],
    )

    report = reconstruction_analysis.build_analysis_report(run_dirs=sorted(run_root.iterdir()))

    assert report.run_summaries["phase4-trace-a"]["case_count"] == 2
    assert report.run_summaries["phase4-trace-a"]["failure_case_count"] == 1
    assert report.run_summaries["phase4-trace-b"]["case_count"] == 1
    assert report.cases[0].cases_path.endswith(
        f"{report.cases[0].run_id}/prompt_baseline_cases.json"
    )
    assert report.cases[0].manifest_path.endswith(f"{report.cases[0].run_id}/manifest.json")


def test_run_comparisons_capture_case_level_deltas(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-comp-a",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.40,
            ),
            _make_result(
                case_id="case-shared-2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.50,
            ),
        ],
    )
    _write_run(
        run_root,
        run_id="phase4-comp-b",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.55,
            ),
            _make_result(
                case_id="case-shared-2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.45,
                semantic_pass=False,
            ),
        ],
    )

    report = reconstruction_analysis.build_analysis_report(
        run_dirs=sorted(run_root.iterdir()),
        promotion_criteria=reconstruction_analysis.PromotionCriteria(
            min_overlapping_cases=2,
            min_mean_delta=0.01,
            min_median_delta=0.0,
            min_non_negative_share=0.5,
            max_failure_case_delta=1,
        ),
    )

    assert len(report.run_comparisons) == 1
    comparison = report.run_comparisons[0]
    assert comparison.reference_run_id == "phase4-comp-a"
    assert comparison.candidate_run_id == "phase4-comp-b"
    assert comparison.overlapping_case_count == 2
    assert comparison.improved_case_count == 1
    assert comparison.worsened_case_count == 1
    assert comparison.mean_weighted_objective_delta == pytest.approx(0.05)
    assert comparison.median_weighted_objective_delta == pytest.approx(0.05)
    assert (
        comparison.mean_weighted_objective_delta_ci_low <= comparison.mean_weighted_objective_delta
    )
    assert (
        comparison.mean_weighted_objective_delta_ci_high >= comparison.mean_weighted_objective_delta
    )
    assert comparison.bootstrap_resamples >= 100
    assert comparison.non_negative_case_share == pytest.approx(0.5)
    assert comparison.failure_case_count_delta == 1
    assert comparison.largest_gain_case_id == "case-shared-1"
    assert comparison.largest_gain_delta == pytest.approx(0.15)
    assert comparison.largest_drop_case_id == "case-shared-2"
    assert comparison.largest_drop_delta == pytest.approx(-0.05)
    assert comparison.comparable is True
    assert comparison.comparability_issues == ()
    assert comparison.comparability_checks["git_sha"] is True
    semantic_transition = next(
        record for record in comparison.failure_transitions if record.label == "semantic_drift"
    )
    assert semantic_transition.introduced_count == 1
    assert semantic_transition.resolved_count == 0
    assert semantic_transition.persistent_count == 0
    assert isinstance(comparison.mean_weighted_objective_delta_ci_excludes_zero, bool)


def test_constant_deltas_produce_point_interval(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-ci-a",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.20,
            ),
            _make_result(
                case_id="case-shared-2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.40,
            ),
        ],
    )
    _write_run(
        run_root,
        run_id="phase4-ci-b",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.30,
            ),
            _make_result(
                case_id="case-shared-2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.50,
            ),
        ],
    )

    report = reconstruction_analysis.build_analysis_report(run_dirs=sorted(run_root.iterdir()))

    comparison = report.run_comparisons[0]
    assert comparison.mean_weighted_objective_delta == pytest.approx(0.1)
    assert comparison.mean_weighted_objective_delta_ci_low == pytest.approx(0.1)
    assert comparison.mean_weighted_objective_delta_ci_high == pytest.approx(0.1)
    assert comparison.mean_weighted_objective_delta_ci_excludes_zero is True


def test_promotion_recommendations_are_separate_from_scheduler_decisions(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-promo-a",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.40,
            ),
            _make_result(
                case_id="case-shared-2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.50,
            ),
        ],
    )
    _write_run(
        run_root,
        run_id="phase4-promo-b",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.55,
            ),
            _make_result(
                case_id="case-shared-2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.45,
            ),
        ],
    )
    _write_run(
        run_root,
        run_id="phase4-promo-c",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.56,
            ),
            _make_result(
                case_id="case-shared-2",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.46,
            ),
        ],
    )

    report = reconstruction_analysis.build_analysis_report(
        run_dirs=sorted(run_root.iterdir()),
        promotion_criteria=reconstruction_analysis.PromotionCriteria(
            min_overlapping_cases=2,
            min_mean_delta=0.04,
            min_median_delta=0.0,
            min_non_negative_share=0.5,
            max_failure_case_delta=0,
        ),
    )

    assert report.promotion_recommendations[0].reference_run_id == "phase4-promo-a"
    assert report.promotion_recommendations[0].candidate_run_id == "phase4-promo-b"
    assert report.promotion_recommendations[0].recommendation == "promote"
    assert report.promotion_recommendations[0].criteria_results["comparable_provenance"] is True
    assert report.promotion_recommendations[0].criteria_results["mean_delta"] is True

    assert report.promotion_recommendations[1].reference_run_id == "phase4-promo-b"
    assert report.promotion_recommendations[1].candidate_run_id == "phase4-promo-c"
    assert report.promotion_recommendations[1].recommendation == "reject"
    assert report.promotion_recommendations[1].criteria_results["mean_delta"] is False


def test_promotion_recommendation_holds_when_provenance_is_not_comparable(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-prov-a",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.40,
            )
        ],
    )
    second_run = _write_run(
        run_root,
        run_id="phase4-prov-b",
        results=[
            _make_result(
                case_id="case-shared-1",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.60,
            )
        ],
    )
    manifest_path = second_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_payload"]["generation_seed"] = 7
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    report = reconstruction_analysis.build_analysis_report(run_dirs=sorted(run_root.iterdir()))

    assert len(report.run_comparisons) == 1
    comparison = report.run_comparisons[0]
    assert comparison.comparable is False
    assert "generation_seed" in comparison.comparability_issues[0]
    recommendation = report.promotion_recommendations[0]
    assert recommendation.recommendation == "hold"
    assert recommendation.criteria_results["comparable_provenance"] is False
    assert "provenance" in recommendation.rationale


def test_promotion_recommendation_holds_when_runs_do_not_overlap(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-no-overlap-a",
        results=[
            _make_result(
                case_id="case-a",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.40,
            )
        ],
    )
    _write_run(
        run_root,
        run_id="phase4-no-overlap-b",
        results=[
            _make_result(
                case_id="case-b",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="bolano_detectivessalvajes",
                target_author="Bolano",
                weighted_objective=0.60,
            )
        ],
    )

    report = reconstruction_analysis.build_analysis_report(run_dirs=sorted(run_root.iterdir()))

    assert report.run_comparisons == ()
    assert report.promotion_recommendations[0].recommendation == "hold"
    assert report.promotion_recommendations[0].criteria_results["comparable_provenance"] is False
    assert "no overlapping cases" in report.promotion_recommendations[0].rationale


def test_article_inputs_exist(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-article",
        results=[
            _make_result(
                case_id="case-article",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.59,
            )
        ],
    )
    report = reconstruction_analysis.build_analysis_report(run_dirs=[run_root / "phase4-article"])

    paths = reconstruction_analysis.write_analysis_artifacts(
        report,
        output_dir=tmp_path / "analysis",
    )

    assert paths["summary"].exists()
    assert paths["report"].exists()
    assert paths["article_inputs"].exists()

    article_inputs = json.loads(paths["article_inputs"].read_text(encoding="utf-8"))
    assert article_inputs["total_cases"] == 1
    assert article_inputs["close_reading_queue"][0]["run_id"] == "phase4-article"
    assert article_inputs["run_comparisons"] == []
    assert article_inputs["promotion_recommendations"] == []
    assert article_inputs["promotion_criteria"]["min_overlapping_cases"] >= 1
    assert article_inputs["comparability_summary"]["comparable_comparison_count"] == 0
    assert article_inputs["run_provenance"]["phase4-article"]["generation_seed"] == 42


def test_write_analysis_artifacts_optionally_logs_to_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {
        "init": None,
        "logs": [],
        "summary_updates": [],
        "artifacts": [],
        "finished": False,
    }

    class _Artifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type
            self.files: list[tuple[str, str | None]] = []

        def add_file(self, local_path: str, name: str | None = None) -> None:
            self.files.append((local_path, name))

    class _Table:
        def __init__(self, *, columns: list[str], data: list[list[object]]) -> None:
            self.columns = columns
            self.data = data

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
            calls["finished"] = True

    class _WandbStub:
        Artifact = _Artifact
        Table = _Table

        @staticmethod
        def init(**kwargs: object) -> _Run:
            calls["init"] = kwargs
            return _Run()

    monkeypatch.setattr(reconstruction_analysis, "_load_wandb_module", lambda: _WandbStub)

    run_root = tmp_path / "runs"
    _write_run(
        run_root,
        run_id="phase4-analysis-a",
        results=[
            _make_result(
                case_id="case-top",
                source_work_id="borges_elaleph",
                source_author="Borges",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.81,
            ),
            _make_result(
                case_id="case-bottom",
                source_work_id="rulfo_pedroparamo",
                source_author="Rulfo",
                target_work_id="cortazar_rayuela",
                target_author="Cortazar",
                weighted_objective=0.19,
                semantic_pass=False,
            ),
        ],
    )
    report = reconstruction_analysis.build_analysis_report(
        run_dirs=[run_root / "phase4-analysis-a"]
    )

    reconstruction_analysis.write_analysis_artifacts(
        report,
        output_dir=tmp_path / "analysis",
        wandb_project="rayuela",
        wandb_entity="macayaven",
        wandb_mode="offline",
        wandb_group="guided-20260310a",
    )

    assert calls["init"] == {
        "project": "rayuela",
        "entity": "macayaven",
        "mode": "offline",
        "group": "guided-20260310a",
        "job_type": "reconstruction_analysis",
        "tags": ["phase6", "analysis", "article-inputs"],
        "name": "analysis-phase4-analysis-a",
        "config": {
            "run_ids": ["phase4-analysis-a"],
            "total_runs": 1,
            "total_cases": 2,
            "failure_labels": [
                "length_guardrail",
                "lexical_overlap",
                "semantic_drift",
                "stalled_revision",
                "target_miss",
            ],
            "promotion_criteria": {
                "min_overlapping_cases": 4,
                "min_mean_delta": 0.005,
                "min_median_delta": 0.0,
                "min_non_negative_share": 0.5,
                "max_failure_case_delta": 0,
                "require_comparable_provenance": True,
            },
            "comparability_invariant_fields": [
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
            ],
        },
    }

    log_calls = calls["logs"]
    assert isinstance(log_calls, list)
    assert log_calls[0]["analysis/total_runs"] == 1.0
    assert log_calls[0]["analysis/total_cases"] == 2.0
    assert log_calls[0]["analysis/failure_mode_semantic_drift_count"] == 1.0
    assert log_calls[0]["article/close_reading_note_count"] == 2.0
    assert isinstance(log_calls[1]["analysis/run_summary_table"], _Table)
    assert isinstance(log_calls[2]["analysis/run_comparison_table"], _Table)
    assert isinstance(log_calls[3]["analysis/failure_transition_table"], _Table)
    assert isinstance(log_calls[4]["analysis/promotion_table"], _Table)
    assert isinstance(log_calls[5]["analysis/run_provenance_table"], _Table)
    assert isinstance(log_calls[6]["article/close_reading_queue"], _Table)

    summary_updates = calls["summary_updates"]
    assert isinstance(summary_updates, list)
    assert summary_updates[0]["summary_path"].endswith("reconstruction_analysis_summary.json")
    assert summary_updates[0]["article_inputs_path"].endswith("reconstruction_article_inputs.json")
    assert summary_updates[0]["headline_findings"][0].startswith(
        "Best observed reconstruction objective"
    )
    assert "comparability_summary" in summary_updates[0]
    assert "promotion_summary" in summary_updates[0]

    artifact_calls = calls["artifacts"]
    assert isinstance(artifact_calls, list)
    assert artifact_calls[0]["type"] == "reconstruction-analysis"
    assert {
        artifact_name
        for _, artifact_name in artifact_calls[0]["files"]
        if artifact_name is not None
    } == {
        "reconstruction_analysis_summary.json",
        "reconstruction_analysis_report.md",
        "reconstruction_article_inputs.json",
    }
    assert calls["finished"] is True


def test_schedule_summary_resolves_kept_run_dirs(tmp_path: Path) -> None:
    summary_path = tmp_path / "scheduler" / "guided-20260310a" / "schedule_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schedule_id": "guided-20260310a",
                "kept_run_ids": ["phase4-a", "phase4-b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_dirs = reconstruction_analysis.run_dirs_from_schedule_summary(
        summary_path,
        project_root=tmp_path,
    )

    assert run_dirs == [
        tmp_path / "outputs" / "reconstruction" / "runs" / "phase4-a",
        tmp_path / "outputs" / "reconstruction" / "runs" / "phase4-b",
    ]
