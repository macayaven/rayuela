from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import reconstruction_baselines
import reconstruction_dataset
import reconstruction_metrics


def _manual_baseline(
    kind: Literal["stylometric", "semantic"],
    names: list[str],
    matrix: np.ndarray,
) -> reconstruction_metrics.MeasurementBaseline:
    registry = reconstruction_metrics.build_dimension_registry(
        kind=kind,
        names=names,
        descriptions={name: f"{name} description" for name in names},
    )
    measurements = reconstruction_metrics.MeasurementMatrix(
        kind=kind,
        dimension_order=tuple(names),
        matrix=matrix,
        segment_ids=tuple(f"seg:{index}" for index in range(matrix.shape[0])),
        source_paths=(),
        dimension_registry=registry,
    )
    return reconstruction_metrics.compute_measurement_baseline(measurements)


def _success_criteria() -> reconstruction_dataset.SuccessCriteria:
    return reconstruction_dataset.build_success_criteria()


def _source_window() -> reconstruction_dataset.WindowRecord:
    return reconstruction_dataset.WindowRecord(
        window_id="source:1",
        work_id="source_work",
        author="Source Author",
        title="Source Title",
        chapter_number=1,
        segment_id="source_work:1",
        chapter_word_count=12,
        word_start=0,
        word_end=12,
        word_count=12,
        text="uno dos tres cuatro cinco seis siete ocho nueve diez once doce",
        stylometric_reference={"sent_len_mean": 1.0, "mattr": 1.0},
        semantic_reference={"existential_questioning": 1.0, "metafiction": 1.0},
        split="test",
    )


def _target_envelope() -> reconstruction_dataset.TargetEnvelope:
    return reconstruction_dataset.TargetEnvelope(
        envelope_id="target:1",
        work_id="target_work",
        author="Target Author",
        title="Target Title",
        aggregation_rule="mean",
        provenance_window_ids=("target:window:1",),
        provenance_segment_ids=("target_work:2",),
        stylometric_target={"sent_len_mean": 3.0, "mattr": 3.0},
        semantic_reference={"existential_questioning": 2.0, "metafiction": 2.0},
    )


def _baseline_case(
    *,
    control_mode: str = "style_shift",
    uses_training_examples: bool = False,
) -> reconstruction_baselines.BaselineCase:
    return reconstruction_baselines.BaselineCase(
        case_id=f"{control_mode}:source:1:target:1",
        control_mode=control_mode,
        source_window=_source_window(),
        target_envelope=_target_envelope(),
        uses_training_examples=uses_training_examples,
    )


class StubPromptBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.requests: list[reconstruction_baselines.PromptRequest] = []

    def generate(self, request: reconstruction_baselines.PromptRequest) -> str:
        self.requests.append(request)
        return self._responses[len(self.requests) - 1]


class StubMeasurementBackend:
    def __init__(
        self,
        measurements: dict[str, reconstruction_baselines.CandidateMeasurements],
    ) -> None:
        self._measurements = measurements

    def measure(
        self,
        *,
        source_window: reconstruction_dataset.WindowRecord,
        target_envelope: reconstruction_dataset.TargetEnvelope,
        candidate_text: str,
        control_mode: str,
    ) -> reconstruction_baselines.CandidateMeasurements:
        del source_window, target_envelope, control_mode
        return self._measurements[candidate_text]


def test_prompt_generation_returns_traceable_output() -> None:
    stylometric_baseline = _manual_baseline(
        "stylometric",
        ["sent_len_mean", "mattr"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    semantic_baseline = _manual_baseline(
        "semantic",
        ["existential_questioning", "metafiction"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    prompt_backend = StubPromptBackend(["respuesta candidata"])
    measurement_backend = StubMeasurementBackend(
        {
            "respuesta candidata": reconstruction_baselines.CandidateMeasurements(
                stylometric=np.array([2.5, 2.5]),
                semantic=np.array([1.2, 1.2]),
            )
        }
    )

    result = reconstruction_baselines.run_prompt_case(
        case=_baseline_case(),
        prompt_backend=prompt_backend,
        measurement_backend=measurement_backend,
        stylometric_baseline=stylometric_baseline,
        semantic_baseline=semantic_baseline,
        success_criteria=_success_criteria(),
        max_iterations=1,
    )

    assert result.prompt_template_id == "style_shift_v1"
    assert result.used_training_examples is False
    assert len(result.iterations) == 1
    assert result.iterations[0].template_id == "style_shift_v1"
    assert result.iterations[0].raw_response == "respuesta candidata"
    assert result.iterations[0].parsed_text == "respuesta candidata"
    assert result.iterations[0].score_history["weighted_objective"] > 0.0


def test_iteration_history_is_saved(tmp_path: Path) -> None:
    result = reconstruction_baselines.BaselineCaseResult(
        case=_baseline_case(),
        prompt_template_id="style_shift_v1",
        iterations=(
            reconstruction_baselines.IterationRecord(
                iteration_index=0,
                template_id="style_shift_v1",
                system_prompt="system",
                user_prompt="user",
                raw_response="raw one",
                parsed_text="text one",
                score=reconstruction_metrics.ReconstructionScore(
                    semantic_source_distance=0.1,
                    stylistic_source_distance=0.2,
                    stylistic_target_distance=0.3,
                    stylistic_target_improvement=0.4,
                    stylistic_target_improvement_ratio=0.5,
                    within_semantic_tolerance=True,
                    within_stylistic_tolerance=True,
                    within_target_tolerance=True,
                    lexical_controls=reconstruction_metrics.compute_lexical_controls(
                        "uno dos tres cuatro",
                        "uno dos tres cinco",
                    ),
                ),
                score_history={"weighted_objective": 0.42},
                accepted_as_best=True,
            ),
            reconstruction_baselines.IterationRecord(
                iteration_index=1,
                template_id="revise_v1",
                system_prompt="system",
                user_prompt="revise",
                raw_response="raw two",
                parsed_text="text two",
                score=reconstruction_metrics.ReconstructionScore(
                    semantic_source_distance=0.05,
                    stylistic_source_distance=0.15,
                    stylistic_target_distance=0.2,
                    stylistic_target_improvement=0.5,
                    stylistic_target_improvement_ratio=0.6,
                    within_semantic_tolerance=True,
                    within_stylistic_tolerance=True,
                    within_target_tolerance=True,
                    lexical_controls=reconstruction_metrics.compute_lexical_controls(
                        "uno dos tres cuatro",
                        "uno dos tres cinco seis",
                    ),
                ),
                score_history={"weighted_objective": 0.57},
                accepted_as_best=True,
            ),
        ),
        best_iteration_index=1,
        stop_reason="max_iterations_reached",
        used_training_examples=False,
    )

    cases_path = tmp_path / "cases.json"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    reconstruction_baselines.write_baseline_artifacts(
        results=[result],
        cases_path=cases_path,
        summary_path=summary_path,
        report_path=report_path,
    )

    cases_payload = json.loads(cases_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(cases_payload["results"]) == 1
    assert len(cases_payload["results"][0]["iterations"]) == 2
    assert summary_payload["total_cases"] == 1
    assert summary_payload["controls"]["style_shift"]["count"] == 1
    assert "weighted objective" in report_path.read_text(encoding="utf-8").lower()


def test_revision_loop_improves_weighted_objective_or_stops_cleanly() -> None:
    stylometric_baseline = _manual_baseline(
        "stylometric",
        ["sent_len_mean", "mattr"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    semantic_baseline = _manual_baseline(
        "semantic",
        ["existential_questioning", "metafiction"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    prompt_backend = StubPromptBackend(["primer intento", "segundo intento"])
    measurement_backend = StubMeasurementBackend(
        {
            "primer intento": reconstruction_baselines.CandidateMeasurements(
                stylometric=np.array([1.4, 1.4]),
                semantic=np.array([1.1, 1.1]),
            ),
            "segundo intento": reconstruction_baselines.CandidateMeasurements(
                stylometric=np.array([2.8, 2.8]),
                semantic=np.array([1.05, 1.05]),
            ),
        }
    )

    result = reconstruction_baselines.run_prompt_case(
        case=_baseline_case(),
        prompt_backend=prompt_backend,
        measurement_backend=measurement_backend,
        stylometric_baseline=stylometric_baseline,
        semantic_baseline=semantic_baseline,
        success_criteria=_success_criteria(),
        max_iterations=2,
    )

    objectives = [step.score_history["weighted_objective"] for step in result.iterations]

    assert len(result.iterations) == 2
    assert objectives[1] > objectives[0]
    assert result.best_iteration_index == 1
    assert result.stop_reason == "max_iterations_reached"


def test_prompt_baseline_respects_length_guardrails() -> None:
    stylometric_baseline = _manual_baseline(
        "stylometric",
        ["sent_len_mean", "mattr"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    semantic_baseline = _manual_baseline(
        "semantic",
        ["existential_questioning", "metafiction"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    candidate_text = " ".join(f"palabra{i}" for i in range(40))
    prompt_backend = StubPromptBackend([candidate_text])
    measurement_backend = StubMeasurementBackend(
        {
            candidate_text: reconstruction_baselines.CandidateMeasurements(
                stylometric=np.array([2.5, 2.5]),
                semantic=np.array([1.1, 1.1]),
            )
        }
    )

    result = reconstruction_baselines.run_prompt_case(
        case=_baseline_case(),
        prompt_backend=prompt_backend,
        measurement_backend=measurement_backend,
        stylometric_baseline=stylometric_baseline,
        semantic_baseline=semantic_baseline,
        success_criteria=_success_criteria(),
        max_iterations=1,
    )

    lexical_controls = result.final_iteration.score.lexical_controls

    assert lexical_controls is not None
    assert lexical_controls.length_ratio > 1.2
    assert result.final_iteration.score_history["length_guardrail_pass"] is False


def test_prompt_baseline_does_not_train_on_eval_examples() -> None:
    stylometric_baseline = _manual_baseline(
        "stylometric",
        ["sent_len_mean", "mattr"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    semantic_baseline = _manual_baseline(
        "semantic",
        ["existential_questioning", "metafiction"],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )

    with pytest.raises(ValueError, match="training examples"):
        reconstruction_baselines.run_prompt_case(
            case=_baseline_case(uses_training_examples=True),
            prompt_backend=StubPromptBackend(["respuesta"]),
            measurement_backend=StubMeasurementBackend(
                {
                    "respuesta": reconstruction_baselines.CandidateMeasurements(
                        stylometric=np.array([2.0, 2.0]),
                        semantic=np.array([1.0, 1.0]),
                    )
                }
            ),
            stylometric_baseline=stylometric_baseline,
            semantic_baseline=semantic_baseline,
            success_criteria=_success_criteria(),
        )
