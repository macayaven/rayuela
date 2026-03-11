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


def test_load_target_envelopes_restores_tuple_provenance(tmp_path: Path) -> None:
    path = tmp_path / "target_envelopes.json"
    payload = {
        "target_envelopes": [
            {
                "envelope_id": "target:1",
                "work_id": "target_work",
                "author": "Target Author",
                "title": "Target Title",
                "aggregation_rule": "mean",
                "provenance_window_ids": ["target:window:1", "target:window:2"],
                "provenance_segment_ids": ["target_work:2", "target_work:3"],
                "stylometric_target": {"sent_len_mean": 3.0, "mattr": 3.0},
                "semantic_reference": {
                    "existential_questioning": 2.0,
                    "metafiction": 2.0,
                },
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = reconstruction_baselines.load_target_envelopes(path)

    assert len(loaded) == 1
    assert loaded[0].provenance_window_ids == ("target:window:1", "target:window:2")
    assert loaded[0].provenance_segment_ids == ("target_work:2", "target_work:3")
    assert isinstance(loaded[0].provenance_window_ids, tuple)
    assert isinstance(loaded[0].provenance_segment_ids, tuple)


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


def test_openai_prompt_backend_passes_seed_to_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class _Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            calls["create"] = kwargs
            return type(
                "_Response",
                (),
                {
                    "choices": [
                        type(
                            "_Choice",
                            (),
                            {"message": type("_Message", (), {"content": "respuesta"})()},
                        )()
                    ]
                },
            )()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            calls["client_init"] = {
                "base_url": base_url,
                "api_key": api_key,
            }
            self.chat = _Chat()

    monkeypatch.setattr(reconstruction_baselines, "_load_openai_client", lambda: _Client)
    backend = reconstruction_baselines.OpenAIPromptBackend(
        api_base="http://localhost:8000/v1",
        model="Qwen/Qwen3.5-27B-FP8",
        temperature=0.1,
        max_tokens=64,
        seed=17,
    )

    content = backend.generate(
        reconstruction_baselines.PromptRequest(
            case_id="case-1",
            control_mode="style_shift",
            iteration_index=0,
            template_id="style_shift_v1",
            source_window_id="source:1",
            target_envelope_id="target:1",
            system_prompt="system",
            user_prompt="user",
            metadata={},
        )
    )

    assert content == "respuesta"
    assert calls["client_init"] == {
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-needed",
    }
    assert calls["create"] == {
        "model": "Qwen/Qwen3.5-27B-FP8",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 0.1,
        "max_tokens": 64,
        "seed": 17,
    }


def test_openai_prompt_backend_prefers_final_content_over_reasoning_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            del kwargs
            return type(
                "_Response",
                (),
                {
                    "choices": [
                        type(
                            "_Choice",
                            (),
                            {
                                "message": type(
                                    "_Message",
                                    (),
                                    {
                                        "content": "respuesta final",
                                        "reasoning_content": "razonamiento oculto",
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            del base_url, api_key
            self.chat = _Chat()

    monkeypatch.setattr(reconstruction_baselines, "_load_openai_client", lambda: _Client)
    backend = reconstruction_baselines.OpenAIPromptBackend()

    content = backend.generate(
        reconstruction_baselines.PromptRequest(
            case_id="case-1",
            control_mode="style_shift",
            iteration_index=0,
            template_id="style_shift_v1",
            source_window_id="source:1",
            target_envelope_id="target:1",
            system_prompt="system",
            user_prompt="user",
            metadata={},
        )
    )

    assert content == "respuesta final"


def test_openai_prompt_backend_fails_when_reasoning_exists_without_final_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            del kwargs
            return type(
                "_Response",
                (),
                {
                    "choices": [
                        type(
                            "_Choice",
                            (),
                            {
                                "message": type(
                                    "_Message",
                                    (),
                                    {
                                        "content": "  ",
                                        "reasoning": "Thinking Process: razonamiento",
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            del base_url, api_key
            self.chat = _Chat()

    monkeypatch.setattr(reconstruction_baselines, "_load_openai_client", lambda: _Client)
    backend = reconstruction_baselines.OpenAIPromptBackend(max_tokens=256)

    with pytest.raises(RuntimeError, match="returned reasoning without final content"):
        backend.generate(
            reconstruction_baselines.PromptRequest(
                case_id="case-1",
                control_mode="style_shift",
                iteration_index=0,
                template_id="style_shift_v1",
                source_window_id="source:1",
                target_envelope_id="target:1",
                system_prompt="system",
                user_prompt="user",
                metadata={},
            )
        )


def test_build_argument_parser_exposes_generation_max_tokens() -> None:
    args = reconstruction_baselines.build_argument_parser().parse_args(
        [
            "--run-id",
            "phase4-test",
            "--generation-max-tokens",
            "2048",
        ]
    )

    assert args.generation_max_tokens == 2048


def test_build_argument_parser_exposes_semantic_generation_max_tokens() -> None:
    args = reconstruction_baselines.build_argument_parser().parse_args(
        [
            "--run-id",
            "phase4-test",
            "--semantic-generation-max-tokens",
            "1536",
        ]
    )

    assert args.semantic_generation_max_tokens == 1536


def test_corpus_measurement_backend_passes_semantic_generation_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "semantic_prompt.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    calls: dict[str, object] = {}

    class _Client:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            calls["client_init"] = {
                "base_url": base_url,
                "api_key": api_key,
            }

    def _extract_chapter(
        client: object,
        prompt: str,
        chapter: dict[str, object],
        model: str,
        with_evidence: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, int]:
        calls["extract_chapter"] = {
            "client": client,
            "prompt": prompt,
            "chapter": chapter,
            "model": model,
            "with_evidence": with_evidence,
            "max_tokens": max_tokens,
        }
        return {
            "existential_questioning": 4,
            "metafiction": 7,
            "temporal_clarity": 5,
        }

    def _fake_import_module(name: str) -> object:
        if name == "semantic_extraction":
            return type(
                "_SemanticModule",
                (),
                {
                    "DIMENSIONS": [
                        "existential_questioning",
                        "metafiction",
                        "temporal_clarity",
                    ],
                    "extract_chapter": staticmethod(_extract_chapter),
                },
            )()
        return __import__(name)

    monkeypatch.setattr(reconstruction_baselines, "_load_openai_client", lambda: _Client)
    monkeypatch.setattr(reconstruction_baselines.importlib, "import_module", _fake_import_module)

    backend = reconstruction_baselines.CorpusMeasurementBackend(
        api_base="http://localhost:8000/v1",
        model="Qwen/Qwen3.5-27B-FP8",
        semantic_prompt_path=prompt_path,
        semantic_max_tokens=1536,
    )

    vector = backend._measure_semantic("texto candidato")

    assert calls["client_init"] == {
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-needed",
    }
    assert np.array_equal(vector, np.array([4.0, 7.0]))
    assert calls["extract_chapter"] == {
        "client": backend._client,
        "prompt": "prompt",
        "chapter": {
            "number": 0,
            "section": "rewrite",
            "text": "texto candidato",
        },
        "model": "Qwen/Qwen3.5-27B-FP8",
        "with_evidence": False,
        "max_tokens": 1536,
    }


def test_parse_generated_text_strips_leading_think_block() -> None:
    parsed = reconstruction_baselines.parse_generated_text(
        "<think>razono internamente</think>\n\nTexto final."
    )

    assert parsed == "Texto final."
