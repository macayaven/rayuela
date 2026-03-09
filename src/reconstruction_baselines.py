#!/usr/bin/env python3
"""
Prompt-only baselines and generate-score-revise controls for Part 3.

Phase 4 establishes a no-training floor before any adapter fine-tuning. The
module runs traceable prompt baselines against the locked Phase 3 pilot,
scores every iteration with the Phase 2 measurement contract, and persists
full histories under immutable run directories.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from project_config import CORPUS_DIR, CORPUS_OUTPUT_DIR
from reconstruction_contract import (
    DEFAULT_RECONSTRUCTION_SEED,
    ReconstructionPaths,
    RunStatus,
    build_run_manifest,
    detect_git_sha,
    finalize_run_manifest,
    prepare_run_directory,
    seed_everything,
    to_project_relative,
    write_run_manifest,
)
from reconstruction_dataset import (
    DEFAULT_SOURCE_WINDOWS_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_SUCCESS_CRITERIA_PATH,
    DEFAULT_TARGET_ENVELOPES_PATH,
    SuccessCriteria,
    TargetEnvelope,
    WindowRecord,
)
from reconstruction_metrics import (
    MeasurementBaseline,
    ToleranceConfig,
    build_measurement_baselines,
    score_rewrite,
)

DEFAULT_CASES_PATH = ReconstructionPaths().baselines_dir / "prompt_baseline_cases.json"
DEFAULT_SUMMARY_PATH = ReconstructionPaths().baselines_dir / "prompt_baseline_summary.json"
DEFAULT_REPORT_PATH = ReconstructionPaths().baselines_dir / "prompt_baseline_report.md"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
DEFAULT_SEMANTIC_PROMPT_PATH = (
    ReconstructionPaths().project_root / "prompts" / "semantic_extraction_v1.txt"
)
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")


def _load_openai_client() -> type[Any]:
    """Load the OpenAI client class with a Phase 4-specific error message."""
    try:
        openai_module = importlib.import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional dependency `openai`. "
            "Install local Phase 4 dependencies with "
            "`.venv/bin/python -m pip install -r requirements-dev.txt`."
        ) from exc
    return openai_module.OpenAI


@dataclass(frozen=True)
class CandidateMeasurements:
    """Measured candidate vectors aligned to the Phase 2 baselines."""

    stylometric: np.ndarray
    semantic: np.ndarray


@dataclass(frozen=True)
class PromptTemplate:
    """Versioned prompt template used for one baseline generation step."""

    template_id: str
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class PromptRequest:
    """Concrete prompt request sent to the generation backend."""

    case_id: str
    control_mode: str
    iteration_index: int
    template_id: str
    source_window_id: str
    target_envelope_id: str
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "case_id": self.case_id,
            "control_mode": self.control_mode,
            "iteration_index": self.iteration_index,
            "template_id": self.template_id,
            "source_window_id": self.source_window_id,
            "target_envelope_id": self.target_envelope_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class IterationRecord:
    """Traceable generation + scoring record for one iteration."""

    iteration_index: int
    template_id: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_text: str
    score: Any
    score_history: dict[str, float | bool]
    accepted_as_best: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "iteration_index": self.iteration_index,
            "template_id": self.template_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response,
            "parsed_text": self.parsed_text,
            "score": self.score.to_dict(),
            "score_history": self.score_history,
            "accepted_as_best": self.accepted_as_best,
        }


@dataclass(frozen=True)
class BaselineCase:
    """Locked Phase 4 evaluation case built from the Phase 3 pilot."""

    case_id: str
    control_mode: str
    source_window: WindowRecord
    target_envelope: TargetEnvelope
    uses_training_examples: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "case_id": self.case_id,
            "control_mode": self.control_mode,
            "source_window": self.source_window.to_dict(),
            "target_envelope": self.target_envelope.to_dict(),
            "uses_training_examples": self.uses_training_examples,
        }


@dataclass(frozen=True)
class BaselineCaseResult:
    """Complete prompt-baseline history for one locked case."""

    case: BaselineCase
    prompt_template_id: str
    iterations: tuple[IterationRecord, ...]
    best_iteration_index: int
    stop_reason: str
    used_training_examples: bool

    @property
    def final_iteration(self) -> IterationRecord:
        """Return the best accepted iteration for the case."""
        return self.iterations[self.best_iteration_index]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "case": self.case.to_dict(),
            "prompt_template_id": self.prompt_template_id,
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "best_iteration_index": self.best_iteration_index,
            "stop_reason": self.stop_reason,
            "used_training_examples": self.used_training_examples,
        }


class PromptBackend(Protocol):
    """Prompt-generation backend used by Phase 4."""

    def generate(self, request: PromptRequest) -> str:
        """Generate one raw response for the given request."""


class MeasurementBackend(Protocol):
    """Measurement backend used to score a candidate rewrite."""

    def measure(
        self,
        *,
        source_window: WindowRecord,
        target_envelope: TargetEnvelope,
        candidate_text: str,
        control_mode: str,
    ) -> CandidateMeasurements:
        """Return candidate vectors aligned to the Phase 2 baselines."""


class OpenAIPromptBackend:
    """OpenAI-compatible prompt backend for local vLLM services."""

    def __init__(
        self,
        *,
        api_base: str = VLLM_API_BASE,
        model: str = DEFAULT_MODEL_NAME,
        temperature: float = 0.3,
        max_tokens: int = 768,
    ) -> None:
        openai_client = _load_openai_client()
        self._client = openai_client(base_url=api_base, api_key="not-needed")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, request: PromptRequest) -> str:
        """Generate one raw response from the configured chat model."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""


class DryRunPromptBackend:
    """Deterministic no-network backend for contract-only dry runs."""

    def generate(self, request: PromptRequest) -> str:
        """Return a deterministic text variant for one prompt request."""
        source_text = str(request.metadata["source_text"])
        target_author = str(request.metadata["target_author"])
        if request.iteration_index == 0:
            return f"{source_text}\n\n[reframed toward {target_author}]"
        return f"{source_text}\n\n[revised toward {target_author}]"


class HeuristicMeasurementBackend:
    """Deterministic dry-run measurement backend for prompt-contract checks."""

    def measure(
        self,
        *,
        source_window: WindowRecord,
        target_envelope: TargetEnvelope,
        candidate_text: str,
        control_mode: str,
    ) -> CandidateMeasurements:
        """Approximate candidate vectors from lexical drift for dry-run checks."""
        lexical = reconstruction_lexical_controls(source_window.text, candidate_text)
        divergence = max(0.0, 1.0 - lexical.normalized_edit_similarity)
        if control_mode == "identity":
            style_factor = 0.0
        elif control_mode == "copy_source":
            style_factor = 0.0
        else:
            style_factor = min(1.0, divergence + 0.1)

        style_names = list(source_window.stylometric_reference)
        semantic_names = list(source_window.semantic_reference)
        source_style = np.array(
            [source_window.stylometric_reference[name] for name in style_names],
            dtype=float,
        )
        target_style = np.array(
            [target_envelope.stylometric_target[name] for name in style_names],
            dtype=float,
        )
        source_semantic = np.array(
            [source_window.semantic_reference[name] for name in semantic_names],
            dtype=float,
        )
        semantic_noise = np.full_like(source_semantic, divergence * 0.25)
        return CandidateMeasurements(
            stylometric=source_style + ((target_style - source_style) * style_factor),
            semantic=source_semantic + semantic_noise,
        )


class CorpusMeasurementBackend:
    """Real text measurement backend using local stylometrics plus semantic extraction."""

    def __init__(
        self,
        *,
        api_base: str = VLLM_API_BASE,
        model: str = DEFAULT_MODEL_NAME,
        semantic_prompt_path: Path = DEFAULT_SEMANTIC_PROMPT_PATH,
    ) -> None:
        openai_client = _load_openai_client()
        self._model = model
        self._client = openai_client(base_url=api_base, api_key="not-needed")
        self._semantic_prompt = semantic_prompt_path.read_text(encoding="utf-8")
        self._nlp: Any | None = None

    def _load_nlp(self) -> Any:
        """Load the spaCy model lazily for candidate stylometrics."""
        if self._nlp is None:
            spacy_module = importlib.import_module("spacy")
            self._nlp = spacy_module.load("es_core_news_lg")
            self._nlp.max_length = 2_000_000
        return self._nlp

    def _measure_stylometric(self, text: str) -> np.ndarray:
        """Return a stylometric vector in the Phase 2 feature order."""
        stylometrics_module = importlib.import_module("stylometrics")
        feature_spec = stylometrics_module.FEATURE_SPEC
        extract_basic = stylometrics_module.extract_basic_features
        extract_syntactic = stylometrics_module.extract_syntactic_features

        basic = extract_basic(text)
        doc = self._load_nlp()(text)
        syntactic = extract_syntactic(doc)
        combined = {**basic, **syntactic}
        return np.array([combined[name] for name, _ in feature_spec], dtype=float)

    def _measure_semantic(self, text: str) -> np.ndarray:
        """Return a semantic vector aligned to the Phase 2 kept dimensions."""
        semantic_module = importlib.import_module("semantic_extraction")
        dimensions = semantic_module.DIMENSIONS
        extract_chapter = semantic_module.extract_chapter

        payload = {
            "number": 0,
            "section": "rewrite",
            "text": text,
        }
        scores = extract_chapter(
            self._client,
            self._semantic_prompt,
            payload,
            self._model,
            with_evidence=False,
        )
        if scores is None:
            raise RuntimeError("semantic extraction failed for candidate rewrite")
        kept = [dimension for dimension in dimensions if dimension != "temporal_clarity"]
        return np.array([float(scores[dimension]) for dimension in kept], dtype=float)

    def measure(
        self,
        *,
        source_window: WindowRecord,
        target_envelope: TargetEnvelope,
        candidate_text: str,
        control_mode: str,
    ) -> CandidateMeasurements:
        """Measure one candidate rewrite against the live backend."""
        del source_window, target_envelope, control_mode
        return CandidateMeasurements(
            stylometric=self._measure_stylometric(candidate_text),
            semantic=self._measure_semantic(candidate_text),
        )


def reconstruction_lexical_controls(source_text: str, candidate_text: str) -> Any:
    """Proxy helper that keeps dry-run scoring aligned with Phase 2 controls."""
    from reconstruction_metrics import compute_lexical_controls

    return compute_lexical_controls(source_text, candidate_text)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    from reconstruction_metrics import utc_now as metrics_utc_now

    return metrics_utc_now()


def clamp01(value: float) -> float:
    """Clamp a floating-point value into the unit interval."""
    return min(1.0, max(0.0, value))


def default_prompt_templates() -> dict[str, PromptTemplate]:
    """Return the versioned Phase 4 prompt templates."""
    return {
        "style_shift": PromptTemplate(
            template_id="style_shift_v1",
            system_prompt=(
                "You rewrite literary prose in Spanish. Preserve the scene, narrative facts, "
                "and semantic content, but move the prose toward the requested stylistic "
                "envelope. Do not explain your choices. Output only the rewritten passage."
            ),
            user_prompt=(
                "Source passage:\n{source_text}\n\n"
                "Target work: {target_title} by {target_author}\n"
                "Target style cues:\n{style_summary}\n\n"
                "Rewrite the source passage in Spanish so that it preserves content while "
                "moving toward the target style envelope. Keep the output close in length."
            ),
        ),
        "identity": PromptTemplate(
            template_id="identity_v1",
            system_prompt=(
                "You rewrite literary prose conservatively in Spanish. Preserve content, tone, "
                "and style as much as possible. Output only the rewritten passage."
            ),
            user_prompt=(
                "Source passage:\n{source_text}\n\n"
                "Produce a minimally changed rewrite that preserves the original style and length."
            ),
        ),
        "paraphrase": PromptTemplate(
            template_id="paraphrase_v1",
            system_prompt=(
                "You paraphrase literary prose in Spanish. Preserve narrative content while "
                "changing surface wording. Output only the rewritten passage."
            ),
            user_prompt=(
                "Source passage:\n{source_text}\n\n"
                "Paraphrase the passage in Spanish while preserving the semantic content."
            ),
        ),
        "revise": PromptTemplate(
            template_id="revise_v1",
            system_prompt=(
                "You revise a prior rewrite in Spanish. Preserve semantic content, improve the "
                "target style fit, and respect the requested length guardrails. Output only the "
                "revised passage."
            ),
            user_prompt=(
                "Original source passage:\n{source_text}\n\n"
                "Current rewrite:\n{candidate_text}\n\n"
                "Target work: {target_title} by {target_author}\n"
                "Target style cues:\n{style_summary}\n\n"
                "Current score feedback:\n{feedback}\n\n"
                "Revise the current rewrite to improve the weighted objective while preserving "
                "the scene and staying close in length."
            ),
        ),
    }


def _source_style_vector(window: WindowRecord, baseline: MeasurementBaseline) -> np.ndarray:
    """Return the source stylometric vector in baseline order."""
    return np.array(
        [window.stylometric_reference[name] for name in baseline.dimension_order],
        dtype=float,
    )


def _source_semantic_vector(window: WindowRecord, baseline: MeasurementBaseline) -> np.ndarray:
    """Return the source semantic vector in baseline order."""
    return np.array(
        [window.semantic_reference[name] for name in baseline.dimension_order],
        dtype=float,
    )


def _target_style_vector(envelope: TargetEnvelope, baseline: MeasurementBaseline) -> np.ndarray:
    """Return the target stylometric vector in baseline order."""
    return np.array(
        [envelope.stylometric_target[name] for name in baseline.dimension_order],
        dtype=float,
    )


def _style_summary(case: BaselineCase) -> str:
    """Return compact target-style guidance for prompting."""
    deltas: list[tuple[float, str, float, float]] = []
    for name, target_value in case.target_envelope.stylometric_target.items():
        source_value = case.source_window.stylometric_reference.get(name)
        if source_value is None:
            continue
        deltas.append((abs(target_value - source_value), name, source_value, target_value))
    top_deltas = sorted(deltas, reverse=True)[:5]
    if not top_deltas:
        return "- preserve the operationally decoupled style profile"
    return "\n".join(
        f"- {name}: source={source_value:.3f}, target={target_value:.3f}"
        for _, name, source_value, target_value in top_deltas
    )


def parse_generated_text(raw_response: str) -> str:
    """Normalize raw model output into the candidate text scored downstream."""
    text = raw_response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text.strip()


def _feedback_from_score(score_history: dict[str, float | bool]) -> str:
    """Serialize compact revision feedback from one score history snapshot."""
    return (
        f"- weighted objective: {float(score_history['weighted_objective']):.4f}\n"
        f"- semantic pass: {bool(score_history['semantic_tolerance_pass'])}\n"
        f"- target pass: {bool(score_history['target_tolerance_pass'])}\n"
        f"- length pass: {bool(score_history['length_guardrail_pass'])}\n"
        f"- lexical pass: {bool(score_history['lexical_overlap_pass'])}\n"
    )


def build_prompt_request(
    *,
    case: BaselineCase,
    template: PromptTemplate,
    iteration_index: int,
    candidate_text: str | None = None,
    prior_score_history: dict[str, float | bool] | None = None,
) -> PromptRequest:
    """Render one prompt request from a case and template."""
    style_summary = _style_summary(case)
    metadata = {
        "source_text": case.source_window.text,
        "target_author": case.target_envelope.author,
        "target_title": case.target_envelope.title,
        "source_window_id": case.source_window.window_id,
        "target_envelope_id": case.target_envelope.envelope_id,
        "uses_training_examples": case.uses_training_examples,
    }
    user_prompt = template.user_prompt.format(
        source_text=case.source_window.text,
        target_author=case.target_envelope.author,
        target_title=case.target_envelope.title,
        style_summary=style_summary,
        candidate_text=candidate_text or case.source_window.text,
        feedback=(
            "no prior score"
            if prior_score_history is None
            else _feedback_from_score(prior_score_history)
        ),
    )
    return PromptRequest(
        case_id=case.case_id,
        control_mode=case.control_mode,
        iteration_index=iteration_index,
        template_id=template.template_id,
        source_window_id=case.source_window.window_id,
        target_envelope_id=case.target_envelope.envelope_id,
        system_prompt=template.system_prompt,
        user_prompt=user_prompt,
        metadata=metadata,
    )


def build_score_history(
    score: Any,
    *,
    success_criteria: SuccessCriteria,
) -> dict[str, float | bool]:
    """Return weighted-objective bookkeeping for one scored candidate."""
    lexical_controls = score.lexical_controls
    if lexical_controls is None:
        raise ValueError("lexical controls are required for Phase 4 scoring")

    semantic_component = clamp01(
        1.0
        - (
            score.semantic_source_distance
            / success_criteria.tolerances["semantic_preservation_max"]
        )
    )
    target_component = clamp01(
        1.0
        - (score.stylistic_target_distance / success_criteria.tolerances["stylistic_target_max"])
    )

    length_min = success_criteria.lexical_guardrails["length_ratio_min"]
    length_max = success_criteria.lexical_guardrails["length_ratio_max"]
    length_ratio = lexical_controls.length_ratio
    length_guardrail_pass = length_min <= length_ratio <= length_max
    if length_guardrail_pass:
        length_component = 1.0
    elif length_ratio < length_min:
        length_component = clamp01(length_ratio / length_min)
    else:
        length_component = clamp01(length_max / length_ratio)

    lexical_overlap_pass = (
        lexical_controls.token_jaccard
        >= success_criteria.lexical_guardrails["token_jaccard_min"]
        and lexical_controls.normalized_edit_similarity
        >= success_criteria.lexical_guardrails["normalized_edit_similarity_min"]
    )
    lexical_component = (
        clamp01(1.0 - lexical_controls.normalized_edit_similarity)
        if lexical_overlap_pass
        else 0.0
    )

    weights = success_criteria.objective_weights
    weighted_objective = (
        (weights["semantic_preservation"] * semantic_component)
        + (weights["stylistic_target"] * target_component)
        + (weights["length_guardrail"] * length_component)
        + (weights["lexical_divergence"] * lexical_component)
    )
    return {
        "semantic_component": semantic_component,
        "target_component": target_component,
        "length_component": length_component,
        "lexical_divergence_component": lexical_component,
        "semantic_tolerance_pass": score.within_semantic_tolerance,
        "stylistic_tolerance_pass": score.within_stylistic_tolerance,
        "target_tolerance_pass": score.within_target_tolerance,
        "length_guardrail_pass": length_guardrail_pass,
        "lexical_overlap_pass": lexical_overlap_pass,
        "weighted_objective": weighted_objective,
    }


def run_prompt_case(
    *,
    case: BaselineCase,
    prompt_backend: PromptBackend,
    measurement_backend: MeasurementBackend,
    stylometric_baseline: MeasurementBaseline,
    semantic_baseline: MeasurementBaseline,
    success_criteria: SuccessCriteria,
    max_iterations: int = 2,
    tolerance_config: ToleranceConfig | None = None,
) -> BaselineCaseResult:
    """Run one prompt baseline case with optional revision iterations."""
    if case.uses_training_examples:
        raise ValueError("Phase 4 prompt baselines must not use training examples")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    templates = default_prompt_templates()
    control_key = case.control_mode if case.control_mode in templates else "style_shift"
    initial_template = templates[control_key]
    revise_template = templates["revise"]
    tolerance = tolerance_config or ToleranceConfig(
        semantic_preservation_max=success_criteria.tolerances["semantic_preservation_max"],
        stylistic_preservation_max=success_criteria.tolerances["stylistic_preservation_max"],
        stylistic_target_max=success_criteria.tolerances["stylistic_target_max"],
    )

    source_style = _source_style_vector(case.source_window, stylometric_baseline)
    target_style = _target_style_vector(case.target_envelope, stylometric_baseline)
    source_semantic = _source_semantic_vector(case.source_window, semantic_baseline)

    iterations: list[IterationRecord] = []
    best_iteration_index = 0
    best_objective = float("-inf")
    stop_reason = "max_iterations_reached"

    for iteration_index in range(max_iterations):
        template = initial_template if iteration_index == 0 else revise_template
        prior_history = None
        prior_candidate = None
        if iterations:
            prior_history = iterations[best_iteration_index].score_history
            prior_candidate = iterations[best_iteration_index].parsed_text

        if case.control_mode == "copy_source":
            request = build_prompt_request(
                case=case,
                template=initial_template,
                iteration_index=iteration_index,
            )
            raw_response = case.source_window.text
        else:
            request = build_prompt_request(
                case=case,
                template=template,
                iteration_index=iteration_index,
                candidate_text=prior_candidate,
                prior_score_history=prior_history,
            )
            raw_response = prompt_backend.generate(request)

        parsed_text = parse_generated_text(raw_response) or case.source_window.text
        candidate = measurement_backend.measure(
            source_window=case.source_window,
            target_envelope=case.target_envelope,
            candidate_text=parsed_text,
            control_mode=case.control_mode,
        )
        score = score_rewrite(
            source_stylometric=source_style,
            candidate_stylometric=candidate.stylometric,
            target_stylometric=target_style,
            source_semantic=source_semantic,
            candidate_semantic=candidate.semantic,
            stylometric_baseline=stylometric_baseline,
            semantic_baseline=semantic_baseline,
            tolerances=tolerance,
            source_text=case.source_window.text,
            candidate_text=parsed_text,
        )
        score_history = build_score_history(score, success_criteria=success_criteria)
        weighted_objective = float(score_history["weighted_objective"])
        accepted_as_best = weighted_objective > (best_objective + 1e-9)
        if accepted_as_best:
            best_objective = weighted_objective
            best_iteration_index = iteration_index

        iterations.append(
            IterationRecord(
                iteration_index=iteration_index,
                template_id=request.template_id,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                raw_response=raw_response,
                parsed_text=parsed_text,
                score=score,
                score_history=score_history,
                accepted_as_best=accepted_as_best,
            )
        )

        if iteration_index > 0 and not accepted_as_best:
            stop_reason = "no_objective_improvement"
            break

    return BaselineCaseResult(
        case=case,
        prompt_template_id=initial_template.template_id,
        iterations=tuple(iterations),
        best_iteration_index=best_iteration_index,
        stop_reason=stop_reason,
        used_training_examples=case.uses_training_examples,
    )


def _summary_by_control(results: list[BaselineCaseResult]) -> dict[str, dict[str, Any]]:
    """Summarize case results by control mode."""
    grouped: dict[str, list[BaselineCaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case.control_mode, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    for control_mode, items in grouped.items():
        objectives = [
            float(item.final_iteration.score_history["weighted_objective"])
            for item in items
        ]
        summary[control_mode] = {
            "count": len(items),
            "mean_weighted_objective": float(np.mean(objectives)),
            "median_weighted_objective": float(np.median(objectives)),
            "stop_reasons": sorted({item.stop_reason for item in items}),
        }
    return summary


def write_baseline_artifacts(
    *,
    results: list[BaselineCaseResult],
    cases_path: Path,
    summary_path: Path,
    report_path: Path,
) -> tuple[Path, Path, Path]:
    """Persist full case histories plus compact summary artifacts."""
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cases_payload = {
        "generated_at": utc_now(),
        "results": [result.to_dict() for result in results],
    }
    summary_payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "total_cases": len(results),
        "controls": _summary_by_control(results),
    }
    report_lines = [
        "# Phase 4 Prompt Baseline Report",
        "",
        f"Generated at: {summary_payload['generated_at']}",
        f"Total cases: {summary_payload['total_cases']}",
        "",
        "## Control Summary",
        "",
    ]
    for control_mode, stats in summary_payload["controls"].items():
        report_lines.append(
            f"- `{control_mode}`: count={stats['count']}, "
            f"mean weighted objective={stats['mean_weighted_objective']:.4f}, "
            f"median weighted objective={stats['median_weighted_objective']:.4f}, "
            f"stop reasons={', '.join(stats['stop_reasons'])}"
        )

    cases_path.write_text(
        json.dumps(cases_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return cases_path, summary_path, report_path


def load_source_windows(path: Path = DEFAULT_SOURCE_WINDOWS_PATH) -> list[WindowRecord]:
    """Load Phase 3 source windows from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [WindowRecord(**record) for record in payload["source_windows"]]


def load_target_envelopes(path: Path = DEFAULT_TARGET_ENVELOPES_PATH) -> list[TargetEnvelope]:
    """Load Phase 3 target envelopes from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TargetEnvelope(**record) for record in payload["target_envelopes"]]


def load_success_criteria(path: Path = DEFAULT_SUCCESS_CRITERIA_PATH) -> SuccessCriteria:
    """Load Phase 3 success criteria from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SuccessCriteria(**payload)


def build_style_shift_cases(
    source_windows: list[WindowRecord],
    target_envelopes: list[TargetEnvelope],
    *,
    max_cases: int | None = None,
) -> list[BaselineCase]:
    """Build deterministic style-shift cases from the locked Phase 3 pilot."""
    cases: list[BaselineCase] = []
    for source_window in source_windows:
        for target_envelope in target_envelopes:
            if source_window.work_id == target_envelope.work_id:
                continue
            cases.append(
                BaselineCase(
                    case_id=(
                        f"style_shift:{source_window.window_id}:"
                        f"{target_envelope.envelope_id}"
                    ),
                    control_mode="style_shift",
                    source_window=source_window,
                    target_envelope=target_envelope,
                )
            )
    if max_cases is not None:
        return cases[:max_cases]
    return cases


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Phase 4 prompt-baseline runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique immutable run identifier.")
    parser.add_argument(
        "--phase",
        default="phase-4-prompt-baselines",
        help="Phase label recorded in the run manifest.",
    )
    parser.add_argument("--source-windows-path", type=Path, default=DEFAULT_SOURCE_WINDOWS_PATH)
    parser.add_argument("--target-envelopes-path", type=Path, default=DEFAULT_TARGET_ENVELOPES_PATH)
    parser.add_argument("--success-criteria-path", type=Path, default=DEFAULT_SUCCESS_CRITERIA_PATH)
    parser.add_argument("--split-manifest-path", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_RECONSTRUCTION_SEED)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--api-base", default=VLLM_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Phase 4 prompt baselines against the locked Phase 3 pilot."""
    args = build_argument_parser().parse_args(argv)
    seed_everything(args.seed)
    paths = ReconstructionPaths()
    run_dir = prepare_run_directory(args.run_id, paths=paths)

    config_payload = {
        "phase": args.phase,
        "seed": args.seed,
        "source_windows_path": to_project_relative(args.source_windows_path, paths.project_root),
        "target_envelopes_path": to_project_relative(
            args.target_envelopes_path, paths.project_root
        ),
        "success_criteria_path": to_project_relative(
            args.success_criteria_path, paths.project_root
        ),
        "split_manifest_path": to_project_relative(args.split_manifest_path, paths.project_root),
        "max_cases": args.max_cases,
        "max_iterations": args.max_iterations,
        "dry_run": args.dry_run,
        "api_base": args.api_base,
        "model": args.model,
    }
    manifest = build_run_manifest(
        run_id=args.run_id,
        phase=args.phase,
        model_id="dry-run-prompt-baseline" if args.dry_run else args.model,
        seed=args.seed,
        git_sha=detect_git_sha(paths.project_root),
        config_payload=config_payload,
        corpus_manifest="outputs/corpus/corpus_metadata.json",
        prompt_template_id="style_shift_v1",
        split_manifest=args.split_manifest_path,
        paths=paths,
    )
    write_run_manifest(manifest, paths=paths)

    try:
        source_windows = load_source_windows(args.source_windows_path)
        target_envelopes = load_target_envelopes(args.target_envelopes_path)
        success_criteria = load_success_criteria(args.success_criteria_path)
        baselines = build_measurement_baselines(
            corpus_dir=CORPUS_DIR,
            corpus_output_dir=CORPUS_OUTPUT_DIR,
        )
        cases = build_style_shift_cases(
            source_windows,
            target_envelopes,
            max_cases=args.max_cases,
        )
        prompt_backend: PromptBackend
        measurement_backend: MeasurementBackend
        if args.dry_run:
            prompt_backend = DryRunPromptBackend()
            measurement_backend = HeuristicMeasurementBackend()
        else:
            prompt_backend = OpenAIPromptBackend(
                api_base=args.api_base,
                model=args.model,
            )
            measurement_backend = CorpusMeasurementBackend(
                api_base=args.api_base,
                model=args.model,
            )

        results = [
            run_prompt_case(
                case=case,
                prompt_backend=prompt_backend,
                measurement_backend=measurement_backend,
                stylometric_baseline=baselines.stylometric,
                semantic_baseline=baselines.semantic,
                success_criteria=success_criteria,
                max_iterations=args.max_iterations,
            )
            for case in cases
        ]

        cases_path = run_dir / "prompt_baseline_cases.json"
        summary_path = run_dir / "prompt_baseline_summary.json"
        report_path = run_dir / "prompt_baseline_report.md"
        write_baseline_artifacts(
            results=results,
            cases_path=cases_path,
            summary_path=summary_path,
            report_path=report_path,
        )
        finalize_run_manifest(args.run_id, RunStatus.COMPLETED, paths=paths)
    except Exception as exc:
        finalize_run_manifest(
            args.run_id,
            RunStatus.FAILED,
            paths=paths,
            error_message=str(exc),
        )
        raise

    print("Phase 4 prompt baselines")
    print(f"  Run dir:      {run_dir}")
    print(f"  Cases:        {len(results)}")
    print(f"  Cases path:   {cases_path}")
    print(f"  Summary path: {summary_path}")
    print(f"  Report path:  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
