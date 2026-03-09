#!/usr/bin/env python3
"""
Measurement contract, baselines, and control scoring for Part 3 reconstruction.

Phase 2 locks the evaluation layer before any generation begins. The module
loads the existing corpus-aligned stylometric and semantic outputs, computes
dimension-level baselines, and exposes deterministic scoring for later rewrite
experiments under the language of operational decoupling.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import numpy as np

from project_config import (
    CORPUS_DIR,
    CORPUS_OUTPUT_DIR,
    CORPUS_WORKS,
    DIM_GROUPS,
    DIM_LABELS,
    DIMS_EXCLUDED,
    PROJECT_ROOT,
)
from reconstruction_contract import DEFAULT_RECONSTRUCTION_SEED, ReconstructionPaths

MetricKind = Literal["stylometric", "semantic"]

DEFAULT_STYLOMETRIC_BASELINE_PATH = (
    ReconstructionPaths().baselines_dir / "stylometric_baseline.json"
)
DEFAULT_SEMANTIC_BASELINE_PATH = ReconstructionPaths().baselines_dir / "semantic_baseline.json"
DEFAULT_CONTROL_DIAGNOSTICS_PATH = ReconstructionPaths().baselines_dir / "control_diagnostics.json"

STYLOMETRIC_GROUPS = {
    "sent_len_mean": "Syntax",
    "sent_len_median": "Syntax",
    "sent_len_std": "Syntax",
    "sent_len_max": "Syntax",
    "sent_len_cv": "Syntax",
    "mattr": "Lexical diversity",
    "hapax_ratio": "Lexical diversity",
    "vocab_density": "Lexical diversity",
    "articles_per_k": "Function words",
    "prepositions_per_k": "Function words",
    "conjunctions_per_k": "Function words",
    "pronouns_per_k": "Function words",
    "semicolons_per_k": "Punctuation",
    "colons_per_k": "Punctuation",
    "em_dashes_per_k": "Punctuation",
    "ellipses_per_k": "Punctuation",
    "exclamations_per_k": "Punctuation",
    "questions_per_k": "Punctuation",
    "parens_per_k": "Punctuation",
    "parse_depth_mean": "Parsing",
    "subordinate_ratio": "Parsing",
    "french_per_k": "Multilingual markers",
    "english_per_k": "Multilingual markers",
    "word_len_mean": "Word shape",
    "syllable_mean": "Word shape",
    "para_len_mean": "Paragraph structure",
}

SEMANTIC_DESCRIPTION_MAP = {
    "existential_questioning": "Degree of philosophical or existential inquiry.",
    "art_and_aesthetics": "Salience of art, literature, or aesthetic reflection.",
    "everyday_mundanity": "How strongly the passage is grounded in ordinary daily life.",
    "death_and_mortality": "Prominence of death, endings, or mortality themes.",
    "love_and_desire": "Strength of romantic, erotic, or desire-driven content.",
    "emotional_intensity": "Overall intensity of affective charge in the passage.",
    "humor_and_irony": "Presence of humor, wit, satire, or ironic framing.",
    "melancholy_and_nostalgia": "Weight of sadness, longing, or retrospective loss.",
    "tension_and_anxiety": "Level of tension, unease, suspense, or anxiety.",
    "oliveira_centrality": (
        "How strongly the focal consciousness is centered on Oliveira-like interiority."
    ),
    "la_maga_presence": "Presence of a desire figure analogous to La Maga.",
    "character_density": "Density of active characters or social presence.",
    "interpersonal_conflict": "Degree of active interpersonal friction or confrontation.",
    "interiority": "Depth of interior reflection or inward narration.",
    "dialogue_density": "Amount of dialogue relative to narration.",
    "metafiction": "Self-awareness about narrative, language, or literary construction.",
    "temporal_clarity": "Clarity of temporal progression in the scene.",
    "spatial_grounding": "Specificity and stability of the spatial setting.",
    "language_experimentation": "Extent of formal or linguistic experimentation.",
    "intertextual_density": "Density of allusion, quotation, or intertextual reference.",
}

SEMANTIC_GROUPS = {
    dimension: group_name
    for group_name, dimensions in DIM_GROUPS.items()
    for dimension in dimensions
}


@dataclass(frozen=True)
class DimensionMetadata:
    """Human-readable metadata for one measurable dimension."""

    kind: MetricKind
    name: str
    label: str
    description: str
    group: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable metadata payload."""
        return {
            "kind": self.kind,
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "group": self.group,
        }


@dataclass(frozen=True)
class BaselineStats:
    """Univariate summary statistics for one dimension."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable stats payload."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
        }


@dataclass(frozen=True)
class DimensionBaseline:
    """Baseline metadata + summary stats for one dimension."""

    metadata: DimensionMetadata
    stats: BaselineStats

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "metadata": self.metadata.to_dict(),
            "stats": self.stats.to_dict(),
        }


@dataclass(frozen=True)
class MeasurementMatrix:
    """Aligned corpus measurement matrix for one scale."""

    kind: MetricKind
    dimension_order: tuple[str, ...]
    matrix: np.ndarray
    segment_ids: tuple[str, ...]
    source_paths: tuple[str, ...]
    dimension_registry: dict[str, DimensionMetadata]

    @property
    def chapter_count(self) -> int:
        """Return the row count in the aligned measurement matrix."""
        return int(self.matrix.shape[0])


@dataclass(frozen=True)
class MeasurementBaseline:
    """Serializable baseline report for one measurement space."""

    kind: MetricKind
    generated_at: str
    chapter_count: int
    dimension_order: tuple[str, ...]
    dimensions: dict[str, DimensionBaseline]
    source_paths: tuple[str, ...]

    def std_vector(self) -> np.ndarray:
        """Return per-dimension standard deviations in report order."""
        return np.array(
            [max(self.dimensions[name].stats.std, 1e-8) for name in self.dimension_order],
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": self.kind,
            "generated_at": self.generated_at,
            "chapter_count": self.chapter_count,
            "dimension_order": list(self.dimension_order),
            "dimension_count": len(self.dimension_order),
            "source_paths": list(self.source_paths),
            "dimensions": {name: self.dimensions[name].to_dict() for name in self.dimension_order},
        }


@dataclass(frozen=True)
class MeasurementBaselineBundle:
    """Phase 2 baseline payload across stylometric and semantic spaces."""

    stylometric: MeasurementBaseline
    semantic: MeasurementBaseline


@dataclass(frozen=True)
class ToleranceConfig:
    """Tolerance thresholds used by rewrite scoring and controls."""

    semantic_preservation_max: float = 0.75
    stylistic_preservation_max: float = 0.75
    stylistic_target_max: float = 0.75


@dataclass(frozen=True)
class LexicalControls:
    """Simple lexical guardrails stored alongside vector-space scores."""

    source_token_count: int
    candidate_token_count: int
    length_ratio: float
    token_jaccard: float
    normalized_edit_similarity: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""
        return {
            "source_token_count": self.source_token_count,
            "candidate_token_count": self.candidate_token_count,
            "length_ratio": self.length_ratio,
            "token_jaccard": self.token_jaccard,
            "normalized_edit_similarity": self.normalized_edit_similarity,
        }


@dataclass(frozen=True)
class ReconstructionScore:
    """Deterministic score bundle for one candidate rewrite."""

    semantic_source_distance: float
    stylistic_source_distance: float
    stylistic_target_distance: float
    stylistic_target_improvement: float
    stylistic_target_improvement_ratio: float
    within_semantic_tolerance: bool
    within_stylistic_tolerance: bool
    within_target_tolerance: bool
    lexical_controls: LexicalControls | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "semantic_source_distance": self.semantic_source_distance,
            "stylistic_source_distance": self.stylistic_source_distance,
            "stylistic_target_distance": self.stylistic_target_distance,
            "stylistic_target_improvement": self.stylistic_target_improvement,
            "stylistic_target_improvement_ratio": self.stylistic_target_improvement_ratio,
            "within_semantic_tolerance": self.within_semantic_tolerance,
            "within_stylistic_tolerance": self.within_stylistic_tolerance,
            "within_target_tolerance": self.within_target_tolerance,
            "lexical_controls": (
                None if self.lexical_controls is None else self.lexical_controls.to_dict()
            ),
        }


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def canonical_segment_id(work_id: str, chapter_number: int) -> str:
    """Return the canonical segment identifier used across Phase 1 and Phase 2."""
    return f"{work_id}:{chapter_number}"


def project_relative_path(path: Path) -> str:
    """Return a repository-relative path for manifest-style serialization."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return resolved.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON payload from disk."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_clean_chapter_numbers(corpus_dir: Path, work_id: str) -> list[int]:
    """Load canonical chapter numbers from one cleaned corpus work."""
    payload = _load_json(corpus_dir / f"{work_id}_clean.json")
    return [int(chapter["number"]) for chapter in payload["chapters"]]


def _default_label(kind: MetricKind, name: str) -> str:
    """Generate a reasonable display label when no explicit label exists."""
    if kind == "semantic":
        return DIM_LABELS.get(name, name.replace("_", " ").title())
    return name.replace("_", " ").title()


def _default_description(kind: MetricKind, name: str) -> str:
    """Generate a human-readable description when upstream metadata is absent."""
    if kind == "semantic":
        return SEMANTIC_DESCRIPTION_MAP.get(
            name,
            f"Semantic score for {name.replace('_', ' ')}.",
        )
    return f"Stylometric measurement for {name.replace('_', ' ')}."


def _default_group(kind: MetricKind, name: str) -> str:
    """Return the default grouping label for a dimension."""
    if kind == "semantic":
        return SEMANTIC_GROUPS.get(name, "Semantic")
    return STYLOMETRIC_GROUPS.get(name, "Stylometric")


def build_dimension_registry(
    *,
    kind: MetricKind,
    names: list[str],
    descriptions: dict[str, str] | None = None,
) -> dict[str, DimensionMetadata]:
    """Build a consistent metadata registry for one measurement space."""
    registry: dict[str, DimensionMetadata] = {}
    for name in names:
        registry[name] = DimensionMetadata(
            kind=kind,
            name=name,
            label=_default_label(kind, name),
            description=(descriptions or {}).get(name, _default_description(kind, name)),
            group=_default_group(kind, name),
        )
    return registry


def _semantic_dimension_order(raw_dimensions: list[str]) -> tuple[list[str], list[int]]:
    """Filter excluded semantic dimensions while preserving column order."""
    kept_dimensions: list[str] = []
    kept_indices: list[int] = []
    for index, dimension in enumerate(raw_dimensions):
        if dimension not in DIMS_EXCLUDED:
            kept_dimensions.append(dimension)
            kept_indices.append(index)
    return kept_dimensions, kept_indices


def load_stylometric_measurements(
    corpus_dir: Path = CORPUS_DIR,
    corpus_output_dir: Path = CORPUS_OUTPUT_DIR,
    corpus_works: dict[str, tuple[str, str]] | None = None,
) -> MeasurementMatrix:
    """Load the aligned stylometric corpus matrix from per-work outputs."""
    resolved_corpus_works = CORPUS_WORKS if corpus_works is None else corpus_works
    all_rows: list[np.ndarray] = []
    segment_ids: list[str] = []
    source_paths: list[str] = []
    expected_names: tuple[str, ...] | None = None
    registry: dict[str, DimensionMetadata] | None = None

    for work_id in sorted(resolved_corpus_works):
        clean_chapters = _load_clean_chapter_numbers(corpus_dir, work_id)
        work_dir = corpus_output_dir / work_id
        npy_path = work_dir / "chapter_stylometrics.npy"
        metadata_path = work_dir / "chapter_stylometrics_metadata.json"
        matrix = np.load(npy_path)
        metadata = _load_json(metadata_path)

        feature_names = tuple(str(name) for name in metadata["feature_names"])
        feature_descriptions = {
            str(name): str(description)
            for name, description in metadata.get("feature_descriptions", {}).items()
        }
        if matrix.shape[0] != len(clean_chapters):
            raise ValueError(
                f"stylometric row count mismatch for {work_id}: "
                f"{matrix.shape[0]} != {len(clean_chapters)}"
            )
        if int(metadata["n_chapters"]) != len(clean_chapters):
            raise ValueError(
                f"stylometric metadata count mismatch for {work_id}: "
                f"{metadata['n_chapters']} != {len(clean_chapters)}"
            )
        if matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"stylometric dimension count mismatch for {work_id}: "
                f"{matrix.shape[1]} != {len(feature_names)}"
            )

        if expected_names is None:
            expected_names = feature_names
            registry = build_dimension_registry(
                kind="stylometric",
                names=list(feature_names),
                descriptions=feature_descriptions,
            )
        elif feature_names != expected_names:
            raise ValueError(
                f"inconsistent stylometric feature order for {work_id}: "
                f"{feature_names} != {expected_names}"
            )

        all_rows.append(matrix.astype(float))
        segment_ids.extend(
            canonical_segment_id(work_id, chapter_number) for chapter_number in clean_chapters
        )
        source_paths.append(project_relative_path(npy_path))

    if expected_names is None or registry is None or not all_rows:
        raise ValueError("no stylometric corpus outputs found")

    return MeasurementMatrix(
        kind="stylometric",
        dimension_order=expected_names,
        matrix=np.vstack(all_rows),
        segment_ids=tuple(segment_ids),
        source_paths=tuple(source_paths),
        dimension_registry=registry,
    )


def load_semantic_measurements(
    corpus_dir: Path = CORPUS_DIR,
    corpus_output_dir: Path = CORPUS_OUTPUT_DIR,
    corpus_works: dict[str, tuple[str, str]] | None = None,
) -> MeasurementMatrix:
    """Load the aligned semantic corpus matrix from per-work outputs."""
    resolved_corpus_works = CORPUS_WORKS if corpus_works is None else corpus_works
    all_rows: list[np.ndarray] = []
    segment_ids: list[str] = []
    source_paths: list[str] = []
    expected_dimensions: tuple[str, ...] | None = None
    registry: dict[str, DimensionMetadata] | None = None

    for work_id in sorted(resolved_corpus_works):
        clean_chapters = _load_clean_chapter_numbers(corpus_dir, work_id)
        clean_segment_ids = tuple(
            canonical_segment_id(work_id, chapter_number) for chapter_number in clean_chapters
        )

        work_dir = corpus_output_dir / work_id
        json_path = work_dir / "narrative_dna.json"
        npy_path = work_dir / "narrative_dna_vectors.npy"
        payload = _load_json(json_path)
        matrix = np.load(npy_path).astype(float)

        raw_dimensions = [str(name) for name in payload["dimensions"]]
        raw_segment_ids = tuple(
            canonical_segment_id(work_id, int(chapter["chapter"]))
            for chapter in payload["chapters"]
        )
        if raw_segment_ids != clean_segment_ids:
            raise ValueError(
                f"semantic segment alignment mismatch for {work_id}: "
                f"{raw_segment_ids} != {clean_segment_ids}"
            )
        if matrix.shape[0] != len(raw_segment_ids):
            raise ValueError(
                f"semantic row count mismatch for {work_id}: "
                f"{matrix.shape[0]} != {len(raw_segment_ids)}"
            )
        if matrix.shape[1] != len(raw_dimensions):
            raise ValueError(
                f"semantic dimension count mismatch for {work_id}: "
                f"{matrix.shape[1]} != {len(raw_dimensions)}"
            )

        kept_dimensions, kept_indices = _semantic_dimension_order(raw_dimensions)
        filtered_matrix = matrix[:, kept_indices]
        filtered_dimensions = tuple(kept_dimensions)

        if expected_dimensions is None:
            expected_dimensions = filtered_dimensions
            registry = build_dimension_registry(
                kind="semantic",
                names=list(filtered_dimensions),
                descriptions=SEMANTIC_DESCRIPTION_MAP,
            )
        elif filtered_dimensions != expected_dimensions:
            raise ValueError(
                f"inconsistent semantic dimension order for {work_id}: "
                f"{filtered_dimensions} != {expected_dimensions}"
            )

        all_rows.append(filtered_matrix)
        segment_ids.extend(raw_segment_ids)
        source_paths.append(project_relative_path(npy_path))

    if expected_dimensions is None or registry is None or not all_rows:
        raise ValueError("no semantic corpus outputs found")

    return MeasurementMatrix(
        kind="semantic",
        dimension_order=expected_dimensions,
        matrix=np.vstack(all_rows),
        segment_ids=tuple(segment_ids),
        source_paths=tuple(source_paths),
        dimension_registry=registry,
    )


def compute_measurement_baseline(measurements: MeasurementMatrix) -> MeasurementBaseline:
    """Summarize a measurement matrix into per-dimension baseline stats."""
    dimensions: dict[str, DimensionBaseline] = {}

    for index, name in enumerate(measurements.dimension_order):
        column = measurements.matrix[:, index]
        dimensions[name] = DimensionBaseline(
            metadata=measurements.dimension_registry[name],
            stats=BaselineStats(
                count=int(column.shape[0]),
                mean=float(np.mean(column)),
                std=float(np.std(column)),
                min=float(np.min(column)),
                max=float(np.max(column)),
                median=float(np.median(column)),
            ),
        )

    return MeasurementBaseline(
        kind=measurements.kind,
        generated_at=utc_now(),
        chapter_count=measurements.chapter_count,
        dimension_order=measurements.dimension_order,
        dimensions=dimensions,
        source_paths=measurements.source_paths,
    )


def build_measurement_baselines(
    corpus_dir: Path = CORPUS_DIR,
    corpus_output_dir: Path = CORPUS_OUTPUT_DIR,
    corpus_works: dict[str, tuple[str, str]] | None = None,
    *,
    require_clean_audit: bool = True,
) -> MeasurementBaselineBundle:
    """Build stylometric and semantic baselines from corpus-aligned outputs."""
    if require_clean_audit:
        from reconstruction_audit import audit_corpus_outputs

        report = audit_corpus_outputs(
            corpus_dir=corpus_dir,
            corpus_output_dir=corpus_output_dir,
            corpus_works=corpus_works,
            author_profile_paths={},
        )
        if not report.is_clean:
            issues = "; ".join(report.issues)
            raise ValueError(f"Phase 2 requires synchronized corpus outputs: {issues}")

    stylometric_measurements = load_stylometric_measurements(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
    )
    semantic_measurements = load_semantic_measurements(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
    )

    return MeasurementBaselineBundle(
        stylometric=compute_measurement_baseline(stylometric_measurements),
        semantic=compute_measurement_baseline(semantic_measurements),
    )


def _validate_vector_length(
    vector: np.ndarray,
    baseline: MeasurementBaseline,
    *,
    label: str,
) -> np.ndarray:
    """Validate vector dimensionality against the chosen baseline."""
    normalized = np.asarray(vector, dtype=float)
    if normalized.shape != (len(baseline.dimension_order),):
        raise ValueError(
            f"{label} has shape {normalized.shape}; expected ({len(baseline.dimension_order)},)"
        )
    return normalized


def _normalized_distance(
    left: np.ndarray,
    right: np.ndarray,
    baseline: MeasurementBaseline,
) -> float:
    """Compute root-mean-square distance normalized by corpus variance."""
    deltas = (left - right) / baseline.std_vector()
    return float(np.sqrt(np.mean(np.square(deltas))))


def _tokenize(text: str) -> list[str]:
    """Tokenize lightly for lexical overlap controls."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def compute_lexical_controls(source_text: str, candidate_text: str) -> LexicalControls:
    """Compute simple length and overlap guardrails for one rewrite pair."""
    source_tokens = _tokenize(source_text)
    candidate_tokens = _tokenize(candidate_text)
    source_set = set(source_tokens)
    candidate_set = set(candidate_tokens)
    union = source_set | candidate_set
    return LexicalControls(
        source_token_count=len(source_tokens),
        candidate_token_count=len(candidate_tokens),
        length_ratio=(float(len(candidate_tokens) / len(source_tokens)) if source_tokens else 0.0),
        token_jaccard=(float(len(source_set & candidate_set) / len(union)) if union else 1.0),
        normalized_edit_similarity=float(
            SequenceMatcher(None, source_text, candidate_text).ratio()
        ),
    )


def score_rewrite(
    *,
    source_stylometric: np.ndarray,
    candidate_stylometric: np.ndarray,
    target_stylometric: np.ndarray,
    source_semantic: np.ndarray,
    candidate_semantic: np.ndarray,
    stylometric_baseline: MeasurementBaseline,
    semantic_baseline: MeasurementBaseline,
    tolerances: ToleranceConfig | None = None,
    source_text: str | None = None,
    candidate_text: str | None = None,
) -> ReconstructionScore:
    """Score one candidate rewrite against preservation and style-target controls."""
    resolved_tolerances = tolerances or ToleranceConfig()

    normalized_source_stylometric = _validate_vector_length(
        source_stylometric,
        stylometric_baseline,
        label="source_stylometric",
    )
    normalized_candidate_stylometric = _validate_vector_length(
        candidate_stylometric,
        stylometric_baseline,
        label="candidate_stylometric",
    )
    normalized_target_stylometric = _validate_vector_length(
        target_stylometric,
        stylometric_baseline,
        label="target_stylometric",
    )
    normalized_source_semantic = _validate_vector_length(
        source_semantic,
        semantic_baseline,
        label="source_semantic",
    )
    normalized_candidate_semantic = _validate_vector_length(
        candidate_semantic,
        semantic_baseline,
        label="candidate_semantic",
    )

    semantic_source_distance = _normalized_distance(
        normalized_source_semantic,
        normalized_candidate_semantic,
        semantic_baseline,
    )
    stylistic_source_distance = _normalized_distance(
        normalized_source_stylometric,
        normalized_candidate_stylometric,
        stylometric_baseline,
    )
    stylistic_target_distance = _normalized_distance(
        normalized_candidate_stylometric,
        normalized_target_stylometric,
        stylometric_baseline,
    )
    source_target_distance = _normalized_distance(
        normalized_source_stylometric,
        normalized_target_stylometric,
        stylometric_baseline,
    )
    stylistic_target_improvement = source_target_distance - stylistic_target_distance
    stylistic_target_improvement_ratio = (
        1.0
        if source_target_distance == 0.0
        else float(stylistic_target_improvement / source_target_distance)
    )

    lexical_controls = None
    if source_text is not None and candidate_text is not None:
        lexical_controls = compute_lexical_controls(source_text, candidate_text)

    return ReconstructionScore(
        semantic_source_distance=semantic_source_distance,
        stylistic_source_distance=stylistic_source_distance,
        stylistic_target_distance=stylistic_target_distance,
        stylistic_target_improvement=stylistic_target_improvement,
        stylistic_target_improvement_ratio=stylistic_target_improvement_ratio,
        within_semantic_tolerance=(
            semantic_source_distance <= resolved_tolerances.semantic_preservation_max
        ),
        within_stylistic_tolerance=(
            stylistic_source_distance <= resolved_tolerances.stylistic_preservation_max
        ),
        within_target_tolerance=(
            stylistic_target_distance <= resolved_tolerances.stylistic_target_max
        ),
        lexical_controls=lexical_controls,
    )


def _derangement_indices(length: int, seed: int) -> np.ndarray:
    """Generate a deterministic derangement for shuffled-label controls."""
    if length < 2:
        return np.arange(length, dtype=int)

    rng = np.random.default_rng(seed)
    base = np.arange(length, dtype=int)
    for _ in range(128):
        shuffled = rng.permutation(length)
        if not np.any(shuffled == base):
            return shuffled
    return np.roll(base, 1)


def build_control_diagnostics(
    stylometric_measurements: MeasurementMatrix,
    semantic_measurements: MeasurementMatrix,
    stylometric_baseline: MeasurementBaseline,
    semantic_baseline: MeasurementBaseline,
    *,
    seed: int = DEFAULT_RECONSTRUCTION_SEED,
    tolerances: ToleranceConfig | None = None,
) -> dict[str, Any]:
    """Build deterministic identity/copy/random control summaries from corpus vectors."""
    resolved_tolerances = tolerances or ToleranceConfig()
    if stylometric_measurements.segment_ids != semantic_measurements.segment_ids:
        raise ValueError("stylometric and semantic segment IDs must align for control diagnostics")

    style_matrix = stylometric_measurements.matrix
    semantic_matrix = semantic_measurements.matrix
    chapter_count = style_matrix.shape[0]
    random_indices = _derangement_indices(chapter_count, seed)
    shifted_indices = np.roll(np.arange(chapter_count, dtype=int), -1)

    def summarize(scores: list[ReconstructionScore]) -> dict[str, Any]:
        return {
            "count": len(scores),
            "semantic_pass_rate": float(
                np.mean([score.within_semantic_tolerance for score in scores])
            ),
            "stylistic_pass_rate": float(
                np.mean([score.within_stylistic_tolerance for score in scores])
            ),
            "target_pass_rate": float(np.mean([score.within_target_tolerance for score in scores])),
            "mean_semantic_source_distance": float(
                np.mean([score.semantic_source_distance for score in scores])
            ),
            "mean_stylistic_source_distance": float(
                np.mean([score.stylistic_source_distance for score in scores])
            ),
            "mean_stylistic_target_distance": float(
                np.mean([score.stylistic_target_distance for score in scores])
            ),
            "mean_stylistic_target_improvement": float(
                np.mean([score.stylistic_target_improvement for score in scores])
            ),
        }

    identity_scores = [
        score_rewrite(
            source_stylometric=style_matrix[index],
            candidate_stylometric=style_matrix[index],
            target_stylometric=style_matrix[index],
            source_semantic=semantic_matrix[index],
            candidate_semantic=semantic_matrix[index],
            stylometric_baseline=stylometric_baseline,
            semantic_baseline=semantic_baseline,
            tolerances=resolved_tolerances,
        )
        for index in range(chapter_count)
    ]
    copy_source_scores = [
        score_rewrite(
            source_stylometric=style_matrix[index],
            candidate_stylometric=style_matrix[index],
            target_stylometric=style_matrix[target_index],
            source_semantic=semantic_matrix[index],
            candidate_semantic=semantic_matrix[index],
            stylometric_baseline=stylometric_baseline,
            semantic_baseline=semantic_baseline,
            tolerances=resolved_tolerances,
        )
        for index, target_index in enumerate(shifted_indices)
    ]
    random_target_scores = [
        score_rewrite(
            source_stylometric=style_matrix[index],
            candidate_stylometric=style_matrix[index],
            target_stylometric=style_matrix[target_index],
            source_semantic=semantic_matrix[index],
            candidate_semantic=semantic_matrix[index],
            stylometric_baseline=stylometric_baseline,
            semantic_baseline=semantic_baseline,
            tolerances=resolved_tolerances,
        )
        for index, target_index in enumerate(random_indices)
    ]

    return {
        "generated_at": utc_now(),
        "seed": seed,
        "tolerances": {
            "semantic_preservation_max": resolved_tolerances.semantic_preservation_max,
            "stylistic_preservation_max": resolved_tolerances.stylistic_preservation_max,
            "stylistic_target_max": resolved_tolerances.stylistic_target_max,
        },
        "controls": {
            "identity": summarize(identity_scores),
            "copy_source": summarize(copy_source_scores),
            "random_target": summarize(random_target_scores),
            "shuffled_label_seed": seed,
        },
    }


def write_measurement_baseline(
    baseline: MeasurementBaseline,
    path: Path,
) -> Path:
    """Persist one baseline report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(baseline.to_dict(), handle, ensure_ascii=False, indent=2)
    return path


def write_control_diagnostics(payload: dict[str, Any], path: Path) -> Path:
    """Persist control diagnostics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Phase 2 baseline generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--corpus-output-dir", type=Path, default=CORPUS_OUTPUT_DIR)
    parser.add_argument(
        "--stylometric-baseline-path",
        type=Path,
        default=DEFAULT_STYLOMETRIC_BASELINE_PATH,
    )
    parser.add_argument(
        "--semantic-baseline-path",
        type=Path,
        default=DEFAULT_SEMANTIC_BASELINE_PATH,
    )
    parser.add_argument(
        "--control-diagnostics-path",
        type=Path,
        default=DEFAULT_CONTROL_DIAGNOSTICS_PATH,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RECONSTRUCTION_SEED)
    parser.add_argument(
        "--allow-audit-issues",
        action="store_true",
        help="Bypass the Phase 1 synchronization gate and build baselines from current files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build Phase 2 baseline artifacts and deterministic control diagnostics."""
    args = build_argument_parser().parse_args(argv)
    baselines = build_measurement_baselines(
        corpus_dir=args.corpus_dir,
        corpus_output_dir=args.corpus_output_dir,
        require_clean_audit=not args.allow_audit_issues,
    )
    stylometric_measurements = load_stylometric_measurements(
        corpus_dir=args.corpus_dir,
        corpus_output_dir=args.corpus_output_dir,
    )
    semantic_measurements = load_semantic_measurements(
        corpus_dir=args.corpus_dir,
        corpus_output_dir=args.corpus_output_dir,
    )
    diagnostics = build_control_diagnostics(
        stylometric_measurements,
        semantic_measurements,
        baselines.stylometric,
        baselines.semantic,
        seed=args.seed,
    )

    stylometric_path = write_measurement_baseline(
        baselines.stylometric,
        args.stylometric_baseline_path,
    )
    semantic_path = write_measurement_baseline(
        baselines.semantic,
        args.semantic_baseline_path,
    )
    diagnostics_path = write_control_diagnostics(diagnostics, args.control_diagnostics_path)

    print("Phase 2 measurement baselines")
    print(f"  Stylometric: {project_relative_path(stylometric_path)}")
    print(f"  Semantic:    {project_relative_path(semantic_path)}")
    print(f"  Controls:    {project_relative_path(diagnostics_path)}")
    print(f"  Chapters:    {baselines.stylometric.chapter_count}")
    print(f"  Style dims:  {len(baselines.stylometric.dimension_order)}")
    print(f"  Sem dims:    {len(baselines.semantic.dimension_order)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
