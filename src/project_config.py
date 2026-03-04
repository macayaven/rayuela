#!/usr/bin/env python3
"""
Centralized configuration for Project Rayuela.

Single source of truth for constants, enums, and shared utilities used
across all analysis scripts. Prevents magic strings, duplicated constants,
and inconsistent methodology.

All scripts should import from here rather than defining their own copies.
"""

import json
import numpy as np
from enum import Enum
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "rayuela_raw.json"

# Embedding / feature outputs
EMB_A_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy"
EMB_B_PATH = PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna_vectors.npy"
STYLO_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_stylometrics.npy"
STYLO_META_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_stylometrics_metadata.json"
STYLE_LLM_PATH = PROJECT_ROOT / "outputs" / "stylistic" / "stylistic_dna_vectors.npy"
STYLE_LLM_JSON_PATH = PROJECT_ROOT / "outputs" / "stylistic" / "stylistic_dna.json"
NARRATIVE_DNA_PATH = PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json"

OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ---------------------------------------------------------------------------
# Distance metrics (enum eliminates magic strings)
# ---------------------------------------------------------------------------

class DistanceMetric(Enum):
    """
    Distance metric for permutation tests and trajectory analysis.

    The sign property encodes the unified z-score convention:
      positive z = smoother than random (all scales, all scripts).

    Euclidean: lower distance = smoother → sign = -1
      z = -1 * (observed - null_mean) / null_std = (null_mean - observed) / null_std
    Cosine similarity: higher similarity = smoother → sign = +1
      z = +1 * (observed - null_mean) / null_std
    """
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"

    @property
    def sign(self) -> int:
        """Sign multiplier for z-score: positive = smoother than random."""
        return -1 if self == DistanceMetric.EUCLIDEAN else 1


# ---------------------------------------------------------------------------
# Statistical constants
# ---------------------------------------------------------------------------

RNG_SEED = 42
N_PERMS = 5_000
NUM_CHAPTERS = 155


# ---------------------------------------------------------------------------
# Shared statistical utilities
# ---------------------------------------------------------------------------

def z_score(observed: float, null_dist: np.ndarray, metric: DistanceMetric) -> float:
    """
    Compute z-score with unified sign convention: positive = smoother than random.

    Args:
        observed: The observed metric value for the actual reading path.
        null_dist: Array of metric values from permuted (random) orderings.
        metric: DistanceMetric enum — determines sign convention.

    Returns:
        Z-score where positive means the path is smoother than random.

    Raises:
        TypeError: If metric is not a DistanceMetric enum member.
    """
    if not isinstance(metric, DistanceMetric):
        raise TypeError(
            f"metric must be a DistanceMetric enum, got {type(metric).__name__}: {metric!r}. "
            f"Use DistanceMetric.EUCLIDEAN or DistanceMetric.COSINE."
        )
    std = null_dist.std()
    if std == 0:
        return 0.0
    return metric.sign * (observed - null_dist.mean()) / std


def z_standardize(matrix: np.ndarray) -> np.ndarray:
    """
    Z-standardize each column (feature dimension) of a matrix.

    Equalizes dimension contributions before computing Euclidean distances.
    Columns with zero variance are left as zeros (not divided).

    Args:
        matrix: (N, D) feature matrix (e.g., 155 chapters × 20 dimensions).

    Returns:
        (N, D) matrix with each column having mean=0, std=1.
    """
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (matrix - mu) / sigma


def z_standardize_scores_dict(
    scores: dict[int, dict[str, float]],
    dims: list[str],
) -> dict[int, np.ndarray]:
    """
    Z-standardize a dict of per-chapter score dicts for Euclidean distance.

    This is the dict-based counterpart to z_standardize(). Use when code
    needs O(1) chapter lookup (e.g., trajectory traversal) rather than a
    dense matrix.

    Args:
        scores: {chapter_number: {dimension_name: score_value}}
        dims: Ordered list of dimension names.

    Returns:
        {chapter_number: z-standardized numpy vector}
    """
    sorted_chs = sorted(scores.keys())
    matrix = np.array([[scores[ch][d] for d in dims] for ch in sorted_chs])
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0)
    sigma[sigma == 0] = 1.0
    return {ch: (np.array([scores[ch][d] for d in dims]) - mu) / sigma
            for ch in sorted_chs}


def continuity_corrected_percentile(
    observed: float,
    null_dist: np.ndarray,
    metric: DistanceMetric,
) -> float:
    """
    Continuity-corrected percentile: (n_extreme + 1) / (n_perm + 1) * 100.

    For Euclidean: counts how many null values are GREATER (less smooth) than observed.
    For Cosine: counts how many null values are LESS (less smooth) than observed.

    Args:
        observed: The observed metric value.
        null_dist: Array of null distribution values.
        metric: DistanceMetric enum.

    Returns:
        Percentile (0-100) indicating how smooth the path is vs random.
    """
    if not isinstance(metric, DistanceMetric):
        raise TypeError(
            f"metric must be a DistanceMetric enum, got {type(metric).__name__}: {metric!r}"
        )
    if metric == DistanceMetric.COSINE:
        n_extreme = int(np.sum(null_dist < observed))
    else:
        n_extreme = int(np.sum(null_dist > observed))
    return float((n_extreme + 1) / (len(null_dist) + 1) * 100)


# ---------------------------------------------------------------------------
# Reading paths (loaded once from the canonical JSON)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_raw_data() -> dict:
    """Load rayuela_raw.json once and cache it."""
    with open(DATA_PATH) as f:
        return json.load(f)


def get_reading_paths() -> tuple[list[int], list[int]]:
    """
    Load reading paths from the canonical data file.

    Returns:
        (hopscotch_path, linear_path) — both as lists of 1-indexed chapter numbers.
    """
    data = _load_raw_data()
    return data["reading_paths"]["hopscotch"], data["reading_paths"]["linear"]


def get_all_chapters() -> list[int]:
    """Return list of all chapter numbers [1, 2, ..., 155]."""
    return list(range(1, NUM_CHAPTERS + 1))


# Module-level constants loaded from JSON — fail hard if data file missing.
# Every analysis script depends on these; silently continuing with empty
# paths would produce wrong results without any error.
TABLERO, LINEAR_ORDER = get_reading_paths()


# ---------------------------------------------------------------------------
# Section metadata
# ---------------------------------------------------------------------------

SECTION_COLORS = {
    "Del lado de allá": "#2196F3",                         # Blue — Paris
    "Del lado de acá": "#FF9800",                          # Orange — Buenos Aires
    "De otros lados (Capítulos prescindibles)": "#9C27B0", # Purple — Expendable
}

SECTION_SHORT = {
    "Del lado de allá": "Allá (Paris)",
    "Del lado de acá": "Acá (Buenos Aires)",
    "De otros lados (Capítulos prescindibles)": "Otros lados (Expendable)",
}


# ---------------------------------------------------------------------------
# Narrative DNA dimensions (Scale B)
# ---------------------------------------------------------------------------

DIM_GROUPS = {
    "Thematic": [
        "existential_questioning", "art_and_aesthetics",
        "everyday_mundanity", "death_and_mortality", "love_and_desire",
    ],
    "Emotional": [
        "emotional_intensity", "humor_and_irony",
        "melancholy_and_nostalgia", "tension_and_anxiety",
    ],
    "Character": [
        "oliveira_centrality", "la_maga_presence",
        "character_density", "interpersonal_conflict",
    ],
    "Narrative": [
        "interiority", "dialogue_density", "metafiction", "temporal_clarity",
    ],
    "Formal": [
        "spatial_grounding", "language_experimentation", "intertextual_density",
    ],
}

# Flat ordered list matching the grouping above (all 20 — used by replication scripts)
DIMS_ORDERED_ALL = [d for dims in DIM_GROUPS.values() for d in dims]

# Dimensions excluded after inter-rater reliability testing (Scale B replication).
# temporal_clarity: ρ = -0.30, κw = -0.16 (anti-correlated between Qwen 3.5 27B
# and Nemotron 70B). Bootstrap 95% CI entirely negative [-0.47, -0.12].
# Root cause: rubric polarity confusion ("1=Clear, 10=Fragmented" vs label
# "Clarity"). Reverse-coding does NOT fix it (ρ only reaches +0.30).
DIMS_EXCLUDED = {"temporal_clarity"}

# Validated dimensions for all downstream analysis (19D after exclusion)
DIMS_ORDERED = [d for d in DIMS_ORDERED_ALL if d not in DIMS_EXCLUDED]

# Column indices to keep when loading the 20D .npy matrices
_VALIDATED_COLS = [i for i, d in enumerate(DIMS_ORDERED_ALL) if d not in DIMS_EXCLUDED]


def filter_excluded_dims(matrix: np.ndarray) -> np.ndarray:
    """
    Remove excluded dimension columns from a (N, 20) Scale B matrix.

    Use when loading narrative_dna_vectors.npy (which was saved with all 20
    dimensions) for downstream analysis that should only use validated dims.
    """
    return matrix[:, _VALIDATED_COLS]

DIM_LABELS = {
    "existential_questioning": "Existential",
    "art_and_aesthetics": "Art/Aesthetics",
    "everyday_mundanity": "Mundanity",
    "death_and_mortality": "Death",
    "love_and_desire": "Love/Desire",
    "emotional_intensity": "Emotion",
    "humor_and_irony": "Humor/Irony",
    "melancholy_and_nostalgia": "Melancholy",
    "tension_and_anxiety": "Tension",
    "oliveira_centrality": "Oliveira",
    "la_maga_presence": "La Maga",
    "character_density": "Characters",
    "interpersonal_conflict": "Conflict",
    "interiority": "Interiority",
    "dialogue_density": "Dialogue",
    "metafiction": "Metafiction",
    "temporal_clarity": "Temporal frag.",
    "spatial_grounding": "Spatial",
    "language_experimentation": "Lang. experiment",
    "intertextual_density": "Intertextual",
}


# ---------------------------------------------------------------------------
# UMAP defaults
# ---------------------------------------------------------------------------

UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.1
