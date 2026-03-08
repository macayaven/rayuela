#!/usr/bin/env python3
"""
Permutation baselines for trajectory smoothness.

Tests whether the linear and hopscotch reading paths are texturally smoother
than random orderings of the same chapters. This addresses a key confound
identified in ANALYSIS_FRAMEWORK.md: observed smoothness might be a property
of the chapter pool, not the ordering.

We generate four trajectory classes:
  1. Linear path (Ch. 1→56, as written)
  2. Random orderings of Ch. 1–56 (same chapters, shuffled — 10,000 permutations)
  3. Hopscotch path (Tablero de Dirección, all 155 chapters)
  4. Random permutations of all 155 chapters (10,000 permutations)

If the real paths are smoother than the random baselines, the ordering
is intentional. If not, the smoothness is an artifact of the chapter pool.

Cortázar himself suggested that readers could create their own novel by
choosing their own chapter sequence — these random baselines are literally
the paths of that imagined reader.

Usage (inside Docker container):
    python src/trajectory_baselines.py
"""

import json

import numpy as np

from project_config import (
    DATA_PATH,
    N_PERMS,
    RNG_SEED,
    DistanceMetric,
    get_all_chapters,
)
from project_config import (
    EMB_A_PATH as EMB_PATH,
)
from project_config import (
    z_score as compute_z_score,
)

N_PERMUTATIONS = N_PERMS


def mean_consecutive_similarity(embeddings: np.ndarray, path: list[int]) -> float:
    """
    Mean cosine similarity between consecutive chapters in a reading path.

    Args:
        embeddings: (155, 1024) L2-normalized
        path: list of 1-indexed chapter numbers
    """
    indices = [ch - 1 for ch in path]  # convert to 0-indexed
    total = 0.0
    for i in range(len(indices) - 1):
        total += float(np.dot(embeddings[indices[i]], embeddings[indices[i + 1]]))
    return total / (len(indices) - 1)


def permutation_distribution(
    embeddings: np.ndarray,
    chapters: list[int],
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a distribution of mean consecutive similarities from random orderings.

    Args:
        embeddings: (155, 1024) L2-normalized
        chapters: list of 1-indexed chapter numbers to permute
        n_permutations: how many random orderings to generate
        rng: numpy random generator for reproducibility

    Returns:
        Array of shape (n_permutations,) with mean consecutive similarity per permutation
    """
    chapters_arr = np.array(chapters)
    results = np.empty(n_permutations)

    for i in range(n_permutations):
        shuffled = rng.permutation(chapters_arr).tolist()
        results[i] = mean_consecutive_similarity(embeddings, shuffled)

    return results


def percentile_rank(observed: float, distribution: np.ndarray) -> float:
    """What percentage of the distribution falls below the observed value."""
    return float(np.mean(distribution < observed) * 100)


def main():
    embeddings = np.load(EMB_PATH)
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    linear_path = data["reading_paths"]["linear"]
    hopscotch_path = data["reading_paths"]["hopscotch"]
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 65)
    print("Permutation baselines for trajectory smoothness")
    print(f"({N_PERMUTATIONS:,} random permutations per test)")
    print("=" * 65)
    print()

    # -----------------------------------------------------------------
    # Test 1: Linear path vs. random orderings of chapters 1-56
    # -----------------------------------------------------------------
    linear_observed = mean_consecutive_similarity(embeddings, linear_path)

    print(f"TEST 1: Linear path (Ch. 1→56, {len(linear_path)} chapters)")
    print(f"  Observed mean consecutive similarity: {linear_observed:.4f}")
    print(f"  Generating {N_PERMUTATIONS:,} random orderings of Ch. 1-56...")

    linear_distribution = permutation_distribution(
        embeddings, linear_path, N_PERMUTATIONS, rng
    )

    linear_pct = percentile_rank(linear_observed, linear_distribution)

    print(f"  Random baseline: mean={linear_distribution.mean():.4f}  "
          f"std={linear_distribution.std():.4f}")
    print(f"  Random baseline: min={linear_distribution.min():.4f}  "
          f"max={linear_distribution.max():.4f}")
    print(f"  Observed percentile rank: {linear_pct:.1f}%")
    print()

    if linear_pct > 95:
        print(f"  ✓ The linear ordering is smoother than {linear_pct:.1f}% of "
              f"random orderings.")
        print(f"    → The sequence 1→2→3→...→56 is intentionally smooth (p < "
              f"{(100 - linear_pct) / 100:.4f}).")
    elif linear_pct > 50:
        print(f"  ~ The linear ordering is smoother than {linear_pct:.1f}% of "
              f"random orderings.")
        print("    → Some signal, but not statistically remarkable.")
    else:
        print("  ✗ The linear ordering is NOT smoother than random.")
        print("    → The 1→56 sequence has no special textural continuity.")

    print()

    # -----------------------------------------------------------------
    # Test 2: Hopscotch path vs. random permutations of all 155 chapters
    # -----------------------------------------------------------------
    hopscotch_observed = mean_consecutive_similarity(embeddings, hopscotch_path)

    print(f"TEST 2: Hopscotch path (Tablero, {len(hopscotch_path)} chapters)")
    print(f"  Observed mean consecutive similarity: {hopscotch_observed:.4f}")
    print(f"  Generating {N_PERMUTATIONS:,} random permutations of all 155 chapters...")

    all_chapters = get_all_chapters()
    hopscotch_distribution = permutation_distribution(
        embeddings, all_chapters, N_PERMUTATIONS, rng
    )

    hopscotch_pct = percentile_rank(hopscotch_observed, hopscotch_distribution)

    print(f"  Random baseline: mean={hopscotch_distribution.mean():.4f}  "
          f"std={hopscotch_distribution.std():.4f}")
    print(f"  Random baseline: min={hopscotch_distribution.min():.4f}  "
          f"max={hopscotch_distribution.max():.4f}")
    print(f"  Observed percentile rank: {hopscotch_pct:.1f}%")
    print()

    if hopscotch_pct > 95:
        print(f"  ✓ The Tablero is smoother than {hopscotch_pct:.1f}% of "
              f"random permutations.")
        print("    → Cortázar's hopscotch order has intentional textural continuity.")
    elif hopscotch_pct > 50:
        print(f"  ~ The Tablero is smoother than {hopscotch_pct:.1f}% of "
              f"random permutations.")
        print("    → Some signal, but not statistically remarkable.")
    else:
        print("  ✗ The Tablero is NOT smoother than random permutations.")
        print("    → The hopscotch order was not designed for textural flow.")

    print()

    # -----------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print()
    print(f"  {'Path':<20} {'Observed':>10} {'Random μ':>10} "
          f"{'Random σ':>10} {'Percentile':>12}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 12}")
    print(f"  {'Linear (1→56)':<20} {linear_observed:>10.4f} "
          f"{linear_distribution.mean():>10.4f} "
          f"{linear_distribution.std():>10.4f} "
          f"{linear_pct:>11.1f}%")
    print(f"  {'Hopscotch (Tablero)':<20} {hopscotch_observed:>10.4f} "
          f"{hopscotch_distribution.mean():>10.4f} "
          f"{hopscotch_distribution.std():>10.4f} "
          f"{hopscotch_pct:>11.1f}%")
    print()

    # Z-scores: positive = smoother than random (cosine similarity: higher = smoother)
    linear_z = compute_z_score(linear_observed, linear_distribution, DistanceMetric.COSINE)
    hopscotch_z = compute_z_score(hopscotch_observed, hopscotch_distribution, DistanceMetric.COSINE)

    print("  Z-scores (positive = smoother than random):")
    print(f"    Linear:    {linear_z:+.2f}σ")
    print(f"    Hopscotch: {hopscotch_z:+.2f}σ")
    print()
    print("  The path with higher z-score shows more intentional textural")
    print("  ordering relative to its random baseline.")


if __name__ == "__main__":
    main()
