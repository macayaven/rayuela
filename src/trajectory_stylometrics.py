#!/usr/bin/env python3
"""
Permutation test for Scale A' (stylometric features).

Same methodology as trajectory_baselines.py but adapted for heterogeneous
feature vectors that need standardization before distance computation.

Scale A (E5 embeddings) used cosine similarity on L2-normalized vectors.
Scale A' features are on different scales (TTR is 0-1, sentence length is
10-200, punctuation counts vary wildly), so we z-score standardize each
feature before computing Euclidean distance between consecutive chapters.

Lower mean consecutive distance = smoother path.

Also runs the same test on Scales A and B for direct three-way comparison.

Usage (inside Docker container):
    python src/trajectory_stylometrics.py

    # Custom permutation count:
    python src/trajectory_stylometrics.py --n-perms 10000
"""

import argparse
import json
import numpy as np
from pathlib import Path

from project_config import (
    PROJECT_ROOT, DATA_PATH, STYLO_PATH, EMB_A_PATH, EMB_B_PATH,
    DistanceMetric, RNG_SEED, N_PERMS,
    z_score as compute_z_score,
    z_standardize, continuity_corrected_percentile, get_all_chapters,
    filter_excluded_dims,
)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def mean_consecutive_distance(matrix: np.ndarray, path: list[int]) -> float:
    """
    Mean Euclidean distance between consecutive chapters in a reading path.

    Args:
        matrix: (155, D) feature matrix (should be standardized)
        path: list of 1-indexed chapter numbers
    """
    indices = [ch - 1 for ch in path]
    total = 0.0
    for i in range(len(indices) - 1):
        diff = matrix[indices[i]] - matrix[indices[i + 1]]
        total += float(np.sqrt(np.dot(diff, diff)))
    return total / (len(indices) - 1)


def mean_consecutive_cosine_sim(matrix: np.ndarray, path: list[int]) -> float:
    """
    Mean cosine similarity between consecutive chapters (for L2-normalized embeddings).
    Higher = smoother.
    """
    indices = [ch - 1 for ch in path]
    total = 0.0
    for i in range(len(indices) - 1):
        total += float(np.dot(matrix[indices[i]], matrix[indices[i + 1]]))
    return total / (len(indices) - 1)


def mean_curvature(matrix: np.ndarray, path: list[int]) -> float:
    """
    Mean angular curvature along a reading path.

    For each consecutive triplet (a, b, c), compute the angle at b between
    direction vectors (a→b) and (b→c). The angle is in radians [0, π].
    Higher curvature = sharper turns = more "collision" between chapters.

    This directly measures direction change, complementing mean_consecutive_distance
    which only measures step size.
    """
    indices = [ch - 1 for ch in path]
    if len(indices) < 3:
        return 0.0

    angles = []
    for i in range(len(indices) - 2):
        v1 = matrix[indices[i + 1]] - matrix[indices[i]]      # a→b
        v2 = matrix[indices[i + 2]] - matrix[indices[i + 1]]  # b→c
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue  # skip degenerate steps (identical chapters)
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        # Clamp to [-1, 1] for numerical stability
        cos_angle = max(-1.0, min(1.0, float(cos_angle)))
        angles.append(np.arccos(cos_angle))

    return float(np.mean(angles)) if angles else 0.0


def distance_autocorrelation(matrix: np.ndarray, path: list[int],
                             max_lag: int = 20) -> list[float]:
    """
    Compute serial autocorrelation of step distances at multiple lags.

    For each lag k, compute the Pearson correlation between consecutive
    distances at positions (i) and (i+k). This reveals how smoothness
    "decays" with distance along the reading path.

    Note: This is serial autocorrelation, NOT Moran's I (which uses a
    spatial weight matrix). The name was corrected during external review.

    Returns a list of correlations for lags 1 through max_lag.
    """
    indices = [ch - 1 for ch in path]
    # Compute all consecutive distances
    dists = []
    for i in range(len(indices) - 1):
        diff = matrix[indices[i]] - matrix[indices[i + 1]]
        dists.append(float(np.sqrt(np.dot(diff, diff))))
    dists = np.array(dists)

    if len(dists) < max_lag + 2:
        max_lag = len(dists) - 2

    correlations = []
    for lag in range(1, max_lag + 1):
        if len(dists) - lag < 3:
            correlations.append(0.0)
            continue
        x = dists[:len(dists) - lag]
        y = dists[lag:]
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))

    return correlations


# ---------------------------------------------------------------------------
# Permutation distribution
# ---------------------------------------------------------------------------

def permutation_distribution(
    matrix: np.ndarray,
    chapters: list[int],
    n_permutations: int,
    rng: np.random.Generator,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
) -> np.ndarray:
    """Generate distribution of path smoothness from random orderings."""
    chapters_arr = np.array(chapters)
    results = np.empty(n_permutations)

    metric_fn = (mean_consecutive_distance if metric == DistanceMetric.EUCLIDEAN
                 else mean_consecutive_cosine_sim)

    for i in range(n_permutations):
        shuffled = rng.permutation(chapters_arr).tolist()
        results[i] = metric_fn(matrix, shuffled)

    return results


# ---------------------------------------------------------------------------
# Run test for one scale
# ---------------------------------------------------------------------------

def run_test(
    name: str,
    matrix: np.ndarray,
    linear_path: list[int],
    hopscotch_path: list[int],
    n_permutations: int,
    rng: np.random.Generator,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
) -> dict:
    """
    Run permutation test for one scale and print results.

    Returns dict with z-scores and percentiles for both paths.
    """
    metric_fn = (mean_consecutive_distance if metric == DistanceMetric.EUCLIDEAN
                 else mean_consecutive_cosine_sim)

    linear_observed = metric_fn(matrix, linear_path)
    hopscotch_observed = metric_fn(matrix, hopscotch_path)

    print(f"\n{'─' * 65}")
    print(f"  {name}")
    print(f"  Matrix shape: {matrix.shape}, metric: {metric.value}")
    print(f"{'─' * 65}")

    # --- Linear path ---
    print(f"\n  Linear path ({len(linear_path)} chapters):")
    print(f"    Observed: {linear_observed:.4f}")

    linear_dist = permutation_distribution(
        matrix, linear_path, n_permutations, rng, metric
    )
    print(f"    Random:   mean={linear_dist.mean():.4f}  std={linear_dist.std():.4f}")

    linear_z = compute_z_score(linear_observed, linear_dist, metric)
    linear_pct = continuity_corrected_percentile(linear_observed, linear_dist, metric)

    print(f"    Z-score:  {linear_z:+.2f}σ (positive = smoother)")
    print(f"    Smoother than {linear_pct:.1f}% of random orderings")

    # --- Hopscotch path ---
    print(f"\n  Hopscotch path ({len(hopscotch_path)} chapters):")
    print(f"    Observed: {hopscotch_observed:.4f}")

    hopscotch_dist = permutation_distribution(
        matrix, get_all_chapters(), n_permutations, rng, metric
    )
    print(f"    Random:   mean={hopscotch_dist.mean():.4f}  std={hopscotch_dist.std():.4f}")

    hopscotch_z = compute_z_score(hopscotch_observed, hopscotch_dist, metric)
    hopscotch_pct = continuity_corrected_percentile(hopscotch_observed, hopscotch_dist, metric)

    print(f"    Z-score:  {hopscotch_z:+.2f}σ (positive = smoother)")
    print(f"    Smoother than {hopscotch_pct:.1f}% of random orderings")

    return {
        "name": name,
        "metric": metric.value,
        "linear_z": float(linear_z),
        "linear_pct": linear_pct,
        "hopscotch_z": float(hopscotch_z),
        "hopscotch_pct": hopscotch_pct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Permutation test across three scales")
    parser.add_argument("--n-perms", type=int, default=N_PERMS,
                        help=f"Number of random permutations (default: {N_PERMS})")
    args = parser.parse_args()

    n_perms = args.n_perms

    # Load reading paths from single source of truth
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    linear_path = data["reading_paths"]["linear"]
    hopscotch_path = data["reading_paths"]["hopscotch"]

    print("=" * 65)
    print("Three-Scale Permutation Test")
    print(f"({n_perms:,} random permutations per test)")
    print("=" * 65)

    results = []

    # --- Scale A': Stylometric features ---
    if STYLO_PATH.exists():
        stylo = z_standardize(np.load(STYLO_PATH))

        rng = np.random.default_rng(RNG_SEED)
        res = run_test("Scale A' (Stylometric Features — content-free)",
                       stylo, linear_path, hopscotch_path, n_perms, rng,
                       metric=DistanceMetric.EUCLIDEAN)
        results.append(res)
    else:
        print(f"\n  [SKIP] Scale A' — {STYLO_PATH} not found")
        print(f"         Run: python src/stylometrics.py")

    # --- Scale A: E5 embeddings ---
    if EMB_A_PATH.exists():
        emb_a = np.load(EMB_A_PATH)
        rng = np.random.default_rng(RNG_SEED)
        res = run_test("Scale A (E5 Holistic Embeddings — meaning+form)",
                       emb_a, linear_path, hopscotch_path, n_perms, rng,
                       metric=DistanceMetric.COSINE)
        results.append(res)
    else:
        print(f"\n  [SKIP] Scale A — {EMB_A_PATH} not found")

    # --- Scale B: Narrative DNA ---
    if EMB_B_PATH.exists():
        emb_b = z_standardize(filter_excluded_dims(np.load(EMB_B_PATH)))

        rng = np.random.default_rng(RNG_SEED)
        res = run_test("Scale B (Narrative DNA — explicit semantic decomposition)",
                       emb_b, linear_path, hopscotch_path, n_perms, rng,
                       metric=DistanceMetric.EUCLIDEAN)
        results.append(res)
    else:
        print(f"\n  [SKIP] Scale B — {EMB_B_PATH} not found")

    # --- Summary comparison ---
    if results:
        print("\n" + "=" * 65)
        print("CROSS-SCALE COMPARISON")
        print("=" * 65)
        print()
        print(f"  {'Scale':<50} {'Linear z':>10} {'Hopscotch z':>12}")
        print(f"  {'─' * 50} {'─' * 10} {'─' * 12}")

        for r in results:
            print(f"  {r['name']:<50} {r['linear_z']:>+10.2f}σ {r['hopscotch_z']:>+12.2f}σ")

        print()
        print("  Sign convention: positive z = smoother than random (all scales)")
        print()
        print("  If linear is extreme and hopscotch is ~0 across ALL three scales")
        print("  (including A' which is provably content-free), the finding is")
        print("  airtight: the ordering effects are real and scale-independent.")

    # --- Curvature analysis ---
    print("\n" + "=" * 65)
    print("CURVATURE ANALYSIS (direction change at each step)")
    print("=" * 65)
    print()
    print("  Mean angular curvature (radians; higher = sharper turns = more collision)")
    print(f"  {'Scale':<50} {'Linear':>10} {'Hopscotch':>12} {'Random μ':>10}")
    print(f"  {'─' * 50} {'─' * 10} {'─' * 12} {'─' * 10}")

    curvature_results = []
    scale_configs = [
        ("A' (Stylometric)", STYLO_PATH, True),
        ("A (E5 Holistic)", EMB_A_PATH, False),
        ("B (Narrative DNA)", EMB_B_PATH, True),
    ]
    for name, matrix_path, needs_standardize in scale_configs:
        if not matrix_path.exists():
            continue
        mat = np.load(matrix_path)
        if needs_standardize:
            mat = z_standardize(mat)

        lin_curv = mean_curvature(mat, linear_path)
        hop_curv = mean_curvature(mat, hopscotch_path)

        # Separate null distributions (same approach as smoothness tests)
        # Linear null: permute Ch.1-56 only (tests ordering, not pool)
        rng_c = np.random.default_rng(RNG_SEED)
        lin_rand_curvs = []
        for _ in range(n_perms):
            perm = list(rng_c.permutation(linear_path))
            lin_rand_curvs.append(mean_curvature(mat, perm))
        lin_rand_curvs = np.array(lin_rand_curvs)

        # Hopscotch null: permute all 155 chapters
        hop_rand_curvs = []
        for _ in range(n_perms):
            perm = list(rng_c.permutation(get_all_chapters()))
            hop_rand_curvs.append(mean_curvature(mat, perm))
        hop_rand_curvs = np.array(hop_rand_curvs)

        ndims = mat.shape[1]
        print(f"  {name:<50} {lin_curv:>10.4f} {hop_curv:>12.4f} (d={ndims})")

        # Curvature uses Euclidean-like convention: lower = smoother
        # z_score with EUCLIDEAN gives: (null_mean - observed) / null_std
        curvature_results.append({
            "name": name,
            "dimensions": int(ndims),
            "linear_curvature": float(lin_curv),
            "hopscotch_curvature": float(hop_curv),
            "linear_null_mean": float(lin_rand_curvs.mean()),
            "linear_null_std": float(lin_rand_curvs.std()),
            "hopscotch_null_mean": float(hop_rand_curvs.mean()),
            "hopscotch_null_std": float(hop_rand_curvs.std()),
            "linear_z": float(compute_z_score(lin_curv, lin_rand_curvs, DistanceMetric.EUCLIDEAN)),
            "hopscotch_z": float(compute_z_score(hop_curv, hop_rand_curvs, DistanceMetric.EUCLIDEAN)),
        })

    print()
    print("  Curvature z-scores (positive = smoother/gentler turns than random):")
    print("  (Each path tested against its own null: linear vs shuffled Ch.1-56,")
    print("   hopscotch vs shuffled all 155)")
    for r in curvature_results:
        print(f"    {r['name']:<50} Lin: {r['linear_z']:+.2f}σ  Hop: {r['hopscotch_z']:+.2f}σ")
        if r['dimensions'] > 100:
            print(f"      ⚠ d={r['dimensions']}: concentration of measure — all angles cluster "
                  f"near {r['hopscotch_null_mean']:.3f} rad, z-scores have limited interpretive value")

    # --- Distance autocorrelation ---
    print("\n" + "=" * 65)
    print("DISTANCE AUTOCORRELATION (how smoothness decays with lag)")
    print("=" * 65)

    autocorrelation_results = {}
    for name, matrix_path, needs_standardize in scale_configs:
        if not matrix_path.exists():
            continue
        mat = np.load(matrix_path)
        if needs_standardize:
            mat = z_standardize(mat)

        lin_auto = distance_autocorrelation(mat, linear_path, max_lag=10)
        hop_auto = distance_autocorrelation(mat, hopscotch_path, max_lag=10)

        print(f"\n  {name}:")
        print(f"    {'Lag':>5} {'Linear r':>10} {'Hopscotch r':>12}")
        print(f"    {'─' * 5} {'─' * 10} {'─' * 12}")
        for lag, (lr, hr) in enumerate(zip(lin_auto, hop_auto), start=1):
            print(f"    {lag:>5} {lr:>+10.3f} {hr:>+12.3f}")

        autocorrelation_results[name] = {
            "linear": lin_auto,
            "hopscotch": hop_auto,
        }

    # Save all results
    all_results = {
        "permutation_tests": results,
        "curvature": curvature_results,
        "autocorrelation": autocorrelation_results,
    }
    output_path = PROJECT_ROOT / "outputs" / "trajectory_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
