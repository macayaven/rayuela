#!/usr/bin/env python3
"""
Cross-scale comparison: A' (computational stylometrics) vs B' (LLM style).

Computes:
  1. Correlation between each B' dimension and each A' feature
  2. Canonical correlation between the two feature sets
  3. Permutation test on B' (same as other scales)
  4. Full 4-scale summary table

Usage (inside Docker container):
    python src/scale_comparison.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

from project_config import (
    PROJECT_ROOT, DATA_PATH,
    STYLO_PATH, STYLO_META_PATH, EMB_A_PATH, EMB_B_PATH,
    STYLE_LLM_PATH, STYLE_LLM_JSON_PATH,
    DistanceMetric, RNG_SEED, N_PERMS,
    z_score as compute_z_score, z_standardize, get_all_chapters,
    filter_excluded_dims,
)


# ---------------------------------------------------------------------------
# Distance / similarity metrics
# ---------------------------------------------------------------------------

def mean_consecutive_distance(matrix, path):
    indices = [ch - 1 for ch in path]
    total = 0.0
    for i in range(len(indices) - 1):
        diff = matrix[indices[i]] - matrix[indices[i + 1]]
        total += float(np.sqrt(np.dot(diff, diff)))
    return total / (len(indices) - 1)


def mean_consecutive_cosine(matrix, path):
    indices = [ch - 1 for ch in path]
    total = 0.0
    for i in range(len(indices) - 1):
        total += float(np.dot(matrix[indices[i]], matrix[indices[i + 1]]))
    return total / (len(indices) - 1)


def permutation_dist(matrix, chapters, n_perms, rng, metric=DistanceMetric.EUCLIDEAN):
    fn = mean_consecutive_distance if metric == DistanceMetric.EUCLIDEAN else mean_consecutive_cosine
    chapters_arr = np.array(chapters)
    results = np.empty(n_perms)
    for i in range(n_perms):
        results[i] = fn(matrix, rng.permutation(chapters_arr).tolist())
    return results


# ---------------------------------------------------------------------------
# Correlation analysis: A' vs B'
# ---------------------------------------------------------------------------

def correlate_scales(stylo, stylo_meta, style_llm, style_llm_json):
    """Compute dimension-level correlations between A' and B'."""

    # Load feature names
    with open(stylo_meta) as f:
        meta = json.load(f)
    a_names = meta["feature_names"]

    with open(style_llm_json) as f:
        b_data = json.load(f)
    b_names = b_data["dimensions"]

    print("\n" + "=" * 65)
    print("CORRELATION: A' (computational) vs B' (LLM-perceived)")
    print("=" * 65)

    # For each B' dimension, find the most correlated A' feature
    print(f"\n  Best A' correlate for each B' dimension:")
    print(f"  {'B dimension':<30} {'Best A feature':<25} {'r':>8} {'p':>10}")
    print(f"  {'─' * 30} {'─' * 25} {'─' * 8} {'─' * 10}")

    for j, b_name in enumerate(b_names):
        best_r = 0
        best_p = 1
        best_a = ""
        for k, a_name in enumerate(a_names):
            r, p = stats.spearmanr(style_llm[:, j], stylo[:, k])
            if abs(r) > abs(best_r):
                best_r = r
                best_p = p
                best_a = a_name
        sig = "***" if best_p < 0.001 else "**" if best_p < 0.01 else "*" if best_p < 0.05 else ""
        print(f"  {b_name:<30} {best_a:<25} {best_r:>+8.3f} {best_p:>9.2e} {sig}")

    # Overall matrix correlation (Mantel-like: flatten pairwise distances)
    # Simpler: mean absolute Spearman across all dimension pairs
    all_r = []
    for j in range(style_llm.shape[1]):
        for k in range(stylo.shape[1]):
            r, _ = stats.spearmanr(style_llm[:, j], stylo[:, k])
            all_r.append(abs(r))
    print(f"\n  Mean |Spearman r| across all A'×B' dimension pairs: {np.mean(all_r):.3f}")

    # Also: correlation between A' and B' pairwise distance matrices
    # (Do chapters that are stylistically similar in A' also look similar in B'?)
    from scipy.spatial.distance import pdist, squareform
    dist_a = pdist(stylo, metric=DistanceMetric.EUCLIDEAN.value)
    dist_b = pdist(style_llm, metric=DistanceMetric.EUCLIDEAN.value)
    mantel_r, mantel_p = stats.spearmanr(dist_a, dist_b)
    print(f"  Mantel-like correlation (pairwise distance matrices): r={mantel_r:.3f}, p={mantel_p:.2e}")


# ---------------------------------------------------------------------------
# Cross-scale correlation: all pairs
# ---------------------------------------------------------------------------

def correlate_all_scales(scales: dict):
    """Compute pairwise Mantel-like correlations between all scales."""
    from scipy.spatial.distance import pdist

    names = list(scales.keys())
    print(f"\n  Pairwise distance-matrix correlations (Mantel Spearman r):")
    print(f"  {'':>25}", end="")
    for n in names:
        print(f" {n:>12}", end="")
    print()

    for i, n1 in enumerate(names):
        print(f"  {n1:>25}", end="")
        d1 = pdist(scales[n1], metric=DistanceMetric.EUCLIDEAN.value)
        for j, n2 in enumerate(names):
            if j <= i:
                d2 = pdist(scales[n2], metric=DistanceMetric.EUCLIDEAN.value)
                r, _ = stats.spearmanr(d1, d2)
                print(f" {r:>+12.3f}", end="")
            else:
                print(f" {'':>12}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    linear_path = data["reading_paths"]["linear"]
    hopscotch_path = data["reading_paths"]["hopscotch"]

    print("=" * 65)
    print("Four-Scale Comparison")
    print(f"({N_PERMS:,} random permutations per test)")
    print("=" * 65)

    results = []
    scales_for_mantel = {}
    all_chapters = get_all_chapters()

    # --- Scale A': Stylometric features ---
    if STYLO_PATH.exists():
        stylo = z_standardize(np.load(STYLO_PATH))
        scales_for_mantel["A' (stylometric)"] = stylo

        rng = np.random.default_rng(RNG_SEED)
        obs_lin = mean_consecutive_distance(stylo, linear_path)
        obs_hop = mean_consecutive_distance(stylo, hopscotch_path)
        dist_lin = permutation_dist(stylo, linear_path, N_PERMS, rng)
        dist_hop = permutation_dist(stylo, all_chapters, N_PERMS, rng)
        results.append({
            "name": "A' (stylometric)",
            "linear_z": compute_z_score(obs_lin, dist_lin, DistanceMetric.EUCLIDEAN),
            "hopscotch_z": compute_z_score(obs_hop, dist_hop, DistanceMetric.EUCLIDEAN),
        })

    # --- Scale A: E5 embeddings ---
    if EMB_A_PATH.exists():
        emb_a = np.load(EMB_A_PATH)
        scales_for_mantel["A (E5 holistic)"] = emb_a

        rng = np.random.default_rng(RNG_SEED)
        obs_lin = mean_consecutive_cosine(emb_a, linear_path)
        obs_hop = mean_consecutive_cosine(emb_a, hopscotch_path)
        dist_lin = permutation_dist(emb_a, linear_path, N_PERMS, rng, DistanceMetric.COSINE)
        dist_hop = permutation_dist(emb_a, all_chapters, N_PERMS, rng, DistanceMetric.COSINE)
        results.append({
            "name": "A (E5 holistic)",
            "linear_z": compute_z_score(obs_lin, dist_lin, DistanceMetric.COSINE),
            "hopscotch_z": compute_z_score(obs_hop, dist_hop, DistanceMetric.COSINE),
        })

    # --- Scale B: Narrative DNA ---
    if EMB_B_PATH.exists():
        emb_b = z_standardize(filter_excluded_dims(np.load(EMB_B_PATH)))
        scales_for_mantel["B (narrative DNA)"] = emb_b

        rng = np.random.default_rng(RNG_SEED)
        obs_lin = mean_consecutive_distance(emb_b, linear_path)
        obs_hop = mean_consecutive_distance(emb_b, hopscotch_path)
        dist_lin = permutation_dist(emb_b, linear_path, N_PERMS, rng)
        dist_hop = permutation_dist(emb_b, all_chapters, N_PERMS, rng)
        results.append({
            "name": "B (narrative DNA)",
            "linear_z": compute_z_score(obs_lin, dist_lin, DistanceMetric.EUCLIDEAN),
            "hopscotch_z": compute_z_score(obs_hop, dist_hop, DistanceMetric.EUCLIDEAN),
        })

    # --- Scale B': LLM stylistic ---
    if STYLE_LLM_PATH.exists():
        style_llm = z_standardize(np.load(STYLE_LLM_PATH))
        scales_for_mantel["B' (LLM style)"] = style_llm

        rng = np.random.default_rng(RNG_SEED)
        obs_lin = mean_consecutive_distance(style_llm, linear_path)
        obs_hop = mean_consecutive_distance(style_llm, hopscotch_path)
        dist_lin = permutation_dist(style_llm, linear_path, N_PERMS, rng)
        dist_hop = permutation_dist(style_llm, all_chapters, N_PERMS, rng)
        results.append({
            "name": "B' (LLM style)",
            "linear_z": compute_z_score(obs_lin, dist_lin, DistanceMetric.EUCLIDEAN),
            "hopscotch_z": compute_z_score(obs_hop, dist_hop, DistanceMetric.EUCLIDEAN),
        })
    else:
        print(f"\n  [SKIP] Scale B' — {STYLE_LLM_PATH} not found")
        print(f"         Still running? Check: docker logs rayuela-extract-style")

    # --- Summary table ---
    print("\n" + "=" * 65)
    print("PERMUTATION TEST RESULTS (4-scale)")
    print("=" * 65)
    print()
    print(f"  {'Scale':<25} {'Linear z':>12} {'Hopscotch z':>14}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 14}")
    for r in results:
        print(f"  {r['name']:<25} {r['linear_z']:>+12.2f}σ {r['hopscotch_z']:>+14.2f}σ")

    print()
    print("  Sign convention: positive z = smoother than random (all scales)")

    # --- Mantel correlations ---
    if len(scales_for_mantel) > 1:
        correlate_all_scales(scales_for_mantel)

    # --- A' vs B' dimension-level correlation ---
    if STYLO_PATH.exists() and STYLE_LLM_PATH.exists():
        correlate_scales(
            np.load(STYLO_PATH),
            STYLO_META_PATH,
            np.load(STYLE_LLM_PATH),
            STYLE_LLM_JSON_PATH,
        )

    # Save results
    output_path = PROJECT_ROOT / "outputs" / "four_scale_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
