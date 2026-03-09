#!/usr/bin/env python3
"""
Convergent validity: compare Scale B scores from two independent LLMs.

Loads Narrative DNA scores from two extractors (Qwen 3.5 27B and Nemotron 70B)
and computes inter-rater reliability metrics. If both models agree on chapter
scores, the signal is in the text — not in one model's idiosyncrasies.

Metrics computed:
  - Per-dimension Spearman ρ (rank correlation)
  - Per-dimension Pearson r (linear correlation)
  - Per-dimension quadratic weighted Cohen's κ (ordinal agreement)
  - Per-dimension MAE (mean absolute error)
  - Overall matrix correlation
  - Agreement rate (% within ±1, ±2)

Usage (inside Docker container):
    python src/replication_scale_b.py

    # Custom paths:
    python src/replication_scale_b.py \
        --qwen outputs/semantic/narrative_dna.json \
        --nemotron outputs/semantic_nemotron/narrative_dna.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from project_config import (
    DIM_GROUPS,
    DIM_LABELS,
    DIMS_EXCLUDED,
    DIMS_ORDERED_ALL,
    PROJECT_ROOT,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

QWEN_PATH = PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json"
NEMOTRON_PATH = PROJECT_ROOT / "outputs" / "semantic_nemotron" / "narrative_dna.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(path: Path) -> dict[int, dict[str, int]]:
    """
    Load narrative DNA scores from a JSON file.

    Returns {chapter_number: {dimension: score}}.
    Handles both v1 (score as int) and v2 (score as {"score": int, "evidence": str}).
    """
    with open(path) as f:
        data = json.load(f)

    result = {}
    for entry in data["chapters"]:
        ch_num = entry["chapter"]
        scores = {}
        for dim, val in entry["scores"].items():
            if isinstance(val, dict):
                scores[dim] = val["score"]
            else:
                scores[dim] = int(val)
        result[ch_num] = scores

    return result


def align_scores(
    scores_a: dict[int, dict[str, int]],
    scores_b: dict[int, dict[str, int]],
    dims: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Align two score dicts on common chapters and return as matrices.

    Returns:
        (matrix_a, matrix_b, common_chapters) where matrices are (N, D).
    """
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not common:
        raise ValueError("No common chapters between the two score sets!")

    mat_a = np.array([[scores_a[ch][d] for d in dims] for ch in common], dtype=np.float64)
    mat_b = np.array([[scores_b[ch][d] for d in dims] for ch in common], dtype=np.float64)

    return mat_a, mat_b, common


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def per_dimension_analysis(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    dims: list[str],
) -> list[dict]:
    """Compute per-dimension correlation and agreement metrics."""
    results = []
    for i, dim in enumerate(dims):
        col_a = mat_a[:, i]
        col_b = mat_b[:, i]

        # Correlations (handle constant columns gracefully)
        if col_a.std() == 0 or col_b.std() == 0:
            spearman_r, spearman_p = float("nan"), float("nan")
            pearson_r, pearson_p = float("nan"), float("nan")
        else:
            spearman_r, spearman_p = stats.spearmanr(col_a, col_b)
            pearson_r, pearson_p = stats.pearsonr(col_a, col_b)

        # Quadratic weighted Cohen's kappa (ordinal agreement, chance-corrected)
        qwk = cohen_kappa_score(
            col_a.astype(int), col_b.astype(int),
            weights="quadratic", labels=list(range(1, 11)),
        )

        # Agreement
        diff = np.abs(col_a - col_b)
        mae = diff.mean()
        within_1 = (diff <= 1).mean() * 100
        within_2 = (diff <= 2).mean() * 100
        exact_match = (diff == 0).mean() * 100

        # Mean scores
        mean_a = col_a.mean()
        mean_b = col_b.mean()
        bias = mean_b - mean_a  # positive = Nemotron scores higher

        results.append({
            "dimension": dim,
            "label": DIM_LABELS.get(dim, dim),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "qwk": qwk,
            "mae": mae,
            "within_1_pct": within_1,
            "within_2_pct": within_2,
            "exact_match_pct": exact_match,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "bias": bias,
        })

    return results


def overall_analysis(mat_a: np.ndarray, mat_b: np.ndarray) -> dict:
    """Compute overall matrix-level agreement metrics."""
    # Flatten for overall correlation
    flat_a = mat_a.flatten()
    flat_b = mat_b.flatten()

    spearman_r, spearman_p = stats.spearmanr(flat_a, flat_b)
    pearson_r, pearson_p = stats.pearsonr(flat_a, flat_b)

    diff = np.abs(flat_a - flat_b)

    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mae": diff.mean(),
        "within_1_pct": (diff <= 1).mean() * 100,
        "within_2_pct": (diff <= 2).mean() * 100,
        "exact_match_pct": (diff == 0).mean() * 100,
        "max_diff": int(diff.max()),
        "n_chapters": mat_a.shape[0],
        "n_dimensions": mat_a.shape[1],
        "n_scores": len(flat_a),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    dim_results: list[dict],
    overall: dict,
    common_chapters: list[int],
    missing_a: set[int],
    missing_b: set[int],
):
    """Print a formatted comparison report."""
    print()
    print("=" * 75)
    print("SCALE B — INTER-RATER RELIABILITY (Qwen 3.5 27B vs Nemotron 70B)")
    print("=" * 75)
    print()

    # Coverage
    print(f"Common chapters scored by both: {len(common_chapters)}/155")
    if missing_a:
        print(f"  Missing from Qwen:    Ch. {sorted(missing_a)}")
    if missing_b:
        print(f"  Missing from Nemotron: Ch. {sorted(missing_b)}")
    print()

    # Overall
    print("─── OVERALL ───")
    print(f"  Total scores compared: {overall['n_scores']} "
          f"({overall['n_chapters']} ch × {overall['n_dimensions']} dims)")
    print(f"  Spearman ρ:  {overall['spearman_r']:.3f}  (p={overall['spearman_p']:.2e})")
    print(f"  Pearson r:   {overall['pearson_r']:.3f}  (p={overall['pearson_p']:.2e})")
    print(f"  MAE:         {overall['mae']:.2f}")
    print(f"  Exact match: {overall['exact_match_pct']:.1f}%")
    print(f"  Within ±1:   {overall['within_1_pct']:.1f}%")
    print(f"  Within ±2:   {overall['within_2_pct']:.1f}%")
    print(f"  Max |diff|:  {overall['max_diff']}")
    print()

    # Per-dimension table
    print("─── PER-DIMENSION ───")
    header = f"  {'Dimension':<18} {'ρ':>6} {'κw':>6} {'MAE':>5} {'±1':>5} {'±2':>5} {'Bias':>6}"
    print(header)
    print(f"  {'─' * 16}  {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6}")

    for group_name, group_dims in DIM_GROUPS.items():
        print(f"  [{group_name}]")
        group_results = [r for r in dim_results if r["dimension"] in group_dims]
        group_results.sort(key=lambda r: group_dims.index(r["dimension"]))
        for r in group_results:
            excluded = r["dimension"] in DIMS_EXCLUDED
            rho_str = f"{r['spearman_r']:>5.2f}" if not np.isnan(r["spearman_r"]) else "  N/A"
            rho_marker = "!" if (not np.isnan(r["spearman_r"]) and r["spearman_r"] < 0.5) else " "
            qwk_str = f"{r['qwk']:>5.2f}" if not np.isnan(r["qwk"]) else "  N/A"
            excl_tag = " ✗EXCL" if excluded else ""
            print(f"  {r['label']:<18} {rho_str}{rho_marker} "
                  f"{qwk_str} {r['mae']:>5.2f} "
                  f"{r['within_1_pct']:>4.0f}% {r['within_2_pct']:>4.0f}% "
                  f"{r['bias']:>+5.2f}{excl_tag}")
        print()

    # Summary interpretation (using ρ for ranking, κw for chance-corrected reliability)
    n_all = len(dim_results)
    valid = [r for r in dim_results if not np.isnan(r["spearman_r"])]
    validated = [r for r in valid if r["dimension"] not in DIMS_EXCLUDED]
    strong = [r for r in valid if r["spearman_r"] >= 0.7]
    moderate = [r for r in valid if 0.5 <= r["spearman_r"] < 0.7]
    weak = [r for r in valid if r["spearman_r"] < 0.5]

    print("─── INTERPRETATION ───")
    print(f"  Strong agreement (ρ ≥ 0.7):   {len(strong)}/{n_all} dimensions")
    if strong:
        print(f"    {', '.join(r['label'] for r in strong)}")
    print(f"  Moderate agreement (0.5–0.7):  {len(moderate)}/{n_all} dimensions")
    if moderate:
        print(f"    {', '.join(r['label'] for r in moderate)}")
    print(f"  Weak agreement (ρ < 0.5):      {len(weak)}/{n_all} dimensions")
    if weak:
        print(f"    {', '.join(r['label'] for r in weak)}")
    if DIMS_EXCLUDED:
        print(f"\n  Excluded from downstream analysis: {', '.join(sorted(DIMS_EXCLUDED))}")
    print()

    # Weighted kappa summary (Landis & Koch thresholds)
    valid_kw = [r for r in dim_results if not np.isnan(r["qwk"])]
    validated_kw = [r for r in valid_kw if r["dimension"] not in DIMS_EXCLUDED]
    mean_kw_all = np.mean([r["qwk"] for r in valid_kw]) if valid_kw else float("nan")
    mean_kw_val = np.mean([r["qwk"] for r in validated_kw]) if validated_kw else float("nan")
    mean_rho_all = np.mean([r["spearman_r"] for r in valid])
    mean_rho_val = np.mean([r["spearman_r"] for r in validated])
    print(f"  All {n_all}D — mean ρ: {mean_rho_all:.3f}, mean κw: {mean_kw_all:.3f}")
    print(f"  Validated {len(validated)}D — mean ρ: {mean_rho_val:.3f}, mean κw: {mean_kw_val:.3f}")
    print("    (Landis & Koch: >0.80 = almost perfect, 0.61–0.80 = substantial,")
    print("     0.41–0.60 = moderate, 0.21–0.40 = fair, <0.20 = slight)")
    print()

    # Mean bias
    biases = [r["bias"] for r in dim_results]
    mean_bias = np.mean(biases)
    print(f"  Mean bias (Nemotron − Qwen): {mean_bias:+.2f}")
    if mean_bias > 0.5:
        print("    → Nemotron tends to score higher (more generous)")
    elif mean_bias < -0.5:
        print("    → Nemotron tends to score lower (more conservative)")
    else:
        print("    → No systematic scoring bias")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scale B inter-rater reliability: Qwen vs Nemotron"
    )
    parser.add_argument(
        "--qwen", default=str(QWEN_PATH),
        help=f"Qwen narrative DNA JSON (default: {QWEN_PATH.relative_to(PROJECT_ROOT)})"
    )
    parser.add_argument(
        "--nemotron", default=str(NEMOTRON_PATH),
        help=f"Nemotron narrative DNA JSON (default: {NEMOTRON_PATH.relative_to(PROJECT_ROOT)})"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results JSON to this path"
    )
    args = parser.parse_args()

    qwen_path = Path(args.qwen)
    nemotron_path = Path(args.nemotron)

    print("Loading Qwen scores:", qwen_path.name)
    scores_qwen = load_scores(qwen_path)
    print(f"  {len(scores_qwen)} chapters")

    print("Loading Nemotron scores:", nemotron_path.name)
    scores_nemotron = load_scores(nemotron_path)
    print(f"  {len(scores_nemotron)} chapters")

    # Align
    all_chapters = set(range(1, 156))
    missing_qwen = all_chapters - set(scores_qwen.keys())
    missing_nemotron = all_chapters - set(scores_nemotron.keys())

    mat_qwen, mat_nem, common = align_scores(scores_qwen, scores_nemotron, DIMS_ORDERED_ALL)

    # Analyze (all 20 dimensions — includes excluded ones for completeness)
    dim_results = per_dimension_analysis(mat_qwen, mat_nem, DIMS_ORDERED_ALL)
    overall = overall_analysis(mat_qwen, mat_nem)

    # Report
    print_report(dim_results, overall, common, missing_qwen, missing_nemotron)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "outputs" / "semantic_nemotron" / "inter_rater_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model_a": "Qwen/Qwen3.5-27B-FP8",
        "model_b": "RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
        "n_common_chapters": len(common),
        "overall": overall,
        "per_dimension": dim_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
