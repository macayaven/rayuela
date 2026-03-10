#!/usr/bin/env python3
"""
Diagnostic: distribution of pairwise cosine similarities across all chapters.

Answers the question: is there enough spread in the embedding space for
meaningful structure to emerge, or are all chapters effectively identical?

Usage (inside Docker container):
    python src/embedding_diagnostics.py
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy"
META_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_metadata.json"


def main():
    embeddings = np.load(EMB_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)

    n = len(embeddings)
    print(f"Embeddings: {n} chapters, {embeddings.shape[1]} dimensions")
    print()

    # -------------------------------------------------------------------
    # Full pairwise similarity matrix (cosine sim = dot product for L2-normed)
    # -------------------------------------------------------------------
    sim_matrix = embeddings @ embeddings.T

    # Extract upper triangle (exclude self-similarities on diagonal)
    upper = sim_matrix[np.triu_indices(n, k=1)]

    print("Pairwise cosine similarity distribution (all chapter pairs):")
    print(f"  Pairs:  {len(upper):,}")
    print(f"  Min:    {upper.min():.4f}")
    print(f"  Max:    {upper.max():.4f}")
    print(f"  Mean:   {upper.mean():.4f}")
    print(f"  Median: {np.median(upper):.4f}")
    print(f"  Std:    {upper.std():.4f}")
    print(f"  Range:  {upper.max() - upper.min():.4f}")
    print()

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("Percentiles:")
    for p in percentiles:
        print(f"  {p:3d}th:  {np.percentile(upper, p):.4f}")
    print()

    # -------------------------------------------------------------------
    # By section: do sections show internal cohesion?
    # -------------------------------------------------------------------
    sections = {}
    for meta in metadata:
        sec = meta["section"]
        if sec not in sections:
            sections[sec] = []
        sections[sec].append(meta["index"])

    print("Within-section vs. between-section similarity:")
    print("─" * 55)

    within_all = []
    between_all = []

    for sec_name, indices in sections.items():
        idx = np.array(indices)
        # Within-section pairs
        sec_sims = sim_matrix[np.ix_(idx, idx)]
        sec_upper = sec_sims[np.triu_indices(len(idx), k=1)]
        within_all.extend(sec_upper)
        print(f"  {sec_name}")
        print(f"    Chapters: {len(idx)}, Pairs: {len(sec_upper)}")
        print(f"    Mean sim: {sec_upper.mean():.4f}  (std: {sec_upper.std():.4f})")

    # Between-section pairs
    all_sections = list(sections.keys())
    for i, sec_a in enumerate(all_sections):
        for sec_b in all_sections[i + 1 :]:
            idx_a = np.array(sections[sec_a])
            idx_b = np.array(sections[sec_b])
            cross_sims = sim_matrix[np.ix_(idx_a, idx_b)].flatten()
            between_all.extend(cross_sims)

    within_arr = np.array(within_all)
    between_arr = np.array(between_all)

    print()
    print(f"  WITHIN sections:  mean={within_arr.mean():.4f}  std={within_arr.std():.4f}")
    print(f"  BETWEEN sections: mean={between_arr.mean():.4f}  std={between_arr.std():.4f}")
    print(f"  Difference:       {within_arr.mean() - between_arr.mean():.4f}")
    print()

    if within_arr.mean() > between_arr.mean():
        print("  ✓ Within-section similarity > between-section similarity")
        print("    → Sections have detectable textural identity.")
    else:
        print("  ✗ No section-level clustering detected in texture space.")

    # -------------------------------------------------------------------
    # Expendable vs. non-expendable chapters
    # -------------------------------------------------------------------
    exp_idx = np.array([m["index"] for m in metadata if m["is_expendable"]])
    non_idx = np.array([m["index"] for m in metadata if not m["is_expendable"]])

    exp_sims = sim_matrix[np.ix_(exp_idx, exp_idx)]
    non_sims = sim_matrix[np.ix_(non_idx, non_idx)]
    cross_sims = sim_matrix[np.ix_(non_idx, exp_idx)]

    exp_upper = exp_sims[np.triu_indices(len(exp_idx), k=1)]
    non_upper = non_sims[np.triu_indices(len(non_idx), k=1)]

    print()
    print("Expendable (57-155) vs. Non-expendable (1-56):")
    print("─" * 55)
    print(f"  Within non-expendable (1-56):   mean={non_upper.mean():.4f}")
    print(f"  Within expendable (57-155):     mean={exp_upper.mean():.4f}")
    print(f"  Cross (non-exp ↔ exp):          mean={cross_sims.mean():.4f}")

    # -------------------------------------------------------------------
    # Most similar and most dissimilar pairs
    # -------------------------------------------------------------------
    print()
    print("5 most SIMILAR chapter pairs:")
    print("─" * 55)
    flat_idx = np.argsort(upper)[::-1][:5]
    row_idx, col_idx = np.triu_indices(n, k=1)
    for rank, fi in enumerate(flat_idx, 1):
        r, c = row_idx[fi], col_idx[fi]
        ch_r = metadata[r]["number"]
        ch_c = metadata[c]["number"]
        print(f"  {rank}. Ch.{ch_r:3d} ↔ Ch.{ch_c:3d}  sim={upper[fi]:.4f}"
              f"  [{metadata[r]['section'][:15]} / {metadata[c]['section'][:15]}]")

    print()
    print("5 most DISSIMILAR chapter pairs:")
    print("─" * 55)
    flat_idx = np.argsort(upper)[:5]
    for rank, fi in enumerate(flat_idx, 1):
        r, c = row_idx[fi], col_idx[fi]
        ch_r = metadata[r]["number"]
        ch_c = metadata[c]["number"]
        print(f"  {rank}. Ch.{ch_r:3d} ↔ Ch.{ch_c:3d}  sim={upper[fi]:.4f}"
              f"  [{metadata[r]['section'][:15]} / {metadata[c]['section'][:15]}]")


if __name__ == "__main__":
    main()
