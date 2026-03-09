#!/usr/bin/env python3
"""
Preview of Scale C: consecutive-pair similarity along both reading paths.

For each reading path (linear 1→56 and hopscotch/Tablero), compute the
cosine similarity between each consecutive pair of chapters. This measures
"trajectory smoothness" — how much the texture changes at each step.

Usage (inside Docker container):
    python src/trajectory_preview.py
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy"
DATA_PATH = PROJECT_ROOT / "data" / "rayuela_raw.json"


def consecutive_similarities(embeddings: np.ndarray, path: list[int]) -> np.ndarray:
    """
    Compute cosine similarity between each consecutive pair in a reading path.

    Args:
        embeddings: (155, 1024) L2-normalized chapter embeddings
        path: list of chapter numbers (1-indexed) defining the reading order

    Returns:
        Array of similarities: [sim(path[0],path[1]), sim(path[1],path[2]), ...]
    """
    sims = []
    for i in range(len(path) - 1):
        # Convert 1-indexed chapter number to 0-indexed array position
        idx_a = path[i] - 1
        idx_b = path[i + 1] - 1
        sim = float(np.dot(embeddings[idx_a], embeddings[idx_b]))
        sims.append(sim)
    return np.array(sims)


def main():
    embeddings = np.load(EMB_PATH)
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    linear_path = data["reading_paths"]["linear"]       # Ch. 1-56
    hopscotch_path = data["reading_paths"]["hopscotch"]  # Tablero sequence

    print("=" * 65)
    print("Trajectory smoothness: consecutive-pair cosine similarity")
    print("=" * 65)
    print()

    # --- Linear path ---
    linear_sims = consecutive_similarities(embeddings, linear_path)

    print(f"LINEAR PATH (Ch. 1 → 56, {len(linear_path)} chapters, "
          f"{len(linear_sims)} steps)")
    print(f"  Mean similarity:    {linear_sims.mean():.4f}")
    print(f"  Std:                {linear_sims.std():.4f}")
    print(f"  Min (roughest step):{linear_sims.min():.4f}")
    print(f"  Max (smoothest):    {linear_sims.max():.4f}")
    print()

    # Roughest jumps in linear path
    rough_idx = np.argsort(linear_sims)[:5]
    print("  5 roughest transitions (biggest texture shifts):")
    for rank, idx in enumerate(rough_idx, 1):
        ch_from = linear_path[idx]
        ch_to = linear_path[idx + 1]
        print(f"    {rank}. Ch.{ch_from:3d} → Ch.{ch_to:3d}  sim={linear_sims[idx]:.4f}")
    print()

    # --- Hopscotch path ---
    hopscotch_sims = consecutive_similarities(embeddings, hopscotch_path)

    print(f"HOPSCOTCH PATH (Tablero, {len(hopscotch_path)} chapters, "
          f"{len(hopscotch_sims)} steps)")
    print(f"  Mean similarity:    {hopscotch_sims.mean():.4f}")
    print(f"  Std:                {hopscotch_sims.std():.4f}")
    print(f"  Min (roughest step):{hopscotch_sims.min():.4f}")
    print(f"  Max (smoothest):    {hopscotch_sims.max():.4f}")
    print()

    # Roughest jumps in hopscotch path
    rough_idx = np.argsort(hopscotch_sims)[:5]
    print("  5 roughest transitions (biggest texture shifts):")
    for rank, idx in enumerate(rough_idx, 1):
        ch_from = hopscotch_path[idx]
        ch_to = hopscotch_path[idx + 1]
        print(f"    {rank}. Ch.{ch_from:3d} → Ch.{ch_to:3d}  sim={hopscotch_sims[idx]:.4f}")
    print()

    # --- Comparison ---
    print("=" * 65)
    print("COMPARISON")
    print("=" * 65)
    print()
    print(f"  {'Metric':<30} {'Linear':>10} {'Hopscotch':>10}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 10}")
    print(f"  {'Mean consecutive similarity':<30} {linear_sims.mean():>10.4f} "
          f"{hopscotch_sims.mean():>10.4f}")
    print(f"  {'Std (consistency)':<30} {linear_sims.std():>10.4f} "
          f"{hopscotch_sims.std():>10.4f}")
    print(f"  {'Min (worst jump)':<30} {linear_sims.min():>10.4f} "
          f"{hopscotch_sims.min():>10.4f}")
    print(f"  {'Max (smoothest step)':<30} {linear_sims.max():>10.4f} "
          f"{hopscotch_sims.max():>10.4f}")
    print()

    diff = linear_sims.mean() - hopscotch_sims.mean()
    if diff > 0:
        smoother = "LINEAR"
        rougher = "HOPSCOTCH"
    else:
        smoother = "HOPSCOTCH"
        rougher = "LINEAR"
        diff = -diff

    print(f"  The {smoother} path is texturally smoother by {diff:.4f}")
    print(f"  The {rougher} path makes bigger texture jumps on average.")
    print()
    print("  Interpretation:")
    print("  - A smoother linear path would mean Cortázar arranged chapters")
    print("    1-56 in a texturally flowing sequence.")
    print("  - A smoother hopscotch path would mean the Tablero was designed")
    print("    for textural continuity (unlikely — more likely semantic).")
    print("  - If hopscotch is rougher, Cortázar may have prioritized")
    print("    *semantic* coherence (Scale B) over surface-level flow.")

    # --- The 131↔58 infinite loop ---
    # The hopscotch path ends with 131 → 58 → 131 → 58 → ...
    if 131 in hopscotch_path and 58 in hopscotch_path:
        sim_loop = float(np.dot(embeddings[130], embeddings[57]))
        print()
        print(f"  The infinite loop: Ch.131 ↔ Ch.58  sim={sim_loop:.4f}")
        print("  (The Tablero ends by bouncing between these two chapters forever)")
        if sim_loop > hopscotch_sims.mean():
            print("  → These chapters are MORE similar than the average hop — a tight loop.")
        else:
            print("  → These chapters are LESS similar than the average hop — a jarring loop.")


if __name__ == "__main__":
    main()
