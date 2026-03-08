#!/usr/bin/env python3
"""
Convergent validity: replicate Scale A with nvidia/llama-nemotron-embed-1b-v2.

This script embeds all 155 chapters with a completely different embedding model
and re-runs the trajectory permutation test. If both E5 and Nemotron detect
hopscotch smoothness above random, the finding is model-independent.

Architecture comparison:
  E5 (original):    Encoder-only (XLM-RoBERTa variant), 560M params, 1024 dims
  Nemotron (this):  Decoder-made-bidirectional (Llama 3.2), 1.2B params, 2048 dims

Different company, different architecture, different training pipeline, different
embedding space. Maximum independence for convergent validity.

Usage (inside Docker container):
    python src/replication_nemotron.py
"""

import json
import time

import numpy as np

from project_config import (
    DATA_PATH,
    EMB_A_PATH,
    LINEAR_ORDER,
    N_PERMS,
    PROJECT_ROOT,
    RNG_SEED,
    TABLERO,
    DistanceMetric,
    continuity_corrected_percentile,
    get_all_chapters,
)
from project_config import (
    z_score as compute_z_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "nvidia/llama-nemotron-embed-1b-v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings_nemotron"


# ---------------------------------------------------------------------------
# Phase 1: Embed all 155 chapters
# ---------------------------------------------------------------------------

def embed_all_chapters(chapters: list[dict]) -> np.ndarray:
    """
    Embed all chapters with Nemotron Embed.

    Uses standard model.encode() with L2 normalization.
    The model's native 2048 dimensions are preserved (no truncation).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    print(f"  Max seq length: {model.max_seq_length}")
    print(f"  Device: {model.device}")
    print()

    texts = [ch["text"] for ch in chapters]

    print(f"Embedding {len(texts)} chapters...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=4,  # slightly smaller batch for 1.2B model
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0

    print(f"  Shape: {embeddings.shape}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / len(texts):.2f}s/chapter)")
    print(f"  L2 norms: min={np.linalg.norm(embeddings, axis=1).min():.4f}, "
          f"max={np.linalg.norm(embeddings, axis=1).max():.4f}")
    print()

    return embeddings


# ---------------------------------------------------------------------------
# Phase 2: Trajectory permutation test (reused from trajectory_baselines.py)
# ---------------------------------------------------------------------------

def mean_consecutive_similarity(embeddings: np.ndarray, path: list[int]) -> float:
    """Mean cosine similarity between consecutive chapters in a reading path."""
    indices = [ch - 1 for ch in path]
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
    """Distribution of mean consecutive similarities from random orderings."""
    chapters_arr = np.array(chapters)
    results = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(chapters_arr).tolist()
        results[i] = mean_consecutive_similarity(embeddings, shuffled)
    return results


def run_trajectory_analysis(embeddings: np.ndarray):
    """Run the same permutation test as trajectory_baselines.py."""
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 65)
    print("TRAJECTORY PERMUTATION TEST — Nemotron Embed")
    print(f"({N_PERMS:,} permutations per path)")
    print("=" * 65)
    print()

    # --- Linear path ---
    linear_obs = mean_consecutive_similarity(embeddings, LINEAR_ORDER)
    print(f"LINEAR PATH (Ch. 1→56, {len(LINEAR_ORDER)} chapters)")
    print(f"  Observed similarity: {linear_obs:.6f}")

    linear_null = permutation_distribution(
        embeddings, list(LINEAR_ORDER), N_PERMS, rng
    )
    linear_z = compute_z_score(linear_obs, linear_null, DistanceMetric.COSINE)
    linear_pct = continuity_corrected_percentile(
        linear_obs, linear_null, DistanceMetric.COSINE
    )

    print(f"  Null: mean={linear_null.mean():.6f}, std={linear_null.std():.6f}")
    print(f"  Z-score: {linear_z:+.2f}σ")
    print(f"  Percentile: {linear_pct:.1f}%")
    print()

    # --- Hopscotch path ---
    hop_obs = mean_consecutive_similarity(embeddings, TABLERO)
    print(f"HOPSCOTCH PATH (Tablero, {len(TABLERO)} steps)")
    print(f"  Observed similarity: {hop_obs:.6f}")

    all_chapters = get_all_chapters()
    hop_null = permutation_distribution(
        embeddings, all_chapters, N_PERMS, rng
    )
    hop_z = compute_z_score(hop_obs, hop_null, DistanceMetric.COSINE)
    hop_pct = continuity_corrected_percentile(
        hop_obs, hop_null, DistanceMetric.COSINE
    )

    print(f"  Null: mean={hop_null.mean():.6f}, std={hop_null.std():.6f}")
    print(f"  Z-score: {hop_z:+.2f}σ")
    print(f"  Percentile: {hop_pct:.1f}%")
    print()

    return {
        "linear_z": linear_z,
        "linear_pct": linear_pct,
        "hopscotch_z": hop_z,
        "hopscotch_pct": hop_pct,
    }


# ---------------------------------------------------------------------------
# Phase 3: Compare with E5 (if available)
# ---------------------------------------------------------------------------

def compare_with_e5(nemotron_results: dict):
    """Load E5 results and compare z-scores."""
    if not EMB_A_PATH.exists():
        print("E5 embeddings not found — skipping comparison")
        return

    e5_emb = np.load(EMB_A_PATH)
    rng = np.random.default_rng(RNG_SEED)

    # Recompute E5 z-scores with same methodology
    e5_linear = mean_consecutive_similarity(e5_emb, LINEAR_ORDER)
    e5_linear_null = permutation_distribution(
        e5_emb, list(LINEAR_ORDER), N_PERMS, rng
    )
    e5_linear_z = compute_z_score(e5_linear, e5_linear_null, DistanceMetric.COSINE)

    e5_hop = mean_consecutive_similarity(e5_emb, TABLERO)
    e5_hop_null = permutation_distribution(
        e5_emb, get_all_chapters(), N_PERMS, rng
    )
    e5_hop_z = compute_z_score(e5_hop, e5_hop_null, DistanceMetric.COSINE)

    print("=" * 65)
    print("CONVERGENT VALIDITY — E5 vs Nemotron")
    print("=" * 65)
    print()
    print(f"  {'':20} {'E5 (1024d)':>12} {'Nemotron (2048d)':>16} {'Agree?':>8}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 16} {'─' * 8}")

    for label, e5_z, nem_z in [
        ("Linear z-score", e5_linear_z, nemotron_results["linear_z"]),
        ("Hopscotch z-score", e5_hop_z, nemotron_results["hopscotch_z"]),
    ]:
        agree = "YES" if (e5_z > 0) == (nem_z > 0) else "NO"
        print(f"  {label:<20} {e5_z:>+12.2f}σ {nem_z:>+16.2f}σ {agree:>8}")

    print()

    # Interpretation
    both_linear = e5_linear_z > 2 and nemotron_results["linear_z"] > 2
    both_hop = e5_hop_z > 2 and nemotron_results["hopscotch_z"] > 2

    if both_linear:
        print("  ✓ LINEAR: Both models detect intentional ordering (>2σ)")
    elif e5_linear_z > 0 and nemotron_results["linear_z"] > 0:
        print("  ~ LINEAR: Both positive but not both >2σ")
    else:
        print("  ✗ LINEAR: Models disagree on direction")

    if both_hop:
        print("  ✓ HOPSCOTCH: Both models detect intentional ordering (>2σ)")
        print("    → FINDING IS MODEL-INDEPENDENT")
    elif e5_hop_z > 0 and nemotron_results["hopscotch_z"] > 0:
        print("  ~ HOPSCOTCH: Both positive but not both >2σ")
    else:
        print("  ✗ HOPSCOTCH: Models disagree — finding may be model-specific")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("CONVERGENT VALIDITY REPLICATION")
    print(f"Model: {MODEL_NAME}")
    print("=" * 65)
    print()

    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)
    chapters = data["chapters"]
    print(f"Loaded {len(chapters)} chapters from {DATA_PATH.name}")
    print()

    # Phase 1: Embed
    embeddings = embed_all_chapters(chapters)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = OUTPUT_DIR / "chapter_embeddings_nemotron.npy"
    np.save(emb_path, embeddings)
    print(f"Saved: {emb_path}")

    meta = [
        {"index": i, "number": ch["number"], "section": ch["section"]}
        for i, ch in enumerate(chapters)
    ]
    meta_path = OUTPUT_DIR / "chapter_metadata_nemotron.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")
    print()

    # Phase 2: Trajectory analysis
    results = run_trajectory_analysis(embeddings)

    # Phase 3: Compare with E5
    compare_with_e5(results)

    # Save results summary
    summary = {
        "model": MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "n_chapters": len(chapters),
        "n_permutations": N_PERMS,
        "rng_seed": RNG_SEED,
        **results,
    }
    summary_path = OUTPUT_DIR / "replication_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {summary_path}")


if __name__ == "__main__":
    main()
