#!/usr/bin/env python3
"""
Scale A — Second pass: sliding-window embeddings for long chapters.

The first pass (embeddings.py) produced one 1024-dim vector per chapter,
but the model truncates at 512 tokens. For long chapters, we only saw
the opening ~350 words. This second pass slides a 512-token window
across the full text to detect *internal* texture shifts.

Usage (inside Docker container):
    python src/embeddings_windowed.py

Input:  data/rayuela_raw.json
Output: outputs/embeddings/window_embeddings.npz
        outputs/embeddings/window_metadata.json
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "rayuela_raw.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings"

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# Same instruction as first pass — ensures embeddings live in the same space
EMBED_INSTRUCTION = (
    "Instruct: Represent this literary passage for stylistic and thematic clustering\n"
    "Query: "
)

# Sliding window parameters
WINDOW_SIZE = 512    # tokens (model tokenizer units)
STRIDE = 256         # 50% overlap — each token appears in ~2 windows
MIN_WORDS = 512      # only window chapters longer than this (word count)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chapters(data_path: Path) -> list[dict]:
    """Load chapter data from rayuela_raw.json."""
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["chapters"]


# ---------------------------------------------------------------------------
# Core: tokenize a chapter and split into overlapping windows
# ---------------------------------------------------------------------------

def create_token_windows(
    text: str,
    tokenizer,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[str]:
    """
    Split text into overlapping windows of `window_size` tokens.

    We tokenize with the model's own tokenizer so that "512 tokens" means
    exactly what the model expects — no guessing about word-to-token ratios.

    Args:
        text: raw chapter text
        tokenizer: the model's tokenizer (from sentence-transformers)
        window_size: number of tokens per window
        stride: how many tokens to advance between windows

    Returns:
        List of text strings, one per window. Each string is the decoded
        text of that window's tokens. We decode back to text because
        sentence-transformers.encode() expects strings, not token IDs.
    """
    # Tokenize without special tokens — we just want the content tokens.
    # sentence-transformers adds its own special tokens during encode().
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)

    if total_tokens <= window_size:
        # Chapter fits in one window — return as-is
        return [text]

    windows = []
    start = 0
    while start < total_tokens:
        end = min(start + window_size, total_tokens)
        window_ids = token_ids[start:end]

        # Decode back to text so sentence-transformers can re-encode it
        # with the proper instruction prefix
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        windows.append(window_text)

        # Stop if we've reached the end
        if end == total_tokens:
            break

        start += stride

    return windows


# ---------------------------------------------------------------------------
# Embedding + internal drift analysis
# ---------------------------------------------------------------------------

def embed_windows(
    windows: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Embed a list of text windows. Returns (n_windows, 1024), L2-normalized.
    """
    return model.encode(
        windows,
        prompt=EMBED_INSTRUCTION,
        show_progress_bar=False,
        batch_size=8,
        normalize_embeddings=True,
    )


def compute_chapter_drift(window_embeddings: np.ndarray) -> dict:
    """
    Measure how much the texture changes within a chapter.

    For a chapter with N windows, we compute:
    - consecutive_sims: cosine similarity between adjacent windows (N-1 values)
    - mean_drift: 1 - mean(consecutive_sims). Higher = more internal variation.
    - max_jump: the biggest single-step drop in similarity (where the texture
      shifts most abruptly). This is our "register shift detector."
    - max_jump_position: where in the chapter (0.0 = start, 1.0 = end) the
      biggest jump occurs.
    - overall_span: similarity between first and last window. Low = the chapter
      ends in a very different texture than it started.

    Returns dict with all metrics.
    """
    n = len(window_embeddings)
    if n < 2:
        return {
            "n_windows": n,
            "consecutive_sims": [],
            "mean_drift": 0.0,
            "max_jump": 0.0,
            "max_jump_position": 0.0,
            "overall_span": 1.0,
        }

    # Consecutive similarities (embeddings are L2-normalized → dot = cosine)
    consecutive_sims = []
    for i in range(n - 1):
        sim = float(np.dot(window_embeddings[i], window_embeddings[i + 1]))
        consecutive_sims.append(sim)

    sims_arr = np.array(consecutive_sims)

    # The "jump" at each step is how much similarity DROPS (1 - sim)
    jumps = 1.0 - sims_arr
    max_jump_idx = int(np.argmax(jumps))

    return {
        "n_windows": n,
        "consecutive_sims": consecutive_sims,
        "mean_drift": float(1.0 - sims_arr.mean()),
        "max_jump": float(jumps[max_jump_idx]),
        "max_jump_position": max_jump_idx / (n - 1),  # normalized 0..1
        "overall_span": float(np.dot(window_embeddings[0], window_embeddings[-1])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Scale A — Sliding-window embeddings (second pass)")
    print("=" * 65)
    print(f"Model:       {MODEL_NAME}")
    print(f"Window size: {WINDOW_SIZE} tokens")
    print(f"Stride:      {STRIDE} tokens ({STRIDE/WINDOW_SIZE:.0%} of window)")
    print(f"Threshold:   chapters with >{MIN_WORDS} words")
    print()

    # Load data
    chapters = load_chapters(DATA_PATH)
    print(f"Loaded {len(chapters)} chapters")

    # Which chapters need windowing?
    windowed_chapters = [
        ch for ch in chapters if ch["token_count"] > MIN_WORDS
    ]
    skipped = len(chapters) - len(windowed_chapters)
    print(f"  Chapters above {MIN_WORDS} words: {len(windowed_chapters)}")
    print(f"  Chapters skipped (short):    {skipped}")
    print()

    # Load model
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length: {model.max_seq_length}")
    print()

    # Process each chapter
    all_window_embeddings = {}  # chapter_number -> np.ndarray
    all_drift_reports = []
    total_windows = 0

    for i, ch in enumerate(windowed_chapters):
        ch_num = ch["number"]
        text = ch["text"]

        # Create token windows
        windows = create_token_windows(text, tokenizer)
        n_win = len(windows)
        total_windows += n_win

        # Embed all windows for this chapter
        win_emb = embed_windows(windows, model)
        all_window_embeddings[ch_num] = win_emb

        # Measure internal drift
        drift = compute_chapter_drift(win_emb)
        drift["chapter_number"] = ch_num
        drift["section"] = ch["section"]
        drift["word_count"] = ch["token_count"]
        drift["is_expendable"] = ch["is_expendable"]
        all_drift_reports.append(drift)

        if (i + 1) % 10 == 0 or (i + 1) == len(windowed_chapters):
            print(f"  Processed {i + 1}/{len(windowed_chapters)} chapters "
                  f"({total_windows} windows so far)")

    print()
    print(f"Total windows embedded: {total_windows}")
    print()

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save window embeddings as compressed .npz (one array per chapter)
    npz_data = {f"ch{num}": emb for num, emb in all_window_embeddings.items()}
    npz_path = OUTPUT_DIR / "window_embeddings.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Window embeddings saved: {npz_path}")

    # Save drift metadata (strip the raw consecutive_sims to keep JSON small)
    meta_for_json = []
    for d in all_drift_reports:
        entry = {k: v for k, v in d.items() if k != "consecutive_sims"}
        entry["n_windows"] = d["n_windows"]
        meta_for_json.append(entry)

    meta_path = OUTPUT_DIR / "window_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_for_json, f, ensure_ascii=False, indent=2)
    print(f"Window metadata saved:  {meta_path}")

    # Also save the full consecutive similarities for later visualization
    sims_data = {
        f"ch{d['chapter_number']}": d["consecutive_sims"]
        for d in all_drift_reports
    }
    sims_path = OUTPUT_DIR / "window_consecutive_sims.json"
    with open(sims_path, "w", encoding="utf-8") as f:
        json.dump(sims_data, f, indent=2)
    print(f"Consecutive sims saved: {sims_path}")
    print()

    # -----------------------------------------------------------------------
    # Report: chapters ranked by internal drift
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("INTERNAL DRIFT REPORT — which chapters shift texture most?")
    print("=" * 65)
    print()

    # Sort by mean drift (highest first = most internal variation)
    sorted_drift = sorted(all_drift_reports, key=lambda d: d["mean_drift"], reverse=True)

    print(f"  {'Ch':>4}  {'Words':>6}  {'Windows':>4}  {'Mean Drift':>10}  "
          f"{'Max Jump':>9}  {'Jump @':>7}  {'Span':>6}  Section")
    print(f"  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*10}  {'─'*9}  {'─'*7}  {'─'*6}  {'─'*20}")

    for d in sorted_drift[:20]:
        jump_pct = f"{d['max_jump_position']:.0%}"
        sec_short = d["section"][:20]
        print(f"  {d['chapter_number']:>4}  {d['word_count']:>6}  {d['n_windows']:>4}  "
              f"{d['mean_drift']:>10.4f}  {d['max_jump']:>9.4f}  {jump_pct:>7}  "
              f"{d['overall_span']:>6.3f}  {sec_short}")

    print()

    # -----------------------------------------------------------------------
    # Special cases: chapters Carlos and the framework call out
    # -----------------------------------------------------------------------
    special = {34: "interleaved two-column", 36: "Carlos's prediction (no shift)",
               28: "longest chapter", 56: "section II finale",
               23: "Club de la Serpiente", 41: "Berthe Trépat concert"}
    print("SPOTLIGHT — key chapters:")
    print("─" * 65)

    drift_by_ch = {d["chapter_number"]: d for d in all_drift_reports}

    for ch_num, label in special.items():
        if ch_num in drift_by_ch:
            d = drift_by_ch[ch_num]
            print(f"  Ch.{ch_num} ({label}):")
            print(f"    {d['n_windows']} windows, mean drift={d['mean_drift']:.4f}, "
                  f"max jump={d['max_jump']:.4f} at {d['max_jump_position']:.0%}, "
                  f"span={d['overall_span']:.3f}")
        else:
            print(f"  Ch.{ch_num} ({label}): below threshold, skipped")
    print()

    # -----------------------------------------------------------------------
    # Carlos's prediction: Ch. 36 shows no special transition
    # -----------------------------------------------------------------------
    if 36 in drift_by_ch:
        d36 = drift_by_ch[36]
        all_drifts = [d["mean_drift"] for d in all_drift_reports]
        rank = sorted(all_drifts, reverse=True).index(d36["mean_drift"]) + 1
        pct = rank / len(all_drifts) * 100
        print("CARLOS'S PREDICTION — Ch. 36 texture consistency")
        print("─" * 65)
        print(f"  Mean drift: {d36['mean_drift']:.4f} "
              f"(rank {rank}/{len(all_drifts)}, top {pct:.0f}%)")
        print(f"  Max jump:   {d36['max_jump']:.4f} at position {d36['max_jump_position']:.0%}")

        median_drift = float(np.median(all_drifts))
        if d36["mean_drift"] < median_drift:
            print(f"  ✓ CONFIRMED: Ch. 36 has LESS internal drift than median ({median_drift:.4f})")
            print("    → Cortázar's voice stays consistent across register changes.")
        else:
            print(f"  ✗ SURPRISED: Ch. 36 has MORE drift than median ({median_drift:.4f})")
            print(
                "    → The model detects the register shifts that Carlos "
                "expected to be invisible."
            )


if __name__ == "__main__":
    main()
