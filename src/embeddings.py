#!/usr/bin/env python3
"""
Scale A: Micro-textural embeddings of Rayuela chapters.

First pass — one embedding per chapter using intfloat/multilingual-e5-large-instruct.
The model captures surface-level texture (vocabulary, syntax, rhythm) as a
1024-dimensional vector per chapter.

Usage (inside Docker container):
    python src/embeddings.py

Input:  data/rayuela_raw.json
Output: outputs/embeddings/chapter_embeddings.npy
        outputs/embeddings/chapter_metadata.json
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

# The instruction steers the embedding toward stylistic/thematic features.
# E5-instruct format: "Instruct: {task}\nQuery: {text}"
# sentence-transformers prepends `prompt` to each text automatically.
EMBED_INSTRUCTION = (
    "Instruct: Represent this literary passage for stylistic and thematic clustering\n"
    "Query: "
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chapters(data_path: Path) -> list[dict]:
    """Load chapter data from rayuela_raw.json."""
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["chapters"]


# ---------------------------------------------------------------------------
# Embedding — First pass (one vector per chapter)
# ---------------------------------------------------------------------------

def embed_chapters(chapters: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Embed each chapter as a single vector (first pass).

    The model truncates chapters longer than its max sequence length (512 tokens)
    automatically — this is intentional for the first pass, since the opening
    of a chapter sets its texture.

    Returns:
        np.ndarray of shape (155, 1024), L2-normalized.
    """
    texts = [ch["text"] for ch in chapters]

    embeddings = model.encode(
        texts,
        prompt=EMBED_INSTRUCTION,
        show_progress_bar=True,
        batch_size=8,
        normalize_embeddings=True,  # cosine similarity = dot product
    )

    return embeddings


# ---------------------------------------------------------------------------
# Sanity check — Carlos's CHECKPOINT A1 predictions
# ---------------------------------------------------------------------------

def checkpoint_a1(embeddings: np.ndarray):
    """
    Compare cosine similarities for the three chapters Carlos read
    (Ch. 1, 8, 36) against his texture predictions.

    Carlos predicted Ch. 36 sits between Ch. 1 and Ch. 8 —
    sharing description with Ch. 1 and intensity with Ch. 8.
    """
    # 0-indexed: Ch.1 → [0], Ch.8 → [7], Ch.36 → [35]
    ch1, ch8, ch36 = embeddings[0], embeddings[7], embeddings[35]

    # Embeddings are L2-normalized, so cosine similarity = dot product
    sim_1_8 = float(np.dot(ch1, ch8))
    sim_1_36 = float(np.dot(ch1, ch36))
    sim_8_36 = float(np.dot(ch8, ch36))

    print("CHECKPOINT A1 — Carlos's texture predictions")
    print("─" * 50)
    print(f"  Ch.1  ↔ Ch.8:   {sim_1_8:.4f}")
    print(f"  Ch.1  ↔ Ch.36:  {sim_1_36:.4f}")
    print(f"  Ch.8  ↔ Ch.36:  {sim_8_36:.4f}")
    print()

    # Interpret the triangle
    pairs = [("Ch.1↔Ch.8", sim_1_8), ("Ch.1↔Ch.36", sim_1_36), ("Ch.8↔Ch.36", sim_8_36)]
    closest = max(pairs, key=lambda p: p[1])
    farthest = min(pairs, key=lambda p: p[1])
    print(f"  Most similar:  {closest[0]} ({closest[1]:.4f})")
    print(f"  Least similar: {farthest[0]} ({farthest[1]:.4f})")
    print()
    print("  Carlos predicted Ch.36 between Ch.1 and Ch.8.")
    print("  If correct, Ch.1↔Ch.36 and Ch.8↔Ch.36 should both be")
    print("  higher than Ch.1↔Ch.8.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Scale A — Chapter-level embeddings (first pass)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Instruction: {EMBED_INSTRUCTION.strip()!r}")
    print()

    # Load data
    chapters = load_chapters(DATA_PATH)
    print(f"Loaded {len(chapters)} chapters")

    # Token count summary
    counts = [ch["token_count"] for ch in chapters]
    print(f"  Word counts: min={min(counts)}, max={max(counts)}, "
          f"median={sorted(counts)[len(counts)//2]}")
    print()

    # Load model (downloads ~2.2 GB on first run, cached in ~/.cache/huggingface)
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length: {model.max_seq_length}")
    print()

    # Embed all chapters
    print("Embedding 155 chapters...")
    embeddings = embed_chapters(chapters, model)
    print(f"  Result shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}")
    print()

    # Save embeddings
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = OUTPUT_DIR / "chapter_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Embeddings saved: {emb_path}")

    # Save chapter metadata (for downstream analysis — links indices to chapters)
    metadata = [
        {
            "index": i,
            "number": ch["number"],
            "section": ch["section"],
            "token_count": ch["token_count"],
            "is_expendable": ch["is_expendable"],
        }
        for i, ch in enumerate(chapters)
    ]
    meta_path = OUTPUT_DIR / "chapter_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved:  {meta_path}")
    print()

    # Sanity check against Carlos's predictions
    checkpoint_a1(embeddings)


if __name__ == "__main__":
    main()
