#!/usr/bin/env python3
"""
Scale B: Semantic Profiling ("Narrative DNA") extraction.

Uses an LLM via vLLM to score each chapter of Rayuela on 20 semantic
dimensions, producing a structured feature vector per chapter.

Usage (inside Docker container):
    python src/semantic_extraction.py

    # Extract specific chapters only (e.g., for CHECKPOINT B1):
    python src/semantic_extraction.py --chapters 1 36 68

    # Use a different API endpoint (e.g., running outside Docker):
    python src/semantic_extraction.py --api-base http://localhost:8000/v1

Input:  data/rayuela_raw.json
        prompts/semantic_extraction_v1.txt
Output: outputs/semantic/narrative_dna.json
        outputs/semantic/narrative_dna_vectors.npy
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "rayuela_raw.json"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "semantic_extraction_v1.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "semantic"

VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"

# The 20 semantic dimensions in canonical order (must match prompt)
DIMENSIONS = [
    "existential_questioning",
    "art_and_aesthetics",
    "everyday_mundanity",
    "death_and_mortality",
    "love_and_desire",
    "emotional_intensity",
    "humor_and_irony",
    "melancholy_and_nostalgia",
    "tension_and_anxiety",
    "oliveira_centrality",
    "la_maga_presence",
    "character_density",
    "interpersonal_conflict",
    "interiority",
    "dialogue_density",
    "metafiction",
    "temporal_clarity",
    "spatial_grounding",
    "language_experimentation",
    "intertextual_density",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chapters(data_path: Path) -> list[dict]:
    """Load chapter data from rayuela_raw.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chapters"]


def load_system_prompt(prompt_path: Path) -> str:
    """Load the system prompt template."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_chapter(
    client: OpenAI,
    system_prompt: str,
    chapter: dict,
    model_name: str,
    with_evidence: bool = False,
    max_retries: int = 2,
) -> dict | None:
    """
    Score a single chapter on 20 dimensions via the vLLM API.

    When with_evidence=True, each dimension returns {"score": int, "evidence": str}.
    Otherwise, returns {dim: int}.

    Returns the parsed scores dict, or None on failure.
    """
    user_message = (
        f"Chapter {chapter['number']} — Section: {chapter['section']}\n\n"
        f"{chapter['text']}"
    )

    # Evidence mode needs more output tokens for the justification strings
    max_tokens = 2048 if with_evidence else 512

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "narrative_dna",
                        "schema": _json_schema(with_evidence=with_evidence),
                    },
                },
            )

            raw = response.choices[0].message.content
            result = json.loads(raw)
            scores = validate_scores(result, chapter["number"], with_evidence=with_evidence)

            if scores is not None:
                return scores

            # Validation failed — retry
            if attempt < max_retries:
                print(f"retry {attempt + 1}...", end=" ", flush=True)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}", end=" ", flush=True)
            if attempt < max_retries:
                print(f"retry {attempt + 1}...", end=" ", flush=True)
        except Exception as e:
            print(f"API error: {e}", end=" ", flush=True)
            if attempt < max_retries:
                print(f"retry {attempt + 1}...", end=" ", flush=True)
                time.sleep(2)

    return None


def _json_schema(with_evidence: bool = False) -> dict:
    """
    JSON schema for vLLM's guided decoding.

    This forces the model to produce valid JSON matching our exact structure,
    eliminating parse failures.

    When with_evidence=True, each dimension produces {"score": int, "evidence": str}
    instead of just an int.
    """
    if with_evidence:
        score_prop = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 10},
                "evidence": {"type": "string"},
            },
            "required": ["score", "evidence"],
        }
    else:
        score_prop = {"type": "integer", "minimum": 1, "maximum": 10}

    return {
        "type": "object",
        "properties": {
            "chapter": {"type": "integer"},
            "scores": {
                "type": "object",
                "properties": {dim: score_prop for dim in DIMENSIONS},
                "required": DIMENSIONS,
            },
        },
        "required": ["chapter", "scores"],
    }


def validate_scores(result: dict, chapter_number: int, with_evidence: bool = False) -> dict | None:
    """
    Validate that the result has all 20 dimensions with scores in [1, 10].

    When with_evidence=True, expects {"dim": {"score": int, "evidence": str}}.
    Returns a clean dict in the same format, or None on validation failure.
    """
    # Handle both {"scores": {...}} and flat {...} formats
    scores = result.get("scores", result)

    missing = [d for d in DIMENSIONS if d not in scores]
    if missing:
        print(f"WARNING Ch.{chapter_number}: missing {missing}", end=" ")
        return None

    clean = {}
    for dim in DIMENSIONS:
        val = scores[dim]

        if with_evidence:
            # Expect {"score": int, "evidence": str}
            if not isinstance(val, dict) or "score" not in val:
                print(f"WARNING Ch.{chapter_number}: bad format {dim}={val}", end=" ")
                return None
            score_val = val["score"]
            if not isinstance(score_val, (int, float)):
                print(f"WARNING Ch.{chapter_number}: non-numeric {dim}={score_val}", end=" ")
                return None
            clean[dim] = {
                "score": max(1, min(10, int(round(score_val)))),
                "evidence": val.get("evidence", ""),
            }
        else:
            if not isinstance(val, (int, float)):
                print(f"WARNING Ch.{chapter_number}: non-numeric {dim}={val}", end=" ")
                return None
            clean[dim] = max(1, min(10, int(round(val))))

    return clean


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _get_score(val) -> int:
    """Extract integer score from either int or {"score": int, "evidence": str}."""
    if isinstance(val, dict):
        return val["score"]
    return val


def save_results(results: list[dict], output_dir: Path):
    """Save results as JSON and numpy array."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON with full metadata
    json_path = output_dir / "narrative_dna.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dimensions": DIMENSIONS, "chapters": results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Numpy matrix (N x 20) for downstream analysis — always integer scores
    if results:
        vectors = np.array(
            [[_get_score(r["scores"][dim]) for dim in DIMENSIONS] for r in results],
            dtype=np.float32,
        )
        npy_path = output_dir / "narrative_dna_vectors.npy"
        np.save(npy_path, vectors)
        return vectors

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scale B: Narrative DNA extraction via LLM"
    )
    parser.add_argument(
        "--chapters", nargs="+", type=int,
        help="Extract specific chapters only (e.g., --chapters 1 36 68)"
    )
    parser.add_argument(
        "--api-base", default=VLLM_API_BASE,
        help=f"vLLM API base URL (default: {VLLM_API_BASE})"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--prompt", default=str(DEFAULT_PROMPT_PATH),
        help=f"System prompt file (default: {DEFAULT_PROMPT_PATH.name})"
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--with-evidence", action="store_true",
        help="Extract evidence strings alongside scores (v2 format)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing narrative_dna.json (skip already-extracted chapters)"
    )
    args = parser.parse_args()

    # Resolve paths
    PROMPT_PATH = Path(args.prompt)
    OUTPUT_DIR = Path(args.output_dir)

    print("=" * 60)
    print("Scale B — Narrative DNA extraction")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"API:   {args.api_base}")
    print(f"Dimensions: {len(DIMENSIONS)}")
    print()

    # Load data
    chapters = load_chapters(DATA_PATH)
    system_prompt = load_system_prompt(PROMPT_PATH)

    if args.with_evidence:
        print(f"Mode: scores + evidence (v2)")
    else:
        print(f"Mode: scores only (v1)")
    print(f"Loaded {len(chapters)} chapters")
    print(f"System prompt: {len(system_prompt)} chars")

    # Filter to specific chapters if requested
    if args.chapters:
        chapters = [ch for ch in chapters if ch["number"] in args.chapters]
        print(f"Filtering to {len(chapters)} chapters: {[ch['number'] for ch in chapters]}")

    # Resume support: skip already-extracted chapters
    existing_results = []
    extracted_numbers = set()
    if args.resume:
        existing_path = OUTPUT_DIR / "narrative_dna.json"
        if existing_path.exists():
            with open(existing_path) as f:
                existing_data = json.load(f)
            existing_results = existing_data.get("chapters", [])
            extracted_numbers = {r["chapter"] for r in existing_results}
            chapters = [ch for ch in chapters if ch["number"] not in extracted_numbers]
            print(f"Resuming: {len(extracted_numbers)} already done, {len(chapters)} remaining")

    if not chapters:
        print("Nothing to extract!")
        return

    print()

    # Connect to vLLM
    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    # Verify connection
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"vLLM models available: {available}")
        if args.model not in available:
            print(f"ERROR: {args.model} not found on vLLM server.")
            print(f"  Available models: {available}")
            print(f"  Refusing to fall back — model identity is critical for replication.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR connecting to vLLM at {args.api_base}: {e}")
        print("Is the vLLM server running? Start it with:")
        print("  docker compose --profile llm up vllm")
        sys.exit(1)

    print()

    # Extract chapter by chapter
    results = list(existing_results)  # start with any resumed results
    failures = []
    total_time = 0

    for i, ch in enumerate(chapters):
        label = f"[{i + 1}/{len(chapters)}]"
        print(f"{label} Ch.{ch['number']:3d} ({ch['token_count']:,} words)...", end=" ", flush=True)

        t0 = time.time()
        scores = extract_chapter(client, system_prompt, ch, args.model,
                                 with_evidence=args.with_evidence)
        elapsed = time.time() - t0
        total_time += elapsed

        if scores is not None:
            entry = {
                "chapter": ch["number"],
                "section": ch["section"],
                "is_expendable": ch["is_expendable"],
                "scores": scores,
            }
            results.append(entry)
            print(f"OK ({elapsed:.1f}s)")
        else:
            failures.append(ch["number"])
            print(f"FAILED ({elapsed:.1f}s)")

        # Save incrementally every 10 chapters
        if (i + 1) % 10 == 0:
            save_results(results, OUTPUT_DIR)
            print(f"  [checkpoint saved: {len(results)} chapters]")

    # Final save
    vectors = save_results(results, OUTPUT_DIR)

    # Summary
    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Succeeded: {len(results)}/{len(results) + len(failures)}")
    if failures:
        print(f"  Failed: {failures}")
    print(f"  Total time: {total_time:.0f}s ({total_time / max(len(chapters), 1):.1f}s/chapter)")
    if vectors is not None:
        print(f"  Vector matrix: {vectors.shape}")
    print(f"  JSON: {OUTPUT_DIR / 'narrative_dna.json'}")
    print(f"  Numpy: {OUTPUT_DIR / 'narrative_dna_vectors.npy'}")


if __name__ == "__main__":
    main()
