#!/usr/bin/env python3
"""
Scale B': LLM-perceived stylistic features.

Same infrastructure as semantic_extraction.py (Scale B) but with a prompt
focused on FORM (how the text is written) rather than CONTENT (what it says).

The correlation between B' (LLM style perception) and A' (computational
stylometrics) tells us how well the LLM perceives measurable style features.

Usage (inside Docker container):
    python src/stylistic_extraction.py

    # Extract specific chapters only:
    python src/stylistic_extraction.py --chapters 1 36 68

    # Resume interrupted run:
    python src/stylistic_extraction.py --resume
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
PROMPT_PATH = PROJECT_ROOT / "prompts" / "stylistic_extraction_v1.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stylistic"

VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"

# The 12 stylistic dimensions (must match prompt)
DIMENSIONS = [
    "sentence_complexity",
    "vocabulary_register",
    "punctuation_expressiveness",
    "prose_rhythm",
    "descriptive_density",
    "dialogue_vs_narration",
    "code_switching_intensity",
    "paragraph_density",
    "narrative_distance",
    "repetition_and_pattern",
    "typographic_experimentation",
    "syntactic_variety",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chapters(data_path: Path) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chapters"]


def load_system_prompt(prompt_path: Path) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# JSON schema for guided decoding
# ---------------------------------------------------------------------------

def json_schema() -> dict:
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


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_chapter(
    client: OpenAI,
    system_prompt: str,
    chapter: dict,
    model_name: str,
    max_retries: int = 2,
) -> dict | None:
    user_message = (
        f"Chapter {chapter['number']} — Section: {chapter['section']}\n\n"
        f"{chapter['text']}"
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=512,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "stylistic_profile",
                        "schema": json_schema(),
                    },
                },
            )

            raw = response.choices[0].message.content
            result = json.loads(raw)
            scores = result.get("scores", result)

            # Validate all dimensions present and in range
            missing = [d for d in DIMENSIONS if d not in scores]
            if missing:
                print(f"missing {missing}", end=" ")
                if attempt < max_retries:
                    print(f"retry {attempt + 1}...", end=" ", flush=True)
                continue

            clean = {}
            for dim in DIMENSIONS:
                val = scores[dim]
                if not isinstance(val, (int, float)):
                    print(f"non-numeric {dim}={val}", end=" ")
                    break
                clean[dim] = max(1, min(10, int(round(val))))
            else:
                return clean

            if attempt < max_retries:
                print(f"retry {attempt + 1}...", end=" ", flush=True)

        except json.JSONDecodeError as e:
            print(f"JSON error: {e}", end=" ")
            if attempt < max_retries:
                print(f"retry {attempt + 1}...", end=" ", flush=True)
        except Exception as e:
            print(f"API error: {e}", end=" ")
            if attempt < max_retries:
                print(f"retry {attempt + 1}...", end=" ", flush=True)
                time.sleep(2)

    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "stylistic_dna.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dimensions": DIMENSIONS, "chapters": results},
            f, ensure_ascii=False, indent=2,
        )

    if results:
        vectors = np.array(
            [[r["scores"][dim] for dim in DIMENSIONS] for r in results],
            dtype=np.float32,
        )
        npy_path = output_dir / "stylistic_dna_vectors.npy"
        np.save(npy_path, vectors)
        return vectors

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scale B': Stylistic profile extraction via LLM"
    )
    parser.add_argument("--chapters", nargs="+", type=int)
    parser.add_argument("--api-base", default=VLLM_API_BASE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Scale B' — Stylistic Profile extraction (LLM-perceived form)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"API:   {args.api_base}")
    print(f"Dimensions: {len(DIMENSIONS)}")
    print()

    chapters = load_chapters(DATA_PATH)
    system_prompt = load_system_prompt(PROMPT_PATH)

    print(f"Loaded {len(chapters)} chapters")
    print(f"System prompt: {len(system_prompt)} chars")

    if args.chapters:
        chapters = [ch for ch in chapters if ch["number"] in args.chapters]
        print(f"Filtering to {len(chapters)} chapters: {[ch['number'] for ch in chapters]}")

    # Resume support
    existing_results = []
    extracted_numbers = set()
    if args.resume:
        existing_path = OUTPUT_DIR / "stylistic_dna.json"
        if existing_path.exists():
            with open(existing_path) as f:
                existing_data = json.load(f)
            existing_results = existing_data.get("chapters", [])
            extracted_numbers = {r["chapter"] for r in existing_results}
            chapters = [ch for ch in chapters if ch["number"] not in extracted_numbers]
            print(f"Resuming: {len(extracted_numbers)} done, {len(chapters)} remaining")

    if not chapters:
        print("Nothing to extract!")
        return

    print()

    # Connect to vLLM
    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"vLLM models available: {available}")
        if args.model not in available:
            print(f"WARNING: {args.model} not found. Available: {available}")
            if available:
                args.model = available[0]
                print(f"Using: {args.model}")
    except Exception as e:
        print(f"ERROR connecting to vLLM at {args.api_base}: {e}")
        print("Is the vLLM server running?")
        sys.exit(1)

    print()

    results = list(existing_results)
    failures = []
    total_time = 0

    for i, ch in enumerate(chapters):
        label = f"[{i + 1}/{len(chapters)}]"
        print(f"{label} Ch.{ch['number']:3d} ({ch['token_count']:,} words)...", end=" ", flush=True)

        t0 = time.time()
        scores = extract_chapter(client, system_prompt, ch, args.model)
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

    vectors = save_results(results, OUTPUT_DIR)

    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Succeeded: {len(results)}/{len(results) + len(failures)}")
    if failures:
        print(f"  Failed: {failures}")
    print(f"  Total time: {total_time:.0f}s ({total_time / max(len(chapters), 1):.1f}s/chapter)")
    if vectors is not None:
        print(f"  Vector matrix: {vectors.shape}")
    print(f"  JSON: {OUTPUT_DIR / 'stylistic_dna.json'}")
    print(f"  Numpy: {OUTPUT_DIR / 'stylistic_dna_vectors.npy'}")


if __name__ == "__main__":
    main()
