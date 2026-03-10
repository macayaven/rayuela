#!/usr/bin/env python3
"""
Run Scale B (narrative DNA) on all cleaned corpus works and compute
per-author semantic profiles.

Usage (inside Docker container):
    python src/corpus_semantic.py

    # Single work only:
    python src/corpus_semantic.py --work borges_ficciones

    # Use Nemotron 70B instead of default Qwen:
    python src/corpus_semantic.py --model RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic

    # Skip extraction, just compute profiles from existing JSON files:
    python src/corpus_semantic.py --profiles-only

Input:  data/corpus/{work_id}_clean.json
Output: outputs/corpus/{work_id}/narrative_dna.json
        outputs/corpus/{work_id}/narrative_dna_vectors.npy
        outputs/corpus/author_profiles_semantic.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import (
    CORPUS_DIR,
    CORPUS_OUTPUT_DIR,
    CORPUS_WORKS,
    DIMS_ORDERED,
    PROJECT_ROOT,
)

SEMANTIC_SCRIPT = PROJECT_ROOT / "src" / "semantic_extraction.py"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "semantic_extraction_v1.txt"
DEFAULT_MODEL = "Qwen/Qwen3.5-27B-FP8"


def run_semantic(work_id: str, model: str, prompt: str) -> bool:
    """Run semantic_extraction.py on a single corpus work."""
    input_path = CORPUS_DIR / f"{work_id}_clean.json"
    output_dir = CORPUS_OUTPUT_DIR / work_id

    if not input_path.exists():
        print(f"  SKIP: {input_path} not found")
        return False

    # Check if already extracted
    json_path = output_dir / "narrative_dna.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        existing = len(data.get("chapters", []))
        # Load expected count
        with open(input_path, encoding="utf-8") as f:
            expected = len(json.load(f)["chapters"])
        if existing >= expected:
            print(f"  SKIP: {json_path} already has {existing}/{expected} chapters")
            return True
        print(f"  RESUMING: {existing}/{expected} chapters done")

    print(f"  Running semantic extraction on {work_id}...")
    cmd = [
        sys.executable, str(SEMANTIC_SCRIPT),
        "--input", str(input_path),
        "--model", model,
        "--prompt", prompt,
        "--output-dir", str(output_dir),
        "--resume",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return False

    lines = result.stdout.strip().split("\n")
    for line in lines[-5:]:
        print(f"    {line}")
    return True


def compute_author_profiles(include_rayuela: bool = False) -> dict:
    """Compute per-author mean + std semantic profiles (19D)."""
    author_works = {}
    for work_id, (author, _title) in CORPUS_WORKS.items():
        author_works.setdefault(author, []).append(work_id)

    # Optional cross-corpus comparison helper; default corpus profiles stay corpus-only.
    rayuela_dna = PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna_vectors.npy"
    if include_rayuela and rayuela_dna.exists():
        author_works.setdefault("Cortázar", []).append("__rayuela__")

    dims = list(DIMS_ORDERED)

    profiles = {}
    for author, work_ids in sorted(author_works.items()):
        all_vectors = []
        work_details = []

        for wid in work_ids:
            if wid == "__rayuela__":
                npy_path = rayuela_dna
                label = "Rayuela"
                # Rayuela's .npy is 20D — need to filter to 19D
                vectors = np.load(npy_path)
                from project_config import filter_excluded_dims
                vectors = filter_excluded_dims(vectors)
            else:
                npy_path = CORPUS_OUTPUT_DIR / wid / "narrative_dna_vectors.npy"
                label = CORPUS_WORKS[wid][1]
                if not npy_path.exists():
                    continue
                vectors = np.load(npy_path)
                # Corpus extractions may be 20D if temporal_clarity included
                if vectors.shape[1] == 20:
                    from project_config import filter_excluded_dims
                    vectors = filter_excluded_dims(vectors)

            all_vectors.append(vectors)
            work_details.append({
                "work_id": wid,
                "title": label,
                "n_chapters": vectors.shape[0],
            })

        if not all_vectors:
            continue

        combined = np.vstack(all_vectors)
        profile = {
            "author": author,
            "n_chapters_total": combined.shape[0],
            "n_works": len(work_details),
            "works": work_details,
            "dimensions": dims,
            "mean": {name: float(np.mean(combined[:, j]))
                     for j, name in enumerate(dims)},
            "std": {name: float(np.std(combined[:, j]))
                    for j, name in enumerate(dims)},
        }
        profiles[author] = profile
        print(f"  {author}: {combined.shape[0]} chapters across {len(work_details)} works")

    return {"dimensions": dims, "profiles": profiles}


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic extraction on all corpus works + compute author profiles"
    )
    parser.add_argument("--work", help="Process single work only")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="Prompt file")
    parser.add_argument("--profiles-only", action="store_true",
                        help="Skip extraction, just compute profiles")
    parser.add_argument(
        "--include-rayuela",
        action="store_true",
        help="Include Rayuela in the Cortázar aggregate instead of keeping corpus-only profiles",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Corpus Scale B — Semantic Extraction + Author Profiles")
    print(f"Model: {args.model}")
    print("=" * 65)
    print()

    if not args.profiles_only:
        works = {args.work: CORPUS_WORKS[args.work]} if args.work else CORPUS_WORKS
        successes = 0
        for work_id in works:
            author, title = CORPUS_WORKS[work_id]
            print(f"[{author}] {title}")
            if run_semantic(work_id, args.model, args.prompt):
                successes += 1
            print()

        print(f"Extraction: {successes}/{len(works)} works succeeded")
        print()

    # Compute author profiles
    print("Computing per-author semantic profiles (19D)...")
    profiles = compute_author_profiles(include_rayuela=args.include_rayuela)

    if profiles:
        CORPUS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        profile_path = CORPUS_OUTPUT_DIR / "author_profiles_semantic.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        print(f"\nProfiles saved: {profile_path}")
        print(f"  Authors: {len(profiles['profiles'])}")


if __name__ == "__main__":
    main()
