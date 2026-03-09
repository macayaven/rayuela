#!/usr/bin/env python3
"""
Run Scale A' (stylometrics) on all cleaned corpus works and compute
per-author profiles.

Usage (inside Docker container):
    python src/corpus_stylometrics.py

    # Single work only:
    python src/corpus_stylometrics.py --work borges_ficciones

    # Skip extraction, just compute profiles from existing .npy files:
    python src/corpus_stylometrics.py --profiles-only

Input:  data/corpus/{work_id}_clean.json
Output: outputs/corpus/{work_id}/chapter_stylometrics.npy
        outputs/corpus/{work_id}/chapter_stylometrics_metadata.json
        outputs/corpus/author_profiles_stylo.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Import from project config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import (
    CORPUS_DIR,
    CORPUS_OUTPUT_DIR,
    CORPUS_WORKS,
    PROJECT_ROOT,
)

STYLOMETRICS_SCRIPT = PROJECT_ROOT / "src" / "stylometrics.py"


def _load_expected_chapter_count(input_path: Path) -> int:
    """Return the cleaned chapter count for one corpus work."""
    with open(input_path) as f:
        return len(json.load(f)["chapters"])


def run_stylometrics(work_id: str) -> bool:
    """Run stylometrics.py on a single corpus work. Returns True on success."""
    input_path = CORPUS_DIR / f"{work_id}_clean.json"
    output_dir = CORPUS_OUTPUT_DIR / work_id

    if not input_path.exists():
        print(f"  SKIP: {input_path} not found")
        return False

    # Check if already extracted
    npy_path = output_dir / "chapter_stylometrics.npy"
    meta_path = output_dir / "chapter_stylometrics_metadata.json"
    if npy_path.exists():
        expected = _load_expected_chapter_count(input_path)
        existing_rows = int(np.load(npy_path).shape[0])
        metadata_rows = None
        if meta_path.exists():
            with open(meta_path) as f:
                metadata_rows = int(json.load(f)["n_chapters"])

        if existing_rows == expected and metadata_rows == expected:
            print(f"  SKIP: {npy_path} already has {existing_rows}/{expected} chapters")
            return True

        print(
            f"  REBUILDING: stale stylometrics for {work_id} "
            f"({existing_rows}/{expected} rows, metadata={metadata_rows})"
        )

    print(f"  Running stylometrics on {work_id}...")
    result = subprocess.run(
        [sys.executable, str(STYLOMETRICS_SCRIPT),
         "--input", str(input_path),
         "--output-dir", str(output_dir)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return False

    # Print last few lines of output
    lines = result.stdout.strip().split("\n")
    for line in lines[-3:]:
        print(f"    {line}")
    return True


def compute_author_profiles(include_rayuela: bool = False) -> dict:
    """Compute per-author mean + std stylometric profiles."""
    # Group works by author
    author_works = {}
    for work_id, (author, _title) in CORPUS_WORKS.items():
        author_works.setdefault(author, []).append(work_id)

    # Optional cross-corpus comparison helper; default corpus profiles stay corpus-only.
    rayuela_stylo = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_stylometrics.npy"
    if include_rayuela and rayuela_stylo.exists():
        author_works.setdefault("Cortázar", []).append("__rayuela__")

    # Load feature names from any metadata file
    feature_names = None
    for work_id in CORPUS_WORKS:
        meta_path = CORPUS_OUTPUT_DIR / work_id / "chapter_stylometrics_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                feature_names = json.load(f)["feature_names"]
            break

    if feature_names is None:
        print("ERROR: No metadata files found. Run extraction first.")
        return {}

    profiles = {}
    for author, work_ids in sorted(author_works.items()):
        all_vectors = []
        work_details = []

        for wid in work_ids:
            if wid == "__rayuela__":
                npy_path = rayuela_stylo
                label = "Rayuela"
            else:
                npy_path = CORPUS_OUTPUT_DIR / wid / "chapter_stylometrics.npy"
                label = CORPUS_WORKS[wid][1]

            if not npy_path.exists():
                continue

            vectors = np.load(npy_path)
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
            "mean": {name: float(np.mean(combined[:, j]))
                     for j, name in enumerate(feature_names)},
            "std": {name: float(np.std(combined[:, j]))
                    for j, name in enumerate(feature_names)},
        }
        profiles[author] = profile
        print(f"  {author}: {combined.shape[0]} chapters across {len(work_details)} works")

    return {"feature_names": feature_names, "profiles": profiles}


def main():
    parser = argparse.ArgumentParser(
        description="Run stylometrics on all corpus works + compute author profiles"
    )
    parser.add_argument("--work", help="Process single work only (e.g., borges_ficciones)")
    parser.add_argument("--profiles-only", action="store_true",
                        help="Skip extraction, just compute profiles from existing .npy files")
    parser.add_argument(
        "--include-rayuela",
        action="store_true",
        help="Include Rayuela in the Cortázar aggregate instead of keeping corpus-only profiles",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Corpus Scale A' — Stylometric Extraction + Author Profiles")
    print("=" * 65)
    print()

    if not args.profiles_only:
        works = {args.work: CORPUS_WORKS[args.work]} if args.work else CORPUS_WORKS
        successes = 0
        for work_id in works:
            author, title = CORPUS_WORKS[work_id]
            print(f"[{author}] {title}")
            if run_stylometrics(work_id):
                successes += 1
            print()

        print(f"Extraction: {successes}/{len(works)} works succeeded")
        print()

    # Compute author profiles
    print("Computing per-author profiles...")
    profiles = compute_author_profiles(include_rayuela=args.include_rayuela)

    if profiles:
        CORPUS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        profile_path = CORPUS_OUTPUT_DIR / "author_profiles_stylo.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        print(f"\nProfiles saved: {profile_path}")
        print(f"  Authors: {len(profiles['profiles'])}")


if __name__ == "__main__":
    main()
