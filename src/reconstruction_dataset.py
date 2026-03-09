#!/usr/bin/env python3
"""
Leakage-safe dataset and pilot design for Part 3 reconstruction work.

Phase 3 turns the synchronized comparison corpus into deterministic window-level
experiment units, split manifests, target work envelopes, and explicit success
criteria without starting any generation or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project_config import CORPUS_DIR, CORPUS_OUTPUT_DIR, CORPUS_WORKS
from reconstruction_contract import DEFAULT_RECONSTRUCTION_SEED, ReconstructionPaths
from reconstruction_metrics import (
    MeasurementMatrix,
    ToleranceConfig,
    load_semantic_measurements,
    load_stylometric_measurements,
)

DEFAULT_SOURCE_WINDOWS_PATH = ReconstructionPaths().pilots_dir / "source_windows.json"
DEFAULT_TARGET_ENVELOPES_PATH = ReconstructionPaths().pilots_dir / "target_envelopes.json"
DEFAULT_SPLIT_MANIFEST_PATH = ReconstructionPaths().pilots_dir / "split_manifest.json"
DEFAULT_SUCCESS_CRITERIA_PATH = ReconstructionPaths().pilots_dir / "success_criteria.json"

SplitName = str


@dataclass(frozen=True)
class WindowRecord:
    """Deterministic reconstruction unit extracted from one chapter span."""

    window_id: str
    work_id: str
    author: str
    title: str
    chapter_number: int
    segment_id: str
    chapter_word_count: int
    word_start: int
    word_end: int
    word_count: int
    text: str
    stylometric_reference: dict[str, float]
    semantic_reference: dict[str, float]
    split: str | None = None

    def with_split(self, split: str) -> WindowRecord:
        """Return a split-annotated copy of the window."""
        return WindowRecord(
            window_id=self.window_id,
            work_id=self.work_id,
            author=self.author,
            title=self.title,
            chapter_number=self.chapter_number,
            segment_id=self.segment_id,
            chapter_word_count=self.chapter_word_count,
            word_start=self.word_start,
            word_end=self.word_end,
            word_count=self.word_count,
            text=self.text,
            stylometric_reference=self.stylometric_reference,
            semantic_reference=self.semantic_reference,
            split=split,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "window_id": self.window_id,
            "work_id": self.work_id,
            "author": self.author,
            "title": self.title,
            "chapter_number": self.chapter_number,
            "segment_id": self.segment_id,
            "chapter_word_count": self.chapter_word_count,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "word_count": self.word_count,
            "text": self.text,
            "stylometric_reference": self.stylometric_reference,
            "semantic_reference": self.semantic_reference,
            "split": self.split,
        }


@dataclass(frozen=True)
class SplitManifest:
    """Window-level split assignment with leakage audit metadata."""

    generated_at: str
    seed: int
    min_words: int
    max_words: int
    train_ratio: float
    val_ratio: float
    total_windows: int
    split_counts: dict[str, int]
    assignments: tuple[dict[str, Any], ...]
    leakage_issues: tuple[str, ...]

    def assignment_lookup(self) -> dict[str, str]:
        """Return window_id -> split lookup."""
        return {
            str(assignment["window_id"]): str(assignment["split"])
            for assignment in self.assignments
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "generated_at": self.generated_at,
            "seed": self.seed,
            "min_words": self.min_words,
            "max_words": self.max_words,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": 1.0 - self.train_ratio - self.val_ratio,
            "total_windows": self.total_windows,
            "split_counts": self.split_counts,
            "assignments": list(self.assignments),
            "leakage_issues": list(self.leakage_issues),
        }


@dataclass(frozen=True)
class TargetEnvelope:
    """Aggregated target work envelope for pilot style-shift experiments."""

    envelope_id: str
    work_id: str
    author: str
    title: str
    aggregation_rule: str
    provenance_window_ids: tuple[str, ...]
    provenance_segment_ids: tuple[str, ...]
    stylometric_target: dict[str, float]
    semantic_reference: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "envelope_id": self.envelope_id,
            "work_id": self.work_id,
            "author": self.author,
            "title": self.title,
            "aggregation_rule": self.aggregation_rule,
            "provenance_window_ids": list(self.provenance_window_ids),
            "provenance_segment_ids": list(self.provenance_segment_ids),
            "stylometric_target": self.stylometric_target,
            "semantic_reference": self.semantic_reference,
        }


@dataclass(frozen=True)
class SuccessCriteria:
    """Serialized tolerance bands and weighted objective for the pilot."""

    generated_at: str
    claim_language: str
    tolerances: dict[str, float]
    objective_weights: dict[str, float]
    lexical_guardrails: dict[str, float]
    minimum_pass_requirements: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "generated_at": self.generated_at,
            "claim_language": self.claim_language,
            "tolerances": self.tolerances,
            "objective_weights": self.objective_weights,
            "lexical_guardrails": self.lexical_guardrails,
            "minimum_pass_requirements": self.minimum_pass_requirements,
        }


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON payload from disk."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _token_spans(text: str) -> list[tuple[int, int]]:
    """Return character spans for non-whitespace token boundaries."""
    return [match.span() for match in re.finditer(r"\S+", text, flags=re.UNICODE)]


def _segment_vector_lookup(measurements: MeasurementMatrix) -> dict[str, dict[str, float]]:
    """Return segment_id -> ordered dimension mapping."""
    return {
        segment_id: {
            name: float(measurements.matrix[row_index, col_index])
            for col_index, name in enumerate(measurements.dimension_order)
        }
        for row_index, segment_id in enumerate(measurements.segment_ids)
    }


def _chapter_hash(segment_id: str, seed: int) -> str:
    """Return a deterministic hash key for chapter-level split assignment."""
    return hashlib.sha256(f"{seed}:{segment_id}".encode()).hexdigest()


def _compute_split_sizes(
    chapter_count: int, train_ratio: float, val_ratio: float
) -> dict[str, int]:
    """Return exact chapter counts per split while preserving determinism."""
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    raw = {name: chapter_count * ratio for name, ratio in ratios.items()}
    sizes = {name: int(value) for name, value in raw.items()}
    remainder = chapter_count - sum(sizes.values())
    ranked = sorted(
        ratios,
        key=lambda name: (raw[name] - sizes[name], ratios[name], name),
        reverse=True,
    )
    for name in ranked[:remainder]:
        sizes[name] += 1

    if chapter_count >= 3:
        for name in ("train", "val", "test"):
            if sizes[name] == 0:
                donor = max(
                    ("train", "val", "test"),
                    key=lambda candidate: (sizes[candidate], ratios[candidate], candidate),
                )
                if sizes[donor] <= 1:
                    continue
                sizes[donor] -= 1
                sizes[name] += 1

    return sizes


def extract_windows(
    *,
    corpus_dir: Path = CORPUS_DIR,
    corpus_output_dir: Path = CORPUS_OUTPUT_DIR,
    corpus_works: dict[str, tuple[str, str]] | None = None,
    min_words: int = 128,
    max_words: int = 256,
) -> list[WindowRecord]:
    """Extract deterministic non-overlapping windows with chapter-level metric references."""
    if min_words <= 0 or max_words < min_words:
        raise ValueError("word bounds must satisfy 0 < min_words <= max_words")

    resolved_corpus_works = CORPUS_WORKS if corpus_works is None else corpus_works
    stylometric_lookup = _segment_vector_lookup(
        load_stylometric_measurements(
            corpus_dir=corpus_dir,
            corpus_output_dir=corpus_output_dir,
            corpus_works=resolved_corpus_works,
        )
    )
    semantic_lookup = _segment_vector_lookup(
        load_semantic_measurements(
            corpus_dir=corpus_dir,
            corpus_output_dir=corpus_output_dir,
            corpus_works=resolved_corpus_works,
        )
    )

    windows: list[WindowRecord] = []
    for work_id in sorted(resolved_corpus_works):
        author, title = resolved_corpus_works[work_id]
        clean_payload = _load_json(corpus_dir / f"{work_id}_clean.json")
        for chapter in clean_payload["chapters"]:
            chapter_number = int(chapter["number"])
            text = str(chapter["text"])
            spans = _token_spans(text)
            chapter_word_count = len(spans)
            if chapter_word_count < min_words:
                continue

            segment_id = f"{work_id}:{chapter_number}"
            if segment_id not in stylometric_lookup or segment_id not in semantic_lookup:
                raise ValueError(f"missing chapter reference scores for {segment_id}")

            word_start = 0
            window_index = 0
            while word_start + min_words <= chapter_word_count:
                remaining = chapter_word_count - word_start
                word_end = chapter_word_count if remaining <= max_words else word_start + max_words
                start_char = spans[word_start][0]
                end_char = spans[word_end - 1][1]
                window_text = text[start_char:end_char]
                window_id = (
                    f"{work_id}:ch{chapter_number}:w{window_index}:"
                    f"{word_start}-{word_end}"
                )
                windows.append(
                    WindowRecord(
                        window_id=window_id,
                        work_id=work_id,
                        author=author,
                        title=title,
                        chapter_number=chapter_number,
                        segment_id=segment_id,
                        chapter_word_count=chapter_word_count,
                        word_start=word_start,
                        word_end=word_end,
                        word_count=word_end - word_start,
                        text=window_text,
                        stylometric_reference=stylometric_lookup[segment_id],
                        semantic_reference=semantic_lookup[segment_id],
                    )
                )
                window_index += 1
                word_start = word_end

    if not windows:
        raise ValueError("no eligible reconstruction windows were extracted")
    return windows


def _token_jaccard(left_text: str, right_text: str) -> float:
    """Return token-set Jaccard overlap for near-duplicate checks."""
    left = set(re.findall(r"\w+", left_text.lower(), flags=re.UNICODE))
    right = set(re.findall(r"\w+", right_text.lower(), flags=re.UNICODE))
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _window_token_set(text: str) -> frozenset[str]:
    """Return a normalized unique token set for near-duplicate checks."""
    return frozenset(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _prefix_length(token_count: int, threshold: float) -> int:
    """Return the deterministic prefix length for Jaccard candidate generation."""
    return max(1, token_count - math.ceil(threshold * token_count) + 1)


def _audit_leakage(
    windows: list[WindowRecord],
    assignment_lookup: dict[str, str],
    *,
    near_duplicate_threshold: float,
) -> tuple[str, ...]:
    """Audit chapter overlap and near-duplicate leakage across splits."""
    issues: list[str] = []
    by_segment: dict[str, set[str]] = {}

    for window in windows:
        split = assignment_lookup[window.window_id]
        by_segment.setdefault(window.segment_id, set()).add(split)

    for segment_id, splits in sorted(by_segment.items()):
        if len(splits) > 1:
            issues.append(f"{segment_id}: chapter assigned to multiple splits {sorted(splits)}")

    token_sets = {window.window_id: _window_token_set(window.text) for window in windows}
    token_frequency = Counter(
        token
        for token_set in token_sets.values()
        for token in token_set
    )
    ordered_tokens = {
        window.window_id: tuple(
            sorted(
                token_sets[window.window_id],
                key=lambda token: (token_frequency[token], token),
            )
        )
        for window in windows
    }
    prefix_index: dict[str, list[int]] = {}

    for left_index, left in enumerate(windows):
        left_split = assignment_lookup[left.window_id]
        left_tokens = token_sets[left.window_id]
        left_token_count = len(left_tokens)
        for right in windows[left_index + 1 :]:
            right_split = assignment_lookup[right.window_id]
            if left_split == right_split:
                continue
            if left.segment_id == right.segment_id:
                overlaps = not (
                    left.word_end <= right.word_start or right.word_end <= left.word_start
                )
                adjacent = left.word_end == right.word_start or right.word_end == left.word_start
                if overlaps or adjacent:
                    issues.append(
                        f"{left.window_id} and {right.window_id}: cross-split chapter overlap"
                    )

        candidates: set[int] = set()
        left_ordered_tokens = ordered_tokens[left.window_id]
        prefix_length = _prefix_length(len(left_ordered_tokens), near_duplicate_threshold)
        for token in left_ordered_tokens[:prefix_length]:
            candidates.update(prefix_index.get(token, []))
            prefix_index.setdefault(token, []).append(left_index)

        for candidate_index in candidates:
            right = windows[candidate_index]
            right_split = assignment_lookup[right.window_id]
            if left_split == right_split:
                continue

            right_tokens = token_sets[right.window_id]
            if not left_tokens or not right_tokens:
                continue

            smaller = min(left_token_count, len(right_tokens))
            larger = max(left_token_count, len(right_tokens))
            if float(smaller / larger) < near_duplicate_threshold:
                continue

            union = left_tokens | right_tokens
            overlap = float(len(left_tokens & right_tokens) / len(union)) if union else 1.0
            if overlap >= near_duplicate_threshold:
                issues.append(
                    f"{right.window_id} and {left.window_id}: near-duplicate cross-split windows"
                )

    return tuple(sorted(set(issues)))


def build_split_manifest(
    windows: list[WindowRecord],
    *,
    seed: int = DEFAULT_RECONSTRUCTION_SEED,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    near_duplicate_threshold: float = 0.95,
) -> SplitManifest:
    """Build a deterministic chapter-aware split manifest with leakage audit."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave room for test_ratio")

    chapter_ids = sorted(
        {window.segment_id for window in windows},
        key=lambda segment_id: _chapter_hash(segment_id, seed),
    )
    split_sizes = _compute_split_sizes(len(chapter_ids), train_ratio, val_ratio)
    chapter_splits: dict[str, str] = {}
    chapter_index = 0
    for split_name in ("train", "val", "test"):
        for segment_id in chapter_ids[chapter_index : chapter_index + split_sizes[split_name]]:
            chapter_splits[segment_id] = split_name
        chapter_index += split_sizes[split_name]

    assignments = tuple(
        {
            "window_id": window.window_id,
            "split": chapter_splits[window.segment_id],
            "segment_id": window.segment_id,
            "work_id": window.work_id,
            "chapter_number": window.chapter_number,
            "word_start": window.word_start,
            "word_end": window.word_end,
        }
        for window in sorted(
            windows,
            key=lambda item: (item.work_id, item.chapter_number, item.word_start, item.window_id),
        )
    )
    assignment_lookup = {str(item["window_id"]): str(item["split"]) for item in assignments}
    leakage_issues = _audit_leakage(
        windows,
        assignment_lookup,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    split_counts = {
        split_name: sum(1 for assignment in assignments if assignment["split"] == split_name)
        for split_name in ("train", "val", "test")
    }

    return SplitManifest(
        generated_at=utc_now(),
        seed=seed,
        min_words=min(window.word_count for window in windows),
        max_words=max(window.word_count for window in windows),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        total_windows=len(windows),
        split_counts=split_counts,
        assignments=assignments,
        leakage_issues=leakage_issues,
    )


def build_target_envelopes(
    windows: list[WindowRecord],
    split_manifest: SplitManifest,
    *,
    target_work_count: int = 3,
    min_train_windows: int = 2,
) -> list[TargetEnvelope]:
    """Build deterministic target work envelopes from train-split provenance."""
    assignment_lookup = split_manifest.assignment_lookup()
    train_windows = [
        window.with_split(assignment_lookup[window.window_id])
        for window in windows
        if assignment_lookup[window.window_id] == "train"
    ]
    by_work: dict[str, list[WindowRecord]] = {}
    for window in train_windows:
        by_work.setdefault(window.work_id, []).append(window)

    eligible_work_ids = sorted(
        (
            work_id
            for work_id, work_windows in by_work.items()
            if len(work_windows) >= min_train_windows
        ),
        key=lambda work_id: (-len(by_work[work_id]), work_id),
    )
    selected_work_ids = eligible_work_ids[:target_work_count]
    if len(selected_work_ids) < target_work_count:
        raise ValueError("not enough train windows to build the requested target envelopes")

    envelopes: list[TargetEnvelope] = []
    for work_id in selected_work_ids:
        work_windows = sorted(
            by_work[work_id],
            key=lambda window: (window.chapter_number, window.word_start, window.window_id),
        )
        segment_representatives: dict[str, WindowRecord] = {}
        for window in work_windows:
            segment_representatives.setdefault(window.segment_id, window)

        reference_windows = [
            segment_representatives[segment_id]
            for segment_id in sorted(segment_representatives)
        ]
        stylometric_names = list(reference_windows[0].stylometric_reference)
        semantic_names = list(reference_windows[0].semantic_reference)
        stylometric_target = {
            name: float(
                sum(window.stylometric_reference[name] for window in reference_windows)
                / len(reference_windows)
            )
            for name in stylometric_names
        }
        semantic_reference = {
            name: float(
                sum(window.semantic_reference[name] for window in reference_windows)
                / len(reference_windows)
            )
            for name in semantic_names
        }

        sample = work_windows[0]
        envelopes.append(
            TargetEnvelope(
                envelope_id=f"target:{work_id}",
                work_id=work_id,
                author=sample.author,
                title=sample.title,
                aggregation_rule="mean_over_unique_chapter_reference_vectors_from_train_windows",
                provenance_window_ids=tuple(window.window_id for window in work_windows),
                provenance_segment_ids=tuple(sorted(segment_representatives)),
                stylometric_target=stylometric_target,
                semantic_reference=semantic_reference,
            )
        )

    return envelopes


def select_source_windows(
    windows: list[WindowRecord],
    split_manifest: SplitManifest,
    target_envelopes: list[TargetEnvelope],
    *,
    source_window_count: int = 5,
    min_source_window_count: int = 3,
) -> list[WindowRecord]:
    """Select deterministic pilot source windows from the held-out test split."""
    assignment_lookup = split_manifest.assignment_lookup()
    target_work_ids = {envelope.work_id for envelope in target_envelopes}
    candidates = [
        window.with_split(assignment_lookup[window.window_id])
        for window in windows
        if assignment_lookup[window.window_id] == "test"
    ]
    preferred = [window for window in candidates if window.work_id not in target_work_ids]
    pool = preferred if len(preferred) >= min_source_window_count else candidates
    pool = sorted(
        pool,
        key=lambda window: (window.work_id, window.chapter_number, window.word_start),
    )

    selected: list[WindowRecord] = []
    seen_works: set[str] = set()
    for window in pool:
        if window.work_id in seen_works:
            continue
        selected.append(window)
        seen_works.add(window.work_id)
        if len(selected) == source_window_count:
            break

    if len(selected) < source_window_count:
        selected_ids = {window.window_id for window in selected}
        for window in pool:
            if window.window_id in selected_ids:
                continue
            selected.append(window)
            selected_ids.add(window.window_id)
            if len(selected) == source_window_count:
                break

    if len(selected) < min_source_window_count:
        raise ValueError("not enough held-out source windows for the requested pilot")
    return selected


def build_success_criteria(
    tolerances: ToleranceConfig | None = None,
) -> SuccessCriteria:
    """Return explicit tolerances and objective weights for the pilot."""
    resolved_tolerances = tolerances or ToleranceConfig()
    return SuccessCriteria(
        generated_at=utc_now(),
        claim_language="operational_decoupling",
        tolerances={
            "semantic_preservation_max": resolved_tolerances.semantic_preservation_max,
            "stylistic_preservation_max": resolved_tolerances.stylistic_preservation_max,
            "stylistic_target_max": resolved_tolerances.stylistic_target_max,
        },
        objective_weights={
            "semantic_preservation": 0.45,
            "stylistic_target": 0.35,
            "length_guardrail": 0.10,
            "lexical_divergence": 0.10,
        },
        lexical_guardrails={
            "length_ratio_min": 0.80,
            "length_ratio_max": 1.20,
            "token_jaccard_min": 0.10,
            "normalized_edit_similarity_min": 0.10,
        },
        minimum_pass_requirements={
            "require_semantic_tolerance": True,
            "require_target_tolerance": True,
            "require_length_guardrail": True,
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a JSON payload with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Phase 3 pilot dataset generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--corpus-output-dir", type=Path, default=CORPUS_OUTPUT_DIR)
    parser.add_argument("--pilots-dir", type=Path, default=ReconstructionPaths().pilots_dir)
    parser.add_argument("--min-words", type=int, default=128)
    parser.add_argument("--max-words", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_RECONSTRUCTION_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--target-work-count", type=int, default=3)
    parser.add_argument("--min-train-windows-per-target", type=int, default=2)
    parser.add_argument("--source-window-count", type=int, default=5)
    parser.add_argument("--min-source-window-count", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build Phase 3 source windows, target envelopes, split manifest, and criteria."""
    args = build_argument_parser().parse_args(argv)
    pilots_dir = args.pilots_dir
    pilots_dir.mkdir(parents=True, exist_ok=True)

    windows = extract_windows(
        corpus_dir=args.corpus_dir,
        corpus_output_dir=args.corpus_output_dir,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    split_manifest = build_split_manifest(
        windows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    if split_manifest.leakage_issues:
        raise ValueError(
            "Phase 3 split manifest contains leakage issues: "
            + "; ".join(split_manifest.leakage_issues)
        )

    target_envelopes = build_target_envelopes(
        windows,
        split_manifest,
        target_work_count=args.target_work_count,
        min_train_windows=args.min_train_windows_per_target,
    )
    source_windows = select_source_windows(
        windows,
        split_manifest,
        target_envelopes,
        source_window_count=args.source_window_count,
        min_source_window_count=args.min_source_window_count,
    )
    success_criteria = build_success_criteria()

    source_windows_path = _write_json(
        pilots_dir / "source_windows.json",
        {
            "generated_at": utc_now(),
            "source_windows": [window.to_dict() for window in source_windows],
        },
    )
    target_envelopes_path = _write_json(
        pilots_dir / "target_envelopes.json",
        {
            "generated_at": utc_now(),
            "target_envelopes": [envelope.to_dict() for envelope in target_envelopes],
        },
    )
    split_manifest_path = _write_json(
        pilots_dir / "split_manifest.json",
        split_manifest.to_dict(),
    )
    success_criteria_path = _write_json(
        pilots_dir / "success_criteria.json",
        success_criteria.to_dict(),
    )

    print("Phase 3 pilot dataset")
    print(f"  Source windows:   {source_windows_path}")
    print(f"  Target envelopes: {target_envelopes_path}")
    print(f"  Split manifest:   {split_manifest_path}")
    print(f"  Success criteria: {success_criteria_path}")
    print(f"  Total windows:    {len(windows)}")
    print(f"  Pilot windows:    {len(source_windows)}")
    print(f"  Target works:     {len(target_envelopes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
