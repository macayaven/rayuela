#!/usr/bin/env python3
"""
Corpus synchronization audit for Part 3 reconstruction work.

Phase 1 verifies that the cleaned corpus and its derived outputs stay in a
single operationally decoupled state before any rewrite experiments begin.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
DEFAULT_CORPUS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "corpus"
DEFAULT_CORPUS_METADATA_PATH = DEFAULT_CORPUS_OUTPUT_DIR / "corpus_metadata.json"
DEFAULT_AUDIT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reconstruction" / "analysis" / "corpus_sync_audit.json"
)
DEFAULT_AUTHOR_PROFILE_PATHS: dict[str, Path] = {
    "stylo": DEFAULT_CORPUS_OUTPUT_DIR / "author_profiles_stylo.json",
    "semantic": DEFAULT_CORPUS_OUTPUT_DIR / "author_profiles_semantic.json",
}


@dataclass(frozen=True)
class WorkAuditRecord:
    """Coverage summary for one cleaned corpus work."""

    work_id: str
    author: str
    title: str
    cleaned_segment_ids: tuple[str, ...]
    stylometric_segment_count: int | None
    stylometric_metadata_count: int | None
    semantic_segment_ids: tuple[str, ...] | None
    semantic_vector_rows: int | None

    @property
    def cleaned_segment_count(self) -> int:
        """Return the canonical cleaned segment count for this work."""
        return len(self.cleaned_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record."""
        semantic_ids = (
            list(self.semantic_segment_ids)
            if self.semantic_segment_ids is not None
            else None
        )
        return {
            "work_id": self.work_id,
            "author": self.author,
            "title": self.title,
            "cleaned_segment_count": self.cleaned_segment_count,
            "cleaned_segment_ids": list(self.cleaned_segment_ids),
            "stylometric_segment_count": self.stylometric_segment_count,
            "stylometric_metadata_count": self.stylometric_metadata_count,
            "semantic_segment_count": len(semantic_ids) if semantic_ids is not None else None,
            "semantic_segment_ids": semantic_ids,
            "semantic_vector_rows": self.semantic_vector_rows,
        }


@dataclass(frozen=True)
class AuditReport:
    """Structured result for the corpus synchronization audit."""

    generated_at: str
    metadata: dict[str, Any]
    works: dict[str, WorkAuditRecord]
    issues: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        """Return True when the audit found no synchronization issues."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""
        return {
            "generated_at": self.generated_at,
            "is_clean": self.is_clean,
            "issue_count": len(self.issues),
            "issues": list(self.issues),
            "metadata": self.metadata,
            "works": {work_id: record.to_dict() for work_id, record in self.works.items()},
        }


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def canonical_segment_id(work_id: str, chapter_number: int) -> str:
    """Return the canonical segment identifier for a cleaned chapter."""
    return f"{work_id}:{chapter_number}"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_corpus_works() -> dict[str, tuple[str, str]]:
    """Load the canonical corpus work mapping lazily."""
    from project_config import CORPUS_WORKS

    return CORPUS_WORKS


def project_relative_path(path: Path) -> str:
    """Return a repo-relative string for console output."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_corpus_metadata(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    corpus_works: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build counts and canonical segment IDs from the cleaned corpus only."""
    resolved_corpus_works = load_corpus_works() if corpus_works is None else corpus_works
    works_payload: dict[str, dict[str, Any]] = {}
    authors_payload: dict[str, dict[str, Any]] = {}
    author_work_ids: defaultdict[str, list[str]] = defaultdict(list)
    author_segment_counts: defaultdict[str, int] = defaultdict(int)

    for work_id in sorted(resolved_corpus_works):
        author, title = resolved_corpus_works[work_id]
        clean_path = corpus_dir / f"{work_id}_clean.json"
        payload = _load_json(clean_path)
        chapter_numbers = [int(chapter["number"]) for chapter in payload["chapters"]]
        segment_ids = [
            canonical_segment_id(work_id, chapter_number)
            for chapter_number in chapter_numbers
        ]

        works_payload[work_id] = {
            "author": author,
            "title": title,
            "segment_count": len(segment_ids),
            "segment_ids": segment_ids,
        }
        author_work_ids[author].append(work_id)
        author_segment_counts[author] += len(segment_ids)

    for author in sorted(author_work_ids):
        authors_payload[author] = {
            "segment_count": author_segment_counts[author],
            "work_ids": sorted(author_work_ids[author]),
        }

    total_segments = sum(work["segment_count"] for work in works_payload.values())
    return {
        "generated_at": utc_now(),
        "total_works": len(works_payload),
        "total_segments": total_segments,
        "works": works_payload,
        "authors": authors_payload,
    }


def write_corpus_metadata(
    metadata: dict[str, Any],
    path: Path = DEFAULT_CORPUS_METADATA_PATH,
) -> Path:
    """Persist the cleaned corpus metadata manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return path


def _load_stylometric_counts(work_dir: Path) -> tuple[int | None, int | None]:
    """Return row counts from stylometric outputs and metadata."""
    npy_path = work_dir / "chapter_stylometrics.npy"
    meta_path = work_dir / "chapter_stylometrics_metadata.json"

    matrix_rows = None
    if npy_path.exists():
        matrix_rows = int(np.load(npy_path).shape[0])

    metadata_rows = None
    if meta_path.exists():
        metadata_rows = int(_load_json(meta_path)["n_chapters"])

    return matrix_rows, metadata_rows


def _load_semantic_counts(
    work_dir: Path,
    work_id: str,
) -> tuple[tuple[str, ...] | None, int | None]:
    """Return canonical semantic segment IDs and vector rows for one work."""
    json_path = work_dir / "narrative_dna.json"
    vectors_path = work_dir / "narrative_dna_vectors.npy"

    semantic_ids = None
    if json_path.exists():
        payload = _load_json(json_path)
        chapter_numbers = [int(chapter["chapter"]) for chapter in payload["chapters"]]
        semantic_ids = tuple(
            canonical_segment_id(work_id, chapter_number)
            for chapter_number in chapter_numbers
        )

    vector_rows = int(np.load(vectors_path).shape[0]) if vectors_path.exists() else None
    return semantic_ids, vector_rows


def _audit_author_profiles(
    metadata: dict[str, Any],
    author_profile_paths: dict[str, Path],
) -> list[str]:
    """Validate author profile totals and work membership against the cleaned corpus."""
    issues: list[str] = []
    expected_authors = metadata["authors"]
    expected_work_counts = metadata["works"]

    for label, path in sorted(author_profile_paths.items()):
        if not path.exists():
            issues.append(f"{label} profile missing: {path.name}")
            continue

        payload = _load_json(path)
        profiles = payload.get("profiles", {})
        actual_authors = set(profiles)
        missing_authors = sorted(set(expected_authors) - actual_authors)
        extra_authors = sorted(actual_authors - set(expected_authors))
        if missing_authors:
            issues.append(f"{label} profile missing authors: {missing_authors}")
        if extra_authors:
            issues.append(f"{label} profile has unexpected authors: {extra_authors}")

        for author in sorted(set(expected_authors) & actual_authors):
            expected_author = expected_authors[author]
            profile = profiles[author]
            actual_total = int(profile.get("n_chapters_total", -1))
            if actual_total != expected_author["segment_count"]:
                issues.append(
                    f"{label} profile n_chapters_total mismatch for {author}: "
                    f"{actual_total} != {expected_author['segment_count']}"
                )

            actual_work_counts = {
                str(work["work_id"]): int(work["n_chapters"])
                for work in profile.get("works", [])
            }
            expected_author_work_counts = {
                work_id: int(expected_work_counts[work_id]["segment_count"])
                for work_id in expected_author["work_ids"]
            }
            if actual_work_counts != expected_author_work_counts:
                issues.append(
                    f"{label} profile work counts mismatch for {author}: "
                    f"{actual_work_counts} != {expected_author_work_counts}"
                )

    return issues


def audit_corpus_outputs(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    corpus_output_dir: Path = DEFAULT_CORPUS_OUTPUT_DIR,
    corpus_works: dict[str, tuple[str, str]] | None = None,
    author_profile_paths: dict[str, Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditReport:
    """Audit cleaned corpus coverage against stylometric, semantic, and profile outputs."""
    resolved_corpus_works = load_corpus_works() if corpus_works is None else corpus_works
    resolved_metadata = (
        build_corpus_metadata(corpus_dir=corpus_dir, corpus_works=resolved_corpus_works)
        if metadata is None
        else metadata
    )
    issues: list[str] = []
    works: dict[str, WorkAuditRecord] = {}

    for work_id in sorted(resolved_corpus_works):
        work_meta = resolved_metadata["works"][work_id]
        cleaned_segment_ids = tuple(work_meta["segment_ids"])
        cleaned_id_set = set(cleaned_segment_ids)
        work_dir = corpus_output_dir / work_id

        stylometric_rows, stylometric_metadata_rows = _load_stylometric_counts(work_dir)
        if (
            stylometric_rows != len(cleaned_segment_ids)
            or stylometric_metadata_rows != len(cleaned_segment_ids)
        ):
            issues.append(
                f"{work_id}: stylometric coverage mismatch "
                f"(clean={len(cleaned_segment_ids)}, npy={stylometric_rows}, "
                f"metadata={stylometric_metadata_rows})"
            )

        semantic_segment_ids, semantic_vector_rows = _load_semantic_counts(work_dir, work_id)
        if semantic_segment_ids is None or semantic_vector_rows is None:
            issues.append(f"{work_id}: semantic outputs missing")
        else:
            duplicate_semantic_ids = sorted(
                segment_id
                for segment_id in set(semantic_segment_ids)
                if semantic_segment_ids.count(segment_id) > 1
            )
            if duplicate_semantic_ids:
                issues.append(f"{work_id}: duplicate semantic segment ids {duplicate_semantic_ids}")
            if (
                set(semantic_segment_ids) != cleaned_id_set
                or semantic_vector_rows != len(cleaned_segment_ids)
            ):
                issues.append(
                    f"{work_id}: semantic coverage mismatch "
                    f"(clean={len(cleaned_segment_ids)}, json={len(semantic_segment_ids)}, "
                    f"vectors={semantic_vector_rows})"
                )

        works[work_id] = WorkAuditRecord(
            work_id=work_id,
            author=str(work_meta["author"]),
            title=str(work_meta["title"]),
            cleaned_segment_ids=cleaned_segment_ids,
            stylometric_segment_count=stylometric_rows,
            stylometric_metadata_count=stylometric_metadata_rows,
            semantic_segment_ids=semantic_segment_ids,
            semantic_vector_rows=semantic_vector_rows,
        )

    if corpus_output_dir.exists():
        orphan_work_dirs = sorted(
            path.name
            for path in corpus_output_dir.iterdir()
            if path.is_dir() and path.name not in resolved_corpus_works
        )
        if orphan_work_dirs:
            issues.append(f"orphan output work directories: {orphan_work_dirs}")

    resolved_profile_paths = (
        DEFAULT_AUTHOR_PROFILE_PATHS
        if author_profile_paths is None
        else author_profile_paths
    )
    issues.extend(_audit_author_profiles(resolved_metadata, resolved_profile_paths))

    return AuditReport(
        generated_at=utc_now(),
        metadata=resolved_metadata,
        works=works,
        issues=tuple(issues),
    )


def write_audit_report(report: AuditReport, path: Path = DEFAULT_AUDIT_REPORT_PATH) -> Path:
    """Persist the audit summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Phase 1 corpus audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--corpus-output-dir", type=Path, default=DEFAULT_CORPUS_OUTPUT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_CORPUS_METADATA_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_AUDIT_REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 1 corpus audit and write machine-readable artifacts."""
    args = build_argument_parser().parse_args(argv)

    metadata = build_corpus_metadata(corpus_dir=args.corpus_dir)
    metadata_path = write_corpus_metadata(metadata, args.metadata_path)
    report = audit_corpus_outputs(
        corpus_dir=args.corpus_dir,
        corpus_output_dir=args.corpus_output_dir,
        metadata=metadata,
    )
    report_path = write_audit_report(report, args.report_path)

    print("Phase 1 corpus audit")
    print(f"  Metadata: {project_relative_path(metadata_path)}")
    print(f"  Report:   {project_relative_path(report_path)}")
    print(f"  Works:    {metadata['total_works']}")
    print(f"  Segments: {metadata['total_segments']}")

    if report.is_clean:
        print("  Status:   clean")
        return 0

    print(f"  Status:   {len(report.issues)} issues")
    for issue in report.issues:
        print(f"  - {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
