from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import reconstruction_audit


def _write_clean_work(
    corpus_dir: Path,
    work_id: str,
    author: str,
    title: str,
    chapter_numbers: list[int],
) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    chapters = [
        {
            "number": chapter_number,
            "title": None,
            "section": None,
            "text": f"{work_id} chapter {chapter_number}",
            "word_count": 10,
        }
        for chapter_number in chapter_numbers
    ]
    payload = {
        "title": title,
        "author": author,
        "year_published": 1963,
        "primary_language": "es",
        "other_languages_present": [],
        "total_chapters": len(chapters),
        "total_words": len(chapters) * 10,
        "structure": "test",
        "prescribed_reading_orders": [],
        "metadata": {},
        "chapters": chapters,
    }
    (corpus_dir / f"{work_id}_clean.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_stylometric_output(
    corpus_output_dir: Path,
    work_id: str,
    rows: int,
    *,
    metadata_n_chapters: int | None = None,
) -> None:
    work_dir = corpus_output_dir / work_id
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(work_dir / "chapter_stylometrics.npy", np.ones((rows, 2)))
    metadata = {
        "feature_names": ["sent_len_mean", "mattr"],
        "feature_descriptions": {},
        "n_chapters": rows if metadata_n_chapters is None else metadata_n_chapters,
        "n_features": 2,
        "feature_stats": {},
    }
    (work_dir / "chapter_stylometrics_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _write_semantic_output(
    corpus_output_dir: Path,
    work_id: str,
    chapter_numbers: list[int],
    *,
    vector_rows: int | None = None,
) -> None:
    work_dir = corpus_output_dir / work_id
    work_dir.mkdir(parents=True, exist_ok=True)
    dimensions = ["existential_questioning", "art_and_aesthetics"]
    payload = {
        "dimensions": dimensions,
        "chapters": [
            {
                "chapter": chapter_number,
                "section": "N/A",
                "is_expendable": False,
                "scores": {dimension: 1 for dimension in dimensions},
            }
            for chapter_number in chapter_numbers
        ],
    }
    (work_dir / "narrative_dna.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    semantic_rows = len(chapter_numbers) if vector_rows is None else vector_rows
    np.save(work_dir / "narrative_dna_vectors.npy", np.ones((semantic_rows, len(dimensions))))


def _write_profile(path: Path, kind: str, profiles: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_key = "feature_names" if kind == "stylo" else "dimensions"
    payload = {
        header_key: ["metric_1"],
        "profiles": profiles,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_cleaned_segment_count_matches_manifest(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {
        "alpha": ("Author A", "Alpha"),
        "beta": ("Author A", "Beta"),
    }
    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2, 3])
    _write_clean_work(corpus_dir, "beta", "Author A", "Beta", [1, 2])

    metadata = reconstruction_audit.build_corpus_metadata(
        corpus_dir=corpus_dir,
        corpus_works=corpus_works,
    )
    metadata_path = corpus_output_dir / "corpus_metadata.json"
    reconstruction_audit.write_corpus_metadata(metadata, metadata_path)

    written = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert written["total_works"] == 2
    assert written["total_segments"] == 5
    assert written["works"]["alpha"]["segment_count"] == 3
    assert written["works"]["alpha"]["segment_ids"] == ["alpha:1", "alpha:2", "alpha:3"]
    assert written["authors"]["Author A"]["segment_count"] == 5
    assert written["authors"]["Author A"]["work_ids"] == ["alpha", "beta"]


def test_stylometric_coverage_is_complete(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {"alpha": ("Author A", "Alpha")}
    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2, 3])
    _write_stylometric_output(corpus_output_dir, "alpha", rows=2, metadata_n_chapters=2)
    _write_semantic_output(corpus_output_dir, "alpha", [1, 2, 3])

    report = reconstruction_audit.audit_corpus_outputs(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        author_profile_paths={},
    )

    assert report.is_clean is False
    assert any(
        "alpha" in issue and "stylometric coverage mismatch" in issue
        for issue in report.issues
    )


def test_semantic_coverage_is_complete(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {"alpha": ("Author A", "Alpha")}
    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2, 3])
    _write_stylometric_output(corpus_output_dir, "alpha", rows=3)
    _write_semantic_output(corpus_output_dir, "alpha", [1, 3], vector_rows=2)

    report = reconstruction_audit.audit_corpus_outputs(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        author_profile_paths={},
    )

    assert report.is_clean is False
    assert any(
        "alpha" in issue and "semantic coverage mismatch" in issue
        for issue in report.issues
    )


def test_author_and_work_profiles_match_cleaned_counts(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {
        "alpha": ("Author A", "Alpha"),
        "beta": ("Author A", "Beta"),
    }
    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2])
    _write_clean_work(corpus_dir, "beta", "Author A", "Beta", [1])
    _write_stylometric_output(corpus_output_dir, "alpha", rows=2)
    _write_stylometric_output(corpus_output_dir, "beta", rows=1)
    _write_semantic_output(corpus_output_dir, "alpha", [1, 2])
    _write_semantic_output(corpus_output_dir, "beta", [1])

    stylo_profile_path = corpus_output_dir / "author_profiles_stylo.json"
    semantic_profile_path = corpus_output_dir / "author_profiles_semantic.json"
    mismatched_profile = {
        "Author A": {
            "author": "Author A",
            "n_chapters_total": 5,
            "n_works": 2,
            "works": [
                {"work_id": "alpha", "title": "Alpha", "n_chapters": 2},
                {"work_id": "__rayuela__", "title": "Rayuela", "n_chapters": 3},
            ],
        }
    }
    _write_profile(stylo_profile_path, "stylo", mismatched_profile)
    _write_profile(semantic_profile_path, "semantic", mismatched_profile)

    report = reconstruction_audit.audit_corpus_outputs(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        author_profile_paths={
            "stylo": stylo_profile_path,
            "semantic": semantic_profile_path,
        },
    )

    assert report.is_clean is False
    assert any("__rayuela__" in issue for issue in report.issues)
    assert any("n_chapters_total" in issue for issue in report.issues)


def test_no_orphaned_or_duplicate_outputs(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {"alpha": ("Author A", "Alpha")}
    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2])
    _write_stylometric_output(corpus_output_dir, "alpha", rows=2)
    _write_semantic_output(corpus_output_dir, "alpha", [1, 1], vector_rows=2)
    _write_stylometric_output(corpus_output_dir, "orphan_work", rows=1)

    report = reconstruction_audit.audit_corpus_outputs(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        author_profile_paths={},
    )

    assert report.is_clean is False
    assert any("duplicate semantic segment ids" in issue for issue in report.issues)
    assert any("orphan output work directories" in issue for issue in report.issues)


def test_main_writes_clean_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    metadata_path = tmp_path / "outputs" / "reconstruction" / "analysis" / "corpus_metadata.json"
    report_path = tmp_path / "outputs" / "reconstruction" / "analysis" / "corpus_sync_audit.json"
    corpus_works = {"alpha": ("Author A", "Alpha")}

    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2])
    _write_stylometric_output(corpus_output_dir, "alpha", rows=2)
    _write_semantic_output(corpus_output_dir, "alpha", [1, 2])

    matching_profile = {
        "Author A": {
            "author": "Author A",
            "n_chapters_total": 2,
            "n_works": 1,
            "works": [{"work_id": "alpha", "title": "Alpha", "n_chapters": 2}],
        }
    }
    stylo_profile_path = corpus_output_dir / "author_profiles_stylo.json"
    semantic_profile_path = corpus_output_dir / "author_profiles_semantic.json"
    _write_profile(stylo_profile_path, "stylo", matching_profile)
    _write_profile(semantic_profile_path, "semantic", matching_profile)

    monkeypatch.setattr(reconstruction_audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_audit, "DEFAULT_AUTHOR_PROFILE_PATHS", {
        "stylo": stylo_profile_path,
        "semantic": semantic_profile_path,
    })
    monkeypatch.setattr(reconstruction_audit, "load_corpus_works", lambda: corpus_works)

    exit_code = reconstruction_audit.main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--metadata-path",
            str(metadata_path),
            "--report-path",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Status:   clean" in captured.out
    assert report_payload["is_clean"] is True
    assert report_payload["works"]["alpha"]["cleaned_segment_count"] == 2
    assert report_payload["works"]["alpha"]["semantic_segment_ids"] == ["alpha:1", "alpha:2"]


def test_main_returns_nonzero_when_audit_finds_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    metadata_path = tmp_path / "outputs" / "reconstruction" / "analysis" / "corpus_metadata.json"
    report_path = tmp_path / "outputs" / "reconstruction" / "analysis" / "corpus_sync_audit.json"
    corpus_works = {"alpha": ("Author A", "Alpha")}

    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2, 3])
    _write_stylometric_output(corpus_output_dir, "alpha", rows=2, metadata_n_chapters=2)

    monkeypatch.setattr(reconstruction_audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_audit, "DEFAULT_AUTHOR_PROFILE_PATHS", {
        "stylo": corpus_output_dir / "author_profiles_stylo.json",
        "semantic": corpus_output_dir / "author_profiles_semantic.json",
    })
    monkeypatch.setattr(reconstruction_audit, "load_corpus_works", lambda: corpus_works)

    exit_code = reconstruction_audit.main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--metadata-path",
            str(metadata_path),
            "--report-path",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert "stylometric coverage mismatch" in captured.out
    assert "semantic outputs missing" in captured.out
    assert report_payload["is_clean"] is False
    assert report_payload["issue_count"] >= 3
