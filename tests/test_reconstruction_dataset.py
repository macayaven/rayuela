from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import reconstruction_dataset


def _write_clean_work(
    corpus_dir: Path,
    work_id: str,
    author: str,
    title: str,
    chapter_texts: list[str],
) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    chapter_word_counts = [len(text.split()) for text in chapter_texts]
    chapters = [
        {
            "number": index + 1,
            "title": f"Chapter {index + 1}",
            "section": None,
            "text": text,
            "word_count": chapter_word_counts[index],
        }
        for index, text in enumerate(chapter_texts)
    ]
    payload = {
        "title": title,
        "author": author,
        "year_published": 1963,
        "primary_language": "es",
        "other_languages_present": [],
        "total_chapters": len(chapters),
        "total_words": sum(chapter_word_counts),
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
    matrix: np.ndarray,
    feature_names: list[str],
) -> None:
    work_dir = corpus_output_dir / work_id
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(work_dir / "chapter_stylometrics.npy", matrix)
    metadata = {
        "feature_names": feature_names,
        "feature_descriptions": {
            "sent_len_mean": "Mean sentence length in words",
            "mattr": "Moving-average type-token ratio",
        },
        "n_chapters": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "feature_stats": {},
    }
    (work_dir / "chapter_stylometrics_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _write_semantic_output(
    corpus_output_dir: Path,
    work_id: str,
    chapter_count: int,
    matrix: np.ndarray,
) -> None:
    work_dir = corpus_output_dir / work_id
    work_dir.mkdir(parents=True, exist_ok=True)
    dimensions = ["existential_questioning", "temporal_clarity", "metafiction"]
    payload = {
        "dimensions": dimensions,
        "chapters": [
            {
                "chapter": chapter_number,
                "section": "N/A",
                "is_expendable": False,
                "scores": {
                    dimension: int(matrix[row_index, col_index])
                    for col_index, dimension in enumerate(dimensions)
                },
            }
            for row_index, chapter_number in enumerate(range(1, chapter_count + 1))
        ],
    }
    (work_dir / "narrative_dna.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    np.save(work_dir / "narrative_dna_vectors.npy", matrix)


def _build_fixture_dataset(tmp_path: Path) -> tuple[Path, Path, dict[str, tuple[str, str]]]:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {
        "alpha": ("Author A", "Alpha"),
        "beta": ("Author B", "Beta"),
        "gamma": ("Author C", "Gamma"),
    }

    _write_clean_work(
        corpus_dir,
        "alpha",
        "Author A",
        "Alpha",
        [
            "a1 a2 a3 a4 a5 a6 a7 a8 a9",
            "a10 a11 a12 a13 a14 a15 a16 a17 a18",
        ],
    )
    _write_clean_work(
        corpus_dir,
        "beta",
        "Author B",
        "Beta",
        [
            "b1 b2 b3 b4 b5 b6 b7 b8 b9",
            "b10 b11 b12 b13 b14 b15 b16 b17 b18",
        ],
    )
    _write_clean_work(
        corpus_dir,
        "gamma",
        "Author C",
        "Gamma",
        [
            "g1 g2 g3 g4 g5 g6 g7 g8 g9",
            "g10 g11 g12 g13 g14 g15 g16 g17 g18",
        ],
    )

    feature_names = ["sent_len_mean", "mattr"]
    for index, work_id in enumerate(corpus_works, start=1):
        _write_stylometric_output(
            corpus_output_dir,
            work_id,
            np.array([[index * 1.0, index * 2.0], [index * 3.0, index * 4.0]]),
            feature_names=feature_names,
        )
        _write_semantic_output(
            corpus_output_dir,
            work_id,
            2,
            np.array([[index * 2.0, 4.0, index * 6.0], [index * 4.0, 6.0, index * 8.0]]),
        )

    return corpus_dir, corpus_output_dir, corpus_works


def test_window_extraction_respects_word_bounds(tmp_path: Path) -> None:
    corpus_dir, corpus_output_dir, corpus_works = _build_fixture_dataset(tmp_path)

    windows = reconstruction_dataset.extract_windows(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        min_words=4,
        max_words=5,
    )
    alpha_chapter_one = [
        window
        for window in windows
        if window.work_id == "alpha" and window.chapter_number == 1
    ]

    assert [window.word_count for window in alpha_chapter_one] == [5, 4]
    assert alpha_chapter_one[0].text == "a1 a2 a3 a4 a5"
    assert alpha_chapter_one[1].text == "a6 a7 a8 a9"
    assert alpha_chapter_one[0].word_start == 0
    assert alpha_chapter_one[0].word_end == 5
    assert alpha_chapter_one[1].word_start == 5
    assert alpha_chapter_one[1].word_end == 9


def test_source_windows_do_not_overlap_eval_targets(tmp_path: Path) -> None:
    corpus_dir, corpus_output_dir, corpus_works = _build_fixture_dataset(tmp_path)
    windows = reconstruction_dataset.extract_windows(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        min_words=4,
        max_words=5,
    )
    split_manifest = reconstruction_dataset.build_split_manifest(
        windows,
        seed=7,
        train_ratio=0.34,
        val_ratio=0.33,
    )
    envelopes = reconstruction_dataset.build_target_envelopes(
        windows,
        split_manifest,
        target_work_count=1,
        min_train_windows=2,
    )
    source_windows = reconstruction_dataset.select_source_windows(
        windows,
        split_manifest,
        envelopes,
        source_window_count=3,
        min_source_window_count=3,
    )

    envelope_window_ids = {
        window_id
        for envelope in envelopes
        for window_id in envelope.provenance_window_ids
    }
    envelope_segments = {
        segment_id
        for envelope in envelopes
        for segment_id in envelope.provenance_segment_ids
    }

    assert len(source_windows) == 3
    assert all(window.window_id not in envelope_window_ids for window in source_windows)
    assert all(window.segment_id not in envelope_segments for window in source_windows)
    assert all(window.split == "test" for window in source_windows)


def test_split_manifest_has_no_leakage(tmp_path: Path) -> None:
    corpus_dir, corpus_output_dir, corpus_works = _build_fixture_dataset(tmp_path)
    windows = reconstruction_dataset.extract_windows(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        min_words=4,
        max_words=5,
    )

    split_manifest = reconstruction_dataset.build_split_manifest(
        windows,
        seed=7,
        train_ratio=0.34,
        val_ratio=0.33,
    )

    assert split_manifest.leakage_issues == ()
    assert split_manifest.total_windows == len(windows)
    assert split_manifest.split_counts == {"train": 4, "val": 4, "test": 4}
    assert sum(split_manifest.split_counts.values()) == len(split_manifest.assignments)


def test_near_duplicate_leakage_is_reported() -> None:
    windows = [
        reconstruction_dataset.WindowRecord(
            window_id="alpha:1",
            work_id="alpha",
            author="Author A",
            title="Alpha",
            chapter_number=1,
            segment_id="alpha:1",
            chapter_word_count=5,
            word_start=0,
            word_end=5,
            word_count=5,
            text="uno dos tres cuatro cinco",
            stylometric_reference={"sent_len_mean": 1.0},
            semantic_reference={"metafiction": 2.0},
        ),
        reconstruction_dataset.WindowRecord(
            window_id="beta:1",
            work_id="beta",
            author="Author B",
            title="Beta",
            chapter_number=1,
            segment_id="beta:1",
            chapter_word_count=5,
            word_start=0,
            word_end=5,
            word_count=5,
            text="uno dos tres cuatro cinco",
            stylometric_reference={"sent_len_mean": 3.0},
            semantic_reference={"metafiction": 4.0},
        ),
    ]

    leakage_issues = reconstruction_dataset._audit_leakage(
        windows,
        {"alpha:1": "train", "beta:1": "test"},
        near_duplicate_threshold=0.95,
    )

    assert leakage_issues == ("alpha:1 and beta:1: near-duplicate cross-split windows",)


def test_target_envelopes_are_reproducible(tmp_path: Path) -> None:
    corpus_dir, corpus_output_dir, corpus_works = _build_fixture_dataset(tmp_path)
    windows = reconstruction_dataset.extract_windows(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        min_words=4,
        max_words=5,
    )
    split_manifest = reconstruction_dataset.build_split_manifest(
        windows,
        seed=7,
        train_ratio=0.34,
        val_ratio=0.33,
    )

    first = reconstruction_dataset.build_target_envelopes(
        windows,
        split_manifest,
        target_work_count=1,
        min_train_windows=2,
    )
    second = reconstruction_dataset.build_target_envelopes(
        windows,
        split_manifest,
        target_work_count=1,
        min_train_windows=2,
    )

    assert [envelope.to_dict() for envelope in first] == [envelope.to_dict() for envelope in second]


def test_success_criteria_are_explicit_and_serialized() -> None:
    criteria = reconstruction_dataset.build_success_criteria()
    payload = criteria.to_dict()

    assert payload["claim_language"] == "operational_decoupling"
    assert payload["tolerances"]["semantic_preservation_max"] > 0
    assert payload["objective_weights"]["semantic_preservation"] > 0
    assert payload["objective_weights"]["stylistic_target"] > 0
    assert payload["lexical_guardrails"]["length_ratio_min"] < 1.0
    assert payload["lexical_guardrails"]["length_ratio_max"] > 1.0
    assert payload["minimum_pass_requirements"]["require_target_tolerance"] is True


def test_main_writes_pilot_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus_dir, corpus_output_dir, corpus_works = _build_fixture_dataset(tmp_path)
    pilots_dir = tmp_path / "outputs" / "reconstruction" / "pilots"

    monkeypatch.setattr(reconstruction_dataset, "CORPUS_WORKS", corpus_works)

    exit_code = reconstruction_dataset.main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--pilots-dir",
            str(pilots_dir),
            "--min-words",
            "4",
            "--max-words",
            "5",
            "--train-ratio",
            "0.34",
            "--val-ratio",
            "0.33",
            "--target-work-count",
            "1",
            "--source-window-count",
            "3",
        ]
    )

    captured = capsys.readouterr()
    source_windows = json.loads((pilots_dir / "source_windows.json").read_text(encoding="utf-8"))
    target_envelopes = json.loads(
        (pilots_dir / "target_envelopes.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads((pilots_dir / "split_manifest.json").read_text(encoding="utf-8"))
    success_criteria = json.loads(
        (pilots_dir / "success_criteria.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert "Phase 3 pilot dataset" in captured.out
    assert len(source_windows["source_windows"]) == 3
    assert len(target_envelopes["target_envelopes"]) == 1
    assert split_manifest["split_counts"] == {"train": 4, "val": 4, "test": 4}
    assert success_criteria["claim_language"] == "operational_decoupling"
