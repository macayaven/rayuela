from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import reconstruction_metrics


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
    matrix: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
) -> None:
    work_dir = corpus_output_dir / work_id
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(work_dir / "chapter_stylometrics.npy", matrix)
    metadata = {
        "feature_names": feature_names,
        "feature_descriptions": feature_descriptions,
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
    chapter_numbers: list[int],
    dimensions: list[str],
    matrix: np.ndarray,
) -> None:
    work_dir = corpus_output_dir / work_id
    work_dir.mkdir(parents=True, exist_ok=True)
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
            for row_index, chapter_number in enumerate(chapter_numbers)
        ],
    }
    (work_dir / "narrative_dna.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    np.save(work_dir / "narrative_dna_vectors.npy", matrix)


def _build_baselines(tmp_path: Path) -> reconstruction_metrics.MeasurementBaselineBundle:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {
        "alpha": ("Author A", "Alpha"),
        "beta": ("Author B", "Beta"),
    }

    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2])
    _write_clean_work(corpus_dir, "beta", "Author B", "Beta", [1])

    _write_stylometric_output(
        corpus_output_dir,
        "alpha",
        np.array([[1.0, 3.0], [3.0, 5.0]]),
        feature_names=["sent_len_mean", "mattr"],
        feature_descriptions={
            "sent_len_mean": "Mean sentence length in words",
            "mattr": "Moving-average type-token ratio",
        },
    )
    _write_stylometric_output(
        corpus_output_dir,
        "beta",
        np.array([[5.0, 7.0]]),
        feature_names=["sent_len_mean", "mattr"],
        feature_descriptions={
            "sent_len_mean": "Mean sentence length in words",
            "mattr": "Moving-average type-token ratio",
        },
    )

    semantic_dimensions = [
        "existential_questioning",
        "temporal_clarity",
        "metafiction",
    ]
    _write_semantic_output(
        corpus_output_dir,
        "alpha",
        [1, 2],
        semantic_dimensions,
        np.array([[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]]),
    )
    _write_semantic_output(
        corpus_output_dir,
        "beta",
        [1],
        semantic_dimensions,
        np.array([[6.0, 8.0, 10.0]]),
    )

    return reconstruction_metrics.build_measurement_baselines(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
    )


def _manual_baseline(
    kind: Literal["stylometric", "semantic"],
    names: list[str],
    matrix: np.ndarray,
) -> reconstruction_metrics.MeasurementBaseline:
    registry = reconstruction_metrics.build_dimension_registry(
        kind=kind,
        names=names,
        descriptions={name: f"{name} description" for name in names},
    )
    measurements = reconstruction_metrics.MeasurementMatrix(
        kind=kind,
        dimension_order=tuple(names),
        matrix=matrix,
        segment_ids=tuple(f"seg:{index}" for index in range(matrix.shape[0])),
        source_paths=(),
        dimension_registry=registry,
    )
    return reconstruction_metrics.compute_measurement_baseline(measurements)


def test_stylometric_baseline_stats_are_complete(tmp_path: Path) -> None:
    baselines = _build_baselines(tmp_path)
    baseline = baselines.stylometric

    assert baseline.kind == "stylometric"
    assert baseline.chapter_count == 3
    assert baseline.dimension_order == ("sent_len_mean", "mattr")
    assert baseline.dimensions["sent_len_mean"].stats.mean == pytest.approx(3.0)
    assert baseline.dimensions["sent_len_mean"].stats.min == pytest.approx(1.0)
    assert baseline.dimensions["sent_len_mean"].stats.max == pytest.approx(5.0)
    assert baseline.dimensions["sent_len_mean"].stats.median == pytest.approx(3.0)
    assert baseline.dimensions["sent_len_mean"].metadata.description == (
        "Mean sentence length in words"
    )


def test_semantic_baseline_stats_are_complete(tmp_path: Path) -> None:
    baselines = _build_baselines(tmp_path)
    baseline = baselines.semantic

    assert baseline.kind == "semantic"
    assert baseline.chapter_count == 3
    assert baseline.dimension_order == ("existential_questioning", "metafiction")
    assert "temporal_clarity" not in baseline.dimensions
    assert baseline.dimensions["existential_questioning"].stats.mean == pytest.approx(4.0)
    assert baseline.dimensions["metafiction"].stats.max == pytest.approx(10.0)


def test_metric_reproducibility_on_same_text() -> None:
    stylometric_baseline = _manual_baseline(
        "stylometric",
        ["sent_len_mean", "mattr"],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    semantic_baseline = _manual_baseline(
        "semantic",
        ["existential_questioning", "metafiction"],
        np.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]),
    )
    tolerances = reconstruction_metrics.ToleranceConfig(
        semantic_preservation_max=0.25,
        stylistic_preservation_max=0.25,
        stylistic_target_max=0.25,
    )

    first = reconstruction_metrics.score_rewrite(
        source_stylometric=np.array([2.0, 3.0]),
        candidate_stylometric=np.array([2.0, 3.0]),
        target_stylometric=np.array([2.0, 3.0]),
        source_semantic=np.array([3.0, 5.0]),
        candidate_semantic=np.array([3.0, 5.0]),
        stylometric_baseline=stylometric_baseline,
        semantic_baseline=semantic_baseline,
        tolerances=tolerances,
        source_text="uno dos tres",
        candidate_text="uno dos tres",
    )
    second = reconstruction_metrics.score_rewrite(
        source_stylometric=np.array([2.0, 3.0]),
        candidate_stylometric=np.array([2.0, 3.0]),
        target_stylometric=np.array([2.0, 3.0]),
        source_semantic=np.array([3.0, 5.0]),
        candidate_semantic=np.array([3.0, 5.0]),
        stylometric_baseline=stylometric_baseline,
        semantic_baseline=semantic_baseline,
        tolerances=tolerances,
        source_text="uno dos tres",
        candidate_text="uno dos tres",
    )

    assert first.to_dict() == second.to_dict()


def test_identity_control_scores_as_expected() -> None:
    stylometric_baseline = _manual_baseline(
        "stylometric",
        ["sent_len_mean", "mattr"],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    semantic_baseline = _manual_baseline(
        "semantic",
        ["existential_questioning", "metafiction"],
        np.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]),
    )
    score = reconstruction_metrics.score_rewrite(
        source_stylometric=np.array([2.0, 3.0]),
        candidate_stylometric=np.array([2.0, 3.0]),
        target_stylometric=np.array([2.0, 3.0]),
        source_semantic=np.array([3.0, 5.0]),
        candidate_semantic=np.array([3.0, 5.0]),
        stylometric_baseline=stylometric_baseline,
        semantic_baseline=semantic_baseline,
        tolerances=reconstruction_metrics.ToleranceConfig(
            semantic_preservation_max=0.1,
            stylistic_preservation_max=0.1,
            stylistic_target_max=0.1,
        ),
        source_text="hola mundo",
        candidate_text="hola mundo",
    )

    assert score.semantic_source_distance == pytest.approx(0.0)
    assert score.stylistic_source_distance == pytest.approx(0.0)
    assert score.stylistic_target_distance == pytest.approx(0.0)
    assert score.within_semantic_tolerance is True
    assert score.within_stylistic_tolerance is True
    assert score.within_target_tolerance is True
    assert score.lexical_controls is not None
    assert score.lexical_controls.length_ratio == pytest.approx(1.0)


def test_random_target_control_fails_tolerance() -> None:
    stylometric_baseline = _manual_baseline(
        "stylometric",
        ["sent_len_mean", "mattr"],
        np.array([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]]),
    )
    semantic_baseline = _manual_baseline(
        "semantic",
        ["existential_questioning", "metafiction"],
        np.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]),
    )
    score = reconstruction_metrics.score_rewrite(
        source_stylometric=np.array([1.0, 1.0]),
        candidate_stylometric=np.array([1.0, 1.0]),
        target_stylometric=np.array([9.0, 9.0]),
        source_semantic=np.array([3.0, 5.0]),
        candidate_semantic=np.array([3.0, 5.0]),
        stylometric_baseline=stylometric_baseline,
        semantic_baseline=semantic_baseline,
        tolerances=reconstruction_metrics.ToleranceConfig(
            semantic_preservation_max=0.1,
            stylistic_preservation_max=0.1,
            stylistic_target_max=0.75,
        ),
    )

    assert score.within_semantic_tolerance is True
    assert score.within_stylistic_tolerance is True
    assert score.within_target_tolerance is False
    assert score.stylistic_target_improvement == pytest.approx(0.0)


def test_dimension_metadata_is_human_readable(tmp_path: Path) -> None:
    baselines = _build_baselines(tmp_path)

    for baseline in (baselines.stylometric, baselines.semantic):
        for name, dimension in baseline.dimensions.items():
            assert dimension.metadata.name == name
            assert dimension.metadata.label
            assert " " in dimension.metadata.description
            assert dimension.metadata.group


def test_main_writes_baseline_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_output_dir = tmp_path / "outputs" / "corpus"
    corpus_works = {"alpha": ("Author A", "Alpha")}

    _write_clean_work(corpus_dir, "alpha", "Author A", "Alpha", [1, 2])
    _write_stylometric_output(
        corpus_output_dir,
        "alpha",
        np.array([[1.0, 3.0], [3.0, 5.0]]),
        feature_names=["sent_len_mean", "mattr"],
        feature_descriptions={
            "sent_len_mean": "Mean sentence length in words",
            "mattr": "Moving-average type-token ratio",
        },
    )
    _write_semantic_output(
        corpus_output_dir,
        "alpha",
        [1, 2],
        ["existential_questioning", "temporal_clarity", "metafiction"],
        np.array([[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]]),
    )

    stylometric_path = tmp_path / "outputs" / "reconstruction" / "baselines" / "stylometric.json"
    semantic_path = tmp_path / "outputs" / "reconstruction" / "baselines" / "semantic.json"
    controls_path = tmp_path / "outputs" / "reconstruction" / "baselines" / "controls.json"

    monkeypatch.setattr(reconstruction_metrics, "CORPUS_WORKS", corpus_works)
    monkeypatch.setattr(
        "reconstruction_audit.load_corpus_works",
        lambda: corpus_works,
    )

    exit_code = reconstruction_metrics.main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--stylometric-baseline-path",
            str(stylometric_path),
            "--semantic-baseline-path",
            str(semantic_path),
            "--control-diagnostics-path",
            str(controls_path),
        ]
    )

    captured = capsys.readouterr()
    stylometric_payload = json.loads(stylometric_path.read_text(encoding="utf-8"))
    semantic_payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    controls_payload = json.loads(controls_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Phase 2 measurement baselines" in captured.out
    assert stylometric_payload["dimension_order"] == ["sent_len_mean", "mattr"]
    assert semantic_payload["dimension_order"] == ["existential_questioning", "metafiction"]
    assert controls_payload["controls"]["identity"]["target_pass_rate"] == pytest.approx(1.0)
