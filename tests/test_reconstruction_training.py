from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import reconstruction_dataset
import reconstruction_infer
import reconstruction_train


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


def _build_phase5_artifacts(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, tuple[str, str]], Path, Path]:
    corpus_dir, corpus_output_dir, corpus_works = _build_fixture_dataset(tmp_path)
    pilots_dir = tmp_path / "outputs" / "reconstruction" / "pilots"
    windows = reconstruction_dataset.extract_windows(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        min_words=4,
        max_words=5,
    )
    split_manifest = reconstruction_dataset.build_split_manifest(
        windows,
        seed=11,
        min_words=4,
        max_words=5,
        train_ratio=0.34,
        val_ratio=0.33,
    )
    envelopes = reconstruction_dataset.build_target_envelopes(
        windows,
        split_manifest,
        target_work_count=2,
        min_train_windows=2,
    )
    success_criteria = reconstruction_dataset.build_success_criteria()

    pilots_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = pilots_dir / "split_manifest.json"
    target_envelopes_path = pilots_dir / "target_envelopes.json"
    success_criteria_path = pilots_dir / "success_criteria.json"
    corpus_manifest_path = tmp_path / "outputs" / "corpus" / "corpus_metadata.json"

    split_manifest_path.write_text(
        json.dumps(split_manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    target_envelopes_path.write_text(
        json.dumps(
            {
                "generated_at": reconstruction_dataset.utc_now(),
                "target_envelopes": [envelope.to_dict() for envelope in envelopes],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    success_criteria_path.write_text(
        json.dumps(success_criteria.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    corpus_manifest_path.write_text(
        json.dumps({"works": list(corpus_works)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return corpus_dir, corpus_output_dir, corpus_works, split_manifest_path, target_envelopes_path


def test_model_config_is_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_dir, corpus_output_dir, _, split_manifest_path, target_envelopes_path = (
        _build_phase5_artifacts(tmp_path)
    )
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")

    exit_code = reconstruction_train.main(
        [
            "--run-id",
            "phase5-smoke",
            "--corpus-dir",
            str(corpus_dir),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--split-manifest-path",
            str(split_manifest_path),
            "--target-envelopes-path",
            str(target_envelopes_path),
            "--allow-corpus-discovery",
        ]
    )

    config_path = (
        tmp_path / "outputs" / "reconstruction" / "runs" / "phase5-smoke" / "training_config.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["model_id"] == "google/mt5-xl"
    assert payload["dataset_mode"] == "identity_smoke"
    assert payload["wandb_project"] is None
    assert payload["seed"] == reconstruction_train.DEFAULT_RECONSTRUCTION_SEED


def test_train_val_test_splits_match_manifest(tmp_path: Path) -> None:
    corpus_dir, corpus_output_dir, corpus_works, split_manifest_path, target_envelopes_path = (
        _build_phase5_artifacts(tmp_path)
    )
    split_manifest = reconstruction_train.load_split_manifest(split_manifest_path)
    target_envelopes = reconstruction_train.load_target_envelopes(target_envelopes_path)
    windows = reconstruction_dataset.extract_windows(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        corpus_works=corpus_works,
        min_words=split_manifest.min_words,
        max_words=split_manifest.max_words,
    )

    examples = reconstruction_train.build_training_examples(
        windows,
        split_manifest,
        target_envelopes,
    )

    counts = reconstruction_train.count_examples_by_split(examples)
    assert counts == split_manifest.split_counts


def test_checkpoint_metadata_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir, corpus_output_dir, _, split_manifest_path, target_envelopes_path = (
        _build_phase5_artifacts(tmp_path)
    )
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")

    reconstruction_train.main(
        [
            "--run-id",
            "phase5-metadata",
            "--corpus-dir",
            str(corpus_dir),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--split-manifest-path",
            str(split_manifest_path),
            "--target-envelopes-path",
            str(target_envelopes_path),
            "--allow-corpus-discovery",
        ]
    )

    metadata_path = (
        tmp_path
        / "outputs"
        / "reconstruction"
        / "runs"
        / "phase5-metadata"
        / "checkpoint_metadata.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["run_id"] == "phase5-metadata"
    assert payload["git_sha"] == "feedface"
    assert payload["phase"] == "phase-5-training-scaffold"
    assert payload["model_id"] == "google/mt5-xl"
    assert payload["adapter_artifact_path"].endswith("adapter/adapter_model.safetensors")
    assert payload["adapter_is_placeholder"] is True
    assert payload["config_path"].endswith("training_config.json")
    assert payload["tokenizer_config_path"].endswith("tokenizer_config.json")
    assert payload["metrics_path"].endswith("training_metrics.json")
    assert payload["split_counts"] == {"train": 4, "val": 4, "test": 4}


def test_inference_pipeline_refuses_placeholder_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir, corpus_output_dir, _, split_manifest_path, target_envelopes_path = (
        _build_phase5_artifacts(tmp_path)
    )
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")

    reconstruction_train.main(
        [
            "--run-id",
            "phase5-infer",
            "--corpus-dir",
            str(corpus_dir),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--split-manifest-path",
            str(split_manifest_path),
            "--target-envelopes-path",
            str(target_envelopes_path),
            "--allow-corpus-discovery",
        ]
    )

    metadata_path = (
        tmp_path
        / "outputs"
        / "reconstruction"
        / "runs"
        / "phase5-infer"
        / "checkpoint_metadata.json"
    )

    with pytest.raises(ValueError, match="placeholder adapter"):
        reconstruction_infer.load_saved_adapter(metadata_path)


def test_optional_wandb_logging_records_run_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _Run:
        path = "rayuela/phase5/test-run"

        def log(self, payload: dict[str, object]) -> None:
            calls["log"] = payload

        def finish(self) -> None:
            calls["finished"] = True

    class _WandbStub:
        @staticmethod
        def init(**kwargs: object) -> _Run:
            calls["init"] = kwargs
            return _Run()

    monkeypatch.setattr(reconstruction_train, "_load_wandb_module", lambda: _WandbStub)
    config = reconstruction_train.TrainingConfig(
        run_id="phase5-wandb",
        model_id="google/mt5-xl",
        dataset_mode="identity_smoke",
        seed=17,
        wandb_project="rayuela",
        wandb_entity="macayaven",
        wandb_mode="offline",
    )

    logger = reconstruction_train.build_experiment_logger(
        config=config,
        git_sha="feedface",
        split_counts={"train": 4, "val": 2, "test": 2},
    )
    logger.log_metrics({"train_loss": 0.42})
    logger.finish()

    assert calls["init"] == {
        "project": "rayuela",
        "entity": "macayaven",
        "mode": "offline",
        "config": {
            "run_id": "phase5-wandb",
            "model_id": "google/mt5-xl",
            "dataset_mode": "identity_smoke",
            "seed": 17,
            "git_sha": "feedface",
            "split_counts": {"train": 4, "val": 2, "test": 2},
        },
        "name": "phase5-wandb",
    }
    assert calls["log"] == {"train_loss": 0.42}
    assert calls["finished"] is True


def test_extract_windows_requires_opt_in_for_noncanonical_corpus(tmp_path: Path) -> None:
    corpus_dir, corpus_output_dir, _, _, _ = _build_phase5_artifacts(tmp_path)

    with pytest.raises(ValueError, match="missing canonical clean corpus files"):
        reconstruction_dataset.extract_windows(
            corpus_dir=corpus_dir,
            corpus_output_dir=corpus_output_dir,
            min_words=4,
            max_words=5,
        )

    windows = reconstruction_dataset.extract_windows(
        corpus_dir=corpus_dir,
        corpus_output_dir=corpus_output_dir,
        min_words=4,
        max_words=5,
        allow_discovery=True,
    )

    assert len(windows) == 12
    assert {window.work_id for window in windows} == {"alpha", "beta", "gamma"}


def test_failed_setup_still_finalizes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, split_manifest_path, target_envelopes_path = _build_phase5_artifacts(tmp_path)
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")
    monkeypatch.setattr(
        reconstruction_train,
        "load_split_manifest",
        lambda path: (_ for _ in ()).throw(RuntimeError("split load failed")),
    )

    with pytest.raises(RuntimeError, match="split load failed"):
        reconstruction_train.main(
            [
                "--run-id",
                "phase5-failed-setup",
                "--split-manifest-path",
                str(split_manifest_path),
                "--target-envelopes-path",
                str(target_envelopes_path),
            ]
        )

    manifest_path = (
        tmp_path / "outputs" / "reconstruction" / "runs" / "phase5-failed-setup" / "manifest.json"
    )
    indexed_manifest_path = (
        tmp_path / "outputs" / "reconstruction" / "manifests" / "phase5-failed-setup.json"
    )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed_payload = json.loads(indexed_manifest_path.read_text(encoding="utf-8"))

    assert manifest_payload["status"] == "failed"
    assert manifest_payload["error_message"] == "split load failed"
    assert indexed_payload["status"] == "failed"
