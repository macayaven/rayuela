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
    metrics_path = (
        tmp_path / "outputs" / "reconstruction" / "runs" / "phase5-smoke" / "training_metrics.json"
    )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["model_id"] == "google/mt5-xl"
    assert payload["dataset_mode"] == "identity_smoke"
    assert payload["training_mode"] == "scaffold"
    assert payload["wandb_project"] is None
    assert payload["seed"] == reconstruction_train.DEFAULT_RECONSTRUCTION_SEED
    assert metrics_payload["status"] == "scaffold_only"
    assert metrics_payload["training_mode"] == "scaffold"
    assert metrics_payload["dataset_paths"]["train"].endswith("training_dataset/train.jsonl")


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


def test_contract_smoke_examples_encode_final_answer_contract(tmp_path: Path) -> None:
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
        dataset_mode="contract_smoke",
    )

    assert examples[0].dataset_mode == "contract_smoke"
    assert examples[0].target_text == examples[0].source_text
    assert "sin razonamiento" in examples[0].instruction
    assert "Devuelve solamente el pasaje final" in examples[0].instruction


def test_training_dataset_jsonl_is_written_by_split(tmp_path: Path) -> None:
    examples = [
        reconstruction_train.TrainingExample(
            window_id="w1",
            split="train",
            instruction="instr",
            source_text="source",
            target_text="target",
            target_envelope_id="target:1",
            dataset_mode="contract_smoke",
        ),
        reconstruction_train.TrainingExample(
            window_id="w2",
            split="val",
            instruction="instr",
            source_text="source",
            target_text="target",
            target_envelope_id="target:1",
            dataset_mode="contract_smoke",
        ),
    ]

    paths = reconstruction_train.write_training_dataset(examples, tmp_path / "dataset")

    train_lines = (tmp_path / "dataset" / paths["train"]).read_text(encoding="utf-8").splitlines()
    val_lines = (tmp_path / "dataset" / paths["val"]).read_text(encoding="utf-8").splitlines()
    test_lines = (tmp_path / "dataset" / paths["test"]).read_text(encoding="utf-8").splitlines()

    assert len(train_lines) == 1
    assert len(val_lines) == 1
    assert test_lines == []
    assert json.loads(train_lines[0])["window_id"] == "w1"


def test_seq2seq_input_format_includes_instruction_and_source() -> None:
    example = reconstruction_train.TrainingExample(
        window_id="w1",
        split="train",
        instruction="Devuelve solamente el pasaje final.",
        source_text="Texto fuente.",
        target_text="Texto destino.",
        target_envelope_id="target:1",
        dataset_mode="contract_smoke",
    )

    formatted = reconstruction_train.format_seq2seq_input(example)

    assert formatted == "Devuelve solamente el pasaje final.\n\nPasaje:\nTexto fuente."


def test_sft_text_format_includes_response_and_eos() -> None:
    example = reconstruction_train.TrainingExample(
        window_id="w1",
        split="train",
        instruction="Devuelve solamente el pasaje final.",
        source_text="Texto fuente.",
        target_text="Texto destino.",
        target_envelope_id="target:1",
        dataset_mode="contract_smoke",
    )

    formatted = reconstruction_train.format_sft_text(example, "</s>")

    assert "### Instrucción:\nDevuelve solamente el pasaje final." in formatted
    assert "### Pasaje:\nTexto fuente." in formatted
    assert formatted.endswith("### Respuesta:\nTexto destino.</s>")


def test_select_training_examples_is_split_bounded() -> None:
    examples = [
        reconstruction_train.TrainingExample(
            window_id=f"w{index}",
            split="train" if index < 3 else "val",
            instruction="instr",
            source_text="source",
            target_text="target",
            target_envelope_id="target:1",
            dataset_mode="contract_smoke",
        )
        for index in range(5)
    ]

    selected = reconstruction_train.select_training_examples(examples, split="train", limit=2)

    assert [example.window_id for example in selected] == ["w0", "w1"]


def test_contract_probe_prompt_and_marker_detection() -> None:
    example = {
        "instruction": "Devuelve solamente el pasaje final.",
        "source_text": "Texto fuente.",
    }

    prompt = reconstruction_infer.build_contract_probe_prompt(example)
    markers = reconstruction_infer.detect_forbidden_markers("### Respuesta:\nNota: algo")

    assert prompt == (
        "### Instrucción:\n"
        "Devuelve solamente el pasaje final.\n\n"
        "### Pasaje:\n"
        "Texto fuente.\n\n"
        "### Respuesta:\n"
    )
    assert markers == ["###", "Respuesta:", "Nota:", "# "]


def test_contract_probe_summary_counts_failures() -> None:
    records = [
        {
            "output_words": 10,
            "length_ratio": 0.5,
            "empty": False,
            "forbidden_markers": [],
            "starts_with_prompt_scaffold": False,
        },
        {
            "output_words": 0,
            "length_ratio": 0.0,
            "empty": True,
            "forbidden_markers": ["Nota:"],
            "starts_with_prompt_scaffold": True,
        },
    ]

    summary = reconstruction_infer.summarize_contract_probe_records(
        run_id="run",
        model_id="model",
        adapter_path="adapter",
        records=records,
    )

    assert summary["probe_examples"] == 2
    assert summary["empty_count"] == 1
    assert summary["forbidden_marker_count"] == 1
    assert summary["prompt_scaffold_count"] == 1
    assert summary["mean_output_words"] == 5.0
    assert summary["mean_length_ratio"] == 0.25


def test_load_probe_examples_reads_bounded_jsonl(tmp_path: Path) -> None:
    probe_path = tmp_path / "probe.jsonl"
    probe_path.write_text(
        "\n".join(
            [
                json.dumps({"window_id": "w1"}),
                json.dumps({"window_id": "w2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = reconstruction_infer.load_probe_examples(probe_path, limit=1)

    assert examples == [{"window_id": "w1"}]


def test_load_probe_examples_rejects_empty_jsonl(tmp_path: Path) -> None:
    probe_path = tmp_path / "empty.jsonl"
    probe_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain any examples"):
        reconstruction_infer.load_probe_examples(probe_path, limit=8)


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


def test_seq2seq_smoke_mode_writes_non_placeholder_artifact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir, corpus_output_dir, _, split_manifest_path, target_envelopes_path = (
        _build_phase5_artifacts(tmp_path)
    )
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")

    def _fake_train(
        *,
        examples: list[reconstruction_train.TrainingExample],
        config: reconstruction_train.TrainingConfig,
        model_output_dir: Path,
    ) -> dict[str, float | int | str]:
        assert examples
        assert config.training_mode == "seq2seq_smoke"
        model_output_dir.mkdir(parents=True)
        (model_output_dir / "config.json").write_text("{}", encoding="utf-8")
        return {"train_loss": 0.25, "trained_examples": 2}

    monkeypatch.setattr(reconstruction_train, "run_seq2seq_smoke_training", _fake_train)

    reconstruction_train.main(
        [
            "--run-id",
            "phase5-real-smoke",
            "--training-mode",
            "seq2seq_smoke",
            "--max-train-examples",
            "2",
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

    run_dir = tmp_path / "outputs" / "reconstruction" / "runs" / "phase5-real-smoke"
    metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))

    assert metadata["adapter_type"] == "seq2seq_full_model_smoke"
    assert metadata["adapter_is_placeholder"] is False
    assert metadata["adapter_artifact_path"].endswith("model")
    assert metrics["status"] == "trained_smoke"
    assert metrics["training_metrics"]["train_loss"] == 0.25


def test_lora_sft_mode_writes_adapter_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir, corpus_output_dir, _, split_manifest_path, target_envelopes_path = (
        _build_phase5_artifacts(tmp_path)
    )
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")

    def _fake_train(
        *,
        examples: list[reconstruction_train.TrainingExample],
        config: reconstruction_train.TrainingConfig,
        adapter_output_dir: Path,
    ) -> dict[str, float | int | str]:
        assert examples
        assert config.training_mode == "lora_sft"
        assert config.lora_rank == 4
        adapter_output_dir.mkdir(parents=True)
        (adapter_output_dir / "adapter_model.safetensors").write_bytes(b"adapter")
        return {"train_loss": 0.5, "artifact_type": "lora_sft_adapter"}

    monkeypatch.setattr(reconstruction_train, "run_lora_sft_training", _fake_train)

    reconstruction_train.main(
        [
            "--run-id",
            "phase5-lora-smoke",
            "--training-mode",
            "lora_sft",
            "--dataset-mode",
            "contract_smoke",
            "--lora-rank",
            "4",
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

    run_dir = tmp_path / "outputs" / "reconstruction" / "runs" / "phase5-lora-smoke"
    metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))

    assert metadata["adapter_type"] == "lora_sft"
    assert metadata["adapter_is_placeholder"] is False
    assert metadata["adapter_artifact_path"].endswith("adapter")
    assert metrics["status"] == "trained_lora_sft"
    assert metrics["training_metrics"]["artifact_type"] == "lora_sft_adapter"


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
        training_mode="scaffold",
        seed=17,
        wandb_project="rayuela",
        wandb_entity="macayaven",
        wandb_mode="offline",
        max_steps=5,
        max_train_examples=32,
        max_eval_examples=16,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        dtype="bfloat16",
        gradient_checkpointing=False,
        max_source_length=512,
        max_target_length=512,
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
            "training_mode": "scaffold",
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


def test_manifest_write_failure_removes_empty_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, split_manifest_path, target_envelopes_path = _build_phase5_artifacts(tmp_path)
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")
    monkeypatch.setattr(
        reconstruction_train,
        "write_run_manifest",
        lambda manifest, paths=None: (_ for _ in ()).throw(RuntimeError("manifest write failed")),
    )

    with pytest.raises(RuntimeError, match="manifest write failed"):
        reconstruction_train.main(
            [
                "--run-id",
                "phase5-manifest-failure",
                "--split-manifest-path",
                str(split_manifest_path),
                "--target-envelopes-path",
                str(target_envelopes_path),
            ]
        )

    run_dir = tmp_path / "outputs" / "reconstruction" / "runs" / "phase5-manifest-failure"
    indexed_manifest_path = (
        tmp_path / "outputs" / "reconstruction" / "manifests" / "phase5-manifest-failure.json"
    )

    assert not run_dir.exists()
    assert not indexed_manifest_path.exists()


def test_keyboard_interrupt_marks_manifest_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, split_manifest_path, target_envelopes_path = _build_phase5_artifacts(tmp_path)
    monkeypatch.setattr(reconstruction_train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_train, "detect_git_sha", lambda project_root: "feedface")
    monkeypatch.setattr(
        reconstruction_train,
        "load_split_manifest",
        lambda path: (_ for _ in ()).throw(KeyboardInterrupt("stop requested")),
    )

    with pytest.raises(KeyboardInterrupt, match="stop requested"):
        reconstruction_train.main(
            [
                "--run-id",
                "phase5-interrupted",
                "--split-manifest-path",
                str(split_manifest_path),
                "--target-envelopes-path",
                str(target_envelopes_path),
            ]
        )

    manifest_path = (
        tmp_path / "outputs" / "reconstruction" / "runs" / "phase5-interrupted" / "manifest.json"
    )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_payload["status"] == "failed"
    assert manifest_payload["error_message"] == "stop requested"
