from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

import reconstruction_contract


class _CudaStub:
    def __init__(self) -> None:
        self.seed: int | None = None

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed


class _CudnnStub:
    def __init__(self) -> None:
        self.deterministic = False
        self.benchmark = True


class _BackendsStub:
    def __init__(self) -> None:
        self.cudnn = _CudnnStub()


class _TorchStub:
    def __init__(self) -> None:
        self.seed: int | None = None
        self.cuda = _CudaStub()
        self.backends = _BackendsStub()
        self.use_deterministic_algorithms_value: bool | None = None

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def use_deterministic_algorithms(self, value: bool) -> None:
        self.use_deterministic_algorithms_value = value


def test_run_manifest_schema(tmp_path: Path) -> None:
    paths = reconstruction_contract.ReconstructionPaths(project_root=tmp_path)
    run_id = "phase0-dry-run"

    run_dir = reconstruction_contract.prepare_run_directory(run_id, paths=paths)
    manifest = reconstruction_contract.build_run_manifest(
        run_id=run_id,
        phase="phase-0-quality-envelope",
        model_id="no-generation",
        seed=17,
        git_sha="deadbeef",
        config_payload={"phase": 0, "dry_run": True},
        corpus_manifest=tmp_path / "outputs" / "corpus" / "corpus_metadata.json",
        prompt_template_id=None,
        paths=paths,
    )
    manifest_path = reconstruction_contract.write_run_manifest(manifest, paths=paths)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert run_dir == paths.run_dir(run_id)
    assert payload["schema_version"] == reconstruction_contract.MANIFEST_SCHEMA_VERSION
    assert payload["run_id"] == run_id
    assert payload["phase"] == "phase-0-quality-envelope"
    assert payload["model_id"] == "no-generation"
    assert payload["git_sha"] == "deadbeef"
    assert payload["seed"] == {
        "python": 17,
        "numpy": 17,
        "torch": 17,
        "data_splitter": 17,
    }
    assert payload["paths"]["run_dir"] == "outputs/reconstruction/runs/phase0-dry-run"
    assert payload["paths"]["manifest_path"] == (
        "outputs/reconstruction/runs/phase0-dry-run/manifest.json"
    )
    assert set(reconstruction_contract.REQUIRED_MANIFEST_FIELDS).issubset(payload)


def test_seed_is_propagated_to_all_components() -> None:
    torch_stub = _TorchStub()

    reconstruction_contract.seed_everything(23, torch_module=torch_stub)
    first_python = random.random()
    first_numpy = float(np.random.rand())

    bundle = reconstruction_contract.seed_everything(23, torch_module=torch_stub)
    second_python = random.random()
    second_numpy = float(np.random.rand())

    assert bundle.python == 23
    assert bundle.numpy == 23
    assert bundle.torch == 23
    assert bundle.data_splitter == 23
    assert first_python == second_python
    assert first_numpy == second_numpy
    assert torch_stub.seed == 23
    assert torch_stub.cuda.seed == 23
    assert torch_stub.backends.cudnn.deterministic is True
    assert torch_stub.backends.cudnn.benchmark is False
    assert torch_stub.use_deterministic_algorithms_value is True


def test_output_paths_are_project_relative(tmp_path: Path) -> None:
    paths = reconstruction_contract.ReconstructionPaths(project_root=tmp_path)

    manifest_path = paths.manifest_path("demo-run")

    assert reconstruction_contract.to_project_relative(manifest_path, project_root=tmp_path) == (
        "outputs/reconstruction/runs/demo-run/manifest.json"
    )

    with pytest.raises(ValueError):
        paths.run_dir("../escape")

    with pytest.raises(ValueError):
        reconstruction_contract.to_project_relative(
            tmp_path.parent / "outside.json",
            project_root=tmp_path,
        )


def test_failed_runs_are_retained(tmp_path: Path) -> None:
    paths = reconstruction_contract.ReconstructionPaths(project_root=tmp_path)
    run_id = "phase0-failed"
    run_dir = reconstruction_contract.prepare_run_directory(run_id, paths=paths)
    error_log = run_dir / "stderr.log"
    error_log.write_text("traceback", encoding="utf-8")

    manifest = reconstruction_contract.build_run_manifest(
        run_id=run_id,
        phase="phase-0-quality-envelope",
        model_id="no-generation",
        seed=29,
        git_sha="cafebabe",
        config_payload={"phase": 0, "dry_run": True},
        corpus_manifest=tmp_path / "outputs" / "corpus" / "corpus_metadata.json",
        prompt_template_id=None,
        paths=paths,
    )
    reconstruction_contract.write_run_manifest(manifest, paths=paths)

    final_manifest_path = reconstruction_contract.finalize_run_manifest(
        run_id=run_id,
        status=reconstruction_contract.RunStatus.FAILED,
        paths=paths,
        error_message="generation failed",
    )
    payload = json.loads(final_manifest_path.read_text(encoding="utf-8"))

    assert run_dir.exists()
    assert error_log.exists()
    assert payload["status"] == reconstruction_contract.RunStatus.FAILED.value
    assert payload["error_message"] == "generation failed"

    with pytest.raises(FileExistsError):
        reconstruction_contract.prepare_run_directory(run_id, paths=paths)


def test_seed_bundle_is_preserved_when_torch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = reconstruction_contract.SeedBundle(
        python=31,
        numpy=31,
        torch=31,
        data_splitter=31,
    )

    monkeypatch.setattr(reconstruction_contract, "_load_torch_module", lambda: None)

    seeded_bundle = reconstruction_contract.seed_everything(bundle)

    assert seeded_bundle == bundle
    assert random.random() == pytest.approx(0.01227824739797545)
    assert float(np.random.rand()) == pytest.approx(0.28605382166051563)


def test_detect_git_sha_handles_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CompletedProcess:
        stdout = "abc123\n"

    monkeypatch.setattr(
        reconstruction_contract.subprocess,
        "run",
        lambda *args, **kwargs: _CompletedProcess(),
    )
    assert reconstruction_contract.detect_git_sha() == "abc123"

    def _raise_git_error(*args: object, **kwargs: object) -> object:
        raise subprocess.CalledProcessError(returncode=1, cmd=["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(reconstruction_contract.subprocess, "run", _raise_git_error)
    assert reconstruction_contract.detect_git_sha() == "unknown"


def test_main_writes_dry_run_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reconstruction_contract, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reconstruction_contract, "detect_git_sha", lambda project_root: "feedface")

    exit_code = reconstruction_contract.main(
        [
            "--run-id",
            "phase0-cli",
            "--phase",
            "phase-0-quality-envelope",
            "--seed",
            "41",
        ]
    )
    captured = capsys.readouterr()

    manifest_path = (
        tmp_path
        / "outputs"
        / "reconstruction"
        / "runs"
        / "phase0-cli"
        / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["git_sha"] == "feedface"
    assert payload["config_payload"]["dry_run"] is True
    assert payload["config_payload"]["seed"] == 41
    assert (
        "Dry-run manifest written to outputs/reconstruction/runs/phase0-cli/manifest.json"
        in captured.out
    )
