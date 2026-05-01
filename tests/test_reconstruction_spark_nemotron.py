from __future__ import annotations

import json
from pathlib import Path

import pytest

import reconstruction_spark_nemotron


def test_default_paths_follow_spark_playbook() -> None:
    config = reconstruction_spark_nemotron.SparkNemotronConfig()

    assert config.model_repo_id == "unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
    assert config.model_filename == "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"
    assert config.server_port == 30000
    assert config.api_base == "http://localhost:30000/v1"
    assert config.generation_temperature == 0.0
    assert config.generation_max_tokens == 4096
    assert config.semantic_generation_max_tokens == 3072
    assert config.llama_server_path == Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
    assert config.model_path == (
        Path.home() / "models" / "nemotron3-gguf" / "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"
    )


def test_build_server_command_matches_official_playbook() -> None:
    config = reconstruction_spark_nemotron.SparkNemotronConfig(
        llama_cpp_root=Path("/tmp/llama.cpp"),
        model_dir=Path("/tmp/models/nemotron3-gguf"),
    )

    command = reconstruction_spark_nemotron.build_server_command(config)

    assert command == [
        "/tmp/llama.cpp/build/bin/llama-server",
        "--model",
        "/tmp/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--n-gpu-layers",
        "99",
        "--ctx-size",
        "16384",
        "--threads",
        "8",
    ]


def test_build_download_command_uses_hf_cli_from_playbook_venv() -> None:
    config = reconstruction_spark_nemotron.SparkNemotronConfig(
        hf_venv_root=Path("/tmp/nemotron-venv"),
        model_dir=Path("/tmp/models/nemotron3-gguf"),
    )

    command = reconstruction_spark_nemotron.build_model_download_command(config)

    assert command == [
        "/tmp/nemotron-venv/bin/hf",
        "download",
        "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
        "--local-dir",
        "/tmp/models/nemotron3-gguf",
    ]


def test_write_launchcheck_plan_targets_llamacpp_api_base(tmp_path: Path) -> None:
    config = reconstruction_spark_nemotron.SparkNemotronConfig(
        server_port=30000,
        openai_model_name="Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
    )
    plan_path = tmp_path / "launchcheck.json"

    written = reconstruction_spark_nemotron.write_launchcheck_plan(
        config=config,
        plan_path=plan_path,
        schedule_id="guided-phase4-spark-nemotron-launchcheck-20260311a",
        run_id="phase4-launchcheck-spark-nemotron-20260311a",
    )

    payload = json.loads(written.read_text(encoding="utf-8"))
    command = payload["experiments"][0]["command"]

    assert payload["schedule_id"] == "guided-phase4-spark-nemotron-launchcheck-20260311a"
    assert payload["experiments"][0]["run_id"] == "phase4-launchcheck-spark-nemotron-20260311a"
    assert "--api-base" in command
    assert command[command.index("--api-base") + 1] == "http://localhost:30000/v1"
    assert command[command.index("--generation-temperature") + 1] == "0.0"
    assert command[command.index("--generation-max-tokens") + 1] == "4096"
    assert command[command.index("--semantic-generation-max-tokens") + 1] == "3072"
    assert command[command.index("--model") + 1] == "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"
    assert command[command.index("--reasoning-parser") + 1] == "llamacpp"


def test_launchcheck_plan_accepts_custom_generation_budgets(tmp_path: Path) -> None:
    config = reconstruction_spark_nemotron.SparkNemotronConfig(
        generation_temperature=0.1,
        generation_max_tokens=4096,
        semantic_generation_max_tokens=2048,
    )
    plan_path = tmp_path / "launchcheck.json"

    written = reconstruction_spark_nemotron.write_launchcheck_plan(
        config=config,
        plan_path=plan_path,
        schedule_id="guided-phase4-spark-nemotron-launchcheck-20260311b",
        run_id="phase4-launchcheck-spark-nemotron-20260311b",
    )

    payload = json.loads(written.read_text(encoding="utf-8"))
    command = payload["experiments"][0]["command"]

    assert command[command.index("--generation-temperature") + 1] == "0.1"
    assert command[command.index("--generation-max-tokens") + 1] == "4096"
    assert command[command.index("--semantic-generation-max-tokens") + 1] == "2048"


def test_print_commands_includes_reproducible_build_download_and_server_steps() -> None:
    config = reconstruction_spark_nemotron.SparkNemotronConfig(
        llama_cpp_root=Path("/tmp/llama.cpp"),
        hf_venv_root=Path("/tmp/nemotron-venv"),
        model_dir=Path("/tmp/models/nemotron3-gguf"),
    )

    rendered = reconstruction_spark_nemotron.print_commands(config)

    assert "https://build.nvidia.com/spark/nemotron/instructions" in rendered
    assert "python3 -m venv /tmp/nemotron-venv" in rendered
    assert "git clone --recursive https://github.com/ggml-org/llama.cpp /tmp/llama.cpp" in rendered
    assert "-DCMAKE_CUDA_ARCHITECTURES=121" in rendered
    assert "/tmp/nemotron-venv/bin/hf download" in rendered
    assert "/tmp/llama.cpp/build/bin/llama-server --model" in rendered


def test_main_print_server_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = reconstruction_spark_nemotron.main(
        [
            "--llama-cpp-root",
            "/tmp/llama.cpp",
            "--model-dir",
            "/tmp/models/nemotron3-gguf",
            "print-server-command",
        ]
    )

    assert exit_code == 0
    assert "/tmp/llama.cpp/build/bin/llama-server --model" in capsys.readouterr().out


def test_main_write_launchcheck_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan_path = tmp_path / "plans" / "launchcheck.json"

    exit_code = reconstruction_spark_nemotron.main(
        [
            "--server-port",
            "31000",
            "write-launchcheck-plan",
            "--plan-path",
            str(plan_path),
            "--schedule-id",
            "guided-test",
            "--run-id",
            "phase4-test",
            "--python-path",
            "/tmp/project/.venv/bin/python",
        ]
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    command = payload["experiments"][0]["command"]
    assert exit_code == 0
    assert str(plan_path) in capsys.readouterr().out
    assert command[0] == "/tmp/project/.venv/bin/python"
    assert command[command.index("--api-base") + 1] == "http://localhost:31000/v1"


def test_main_command_subprocess_paths_are_injectable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path | None]] = []

    def fake_run(command: list[str], *, cwd: Path | None = None) -> None:
        calls.append((command, cwd))

    monkeypatch.setattr(reconstruction_spark_nemotron, "_run_command", fake_run)

    hf_root = tmp_path / "hf"
    model_dir = tmp_path / "models"
    llama_root = tmp_path / "llama.cpp"

    assert (
        reconstruction_spark_nemotron.main(
            [
                "--hf-venv-root",
                str(hf_root),
                "install-hf-cli",
            ]
        )
        == 0
    )
    assert (
        reconstruction_spark_nemotron.main(
            [
                "--model-dir",
                str(model_dir),
                "--hf-venv-root",
                str(hf_root),
                "download-model",
            ]
        )
        == 0
    )
    assert (
        reconstruction_spark_nemotron.main(
            [
                "--llama-cpp-root",
                str(llama_root),
                "build-llama-cpp",
            ]
        )
        == 0
    )

    assert calls[0][0] == ["python3", "-m", "venv", str(hf_root)]
    assert calls[1][0] == [
        str(hf_root / "bin" / "python"),
        "-m",
        "pip",
        "install",
        "-U",
        "huggingface_hub",
    ]
    assert calls[2][0][:3] == [
        str(hf_root / "bin" / "hf"),
        "download",
        "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
    ]
    assert calls[3][0][:3] == ["git", "clone", "--recursive"]
    assert calls[4][0] == [
        "cmake",
        "..",
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=121",
        "-DLLAMA_CURL=OFF",
    ]
    assert calls[4][1] == llama_root / "build"
    assert calls[5][0] == ["make", "-j8"]
