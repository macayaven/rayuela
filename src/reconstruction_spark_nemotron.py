#!/usr/bin/env python3
"""
Helpers for the official Spark-aligned Nemotron 3 Nano llama.cpp lane.

This module codifies the published DGX Spark playbook into small, testable
commands rather than relying on ad hoc shell history. It does not run
experiments by itself; it prepares the local build/download/server surfaces and
can write a bounded Phase 4 launchcheck plan targeting the local OpenAI-style
endpoint exposed by llama.cpp.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

PLAYBOOK_URL = "https://build.nvidia.com/spark/nemotron/instructions"
DEFAULT_MODEL_REPO_ID = "unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
DEFAULT_MODEL_FILENAME = "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"
DEFAULT_OPENAI_MODEL_NAME = DEFAULT_MODEL_FILENAME
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 30000
DEFAULT_CTX_SIZE = 16384
DEFAULT_THREADS = 8
DEFAULT_GPU_LAYERS = 99
DEFAULT_CUDA_ARCH = "121"
DEFAULT_GENERATION_TEMPERATURE = 0.0
DEFAULT_GENERATION_MAX_TOKENS = 4096
DEFAULT_SEMANTIC_GENERATION_MAX_TOKENS = 3072
DEFAULT_LLAMA_CPP_ROOT = Path.home() / "llama.cpp"
DEFAULT_MODEL_DIR = Path.home() / "models" / "nemotron3-gguf"
DEFAULT_HF_VENV_ROOT = Path.home() / "nemotron-venv"
DEFAULT_PROJECT_PYTHON_PATH = Path.home() / "education" / "rayuela" / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class SparkNemotronConfig:
    """Local paths and runtime knobs for the Spark Nemotron lane."""

    llama_cpp_root: Path = DEFAULT_LLAMA_CPP_ROOT
    model_dir: Path = DEFAULT_MODEL_DIR
    hf_venv_root: Path = DEFAULT_HF_VENV_ROOT
    model_repo_id: str = DEFAULT_MODEL_REPO_ID
    model_filename: str = DEFAULT_MODEL_FILENAME
    openai_model_name: str = DEFAULT_OPENAI_MODEL_NAME
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT
    ctx_size: int = DEFAULT_CTX_SIZE
    threads: int = DEFAULT_THREADS
    n_gpu_layers: int = DEFAULT_GPU_LAYERS
    cuda_architecture: str = DEFAULT_CUDA_ARCH
    generation_temperature: float = DEFAULT_GENERATION_TEMPERATURE
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS
    semantic_generation_max_tokens: int = DEFAULT_SEMANTIC_GENERATION_MAX_TOKENS

    @property
    def api_base(self) -> str:
        """Return the local OpenAI-compatible base URL."""
        return f"http://localhost:{self.server_port}/v1"

    @property
    def hf_python(self) -> Path:
        """Return the Python executable used for the Hugging Face CLI venv."""
        return self.hf_venv_root / "bin" / "python"

    @property
    def hf_cli(self) -> Path:
        """Return the HF CLI path inside the dedicated playbook venv."""
        return self.hf_venv_root / "bin" / "hf"

    @property
    def llama_server_path(self) -> Path:
        """Return the built llama-server binary path."""
        return self.llama_cpp_root / "build" / "bin" / "llama-server"

    @property
    def model_path(self) -> Path:
        """Return the GGUF model path referenced by llama-server."""
        return self.model_dir / self.model_filename


def build_hf_cli_install_commands(config: SparkNemotronConfig) -> list[list[str]]:
    """Return the official HF CLI venv bootstrap commands."""
    return [
        ["python3", "-m", "venv", str(config.hf_venv_root)],
        [str(config.hf_python), "-m", "pip", "install", "-U", "huggingface_hub"],
    ]


def build_clone_llama_cpp_command(config: SparkNemotronConfig) -> list[str]:
    """Return the published git clone command for llama.cpp."""
    return [
        "git",
        "clone",
        "--recursive",
        "https://github.com/ggml-org/llama.cpp",
        str(config.llama_cpp_root),
    ]


def build_cmake_command(config: SparkNemotronConfig) -> list[str]:
    """Return the Spark-specific CMake configure command."""
    return [
        "cmake",
        "..",
        "-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={config.cuda_architecture}",
        "-DLLAMA_CURL=OFF",
    ]


def build_make_command() -> list[str]:
    """Return the published build command."""
    return ["make", "-j8"]


def build_model_download_command(config: SparkNemotronConfig) -> list[str]:
    """Return the official GGUF download command."""
    return [
        str(config.hf_cli),
        "download",
        config.model_repo_id,
        config.model_filename,
        "--local-dir",
        str(config.model_dir),
    ]


def build_server_command(config: SparkNemotronConfig) -> list[str]:
    """Return the llama-server launch command from the playbook."""
    return [
        str(config.llama_server_path),
        "--model",
        str(config.model_path),
        "--host",
        config.server_host,
        "--port",
        str(config.server_port),
        "--n-gpu-layers",
        str(config.n_gpu_layers),
        "--ctx-size",
        str(config.ctx_size),
        "--threads",
        str(config.threads),
    ]


def build_launchcheck_plan_payload(
    *,
    config: SparkNemotronConfig,
    schedule_id: str,
    run_id: str,
    python_path: Path,
) -> dict[str, object]:
    """Return a bounded Phase 4 launchcheck plan for the Spark lane."""
    return {
        "schedule_id": schedule_id,
        "experiments": [
            {
                "experiment_id": "baseline-live-seeded-1case-1iter-spark-nemotron-launchcheck",
                "run_id": run_id,
                "phase": "phase-4-prompt-baselines",
                "command": [
                    str(python_path),
                    "src/reconstruction_baselines.py",
                    "--run-id",
                    "{run_id}",
                    "--generation-temperature",
                    str(config.generation_temperature),
                    "--generation-max-tokens",
                    str(config.generation_max_tokens),
                    "--semantic-generation-max-tokens",
                    str(config.semantic_generation_max_tokens),
                    "--max-cases",
                    "1",
                    "--max-iterations",
                    "1",
                    "--seed",
                    "42",
                    "--api-base",
                    config.api_base,
                    "--model",
                    config.openai_model_name,
                    "--reasoning-parser",
                    "llamacpp",
                ],
                "timeout_seconds": 2400,
                "metric_path_template": (
                    "outputs/reconstruction/runs/{run_id}/prompt_baseline_summary.json"
                ),
                "metric_key": "controls.style_shift.mean_weighted_objective",
                "higher_is_better": True,
            }
        ],
    }


def write_launchcheck_plan(
    *,
    config: SparkNemotronConfig,
    plan_path: Path,
    schedule_id: str,
    run_id: str,
    python_path: Path = DEFAULT_PROJECT_PYTHON_PATH,
) -> Path:
    """Write one bounded launchcheck plan for the Spark lane."""
    payload = build_launchcheck_plan_payload(
        config=config,
        schedule_id=schedule_id,
        run_id=run_id,
        python_path=python_path,
    )
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan_path


def _shell_join(command: list[str]) -> str:
    """Render one command as a shell-safe line."""
    return " ".join(shlex.quote(part) for part in command)


def print_commands(config: SparkNemotronConfig) -> str:
    """Return the official Spark lane commands as a readable block."""
    lines = [
        f"# Playbook: {PLAYBOOK_URL}",
        "",
        "# 1. Install HF CLI in a dedicated venv",
        *(_shell_join(command) for command in build_hf_cli_install_commands(config)),
        "",
        "# 2. Clone and build llama.cpp for GB10 / SM_121",
        _shell_join(build_clone_llama_cpp_command(config)),
        f"cd {shlex.quote(str(config.llama_cpp_root))}",
        "mkdir -p build && cd build",
        _shell_join(build_cmake_command(config)),
        _shell_join(build_make_command()),
        "",
        "# 3. Download the official GGUF model",
        _shell_join(build_model_download_command(config)),
        "",
        "# 4. Start the OpenAI-compatible server",
        _shell_join(build_server_command(config)),
    ]
    return "\n".join(lines) + "\n"


def _run_command(command: list[str], *, cwd: Path | None = None) -> None:
    """Run one subprocess with streaming stderr/stdout."""
    subprocess.run(command, check=True, cwd=cwd)


def _build_llama_cpp(config: SparkNemotronConfig) -> None:
    """Clone llama.cpp if missing and build it using the official flags."""
    if not config.llama_cpp_root.exists():
        _run_command(build_clone_llama_cpp_command(config))
    build_dir = config.llama_cpp_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _run_command(build_cmake_command(config), cwd=build_dir)
    _run_command(build_make_command(), cwd=build_dir)


def _install_hf_cli(config: SparkNemotronConfig) -> None:
    """Install the Hugging Face CLI in the dedicated venv."""
    for command in build_hf_cli_install_commands(config):
        _run_command(command)


def _download_model(config: SparkNemotronConfig) -> None:
    """Download the official GGUF model artifact."""
    config.model_dir.mkdir(parents=True, exist_ok=True)
    _run_command(build_model_download_command(config))


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for Spark Nemotron helpers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-venv-root", type=Path, default=DEFAULT_HF_VENV_ROOT)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--openai-model-name", default=DEFAULT_OPENAI_MODEL_NAME)
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=DEFAULT_GENERATION_TEMPERATURE,
    )
    parser.add_argument(
        "--generation-max-tokens",
        type=int,
        default=DEFAULT_GENERATION_MAX_TOKENS,
    )
    parser.add_argument(
        "--semantic-generation-max-tokens",
        type=int,
        default=DEFAULT_SEMANTIC_GENERATION_MAX_TOKENS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("print-commands")
    subparsers.add_parser("install-hf-cli")
    subparsers.add_parser("build-llama-cpp")
    subparsers.add_parser("download-model")
    subparsers.add_parser("print-server-command")

    write_plan = subparsers.add_parser("write-launchcheck-plan")
    write_plan.add_argument("--plan-path", type=Path, required=True)
    write_plan.add_argument("--schedule-id", required=True)
    write_plan.add_argument("--run-id", required=True)
    write_plan.add_argument(
        "--python-path",
        type=Path,
        default=DEFAULT_PROJECT_PYTHON_PATH,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Spark Nemotron helper subcommands."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = SparkNemotronConfig(
        llama_cpp_root=args.llama_cpp_root,
        model_dir=args.model_dir,
        hf_venv_root=args.hf_venv_root,
        openai_model_name=args.openai_model_name,
        server_port=args.server_port,
        generation_temperature=args.generation_temperature,
        generation_max_tokens=args.generation_max_tokens,
        semantic_generation_max_tokens=args.semantic_generation_max_tokens,
    )

    if args.command == "print-commands":
        print(print_commands(config), end="")
        return 0
    if args.command == "install-hf-cli":
        _install_hf_cli(config)
        return 0
    if args.command == "build-llama-cpp":
        _build_llama_cpp(config)
        return 0
    if args.command == "download-model":
        _download_model(config)
        return 0
    if args.command == "print-server-command":
        print(_shell_join(build_server_command(config)))
        return 0
    if args.command == "write-launchcheck-plan":
        path = write_launchcheck_plan(
            config=config,
            plan_path=args.plan_path,
            schedule_id=args.schedule_id,
            run_id=args.run_id,
            python_path=args.python_path,
        )
        print(path)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
