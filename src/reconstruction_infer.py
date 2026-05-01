#!/usr/bin/env python3
"""
Phase 5 inference scaffold for reconstruction experiments.

This module focuses on adapter metadata loading and smoke-safe inference plumbing.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

FORBIDDEN_CONTRACT_MARKERS = (
    "###",
    "Instrucción:",
    "Pasaje:",
    "Respuesta:",
    "Nota:",
    "Note:",
    "Explicación:",
    "Explanation:",
    "markdown",
    "razonamiento",
    "reasoning",
    "Thinking Process",
    "<think>",
    "</think>",
    "* ",
    "- ",
    "# ",
)


def load_saved_adapter(checkpoint_metadata_path: Path) -> dict[str, Any]:
    """Load adapter metadata saved by the Phase 5 training scaffold."""
    payload = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    if bool(payload.get("adapter_is_placeholder", False)):
        raise ValueError(
            "checkpoint metadata refers to a placeholder adapter from a scaffold-only run; "
            "inference is unavailable until a real adapter artifact exists."
        )
    return {
        "run_id": payload["run_id"],
        "git_sha": payload["git_sha"],
        "phase": payload["phase"],
        "model_id": payload["model_id"],
        "adapter_type": payload["adapter_type"],
        "adapter_artifact_path": payload["adapter_artifact_path"],
        "adapter_is_placeholder": bool(payload.get("adapter_is_placeholder", False)),
        "config_path": payload["config_path"],
        "tokenizer_config_path": payload["tokenizer_config_path"],
        "metrics_path": payload["metrics_path"],
        "split_counts": payload["split_counts"],
    }


def build_contract_probe_prompt(example: dict[str, Any]) -> str:
    """Build the causal-LM prompt used for final-answer contract probing."""
    return (
        "### Instrucción:\n"
        f"{example['instruction']}\n\n"
        "### Pasaje:\n"
        f"{example['source_text']}\n\n"
        "### Respuesta:\n"
    )


def detect_forbidden_markers(
    text: str,
    markers: tuple[str, ...] = FORBIDDEN_CONTRACT_MARKERS,
) -> list[str]:
    """Return output-contract markers found in generated text."""
    lowered = text.lower()
    return [marker for marker in markers if marker.lower() in lowered]


def summarize_contract_probe_records(
    *,
    run_id: str,
    model_id: str,
    adapter_path: str | None,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize contract-probe records without interpreting literary quality."""
    if not records:
        raise ValueError("contract probe requires at least one generated record")
    return {
        "run_id": run_id,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "probe_examples": len(records),
        "empty_count": sum(1 for record in records if bool(record["empty"])),
        "forbidden_marker_count": sum(
            1 for record in records if bool(record["forbidden_markers"])
        ),
        "prompt_scaffold_count": sum(
            1 for record in records if bool(record["starts_with_prompt_scaffold"])
        ),
        "mean_output_words": sum(float(record["output_words"]) for record in records)
        / len(records),
        "mean_length_ratio": sum(float(record["length_ratio"]) for record in records)
        / len(records),
        "records": records,
    }


def load_probe_examples(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Load a deterministic prefix of JSONL probe examples."""
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line))
            if limit > 0 and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"probe split {path} did not contain any examples")
    return examples


def _load_contract_probe_backend() -> dict[str, Any]:  # pragma: no cover
    """Load optional generation dependencies only for contract probes."""
    try:
        torch = importlib.import_module("torch")
        peft = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Contract probing requires torch, transformers, and peft. "
            "Run inside the pinned NVIDIA DGX Spark fine-tuning container."
        ) from exc
    return {
        "torch": torch,
        "PeftModel": peft.PeftModel,
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoTokenizer": transformers.AutoTokenizer,
    }


def run_contract_probe(  # pragma: no cover
    *,
    checkpoint_metadata_path: Path,
    probe_split_path: Path,
    output_path: Path,
    limit: int,
    max_new_tokens: int,
    base_only: bool,
) -> dict[str, Any]:
    """Generate a small contract probe for an adapter or its base model."""
    backend = _load_contract_probe_backend()
    metadata = load_saved_adapter(checkpoint_metadata_path)
    model_id = str(metadata["model_id"])
    adapter_path = str(metadata["adapter_artifact_path"])
    tokenizer_path = model_id if base_only else adapter_path

    tokenizer = backend["AutoTokenizer"].from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = backend["AutoModelForCausalLM"].from_pretrained(
        model_id,
        dtype=backend["torch"].bfloat16,
        device_map="auto",
    )
    if not base_only:
        model = backend["PeftModel"].from_pretrained(model, adapter_path)
    model.eval()

    records: list[dict[str, Any]] = []
    for example in load_probe_examples(probe_split_path, limit=limit):
        prompt = build_contract_probe_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
            model.device
        )
        with backend["torch"].no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        source_words = len(str(example["source_text"]).split())
        output_words = len(text.split())
        records.append(
            {
                "window_id": example["window_id"],
                "output_text": text,
                "output_words": output_words,
                "source_words": source_words,
                "length_ratio": (output_words / source_words) if source_words else 0.0,
                "empty": not bool(text),
                "forbidden_markers": detect_forbidden_markers(text),
                "starts_with_prompt_scaffold": text.startswith("###"),
            }
        )

    summary = summarize_contract_probe_records(
        run_id=f"base-{model_id}-contract-probe" if base_only else str(metadata["run_id"]),
        model_id=model_id,
        adapter_path=None if base_only else adapter_path,
        records=records,
    )
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the Phase 5 inference scaffold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-metadata-path", type=Path, required=True)
    parser.add_argument("--probe-split-path", type=Path, default=None)
    parser.add_argument("--contract-probe-output-path", type=Path, default=None)
    parser.add_argument("--probe-examples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--base-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load adapter metadata and print a minimal confirmation."""
    args = build_argument_parser().parse_args(argv)
    if args.contract_probe_output_path is not None:
        if args.probe_split_path is None:
            raise ValueError("--probe-split-path is required with --contract-probe-output-path")
        summary = run_contract_probe(
            checkpoint_metadata_path=args.checkpoint_metadata_path,
            probe_split_path=args.probe_split_path,
            output_path=args.contract_probe_output_path,
            limit=args.probe_examples,
            max_new_tokens=args.max_new_tokens,
            base_only=args.base_only,
        )
        print("Contract probe completed")
        print(f"  Examples: {summary['probe_examples']}")
        print(f"  Empty outputs: {summary['empty_count']}")
        print(f"  Forbidden marker outputs: {summary['forbidden_marker_count']}")
        return 0

    payload = load_saved_adapter(args.checkpoint_metadata_path)
    print("Loaded adapter metadata")
    print(f"  Run ID: {payload['run_id']}")
    print(f"  Model:  {payload['model_id']}")
    print(f"  Adapter: {payload['adapter_artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
