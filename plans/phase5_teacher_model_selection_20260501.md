# Phase 5 Teacher Model Selection

Date: 2026-05-01
Status: initial recommendation

## Goal

Select the next local teacher model for Phase 4/5 style-transfer data generation
on the DGX Spark workstation.

The teacher must improve on the current Nemotron lane in two ways:

- produce final-answer-only passages reliably, without reasoning-only failures
- generate high-quality Spanish literary rewrites suitable for scored
  distillation into the Phase 5 style-transfer adapter dataset

## Current Local Constraints

- DGX Spark: NVIDIA GB10, CUDA 13.0, 128 GB unified memory.
- Current GPU consumers include the Nemotron `llama-server` process and one
  vLLM engine, so a large teacher probe should not start until the active
  services are intentionally drained.
- The first Qwen3 launch attempt at `--gpu-memory-utilization 0.85` requested
  `101.74 GiB`, while the clean desktop workstation reported `99.72 GiB` free.
  The tracked compose profile therefore uses `0.80` to avoid a brittle startup
  threshold.
- The second launch progressed to model load at 32k context but did not publish
  the API socket after several minutes of CPU-heavy initialization. The tracked
  first-probe profile uses `--max-model-len 16384` and `--enforce-eager` so the
  first teacher batch prioritizes startup reliability over maximum throughput.
- Existing local OpenAI-compatible services:
  - `http://localhost:30000/v1`: `Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf`
  - `http://localhost:8001/v1`: `google/medgemma-1.5-4b-it`
  - `http://localhost:8000/v1`: container is present, but the API currently
    resets the connection
- No checked local model cache was found for Mistral Small 4, Qwen3 30B 2507,
  or gpt-oss.

## Selection Criteria

1. Explicit non-reasoning or final-only control.
2. vLLM, llama.cpp, or NVIDIA-supported serving path on ARM64/CUDA 13 class
   software.
3. Feasible memory profile for a single DGX Spark.
4. Strong multilingual and Spanish generation.
5. Open weights with a research-compatible license.
6. Low integration risk with the current `reconstruction_baselines.py`
   OpenAI-compatible client.

## Recommendation

Use `Qwen/Qwen3-30B-A3B-Instruct-2507` as the first new teacher candidate.

Keep `mistralai/Mistral-Small-4-119B-2603-NVFP4` as the higher-upside second
probe after the Qwen lane proves the new teacher workflow. Keep
`openai/gpt-oss-120b` as a comparison model, not the first replacement teacher.

## Why Qwen3-30B First

Qwen3-30B-A3B-Instruct-2507 is the best first bet because it directly addresses
the failure that blocked the Nemotron teacher lane without introducing a large
serving-stack experiment. The model card says this checkpoint supports only
non-thinking mode and does not emit `<think>` blocks, so it matches the
final-answer-only requirement directly.

It is also much easier to fit into the current workstation state. Qwen documents
a single-process vLLM serve command, while Mistral Small 4 NVFP4 currently
documents a custom vLLM image and a `--tensor-parallel-size 2` serving command.
That does not rule Mistral out, but it makes it a risky first move on one DGX
Spark. Qwen lets us generate usable teacher data sooner, then test Mistral Small
4 once the replacement lane is already producing scoreable artifacts.

The risk is that Qwen3-30B may be less stylistically rich than a 100B+ teacher.
That is acceptable for the first replacement lane because the current blocking
problem is completion reliability, not maximum literary quality.

## Candidate Ranking

### 1. Qwen3-30B-A3B-Instruct-2507

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Strengths:
  - explicitly non-thinking; the card says it does not emit `<think>` blocks
  - 30.5B total, 3.3B active MoE, so it is much easier to serve than 100B+
    teachers
  - Apache 2.0
  - vLLM, SGLang, llama.cpp, Ollama, and LM Studio support are documented
  - long context is available, though Phase 5 should start with shorter context
- Risks:
  - smaller teacher may be less stylistically rich than Mistral Small 4 or
    gpt-oss-120b
  - prior project Qwen lane used hidden reasoning; this specific instruct model
    should be treated as a fresh endpoint, not assumed equivalent
- Decision: first teacher candidate.

### 2. Mistral Small 4 119B A6B

- Model: `mistralai/Mistral-Small-4-119B-2603`
- Preferred local variant: `mistralai/Mistral-Small-4-119B-2603-NVFP4`, if
  serving support is healthy
- Strengths:
  - explicit `reasoning_effort="none"` for final-only teacher generation
  - multilingual support including Spanish
  - vLLM and llama.cpp deployment paths
  - Apache 2.0
  - recent release: March 2026
- Risks:
  - official NVFP4 card currently recommends a custom Docker image and
    `--tensor-parallel-size 2`, which is not a clean first fit for one DGX Spark
  - needs very recent serving stack
  - the full BF16/FP8 checkpoint is too large for casual single-service probing
  - quantized path must be validated before a long unattended batch
- Decision: second teacher probe after Qwen produces scoreable artifacts.

### 3. OpenAI gpt-oss-120b

- Model: `openai/gpt-oss-120b`
- Strengths:
  - open-weight Apache 2.0 model with strong reasoning benchmark positioning
  - native MXFP4 quantization is designed to fit on a single 80 GB GPU
  - OpenAI and NVIDIA both publish deployment paths
  - vLLM, llama.cpp, Ollama, Transformers, and NVIDIA NIM ecosystem support are
    available
- Risks:
  - it is a reasoning model and uses the harmony format; misconfigured serving
    can recreate the same "reasoning instead of final answer" failure class
  - the project needs literary final-output reliability more urgently than
    benchmark reasoning depth
  - fine-tuning and serving require format discipline
- Decision: second comparison teacher after a non-reasoning teacher is healthy.

### 4. Mistral Medium 3.5 128B

- Model: `mistralai/Mistral-Medium-3.5-128B`
- Strengths:
  - very recent April 2026 flagship model
  - dense 128B with configurable reasoning
  - potentially strong for long, literary, and agentic tasks
- Risks:
  - dense 128B is a poor first fit for a single DGX Spark teacher lane
  - released under a Modified MIT license, so terms need review before article
    pipeline dependency
  - very new ecosystem support; likely a time sink before we get text examples
- Decision: exclude from the first pass.

### 5. Mistral Small 3.2 24B

- Model: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- Strengths:
  - stable vLLM path and lower memory demand
  - strong instruction-following update over Small 3.1
  - Apache 2.0
- Risks:
  - superseded by Mistral Small 4 for this use case
  - less likely to be a top-quality literary teacher
- Decision: operational fallback if both Small 4 and Qwen3 30B stall.

## Immediate Execution Plan

1. Drain or pause nonessential GPU services before a large teacher probe.
2. Start with a short Qwen3-30B-A3B-Instruct-2507 serving probe in a pinned DGX
   Spark container or the existing ARM64 vLLM image.
3. Run the same 2-case teacher probe currently used for Phase 4, but cap
   generation tightly and require final-answer-only parsing.
4. If Qwen fails within the timebox, switch to Mistral Small 3.2 24B as the
   operational fallback; if Qwen succeeds, schedule Mistral Small 4 NVFP4 as the
   higher-quality comparison probe.
5. Distill only scoreable teacher outputs above the existing threshold into a
   new Phase 5 dataset ID.
6. Compare the new teacher against the best existing Nemotron offset batch by:
   - scoreable completion rate
   - weighted objective
   - final-answer extraction success
   - semantic preservation pass rate
   - manual literary usefulness label

## Launch Log

- 2026-05-01 17:42 CEST: stopped `sparkclaw-vllm-vision`,
  `sparkclaw-vllm-medgemma`, and the Nemotron `llama-server` on port `30000`.
- 2026-05-01 17:43 CEST: first Qwen3 launch at 32k context and
  `gpu_memory_utilization=0.85` failed because vLLM requested more memory than
  was free after desktop overhead.
- 2026-05-01 17:50 CEST: switched first-probe serving to 16k context,
  `gpu_memory_utilization=0.80`, and eager execution.
- 2026-05-01 18:03 CEST: endpoint became live at `http://localhost:8000/v1`.
  vLLM loaded `16` safetensor shards in `343.37` seconds, used `56.93 GiB` for
  model weights, reported `41.99 GiB` available for KV cache, and exposed
  `max_model_len=16384`.
- 2026-05-01 18:04 CEST: final-answer smoke request succeeded with
  `reasoning=null` and no `<think>` block.
- 2026-05-01 18:05 CEST: one-case Phase 4 pipeline smoke completed in the
  warmed endpoint with no rescue needed. The case was scoreable, with weighted
  objective `0.210`; semantic tolerance failed, while style, target, length,
  and lexical guardrails passed.
- 2026-05-01 18:06 CEST: three-case Qwen3 teacher batch
  `phase4-qwen3-instruct-3cases-20260501a` completed with `3/3` scoreable
  cases, no rescue, no visible meta suffix, mean weighted objective `0.197`,
  and median `0.176`. All three failed semantic tolerance under the current
  scorer, so the generated examples remain provisional.
- 2026-05-01 18:07 CEST: distilled the three scoreable Qwen3 examples into
  `phase5-style-distill-qwen3-instruct-20260501a` using the existing
  `min_weighted_objective=0.14` threshold. Split counts: `2 train`, `0 val`,
  `1 test`.
- 2026-05-01 18:15 CEST: added and launched the first agentic Qwen3 teacher
  loop. Its first offset-3 cycle produced a valid teacher batch and a 5-example
  cumulative distillation dataset, then exposed a loop bug: the loop expected
  `manifest.json` while the distiller writes `distill_manifest.json`.
- 2026-05-01 18:20 CEST: fixed the manifest path bug, made the launcher accept
  `RAYUELA_QWEN3_START_OFFSET` and `RAYUELA_QWEN3_EXTRA_CASES`, then relaunched
  the forever loop as `qwen3-teacher-loop-20260501182025` from offset `6`.
- 2026-05-01 19:28 CEST: stopped the loop after it exhausted available case
  offsets and began producing zero-case cycles. The useful cumulative distilled
  dataset had reached `8` examples: `7 train`, `0 val`, `1 test`. Added a
  `no_cases_available` guard so future loops halt before creating empty
  distillation artifacts.

## Sources Checked

- NVIDIA DGX Spark hardware guide:
  https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- NVIDIA DGX Spark NGC guidance:
  https://docs.nvidia.com/dgx/dgx-spark/ngc.html
- NVIDIA DGX Spark Unsloth playbook:
  https://build.nvidia.com/spark/unsloth/instructions
- Mistral Small 4 model card:
  https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
- Mistral Small 4 docs:
  https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03
- Mistral Small 4 NVFP4 model card:
  https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4
- Qwen3-30B-A3B-Instruct-2507 model card:
  https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen3-235B-A22B-Instruct-2507 model card:
  https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507
- OpenAI gpt-oss announcement:
  https://openai.com/index/introducing-gpt-oss/
- OpenAI gpt-oss-120b Hugging Face model card:
  https://huggingface.co/openai/gpt-oss-120b
- NVIDIA NIM listing for gpt-oss-120b:
  https://build.nvidia.com/openai/gpt-oss-120b
- Mistral Small 3.2 model card:
  https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
- Mistral Medium 3.5 model card:
  https://huggingface.co/mistralai/Mistral-Medium-3.5-128B
