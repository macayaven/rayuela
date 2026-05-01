# Phase 5 DGX Spark Fine-Tuning Playbook Adaptation

Date: 2026-05-01

## Source Playbooks

This plan adapts two official NVIDIA DGX Spark playbooks:

- NVIDIA, "Fine-tune with Pytorch": https://build.nvidia.com/spark/pytorch-fine-tune
- NVIDIA, "Unsloth on DGX Spark": https://build.nvidia.com/spark/unsloth
- Upstream playbook repository pinned during review: `NVIDIA/dgx-spark-playbooks@599cf838a007d6121929ce84f117de0e9f371e2c`

## Recommendation

Use the NVIDIA PyTorch PEFT/TRL path as the first production adapter lane, then add
Unsloth as an acceleration lane after the PyTorch LoRA path is green on our Rayuela
contract dataset.

The PyTorch path matches our current Phase 5 runner more directly: Dockerized
NGC PyTorch, Hugging Face `transformers`, `datasets`, `trl`, `peft`, and
`bitsandbytes`, with separate full SFT, LoRA, and QLoRA examples. The Unsloth
path is attractive for throughput, but it introduces a second failure surface:
Triton/kernel/CUDA compatibility. For the article research loop, correctness
and reproducibility should come before speed.

DGX Spark is still a narrow hardware/software target. Do not treat generic
PyPI/PyTorch advice as portable to this machine. The adapter lane should prefer
NVIDIA-published NGC PyTorch images and NVIDIA-validated dependency sets. Host
Python environments should not be used for real training, and `pip install torch`
inside the training container is forbidden unless we deliberately rebuild the
whole lane around a new NVIDIA image.

## Rayuela Adaptation

Keep Rayuela's existing run contract as the source of truth:

- immutable run IDs under `outputs/reconstruction/runs/<run_id>/`
- split-specific `training_dataset/{train,val,test}.jsonl`
- `training_config.json`, `training_metrics.json`, and `checkpoint_metadata.json`
- explicit `git_sha`, `dataset_mode`, `training_mode`, and model identifier

Adapt NVIDIA's script pattern into `src/reconstruction_train.py` instead of
vendoring the playbook scripts directly:

- replace Alpaca loading with Rayuela JSONL examples
- format examples as instruction/input/response SFT text
- use `AutoModelForCausalLM` plus `SFTTrainer` for the first serious adapter
- use `LoraConfig` for LoRA and `BitsAndBytesConfig` plus
  `prepare_model_for_kbit_training` for QLoRA
- save only adapter artifacts for LoRA/QLoRA runs, not a full merged model by default
- keep `seq2seq_smoke` as the fast health check for the training container

## Execution Order

1. Bootstrap a DGX Spark fine-tuning container with PyTorch dependencies:
   `transformers`, `peft`, `datasets`, `trl`, `bitsandbytes`, and `hf_transfer`.
2. Run the existing `seq2seq_smoke` health check inside that container to verify
   mounts, GPU visibility, Hugging Face cache, and output paths.
3. Add `training_mode=lora_sft` to Rayuela using the NVIDIA 8B LoRA recipe shape,
   but with Rayuela's contract dataset.
4. Run a tiny LoRA SFT smoke on a small open model with `max_steps=1`.
5. Run a contract LoRA batch on the selected base model with a bounded sample
   size, then evaluate final-answer-only behavior before any style claim.
6. Add `training_mode=qlora_sft` using NVIDIA's 70B QLoRA recipe shape only after
   LoRA SFT is stable.
7. Evaluate Unsloth as `training_mode=unsloth_lora_sft` only after PyTorch LoRA
   produces a valid adapter and metrics.

## Initial Parameter Defaults

For the first PyTorch LoRA adapter:

- `dtype=bfloat16`
- `seq_length=2048`
- `lora_rank=8`
- `lora_alpha=16`
- `lora_dropout=0`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `learning_rate=1e-4`
- `max_steps=20` for smoke, then expand only after artifact validation
- `gradient_checkpointing=true` for larger models

For QLoRA:

- `load_in_4bit=true`
- `bnb_4bit_quant_type=nf4`
- `bnb_4bit_use_double_quant=true`
- `bnb_4bit_compute_dtype=bfloat16`
- `target_modules=all-linear`

## Risks And Controls

- Hardware/software mismatch: use `nvcr.io/nvidia/pytorch:25.11-py3` as the
  default playbook image until a newer NVIDIA DGX Spark playbook supersedes it.
  Record any image override in the run manifest or research log.
- PyTorch replacement: install fine-tuning dependencies under constraints
  generated from the NGC image's existing Torch, CUDA, NCCL, cuDNN, NVIDIA, and
  Triton packages.
- Silent CPU fallback: fail before installation if `nvidia-smi`, `nvcc`, or
  `torch.cuda.is_available()` is not healthy inside the container.
- Dependency drift: pin the upstream playbook commit in logs and record package
  versions in `training_metrics.json`.
- Gated model downloads: fail early if `HF_TOKEN` is missing for gated models.
- UMA memory pressure: use small smoke runs first; if memory appears available
  but allocation fails, flush host caches before rerun as NVIDIA recommends.
- Speed optimism: do not adopt Unsloth until PyTorch LoRA establishes a baseline.
- Research overclaiming: a lower training loss is not a literary result. The
  first adapter claim must be completion reliability and output-contract behavior.

## Expected Products

- `lora_sft` training mode in `src/reconstruction_train.py`
- a DGX Spark container bootstrap script
- one tiny LoRA smoke artifact with non-placeholder adapter metadata
- one contract-adapter run on the locked Phase 5 dataset
- Phase 6 comparison of prompt-only vs contract-adapter completion reliability
