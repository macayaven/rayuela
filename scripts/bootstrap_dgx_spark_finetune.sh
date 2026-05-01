#!/usr/bin/env bash
set -euo pipefail

LANE="${1:-pytorch}"
IMAGE="${RAYUELA_FINETUNE_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
WORKDIR_IN_CONTAINER="${RAYUELA_CONTAINER_WORKDIR:-/workspace/rayuela}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

case "$LANE" in
  pytorch | unsloth) ;;
  *)
    echo "usage: $0 [pytorch|unsloth]" >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for DGX Spark fine-tuning bootstrap" >&2
  exit 1
fi

echo "Rayuela fine-tuning runs are container-only; host Python environments are not used."
echo "Container image: $IMAGE"
echo "Lane: $LANE"

mkdir -p "$HF_CACHE"

if [[ "$LANE" == "pytorch" ]]; then
  INSTALL_COMMAND='pip install --constraint /tmp/rayuela-torch-constraints.txt "transformers==4.57.1" "peft==0.17.1" "datasets==4.3.0" "trl==0.26.1" "bitsandbytes==0.49.2" "hf_transfer==0.1.9"'
else
  INSTALL_COMMAND='pip install --constraint /tmp/rayuela-torch-constraints.txt "transformers==4.57.1" "peft==0.17.1" "hf_transfer==0.1.9" "datasets==4.3.0" "trl==0.26.1" && pip install --no-deps unsloth unsloth_zoo "bitsandbytes==0.49.2"'
fi

exec docker run --gpus all --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -it --rm \
  -v "$PWD:$WORKDIR_IN_CONTAINER" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -w "$WORKDIR_IN_CONTAINER" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    nvidia-smi
    nvcc --version
    python - <<'PY'
import sys
import torch

print(f'torch: {torch.__version__}')
print(f'torch cuda: {torch.version.cuda}')
if not torch.cuda.is_available():
    sys.exit('CUDA is not visible to PyTorch inside the container')
print(f'cuda device count: {torch.cuda.device_count()}')
print(f'cuda device 0: {torch.cuda.get_device_name(0)}')
PY
    pip list --format=freeze | grep -iE '^(torch|nvidia|cuda|cudnn|nccl|triton)' > /tmp/rayuela-torch-constraints.txt
    echo '--- protected NVIDIA/PyTorch packages ---'
    cat /tmp/rayuela-torch-constraints.txt
    $INSTALL_COMMAND
    python - <<'PY'
import importlib.util
mods = ['torch', 'transformers', 'datasets', 'peft', 'trl', 'bitsandbytes']
if '$LANE' == 'unsloth':
    mods.extend(['unsloth', 'unsloth_zoo'])
for mod in mods:
    print(f'{mod}: {importlib.util.find_spec(mod) is not None}')
PY
    exec bash
  "
