#!/usr/bin/env bash
# CorpusStudio — native-Linux GPU training bootstrap (Ubuntu 24.04+, NVIDIA RTX 50-series / Blackwell).
#
# Purpose: build a manual first-party QLoRA environment after the final Linux/RTX host exists. This
# script is not native-Linux, offload, long-sequence, or hardware proof. The managed Environment
# Manager workflow is preferred; this remains a transparent diagnostic fallback.
#
# It does NOT install the NVIDIA DRIVER (that needs sudo + a reboot + Secure-Boot MOK enrolment). Do
# that first (see docs/RUNNING_ON_LINUX.md) and confirm `nvidia-smi` shows your GPU; this script then
# builds the userspace Python env only.
#
# Usage:  bash scripts/setup_linux_training.sh  [/path/to/CorpusStudio/engine]
set -euo pipefail

ENGINE_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../engine" && pwd)}"
VENV="${CORPUS_STUDIO_TRAIN_VENV:-/mnt/training-nvme/environments/backend-corpus-studio-manual}"
PY_VERSION="3.12"

echo "== 0. preflight: NVIDIA driver =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi not found. Install the NVIDIA driver first (needs a recent 570+ for Blackwell):"
  echo "     sudo ubuntu-drivers install    # or the graphics-drivers PPA / the .run installer"
  echo "   Reboot, confirm 'nvidia-smi' shows your GPU, then re-run this script."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
if [ ! -d "$(dirname "$VENV")" ]; then
  echo "!! Training NVMe environment root is missing: $(dirname "$VENV")"
  echo "   Prepare and mount /mnt/training-nvme first; see docs/RUNNING_ON_LINUX.md."
  exit 1
fi

echo "== 1. uv (userspace, no sudo) + CPython ${PY_VERSION} =="
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  mkdir -p "$HOME/.local/bin"
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    -o /tmp/uv.tar.gz
  tar xzf /tmp/uv.tar.gz -C /tmp
  cp /tmp/uv-x86_64-unknown-linux-gnu/uv "$HOME/.local/bin/"
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV" python install "$PY_VERSION"

echo "== 2. venv at ${VENV} =="
"$UV" venv "$VENV" --python "$PY_VERSION"
PY="$VENV/bin/python"

echo "== 3. torch 2.11.0 + cu128 (Blackwell sm_120) =="
"$UV" pip install --python "$PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

echo "== 4. first-party training stack (no DeepSpeed/FSDP/NVMe backend) =="
"$UV" pip install --python "$PY" \
  "transformers==5.13.1" "trl==1.8.0" "peft==0.19.0" \
  "accelerate==1.14.0" "datasets==5.0.0" "bitsandbytes==0.49.2" \
  "liger-kernel"

echo "== 5. CorpusStudio engine (editable) from ${ENGINE_DIR} =="
"$UV" pip install --python "$PY" -e "$ENGINE_DIR"

echo "== 6. verify =="
"$PY" - <<'PYEOF'
import torch
print("torch        ", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("capability   ", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
from corpus_studio.platform.host_platform import detect_operating_system
from corpus_studio.platform.gpu_health import probe_gpu_responsive, classify_gpu_health
os_val, resid = detect_operating_system()
print("os / residency", os_val.value, "/", resid.value, "(expect linux / linux_dedicated on bare metal)")
print("gpu health   ", classify_gpu_health(probe_gpu_responsive()))
PYEOF

cat <<EOF

Done. Activate with:  source ${VENV}/bin/activate
Then:  corpus-studio train-check          # must prove readiness on this exact host
This manual environment does not implement DeepSpeed/FSDP/NVMe offload. Establish the sequence-1024
baseline first and follow docs/RUNNING_ON_LINUX.md without promoting installation to proof.
EOF
