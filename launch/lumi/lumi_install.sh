#!/bin/bash
# Native VATIX install on LUMI (run once, on a login node). See docs/lumi.md.
# Usage: LUMI_PROJECT=project_XXXXXXXXX bash launch/lumi/lumi_install.sh
set -euo pipefail

LUMI_PROJECT="${LUMI_PROJECT:?set LUMI_PROJECT=project_XXXXXXXXX}"
VATIX_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VATIX_NATIVE="${VATIX_NATIVE:-/scratch/$LUMI_PROJECT/$USER/native}"   # miniforge + env
# Container we copy MIOpen's kernel db and the RCCL plugin out of (not used at runtime).
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1.sif

mkdir -p "$VATIX_NATIVE"

# Python 3.12 (cray-python only offers 3.10/3.11)
if [ ! -x "$VATIX_NATIVE/miniforge/bin/conda" ]; then
  curl -fsSL -o "$VATIX_NATIVE/Miniforge3.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$VATIX_NATIVE/Miniforge3.sh" -b -p "$VATIX_NATIVE/miniforge"
fi
[ -x "$VATIX_NATIVE/vatix/bin/python" ] || \
  "$VATIX_NATIVE/miniforge/bin/conda" create -y -p "$VATIX_NATIVE/vatix" python=3.12.7
source "$VATIX_NATIVE/miniforge/bin/activate" "$VATIX_NATIVE/vatix"
export PYTHONNOUSERSITE=1

# Pins that keep pip from replacing the ROCm torch or breaking the Hugging Face stack
C="$VATIX_NATIVE/constraints.txt"
cat > "$C" << 'EOF'
torch==2.5.1
torchvision==0.20.1
numpy<2
huggingface_hub==0.26.2
transformers==4.46.3
tokenizers==0.20.3
accelerate==0.34.2
safetensors==0.4.5
EOF

# PyTorch for ROCm
pip install -c "$C" --index-url https://download.pytorch.org/whl/rocm6.2 \
  --extra-index-url https://pypi.org/simple torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2

# Dependencies (moviepy must stay <2 for TensorBoard videos)
pip install -c "$C" hydra-core omegaconf einops easydict ftfy huggingface_hub transformers \
  tokenizers safetensors accelerate diffusers==0.31.0 tensorboard==2.18.0 "moviepy<2" webdataset \
  imageio imageio-ffmpeg scipy scikit-learn torchmetrics torch-fidelity tqdm
# --no-deps: these would otherwise pull a CUDA/CPU torch over the ROCm build
pip install --no-deps clean-fid "torchcodec==0.1.1"
pip install --no-deps -e "$VATIX_CODE"

# MIOpen precompiled kernels (the ROCm wheel lacks gfx90a.kdb; without it convolutions JIT-compile)
MIOPEN_DB="$(python -c 'import site; print(site.getsitepackages()[0])')/torch/share/miopen/db"
[ -s "$MIOPEN_DB/gfx90a.kdb" ] || \
  singularity exec -B "$MIOPEN_DB:/dest" "$SIF" cp -L /opt/rocm/share/miopen/db/gfx90a.kdb /dest/

# RCCL Slingshot plugin (multi-node)
mkdir -p "$VATIX_NATIVE/ofi-rccl"
[ -e "$VATIX_NATIVE/ofi-rccl/librccl-net.so.0.0.0" ] || \
  singularity exec -B "$VATIX_NATIVE/ofi-rccl:/dest" "$SIF" cp -aL /opt/aws-ofi-rccl/librccl-net.so.0.0.0 /dest/
ln -sf librccl-net.so.0.0.0 "$VATIX_NATIVE/ofi-rccl/librccl-net.so"

# WAN 2.1 VAE
cd "$VATIX_CODE"
huggingface-cli download llvictorll/Vatix wan21/wan_2.1_vae.pth --repo-type model --local-dir ./ckpt

echo "Installed: $VATIX_NATIVE/vatix  (source launch/lumi/env_lumi.sh in every job)"
