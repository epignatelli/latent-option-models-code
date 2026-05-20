#!/usr/bin/env bash
# Create the conda environment and install the lom package.
#
# Usage:
#   bash install.sh          # CUDA 13.2 (default)
#   CUDA=cpu bash install.sh # CPU-only build
set -euo pipefail

CUDA=${CUDA:-cu132}
ENV_NAME=lom

echo "=== Creating conda environment: $ENV_NAME ==="
conda env create -f environment.yml --name "$ENV_NAME" || \
  conda env update -f environment.yml --name "$ENV_NAME" --prune

echo "=== Installing pip dependencies (CUDA=$CUDA) ==="
conda run -n "$ENV_NAME" pip install \
  --index-url "https://download.pytorch.org/whl/$CUDA" \
  --extra-index-url https://pypi.org/simple \
  -r requirements.txt

echo "=== Installing lom package (editable) ==="
conda run -n "$ENV_NAME" pip install -e .

echo ""
echo "Done. Activate with:  conda activate $ENV_NAME"
