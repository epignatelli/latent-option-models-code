#!/usr/bin/env bash
# Bootstrap script: replicate local shell + conda environment on Myriad.
# Run once after first SSH login:
#   bash setup_myriad.sh
set -euo pipefail

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
MINIFORGE_INSTALLER="$HOME/miniforge3_installer.sh"
ZSH_CUSTOM="$HOME/.oh-my-zsh/custom"

echo "==> [1/6] Installing oh-my-zsh"
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    RUNZSH=no KEEP_ZSHRC=yes \
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
else
    echo "    oh-my-zsh already installed, skipping"
fi

echo "==> [2/6] Installing zsh plugins"
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]; then
    git clone https://github.com/zsh-users/zsh-autosuggestions \
        "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
fi
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]; then
    git clone https://github.com/zsh-users/zsh-syntax-highlighting \
        "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
fi

echo "==> [3/6] Writing .zshrc"
cat > "$HOME/.zshrc" << 'ZSHRC'
export PATH="$HOME/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# HuggingFace and wandb caches (home IS scratch on Myriad)
export HF_HOME="$HOME/.cache/huggingface"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"

# CUDA — loaded via modules on Myriad; load before running jobs
# module load cuda/12.8.0

# oh-my-zsh
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source "$ZSH/oh-my-zsh.sh"

alias ccat='pygmentize -g'

# >>> conda initialize >>>
__conda_setup="$("$HOME/miniforge3/bin/conda" 'shell.zsh' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ] && \
        . "$HOME/miniforge3/etc/profile.d/conda.sh" || \
        export PATH="$HOME/miniforge3/bin:$PATH"
fi
unset __conda_setup
# <<< conda initialize <<<
ZSHRC

echo "==> [4/6] Installing miniforge3"
if [ ! -d "$HOME/miniforge3" ]; then
    curl -fsSL "$MINIFORGE_URL" -o "$MINIFORGE_INSTALLER"
    bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
    rm "$MINIFORGE_INSTALLER"
    "$HOME/miniforge3/bin/conda" config --set auto_update_conda false
    "$HOME/miniforge3/bin/conda" config --set auto_activate_base false
    "$HOME/miniforge3/bin/conda" config --add channels conda-forge
else
    echo "    miniforge3 already installed, skipping"
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"

echo "==> [5/6] Creating lom conda environment (Python 3.11)"
if conda env list | grep -q "^lom "; then
    echo "    lom env already exists, skipping"
else
    conda create -y -n lom python=3.11
fi
conda activate lom

echo "==> [6/6] Installing lom dependencies"
# Install PyTorch with CUDA 13.2 (same as local)
pip install torch==2.12.0+cu132 torchvision==0.27.0+cu132 \
    --index-url https://download.pytorch.org/whl/cu132

# NLE needs cmake/flex/bison — install build deps via conda first
conda install -y -n lom cmake flex bison make pkg-config

# Core dependencies
pip install \
    numpy==2.4.6 \
    tqdm==4.67.3 \
    wandb==0.27.0 \
    tyro==1.0.13 \
    gymnasium==1.2.0 \
    pydantic==2.13.4 \
    psutil \
    nle==1.3.0 \
    pytest==9.0.3 \
    pybind11==3.0.4

# Install the lom package itself (editable, from wherever you cloned it)
if [ -d "$HOME/repos/latent-option-models-code" ]; then
    pip install -e "$HOME/repos/latent-option-models-code"
else
    echo "    WARNING: lom repo not found at ~/repos/latent-option-models-code — install manually"
fi

echo ""
echo "Done. To finish:"
echo "  1. Change your default shell:  chsh -s \$(which zsh)"
echo "  2. Log out and back in"
echo "  3. On Myriad, load CUDA before training: module load cuda/12.8.0"
echo "  4. Clone the repo if not done: git clone <url> ~/repos/latent-option-models-code"
