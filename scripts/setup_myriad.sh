#!/usr/bin/env bash
# Bootstrap script: replicate shell environment on Myriad.
# Run once after first SSH login:
#   bash setup_myriad.sh
set -euo pipefail

ZSH_CUSTOM="$HOME/.oh-my-zsh/custom"
ZSH_VERSION="5.9"
ZSH_PREFIX="$HOME/.local"

echo "==> [1/6] Installing zsh $ZSH_VERSION from source"
if command -v zsh &>/dev/null; then
    echo "    zsh already available at $(which zsh), skipping build"
else
    cd /tmp
    curl -fsSL "https://sourceforge.net/projects/zsh/files/zsh/${ZSH_VERSION}/zsh-${ZSH_VERSION}.tar.xz/download" \
        -o "zsh-${ZSH_VERSION}.tar.xz"
    tar -xf "zsh-${ZSH_VERSION}.tar.xz"
    cd "zsh-${ZSH_VERSION}"
    ./configure --prefix="$ZSH_PREFIX" --without-tcsetpgrp
    make -j4
    make install
    export PATH="$ZSH_PREFIX/bin:$PATH"
    echo "    zsh installed at $ZSH_PREFIX/bin/zsh"
    cd "$HOME"
    rm -rf "/tmp/zsh-${ZSH_VERSION}" "/tmp/zsh-${ZSH_VERSION}.tar.xz"
fi

echo "==> [2/6] Installing oh-my-zsh"
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    RUNZSH=no KEEP_ZSHRC=yes \
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
else
    echo "    already installed, skipping"
fi

echo "==> [3/6] Installing zsh plugins"
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]; then
    git clone https://github.com/zsh-users/zsh-autosuggestions \
        "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
fi
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]; then
    git clone https://github.com/zsh-users/zsh-syntax-highlighting \
        "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
fi

echo "==> [4/5] Installing Miniconda"
CONDA_DIR="$HOME/miniconda3"
if [ -d "$CONDA_DIR" ]; then
    echo "    Miniconda already installed at $CONDA_DIR, skipping"
else
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "==> [5/6] Creating lom conda environment"
if conda env list | grep -q "^lom "; then
    echo "    lom env already exists, skipping"
else
    conda env create -f "$HOME/repos/latent-option-models-code/environment.yml"
fi

echo "==> [6/6] Writing .zshrc"
cat > "$HOME/.zshrc" << 'ZSHRC'
export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"

export HF_HOME="$HOME/.cache/huggingface"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"

# Conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source "$ZSH/oh-my-zsh.sh"

alias ccat='pygmentize -g'
ZSHRC

echo ""
echo "Done. Change your default shell and re-login:"
echo "  chsh -s \$(which zsh)"
echo "  (if chsh is not allowed, add 'exec \$HOME/.local/bin/zsh' to ~/.bash_profile)"
