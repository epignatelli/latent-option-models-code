#!/usr/bin/env bash
# Bootstrap script: replicate shell environment on Myriad.
# Run once after first SSH login:
#   bash setup_myriad.sh
set -euo pipefail

ZSH_CUSTOM="$HOME/.oh-my-zsh/custom"

echo "==> [1/3] Installing oh-my-zsh"
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    RUNZSH=no KEEP_ZSHRC=yes \
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
else
    echo "    already installed, skipping"
fi

echo "==> [2/3] Installing zsh plugins"
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]; then
    git clone https://github.com/zsh-users/zsh-autosuggestions \
        "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
fi
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]; then
    git clone https://github.com/zsh-users/zsh-syntax-highlighting \
        "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
fi

echo "==> [3/3] Writing .zshrc"
cat > "$HOME/.zshrc" << 'ZSHRC'
export PATH="$HOME/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

export HF_HOME="$HOME/.cache/huggingface"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"

export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source "$ZSH/oh-my-zsh.sh"

alias ccat='pygmentize -g'
ZSHRC

echo ""
echo "Done. Change your default shell and re-login:"
echo "  chsh -s \$(which zsh)"
