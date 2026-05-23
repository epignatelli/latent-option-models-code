"""Draw the LOM system diagram as a PNG."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── colour palette ────────────────────────────────────────────────────────────
C_DATA   = "#d6eaf8"   # blue-ish  – data pipeline
C_TOK    = "#d5f5e3"   # green     – tokenisation / embedding
C_ENC    = "#fdebd0"   # orange    – LAM / option encoder
C_DYN    = "#f9ebea"   # red-ish   – dynamics models
C_TRAIN  = "#f4ecf7"   # purple    – training loop
C_ORCH   = "#fdfefe"   # near-white– orchestration
C_BORDER = "#2c3e50"   # dark      – all borders
C_ARROW  = "#2c3e50"

FONT = "monospace"
FS   = 8.5   # base font size

fig, ax = plt.subplots(figsize=(18, 26))
ax.set_xlim(0, 18)
ax.set_ylim(0, 26)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, title, lines, bg, title_bg=None, fs=FS):
    """Draw a labelled box with text lines."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.2,
        edgecolor=C_BORDER,
        facecolor=bg,
        zorder=2,
    )
    ax.add_patch(rect)

    title_h = 0.42
    if title:
        tbg = title_bg if title_bg else bg
        trect = mpatches.FancyBboxPatch(
            (x, y + h - title_h), w, title_h,
            boxstyle="round,pad=0.02",
            linewidth=0,
            edgecolor=tbg,
            facecolor="#2c3e50",
            zorder=3,
        )
        ax.add_patch(trect)
        ax.text(
            x + w / 2, y + h - title_h / 2, title,
            ha="center", va="center",
            fontsize=fs + 0.5, fontweight="bold",
            fontfamily=FONT, color="white", zorder=4,
        )

    line_h = (h - title_h - 0.15) / max(len(lines), 1)
    for i, line in enumerate(lines):
        ty = y + h - title_h - 0.12 - (i + 0.5) * line_h
        bold = line.startswith("**") and line.endswith("**")
        txt = line.strip("*")
        ax.text(
            x + 0.18, ty, txt,
            ha="left", va="center",
            fontsize=fs - (0.5 if not bold else 0),
            fontweight="bold" if bold else "normal",
            fontfamily=FONT, color="#1a1a2e", zorder=4,
        )


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.1, my, label, fontsize=FS - 1.5, color="#555",
                fontfamily=FONT, va="center")


def sub_box(ax, x, y, w, h, label, lines, bg="#ffffff"):
    """Lighter inner box for sub-components."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        linewidth=0.8,
        edgecolor="#888",
        facecolor=bg,
        zorder=3,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + 0.12, y + h - 0.22, label,
                fontsize=FS - 0.5, fontweight="bold",
                fontfamily=FONT, color="#333", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.22, y + h - 0.22 - (i + 1) * 0.3, line,
                fontsize=FS - 1, fontfamily=FONT, color="#222", zorder=4)


# ═════════════════════════════════════════════════════════════════════════════
# 1. ORCHESTRATION  (top)
# ═════════════════════════════════════════════════════════════════════════════
box(ax, 0.3, 23.4, 17.4, 2.3,
    "EXPERIMENT ORCHESTRATION  —  experiments/benchmark_nao_top10/run.sh",
    [
        "4× A100 80 GB  |  6 jobs total: LAM × 3 seeds  +  LOM × 3 seeds  (1 GPU each, parallel)",
        "",
        'for seed in [0, 1, 2]:',
        '  GPU[slot] ← bash -c "python3 -m scripts.pretrain lam  --data.horizon 1  --model.num_options 98  ... && touch lam_seedN/done"',
        '  GPU[slot] ← bash -c "python3 -m scripts.pretrain lom  --data.horizon 128 --model.num_options 256 ... && touch lom_seedN/done"',
        "",
        "_flush: wait for all background jobs.    done sentinel → skip already-completed jobs on re-run.",
    ],
    C_ORCH)

arrow(ax, 9, 23.4, 9, 22.85)

# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
box(ax, 0.3, 18.5, 17.4, 4.6,
    "DATA PIPELINE  —  lom/dataset.py",
    [], C_DATA)

# index.npz
sub_box(ax, 0.55, 21.55, 5.0, 1.25, "index.npz  (built once by scripts/convert_to_npz.py)",
        ["paths:   (16 482,) object  —  per-game .npz file path",
         "lengths: (16 482,) int32   —  frame count per game",
         "median game length ≈ 7 917 frames"],
        "#e8f4fb")

# GameBuffer
sub_box(ax, 0.55, 18.75, 8.2, 2.65, "GameBuffer  (background thread, atomic state swap)",
        ["hot pool: buffer_size=200 games in RAM  (~6 GB)",
         "refresh:  replace 10% of buffer every 60 s  (weights ∝ valid_starts)",
         "_load(i):  tty_chars (T,24,80) uint8",
         "           tty_colors clipped to [0,31] uint8",
         "           → np.stack → (T, 24, 80, 2) uint8",
         "sample():  pick game by length-weighted prob,  pick t ∈ [ctx-1, T-horizon-1]"],
        "#d6eaf8")

# NpzTrajectoryDataset outputs
sub_box(ax, 9.0, 18.75, 8.5, 2.65, "NpzTrajectoryDataset.__getitem__  →  DataLoader (batch_size=256, num_workers=0)",
        ["history:       game[t-3 : t+1]          (B,   4, 24, 80, 2)  uint8",
         "next_frame:    game[t+1]                 (B,      24, 80, 2)  uint8",
         "future_frame:  game[t+horizon]           (B,      24, 80, 2)  uint8",
         "sequence:      game[t+1 : t+horizon+1]   (B, 128, 24, 80, 2)  uint8  [LOM only]",
         "",
         "→  .to(device)  →  AMP autocast (float16)"],
        "#d6eaf8")

# arrows inside data box
arrow(ax, 5.55, 22.2, 6.5, 22.2, "")
ax.annotate("", xy=(4.75, 21.55), xytext=(4.75, 21.41),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2), zorder=5)

arrow(ax, 8.75, 20.1, 9.0, 20.1)

arrow(ax, 9, 18.5, 9, 18.0)

# ═════════════════════════════════════════════════════════════════════════════
# 3. TOKENISATION + PATCH EMBEDDING
# ═════════════════════════════════════════════════════════════════════════════
box(ax, 0.3, 15.6, 17.4, 2.15,
    "TOKENISATION + PATCH EMBEDDING  —  lom/modules.py  (inside each model)",
    [
        "ScreenTokeniser:   (*, 24, 80, 2) uint8  →  token_id = char × 32 + color  →  (*, 24, 80) long  ∈ [0, 8 192)",
        "",
        "PatchEmbedding (patch_size=4):",
        "  (B, T, 24, 80) long  →  char_embed  →  (B, T, 24, 80, 256)",
        "  group 4×4 patches   →  (B, T, 6, 20, 16, 256)  →  patch_proj  →  (B, T, 120, 256)",
        "  n_tokens = (24/4) × (80/4) = 120 patch tokens per frame",
    ],
    C_TOK)

arrow(ax, 9, 15.6, 9, 15.1)

# ═════════════════════════════════════════════════════════════════════════════
# 4A. LAM  (left column)
# ═════════════════════════════════════════════════════════════════════════════
box(ax, 0.3, 7.3, 8.4, 8.05,
    "LAM — BASELINE  (horizon=1, num_options=98)",
    [], C_ENC)

sub_box(ax, 0.55, 11.8, 7.9, 3.35, "LAM Encoder  —  LatentActionModel(codebook_size=98, horizon=1)",
        ["Input seq  [h₀ h₁ h₂ h₃ | OPT | x_{t+1}]     T = 6, S = 120",
         "OPT token masked: cannot attend to x_{t+1}",
         "BiDir STP-Transformer  (4 layers, 4 heads, d=256)",
         "  spatial attn: (B×T, 120, 256)  bidirectional",
         "  temporal attn: (B×120, 6, 256) bidirectional",
         "pool OPT token → mean over S → (B, 256) → vq_proj → (B, 512)",
         "VectorQuantizer: cosine dist, STE, entropy reg, dead-code reset",
         "  z_q  (B, 512),   vq_loss = q_loss + β·commit - λ·entropy"],
        "#fef9e7")

sub_box(ax, 0.55, 9.15, 7.9, 2.45, "LAM Dynamics  —  DynamicsModel(predict_sequence=False)",
        ["cond = action_proj(z_q)              (B, 1, 1, 256)  — broadcast",
         "emb  = patch_embed(history) + cond   (B, 4, 120, 256)",
         "Causal STP-Transformer  (4 layers, 4 heads, d=256)",
         "  temporal attn is causal (lower-triangular mask)",
         "take hidden[:, -1]  →  state_head  →  (B, 120, 16×8192)",
         "_unpatch_logits     →  (B, 1920, 8192)"],
        "#fef5f5")

sub_box(ax, 0.55, 7.55, 7.9, 1.45, "LAM Loss",
        ["lam_recon = CrossEntropy( logits,  tokenise(x_{t+1}) )   # (B×1920,) targets",
         "total     = lam_recon + vq_loss"],
        "#f0f0f0")

# ═════════════════════════════════════════════════════════════════════════════
# 4B. LOM  (right column)
# ═════════════════════════════════════════════════════════════════════════════
box(ax, 9.1, 7.3, 8.6, 8.05,
    "LOM — PROPOSED  (horizon=128, num_options=256)",
    [], C_ENC)

sub_box(ax, 9.35, 13.25, 8.1, 1.95, "option_lam  —  LatentActionModel(codebook_size=256, horizon=128)",
        ["seq: [h₀ h₁ h₂ h₃ | OPT | x_{t+1} … x_{t+128}]   T=133, S=120",
         "OPT masked from all 128 future frames",
         "BiDir STP-Transformer  →  VQ(256 codes)  →  z_opt  (B, 512)"],
        "#fef9e7")

sub_box(ax, 9.35, 11.35, 8.1, 1.75, "action_lam  —  LatentActionModel(codebook_size=98, horizon=1, condition_dim=512)",
        ["seq: [h₀ h₁ h₂ h₃ | z_opt_tok | OPT | x_{t+1}]   T=7, S=120",
         "z_opt projected to d_model, prepended as condition token",
         "BiDir STP-Transformer  →  VQ(98 codes)  →  z_act  (B, 512)"],
        "#fef9e7")

sub_box(ax, 9.35, 9.65, 8.1, 1.55, "lam_dynamics  —  DynamicsModel(predict_sequence=False)",
        ["cond = action_proj(z_act)                (B, 1, 1, 256)",
         "same causal transformer over history as LAM",
         "→  logits (B, 1920, 8192)    target: x_{t+1}"],
        "#fef5f5")

sub_box(ax, 9.35, 7.95, 8.1, 1.55, "lom_dynamics  —  DynamicsModel(predict_sequence=False, option_dim=512)",
        ["cond = action_proj(z_act) + goal_proj(z_opt)   (B, 1, 1, 256)",
         "same causal transformer over history",
         "→  logits (B, 1920, 8192)    target: x_{t+128}"],
        "#fef5f5")

sub_box(ax, 9.35, 7.55, 8.1, 0.28, "LOM Loss  =  CE(lam_logits, x_{t+1})  +  CE(lom_logits, x_{t+128})  +  vq_loss_opt  +  vq_loss_act",
        [], "#f0f0f0")

# separator label between LAM and LOM
ax.text(8.85, 11.3, "vs", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#888", fontfamily=FONT)

# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
arrow(ax, 9, 7.3, 9, 6.75)

box(ax, 0.3, 3.8, 17.4, 3.25,
    "TRAINING LOOP  —  lom/training.py  (Trainer.train)",
    [
        "Optimiser:  AdamW  (fused CUDA)   |  decay params: dim≥2,  no-decay: bias+norms",
        "LR schedule: linear warmup 0→3e-4 over 1 000 steps, cosine decay to 1e-6 over 100 000 steps",
        "AMP:         float16 + GradScaler  |  grad_clip = 1.0",
        "Data:        steps_per_epoch=10 000 × batch=256  =  2.56 M samples/epoch   (10 epochs total)",
        "torch.compile:  enabled (each sub-model compiled independently)",
        "",
        "log every   50 steps  →  wandb train/{total_loss, recon, vq_loss, entropy, lr}",
        "eval every 500 steps  →  wandb val/   +  checkpoint saved locally + uploaded to wandb artifact",
        "wandb config includes: git commit, dirty flag, full argv, all hyper-parameters",
    ],
    C_TRAIN)

# ═════════════════════════════════════════════════════════════════════════════
# 6. SPATIO-TEMPORAL TRANSFORMER  (bottom box, shared detail)
# ═════════════════════════════════════════════════════════════════════════════
arrow(ax, 9, 3.8, 9, 3.2)

box(ax, 0.3, 0.3, 17.4, 3.25,
    "SPATIO-TEMPORAL TRANSFORMER (shared building block)  —  lom/modules.py",
    [
        "Input: (B, T, S=120, D=256)  →  spatial_pos embed (S,)  +  temporal_pos embed (T,)",
        "",
        "Each SpatioTemporalBlock:",
        "  1. Spatial  attn:  reshape → (B×T, 120, 256)  →  bidir MHA  →  reshape back   [within-frame]",
        "  2. Temporal attn:  reshape → (B×120, T, 256)  →  MHA (causal for Dynamics, bidir for LAM)  →  reshape back",
        "  3. MLP:  GELU  (4× expansion)",
        "",
        "Flash Attention (scaled_dot_product_attention) with custom additive mask (OPT mask for LAM encoder)",
        "4 layers  ×  4 heads  ×  d=256  |  output: LayerNorm(x)  →  (B, T, S, D)",
    ],
    C_TOK)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(9, 25.9, "LOM System Diagram  —  NAO-TOP10 Benchmark",
        ha="center", va="center",
        fontsize=13, fontweight="bold", fontfamily=FONT, color="#1a1a2e")

plt.tight_layout(pad=0)
out = "experiments/benchmark_nao_top10/system_diagram.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
