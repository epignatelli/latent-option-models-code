"""Shared small-dimension constants for fast CPU tests."""

from lom.modules import TOKEN_VOCAB

OBS_H = 4
OBS_W = 4
S = OBS_H * OBS_W       # 16 spatial positions
VOCAB = TOKEN_VOCAB      # 8192 = 256 chars × 32 colors
LATENT_DIM = 8
D_MODEL = 16
N_LAYERS = 1
N_HEADS = 2
CONTEXT = 4
HORIZON = 3
BATCH = 2
