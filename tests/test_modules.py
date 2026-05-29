import pytest
import torch
from lom.modules import PatchEmbedding, VectorQuantizer, SpatioTemporalTransformer
from torch.nn.attention.flex_attention import create_block_mask

from conftest import BATCH, CONTEXT, D_MODEL, LATENT_DIM, N_HEADS, N_LAYERS, OBS_H, OBS_W, S, VOCAB


# --------------------------------------------------------------------------- #
# --- VectorQuantizer ------------------------------------------------------- #
# --------------------------------------------------------------------------- #

@pytest.fixture
def vq():
    return VectorQuantizer(
        latent_dim=LATENT_DIM,
        num_options=32,
        dropout=0.0,
        entropy_weight=0.01,
        vq_beta=0.25,
        vq_reset_thresh=0,
    )


def test_vq_output_shapes(vq):
    z = torch.randn(BATCH, LATENT_DIM)
    z_q, _, indices = vq(z)
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


def test_vq_loss_keys(vq):
    z = torch.randn(BATCH, LATENT_DIM)
    _, loss_dict, _ = vq(z)
    assert {"vq_loss", "commit_loss", "entropy"} == set(loss_dict.keys())


def test_vq_straight_through(vq):
    z = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
    z_q, loss_dict, _ = vq(z)
    z_q.sum().backward()
    assert z.grad is not None


def test_vq_lookup(vq):
    indices = torch.randint(0, 32, (BATCH,))
    result = vq.lookup(indices)
    assert result.shape == (BATCH, LATENT_DIM)


# --------------------------------------------------------------------------- #
# --- SpatioTemporalTransformer --------------------------------------------- #
# --------------------------------------------------------------------------- #

T = 5
MAX_T = 8


@pytest.fixture
def stt():
    return SpatioTemporalTransformer(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_spatial_positions=S,
        max_temporal_len=MAX_T,
    )


def test_stt_output_shape(stt):
    x = torch.randn(BATCH, T, S, D_MODEL)
    out = stt(x)
    assert out.shape == (BATCH, T, S, D_MODEL)


def test_stt_causal():
    model = SpatioTemporalTransformer(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_spatial_positions=S,
        max_temporal_len=MAX_T,
        causal_temporal=True,
    )
    x = torch.randn(BATCH, T, S, D_MODEL)
    out = model(x)
    assert out.shape == (BATCH, T, S, D_MODEL)


def test_stt_exceeds_max_len(stt):
    x = torch.randn(BATCH, MAX_T + 1, S, D_MODEL)
    with pytest.raises(AssertionError):
        stt(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BlockMask requires CUDA")
def test_stt_temporal_mask():
    model = SpatioTemporalTransformer(
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        n_spatial_positions=S, max_temporal_len=MAX_T,
    ).cuda()
    x = torch.randn(BATCH, T, S, D_MODEL, device="cuda")

    def opt_mask(_b, _h, q_idx, kv_idx):
        return ~((q_idx == 0) & (kv_idx > 0))

    block_mask = create_block_mask(opt_mask, B=None, H=None, Q_LEN=T, KV_LEN=T, device="cuda")
    out = model(x, temporal_mask=block_mask)
    assert out.shape == (BATCH, T, S, D_MODEL)


# --------------------------------------------------------------------------- #
# --- PatchEmbedding -------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def test_patch_embed_no_patch_shape():
    pe = PatchEmbedding(VOCAB, D_MODEL, OBS_H, OBS_W, patch_size=1)
    x = torch.randint(0, VOCAB, (BATCH, CONTEXT, OBS_H, OBS_W))
    out = pe(x)
    assert out.shape == (BATCH, CONTEXT, OBS_H * OBS_W, D_MODEL)


def test_patch_embed_with_patch_shape():
    # OBS_H=OBS_W=4, patch_size=2 → 2×2=4 tokens per frame
    pe = PatchEmbedding(VOCAB, D_MODEL, OBS_H, OBS_W, patch_size=2)
    x = torch.randint(0, VOCAB, (BATCH, CONTEXT, OBS_H, OBS_W))
    out = pe(x)
    n_tokens = (OBS_H // 2) * (OBS_W // 2)
    assert out.shape == (BATCH, CONTEXT, n_tokens, D_MODEL)


def test_patch_embed_token_count():
    pe = PatchEmbedding(VOCAB, D_MODEL, OBS_H, OBS_W, patch_size=2)
    assert pe.n_tokens == (OBS_H // 2) * (OBS_W // 2)


def test_patch_embed_bad_divisibility():
    with pytest.raises(AssertionError):
        PatchEmbedding(VOCAB, D_MODEL, OBS_H, OBS_W, patch_size=3)  # 4 not divisible by 3
