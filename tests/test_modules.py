import pytest
import torch
from lom.modules import VectorQuantizer, SpatioTemporalTransformer

from conftest import BATCH, D_MODEL, LATENT_DIM, N_HEADS, N_LAYERS, S


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
    assert {"vq_loss", "q_loss", "commit_loss", "entropy"} == set(loss_dict.keys())


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


def test_stt_temporal_mask(stt):
    x = torch.randn(BATCH, T, S, D_MODEL)
    mask = torch.zeros(1, 1, T, T)
    mask[0, 0, 0, 1:] = float("-inf")
    out = stt(x, temporal_mask=mask)
    assert out.shape == (BATCH, T, S, D_MODEL)
