import pytest
import torch
from lom.lam import LatentActionModel, ObservableTransitionModel
from lom.encoders import EMAEncoder

from conftest import BATCH, CONTEXT, D_MODEL, HORIZON, LATENT_DIM, N_HEADS, N_LAYERS, OBS_H, OBS_W, S, VOCAB

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention backward requires CUDA"
)


def make_lam(in_dim=D_MODEL):
    return LatentActionModel(in_dim=in_dim, latent_dim=LATENT_DIM, num_options=32)


def make_dynamics(predict_sequence=False):
    return ObservableTransitionModel(
        vocab_size=VOCAB, obs_h=OBS_H, obs_w=OBS_W,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        context_length=CONTEXT, latent_dim=LATENT_DIM,
        predict_sequence=predict_sequence,
        horizon=HORIZON if predict_sequence else 1,
    )


def _screen(*leading):
    """Random (char, color) screen tensor: (*leading, H, W, 2) uint8."""
    chars  = torch.randint(0, 256, (*leading, OBS_H, OBS_W, 1), dtype=torch.uint8)
    colors = torch.randint(0, 32,  (*leading, OBS_H, OBS_W, 1), dtype=torch.uint8)
    return torch.cat([chars, colors], dim=-1)


def history():
    return _screen(BATCH, CONTEXT)


def single_frame():
    return _screen(BATCH)


def frame_sequence(k):
    return _screen(BATCH, k)


# --------------------------------------------------------------------------- #
# --- LatentActionModel (VQ bottleneck) ------------------------------------- #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def test_lam_output_shapes():
    lam = make_lam()
    x = torch.randn(BATCH, D_MODEL)
    z_q, _, indices = lam(x)
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


@needs_cuda
def test_lam_backward():
    lam = make_lam().cuda()
    _, loss_dict, _ = lam(torch.randn(BATCH, D_MODEL).cuda())
    loss_dict["vq_loss"].backward()


# --------------------------------------------------------------------------- #
# --- Encoder classes ------------------------------------------------------- #
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# --- DynamicsModel --------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def action():
    return torch.randn(BATCH, LATENT_DIM)


@torch.no_grad()
def test_dynamics_single_frame():
    dyn = make_dynamics()
    logits = dyn(history(), action())
    assert logits.shape == (BATCH, S, VOCAB)


@torch.no_grad()
def test_dynamics_sequence_teacher():
    dyn = make_dynamics(predict_sequence=True)
    teacher = frame_sequence(HORIZON)
    logits = dyn(history(), action(), horizon=HORIZON, teacher_frames=teacher)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


@torch.no_grad()
def test_dynamics_sequence_autoregressive():
    dyn = make_dynamics(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)



@needs_cuda
def test_dynamics_backward():
    dyn = make_dynamics().cuda()
    logits = dyn(history().cuda(), action().cuda())
    target = torch.randint(0, VOCAB, (BATCH, S), device="cuda")
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()


# --------------------------------------------------------------------------- #
# --- Patch embedding in models --------------------------------------------- #
# --------------------------------------------------------------------------- #

PATCH = 2  # OBS_H=OBS_W=4, so 4//2=2 tokens per dim → 4 tokens/frame


def make_dynamics_patched(predict_sequence=False):
    return ObservableTransitionModel(
        vocab_size=VOCAB, obs_h=OBS_H, obs_w=OBS_W, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, context_length=CONTEXT,
        latent_dim=LATENT_DIM, predict_sequence=predict_sequence,
        horizon=HORIZON if predict_sequence else 1, patch_size=PATCH,
    )


@torch.no_grad()
def test_dynamics_patch_single_frame():
    dyn = make_dynamics_patched()
    logits = dyn(history(), action())
    assert logits.shape == (BATCH, S, VOCAB)


@torch.no_grad()
def test_dynamics_patch_sequence_teacher():
    dyn = make_dynamics_patched(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON, teacher_frames=frame_sequence(HORIZON))
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


@torch.no_grad()
def test_dynamics_patch_sequence_autoregressive():
    dyn = make_dynamics_patched(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


@needs_cuda
def test_dynamics_patch_backward():
    dyn = make_dynamics_patched().cuda()
    logits = dyn(history().cuda(), action().cuda())
    target = torch.randint(0, VOCAB, (BATCH, S), device="cuda")
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()


# --------------------------------------------------------------------------- #
# --- EMAEncoder ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def test_ema_encoder_update_changes_weights():
    linear = torch.nn.Linear(D_MODEL, LATENT_DIM)
    ema = EMAEncoder(linear, decay=0.0)  # decay=0 → EMA params fully replaced by online
    with torch.no_grad():
        for p in linear.parameters():
            p.add_(torch.ones_like(p))
    before = next(ema.encoder.parameters()).clone()
    ema.update(linear)
    after = next(ema.encoder.parameters())
    assert not torch.allclose(before, after)
