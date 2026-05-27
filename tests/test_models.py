import pytest
import torch
from lom.models import DynamicsModel, LatentActionModel

from conftest import BATCH, CONTEXT, D_MODEL, HORIZON, LATENT_DIM, N_HEADS, N_LAYERS, OBS_H, OBS_W, S, VOCAB


def make_lam(horizon=1, condition_dim=None, two_encoder=False):
    return LatentActionModel(
        vocab_size=VOCAB,
        obs_h=OBS_H,
        obs_w=OBS_W,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        context_length=CONTEXT,
        latent_dim=LATENT_DIM,
        codebook_size=32,
        horizon=horizon,
        condition_dim=condition_dim,
        two_encoder=two_encoder,
    )


def make_dynamics(option_dim=None, predict_sequence=False):
    return DynamicsModel(
        vocab_size=VOCAB,
        obs_h=OBS_H,
        obs_w=OBS_W,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        context_length=CONTEXT,
        latent_dim=LATENT_DIM,
        option_dim=option_dim,
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
# --- LatentActionModel ----------------------------------------------------- #
# --------------------------------------------------------------------------- #

def test_lam_single_future():
    lam = make_lam()
    z_q, _, indices = lam(history(), single_frame())
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


def test_lam_sequence_future():
    lam = make_lam(horizon=HORIZON)
    z_q, _, indices = lam(history(), frame_sequence(HORIZON))
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


def test_lam_with_condition():
    lam = make_lam(condition_dim=LATENT_DIM)
    cond = torch.randn(BATCH, LATENT_DIM)
    z_q, _, indices = lam(history(), single_frame(), condition=cond)
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


def test_lam_backward():
    lam = make_lam()
    _, loss_dict, _ = lam(history(), single_frame())
    loss_dict["vq_loss"].backward()


def test_lam_opt_ignores_future():
    lam = make_lam().eval()
    hist = history()
    with torch.no_grad():
        z1, _, _ = lam(hist, single_frame())
        z2, _, _ = lam(hist, single_frame())  # different random future
    assert torch.allclose(z1, z2), "OPT output must not depend on future content"


# --------------------------------------------------------------------------- #
# --- LatentActionModel two_encoder=True ------------------------------------ #
# --------------------------------------------------------------------------- #

def test_two_encoder_lam_shapes():
    lam = make_lam(two_encoder=True)
    z_q, _, indices = lam(history(), single_frame())
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


def test_two_encoder_lam_sequence_future():
    lam = make_lam(horizon=HORIZON, two_encoder=True)
    z_q, _, indices = lam(history(), frame_sequence(HORIZON))
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


def test_two_encoder_lam_with_condition():
    lam = make_lam(condition_dim=LATENT_DIM, two_encoder=True)
    cond = torch.randn(BATCH, LATENT_DIM)
    z_q, _, _ = lam(history(), single_frame(), condition=cond)
    assert z_q.shape == (BATCH, LATENT_DIM)


def test_two_encoder_lam_backward():
    lam = make_lam(two_encoder=True)
    _, loss_dict, _ = lam(history(), single_frame())
    loss_dict["vq_loss"].backward()


def test_lam_serialise_roundtrip(tmp_path):
    lam = make_lam().eval()
    hist, fut = history(), single_frame()
    with torch.no_grad():
        z_before, _, _ = lam(hist, fut)
    lam.save(str(tmp_path / "lam.pt"))
    lam2 = LatentActionModel.load(str(tmp_path / "lam.pt")).eval()
    with torch.no_grad():
        z_after, _, _ = lam2(hist, fut)
    assert torch.allclose(z_before, z_after)


def test_dynamics_serialise_roundtrip(tmp_path):
    dyn = make_dynamics().eval()
    hist, act = history(), action()
    with torch.no_grad():
        logits_before = dyn(hist, act)
    dyn.save(str(tmp_path / "dyn.pt"))
    dyn2 = DynamicsModel.load(str(tmp_path / "dyn.pt")).eval()
    with torch.no_grad():
        logits_after = dyn2(hist, act)
    assert torch.allclose(logits_before, logits_after)


# --------------------------------------------------------------------------- #
# --- DynamicsModel --------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def action():
    return torch.randn(BATCH, LATENT_DIM)


def test_dynamics_single_frame():
    dyn = make_dynamics()
    logits = dyn(history(), action())
    assert logits.shape == (BATCH, S, VOCAB)


def test_dynamics_sequence_teacher():
    dyn = make_dynamics(predict_sequence=True)
    teacher = frame_sequence(HORIZON)
    logits = dyn(history(), action(), horizon=HORIZON, teacher_frames=teacher)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


def test_dynamics_sequence_autoregressive():
    dyn = make_dynamics(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


def test_dynamics_with_option_code():
    dyn = make_dynamics(option_dim=LATENT_DIM)
    option_code = torch.randn(BATCH, LATENT_DIM)
    logits = dyn(history(), action(), option_code=option_code)
    assert logits.shape == (BATCH, S, VOCAB)


def test_dynamics_backward():
    dyn = make_dynamics()
    logits = dyn(history(), action())
    target = torch.randint(0, VOCAB, (BATCH, S))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()


# --------------------------------------------------------------------------- #
# --- Patch embedding in models --------------------------------------------- #
# --------------------------------------------------------------------------- #

PATCH = 2  # OBS_H=OBS_W=4, so 4//2=2 tokens per dim → 4 tokens/frame


def make_lam_patched(horizon=1):
    return LatentActionModel(
        vocab_size=VOCAB, obs_h=OBS_H, obs_w=OBS_W, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, context_length=CONTEXT,
        latent_dim=LATENT_DIM, codebook_size=32, horizon=horizon, patch_size=PATCH,
    )


def make_dynamics_patched(predict_sequence=False):
    return DynamicsModel(
        vocab_size=VOCAB, obs_h=OBS_H, obs_w=OBS_W, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, context_length=CONTEXT,
        latent_dim=LATENT_DIM, predict_sequence=predict_sequence,
        horizon=HORIZON if predict_sequence else 1, patch_size=PATCH,
    )


def test_lam_patch_output_shape():
    lam = make_lam_patched()
    z, _, idx = lam(history(), single_frame())
    assert z.shape == (BATCH, LATENT_DIM)
    assert idx.shape == (BATCH,)


def test_lam_patch_sequence_future():
    lam = make_lam_patched(horizon=HORIZON)
    z, _, _ = lam(history(), frame_sequence(HORIZON))
    assert z.shape == (BATCH, LATENT_DIM)


def test_dynamics_patch_single_frame():
    # output must be character-level regardless of patch size
    dyn = make_dynamics_patched()
    logits = dyn(history(), action())
    assert logits.shape == (BATCH, S, VOCAB)


def test_dynamics_patch_sequence_teacher():
    dyn = make_dynamics_patched(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON, teacher_frames=frame_sequence(HORIZON))
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


def test_dynamics_patch_sequence_autoregressive():
    dyn = make_dynamics_patched(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


def test_dynamics_patch_backward():
    dyn = make_dynamics_patched()
    logits = dyn(history(), action())
    target = torch.randint(0, VOCAB, (BATCH, S))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()
