import pytest
import torch
from lom.models import DynamicsModel, LatentActionModel

from conftest import BATCH, CONTEXT, D_MODEL, HORIZON, LATENT_DIM, N_HEADS, N_LAYERS, OBS_H, OBS_W, S, VOCAB


def make_lam(horizon=1, condition_dim=None):
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


def history():
    return torch.randint(0, VOCAB, (BATCH, CONTEXT, OBS_H, OBS_W))


def single_frame():
    return torch.randint(0, VOCAB, (BATCH, OBS_H, OBS_W))


def frame_sequence(k):
    return torch.randint(0, VOCAB, (BATCH, k, OBS_H, OBS_W))


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
