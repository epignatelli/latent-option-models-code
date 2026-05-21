import math

import pytest
import torch
import torch.nn as nn

from lom.config import LAMCfg, LOMCfg, LOMModelCfg, ModelCfg, DataCfg, EnvCfg, TrainCfg
from lom.training import LAMTrainer, LOMTrainer, get_lr, reconstruction_loss

from conftest import BATCH, CONTEXT, D_MODEL, HORIZON, LATENT_DIM, N_HEADS, N_LAYERS, OBS_H, OBS_W, VOCAB


# --------------------------------------------------------------------------- #
# --- get_lr ---------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

LR = 1e-3
WARMUP = 10
MAX_ITERS = 100
ETA_MIN = 1e-6


def test_get_lr_warmup_start():
    assert get_lr(0, LR, WARMUP, MAX_ITERS, ETA_MIN) == pytest.approx(0.0)


def test_get_lr_warmup_end():
    assert get_lr(WARMUP, LR, WARMUP, MAX_ITERS, ETA_MIN) == pytest.approx(LR)


def test_get_lr_cosine_end():
    assert get_lr(MAX_ITERS, LR, WARMUP, MAX_ITERS, ETA_MIN) == pytest.approx(ETA_MIN)


def test_get_lr_monotone_after_warmup():
    lrs = [get_lr(s, LR, WARMUP, MAX_ITERS, ETA_MIN) for s in range(WARMUP, MAX_ITERS + 1)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


# --------------------------------------------------------------------------- #
# --- reconstruction_loss --------------------------------------------------- #
# --------------------------------------------------------------------------- #

def test_reconstruction_loss_is_scalar():
    logits = torch.randn(BATCH, 16, VOCAB)
    target = torch.randint(0, VOCAB, (BATCH, 16))
    loss = reconstruction_loss(logits, target, VOCAB)
    assert loss.shape == ()


def test_reconstruction_loss_perfect():
    logits = torch.zeros(4, VOCAB)
    logits[:, 0] = 1e9
    target = torch.zeros(4, dtype=torch.long)
    loss = reconstruction_loss(logits, target, VOCAB)
    assert loss.item() < 1e-3


# --------------------------------------------------------------------------- #
# --- Trainer.step ---------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def make_env_cfg():
    return EnvCfg(obs_h=OBS_H, obs_w=OBS_W, vocab_size=VOCAB, n_actions=8)


def make_model_cfg():
    return ModelCfg(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                    max_context=CONTEXT, latent_dim=LATENT_DIM, num_options=16)


def make_lom_model_cfg():
    return LOMModelCfg(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       max_context=CONTEXT, latent_dim=LATENT_DIM, num_options=16)


def make_data_cfg():
    return DataCfg(context_len=CONTEXT, horizon=HORIZON)


def lam_trainer_with_models():
    cfg = LAMCfg(env=make_env_cfg(), model=make_model_cfg(), data=make_data_cfg())
    trainer = object.__new__(LAMTrainer)
    trainer.cfg = cfg
    trainer.models = trainer.build_models().eval()
    return trainer


def lom_trainer_with_models():
    cfg = LOMCfg(env=make_env_cfg(), model=make_lom_model_cfg(), data=make_data_cfg())
    trainer = object.__new__(LOMTrainer)
    trainer.cfg = cfg
    trainer.models = trainer.build_models().eval()
    return trainer


def lam_batch():
    history = torch.randint(0, VOCAB, (BATCH, CONTEXT, OBS_H, OBS_W))
    next_frame = torch.randint(0, VOCAB, (BATCH, OBS_H, OBS_W))
    return [history, next_frame]


def lom_batch():
    history = torch.randint(0, VOCAB, (BATCH, CONTEXT, OBS_H, OBS_W))
    next_frame = torch.randint(0, VOCAB, (BATCH, OBS_H, OBS_W))
    future_frame = torch.randint(0, VOCAB, (BATCH, OBS_H, OBS_W))
    sequence = torch.randint(0, VOCAB, (BATCH, HORIZON, OBS_H, OBS_W))
    return [history, next_frame, future_frame, sequence]


def test_lam_step_keys():
    trainer = lam_trainer_with_models()
    out = trainer.step(lam_batch())
    assert {"recon", "vq_loss", "entropy", "total_loss"} == set(out.keys())


def test_lam_step_backward():
    trainer = lam_trainer_with_models()
    out = trainer.step(lam_batch())
    out["total_loss"].backward()


def test_lom_step_keys():
    trainer = lom_trainer_with_models()
    out = trainer.step(lom_batch())
    assert {"lam_recon", "lom_recon", "vq_loss_option", "vq_loss_action",
            "entropy_option", "entropy_action", "total_loss"} == set(out.keys())


def test_lom_step_backward():
    trainer = lom_trainer_with_models()
    out = trainer.step(lom_batch())
    out["total_loss"].backward()
