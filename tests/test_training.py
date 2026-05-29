import math

import pytest
import torch
import torch.nn as nn

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention backward requires CUDA"
)

from lom.config import LAMCfg, LOMCfg, LOMModelCfg, ModelCfg, DataCfg, EnvCfg, TrainCfg
from lom.training import LAMTrainer, LOMTrainer, get_lr, reconstruction_loss

from conftest import BATCH, CONTEXT, D_MODEL, HORIZON, LATENT_DIM, N_HEADS, N_LAYERS, OBS_H, OBS_W, S, VOCAB


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


def test_reconstruction_loss_sequence():
    logits = torch.randn(BATCH, HORIZON, S, VOCAB)
    target = torch.randint(0, VOCAB, (BATCH, HORIZON, S))
    loss = reconstruction_loss(logits, target, VOCAB)
    assert loss.shape == ()


# --------------------------------------------------------------------------- #
# --- Trainer.step ---------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _screen(*leading):
    chars  = torch.randint(0, 256, (*leading, OBS_H, OBS_W, 1), dtype=torch.uint8)
    colors = torch.randint(0, 32,  (*leading, OBS_H, OBS_W, 1), dtype=torch.uint8)
    return torch.cat([chars, colors], dim=-1)


def make_env_cfg():
    return EnvCfg(obs_h=OBS_H, obs_w=OBS_W, vocab_size=VOCAB, n_actions=8)


def make_model_cfg():
    return ModelCfg(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                    context_length=CONTEXT, latent_dim=LATENT_DIM, num_options=16)


def make_lom_model_cfg():
    return LOMModelCfg(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       context_length=CONTEXT, latent_dim=LATENT_DIM, num_options=16)


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
    return [_screen(BATCH, CONTEXT), _screen(BATCH)]


def lom_batch():
    return [_screen(BATCH, CONTEXT), _screen(BATCH), _screen(BATCH), _screen(BATCH, HORIZON)]


@torch.no_grad()
def test_lam_step_keys():
    trainer = lam_trainer_with_models()
    out = trainer.step(lam_batch())
    assert {"recon", "vq_loss", "commit_loss", "entropy", "total_loss"} == set(out.keys())


@needs_cuda
def test_lam_step_backward():
    trainer = lam_trainer_with_models()
    trainer.models = trainer.models.cuda()
    out = trainer.step([t.cuda() for t in lam_batch()])
    out["total_loss"].backward()


@torch.no_grad()
def test_lom_step_keys():
    trainer = lom_trainer_with_models()
    out = trainer.step(lom_batch())
    assert {"lam_recon", "lom_recon", "vq_loss_option", "vq_loss_action",
            "commit_loss_option", "commit_loss_action",
            "entropy_option", "entropy_action", "total_loss"} == set(out.keys())


@needs_cuda
def test_lom_step_backward():
    trainer = lom_trainer_with_models()
    trainer.models = trainer.models.cuda()
    out = trainer.step([t.cuda() for t in lom_batch()])
    out["total_loss"].backward()


# --------------------------------------------------------------------------- #
# --- Trainer checkpoint ---------------------------------------------------- #
# --------------------------------------------------------------------------- #


def make_trainer_for_checkpoint(tmp_path):
    cfg = LAMCfg(env=make_env_cfg(), model=make_model_cfg(), data=make_data_cfg())
    trainer = object.__new__(LAMTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.models = trainer.build_models()
    trainer.optimizer = torch.optim.Adam(trainer.models.parameters())
    trainer.ckpt_path = str(tmp_path / "lam_pretrain.pt")
    trainer.wandb_run = None
    return trainer


def test_save_restore_checkpoint(tmp_path):
    trainer = make_trainer_for_checkpoint(tmp_path)
    trainer.save_checkpoint(step=42)
    assert (tmp_path / "lam_pretrain.pt").exists()

    for p in trainer.models.parameters():
        p.data.zero_()

    step = trainer.restore_checkpoint()
    assert step == 42
    assert any(p.data.abs().sum() > 0 for p in trainer.models.parameters())


def test_restore_checkpoint_missing_returns_zero(tmp_path):
    trainer = make_trainer_for_checkpoint(tmp_path)
    step = trainer.restore_checkpoint()
    assert step == 0
