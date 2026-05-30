import pytest
from lom.config import (
    LOMCfg,
    ModelCfg,
    LOMModelCfg,
    TrainCfg,
)


def test_model_cfg_valid():
    cfg = ModelCfg(d_model=16, n_heads=2)
    assert cfg.d_model == 16
    assert cfg.n_heads == 2


def test_model_cfg_bad_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelCfg(d_model=6, n_heads=4)


def test_train_cfg_bad_warmup():
    with pytest.raises(ValueError, match="warmup_iters"):
        TrainCfg(warmup_iters=1000, max_iters=1000)


def test_train_cfg_bad_eta_min():
    with pytest.raises(ValueError, match="eta_min"):
        TrainCfg(eta_min=1.0, lr=1e-4)


def test_lom_cfg_default():
    cfg = LOMCfg()
    assert isinstance(cfg.model, LOMModelCfg)
