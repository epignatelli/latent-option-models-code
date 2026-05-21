import os

import numpy as np
import pytest
import torch

from lom.dataset import (
    TrajectoryDataset,
    build_dataloaders,
    _load_from_nao_top10_dir,
    _load_from_numpy,
    load_nao_top10,
    load_nld_aa,
    load_nld_nao,
)

from conftest import CONTEXT, HORIZON, OBS_H, OBS_W, VOCAB


def make_seq(T):
    return np.random.randint(0, VOCAB, (T, OBS_H, OBS_W), dtype=np.uint8)


def test_dataset_basic_shapes():
    ds = TrajectoryDataset([make_seq(20)], context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W)
    history, next_frame, future_frame = ds[0]
    assert history.shape == (CONTEXT, OBS_H, OBS_W)
    assert next_frame.shape == (OBS_H, OBS_W)
    assert future_frame.shape == (OBS_H, OBS_W)
    assert history.dtype == torch.long
    assert next_frame.dtype == torch.long


def test_dataset_return_sequence():
    ds = TrajectoryDataset([make_seq(20)], context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W, return_sequence=True)
    history, next_frame, future_frame, sequence = ds[0]
    assert sequence.shape == (HORIZON, OBS_H, OBS_W)


def test_dataset_with_actions():
    seq = make_seq(20)
    acts = np.random.randint(0, 98, (20,), dtype=np.int64)
    ds = TrajectoryDataset([seq], context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W, action_sequences=[acts])
    *_, action = ds[0]
    assert action.shape == ()
    assert action.dtype == torch.long


def test_dataset_filters_short():
    min_len = CONTEXT + HORIZON + 1
    short = make_seq(min_len - 1)
    long = make_seq(min_len + 5)
    ds = TrajectoryDataset([short, long], context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W)
    assert len(ds._seqs) == 1   # short trajectory dropped
    assert len(ds) > 0


def test_dataset_len():
    T = 20
    ds = TrajectoryDataset([make_seq(T)], context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W)
    valid = T - CONTEXT - HORIZON + 1
    expected = valid // (CONTEXT + HORIZON)
    assert len(ds) == expected


def test_dataset_split():
    seqs = [make_seq(40) for _ in range(20)]
    ds = TrajectoryDataset(seqs, context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W)
    train_ds, val_ds = TrajectoryDataset.split(ds, val_fraction=0.2, seed=0)
    assert len(train_ds._seqs) + len(val_ds._seqs) == len(ds._seqs)
    assert len(val_ds._seqs) >= 1


def test_build_dataloaders():
    seqs = [make_seq(30), make_seq(30)]
    train_loader, val_loader = build_dataloaders(
        seqs, context_len=CONTEXT, horizon=HORIZON,
        batch_size=4, val_fraction=0.2, num_workers=0,
    )
    batch = next(iter(train_loader))
    history, next_frame, future_frame = batch
    assert history.shape[1:] == (CONTEXT, OBS_H, OBS_W)
    assert next_frame.shape[1:] == (OBS_H, OBS_W)
    assert future_frame.shape[1:] == (OBS_H, OBS_W)


# --------------------------------------------------------------------------- #
# --- Loader helpers --------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _write_npz_sessions(root, n_sessions=3, T=20):
    """Write fake NAO-TOP10 .npz files under root/username/session.npz."""
    for i in range(n_sessions):
        user_dir = os.path.join(root, f"player{i}")
        os.makedirs(user_dir, exist_ok=True)
        chars = np.random.randint(0, VOCAB, (T, OBS_H, OBS_W), dtype=np.uint8)
        np.savez(os.path.join(user_dir, f"sess{i}.npz"), tty_chars=chars)


def _write_npy_sequences(root, n=3, T=20):
    """Write fake flat .npy observation files under root."""
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        arr = np.random.randint(0, VOCAB, (T, OBS_H * OBS_W), dtype=np.uint8)
        np.save(os.path.join(root, f"ep{i:03d}.npy"), arr)


# --------------------------------------------------------------------------- #
# --- _load_from_nao_top10_dir ----------------------------------------------- #
# --------------------------------------------------------------------------- #

def test_load_nao_top10_dir_shapes(tmp_path):
    _write_npz_sessions(str(tmp_path), n_sessions=3, T=20)
    obs, acts = _load_from_nao_top10_dir(str(tmp_path), top_n=None)
    assert len(obs) == 3
    assert len(acts) == 0
    assert obs[0].shape == (20, OBS_H * OBS_W)
    assert obs[0].dtype == np.uint8


def test_load_nao_top10_dir_top_n(tmp_path):
    _write_npz_sessions(str(tmp_path), n_sessions=5, T=20)
    obs, _ = _load_from_nao_top10_dir(str(tmp_path), top_n=2)
    assert len(obs) == 2


def test_load_nao_top10_dir_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_from_nao_top10_dir(str(tmp_path), top_n=None)


# --------------------------------------------------------------------------- #
# --- load_nao_top10 --------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def test_load_nao_top10_reads_npz(tmp_path):
    top10_dir = tmp_path / "nao-top10"
    top10_dir.mkdir()
    _write_npz_sessions(str(top10_dir), n_sessions=2, T=20)
    obs, acts = load_nao_top10(nle_data_dir=str(tmp_path))
    assert len(obs) == 2
    assert len(acts) == 0


def test_load_nao_top10_fallback_numpy(tmp_path):
    numpy_dir = tmp_path / "numpy"
    _write_npy_sequences(str(numpy_dir), n=2, T=20)
    obs, _ = load_nao_top10(nle_data_dir=str(tmp_path), fallback_numpy_dir=str(numpy_dir))
    assert len(obs) == 2


def test_load_nao_top10_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        load_nao_top10(nle_data_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# --- load_nld_nao ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def test_load_nld_nao_fallback_numpy(tmp_path):
    numpy_dir = tmp_path / "numpy"
    _write_npy_sequences(str(numpy_dir), n=2, T=20)
    obs, _ = load_nld_nao(nle_data_dir=str(tmp_path), fallback_numpy_dir=str(numpy_dir))
    assert len(obs) == 2


def test_load_nld_nao_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        load_nld_nao(nle_data_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# --- load_nld_aa ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

def test_load_nld_aa_fallback_numpy(tmp_path):
    numpy_dir = tmp_path / "numpy"
    _write_npy_sequences(str(numpy_dir), n=2, T=20)
    obs, _ = load_nld_aa(nle_data_dir=str(tmp_path), fallback_numpy_dir=str(numpy_dir))
    assert len(obs) == 2


def test_load_nld_aa_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        load_nld_aa(nle_data_dir=str(tmp_path))
