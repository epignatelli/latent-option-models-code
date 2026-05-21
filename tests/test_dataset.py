import numpy as np
import pytest
import torch
from lom.dataset import TrajectoryDataset, build_dataloaders

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
    assert len(ds) > 0
    # All items should come from the long sequence only
    for traj_idx, _ in ds._index:
        assert traj_idx == 0   # index 0 because short was dropped


def test_dataset_len():
    T = 20
    ds = TrajectoryDataset([make_seq(T)], context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W)
    expected = T - CONTEXT - HORIZON + 1   # valid t values: [CONTEXT-1, T-HORIZON)
    assert len(ds) == expected


def test_dataset_split():
    ds = TrajectoryDataset([make_seq(40)], context_len=CONTEXT, horizon=HORIZON,
                           obs_h=OBS_H, obs_w=OBS_W)
    train_ds, val_ds = TrajectoryDataset.split(ds, val_fraction=0.2, seed=0)
    assert len(train_ds) + len(val_ds) == len(ds)
    assert len(val_ds) >= 1


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
