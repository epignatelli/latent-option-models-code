"""NAO dataset loading for LOM pre-training.

Supports:
  - NLE's built-in NAO Top-10 dataset (github.com/NetHack-LE/nle)
  - Full NAO dataset (same API, different size filter)
  - Fallback: directory of pre-saved (T, H, W) uint8 numpy arrays

Primary reference: https://github.com/NetHack-LE/nle
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

# NLE observation / action keys
_OBS_KEY = "tty_chars"  # (T, H, W) uint8 — ASCII char codes
_ACTION_KEY = "keypresses"  # (T,) int64   — NLE action index at each step
_SCREEN_H = 24
_SCREEN_W = 80

# --------------------------------------------------------------------------- #
# --- Sequence-level loaders ------------------------------------------------ #
# --------------------------------------------------------------------------- #


def _load_from_nle_db(
    db_path: str,
    top_n: Optional[int],
    include_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load observation (and optionally action) sequences from a NLE DB.

    Returns:
        obs_seqs:    list of (T, H*W) uint8 arrays
        action_seqs: list of (T,) int64 arrays (empty list if include_actions=False)
    """
    try:
        from nle.dataset import dataset as nle_dataset  # nle ≥ 0.9
    except ImportError:
        raise ImportError(
            "NLE is not installed. Install it with: pip install nle\n"
            "See https://github.com/NetHack-LE/nle for system requirements."
        )

    keys = [_OBS_KEY, _ACTION_KEY] if include_actions else [_OBS_KEY]
    ds = nle_dataset.TtyrecDataset(
        dataset_name="nao",
        dbfilename=db_path,
        seq_length=None,
        observation_keys=tuple(keys),
    )

    obs_seqs: List[np.ndarray] = []
    act_seqs: List[np.ndarray] = []
    for i, episode in enumerate(ds):
        if top_n is not None and i >= top_n:
            break
        chars = episode[_OBS_KEY]
        obs_seqs.append(chars.reshape(len(chars), -1).astype(np.uint8))
        if include_actions:
            act_seqs.append(episode[_ACTION_KEY].astype(np.int64))
        log.debug("Loaded episode %d: T=%d", i, len(chars))

    log.info("Loaded %d episodes from %s", len(obs_seqs), db_path)
    return obs_seqs, act_seqs


def _load_from_numpy(
    directory: str,
    top_n: Optional[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Fallback loader: reads (T, H*W) uint8 .npy observation files.

    Looks for matching *_actions.npy files alongside each observation file.
    Returns (obs_seqs, action_seqs); action_seqs is empty if no action files found.
    """
    obs_files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    obs_files = [f for f in obs_files if "_actions" not in f]
    if not obs_files:
        raise FileNotFoundError(f"No .npy files found in {directory}")
    if top_n is not None:
        obs_files = obs_files[:top_n]

    obs_seqs, act_seqs = [], []
    for f in obs_files:
        obs_seqs.append(np.load(f))
        act_path = f.replace(".npy", "_actions.npy")
        if os.path.exists(act_path):
            act_seqs.append(np.load(act_path).astype(np.int64))

    log.info("Loaded %d sequences from %s", len(obs_seqs), directory)
    return obs_seqs, act_seqs


def load_nao_top10(
    nle_data_dir: str = "nle_data",
    fallback_numpy_dir: Optional[str] = None,
    include_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the NAO Top-10 dataset (10 highest-scoring games on nethack.alt.org).

    Args:
        nle_data_dir:      directory where NLE stores nao.db
        fallback_numpy_dir: directory with pre-extracted .npy observation files
        include_actions:   also load action sequences (needed for GAM training)
    Returns:
        (obs_seqs, action_seqs) — action_seqs is empty when include_actions=False
    """
    db = os.path.join(nle_data_dir, "nao.db")
    if os.path.exists(db):
        try:
            return _load_from_nle_db(db_path=db, top_n=10, include_actions=include_actions)
        except ImportError as e:
            log.warning("NLE import failed (%s); trying numpy fallback.", e)

    if fallback_numpy_dir and os.path.isdir(fallback_numpy_dir):
        return _load_from_numpy(fallback_numpy_dir, top_n=10)

    raise RuntimeError(
        "Could not load NAO Top-10 dataset.\n"
        f"  Tried NLE DB at: {db}\n"
        f"  Tried numpy dir: {fallback_numpy_dir}\n"
        "Install NLE (pip install nle) or provide pre-extracted numpy files."
    )


def load_nao_full(
    nle_data_dir: str = "nle_data",
    fallback_numpy_dir: Optional[str] = None,
    max_games: Optional[int] = None,
    include_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the full NAO dataset.

    Same API as load_nao_top10 but without the top-10 restriction.
    """
    db = os.path.join(nle_data_dir, "nao.db")
    if os.path.exists(db):
        try:
            return _load_from_nle_db(db_path=db, top_n=max_games, include_actions=include_actions)
        except ImportError as e:
            log.warning("NLE import failed (%s); trying numpy fallback.", e)

    if fallback_numpy_dir and os.path.isdir(fallback_numpy_dir):
        return _load_from_numpy(fallback_numpy_dir, top_n=max_games)

    raise RuntimeError("Could not load NAO full dataset. See load_nao_top10 for details.")


# --------------------------------------------------------------------------- #
# --- PyTorch Dataset ------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class TrajectoryDataset(Dataset):
    """(history, next_frame, future_frame[, sequence][, action]) dataset.

    Each item:
        history:      (context_len, H, W) long  — frames [t-c+1 … t]
        next_frame:   (H, W) long               — frame at t+1
        future_frame: (H, W) long               — frame at t+horizon
        sequence:     (horizon, H, W) long      — frames [t+1 … t+horizon] (only if return_sequence=True)
        action:       () long                   — action at step t (only if action_seqs given)

    When horizon=1 next_frame and future_frame are the same frame.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        context_len: int = 4,
        horizon: int = 8,
        obs_h: int = _SCREEN_H,
        obs_w: int = _SCREEN_W,
        action_sequences: Optional[List[np.ndarray]] = None,
        return_sequence: bool = False,
    ):
        self.context_len = context_len
        self.horizon = horizon
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.return_sequence = return_sequence

        self._seqs: List[np.ndarray] = []
        self._acts: Optional[List[np.ndarray]] = None if action_sequences is None else []
        self._index: List[Tuple[int, int]] = []

        min_len = context_len + horizon
        for traj_idx, seq in enumerate(sequences):
            if seq.ndim == 2:
                seq = seq.reshape(-1, obs_h, obs_w)
            T = len(seq)
            if T < min_len + 1:
                continue
            self._seqs.append(seq.astype(np.uint8))
            if action_sequences is not None:
                self._acts.append(action_sequences[traj_idx].astype(np.int64))
            for t in range(context_len - 1, T - horizon):
                self._index.append((traj_idx, t))

        log.info(
            "TrajectoryDataset: %d episodes → %d samples (c=%d, k=%d, seq=%s, actions=%s)",
            len(self._seqs),
            len(self._index),
            context_len,
            horizon,
            return_sequence,
            self._acts is not None,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        traj_idx, t = self._index[idx]
        seq = self._seqs[traj_idx]

        history = torch.tensor(seq[t - self.context_len + 1 : t + 1], dtype=torch.long)
        next_frame = torch.tensor(seq[t + 1], dtype=torch.long)
        future_frame = torch.tensor(seq[t + self.horizon], dtype=torch.long)

        out = (history, next_frame, future_frame)

        if self.return_sequence:
            sequence = torch.tensor(seq[t + 1 : t + self.horizon + 1], dtype=torch.long)
            out = out + (sequence,)

        if self._acts is not None:
            action = int(self._acts[traj_idx][t])
            out = out + (torch.tensor(action, dtype=torch.long),)

        return out

    @staticmethod
    def split(
        dataset: TrajectoryDataset,
        val_fraction: float = 0.05,
        seed: int = 42,
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset]:
        """Randomly split a TrajectoryDataset into train / val subsets."""
        import torch.utils.data as tud

        n_val = max(1, int(len(dataset) * val_fraction))
        n_train = len(dataset) - n_val
        gen = torch.Generator().manual_seed(seed)
        return tud.random_split(dataset, [n_train, n_val], generator=gen)


def build_dataloaders(
    sequences: List[np.ndarray],
    context_len: int,
    horizon: int,
    batch_size: int,
    val_fraction: float = 0.05,
    num_workers: int = 4,
    seed: int = 42,
    return_sequence: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience: build train + val DataLoaders from raw sequences."""
    ds = TrajectoryDataset(
        sequences, context_len=context_len, horizon=horizon, return_sequence=return_sequence
    )
    train_ds, val_ds = TrajectoryDataset.split(ds, val_fraction=val_fraction, seed=seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
