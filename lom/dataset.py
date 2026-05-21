"""Dataset loading for LOM pre-training.

Three datasets are supported:

  nld-aa    — NLD-AA (Autoascend AI ttyrec), via NLE SQLite DB
  nld-nao   — NLD-NAO (NetHack.alt.org ttyrec), via NLE SQLite DB
  nao-top10 — NAO Top-10 processed .npz tensors from DeepMind

Primary references:
  https://github.com/NetHack-LE/nle
  https://github.com/google-deepmind/nao_top10
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

_OBS_KEY    = "tty_chars"   # (T, H, W) uint8 — ASCII char codes
_ACTION_KEY = "keypresses"  # (T,) int64 — NLE action index
_SCREEN_H   = 24
_SCREEN_W   = 80


# --------------------------------------------------------------------------- #
# --- NLE DB helpers --------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _ensure_nle_db(
    data_dir: str,
    db_path: str,
    unzipped_subdir: str,
    dataset_name: str,
    use_altorg: bool,
) -> None:
    """Build an NLE SQLite DB from local unzipped ttyrec files if it is missing."""
    if os.path.exists(db_path):
        return

    unzipped = os.path.join(data_dir, unzipped_subdir)
    if not os.path.isdir(unzipped):
        raise RuntimeError(
            f"Database not found at {db_path} and unzipped data not found at {unzipped}.\n"
            f"Download the dataset first:\n"
            f"  python -m scripts.download_datasets {dataset_name} --output_dir {data_dir}"
        )

    log.info("%s.db not found — building from %s ...", dataset_name, unzipped)
    try:
        import nle.dataset as nld
        import nle.dataset.db as nld_db
    except ImportError:
        raise ImportError(
            "NLE is not installed.\n"
            "  pip install git+https://github.com/NetHack-LE/nle.git@main"
        )

    nld_db.create(filename=db_path)
    if use_altorg:
        nld.add_altorg_directory(unzipped, dataset_name, filename=db_path)
    else:
        nld.add_nledata_directory(unzipped, dataset_name, filename=db_path)
    log.info("Database built at %s", db_path)


# --------------------------------------------------------------------------- #
# --- Sequence-level loaders ------------------------------------------------ #
# --------------------------------------------------------------------------- #

def _load_from_nle_db(
    db_path: str,
    dataset_name: str,
    top_n: Optional[int],
    include_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load observation (and optionally action) sequences from an NLE ttyrec DB.

    Returns:
        obs_seqs:    list of (T, H*W) uint8 arrays
        action_seqs: list of (T,) int64 arrays (empty if include_actions=False)
    """
    try:
        from nle.dataset import dataset as nle_dataset
    except ImportError:
        raise ImportError(
            "NLE is not installed.\n"
            "  pip install git+https://github.com/NetHack-LE/nle.git@main"
        )

    keys = [_OBS_KEY, _ACTION_KEY] if include_actions else [_OBS_KEY]
    ds = nle_dataset.TtyrecDataset(
        dataset_name=dataset_name,
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


def _load_from_nao_top10_dir(
    directory: str,
    top_n: Optional[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load sessions from the DeepMind NAO-TOP10 .npz dataset.

    Expects `directory` to contain `[username]/[session_id].npz` files (as
    extracted from nao_top10.tar).  Each .npz has `tty_chars (T, 24, 80)`.

    Returns:
        obs_seqs:    list of (T, H*W) uint8 arrays
        action_seqs: empty list (no action data in this dataset)
    """
    npz_files = sorted(glob.glob(os.path.join(directory, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found under {directory}.\n"
            "Download the dataset first:\n"
            "  python -m scripts.download_datasets nao-top10 --output_dir <nle_data_dir>"
        )
    if top_n is not None:
        npz_files = npz_files[:top_n]

    obs_seqs: List[np.ndarray] = []
    for f in npz_files:
        data = np.load(f)
        chars = data["tty_chars"]                      # (T, 24, 80)
        obs_seqs.append(chars.reshape(len(chars), -1).astype(np.uint8))

    log.info("Loaded %d sessions from %s", len(obs_seqs), directory)
    return obs_seqs, []


def _load_from_numpy(
    directory: str,
    top_n: Optional[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Fallback: reads (T, H*W) uint8 .npy observation files.

    Looks for matching *_actions.npy files alongside each observation file.
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


# --------------------------------------------------------------------------- #
# --- Public dataset loaders ------------------------------------------------ #
# --------------------------------------------------------------------------- #

def load_nld_aa(
    nle_data_dir: str = "nle_data",
    fallback_numpy_dir: Optional[str] = None,
    max_games: Optional[int] = None,
    include_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the NLD-AA dataset (Autoascend AI gameplay, ~100 GB raw).

    Args:
        nle_data_dir:       directory containing nld-aa.db and nld-aa/ unzipped data
        fallback_numpy_dir: directory with pre-extracted .npy files (fallback)
        max_games:          maximum number of episodes to load
        include_actions:    also load action sequences
    """
    db = os.path.join(nle_data_dir, "nld-aa.db")
    try:
        _ensure_nle_db(nle_data_dir, db, "nld-aa", "nld-aa", use_altorg=False)
        return _load_from_nle_db(db, "nld-aa", top_n=max_games, include_actions=include_actions)
    except (RuntimeError, ImportError) as e:
        log.warning("NLE DB unavailable (%s); trying numpy fallback.", e)

    if fallback_numpy_dir and os.path.isdir(fallback_numpy_dir):
        return _load_from_numpy(fallback_numpy_dir, top_n=max_games)

    raise RuntimeError(
        "Could not load NLD-AA dataset.\n"
        f"  Tried NLE DB at: {db}\n"
        f"  Tried numpy dir: {fallback_numpy_dir}\n"
        "Run: python -m scripts.download_datasets nld-aa --output_dir <nle_data_dir>"
    )


def load_nld_nao(
    nle_data_dir: str = "nle_data",
    fallback_numpy_dir: Optional[str] = None,
    max_games: Optional[int] = None,
    include_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the NLD-NAO dataset (NetHack.alt.org gameplay, ~500 GB raw).

    Args:
        nle_data_dir:       directory containing nld-nao.db and nld-nao/ unzipped data
        fallback_numpy_dir: directory with pre-extracted .npy files (fallback)
        max_games:          maximum number of episodes to load
        include_actions:    also load action sequences
    """
    db = os.path.join(nle_data_dir, "nld-nao.db")
    try:
        _ensure_nle_db(nle_data_dir, db, "nld-nao", "nld-nao", use_altorg=True)
        return _load_from_nle_db(db, "nld-nao", top_n=max_games, include_actions=include_actions)
    except (RuntimeError, ImportError) as e:
        log.warning("NLE DB unavailable (%s); trying numpy fallback.", e)

    if fallback_numpy_dir and os.path.isdir(fallback_numpy_dir):
        return _load_from_numpy(fallback_numpy_dir, top_n=max_games)

    raise RuntimeError(
        "Could not load NLD-NAO dataset.\n"
        f"  Tried NLE DB at: {db}\n"
        f"  Tried numpy dir: {fallback_numpy_dir}\n"
        "Run: python -m scripts.download_datasets nld-nao --output_dir <nle_data_dir>"
    )


def load_nao_top10(
    nle_data_dir: str = "nle_data",
    fallback_numpy_dir: Optional[str] = None,
    max_games: Optional[int] = None,
    include_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the NAO Top-10 dataset (DeepMind processed .npz, ~12 GB).

    Data is at `nle_data_dir/nao-top10/` as extracted from nao_top10.tar.
    No NLE database is required; observations are read directly from .npz files.
    Action sequences are not available in this dataset (`include_actions` is ignored).

    Args:
        nle_data_dir:       directory containing the nao-top10/ subdirectory
        fallback_numpy_dir: directory with pre-extracted .npy files (fallback)
        max_games:          maximum number of sessions to load
        include_actions:    ignored (no actions in this dataset)
    """
    top10_dir = os.path.join(nle_data_dir, "nao-top10")
    try:
        return _load_from_nao_top10_dir(top10_dir, top_n=max_games)
    except FileNotFoundError as e:
        log.warning("NAO-TOP10 dir unavailable (%s); trying numpy fallback.", e)

    if fallback_numpy_dir and os.path.isdir(fallback_numpy_dir):
        return _load_from_numpy(fallback_numpy_dir, top_n=max_games)

    raise RuntimeError(
        f"Could not load NAO-TOP10 dataset.\n"
        f"  Tried .npz dir: {top10_dir}\n"
        f"  Tried numpy dir: {fallback_numpy_dir}\n"
        "Run: python -m scripts.download_datasets nao-top10 --output_dir <nle_data_dir>"
    )


# Backward-compat alias (nld-nao was previously called "full")
load_nao_full = load_nld_nao


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

    When horizon=1, next_frame and future_frame are the same frame.
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
        for i, seq in enumerate(sequences):
            if seq.ndim == 2:
                seq = seq.reshape(-1, obs_h, obs_w)
            T = len(seq)
            if T < min_len + 1:
                continue
            internal_idx = len(self._seqs)
            self._seqs.append(seq.astype(np.uint8))
            if action_sequences is not None:
                self._acts.append(action_sequences[i].astype(np.int64))
            for t in range(context_len - 1, T - horizon):
                self._index.append((internal_idx, t))

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

        history      = torch.tensor(seq[t - self.context_len + 1 : t + 1], dtype=torch.long)
        next_frame   = torch.tensor(seq[t + 1],             dtype=torch.long)
        future_frame = torch.tensor(seq[t + self.horizon],  dtype=torch.long)

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

        n_val   = max(1, int(len(dataset) * val_fraction))
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
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
