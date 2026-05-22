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
import threading
from typing import List, Optional, Tuple

# Prevent NLE from creating a stray nle_data/ in the current working directory.
os.environ.setdefault("NLE_DATA_PATH", os.path.abspath("nle_data"))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .modules import COLOR_VOCAB

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
            f"  python -m scripts.prepare_data {dataset_name} --output_dir {data_dir}"
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
        chars = episode[_OBS_KEY]                          # (T, H, W) uint8
        colors_raw = episode.get("tty_colors")
        if colors_raw is not None:
            colors = np.clip(colors_raw.astype(np.int16), 0, COLOR_VOCAB - 1).astype(np.uint8)
        else:
            colors = np.zeros_like(chars)
        stacked = np.stack([chars, colors], axis=-1)       # (T, H, W, 2) uint8
        obs_seqs.append(stacked.reshape(len(chars), -1))
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
            "  python -m scripts.prepare_data nao-top10 --output_dir <nle_data_dir>"
        )
    if top_n is not None:
        npz_files = npz_files[:top_n]

    obs_seqs: List[np.ndarray] = []
    for f in npz_files:
        data = np.load(f)
        chars = data["tty_chars"].astype(np.uint8)     # (T, 24, 80)
        if "tty_colors" in data:
            colors = np.clip(data["tty_colors"].astype(np.int16), 0, COLOR_VOCAB - 1).astype(np.uint8)
        else:
            colors = np.zeros_like(chars)
        stacked = np.stack([chars, colors], axis=-1)   # (T, 24, 80, 2)
        obs_seqs.append(stacked.reshape(len(chars), -1))

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
        "Run: python -m scripts.prepare_data nld-aa --output_dir <nle_data_dir>"
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
        "Run: python -m scripts.prepare_data nld-nao --output_dir <nle_data_dir>"
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
        "Run: python -m scripts.prepare_data nao-top10 --output_dir <nle_data_dir>"
    )


# --------------------------------------------------------------------------- #
# --- PyTorch Dataset ------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class TrajectoryDataset(Dataset):
    """GENIE-style random-sampling trajectory dataset.

    Each call to __getitem__ samples a random trajectory (weighted by length)
    and a random valid starting timestep within it — independent of the index
    argument.  This avoids the dense step-1 sliding window and its ~window_size×
    inflation of highly-correlated samples.

    __len__ is defined as total_valid_timesteps // (context_len + horizon), so
    one "epoch" corresponds to each timestep being seen approximately once on
    average.  The split() method partitions trajectories (not samples) into
    train / val subsets.

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

        min_len = context_len + horizon + 1
        for i, seq in enumerate(sequences):
            if seq.ndim == 2:
                # flat (T, H*W) — char-only legacy format; zero the color channel
                seq = seq.reshape(-1, obs_h, obs_w)
                seq = np.stack([seq, np.zeros_like(seq)], axis=-1)  # (T, H, W, 2)
            elif seq.ndim == 3:
                # (T, H, W) — char-only; zero the color channel
                seq = np.stack([seq, np.zeros_like(seq)], axis=-1)  # (T, H, W, 2)
            if len(seq) < min_len:
                continue
            self._seqs.append(seq.astype(np.uint8))
            if action_sequences is not None:
                self._acts.append(action_sequences[i].astype(np.int64))

        # valid starting positions per trajectory: t in [context_len-1, T-horizon-1]
        valid_starts = [len(s) - context_len - horizon + 1 for s in self._seqs]
        total_valid = sum(valid_starts)
        self._epoch_len = max(1, total_valid // (context_len + horizon))
        weights = np.array(valid_starts, dtype=np.float64)
        self._weights = weights / weights.sum() if weights.sum() > 0 else weights

        log.info(
            "TrajectoryDataset: %d episodes → epoch_len=%d (c=%d, k=%d, seq=%s, actions=%s)",
            len(self._seqs),
            self._epoch_len,
            context_len,
            horizon,
            return_sequence,
            self._acts is not None,
        )

    def __len__(self) -> int:
        return self._epoch_len

    def __getitem__(self, idx: int):
        # idx is unused — sample a random (trajectory, timestep) pair
        traj_idx = int(np.random.choice(len(self._seqs), p=self._weights))
        seq = self._seqs[traj_idx]
        t = int(np.random.randint(self.context_len - 1, len(seq) - self.horizon))

        history      = torch.from_numpy(seq[t - self.context_len + 1 : t + 1].copy())
        next_frame   = torch.from_numpy(seq[t + 1].copy())
        future_frame = torch.from_numpy(seq[t + self.horizon].copy())

        out = (history, next_frame, future_frame)

        if self.return_sequence:
            sequence = torch.from_numpy(seq[t + 1 : t + self.horizon + 1].copy())
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
        """Split trajectories into train / val subsets."""
        rng = np.random.default_rng(seed)
        n = len(dataset._seqs)
        n_val = max(1, int(n * val_fraction))
        perm = rng.permutation(n)
        val_idxs   = perm[:n_val].tolist()
        train_idxs = perm[n_val:].tolist()

        def _make(idxs: list) -> TrajectoryDataset:
            seqs = [dataset._seqs[i] for i in idxs]
            acts = None if dataset._acts is None else [dataset._acts[i] for i in idxs]
            return TrajectoryDataset(
                seqs,
                context_len=dataset.context_len,
                horizon=dataset.horizon,
                obs_h=dataset.obs_h,
                obs_w=dataset.obs_w,
                action_sequences=acts,
                return_sequence=dataset.return_sequence,
            )

        return _make(train_idxs), _make(val_idxs)


# --------------------------------------------------------------------------- #
# --- Buffer-based npz dataset (scalable, O(buffer_size) RAM) --------------- #
# --------------------------------------------------------------------------- #


class _GameBuffer:
    """In-memory pool of loaded game arrays, refreshed by a background thread.

    Uses atomic state replacement so sample() never acquires a lock.

    Args:
        paths:             (N,) object array of .npz file paths
        lengths:           (N,) int32 array of frame counts
        buffer_size:       number of games to keep in memory
        context_len:       frames of history per sample
        horizon:           look-ahead frames per sample
        refresh_fraction:  fraction of buffer replaced per refresh cycle
        refresh_every:     seconds between refresh cycles
        seed:              RNG seed (refresh thread uses seed+1)
    """

    def __init__(
        self,
        paths: np.ndarray,
        lengths: np.ndarray,
        buffer_size: int,
        context_len: int,
        horizon: int,
        refresh_fraction: float = 0.1,
        refresh_every: float = 60.0,
        seed: int = 0,
    ) -> None:
        self._paths = paths
        self._ctx = context_len
        self._horizon = horizon

        valid = np.maximum(lengths.astype(np.float64) - (context_len + horizon - 1), 0.0)
        total = valid.sum()
        self._pool_weights = valid / total if total > 0 else np.ones(len(paths)) / len(paths)

        n_init = min(buffer_size, len(paths))
        self._n_refresh = min(max(1, int(n_init * refresh_fraction)), n_init)
        self._refresh_every = refresh_every

        rng = np.random.default_rng(seed)
        self._refresh_rng = np.random.default_rng(seed + 1)

        init_idxs = rng.choice(len(paths), size=n_init, replace=False, p=self._pool_weights)
        log.info("Loading initial buffer of %d files ...", n_init)
        # Each _load call returns a list of games (one file may contain many games via offsets).
        # _players tracks the per-file grouping for refresh; _state exposes a flat view for sampling.
        self._players: list = [self._load(i) for i in init_idxs]
        flat = [g for pg in self._players for g in pg]
        self._state: tuple = (flat, self._make_weights(flat))
        log.info("Buffer ready (%d games from %d files).", len(flat), n_init)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def _load(self, idx: int) -> list:
        """Load one npz file and return a list of (T, H, W, 2) uint8 game arrays.

        Supports both single-game npz (no ``offsets`` key) and per-player npz
        produced by prepare_data.py (with ``offsets`` marking game boundaries).
        """
        with np.load(self._paths[idx]) as f:
            chars = f["tty_chars"].astype(np.uint8)
            if "tty_colors" in f:
                colors = np.clip(f["tty_colors"].astype(np.int16), 0, COLOR_VOCAB - 1).astype(np.uint8)
            else:
                colors = np.zeros_like(chars)
            offsets = f["offsets"] if "offsets" in f else np.array([0, len(chars)], dtype=np.int64)
        stacked = np.stack([chars, colors], axis=-1)  # (total_T, H, W, 2)
        return [stacked[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]

    def _make_weights(self, games: list) -> np.ndarray:
        valid = np.maximum(
            np.array([len(g) for g in games], dtype=np.float64) - (self._ctx + self._horizon - 1),
            0.0,
        )
        s = valid.sum()
        return valid / s if s > 0 else np.ones(len(games)) / len(games)

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self._refresh_every):
            players = list(self._players)

            new_idxs = self._refresh_rng.choice(
                len(self._paths), size=self._n_refresh, replace=False, p=self._pool_weights
            )
            slots = self._refresh_rng.choice(len(players), size=self._n_refresh, replace=False)

            for slot, pool_idx in zip(slots, new_idxs):
                players[slot] = self._load(pool_idx)

            flat = [g for pg in players for g in pg]
            self._players = players
            self._state = (flat, self._make_weights(flat))

    def sample(self, rng: np.random.Generator) -> tuple:
        games, weights = self._state
        game_idx = int(rng.choice(len(games), p=weights))
        game = games[game_idx]
        lo = self._ctx - 1
        hi = len(game) - self._horizon - 1
        t = int(rng.integers(lo, max(lo, hi), endpoint=True))
        return game, t

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


class NpzTrajectoryDataset(Dataset):
    """Random-sampling trajectory dataset backed by per-game .npz files.

    Maintains a hot buffer of buffer_size games in RAM; a background thread
    replaces refresh_fraction of the buffer every refresh_every seconds.
    __getitem__ samples a random (game, timestep) pair regardless of idx.

    Requires num_workers=0 in DataLoader — IO is handled by the buffer thread.

    Each item:
        history:      (context_len, H, W) long  — frames [t-c+1 … t]
        next_frame:   (H, W) long               — frame t+1
        future_frame: (H, W) long               — frame t+horizon
        sequence:     (horizon, H, W) long      — frames [t+1 … t+horizon] (if return_sequence)
    """

    def __init__(
        self,
        paths: np.ndarray,
        lengths: np.ndarray,
        context_len: int = 4,
        horizon: int = 8,
        buffer_size: int = 1_000,
        refresh_fraction: float = 0.1,
        refresh_every: float = 60.0,
        steps_per_epoch: int = 10_000,
        seed: int = 0,
        obs_h: int = _SCREEN_H,
        obs_w: int = _SCREEN_W,
        return_sequence: bool = False,
    ) -> None:
        self.context_len = context_len
        self.horizon = horizon
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.return_sequence = return_sequence
        self._steps = steps_per_epoch

        self._buffer = _GameBuffer(
            paths, lengths, buffer_size, context_len, horizon,
            refresh_fraction=refresh_fraction, refresh_every=refresh_every, seed=seed,
        )
        self._rng = np.random.default_rng(seed + 2)

        log.info(
            "NpzTrajectoryDataset: %d games in pool, buffer=%d, steps/epoch=%d",
            len(paths), buffer_size, steps_per_epoch,
        )

    @classmethod
    def from_index(cls, index_path: str, **kwargs) -> "NpzTrajectoryDataset":
        """Construct from an index.npz file produced by scripts/prepare_data.py."""
        idx = np.load(index_path)
        if "player_paths" in idx:
            # Rich nld-nao index: one entry per player file.
            paths   = idx["player_paths"].astype(str)
            lengths = idx["player_lengths"].astype(np.int32)
        else:
            # Simple nld-aa / nao-top10 index.
            paths   = idx["paths"].astype(str)
            lengths = idx["lengths"].astype(np.int32)
        return cls(paths, lengths, **kwargs)

    @classmethod
    def split(
        cls,
        index_path: str,
        val_fraction: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> Tuple["NpzTrajectoryDataset", "NpzTrajectoryDataset"]:
        """Split index into train / val datasets (by player for rich index)."""
        idx = np.load(index_path)
        if "player_paths" in idx:
            paths   = idx["player_paths"].astype(str)
            lengths = idx["player_lengths"].astype(np.int32)
        else:
            paths   = idx["paths"].astype(str)
            lengths = idx["lengths"].astype(np.int32)

        rng = np.random.default_rng(seed)
        n_val = max(1, int(len(paths) * val_fraction))
        perm = rng.permutation(len(paths))

        train_ds = cls(paths[perm[n_val:]], lengths[perm[n_val:]], seed=seed,     **kwargs)
        val_ds   = cls(paths[perm[:n_val]], lengths[perm[:n_val]], seed=seed + 1, **kwargs)
        return train_ds, val_ds

    def __len__(self) -> int:
        return self._steps

    def __getitem__(self, idx: int):
        game, t = self._buffer.sample(self._rng)
        # game: (T, H, W, 2) uint8 — (char, color) per cell

        history      = torch.from_numpy(game[t - self.context_len + 1 : t + 1].copy())
        next_frame   = torch.from_numpy(game[t + 1].copy())
        future_frame = torch.from_numpy(game[t + self.horizon].copy())

        out = (history, next_frame, future_frame)
        if self.return_sequence:
            out = out + (torch.from_numpy(game[t + 1 : t + self.horizon + 1].copy()),)
        return out

    def close(self) -> None:
        self._buffer.stop()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_npz_dataloaders(
    index_path: str,
    context_len: int,
    horizon: int,
    batch_size: int,
    buffer_size: int = 1_000,
    val_fraction: float = 0.05,
    steps_per_epoch: int = 10_000,
    refresh_fraction: float = 0.1,
    refresh_every: float = 60.0,
    seed: int = 42,
    return_sequence: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders from a prepare_data index file.

    num_workers must be 0: IO is handled by each dataset's background thread.
    """
    train_ds, val_ds = NpzTrajectoryDataset.split(
        index_path,
        val_fraction=val_fraction,
        seed=seed,
        context_len=context_len,
        horizon=horizon,
        buffer_size=buffer_size,
        refresh_fraction=refresh_fraction,
        refresh_every=refresh_every,
        steps_per_epoch=steps_per_epoch,
        return_sequence=return_sequence,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


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
