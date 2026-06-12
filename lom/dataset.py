from __future__ import annotations

import logging
import os
import threading
from tqdm import tqdm
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokeniser import COLOR_VOCAB

log = logging.getLogger(__name__)

SCREEN_H = 24
SCREEN_W = 80


# --------------------------------------------------------------------------- #
# --- Buffer-based npz dataset (scalable, O(buffer_size) RAM) --------------- #
# --------------------------------------------------------------------------- #


class GameBuffer:
    """In-memory pool of games, refreshed by a background thread.

    Each buffer slot holds one game (one npz file). Games are loaded in
    parallel at init from a v2 per-game index (game_paths / game_lengths).

    buffer_size controls how many games to keep in RAM at once.
    With ~4.6 MB/game, buffer_size=1000 ≈ 4.6 GB.

    Args:
        paths:             (N,) object array of per-game .npz file paths
        lengths:           (N,) int32 array of frame counts per game
        buffer_size:       number of game slots to keep in memory
        context_len:       frames of history per sample
        horizon:           look-ahead frames per sample
        stride:            step between future frames
        refresh_fraction:  fraction of slots replaced per refresh cycle
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
        stride: int = 1,
        refresh_fraction: float = 0.1,
        refresh_every: float = 60.0,
        seed: int = 0,
    ) -> None:
        from multiprocessing.pool import ThreadPool

        self.paths = paths
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        self.min_len = context_len + horizon * stride

        valid = np.maximum(lengths.astype(np.float64) - self.min_len, 0.0)
        total = valid.sum()
        self.weights = valid / total if total > 0 else np.ones(len(paths)) / len(paths)

        n_slots = min(buffer_size, len(paths))
        self.n_refresh = max(1, int(n_slots * refresh_fraction))
        self.refresh_every = refresh_every
        self.refresh_rng = np.random.default_rng(seed + 1)

        rng = np.random.default_rng(seed)
        game_idxs = rng.choice(len(paths), size=n_slots, replace=n_slots > len(paths),
                               p=self.weights)

        log.info("Loading buffer: %d games in parallel ...", n_slots)
        n_workers = min(8, n_slots)
        slots = []
        with ThreadPool(n_workers) as pool:
            with tqdm(total=n_slots, desc="buffer", unit="game", position=0, leave=True) as bar:
                for game in pool.imap_unordered(self.load_game, game_idxs.tolist()):
                    if game is not None:
                        slots.append(game)
                    bar.update(1)
        self.slots: list = slots
        self.state: tuple = (slots, self.make_weights(slots))
        log.info("Buffer ready: %d games loaded.", len(slots))

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.refresh_loop, daemon=True)
        self.thread.start()

    def load_game(self, game_idx: int):
        """Load a single game npz. Returns (T, H, W, 2) uint8 array or None."""
        path = str(self.paths[game_idx])
        try:
            with np.load(path) as f:
                chars = f["tty_chars"].astype(np.uint8)
                colors = (
                    np.clip(f["tty_colors"], 0, COLOR_VOCAB - 1).astype(np.uint8)
                    if "tty_colors" in f
                    else np.zeros_like(chars)
                )
            if len(chars) < self.min_len:
                return None
            return np.stack([chars, colors], axis=-1)
        except Exception as exc:
            log.warning("Failed to load game %s: %s", path, exc)
            return None

    def make_weights(self, games: list) -> np.ndarray:
        valid = np.maximum(
            np.array([len(g) for g in games], dtype=np.float64) - self.min_len,
            0.0,
        )
        s = valid.sum()
        return valid / s if s > 0 else np.ones(len(games)) / len(games)

    def refresh_loop(self) -> None:
        while not self.stop_event.wait(self.refresh_every):
            slots = list(self.slots)
            if not slots:
                continue
            game_idxs = self.refresh_rng.choice(
                len(self.paths), size=self.n_refresh, replace=True, p=self.weights
            )
            slot_idxs = self.refresh_rng.choice(len(slots), size=self.n_refresh, replace=False)
            for slot_i, gi in zip(slot_idxs, game_idxs):
                new_game = self.load_game(int(gi))
                if new_game is not None:
                    slots[slot_i] = new_game
            self.slots = slots
            self.state = (slots, self.make_weights(slots))

    def sample(self, rng: np.random.Generator) -> tuple:
        games, weights = self.state
        game_idx = int(rng.choice(len(games), p=weights))
        game = games[game_idx]
        lo = self.context_len - 1
        hi = len(game) - self.horizon * self.stride - 1
        t = int(rng.integers(lo, max(lo, hi), endpoint=True))
        return game, t

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)


class NpzTrajectoryDataset(Dataset):
    """Random-sampling trajectory dataset backed by per-game .npz files.

    Maintains a hot buffer of buffer_size games in RAM; a background thread
    replaces refresh_fraction of the buffer every refresh_every seconds.
    __getitem__ samples a random (game, timestep) pair regardless of idx.

    Requires num_workers=0 in DataLoader — IO is handled by the buffer thread.

    Each item:
        history:      (context_len, H, W) long  — frames [t-c+1 … t]
        next_frame:   (H, W) long               — frame t+1
        future_frame: (H, W) long               — frame t+horizon*stride
        sequence:     (horizon, H, W) long      — frames [t+stride … t+horizon*stride] (if return_sequence)
    """

    def __init__(
        self,
        paths: np.ndarray,
        lengths: np.ndarray,
        context_len: int = 4,
        horizon: int = 8,
        stride: int = 1,
        buffer_size: int = 1_000,
        refresh_fraction: float = 0.1,
        refresh_every: float = 60.0,
        seed: int = 0,
        obs_h: int = SCREEN_H,
        obs_w: int = SCREEN_W,
        return_sequence: bool = False,
    ) -> None:
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.return_sequence = return_sequence

        self.buffer = GameBuffer(
            paths, lengths, buffer_size, context_len, horizon, stride,
            refresh_fraction=refresh_fraction, refresh_every=refresh_every, seed=seed,
        )
        self.rng = np.random.default_rng(seed + 2)

        log.info(
            "NpzTrajectoryDataset: %d games in pool, buffer=%d games, refresh_every=%.0fs",
            len(paths), buffer_size, refresh_every,
        )

    @classmethod
    def from_index(cls, index_path: str, data_root: str = "", **kwargs) -> "NpzTrajectoryDataset":
        """Construct from a v2 index.npz produced by scripts/prepare_data.py."""
        idx = np.load(index_path)
        paths   = idx["game_paths"].astype(str)
        lengths = idx["game_lengths"].astype(np.int32)
        if data_root:
            paths = np.array([
                os.path.join(data_root, os.path.basename(os.path.dirname(p)), os.path.basename(p))
                for p in paths
            ])
        return cls(paths, lengths, **kwargs)

    @classmethod
    def split(
        cls,
        index_path: str,
        data_root: str = "",
        val_fraction: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> Tuple["NpzTrajectoryDataset", "NpzTrajectoryDataset"]:
        """Split index into train / val datasets by game."""
        idx = np.load(index_path)
        paths   = idx["game_paths"].astype(str)
        lengths = idx["game_lengths"].astype(np.int32)
        if data_root:
            paths = np.array([
                os.path.join(data_root, os.path.basename(os.path.dirname(p)), os.path.basename(p))
                for p in paths
            ])

        rng = np.random.default_rng(seed)
        n_val = max(1, int(len(paths) * val_fraction))
        perm = rng.permutation(len(paths))

        train_ds = cls(paths[perm[n_val:]], lengths[perm[n_val:]], seed=seed,     **kwargs)
        val_ds   = cls(paths[perm[:n_val]], lengths[perm[:n_val]], seed=seed + 1, **kwargs)
        return train_ds, val_ds

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, idx: int):
        game, t = self.buffer.sample(self.rng)

        history      = torch.from_numpy(game[t - self.context_len + 1 : t + 1].copy())
        next_frame   = torch.from_numpy(game[t + 1].copy())
        future_frame = torch.from_numpy(game[t + self.horizon * self.stride].copy())

        out = (history, next_frame, future_frame)
        if self.return_sequence:
            out = out + (torch.from_numpy(
                game[t + self.stride : t + self.horizon * self.stride + 1 : self.stride].copy()
            ),)
        return out

    def close(self) -> None:
        self.buffer.stop()

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
    stride: int = 1,
    buffer_size: int = 1_000,
    val_fraction: float = 0.05,
    refresh_fraction: float = 0.1,
    refresh_every: float = 60.0,
    seed: int = 42,
    return_sequence: bool = False,
    data_root: str = "",
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders from a prepare_data v2 index file.

    num_workers must be 0: IO is handled by each dataset's background thread.
    """
    train_ds, val_ds = NpzTrajectoryDataset.split(
        index_path,
        data_root=data_root,
        val_fraction=val_fraction,
        seed=seed,
        context_len=context_len,
        horizon=horizon,
        stride=stride,
        buffer_size=buffer_size,
        refresh_fraction=refresh_fraction,
        refresh_every=refresh_every,
        return_sequence=return_sequence,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
