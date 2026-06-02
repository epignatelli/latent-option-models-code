from __future__ import annotations

import logging
import os
import threading
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
    """In-memory pool of player games, refreshed by a background thread.

    Each buffer slot holds one player's worth of games. All games from a
    player are loaded in one shot, amortising the decompression cost across
    every game in that player file. Players are loaded in parallel at init.

    buffer_size controls how many players to keep in RAM at once.
    With ~20 games/player and ~4.6 MB/game, buffer_size=50 ≈ 4.6 GB.

    Args:
        paths:             (N,) object array of .npz file paths
        lengths:           (N,) int32 array of total frame counts per player
        buffer_size:       number of player slots to keep in memory
        context_len:       frames of history per sample
        horizon:           look-ahead frames per sample
        stride:            step between future frames
        refresh_fraction:  fraction of player slots replaced per refresh cycle
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

        self._paths = paths
        self._ctx = context_len
        self._horizon = horizon
        self._stride = stride
        self._min_len = context_len + horizon * stride

        valid = np.maximum(lengths.astype(np.float64) - self._min_len, 0.0)
        total = valid.sum()
        self._player_weights = valid / total if total > 0 else np.ones(len(paths)) / len(paths)

        n_slots = min(buffer_size, len(paths))
        self._n_refresh = max(1, int(n_slots * refresh_fraction))
        self._refresh_every = refresh_every
        self._refresh_rng = np.random.default_rng(seed + 1)

        rng = np.random.default_rng(seed)
        player_idxs = rng.choice(len(paths), size=n_slots, replace=n_slots > len(paths),
                                 p=self._player_weights)

        log.info("Loading buffer: %d players in parallel ...", n_slots)
        n_workers = min(8, n_slots)
        slots = []
        with ThreadPool(n_workers) as pool:
            for games in pool.imap_unordered(self._load_player, player_idxs.tolist()):
                if games:
                    slots.append(games)
        self._slots: list[list] = slots
        games = [g for s in slots for g in s]
        self._state: tuple = (games, self._make_weights(games))
        log.info("Buffer ready: %d players, %d games.", len(slots), len(games))

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def _load_player(self, player_idx: int) -> list:
        """Load all valid games from a player file. Returns list of (T, H, W, 2) uint8."""
        path = str(self._paths[player_idx])
        try:
            with np.load(path) as f:
                chars_full = f["tty_chars"].astype(np.uint8)
                n_frames = len(chars_full)
                offsets = (
                    f["offsets"][:]
                    if "offsets" in f
                    else np.array([0, n_frames], dtype=np.int64)
                )
                colors_full = (
                    np.clip(f["tty_colors"], 0, COLOR_VOCAB - 1).astype(np.uint8)
                    if "tty_colors" in f
                    else np.zeros_like(chars_full)
                )
            games = []
            for i in range(len(offsets) - 1):
                a, b = int(offsets[i]), int(offsets[i + 1])
                if b - a < self._min_len:
                    continue
                games.append(np.stack([chars_full[a:b].copy(), colors_full[a:b].copy()], axis=-1))
            return games
        except Exception as exc:
            log.warning("Failed to load player %s: %s", path, exc)
            return []

    def _make_weights(self, games: list) -> np.ndarray:
        valid = np.maximum(
            np.array([len(g) for g in games], dtype=np.float64) - self._min_len,
            0.0,
        )
        s = valid.sum()
        return valid / s if s > 0 else np.ones(len(games)) / len(games)

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self._refresh_every):
            slots = list(self._slots)
            if not slots:
                continue
            player_idxs = self._refresh_rng.choice(
                len(self._paths), size=self._n_refresh, replace=True, p=self._player_weights
            )
            slot_idxs = self._refresh_rng.choice(len(slots), size=self._n_refresh, replace=False)
            for slot_i, pi in zip(slot_idxs, player_idxs):
                new_games = self._load_player(int(pi))
                if new_games:
                    slots[slot_i] = new_games
            self._slots = slots
            games = [g for s in slots for g in s]
            self._state = (games, self._make_weights(games))

    def sample(self, rng: np.random.Generator) -> tuple:
        games, weights = self._state
        game_idx = int(rng.choice(len(games), p=weights))
        game = games[game_idx]
        lo = self._ctx - 1
        hi = len(game) - self._horizon * self._stride - 1
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
        future_frame: (H, W) long               — frame t+horizon*stride
        sequence:     (horizon, H, W) long      — frames [t+stride, t+2*stride … t+horizon*stride] (if return_sequence)
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

        self._buffer = GameBuffer(
            paths, lengths, buffer_size, context_len, horizon, stride,
            refresh_fraction=refresh_fraction, refresh_every=refresh_every, seed=seed,
        )
        self._rng = np.random.default_rng(seed + 2)

        log.info(
            "NpzTrajectoryDataset: %d players in pool, buffer=%d players, refresh_every=%.0fs",
            len(paths), buffer_size, refresh_every,
        )

    @classmethod
    def from_index(cls, index_path: str, **kwargs) -> "NpzTrajectoryDataset":
        """Construct from an index.npz file produced by scripts/prepare_data.py."""
        idx = np.load(index_path)
        if "player_paths" in idx:
            paths   = idx["player_paths"].astype(str)
            lengths = idx["player_lengths"].astype(np.int32)
        else:
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
        return 10_000

    def __getitem__(self, idx: int):
        game, t = self._buffer.sample(self._rng)

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
    stride: int = 1,
    buffer_size: int = 1_000,
    val_fraction: float = 0.05,
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
        stride=stride,
        buffer_size=buffer_size,
        refresh_fraction=refresh_fraction,
        refresh_every=refresh_every,
        return_sequence=return_sequence,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
