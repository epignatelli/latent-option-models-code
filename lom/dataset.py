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
    """In-memory pool of loaded game arrays, refreshed by a background thread.

    Each buffer slot holds one game (T, H, W, 2) uint8 array. A background
    thread replaces refresh_fraction of slots every refresh_every seconds,
    loading one random game per sampled player file. State is replaced
    atomically so sample() never acquires a lock.

    Because each load call only keeps a single game's frames (not the full
    player file), large players with many games are handled without holding
    all of their data simultaneously in RAM.

    Args:
        paths:             (N,) object array of .npz file paths
        lengths:           (N,) int32 array of total frame counts per player
        buffer_size:       number of game slots to keep in memory
        context_len:       frames of history per sample
        horizon:           look-ahead frames per sample
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
        refresh_fraction: float = 0.1,
        refresh_every: float = 60.0,
        seed: int = 0,
    ) -> None:
        self._paths = paths
        self._ctx = context_len
        self._horizon = horizon

        valid = np.maximum(lengths.astype(np.float64) - (context_len + horizon - 1), 0.0)
        total = valid.sum()
        self._player_weights = valid / total if total > 0 else np.ones(len(paths)) / len(paths)

        n_slots = min(buffer_size, len(paths))
        self._n_refresh = min(max(1, int(n_slots * refresh_fraction)), n_slots)
        self._refresh_every = refresh_every

        rng = np.random.default_rng(seed)
        self._refresh_rng = np.random.default_rng(seed + 1)

        player_idxs = rng.choice(len(paths), size=n_slots, replace=True, p=self._player_weights)
        log.info("Loading initial buffer of %d game slots ...", n_slots)
        games = []
        log_every = max(1, n_slots // 10)
        for i, pi in enumerate(player_idxs):
            g = self._load_game(int(pi), rng)
            if g is not None:
                games.append(g)
            if (i + 1) % log_every == 0 or (i + 1) == n_slots:
                log.info("  buffer %d/%d slots loaded (%d games ready)",
                         i + 1, n_slots, len(games))
        self._state: tuple = (games, self._make_weights(games))
        log.info("Buffer ready (%d games).", len(games))

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def _load_game(self, player_idx: int, rng: np.random.Generator):
        """Load one random game from a player file. Returns (T, H, W, 2) uint8 or None.

        Loads chars and colors sequentially (with explicit del between) to keep
        peak RSS to ~one full player array at a time rather than two.
        """
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
                n_games = len(offsets) - 1
                game_lens = (offsets[1:] - offsets[:-1]).astype(np.float64)
                valid_lens = np.maximum(game_lens - (self._ctx + self._horizon - 1), 0.0)
                total_w = valid_lens.sum()
                if total_w <= 0:
                    return None
                gi = int(rng.choice(n_games, p=valid_lens / total_w))
                a, b = int(offsets[gi]), int(offsets[gi + 1])
                chars = chars_full[a:b].copy()
                del chars_full
                if "tty_colors" in f:
                    colors_raw = f["tty_colors"]
                    colors = np.clip(colors_raw[a:b], 0, COLOR_VOCAB - 1).astype(np.uint8)
                    del colors_raw
                else:
                    colors = np.zeros_like(chars)
        except Exception as exc:
            log.warning("Failed to load player %s: %s", path, exc)
            return None
        return np.stack([chars, colors], axis=-1)

    def _make_weights(self, games: list) -> np.ndarray:
        valid = np.maximum(
            np.array([len(g) for g in games], dtype=np.float64) - (self._ctx + self._horizon - 1),
            0.0,
        )
        s = valid.sum()
        return valid / s if s > 0 else np.ones(len(games)) / len(games)

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self._refresh_every):
            games = list(self._state[0])
            n = len(games)
            if n == 0:
                continue
            player_idxs = self._refresh_rng.choice(
                len(self._paths), size=self._n_refresh, replace=True, p=self._player_weights
            )
            slots = self._refresh_rng.choice(n, size=self._n_refresh, replace=False)
            for slot, pi in zip(slots, player_idxs):
                new_game = self._load_game(int(pi), self._refresh_rng)
                if new_game is not None:
                    games[slot] = new_game
            self._state = (games, self._make_weights(games))

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
        obs_h: int = SCREEN_H,
        obs_w: int = SCREEN_W,
        return_sequence: bool = False,
    ) -> None:
        self.context_len = context_len
        self.horizon = horizon
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.return_sequence = return_sequence
        self._steps = steps_per_epoch

        self._buffer = GameBuffer(
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
        return self._steps

    def __getitem__(self, idx: int):
        game, t = self._buffer.sample(self._rng)

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
