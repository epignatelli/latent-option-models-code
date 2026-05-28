"""Profile buffer_size tradeoffs: startup time, RAM, and sample reuse ratio.

Usage:
    python -m scripts.profile_buffer                          # default sizes
    python -m scripts.profile_buffer --sizes 50 200 500       # custom sweep
    python -m scripts.profile_buffer --samp-per-sec 524       # measured GPU speed

The script loads each buffer size, measures startup time, RAM, and unique start
positions, then reports the reuse ratio given a GPU throughput figure.

A reuse ratio close to 1 means the model rarely sees the same (game, t) pair
twice before that game slot is replaced. Higher ratios mean more repetition.
Run scripts/profile_memory.py first to get the samp/s figure.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from lom.dataset import GameBuffer

LOG_FILE = "/scratch/uceeepi/lom/profile_buffer.log"

_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_sh)
_fh = logging.FileHandler(LOG_FILE, mode="a")
_fh.setFormatter(_fmt)
_root.addHandler(_fh)
log = logging.getLogger(__name__)

DEFAULT_INDEX   = "/scratch/uceeepi/lom/datasets/nle/nao/index.npz"
DEFAULT_SIZES   = [50, 100, 200, 500]
DEFAULT_SPS     = 20_000.0
CTX             = 4
HORIZON         = 128
REFRESH_FRAC    = 0.1
REFRESH_EVERY_S = 60.0
SEED            = 42


def load_index(path: str):
    idx = np.load(path)
    if "player_paths" in idx:
        return idx["player_paths"].astype(str), idx["player_lengths"].astype(np.int32)
    return idx["paths"].astype(str), idx["lengths"].astype(np.int32)


def profile_one(paths, lengths, buffer_size: int, horizon: int) -> dict:
    t0 = time.perf_counter()
    buf = GameBuffer(
        paths, lengths,
        buffer_size=buffer_size,
        context_len=CTX,
        horizon=horizon,
        seed=SEED,
        refresh_fraction=REFRESH_FRAC,
        refresh_every=REFRESH_EVERY_S,
    )
    startup_s = time.perf_counter() - t0

    games, _ = buf._state
    game_lens   = np.array([len(g) for g in games], dtype=np.int64)
    ram_bytes   = sum(int(g.nbytes) for g in games)
    valid_lens  = np.maximum(game_lens - (CTX + horizon - 1), 0)
    total_valid = int(valid_lens.sum())
    buf.stop()

    n_refresh       = min(max(1, int(buffer_size * REFRESH_FRAC)), buffer_size)
    load_s_per_game = startup_s / buffer_size
    refresh_cycle_s = n_refresh * load_s_per_game + REFRESH_EVERY_S

    return dict(
        buffer_size     = buffer_size,
        n_games         = len(games),
        startup_s       = startup_s,
        load_s_per_game = load_s_per_game,
        ram_bytes       = ram_bytes,
        avg_game_frames = float(game_lens.mean()),
        total_valid     = total_valid,
        n_refresh       = n_refresh,
        refresh_cycle_s = refresh_cycle_s,
    )


def print_results(rows: list[dict], samp_per_sec: float, horizon: int) -> None:
    log.info("")
    log.info("═" * 80)
    log.info("  Buffer sweep  (horizon=%d  ctx=%d  GPU=%.0f samp/s):",
             horizon, CTX, samp_per_sec)
    log.info("  %9s  %10s  %8s  %12s  %10s",
             "buf_size", "startup(s)", "RAM(MB)", "uniq_starts", "reuse")
    log.info("  " + "-" * 58)
    for r in rows:
        reuse = samp_per_sec * r["refresh_cycle_s"] / max(1, r["total_valid"])
        flag  = "✓" if reuse < 5 else ("~" if reuse < 20 else "✗")
        log.info("  %9d  %10.0f  %8.0f  %12d  %8.1fx %s",
                 r["buffer_size"], r["startup_s"], r["ram_bytes"] / 1e6,
                 r["total_valid"], reuse, flag)

    log.info("")
    log.info("  Reuse ratio: times each unique start is seen per refresh cycle.")
    log.info("  < 5x = good.  5–20x = acceptable.  > 20x = increase buffer.")
    log.info("  Use scripts/profile_memory.py to measure samp/s for your model.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index",        default=DEFAULT_INDEX)
    parser.add_argument("--sizes",        type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--horizon",      type=int, default=HORIZON)
    parser.add_argument("--samp-per-sec", type=float, default=DEFAULT_SPS,
                        help="GPU training throughput in samples/s (from profile_memory.py)")
    args = parser.parse_args()

    log.info("Loading index from %s ...", args.index)
    paths, lengths = load_index(args.index)
    log.info("  %d players  total frames: %.1fM  median: %.0f",
             len(paths), lengths.sum() / 1e6, float(np.median(lengths)))

    if args.samp_per_sec == DEFAULT_SPS:
        log.info("Using default samp/s=%.0f (placeholder). "
                 "Run profile_memory.py --model lam first for an accurate figure.",
                 args.samp_per_sec)

    rows = []
    for bs in args.sizes:
        log.info("")
        log.info("Profiling buffer_size=%d ...", bs)
        r = profile_one(paths, lengths, bs, args.horizon)
        rows.append(r)
        log.info("  done: %.0fs startup  %.0fMB RAM  %d unique starts",
                 r["startup_s"], r["ram_bytes"] / 1e6, r["total_valid"])

    print_results(rows, args.samp_per_sec, args.horizon)


if __name__ == "__main__":
    main()
