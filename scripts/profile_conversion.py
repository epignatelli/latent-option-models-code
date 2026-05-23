"""Profile the per-stage timing of converting one nld-aa group.

Usage:
    python scripts/profile_conversion.py <game_dir> [--n-games N]

Example:
    python scripts/profile_conversion.py \
        /scratch/uceeepi/lom/datasets/nld-aa/nle_data/nle.2020-01-01 \
        --n-games 10

Stages timed per game:
  read    — open() + read() of the bz2 file from disk (pure I/O, no decode)
  decode  — nle_converter.Converter.load_ttyrec + .convert (C extension)
  cast    — .astype(uint8) / .clip() on the returned arrays
  total   — wall time for the whole game (read + decode + cast)

Summary at the end:
  per-game medians + totals, plus concat and savez timings on the full batch.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from statistics import median

import numpy as np
import tyro

ROWS, COLS = 24, 80
TTYREC_VERSION = 3


@dataclass
class Args:
    output_dir: str = "/scratch/uceeepi/lom/datasets"
    """Root output dir (same as --output-dir in prepare_data.py)."""
    n_games: int = 20
    """Number of games to profile per group (0 = all)."""
    group: str = ""
    """Specific group subdir name to profile; empty = pick the first one found."""


def _read_bytes(path: str) -> float:
    """Time reading the file into memory without decoding."""
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        f.read()
    return time.perf_counter() - t0


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def _bar(value: float, total: float, width: int = 30) -> str:
    filled = int(round(width * value / total)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _find_game_dir(output_dir: str, group: str) -> str:
    data_root = os.path.join(output_dir, "nld-aa", "nle_data")
    if not os.path.isdir(data_root):
        print(f"nld-aa data not found at {data_root}")
        sys.exit(1)
    if group:
        return os.path.join(data_root, group)
    for name in sorted(os.listdir(data_root)):
        candidate = os.path.join(data_root, name)
        if os.path.isdir(candidate) and any(f.endswith(".bz2") for f in os.listdir(candidate)):
            return candidate
    print(f"No group dirs with .bz2 files found under {data_root}")
    sys.exit(1)


def main() -> None:
    args = tyro.cli(Args)

    game_dir = _find_game_dir(args.output_dir, args.group)

    bz2_files = sorted(
        [os.path.join(game_dir, f) for f in os.listdir(game_dir) if f.endswith(".bz2")],
        key=lambda p: int(os.path.basename(p).split(".")[2]),
    )
    if not bz2_files:
        print(f"No .bz2 files found in {game_dir}")
        sys.exit(1)

    if args.n_games > 0:
        bz2_files = bz2_files[:args.n_games]

    print(f"Profiling {len(bz2_files)} games in {game_dir}\n")

    from nle import _pyconverter as nle_converter

    times_read   = []
    times_decode = []
    times_cast   = []
    frames_list  = []
    sizes_mb     = []

    chars_parts  = []
    colors_parts = []

    for i, path in enumerate(bz2_files):
        size_mb = os.path.getsize(path) / 1e6
        sizes_mb.append(size_mb)

        # --- read ---
        t_read = _read_bytes(path)

        # --- decode ---
        chunk = 200_000
        tmp_chars  = np.zeros((chunk, ROWS, COLS), dtype=np.uint8)
        tmp_colors = np.zeros((chunk, ROWS, COLS), dtype=np.int8)
        tmp_cursor = np.zeros((chunk, 2),          dtype=np.int16)
        tmp_ts     = np.zeros(chunk,               dtype=np.int64)
        tmp_kp     = np.zeros(chunk,               dtype=np.uint8)
        tmp_scores = np.zeros(chunk,               dtype=np.int32)

        conv = nle_converter.Converter(ROWS, COLS, TTYREC_VERSION)
        t0 = time.perf_counter()
        conv.load_ttyrec(path, gameid=1, part=0)
        remaining = conv.convert(tmp_chars, tmp_colors, tmp_cursor, tmp_ts, tmp_kp, tmp_scores)
        t_decode = time.perf_counter() - t0

        n = chunk - remaining
        if n == 0:
            continue

        # --- cast ---
        t0 = time.perf_counter()
        chars  = tmp_chars[:n].copy().astype(np.uint8)
        colors = tmp_colors[:n].astype(np.int16).clip(0, 31).astype(np.uint8)
        t_cast = time.perf_counter() - t0

        times_read.append(t_read)
        times_decode.append(t_decode)
        times_cast.append(t_cast)
        frames_list.append(n)
        chars_parts.append(chars)
        colors_parts.append(colors)

        total = t_read + t_decode + t_cast
        print(
            f"  game {i+1:3d}/{len(bz2_files)}  "
            f"{size_mb:5.2f} MB  {n:6,} frames  "
            f"read={_fmt(t_read)}  decode={_fmt(t_decode)}  cast={_fmt(t_cast)}  "
            f"total={_fmt(total)}"
        )

    if not times_read:
        print("No valid games decoded.")
        sys.exit(1)

    # --- concatenate ---
    t0 = time.perf_counter()
    all_chars  = np.concatenate(chars_parts)
    all_colors = np.concatenate(colors_parts)
    t_concat = time.perf_counter() - t0

    # --- savez_compressed ---
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    t0 = time.perf_counter()
    np.savez_compressed(
        tmp_path,
        tty_chars=all_chars,
        tty_colors=all_colors,
        offsets=np.zeros(len(frames_list) + 1, dtype=np.int64),
    )
    t_savez = time.perf_counter() - t0
    npz_mb = os.path.getsize(tmp_path) / 1e6
    os.unlink(tmp_path)

    # --- summary ---
    total_read   = sum(times_read)
    total_decode = sum(times_decode)
    total_cast   = sum(times_cast)
    total_frames = sum(frames_list)
    total_size   = sum(sizes_mb)
    grand_total  = total_read + total_decode + total_cast + t_concat + t_savez

    print(f"\n{'─'*70}")
    print(f"  games:   {len(times_read)}   frames: {total_frames:,}   input: {total_size:.1f} MB   output: {npz_mb:.1f} MB")
    print(f"{'─'*70}")

    stages = [
        ("read (NFS I/O)", total_read),
        ("decode (C ext)", total_decode),
        ("cast (numpy)",   total_cast),
        ("concat",         t_concat),
        ("savez_compress", t_savez),
    ]
    for label, t in stages:
        bar = _bar(t, grand_total)
        pct = 100 * t / grand_total if grand_total > 0 else 0
        print(f"  {label:<18} {_fmt(t):>9}  {pct:5.1f}%  {bar}")

    print(f"{'─'*70}")
    print(f"  {'total':<18} {_fmt(grand_total):>9}")
    print(f"\n  throughput:  {total_frames / grand_total:,.0f} frames/s  |  {total_size / grand_total:.1f} MB/s (input)")
    print(f"  median/game: read={_fmt(median(times_read))}  decode={_fmt(median(times_decode))}  cast={_fmt(median(times_cast))}")


if __name__ == "__main__":
    main()
