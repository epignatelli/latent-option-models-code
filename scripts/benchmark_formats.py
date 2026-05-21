"""Measure the size ratio between ttyrec and numpy formats for NLD-AA sessions.

Run on the cluster where NLD-AA data lives:

    python -m scripts.benchmark_formats --nle-data-dir /scratch/uceeepi/lom/datasets --n-games 20
"""

from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass

os.environ.setdefault("NLE_DATA_PATH", os.path.abspath("nle_data"))

import numpy as np
import tyro


@dataclass
class Args:
    nle_data_dir: str = "nle_data"
    """Root directory containing the nld-aa/ subdirectory."""
    n_games: int = 20
    """Number of games to sample for the benchmark."""


def _decode_ttyrec(
    ttyrec_files: list[str], ttyrec_version: int, rows: int = 24, cols: int = 80
) -> tuple[np.ndarray, list[str]]:
    """Decode ttyrec parts into a (T, rows, cols) uint8 array.

    Returns (chars_array, consumed_files) so callers can measure the on-disk
    size of exactly the parts that were decoded (avoids counting unread parts).
    """
    from nle import _pyconverter as nle_converter

    MAX_FRAMES = 200_000  # ~10 parts; large enough for representative ratios
    chars = np.zeros((MAX_FRAMES, rows, cols), dtype=np.uint8)
    colors = np.zeros((MAX_FRAMES, rows, cols), dtype=np.int8)
    cursors = np.zeros((MAX_FRAMES, 2), dtype=np.int16)
    timestamps = np.zeros(MAX_FRAMES, dtype=np.int64)
    inputs = np.zeros(MAX_FRAMES, dtype=np.uint8)
    scores = np.zeros(MAX_FRAMES, dtype=np.int32)

    conv = nle_converter.Converter(rows, cols, ttyrec_version)
    filled = 0
    consumed: list[str] = []

    for part_idx, path in enumerate(ttyrec_files):
        if filled >= MAX_FRAMES:
            break
        conv.load_ttyrec(path, gameid=1, part=part_idx)
        space = MAX_FRAMES - filled
        remaining = conv.convert(
            chars[filled:], colors[filled:], cursors[filled:],
            timestamps[filled:], inputs[filled:], scores[filled:],
        )
        frames_read = space - remaining
        if frames_read > 0:
            consumed.append(path)
        filled += frames_read

    nonzero = np.any(chars[:filled] != 0, axis=(1, 2))
    last = int(np.max(np.where(nonzero))) + 1 if nonzero.any() else 1
    return chars[:last], consumed


def _load_episodes(nle_data_dir: str, n: int):
    """Scan nle_data/ directories and decode the first n complete games.

    Returns (episodes, ttyrec_sizes):
        episodes      – list of (T, 24, 80) uint8 arrays
        ttyrec_sizes  – list of total on-disk compressed sizes (bytes)
    """
    nle_data = os.path.join(nle_data_dir, "nld-aa", "nle_data")
    game_dirs = sorted(os.listdir(nle_data))

    episodes: list[np.ndarray] = []
    ttyrec_sizes: list[int] = []

    for gdir in game_dirs:
        if len(episodes) >= n:
            break
        dir_path = os.path.join(nle_data, gdir)
        bz2_files = glob.glob(os.path.join(dir_path, "*.ttyrec*.bz2"))
        if not bz2_files:
            continue

        # Sort parts numerically: nle.<pid>.<part>.ttyrec<ver>.bz2
        bz2_files.sort(key=lambda f: int(os.path.basename(f).split(".")[2]))

        sample = os.path.basename(bz2_files[0])
        ver_str = sample.split("ttyrec")[-1].replace(".bz2", "")
        ttyrec_version = int(ver_str) if ver_str.isdigit() else 1

        try:
            arr, consumed = _decode_ttyrec(bz2_files, ttyrec_version)
        except Exception as exc:
            print(f"  skip {gdir}: {exc}")
            continue

        # Only count the on-disk size for the parts actually decoded
        on_disk = sum(os.path.getsize(f) for f in consumed)
        episodes.append(arr)
        ttyrec_sizes.append(on_disk)

    return episodes, ttyrec_sizes


def _sizeof_npz_compressed(arr: np.ndarray) -> int:
    buf = io.BytesIO()
    np.savez_compressed(buf, obs=arr)
    return buf.tell()


def _sizeof_npz_uncompressed(arr: np.ndarray) -> int:
    buf = io.BytesIO()
    np.savez(buf, obs=arr)
    return buf.tell()


def _sizeof_npy(arr: np.ndarray) -> int:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.tell()


def main() -> None:
    args = tyro.cli(Args)

    nle_data = os.path.join(args.nle_data_dir, "nld-aa", "nle_data")
    if not os.path.isdir(nle_data):
        raise FileNotFoundError(f"nle_data directory not found: {nle_data}")

    print(f"Loading {args.n_games} episodes from {nle_data} ...")
    episodes, ttyrec_sizes = _load_episodes(args.nle_data_dir, args.n_games)

    print(f"\nLoaded {len(episodes)} episodes.\n")

    rows = []
    for i, arr in enumerate(episodes):
        T = len(arr)
        raw_bytes   = arr.nbytes
        npz_c_bytes = _sizeof_npz_compressed(arr)
        npz_u_bytes = _sizeof_npz_uncompressed(arr)
        npy_bytes   = _sizeof_npy(arr)
        tty_bytes   = ttyrec_sizes[i]

        ratio_c   = tty_bytes / npz_c_bytes
        ratio_u   = tty_bytes / npz_u_bytes
        ratio_raw = tty_bytes / raw_bytes
        print(
            f"  game {i:3d}: T={T:6d}  "
            f"ttyrec={tty_bytes/1e6:6.2f}MB  "
            f"raw={raw_bytes/1e6:6.2f}MB  "
            f"npy={npy_bytes/1e6:6.2f}MB  "
            f"npz={npz_c_bytes/1e6:6.2f}MB  "
            f"| ttyrec/raw={ratio_raw:.3f}  ttyrec/npz={ratio_c:.3f}"
        )
        rows.append((T, raw_bytes, npy_bytes, npz_u_bytes, npz_c_bytes, tty_bytes))

    if rows:
        total_ttyrec = sum(r[5] for r in rows)
        total_raw    = sum(r[1] for r in rows)
        total_npy    = sum(r[2] for r in rows)
        total_npz_u  = sum(r[3] for r in rows)
        total_npz_c  = sum(r[4] for r in rows)

        print(f"\n{'='*70}")
        print(f"Totals over {len(rows)} games:")
        print(f"  ttyrec (on disk):      {total_ttyrec/1e6:.1f} MB")
        print(f"  raw uint8 array:       {total_raw/1e6:.1f} MB")
        print(f"  .npy (uncompressed):   {total_npy/1e6:.1f} MB")
        print(f"  .npz (uncompressed):   {total_npz_u/1e6:.1f} MB")
        print(f"  .npz (compressed):     {total_npz_c/1e6:.1f} MB")
        print(f"\nRatios (ttyrec / format):")
        print(f"  ttyrec / raw:          {total_ttyrec/total_raw:.3f}x")
        print(f"  ttyrec / npy:          {total_ttyrec/total_npy:.3f}x")
        print(f"  ttyrec / npz_u:        {total_ttyrec/total_npz_u:.3f}x")
        print(f"  ttyrec / npz_c:        {total_ttyrec/total_npz_c:.3f}x")
        print(f"\n  → npz_c / ttyrec:     {total_npz_c/total_ttyrec:.1f}x  (>1 means npz is larger)")


if __name__ == "__main__":
    main()
