"""Measure the size ratio between ttyrec and numpy formats for NLD-AA sessions.

Run on the cluster where NLD-AA data lives:

    python -m scripts.benchmark_formats --nle_data_dir /scratch/uceeepi/lom/datasets --n_games 20
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass

os.environ.setdefault("NLE_DATA_PATH", os.path.abspath("nle_data"))

import numpy as np
import tyro


@dataclass
class Args:
    nle_data_dir: str = "nle_data"
    """Root directory containing nld-aa.db and nld-aa/ subdirectory."""
    n_games: int = 20
    """Number of games to sample for the benchmark."""


def _load_episodes(db_path: str, dataset_name: str, n: int):
    from nle.dataset import dataset as nle_dataset
    # TtyrecDataset requires a fixed seq_length; use a large cap and trim trailing blank frames
    SEQ_LEN = 100_000
    ds = nle_dataset.TtyrecDataset(
        dataset_name=dataset_name,
        dbfilename=db_path,
        seq_length=SEQ_LEN,
        batch_size=1,
    )
    episodes = []
    for i, ep in enumerate(ds):
        if i >= n:
            break
        arr = ep["tty_chars"][0]  # (1, T, 24, 80) → (T, 24, 80) uint8, possibly zero-padded
        # trim trailing all-zero frames (padding)
        nonzero = np.any(arr != 0, axis=(1, 2))
        last = int(np.max(np.where(nonzero))) + 1 if nonzero.any() else 1
        episodes.append(arr[:last])
    return episodes


def _ttyrec_paths(nle_data_dir: str, db_path: str, dataset_name: str, n: int) -> list[str]:
    """Return paths to the raw ttyrec files for the first n games via the DB."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT filename FROM games WHERE dataset_name=? LIMIT ?",
        (dataset_name, n),
    ).fetchall()
    conn.close()
    return [os.path.join(nle_data_dir, r[0]) for r in rows if os.path.exists(os.path.join(nle_data_dir, r[0]))]


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

    db_path = os.path.join(args.nle_data_dir, "nld-aa.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"Loading {args.n_games} episodes from {db_path} ...")
    episodes = _load_episodes(db_path, "nld-aa", args.n_games)
    ttyrec_paths = _ttyrec_paths(args.nle_data_dir, db_path, "nld-aa", args.n_games)

    print(f"\nLoaded {len(episodes)} episodes.")
    print(f"Found {len(ttyrec_paths)} matching ttyrec files on disk.\n")

    rows = []
    for i, arr in enumerate(episodes):
        T = len(arr)
        raw_bytes      = arr.nbytes                      # T * 1920
        npz_c_bytes    = _sizeof_npz_compressed(arr)
        npz_u_bytes    = _sizeof_npz_uncompressed(arr)
        npy_bytes      = _sizeof_npy(arr)
        ttyrec_bytes   = os.path.getsize(ttyrec_paths[i]) if i < len(ttyrec_paths) else None

        rows.append((T, raw_bytes, npy_bytes, npz_u_bytes, npz_c_bytes, ttyrec_bytes))

        if ttyrec_bytes:
            ratio_c  = ttyrec_bytes / npz_c_bytes
            ratio_u  = ttyrec_bytes / npz_u_bytes
            ratio_raw = ttyrec_bytes / raw_bytes
            print(
                f"  game {i:3d}: T={T:6d}  "
                f"ttyrec={ttyrec_bytes/1e6:6.2f}MB  "
                f"raw={raw_bytes/1e6:6.2f}MB  "
                f"npy={npy_bytes/1e6:6.2f}MB  "
                f"npz={npz_c_bytes/1e6:6.2f}MB  "
                f"| ttyrec/raw={ratio_raw:.3f}  ttyrec/npz={ratio_c:.3f}"
            )
        else:
            print(
                f"  game {i:3d}: T={T:6d}  "
                f"ttyrec=N/A  "
                f"raw={raw_bytes/1e6:6.2f}MB  "
                f"npy={npy_bytes/1e6:6.2f}MB  "
                f"npz={npz_c_bytes/1e6:6.2f}MB"
            )

    # Aggregate
    valid = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows if r[5] is not None]
    if valid:
        total_ttyrec = sum(r[5] for r in valid)
        total_raw    = sum(r[1] for r in valid)
        total_npy    = sum(r[2] for r in valid)
        total_npz_u  = sum(r[3] for r in valid)
        total_npz_c  = sum(r[4] for r in valid)

        print(f"\n{'='*70}")
        print(f"Totals over {len(valid)} games:")
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
