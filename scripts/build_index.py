"""Build index.npz from an already-converted npz dataset directory.

Run this after scripts/convert_to_npz.py finishes (or while it is still running
to index whatever has been written so far).

Usage:
    python -m scripts.build_index --npz-dir /scratch/uceeepi/lom/datasets/nld-nao-npz
    python -m scripts.build_index --npz-dir /scratch/uceeepi/lom/datasets/nld-aa-npz  --workers 32

Score extraction:
    nld-aa:  reads the saved `scores` array directly (exact, cheap).
    nld-nao: parses `S:<N>` from row 22 of tty_chars scanning backwards from
             the last frame (the status bar shows the running game score there).
             Stops at the first non-blank status line, so it's O(1) per game.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
from dataclasses import dataclass

import numpy as np
import tyro


@dataclass
class Args:
    npz_dir: str
    """Root of the converted dataset (contains player subdirs or autoascend/)."""
    workers: int = 32
    """Parallel workers for reading frame counts and scores."""


_SCORE_RE = re.compile(rb"S:(\d+)")


def _max_score_from_screen(tty_chars: np.ndarray) -> int:
    """Scan all frames for S:N on row 22 and return the maximum.

    A single file can contain multiple game lives (score resets on death),
    so the last frame is not necessarily the highest score.
    """
    max_score = 0
    for t in range(len(tty_chars)):
        m = _SCORE_RE.search(bytes(tty_chars[t, 22]))
        if m:
            s = int(m.group(1))
            if s > max_score:
                max_score = s
    return max_score


def _worker(path: str) -> tuple[str, int, int]:
    try:
        with np.load(path) as f:
            if "scores" in f:
                # nld-aa: scores array is cheap to load (T×4 bytes vs T×24×80)
                scores = f["scores"]
                n = int(scores.shape[0])
                max_score = int(scores.max())
            else:
                # nld-nao: parse from screen
                chars = f["tty_chars"]
                n = int(chars.shape[0])
                max_score = _max_score_from_screen(chars)
        return path, n, max_score
    except Exception:
        return path, -1, -1


def main() -> None:
    args = tyro.cli(Args)

    print(f"Scanning {args.npz_dir} ...", flush=True)
    paths = []
    for dirpath, _, filenames in os.walk(args.npz_dir):
        for fname in filenames:
            if fname.endswith(".npz") and fname != "index.npz":
                paths.append(os.path.join(dirpath, fname))

    total = len(paths)
    print(f"Found {total:,} .npz files. Building index with {args.workers} workers ...", flush=True)

    log_every = max(1, total // 100)
    good_paths:   list[str] = []
    good_lengths: list[int] = []
    good_scores:  list[int] = []
    errors = 0

    with mp.Pool(args.workers) as pool:
        for i, (path, n, max_score) in enumerate(pool.imap_unordered(_worker, paths), 1):
            if n >= 0:
                good_paths.append(path)
                good_lengths.append(n)
                good_scores.append(max_score)
            else:
                errors += 1
            if i % log_every == 0 or i == total:
                print(f"  [{i:>9,} / {total:,}]  ok={len(good_paths):,}  errors={errors}", flush=True)

    index_path = os.path.join(args.npz_dir, "index.npz")
    np.savez(
        index_path,
        paths=np.array(good_paths, dtype=object),
        lengths=np.array(good_lengths, dtype=np.int32),
        max_scores=np.array(good_scores, dtype=np.int32),
    )

    scores_arr = np.array(good_scores, dtype=np.int32)
    print(f"\nWrote {index_path}  ({len(good_paths):,} games, {errors} errors)")
    print(f"Score stats:")
    print(f"  min={scores_arr.min():,}  median={int(np.median(scores_arr)):,}  "
          f"p90={int(np.percentile(scores_arr, 90)):,}  max={scores_arr.max():,}")


if __name__ == "__main__":
    main()
