"""Convert NLD-AA or NLD-NAO ttyrec files to compressed .npz format.

One .npz per episode. Fields saved:
  nld-aa:  tty_chars, tty_colors, tty_cursor, keypresses, scores, done
  nld-nao: tty_chars, tty_colors, tty_cursor, done

Output layout:
  nld-aa:  <output_dir>/autoascend/<game_dir>.npz
  nld-nao: <output_dir>/<player>/<game>.npz

After conversion an index.npz is written to output_dir with:
  paths      – (N,) object array of absolute .npz paths
  lengths    – (N,) int32 array of frame counts
  max_scores – (N,) int32 array of per-game max scores

Usage:
    python -m scripts.convert_to_npz --dataset nld-aa  --nle-data-dir /scratch/uceeepi/lom/datasets
    python -m scripts.convert_to_npz --dataset nld-nao --nle-data-dir /scratch/uceeepi/lom/datasets
    python -m scripts.convert_to_npz --dataset all     --nle-data-dir /scratch/uceeepi/lom/datasets

    # Skip conversion, (re)build index from already-converted files:
    python -m scripts.convert_to_npz --dataset nld-nao --nle-data-dir /scratch/uceeepi/lom/datasets --index-only

Memory note:
    nld-aa peaks at ~15 GB RAM per worker. On this machine (375 GB, 128 cores)
    --workers 20 is safe for nld-aa. nld-nao is I/O-bound; 32 is a good cap.
"""

from __future__ import annotations

import glob
import logging
import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ROWS, COLS = 24, 80

_KEYS_AA  = ("tty_chars", "tty_colors", "tty_cursor", "keypresses", "scores", "done")
_KEYS_NAO = ("tty_chars", "tty_colors", "tty_cursor", "done")

_SCORE_RE = re.compile(rb"S:(\d+)")


@dataclass
class Args:
    dataset: Literal["nld-aa", "nld-nao", "all"]
    """Which dataset to convert. 'all' converts nld-aa then nld-nao."""
    nle_data_dir: str
    """Root directory containing nld-aa/ or nld-nao/ subdirectories."""
    output_dir: str = ""
    """Output directory. Defaults to <nle_data_dir>/<dataset>-npz (ignored for 'all')."""
    workers: int = 20
    """Number of parallel worker processes.
    On this machine (375 GB RAM, 128 cores): nld-aa peaks at ~15 GB/worker
    so 20 workers is safe. nld-nao is I/O-bound; 32 is a reasonable cap."""
    min_frames: int = 50
    """Minimum decoded frames to keep. Applied to nld-nao only (nld-aa: no filter)."""
    index_only: bool = False
    """Skip ttyrec conversion; scan the output directory for existing .npz files
    and (re)build index.npz. Useful when conversion ran without index support."""


# ── score helpers ─────────────────────────────────────────────────────────────

def _max_score_from_arrays(arrays: dict, save_keys: tuple) -> int:
    """Extract max score from already-decoded arrays (called during conversion)."""
    if "scores" in save_keys:
        return int(arrays["scores"].max())
    # nld-nao: parse S:N from row 22 of tty_chars across all frames
    tty_chars = arrays["tty_chars"]
    max_score = 0
    for t in range(len(tty_chars)):
        m = _SCORE_RE.search(bytes(tty_chars[t, 22]))
        if m:
            s = int(m.group(1))
            if s > max_score:
                max_score = s
    return max_score


def _max_score_from_file(path: str) -> tuple[int, int]:
    """Read (n_frames, max_score) from an existing .npz file (called during index-only)."""
    with np.load(path) as f:
        if "scores" in f:
            scores = f["scores"]
            return int(scores.shape[0]), int(scores.max())
        chars = f["tty_chars"]
        n = int(chars.shape[0])
        max_score = 0
        for t in range(n):
            m = _SCORE_RE.search(bytes(chars[t, 22]))
            if m:
                s = int(m.group(1))
                if s > max_score:
                    max_score = s
        return n, max_score


def _index_worker(path: str) -> tuple[str, int, int]:
    try:
        n, max_score = _max_score_from_file(path)
        return path, n, max_score
    except Exception:
        return path, -1, -1


# ── decoding ─────────────────────────────────────────────────────────────────

def _decode(ttyrec_files: list[str], ttyrec_version: int) -> tuple[dict, int]:
    """Decode a list of ttyrec parts into a dict of numpy arrays.

    Returns (arrays_dict, n_frames). n_frames is 0 if decoding produced nothing.
    """
    from nle import _pyconverter as nle_converter

    # Use a larger single buffer for nld-nao (one file, variable length up to ~10k frames).
    # For nld-aa, parts are exactly 25k frames; 30k gives a small safety margin.
    chunk = 200_000 if len(ttyrec_files) == 1 else 30_000

    tmp_chars  = np.zeros((chunk, ROWS, COLS), dtype=np.uint8)
    tmp_colors = np.zeros((chunk, ROWS, COLS), dtype=np.int8)
    tmp_cursor = np.zeros((chunk, 2),          dtype=np.int16)
    tmp_ts     = np.zeros(chunk,               dtype=np.int64)
    tmp_kp     = np.zeros(chunk,               dtype=np.uint8)
    tmp_scores = np.zeros(chunk,               dtype=np.int32)

    conv = nle_converter.Converter(ROWS, COLS, ttyrec_version)

    chars_parts, colors_parts, cursor_parts = [], [], []
    kp_parts, scores_parts = [], []

    for part_idx, path in enumerate(ttyrec_files):
        conv.load_ttyrec(path, gameid=1, part=part_idx)
        remaining = conv.convert(
            tmp_chars, tmp_colors, tmp_cursor,
            tmp_ts, tmp_kp, tmp_scores,
        )
        n = chunk - remaining
        if n == 0:
            continue
        chars_parts.append(tmp_chars[:n].copy())
        colors_parts.append(tmp_colors[:n].copy())
        cursor_parts.append(tmp_cursor[:n].copy())
        kp_parts.append(tmp_kp[:n].copy())
        scores_parts.append(tmp_scores[:n].copy())

    if not chars_parts:
        return {}, 0

    tty_chars  = np.concatenate(chars_parts)
    tty_colors = np.concatenate(colors_parts)
    tty_cursor = np.concatenate(cursor_parts)
    keypresses = np.concatenate(kp_parts)
    scores     = np.concatenate(scores_parts)
    done       = np.zeros(len(tty_chars), dtype=np.uint8)
    done[0]    = 1

    return {
        "tty_chars":  tty_chars,
        "tty_colors": tty_colors,
        "tty_cursor": tty_cursor,
        "keypresses": keypresses,
        "scores":     scores,
        "done":       done,
    }, len(tty_chars)


# ── workers ───────────────────────────────────────────────────────────────────

def _convert_one(task: tuple) -> dict:
    input_files, output_path, ttyrec_version, min_frames, save_keys = task

    if os.path.exists(output_path):
        return {"status": "skip"}

    try:
        arrays, n_frames = _decode(input_files, ttyrec_version)
    except Exception as exc:
        return {"status": "error", "msg": f"{input_files[0]}: {exc}"}

    if n_frames < min_frames:
        return {"status": "filter"}

    max_score = _max_score_from_arrays(arrays, save_keys)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **{k: arrays[k] for k in save_keys})
    return {"status": "ok", "frames": n_frames, "path": output_path, "max_score": max_score}


# ── task discovery ────────────────────────────────────────────────────────────

def _discover_nld_aa(nle_data_dir: str, output_dir: str) -> list[tuple]:
    data_root = os.path.join(nle_data_dir, "nld-aa", "nle_data")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"nld-aa data not found at {data_root}")

    tasks = []
    for gdir in sorted(os.listdir(data_root)):
        dir_path = os.path.join(data_root, gdir)
        parts = sorted(
            glob.glob(os.path.join(dir_path, "*.ttyrec*.bz2")),
            key=lambda f: int(os.path.basename(f).split(".")[2]),
        )
        if not parts:
            continue
        output_path = os.path.join(output_dir, "autoascend", f"{gdir}.npz")
        tasks.append((parts, output_path, 3, 0, _KEYS_AA))
    return tasks


def _discover_nld_nao(nle_data_dir: str, output_dir: str, min_frames: int) -> list[tuple]:
    data_root = os.path.join(nle_data_dir, "nld-nao", "nld-nao-unzipped")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"nld-nao data not found at {data_root}")

    tasks = []
    for player in sorted(os.listdir(data_root)):
        player_dir = os.path.join(data_root, player)
        if not os.path.isdir(player_dir):
            continue
        for fname in sorted(os.listdir(player_dir)):
            if not fname.endswith(".bz2"):
                continue
            src = os.path.join(player_dir, fname)
            safe = fname.replace(":", "-").replace(".bz2", "") + ".npz"
            output_path = os.path.join(output_dir, player, safe)
            tasks.append(([src], output_path, 1, min_frames, _KEYS_NAO))
    return tasks


# ── main ─────────────────────────────────────────────────────────────────────

def _write_index(index_path: str, paths: list[str], lengths: list[int], scores: list[int]) -> None:
    np.savez(
        index_path,
        paths=np.array(paths, dtype=object),
        lengths=np.array(lengths, dtype=np.int32),
        max_scores=np.array(scores, dtype=np.int32),
    )


def _run_index_only(output_dir: str, workers: int) -> None:
    print(f"index-only : scanning {output_dir}", flush=True)

    npz_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(output_dir)
        for f in files
        if f.endswith(".npz") and f != "index.npz"
    ]
    total = len(npz_files)
    print(f"files found: {total:,}\n", flush=True)

    log_every = max(1, total // 100)
    good_paths:   list[str] = []
    good_lengths: list[int] = []
    good_scores:  list[int] = []
    errors = 0

    with mp.Pool(workers) as pool:
        for i, (path, n, max_score) in enumerate(
            pool.imap_unordered(_index_worker, npz_files), 1
        ):
            if n >= 0:
                good_paths.append(path)
                good_lengths.append(n)
                good_scores.append(max_score)
            else:
                errors += 1
            if i % log_every == 0 or i == total:
                print(
                    f"  [{i:>9,} / {total:,}]  "
                    f"ok={len(good_paths):,}  errors={errors}",
                    flush=True,
                )

    index_path = os.path.join(output_dir, "index.npz")
    _write_index(index_path, good_paths, good_lengths, good_scores)

    scores_arr = np.array(good_scores, dtype=np.int32)
    print(f"\n{'='*60}")
    print(f"indexed   : {len(good_paths):,}")
    print(f"errors    : {errors}")
    print(f"index     : {index_path}")
    print(f"scores    : min={scores_arr.min():,}  median={int(np.median(scores_arr)):,}"
          f"  p90={int(np.percentile(scores_arr, 90)):,}  max={scores_arr.max():,}")
    print()


def _run_one(dataset: str, nle_data_dir: str, output_dir: str, workers: int, min_frames: int,
             index_only: bool) -> None:
    output_dir = output_dir or os.path.join(nle_data_dir, f"{dataset}-npz")
    os.makedirs(output_dir, exist_ok=True)

    print(f"dataset    : {dataset}")
    print(f"output     : {output_dir}")
    print(f"workers    : {workers}")
    if dataset == "nld-nao":
        print(f"min_frames : {min_frames}")
    print(flush=True)

    if index_only:
        _run_index_only(output_dir, workers)
        return

    if dataset == "nld-aa":
        tasks = _discover_nld_aa(nle_data_dir, output_dir)
    else:
        tasks = _discover_nld_nao(nle_data_dir, output_dir, min_frames)

    total = len(tasks)
    print(f"games found: {total:,}\n", flush=True)

    # Merge with existing index so re-runs accumulate rather than restart.
    index_path = os.path.join(output_dir, "index.npz")
    if os.path.exists(index_path):
        ex = np.load(index_path, allow_pickle=True)
        ex_paths:   list[str] = list(ex["paths"])
        ex_lengths: list[int] = list(ex["lengths"].astype(int))
        ex_scores:  list[int] = list(ex["max_scores"].astype(int)) if "max_scores" in ex else [0] * len(ex_paths)
        existing_set: set[str] = set(ex_paths)
    else:
        ex_paths, ex_lengths, ex_scores, existing_set = [], [], [], set()

    counts = {"ok": 0, "skip": 0, "filter": 0, "error": 0}
    errors: list[str] = []
    new_entries: list[tuple[str, int, int]] = []
    log_every = max(1, total // 100)

    with mp.Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_convert_one, tasks), 1):
            counts[result["status"]] += 1
            if result["status"] == "ok":
                p = result["path"]
                if p not in existing_set:
                    new_entries.append((p, result["frames"], result["max_score"]))
            elif result["status"] == "error":
                errors.append(result.get("msg", "unknown"))
            if i % log_every == 0 or i == total:
                print(
                    f"  [{i:>9,} / {total:,}]  "
                    f"ok={counts['ok']:,}  "
                    f"skip={counts['skip']:,}  "
                    f"filter={counts['filter']:,}  "
                    f"error={counts['error']:,}",
                    flush=True,
                )

    all_paths   = ex_paths   + [p          for p, _, _ in new_entries]
    all_lengths = ex_lengths + [n          for _, n, _ in new_entries]
    all_scores  = ex_scores  + [s          for _, _, s in new_entries]
    if all_paths:
        _write_index(index_path, all_paths, all_lengths, all_scores)

    print(f"\n{'='*60}")
    print(f"converted : {counts['ok']:,}")
    print(f"skipped   : {counts['skip']:,}  (already existed)")
    print(f"filtered  : {counts['filter']:,}  (< {min_frames} frames)")
    print(f"errors    : {counts['error']:,}")
    if all_paths:
        print(f"index     : {index_path}  ({len(all_paths):,} games)")
    if errors:
        print("\nFirst 10 errors:")
        for msg in errors[:10]:
            print(f"  {msg}")
    print()


def main() -> None:
    args = tyro.cli(Args)

    if args.dataset == "all":
        _run_one("nld-aa",  args.nle_data_dir, "", args.workers, args.min_frames, args.index_only)
        _run_one("nld-nao", args.nle_data_dir, "", args.workers, args.min_frames, args.index_only)
    else:
        _run_one(args.dataset, args.nle_data_dir, args.output_dir, args.workers, args.min_frames,
                 args.index_only)


if __name__ == "__main__":
    main()
