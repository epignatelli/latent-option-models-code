"""Convert NLD-AA or NLD-NAO ttyrec files to compressed .npz format.

One .npz per episode. Fields saved:
  nld-aa:  tty_chars, tty_colors, tty_cursor, keypresses, scores, done
  nld-nao: tty_chars, tty_colors, tty_cursor, done

Output layout:
  nld-aa:  <output_dir>/autoascend/<game_dir>.npz
  nld-nao: <output_dir>/<player>/<game>.npz

Usage:
    python -m scripts.convert_to_npz --dataset nld-aa  --nle-data-dir /scratch/uceeepi/lom/datasets
    python -m scripts.convert_to_npz --dataset nld-nao --nle-data-dir /scratch/uceeepi/lom/datasets
    python -m scripts.convert_to_npz --dataset all     --nle-data-dir /scratch/uceeepi/lom/datasets

Memory note:
    nld-aa peaks at ~15 GB RAM per worker. On this machine (375 GB, 128 cores)
    --workers 20 is safe for nld-aa. nld-nao is I/O-bound; 32 is a good cap.
"""

from __future__ import annotations

import glob
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ROWS, COLS = 24, 80

_KEYS_AA  = ("tty_chars", "tty_colors", "tty_cursor", "keypresses", "scores", "done")
_KEYS_NAO = ("tty_chars", "tty_colors", "tty_cursor", "done")


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


# ── worker ────────────────────────────────────────────────────────────────────

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **{k: arrays[k] for k in save_keys})
    return {"status": "ok", "frames": n_frames, "path": output_path}


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

def _run_one(dataset: str, nle_data_dir: str, output_dir: str, workers: int, min_frames: int) -> None:
    output_dir = output_dir or os.path.join(nle_data_dir, f"{dataset}-npz")
    os.makedirs(output_dir, exist_ok=True)

    print(f"dataset    : {dataset}")
    print(f"output     : {output_dir}")
    print(f"workers    : {workers}")
    if dataset == "nld-nao":
        print(f"min_frames : {min_frames}")
    print(flush=True)

    if dataset == "nld-aa":
        tasks = _discover_nld_aa(nle_data_dir, output_dir)
    else:
        tasks = _discover_nld_nao(nle_data_dir, output_dir, min_frames)

    total = len(tasks)
    print(f"games found: {total:,}\n", flush=True)

    # Load existing index so skipped files are still included.
    index_path = os.path.join(output_dir, "index.npz")
    if os.path.exists(index_path):
        ex = np.load(index_path, allow_pickle=True)
        ex_paths:   list[str] = list(ex["paths"])
        ex_lengths: list[int] = list(ex["lengths"].astype(int))
        existing_set: set[str] = set(ex_paths)
    else:
        ex_paths, ex_lengths, existing_set = [], [], set()

    counts = {"ok": 0, "skip": 0, "filter": 0, "error": 0}
    errors: list[str] = []
    new_entries: list[tuple[str, int]] = []
    log_every = max(1, total // 100)  # ~100 progress lines regardless of dataset size

    with mp.Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_convert_one, tasks), 1):
            counts[result["status"]] += 1
            if result["status"] == "ok":
                p = result["path"]
                if p not in existing_set:
                    new_entries.append((p, result["frames"]))
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

    # Write merged index (existing + newly converted).
    all_paths   = ex_paths   + [p for p, _ in new_entries]
    all_lengths = ex_lengths + [f for _, f in new_entries]
    if all_paths:
        np.savez(
            index_path,
            paths=np.array(all_paths, dtype=object),
            lengths=np.array(all_lengths, dtype=np.int32),
        )

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
        _run_one("nld-aa",  args.nle_data_dir, "", args.workers, args.min_frames)
        _run_one("nld-nao", args.nle_data_dir, "", args.workers, args.min_frames)
    else:
        _run_one(args.dataset, args.nle_data_dir, args.output_dir, args.workers, args.min_frames)


if __name__ == "__main__":
    main()
