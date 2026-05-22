"""Prepare NLE datasets for LOM pre-training.

Pipeline stages (run in order; each is individually skippable):

  download   fetch archives from remote storage
  extract    unzip / untar archives
  db         build NLE SQLite database (nld-aa / nld-nao only)
  convert    decode ttyrec → per-game .npz (nld-aa / nld-nao only)
  index      scan output directory and write index.npz

Datasets:

  nao-top10  NAO Top-10, DeepMind processed .npz, ~12 GB
             stages: download → extract → index
  nld-aa     NLD-AA (Autoascend AI), 16 zip archives, ~100 GB
             stages: download → extract → db → convert → index
  nld-nao    NLD-NAO (NetHack.alt.org), 41 zip archives, ~500 GB
             stages: download → extract → db → convert → index
  all        run all three datasets in sequence

Output layout under --output-dir:

  nao-top10/nao_top10/          extracted npz sessions
  nao-top10/index.npz
  nld-aa/                       extracted ttyrec files
  nld-aa.db                     NLE SQLite database
  nld-aa-npz/autoascend/        converted per-game .npz files
  nld-aa-npz/index.npz
  nld-nao/                      extracted ttyrec files
  nld-nao.db
  nld-nao-npz/                  converted per-player .npz files (one file per player)
  nld-nao-npz/index.npz         rich index with per-player and per-game metadata
  zips/                         downloaded archives (removed unless --keep-archives)

Usage:

  # Full pipeline:
  python -m scripts.prepare_data nao-top10 --output-dir /scratch/uceeepi/lom/datasets
  python -m scripts.prepare_data nld-nao   --output-dir /scratch/uceeepi/lom/datasets
  python -m scripts.prepare_data all       --output-dir /scratch/uceeepi/lom/datasets

  # Data already downloaded — skip download + extract:
  python -m scripts.prepare_data nao-top10 --output-dir /scratch/... \\
      --skip-download --skip-extract

  # Re-index only (conversion already done):
  python -m scripts.prepare_data nld-nao --output-dir /scratch/... \\
      --skip-download --skip-extract --skip-db --skip-convert

  # Skip NLE DB build (not required when using NpzTrajectoryDataset):
  python -m scripts.prepare_data nld-nao --output-dir /scratch/... --skip-db
"""

from __future__ import annotations

import bisect
import glob
import logging
import multiprocessing as mp
import os
import re
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Union

os.environ.setdefault("NLE_DATA_PATH", os.path.abspath("nle_data"))

import numpy as np
import tyro
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROWS, COLS = 24, 80

_KEYS_AA  = ("tty_chars", "tty_colors", "tty_cursor", "keypresses", "scores", "done")
_KEYS_NAO = ("tty_chars", "tty_colors", "tty_cursor", "done")
_SCORE_RE = re.compile(rb"S:(\d+)")
_HEX_RE   = re.compile(r"^0x[0-9a-fA-F]+$")

_NLD_AA_BASE   = "https://dl.fbaipublicfiles.com/nld/nld-aa/"
_NLD_NAO_BASE  = "https://dl.fbaipublicfiles.com/nld/nld-nao/"
_NAO_TOP10_URL = "https://storage.googleapis.com/dm_nethack/nao_top10.tar"

# Populated before Pool creation; workers inherit via fork (Linux copy-on-write).
_xl_by_player: dict[str, list[dict]] = {}

_XLOG_NAMES = [
    "xlogfile.full.txt",
    "xlogfile.nh360", "xlogfile.nh361", "xlogfile.nh361dev",
    "xlogfile.nh362", "xlogfile.nh363+",
]

_GAME_META_DEFAULT: dict = {
    "length": 0, "score": 0, "turns": -1, "dlvl": -1, "conduct": 0,
    "ascended": False, "role": "???", "race": "???", "align": "???",
    "death": "", "flags": 0, "timestamp": 0,
}


# --------------------------------------------------------------------------- #
# --- Config ----------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

@dataclass
class BaseArgs:
    output_dir: str = "nle_data"
    """Root directory for all datasets and outputs."""
    workers: int = 4
    """Parallel workers for download and conversion."""
    keep_archives: bool = False
    """Keep zip / tar archives after extraction."""
    min_frames: int = 50
    """Minimum decoded frames to retain a game (nld-nao only)."""
    nld_aa_subdir: str = "nle_data"
    """Sub-directory inside nld-aa/ holding per-game ttyrec dirs."""

    skip_download: bool = False
    """Skip the download stage (archives must already be present)."""
    skip_extract: bool = False
    """Skip the extract stage (archives must already be extracted)."""
    skip_db: bool = False
    """Skip building the NLE SQLite database (nld-aa / nld-nao only)."""
    skip_convert: bool = False
    """Skip ttyrec → npz conversion (nld-aa / nld-nao only); jump straight to index."""
    skip_index: bool = False
    """Skip building / updating index.npz."""


@dataclass
class NldAaArgs(BaseArgs):
    """NLD-AA (Autoascend AI gameplay, 16 zips, ~100 GB)."""


@dataclass
class NldNaoArgs(BaseArgs):
    """NLD-NAO (NetHack.alt.org gameplay, 41 zips, ~500 GB)."""


@dataclass
class NaoTop10Args(BaseArgs):
    """NAO Top-10 processed .npz dataset from DeepMind (1 tar, ~12 GB)."""


@dataclass
class AllArgs(BaseArgs):
    """Run all three datasets in sequence."""


# --------------------------------------------------------------------------- #
# --- Sentinel helpers ------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _sentinel(directory: str) -> str:
    return os.path.join(directory, ".done")


def _is_done(directory: str) -> bool:
    return os.path.exists(_sentinel(directory))


def _mark_done(directory: str) -> None:
    open(_sentinel(directory), "w").close()


# --------------------------------------------------------------------------- #
# --- Download helpers ------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    tmp = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _parallel_download(base_url: str, filenames: list[str], dest_dir: str, workers: int) -> None:
    pending = [n for n in filenames if not os.path.exists(os.path.join(dest_dir, n))]
    if not pending:
        print(f"  all {len(filenames)} archives already present — skipping download.")
        return
    print(f"  downloading {len(pending)}/{len(filenames)} archives ({workers} workers) ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download, base_url + name, os.path.join(dest_dir, name)): name
            for name in pending
        }
        with tqdm(total=len(futures), unit="file") as bar:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to download {name}: {exc}") from exc
                bar.set_postfix(file=name)
                bar.update(1)


def _remove_archives(filenames: list[str], archive_dir: str) -> None:
    for name in filenames:
        path = os.path.join(archive_dir, name)
        if os.path.exists(path):
            os.remove(path)
    try:
        os.rmdir(archive_dir)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# --- Extract helpers -------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _extract_zips(filenames: list[str], zip_dir: str, dest_dir: str) -> None:
    if _is_done(dest_dir):
        print(f"  already extracted to {dest_dir} — skipping.")
        return
    print(f"  extracting {len(filenames)} archives to {dest_dir} ...")
    for name in tqdm(filenames, unit="file"):
        with zipfile.ZipFile(os.path.join(zip_dir, name), "r") as zf:
            zf.extractall(dest_dir)
    _mark_done(dest_dir)


def _extract_tar(tar_path: str, dest_dir: str) -> None:
    if _is_done(dest_dir):
        print(f"  already extracted to {dest_dir} — skipping.")
        return
    print(f"  extracting {tar_path} to {dest_dir} ...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest_dir)
    _mark_done(dest_dir)


# --------------------------------------------------------------------------- #
# --- NLE DB ----------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _build_nle_db(unzipped_dir: str, db_path: str, dataset_name: str, use_altorg: bool) -> None:
    if os.path.exists(db_path):
        print(f"  NLE database already exists at {db_path} — skipping.")
        return
    print(f"  building NLE database at {db_path} ...")
    try:
        import nle.dataset as nld
        import nle.dataset.db as nld_db
    except ImportError:
        raise ImportError(
            "NLE is required for DB build.\n"
            "  pip install git+https://github.com/NetHack-LE/nle.git@main"
        )
    nld_db.create(filename=db_path)
    if use_altorg:
        nld.add_altorg_directory(unzipped_dir, dataset_name, filename=db_path)
    else:
        nld.add_nledata_directory(unzipped_dir, dataset_name, filename=db_path)


# --------------------------------------------------------------------------- #
# --- Xlogfile helpers ------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _parse_xlog_line(line: str) -> dict[str, str]:
    """Parse one xlogfile line; auto-detects `:` vs `\t` separator."""
    sep = "\t" if "\t" in line else ":"
    result: dict[str, str] = {}
    for part in line.strip().split(sep):
        if "=" in part:
            k, _, v = part.partition("=")
            result[k] = v
    return result


def _load_xlogfiles(nld_nao_dir: str) -> dict[str, list[dict]]:
    """Load all xlogfile variants; group entries by player name, sort by starttime."""
    by_player: dict[str, list] = defaultdict(list)
    total = 0
    for fname in _XLOG_NAMES:
        path = os.path.join(nld_nao_dir, fname)
        if not os.path.exists(path):
            continue
        n = 0
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                entry = _parse_xlog_line(line)
                name = entry.get("name", "")
                if name:
                    by_player[name].append(entry)
                    n += 1
        total += n
        print(f"  xlogfile {fname}: {n:,} entries", flush=True)
    for entries in by_player.values():
        entries.sort(key=lambda e: int(e.get("starttime", 0) or 0))
    print(f"  xlogfiles total: {total:,} entries, {len(by_player):,} players", flush=True)
    return dict(by_player)


def _parse_filename_ts(bz2_path: str) -> int:
    """Extract Unix timestamp from a ttyrec filename: YYYY-MM-DD.HH:MM:SS.ttyrec.bz2"""
    stem = os.path.basename(bz2_path).replace(".ttyrec.bz2", "")
    try:
        return int(datetime.strptime(stem, "%Y-%m-%d.%H:%M:%S")
                   .replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return 0


def _match_xlog_entry(entries: list[dict], file_ts: int) -> dict:
    """Return the xlogfile entry whose starttime is closest to file_ts."""
    if not entries:
        return {}
    times = [int(e.get("starttime", 0) or 0) for e in entries]
    pos = bisect.bisect_left(times, file_ts)
    candidates = []
    if pos < len(entries):
        candidates.append(entries[pos])
    if pos > 0:
        candidates.append(entries[pos - 1])
    return min(candidates, key=lambda e: abs(int(e.get("starttime", 0) or 0) - file_ts))


def _hex_or_int(s: str, default: int = 0) -> int:
    try:
        s = s.strip()
        return int(s, 16) if _HEX_RE.match(s) else int(s)
    except (ValueError, AttributeError):
        return default


def _game_meta_from_xlog(entry: dict, n_frames: int, file_ts: int) -> dict:
    death = entry.get("death", "") or ""
    return {
        "length":   n_frames,
        "score":    int(entry.get("points",  0) or 0),
        "turns":    int(entry.get("turns",  -1) or -1),
        "dlvl":     int(entry.get("maxlvl", -1) or -1),
        "conduct":  _hex_or_int(entry.get("conduct", "0")),
        "ascended": death.lower().startswith("ascended"),
        "role":     (entry.get("role",  "???") or "???")[:3],
        "race":     (entry.get("race",  "???") or "???")[:3],
        "align":    (entry.get("align", "???") or "???")[:3],
        "death":    death[:128],
        "flags":    _hex_or_int(entry.get("flags", "0")),
        "timestamp": file_ts,
    }


# --------------------------------------------------------------------------- #
# --- ttyrec → npz conversion ------------------------------------------------ #
# --------------------------------------------------------------------------- #

def _max_score_from_arrays(arrays: dict, save_keys: tuple) -> int:
    if "scores" in save_keys:
        return int(arrays["scores"].max())
    tty_chars = arrays["tty_chars"]
    max_score = 0
    for t in range(len(tty_chars)):
        m = _SCORE_RE.search(bytes(tty_chars[t, 22]))
        if m:
            s = int(m.group(1))
            if s > max_score:
                max_score = s
    return max_score


def _decode(ttyrec_files: list[str], ttyrec_version: int) -> tuple[dict, int]:
    from nle import _pyconverter as nle_converter

    chunk = 200_000 if len(ttyrec_files) == 1 else 30_000
    tmp_chars  = np.zeros((chunk, ROWS, COLS), dtype=np.uint8)
    tmp_colors = np.zeros((chunk, ROWS, COLS), dtype=np.int8)
    tmp_cursor = np.zeros((chunk, 2),          dtype=np.int16)
    tmp_ts     = np.zeros(chunk,               dtype=np.int64)
    tmp_kp     = np.zeros(chunk,               dtype=np.uint8)
    tmp_scores = np.zeros(chunk,               dtype=np.int32)

    conv = nle_converter.Converter(ROWS, COLS, ttyrec_version)
    chars_parts, colors_parts, cursor_parts, kp_parts, scores_parts = [], [], [], [], []

    for part_idx, path in enumerate(ttyrec_files):
        conv.load_ttyrec(path, gameid=1, part=part_idx)
        remaining = conv.convert(tmp_chars, tmp_colors, tmp_cursor, tmp_ts, tmp_kp, tmp_scores)
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
    done       = np.zeros(len(tty_chars), dtype=np.uint8)
    done[0]    = 1
    return {
        "tty_chars":  tty_chars,
        "tty_colors": np.concatenate(colors_parts),
        "tty_cursor": np.concatenate(cursor_parts),
        "keypresses": np.concatenate(kp_parts),
        "scores":     np.concatenate(scores_parts),
        "done":       done,
    }, len(tty_chars)


def _convert_one(task: tuple) -> dict:
    """Decode one nld-aa game (potentially multi-part) into a single npz."""
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


def _convert_player(task: tuple) -> dict:
    """Decode all games for one nld-nao player into a single per-player npz.

    Each bz2 file is one game.  Valid games are concatenated along the time axis;
    ``offsets`` (shape n_games+1) marks game boundaries;
    ``source_timestamps`` (shape n_games) stores the Unix timestamp parsed from
    each bz2 filename so xlogfile metadata can be reconstructed on restart.

    Uses the module-level ``_xl_by_player`` dict which workers inherit from the
    main process via fork.
    """
    input_files, output_path, ttyrec_version, min_frames, _save_keys, player_name = task
    xl_entries = _xl_by_player.get(player_name, [])

    # --- skip case: file already exists; rebuild game_meta without re-decoding ---
    if os.path.exists(output_path):
        try:
            with np.load(output_path) as f:
                offsets = f["offsets"]
                src_ts = f["source_timestamps"] if "source_timestamps" in f else None
        except Exception as exc:
            return {"status": "error", "msg": f"failed to read {output_path}: {exc}"}

        n_games = len(offsets) - 1
        game_meta: list[dict] = []
        for i in range(n_games):
            n_frames = int(offsets[i + 1]) - int(offsets[i])
            if src_ts is not None and i < len(src_ts):
                ts = int(src_ts[i])
                entry = _match_xlog_entry(xl_entries, ts) if xl_entries else {}
            else:
                ts, entry = 0, {}
            game_meta.append(_game_meta_from_xlog(entry, n_frames, ts))

        return {
            "status": "skip",
            "path": output_path,
            "frames": int(offsets[-1]),
            "games": n_games,
            "game_meta": game_meta,
        }

    # --- convert case: decode each bz2 file ---
    chars_parts: list[np.ndarray] = []
    colors_parts: list[np.ndarray] = []
    offsets_list: list[int] = [0]
    src_timestamps: list[int] = []
    game_meta = []
    total_frames = 0

    for bz2_path in sorted(input_files):
        file_ts = _parse_filename_ts(bz2_path)
        try:
            arrays, n_frames = _decode([bz2_path], ttyrec_version)
        except Exception:
            continue
        if not arrays or n_frames < min_frames:
            continue

        chars_parts.append(arrays["tty_chars"].astype(np.uint8))
        colors = (arrays["tty_colors"].astype(np.int16).clip(0, 31).astype(np.uint8)
                  if "tty_colors" in arrays
                  else np.zeros_like(arrays["tty_chars"], dtype=np.uint8))
        colors_parts.append(colors)
        total_frames += n_frames
        offsets_list.append(total_frames)
        src_timestamps.append(file_ts)

        entry = _match_xlog_entry(xl_entries, file_ts) if xl_entries else {}
        game_meta.append(_game_meta_from_xlog(entry, n_frames, file_ts))

    if not chars_parts:
        return {"status": "filter"}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        tty_chars=np.concatenate(chars_parts),
        tty_colors=np.concatenate(colors_parts),
        offsets=np.array(offsets_list, dtype=np.int64),
        source_timestamps=np.array(src_timestamps, dtype=np.int64),
    )
    return {
        "status": "ok",
        "path": output_path,
        "frames": total_frames,
        "games": len(offsets_list) - 1,
        "game_meta": game_meta,
    }


# --------------------------------------------------------------------------- #
# --- Discovery -------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _discover_nld_aa(nle_data_dir: str, output_dir: str, subdir: str) -> list[tuple]:
    data_root = os.path.join(nle_data_dir, "nld-aa", subdir)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"nld-aa data not found at {data_root}\n"
            f"Pass --nld-aa-subdir to override (current: '{subdir}')"
        )
    tasks = []
    for gdir in sorted(os.listdir(data_root)):
        dir_path = os.path.join(data_root, gdir)
        parts = sorted(
            glob.glob(os.path.join(dir_path, "*.ttyrec*.bz2")),
            key=lambda f: int(os.path.basename(f).split(".")[2]),
        )
        if not parts:
            continue
        tasks.append((parts, os.path.join(output_dir, "autoascend", f"{gdir}.npz"), 3, 0, _KEYS_AA))
    return tasks


def _discover_nld_nao(nle_data_dir: str, output_dir: str, min_frames: int) -> list[tuple]:
    """Build per-player task list and load xlogfile into the module-level global."""
    global _xl_by_player

    data_root = os.path.join(nle_data_dir, "nld-nao", "nld-nao-unzipped")
    if not os.path.isdir(data_root):
        data_root = os.path.join(nle_data_dir, "nld-nao")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"nld-nao data not found at {data_root}")

    nld_nao_dir = os.path.join(nle_data_dir, "nld-nao")
    _xl_by_player = _load_xlogfiles(nld_nao_dir)

    tasks: list[tuple] = []
    for player in sorted(os.listdir(data_root)):
        player_dir = os.path.join(data_root, player)
        if not os.path.isdir(player_dir):
            continue
        bz2_files = [
            os.path.join(player_dir, f)
            for f in os.listdir(player_dir)
            if f.endswith(".bz2")
        ]
        if not bz2_files:
            continue
        tasks.append((
            bz2_files,
            os.path.join(output_dir, f"{player}.npz"),
            1, min_frames, _KEYS_NAO, player,
        ))
    return tasks


# --------------------------------------------------------------------------- #
# --- Index write helpers ---------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _new_rich_accum() -> dict:
    return {
        "pl_paths": [], "pl_lengths": [], "pl_n_games": [],
        "gm_player_id": [], "gm_lengths": [], "gm_scores": [],
        "gm_turns": [], "gm_dlvl": [], "gm_conduct": [],
        "gm_ascended": [], "gm_role": [], "gm_race": [],
        "gm_align": [], "gm_death": [], "gm_timestamps": [], "gm_flags": [],
    }


def _write_index_rich(index_path: str, a: dict) -> None:
    np.savez_compressed(
        index_path,
        format_version=np.int32(1),
        player_paths=np.array(a["pl_paths"],    dtype="U512"),
        player_lengths=np.array(a["pl_lengths"], dtype=np.int32),
        player_n_games=np.array(a["pl_n_games"], dtype=np.int32),
        game_player_id=np.array(a["gm_player_id"], dtype=np.int32),
        game_lengths=np.array(a["gm_lengths"],   dtype=np.int32),
        game_scores=np.array(a["gm_scores"],     dtype=np.int32),
        game_turns=np.array(a["gm_turns"],       dtype=np.int32),
        game_dlvl=np.array(a["gm_dlvl"],         dtype=np.int16),
        game_conduct=np.array(a["gm_conduct"],   dtype=np.int32),
        game_ascended=np.array(a["gm_ascended"], dtype=bool),
        game_role=np.array(a["gm_role"],         dtype="U3"),
        game_race=np.array(a["gm_race"],         dtype="U3"),
        game_align=np.array(a["gm_align"],       dtype="U3"),
        game_death=np.array(a["gm_death"],       dtype="U128"),
        game_timestamps=np.array(a["gm_timestamps"], dtype=np.int64),
        game_flags=np.array(a["gm_flags"],       dtype=np.int32),
    )
    print(
        f"  index: {len(a['pl_paths']):,} players, "
        f"{len(a['gm_lengths']):,} games → {index_path}",
        flush=True,
    )


def _write_index_simple(
    index_path: str, paths: list[str], lengths: list[int], scores: list[int]
) -> None:
    np.savez_compressed(
        index_path,
        paths=np.array(paths, dtype="U512"),
        lengths=np.array(lengths, dtype=np.int32),
        max_scores=np.array(scores, dtype=np.int32),
    )
    scores_arr = np.array(scores, dtype=np.int32)
    print(f"  index written: {index_path}  ({len(paths):,} entries)", flush=True)
    if len(scores_arr):
        print(
            f"  scores: min={scores_arr.min():,}  "
            f"median={int(np.median(scores_arr)):,}  "
            f"p90={int(np.percentile(scores_arr, 90)):,}  "
            f"max={scores_arr.max():,}",
            flush=True,
        )


def _accum_player_result(a: dict, result: dict) -> None:
    """Append one ok/skip result (with game_meta) into the rich accumulator."""
    player_id = len(a["pl_paths"])
    a["pl_paths"].append(result["path"])
    a["pl_lengths"].append(result.get("frames", 0))
    a["pl_n_games"].append(len(result["game_meta"]))
    for gm in result["game_meta"]:
        a["gm_player_id"].append(player_id)
        a["gm_lengths"].append(gm["length"])
        a["gm_scores"].append(gm["score"])
        a["gm_turns"].append(gm["turns"])
        a["gm_dlvl"].append(gm["dlvl"])
        a["gm_conduct"].append(gm["conduct"])
        a["gm_ascended"].append(gm["ascended"])
        a["gm_role"].append(gm["role"])
        a["gm_race"].append(gm["race"])
        a["gm_align"].append(gm["align"])
        a["gm_death"].append(gm["death"])
        a["gm_timestamps"].append(gm["timestamp"])
        a["gm_flags"].append(gm["flags"])


def _load_rich_accum(index_path: str) -> tuple[dict, set[str]]:
    """Reload a previously written rich index into an accumulator dict."""
    a = _new_rich_accum()
    indexed: set[str] = set()
    try:
        ex = np.load(index_path)
        if "player_paths" not in ex:
            return a, indexed
        a["pl_paths"]    = [str(p) for p in ex["player_paths"]]
        a["pl_lengths"]  = list(ex["player_lengths"].astype(int))
        a["pl_n_games"]  = list(ex["player_n_games"].astype(int))
        a["gm_player_id"] = list(ex["game_player_id"].astype(int))
        a["gm_lengths"]  = list(ex["game_lengths"].astype(int))
        a["gm_scores"]   = list(ex["game_scores"].astype(int))
        a["gm_turns"]    = list(ex["game_turns"].astype(int))
        a["gm_dlvl"]     = list(ex["game_dlvl"].astype(int))
        a["gm_conduct"]  = list(ex["game_conduct"].astype(int))
        a["gm_ascended"] = list(ex["game_ascended"].astype(bool))
        a["gm_role"]     = [str(r) for r in ex["game_role"]]
        a["gm_race"]     = [str(r) for r in ex["game_race"]]
        a["gm_align"]    = [str(r) for r in ex["game_align"]]
        a["gm_death"]    = [str(d) for d in ex["game_death"]]
        a["gm_timestamps"] = list(ex["game_timestamps"].astype(int))
        a["gm_flags"]    = list(ex["game_flags"].astype(int))
        indexed = set(a["pl_paths"])
        print(f"  resuming: {len(indexed):,} players already indexed", flush=True)
    except Exception as exc:
        print(f"  warning: could not reload existing index ({exc}), starting fresh", flush=True)
        a = _new_rich_accum()
    return a, indexed


# --------------------------------------------------------------------------- #
# --- Conversion runners ----------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _run_convert_simple(
    tasks: list[tuple],
    workers: int,
    npz_dir: str,
) -> tuple[list[str], list[int], list[int]]:
    """Convert nld-aa tasks; return (paths, lengths, scores) for index writing."""
    total = len(tasks)
    print(f"  games found: {total:,}", flush=True)

    index_path = os.path.join(npz_dir, "index.npz")
    if os.path.exists(index_path):
        ex = np.load(index_path)
        ex_paths   = [str(p) for p in ex["paths"]]
        ex_lengths = list(ex["lengths"].astype(int))
        ex_scores  = (list(ex["max_scores"].astype(int)) if "max_scores" in ex
                      else [0] * len(ex_paths))
        existing   = set(ex_paths)
    else:
        ex_paths, ex_lengths, ex_scores, existing = [], [], [], set()

    counts = {"ok": 0, "skip": 0, "filter": 0, "error": 0}
    errors: list[str] = []
    new_entries: list[tuple[str, int, int]] = []

    with mp.Pool(workers) as pool:
        with tqdm(total=total, unit="game", desc="  convert", dynamic_ncols=True) as bar:
            for result in pool.imap_unordered(_convert_one, tasks):
                counts[result["status"]] += 1
                if result["status"] == "ok":
                    p = result["path"]
                    if p not in existing:
                        new_entries.append((p, result["frames"], result["max_score"]))
                elif result["status"] == "error":
                    errors.append(result.get("msg", "unknown"))
                bar.set_postfix(
                    ok=counts["ok"], skip=counts["skip"],
                    filt=counts["filter"], err=counts["error"],
                )
                bar.update(1)

    if errors:
        print(f"\n  first 10 errors:", flush=True)
        for msg in errors[:10]:
            print(f"    {msg}", flush=True)

    all_paths   = ex_paths   + [p for p, _, _ in new_entries]
    all_lengths = ex_lengths + [n for _, n, _ in new_entries]
    all_scores  = ex_scores  + [s for _, _, s in new_entries]
    print(
        f"  convert summary: ok={counts['ok']} skip={counts['skip']} "
        f"filter={counts['filter']} error={counts['error']}",
        flush=True,
    )
    return all_paths, all_lengths, all_scores


def _run_convert_rich(
    tasks: list[tuple],
    workers: int,
    npz_dir: str,
    write_index: bool = True,
    checkpoint_every: int = 500,
) -> None:
    """Convert per-player tasks and progressively write a rich index.npz.

    Resumes from an existing partial index on restart: players already in the
    index are skipped; players whose npz file exists but are not yet indexed
    have their metadata rebuilt from ``source_timestamps`` without re-decoding.
    """
    total = len(tasks)
    print(f"  players found: {total:,}", flush=True)

    index_path = os.path.join(npz_dir, "index.npz")

    # Reload any existing partial checkpoint.
    accum, indexed_paths = _new_rich_accum(), set()
    if write_index and os.path.exists(index_path):
        accum, indexed_paths = _load_rich_accum(index_path)

    # Tasks whose output is already in the index are truly skipped.
    pending = [t for t in tasks if t[1] not in indexed_paths]
    print(f"  pending: {len(pending):,} players to process", flush=True)

    counts = {"ok": 0, "skip": 0, "filter": 0, "error": 0}
    errors: list[str] = []
    since_ckpt = 0

    with mp.Pool(workers) as pool:
        with tqdm(total=len(pending), unit="player", desc="  convert", dynamic_ncols=True) as bar:
            for result in pool.imap_unordered(_convert_player, pending):
                status = result["status"]
                counts[status] += 1

                if status in ("ok", "skip") and result.get("game_meta"):
                    _accum_player_result(accum, result)
                    since_ckpt += 1
                elif status == "error":
                    errors.append(result.get("msg", "unknown"))

                bar.set_postfix(
                    ok=counts["ok"], skip=counts["skip"],
                    filt=counts["filter"], err=counts["error"],
                )
                bar.update(1)

                if write_index and since_ckpt >= checkpoint_every and accum["pl_paths"]:
                    _write_index_rich(index_path, accum)
                    since_ckpt = 0

    if write_index and accum["pl_paths"]:
        _write_index_rich(index_path, accum)

    if errors:
        print(f"\n  first 10 errors:", flush=True)
        for msg in errors[:10]:
            print(f"    {msg}", flush=True)

    print(
        f"\n  convert summary: ok={counts['ok']} skip={counts['skip']} "
        f"filter={counts['filter']} error={counts['error']}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# --- Index scan (--skip-convert case) --------------------------------------- #
# --------------------------------------------------------------------------- #

def _max_score_from_file(path: str) -> tuple[int, int]:
    with np.load(path) as f:
        if "offsets" in f:
            return int(f["offsets"][-1]), 0
        if "scores" in f:
            scores = f["scores"]
            return int(scores.shape[0]), int(scores.max())
        chars = f["tty_chars"]
        n, max_score = int(chars.shape[0]), 0
        for t in range(n):
            m = _SCORE_RE.search(bytes(chars[t, 22]))
            if m:
                s = int(m.group(1))
                if s > max_score:
                    max_score = s
        return n, max_score


def _index_worker_simple(path: str) -> tuple[str, int, int]:
    try:
        n, max_score = _max_score_from_file(path)
        return path, n, max_score
    except Exception:
        return path, -1, -1


def _index_worker_rich(player_path: str) -> dict:
    """Read one per-player npz; rebuild per-game metadata from source_timestamps."""
    try:
        with np.load(player_path) as f:
            offsets = f["offsets"]
            src_ts = f["source_timestamps"] if "source_timestamps" in f else None
    except Exception as exc:
        return {"error": str(exc), "path": player_path}

    player_name = os.path.splitext(os.path.basename(player_path))[0]
    xl_entries = _xl_by_player.get(player_name, [])
    n_games = len(offsets) - 1
    game_meta: list[dict] = []

    for i in range(n_games):
        n_frames = int(offsets[i + 1]) - int(offsets[i])
        if src_ts is not None and i < len(src_ts):
            ts = int(src_ts[i])
            entry = _match_xlog_entry(xl_entries, ts) if xl_entries else {}
        else:
            ts, entry = 0, {}
        game_meta.append(_game_meta_from_xlog(entry, n_frames, ts))

    return {
        "path": player_path,
        "frames": int(offsets[-1]),
        "games": n_games,
        "game_meta": game_meta,
    }


def _build_index_from_scan(scan_dir: str, workers: int) -> tuple[list[str], list[int], list[int]]:
    """Scan a directory for .npz files; return (paths, lengths, scores) for simple index."""
    npz_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(scan_dir)
        for f in files
        if f.endswith(".npz") and f != "index.npz"
    ]
    total = len(npz_files)
    print(f"  scanning {total:,} files in {scan_dir} ...", flush=True)

    good_paths, good_lengths, good_scores = [], [], []
    errors = 0
    with mp.Pool(workers) as pool:
        with tqdm(total=total, unit="file", desc="  index", dynamic_ncols=True) as bar:
            for path, n, score in pool.imap_unordered(_index_worker_simple, npz_files):
                if n >= 0:
                    good_paths.append(path)
                    good_lengths.append(n)
                    good_scores.append(score)
                else:
                    errors += 1
                bar.set_postfix(ok=len(good_paths), err=errors)
                bar.update(1)
    return good_paths, good_lengths, good_scores


def _build_rich_index_from_scan(
    scan_dir: str, workers: int, index_path: str, nle_data_dir: str
) -> None:
    """Scan nld-nao-npz dir; rebuild rich index using source_timestamps + xlogfile."""
    global _xl_by_player

    npz_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(scan_dir)
        for f in files
        if f.endswith(".npz") and f != "index.npz"
    ]
    total = len(npz_files)
    print(f"  scanning {total:,} player files in {scan_dir} ...", flush=True)

    if not _xl_by_player:
        nld_nao_dir = os.path.join(nle_data_dir, "nld-nao")
        _xl_by_player = _load_xlogfiles(nld_nao_dir)

    accum = _new_rich_accum()
    errors = 0

    with mp.Pool(workers) as pool:
        with tqdm(total=total, unit="player", desc="  index", dynamic_ncols=True) as bar:
            for result in pool.imap_unordered(_index_worker_rich, npz_files):
                if "error" in result:
                    errors += 1
                else:
                    _accum_player_result(accum, result)
                bar.set_postfix(ok=len(accum["pl_paths"]), err=errors)
                bar.update(1)

    if accum["pl_paths"]:
        _write_index_rich(index_path, accum)


# --------------------------------------------------------------------------- #
# --- Dataset pipelines ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

def _nld_aa_zips() -> list[str]:
    return [f"nld-aa-dir-a{c}.zip" for c in "abcdefghijklmnop"]


def _nld_nao_zips() -> list[str]:
    suffixes = [f"a{c}" for c in "abcdefghijklmnopqrstuvwxyz"] + \
               [f"b{c}" for c in "abcdefghijklmn"]
    zips = [f"nld-nao-dir-{s}.zip" for s in suffixes]
    zips.append("nld-nao_xlogfiles.zip")
    return zips


def _run_nao_top10(args: BaseArgs) -> None:
    root        = args.output_dir
    zip_dir     = os.path.join(root, "zips", "nao-top10")
    tar_path    = os.path.join(zip_dir, "nao_top10.tar")
    extract_dir = os.path.join(root, "nao-top10")
    npz_dir     = os.path.join(extract_dir, "nao_top10")
    index_path  = os.path.join(npz_dir, "index.npz")

    print("\n─── nao-top10 ───────────────────────────────────────────────────")

    if not args.skip_download:
        os.makedirs(zip_dir, exist_ok=True)
        print(f"[download] nao_top10.tar (~11.8 GB) → {tar_path}")
        _download(_NAO_TOP10_URL, tar_path)
    else:
        print("[download] skipped")

    if not args.skip_extract:
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[extract]  → {extract_dir}")
        _extract_tar(tar_path, extract_dir)
        if not args.keep_archives and os.path.exists(tar_path):
            os.remove(tar_path)
            try:
                os.rmdir(zip_dir)
            except OSError:
                pass
    else:
        print("[extract]  skipped")

    print("[db]       n/a for nao-top10")
    print("[convert]  n/a for nao-top10")

    if not args.skip_index:
        os.makedirs(npz_dir, exist_ok=True)
        print(f"[index]    → {index_path}")
        paths, lengths, scores = _build_index_from_scan(npz_dir, args.workers)
        if paths:
            _write_index_simple(index_path, paths, lengths, scores)
    else:
        print("[index]    skipped")

    print(f"\nDone. Set in your experiment config:")
    print(f"  data.index_path: {index_path}")


def _run_nld(dataset: str, args: BaseArgs) -> None:
    assert dataset in ("nld-aa", "nld-nao")
    root        = args.output_dir
    zip_dir     = os.path.join(root, "zips", dataset)
    extract_dir = os.path.join(root, dataset)
    db_path     = os.path.join(root, f"{dataset}.db")
    npz_dir     = os.path.join(root, f"{dataset}-npz")
    index_path  = os.path.join(npz_dir, "index.npz")

    filenames  = _nld_aa_zips() if dataset == "nld-aa" else _nld_nao_zips()
    base_url   = _NLD_AA_BASE   if dataset == "nld-aa" else _NLD_NAO_BASE
    use_altorg = dataset == "nld-nao"

    print(f"\n─── {dataset} ───────────────────────────────────────────────────")

    if not args.skip_download:
        os.makedirs(zip_dir, exist_ok=True)
        print(f"[download] {len(filenames)} archives → {zip_dir}")
        _parallel_download(base_url, filenames, zip_dir, args.workers)
    else:
        print("[download] skipped")

    if not args.skip_extract:
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[extract]  → {extract_dir}")
        _extract_zips(filenames, zip_dir, extract_dir)
        if not args.keep_archives:
            _remove_archives(filenames, zip_dir)
    else:
        print("[extract]  skipped")

    if not args.skip_db:
        print(f"[db]       → {db_path}")
        _build_nle_db(extract_dir, db_path, dataset, use_altorg)
    else:
        print("[db]       skipped")

    if not args.skip_convert:
        os.makedirs(npz_dir, exist_ok=True)
        print(f"[convert]  → {npz_dir}")
        if dataset == "nld-aa":
            tasks = _discover_nld_aa(root, npz_dir, args.nld_aa_subdir)
            paths, lengths, scores = _run_convert_simple(tasks, args.workers, npz_dir)
            if not args.skip_index and paths:
                print(f"[index]    → {index_path}")
                _write_index_simple(index_path, paths, lengths, scores)
            elif args.skip_index:
                print("[index]    skipped")
        else:
            tasks = _discover_nld_nao(root, npz_dir, args.min_frames)
            print(f"[index]    progressive → {index_path}")
            _run_convert_rich(
                tasks, args.workers, npz_dir,
                write_index=not args.skip_index,
            )
    else:
        print("[convert]  skipped")
        if not args.skip_index:
            os.makedirs(npz_dir, exist_ok=True)
            print(f"[index]    → {index_path}")
            if dataset == "nld-aa":
                paths, lengths, scores = _build_index_from_scan(npz_dir, args.workers)
                if paths:
                    _write_index_simple(index_path, paths, lengths, scores)
            else:
                _build_rich_index_from_scan(npz_dir, args.workers, index_path, root)
        else:
            print("[index]    skipped")

    print(f"\nDone. Set in your experiment config:")
    print(f"  data.nle_data_dir: {root}")
    print(f"  data.index_path:   {index_path}")


# --------------------------------------------------------------------------- #
# --- Entry point ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

def main() -> None:
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    cfg = tyro.cli(
        Union[
            Annotated[NaoTop10Args, tyro.conf.subcommand("nao-top10")],
            Annotated[NldAaArgs,    tyro.conf.subcommand("nld-aa")],
            Annotated[NldNaoArgs,   tyro.conf.subcommand("nld-nao")],
            Annotated[AllArgs,      tyro.conf.subcommand("all")],
        ]
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    if isinstance(cfg, NaoTop10Args):
        _run_nao_top10(cfg)
    elif isinstance(cfg, NldAaArgs):
        _run_nld("nld-aa", cfg)
    elif isinstance(cfg, NldNaoArgs):
        _run_nld("nld-nao", cfg)
    elif isinstance(cfg, AllArgs):
        _run_nld("nld-aa",  cfg)
        _run_nld("nld-nao", cfg)
        _run_nao_top10(cfg)


if __name__ == "__main__":
    main()
