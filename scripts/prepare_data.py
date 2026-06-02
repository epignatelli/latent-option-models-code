"""Prepare NLE datasets for LOM pre-training.

Pipeline stages (run in order; each is individually skippable):

  download   fetch archives from remote storage
  extract    unzip / untar archives
  db         build NLE SQLite database (nld-aa / nld-nao only)
  convert    decode ttyrec / copy session → one .npz per game
  index      scan output directory and write index.npz (always written after convert)

Datasets:

  nao-top10  NAO Top-10, DeepMind processed .npz, ~12 GB
             stages: download → extract → convert → index
  nld-aa     NLD-AA (Autoascend AI), 16 zip archives, ~100 GB
             stages: download → extract → db → convert → index
  nld-nao    NLD-NAO (NetHack.alt.org), 41 zip archives, ~500 GB
             stages: download → extract → db → convert → index
  all        run all three datasets in sequence

Output layout under --output-dir:

  nao-top10/nao_top10/                  extracted source npz sessions (by player)
  nle/nao-top10/<player>/<session>.npz  one npz per game
  nle/nao-top10/index.npz               flat index (game paths + lengths + metadata)
  nld-aa/                               extracted ttyrec files
  nld-aa.db                             NLE SQLite database
  nle/aa/<group>/<game_id>.npz          one npz per game
  nle/aa/index.npz
  nld-nao/                              extracted ttyrec files
  nld-nao.db
  nle/nao/<player>/<timestamp>.npz      one npz per game
  nle/nao/index.npz
  zips/                                 downloaded archives (removed unless --keep-archives)

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
import multiprocessing as mp
import os
import re
import signal
import sys
import tarfile
import threading
import time
import traceback

import psutil
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

ROWS, COLS = 24, 80

_HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")

_NLD_AA_BASE   = "https://dl.fbaipublicfiles.com/nld/nld-aa/"
_NLD_NAO_BASE  = "https://dl.fbaipublicfiles.com/nld/nld-nao/"
_NAO_TOP10_URL = "https://storage.googleapis.com/dm_nethack/nao_top10.tar"

_xl_by_player: dict[str, list[dict]] = {}
_xl_aa: dict[str, dict[str, dict]] = {}  # group_name -> {bz2_basename -> xlog_entry}
_DEBUG_LOG_PATH: str = ""
_POOL_CONVERTER = None  # set before Pool creation; inherited via fork


def _dbg(msg: str) -> None:
    if _DEBUG_LOG_PATH:
        try:
            with open(_DEBUG_LOG_PATH, "a") as _f:
                _f.write(f"[{time.strftime('%H:%M:%S')}][{os.getpid()}] {msg}\n")
        except OSError:
            pass


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


@dataclass
class BaseArgs:
    output_dir: str = "nle_data"
    """Root directory for npz outputs and index."""
    raw_dir: str = ""
    """Directory for downloads and extraction. Defaults to output_dir if empty."""
    workers: int = 4
    """Parallel workers for download and conversion."""
    keep_archives: bool = False
    """Keep zip / tar archives after extraction."""
    min_frames: int = 50
    """Minimum decoded frames to retain a game."""
    nld_aa_subdir: str = "nle_data"
    """Sub-directory inside nld-aa/ holding per-game ttyrec dirs."""
    skip_download: bool = False
    """Skip the download stage."""
    skip_extract: bool = False
    """Skip the extract stage."""
    skip_db: bool = False
    """Skip building the NLE SQLite database (nld-aa / nld-nao only)."""
    skip_convert: bool = False
    """Skip conversion; rebuild index from existing npz files instead."""
    max_groups: int = 0
    """Maximum number of games to convert. 0 = no limit."""
    log_dir: str = "logs"
    """Directory for debug.log."""


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


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    tmp = dest + ".tmp"
    name = os.path.basename(dest)
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total or None, unit="B", unit_scale=True,
                      desc=f"  {name}", file=sys.stdout, dynamic_ncols=True, smoothing=0) as bar:
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        bar.update(len(chunk))
        os.rename(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _parallel_download(base_url: str, filenames: list[str], dest_dir: str, workers: int) -> None:
    pending = [n for n in filenames if not os.path.exists(os.path.join(dest_dir, n))]
    if not pending:
        print(f"  all {len(filenames)} archives already present — skipping download.", flush=True)
        return
    print(f"  downloading {len(pending)}/{len(filenames)} archives ({workers} workers) ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download, base_url + name, os.path.join(dest_dir, name)): name
            for name in pending
        }
        with tqdm(total=len(futures), unit="file", file=sys.stdout, smoothing=0) as bar:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to download {name}: {exc}") from exc
                bar.set_postfix(file=name)
                bar.update(1)


def _extract_zips(filenames: list[str], zip_dir: str, dest_dir: str, workers: int = 1) -> None:
    _done = os.path.join(dest_dir, ".done")
    if os.path.exists(_done):
        print(f"  already extracted to {dest_dir} — skipping.", flush=True)
        return
    print(f"  extracting {len(filenames)} archives to {dest_dir} ({workers} workers)...", flush=True)
    tasks = [(os.path.join(zip_dir, name), dest_dir) for name in filenames]
    print("  scanning zip manifests...", flush=True)
    for name in filenames:
        with zipfile.ZipFile(os.path.join(zip_dir, name), "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    os.makedirs(os.path.join(dest_dir, member.filename), exist_ok=True)
    with tqdm(total=len(filenames), unit="zip", file=sys.stdout, smoothing=0) as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def _do_zip(t: tuple) -> None:
                with zipfile.ZipFile(t[0], "r") as zf:
                    zf.extractall(t[1])
            futures = {pool.submit(_do_zip, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                bar.set_postfix_str(os.path.basename(futures[fut]))
                fut.result()
                bar.update(1)
    open(_done, "w").close()


def _extract_tar(tar_path: str, dest_dir: str) -> None:
    _done = os.path.join(dest_dir, ".done")
    if os.path.exists(_done):
        print(f"  already extracted to {dest_dir} — skipping.", flush=True)
        return
    print(f"  extracting {tar_path} to {dest_dir} ...", flush=True)
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        total = sum(m.size for m in members)
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(tar_path),
                  file=sys.stdout, smoothing=0) as bar:
            for member in members:
                tf.extract(member, dest_dir)
                bar.update(member.size)
    open(_done, "w").close()


# ---------------------------------------------------------------------------
# NLE database
# ---------------------------------------------------------------------------

def _build_nle_db(unzipped_dir: str, db_path: str, dataset: str) -> None:
    if os.path.exists(db_path):
        print(f"  NLE database already exists at {db_path} — skipping.", flush=True)
        return
    print(f"  building NLE database at {db_path} ...", flush=True)
    print(f"  scanning {unzipped_dir} (this can take several minutes) ...", flush=True)
    try:
        import nle.dataset as nld
        import nle.dataset.db as nld_db
    except ImportError:
        raise ImportError(
            "NLE is required for DB build.\n"
            "  pip install git+https://github.com/NetHack-LE/nle.git@main"
        )
    t0 = time.time()
    nld_db.create(filename=db_path)
    print(f"  [{time.strftime('%H:%M:%S')}] DB created, populating rows ...", flush=True)
    if dataset == "nld-nao":
        nld.add_altorg_directory(unzipped_dir, dataset, filename=db_path)
    else:
        nld.add_nledata_directory(unzipped_dir, dataset, filename=db_path)
    print(f"  [{time.strftime('%H:%M:%S')}] DB done in {(time.time()-t0)/60:.1f} min → {db_path}", flush=True)


# ---------------------------------------------------------------------------
# xlogfile helpers
# ---------------------------------------------------------------------------

def _parse_xlog_line(line: str) -> dict[str, str]:
    sep = "\t" if "\t" in line else ":"
    result: dict[str, str] = {}
    for part in line.strip().split(sep):
        if "=" in part:
            k, _, v = part.partition("=")
            result[k] = v
    return result


def _load_xlogfiles(nld_nao_dir: str) -> dict[str, list[dict]]:
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
    stem = os.path.basename(bz2_path).replace(".ttyrec.bz2", "")
    try:
        return int(datetime.strptime(stem, "%Y-%m-%d.%H:%M:%S")
                   .replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return 0


def _match_xlog_entry(entries: list[dict], file_ts: int) -> dict:
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
        "length":    n_frames,
        "score":     int(entry.get("points",  0) or 0),
        "turns":     int(entry.get("turns",  -1) or -1),
        "dlvl":      int(entry.get("maxlvl", -1) or -1),
        "conduct":   _hex_or_int(entry.get("conduct", "0")),
        "ascended":  death.lower().startswith("ascended"),
        "role":      (entry.get("role",  "???") or "???")[:3],
        "race":      (entry.get("race",  "???") or "???")[:3],
        "align":     (entry.get("align", "???") or "???")[:3],
        "death":     death[:128],
        "flags":     _hex_or_int(entry.get("flags", "0")),
        "timestamp": file_ts,
    }


# ---------------------------------------------------------------------------
# ttyrec decoder
# ---------------------------------------------------------------------------

def _decode(ttyrec_files: list[str], ttyrec_version: int) -> tuple[dict, int]:
    from nle import _pyconverter as nle_converter  # type: ignore[reportAttributeAccessIssue]

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


def _decode_with_timeout(ttyrec_files: list[str], ttyrec_version: int) -> tuple[dict, int]:
    """Run _decode in a daemon thread; raises TimeoutError if NLE hangs past 300s."""
    result_box: list = [None]
    error_box:  list = [None]
    done = threading.Event()

    def _run() -> None:
        try:
            result_box[0] = _decode(ttyrec_files, ttyrec_version)
        except Exception as exc:
            error_box[0] = exc
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not done.wait(300):
        raise TimeoutError(f"_decode timed out after 300s on {ttyrec_files[0]}")
    if error_box[0] is not None:
        raise error_box[0]
    return result_box[0]


# ---------------------------------------------------------------------------
# xlogfile helpers for nld-aa
# ---------------------------------------------------------------------------

def _read_aa_xlogfile(game_dir: str) -> dict[str, dict]:
    """Return mapping ttyrecname → xlogfile entry for all games in game_dir."""
    _dbg(f"XLOG_LISTDIR {game_dir}")
    for fname in os.listdir(game_dir):
        if fname.endswith(".xlogfile"):
            result: dict[str, dict] = {}
            with open(os.path.join(game_dir, fname), "r", errors="replace") as fh:
                for line in fh:
                    entry = _parse_xlog_line(line)
                    key = entry.get("ttyrecname", "")
                    if key:
                        result[key] = entry
            return result
    return {}


# ---------------------------------------------------------------------------
# Per-game converters — one task = one game file
# ---------------------------------------------------------------------------

def _convert_nao_top10_game(task: tuple) -> list[dict]:
    """Copy one nao-top10 session npz → per-game npz, filtering by min_frames."""
    session_path, output_path, min_frames = task
    if os.path.exists(output_path):
        try:
            with np.load(output_path) as f:
                n = f["tty_chars"].shape[0]
            return [{"status": "skip", "path": output_path, "frames": n,
                     "game_meta": [dict(_GAME_META_DEFAULT, length=n)]}]
        except Exception as exc:
            _dbg(f"SKIP_LOAD_ERROR {output_path}: {exc}")

    try:
        with np.load(session_path) as f:
            chars  = f["tty_chars"].astype(np.uint8)
            colors = (np.clip(f["tty_colors"].astype(np.int16), 0, 31).astype(np.uint8)
                      if "tty_colors" in f else np.zeros_like(chars))
    except Exception as exc:
        return [{"status": "error", "path": output_path, "error": str(exc)}]

    n = chars.shape[0]
    if n < min_frames:
        return [{"status": "filter", "filtered_games": 1}]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, tty_chars=chars, tty_colors=colors)
    return [{"status": "ok", "path": output_path, "frames": n,
             "game_meta": [dict(_GAME_META_DEFAULT, length=n)]}]


def _convert_nld_game(task: tuple) -> list[dict]:
    """Decode one ttyrec bz2 → per-game npz (nld-nao)."""
    bz2_path, output_path, ttyrec_version, min_frames, player_name = task
    if os.path.exists(output_path):
        try:
            with np.load(output_path) as f:
                n = f["tty_chars"].shape[0]
            ts    = _parse_filename_ts(bz2_path)
            entry = _match_xlog_entry(_xl_by_player.get(player_name, []), ts)
            return [{"status": "skip", "path": output_path, "frames": n,
                     "game_meta": [_game_meta_from_xlog(entry, n, ts)]}]
        except Exception as exc:
            _dbg(f"SKIP_LOAD_ERROR {output_path}: {exc}")

    try:
        arrays, n = _decode_with_timeout([bz2_path], ttyrec_version)
    except Exception as exc:
        return [{"status": "error", "path": output_path, "error": str(exc)}]

    if not arrays or n < min_frames:
        return [{"status": "filter", "filtered_games": 1}]

    ts    = _parse_filename_ts(bz2_path)
    entry = _match_xlog_entry(_xl_by_player.get(player_name, []), ts)
    chars  = arrays["tty_chars"].astype(np.uint8)
    colors = (arrays["tty_colors"].astype(np.int16).clip(0, 31).astype(np.uint8)
              if "tty_colors" in arrays else np.zeros_like(chars))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, tty_chars=chars, tty_colors=colors)
    return [{"status": "ok", "path": output_path, "frames": n,
             "game_meta": [_game_meta_from_xlog(entry, n, ts)]}]


def _convert_aa_game(task: tuple) -> list[dict]:
    """Decode one ttyrec bz2 → per-game npz (nld-aa)."""
    bz2_path, output_path, ttyrec_version, min_frames, group_name = task
    bz2_basename = os.path.basename(bz2_path)

    if os.path.exists(output_path):
        try:
            with np.load(output_path) as f:
                n = f["tty_chars"].shape[0]
            entry = _xl_aa.get(group_name, {}).get(bz2_basename, {})
            ts    = int(entry.get("starttime", 0) or 0)
            return [{"status": "skip", "path": output_path, "frames": n,
                     "game_meta": [_game_meta_from_xlog(entry, n, ts)]}]
        except Exception as exc:
            _dbg(f"SKIP_LOAD_ERROR {output_path}: {exc}")

    try:
        arrays, n = _decode_with_timeout([bz2_path], ttyrec_version)
    except Exception as exc:
        return [{"status": "error", "path": output_path, "error": str(exc)}]

    if not arrays or n < min_frames:
        return [{"status": "filter", "filtered_games": 1}]

    entry  = _xl_aa.get(group_name, {}).get(bz2_basename, {})
    ts     = int(entry.get("starttime", 0) or 0)
    chars  = arrays["tty_chars"].astype(np.uint8)
    colors = (arrays["tty_colors"].astype(np.int16).clip(0, 31).astype(np.uint8)
              if "tty_colors" in arrays else np.zeros_like(chars))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, tty_chars=chars, tty_colors=colors)
    return [{"status": "ok", "path": output_path, "frames": n,
             "game_meta": [_game_meta_from_xlog(entry, n, ts)]}]


# ---------------------------------------------------------------------------
# Discovery functions — one task per game
# ---------------------------------------------------------------------------

def _discover_nao_top10_games(extract_dir: str, output_dir: str, min_frames: int) -> list[tuple]:
    """One task per session npz: (session_path, output_path, min_frames)."""
    src_dir = os.path.join(extract_dir, "nao_top10")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"nao-top10 data not found at {src_dir}")
    tasks: list[tuple] = []
    for player in sorted(os.listdir(src_dir)):
        player_dir = os.path.join(src_dir, player)
        if not os.path.isdir(player_dir):
            continue
        for fname in sorted(os.listdir(player_dir)):
            if not fname.endswith(".npz"):
                continue
            tasks.append((
                os.path.join(player_dir, fname),
                os.path.join(output_dir, player, fname),
                min_frames,
            ))
    return tasks


def _discover_nld_nao_games(nle_data_dir: str, output_dir: str, min_frames: int) -> list[tuple]:
    """One task per bz2 file: (bz2_path, output_path, version, min_frames, player_name).
    Also loads xlogfiles into _xl_by_player before returning (must run before Pool)."""
    global _xl_by_player

    data_root = os.path.join(nle_data_dir, "nld-nao", "nld-nao-unzipped")
    if not os.path.isdir(data_root):
        data_root = os.path.join(nle_data_dir, "nld-nao")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"nld-nao data not found at {data_root}")

    _xl_by_player = _load_xlogfiles(os.path.join(nle_data_dir, "nld-nao"))

    tasks: list[tuple] = []
    for player in sorted(os.listdir(data_root)):
        player_dir = os.path.join(data_root, player)
        if not os.path.isdir(player_dir):
            continue
        for fname in sorted(os.listdir(player_dir)):
            if not fname.endswith(".bz2"):
                continue
            stem = os.path.splitext(os.path.splitext(fname)[0])[0]
            tasks.append((
                os.path.join(player_dir, fname),
                os.path.join(output_dir, player, stem + ".npz"),
                1,  # ttyrec_version for nld-nao
                min_frames,
                player,
            ))
    return tasks


def _discover_nld_aa_games(nle_data_dir: str, output_dir: str, min_frames: int) -> list[tuple]:
    """One task per bz2 file: (bz2_path, output_path, version, min_frames, group_name).
    Also pre-loads all aa xlogfiles into _xl_aa before returning (must run before Pool)."""
    global _xl_aa

    data_root = os.path.join(nle_data_dir, "nld-aa", "nle_data")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"nld-aa data not found at {data_root}\n"
            "Pass --nld-aa-subdir to point at the right sub-directory."
        )

    tasks: list[tuple] = []
    for group in sorted(os.listdir(data_root)):
        group_dir = os.path.join(data_root, group)
        if not os.path.isdir(group_dir):
            continue
        _xl_aa[group] = _read_aa_xlogfile(group_dir)
        for fname in sorted(os.listdir(group_dir)):
            if not fname.endswith(".bz2"):
                continue
            stem = os.path.splitext(os.path.splitext(fname)[0])[0]
            tasks.append((
                os.path.join(group_dir, fname),
                os.path.join(output_dir, group, stem + ".npz"),
                3,  # ttyrec_version for nld-aa
                min_frames,
                group,
            ))
    return tasks


# ---------------------------------------------------------------------------
# Flat game index (v2)
# ---------------------------------------------------------------------------

def _new_game_accum() -> dict:
    return {
        "gm_paths": [], "gm_lengths": [], "gm_scores": [],
        "gm_turns": [], "gm_dlvl": [], "gm_conduct": [],
        "gm_ascended": [], "gm_role": [], "gm_race": [],
        "gm_align": [], "gm_death": [], "gm_timestamps": [], "gm_flags": [],
    }


def _accum_game_result(a: dict, result: dict) -> None:
    gm = result["game_meta"][0]
    a["gm_paths"].append(result["path"])
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


def _write_game_index(index_path: str, a: dict) -> None:
    np.savez_compressed(
        index_path,
        format_version=np.int32(2),
        game_paths=np.array(a["gm_paths"],      dtype="U512"),
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
    print(f"  index: {len(a['gm_paths']):,} games → {index_path}", flush=True)


def _load_game_accum(index_path: str) -> tuple[dict, set[str]]:
    a = _new_game_accum()
    indexed: set[str] = set()
    try:
        ex = np.load(index_path)
        if "game_paths" not in ex:
            return a, indexed
        a["gm_paths"]      = list(ex["game_paths"].astype(str))
        a["gm_lengths"]    = list(ex["game_lengths"].astype(int))
        a["gm_scores"]     = list(ex["game_scores"].astype(int))
        a["gm_turns"]      = list(ex["game_turns"].astype(int))
        a["gm_dlvl"]       = list(ex["game_dlvl"].astype(int))
        a["gm_conduct"]    = list(ex["game_conduct"].astype(int))
        a["gm_ascended"]   = list(ex["game_ascended"].astype(bool))
        a["gm_role"]       = [str(r) for r in ex["game_role"]]
        a["gm_race"]       = [str(r) for r in ex["game_race"]]
        a["gm_align"]      = [str(r) for r in ex["game_align"]]
        a["gm_death"]      = [str(d) for d in ex["game_death"]]
        a["gm_timestamps"] = list(ex["game_timestamps"].astype(int))
        a["gm_flags"]      = list(ex["game_flags"].astype(int))
        indexed = set(a["gm_paths"])
        print(f"  resuming: {len(indexed):,} games already indexed", flush=True)
    except Exception as exc:
        print(f"  warning: could not reload existing index ({exc}), starting fresh", flush=True)
        a = _new_game_accum()
    return a, indexed


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

def _convert_wrapper(task: tuple) -> tuple:
    """Module-level wrapper so imap_unordered can pickle it.

    Returns (name, n_files, elapsed_s, result_list, exc_str_or_None).
    """
    t0 = time.monotonic()
    name = os.path.splitext(os.path.basename(task[1]))[0]  # task[1] = output_path
    _dbg(f"WRAPPER_START {name}")
    try:
        results = _POOL_CONVERTER(task)  # type: ignore[operator]
        _dbg(f"WRAPPER_END {name} elapsed={time.monotonic()-t0:.1f}s")
        return name, 1, time.monotonic() - t0, results, None
    except Exception:
        tb = traceback.format_exc()
        _dbg(f"WRAPPER_EXCEPTION {name}: {tb}")
        return name, 1, time.monotonic() - t0, [], tb


def _setup_signal_handlers(counts: dict, t0: float) -> None:
    def _handler(signum: int, frame: object) -> None:
        sig_name = {signal.SIGTERM: "SIGTERM", signal.SIGXCPU: "SIGXCPU"}.get(signum, str(signum))
        elapsed  = time.time() - t0
        ram_gb   = psutil.virtual_memory().used / 1024 ** 3
        msg = (
            f"[{time.strftime('%H:%M:%S')}] SIGNAL {sig_name} received  "
            f"elapsed={elapsed:.0f}s  ram={ram_gb:.1f}GB\n"
            f"  counts={counts}\n"
        )
        if _DEBUG_LOG_PATH:
            try:
                with open(_DEBUG_LOG_PATH, "a") as _f:
                    _f.write(msg)
            except OSError:
                pass
        sys.stderr.write(msg)
        sys.stderr.flush()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _handler)
    try:
        signal.signal(signal.SIGXCPU, _handler)
    except (OSError, ValueError, AttributeError):
        pass


def _run_convert_rich(
    tasks: list[tuple],
    workers: int,
    npz_dir: str,
    converter=_convert_nld_game,
    checkpoint_every: int = 500,
    max_groups: int = 0,
) -> None:
    """Convert per-game tasks with mp.Pool.imap_unordered and progressive indexing."""
    global _POOL_CONVERTER
    _POOL_CONVERTER = converter

    n_games = len(tasks)
    print(f"  games found: {n_games:,}", flush=True)

    index_path = os.path.join(npz_dir, "index.npz")
    accum, indexed_paths = _new_game_accum(), set()
    if os.path.exists(index_path):
        accum, indexed_paths = _load_game_accum(index_path)

    pending = [t for t in tasks if t[1] not in indexed_paths]
    if max_groups > 0:
        pending = pending[:max_groups]
    print(f"  pending: {len(pending):,} games", flush=True)
    if not pending:
        return

    try:
        with open("/proc/meminfo") as _f:
            meminfo = {l.split(":")[0].strip(): int(l.split()[1])
                       for l in _f if ":" in l and l.split()[-1] == "kB"}
        limit_gb    = meminfo.get("CommitLimit", 0) / 1024 ** 2
        committed_gb = meminfo.get("Committed_AS", 0) / 1024 ** 2
        print(f"  commit budget: {limit_gb:.1f} GB limit, {committed_gb:.1f} GB used, "
              f"{limit_gb - committed_gb:.1f} GB free", flush=True)
    except Exception:
        pass

    counts = {"ok": 0, "skip": 0, "filter": 0, "error": 0}
    filtered_games_total = 0
    errors: list[str] = []
    error_paths: list[str] = []
    since_ckpt = 0
    _t0 = time.time()
    _last_log = _t0

    n_workers = min(workers, len(pending))
    _dbg(f"POOL_START n_workers={n_workers} pending={len(pending)}")

    _hb_stop = threading.Event()

    def _heartbeat() -> None:
        while not _hb_stop.wait(10.0):
            ram_gb = psutil.virtual_memory().used / 1024 ** 3
            _dbg(
                f"HEARTBEAT ok={counts['ok']} skip={counts['skip']} "
                f"err={counts['error']} filter={counts['filter']} "
                f"ram={ram_gb:.1f}GB"
            )

    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    _setup_signal_handlers(counts, _t0)

    with mp.Pool(processes=n_workers, maxtasksperchild=8) as pool:
        result_iter = pool.imap_unordered(_convert_wrapper, pending)

        with tqdm(total=len(pending), unit="game", desc="  total",
                  ncols=120, smoothing=0, position=0,
                  file=sys.stdout, mininterval=5.0) as bar:

            while True:
                _dbg("MAIN_LOOP_NEXT_RESULT")
                try:
                    raw = next(result_iter)
                except StopIteration:
                    _dbg("MAIN_LOOP_DONE")
                    break
                except Exception as pool_exc:
                    tb = traceback.format_exc()
                    _dbg(f"POOL_EXCEPTION {type(pool_exc).__name__}: {tb}")
                    tqdm.write(
                        f"  [{time.strftime('%H:%M:%S')}] POOL ERROR: {pool_exc}",
                        file=sys.stdout,
                    )
                    counts["error"] += 1
                    continue

                name, _n, elapsed, result_list, exc_str = raw

                if exc_str:
                    _dbg(f"TASK_EXCEPTION {name}:\n{exc_str}")
                    counts["error"] += 1
                    errors.append(exc_str.splitlines()[-1])
                    tqdm.write(
                        f"  [{time.strftime('%H:%M:%S')}] ERR   {name:<48}  {elapsed:.1f}s"
                        f"  {errors[-1]}",
                        file=sys.stdout,
                    )
                    bar.update(1)
                    bar.set_postfix(ok=counts["ok"], skip=counts["skip"],
                                    filt=filtered_games_total, err=counts["error"])
                    continue

                for result in result_list:
                    status = result["status"]
                    counts[status] += 1
                    if status in ("ok", "skip") and result.get("game_meta"):
                        _accum_game_result(accum, result)
                        since_ckpt += 1
                    elif status == "error":
                        errors.append(result.get("error", result.get("msg", "unknown")))
                        if result.get("path"):
                            error_paths.append(result["path"])
                    filtered_games_total += result.get("filtered_games", 0)

                top_status = result_list[0]["status"].upper() if result_list else "UNK"
                tqdm.write(
                    f"  [{time.strftime('%H:%M:%S')}] {top_status:<6}  {name:<48}  {elapsed:>6.1f}s",
                    file=sys.stdout,
                )
                bar.update(1)
                bar.set_postfix(ok=counts["ok"], skip=counts["skip"],
                                filt=filtered_games_total, err=counts["error"])

                now = time.time()
                if now - _last_log >= 60:
                    _last_log = now
                    ram_gb  = psutil.virtual_memory().used  / 1024 ** 3
                    ram_tot = psutil.virtual_memory().total / 1024 ** 3
                    tqdm.write(
                        f"\n  [{time.strftime('%H:%M:%S')}] === "
                        f"{counts['ok']+counts['skip']:,}/{len(pending)}"
                        f"  ok={counts['ok']} skip={counts['skip']} err={counts['error']}"
                        f"  ram={ram_gb:.0f}/{ram_tot:.0f}GB"
                        f"  elapsed={(now - _t0)/60:.1f}min ===\n",
                        file=sys.stdout,
                    )

                if since_ckpt >= checkpoint_every and accum["gm_paths"]:
                    _dbg(f"INDEX_CHECKPOINT games={len(accum['gm_paths'])}")
                    _write_game_index(index_path, accum)
                    since_ckpt = 0

    _hb_stop.set()
    _dbg("POOL_CLOSED writing final index")

    if accum["gm_paths"]:
        _write_game_index(index_path, accum)

    if error_paths:
        retry_path = os.path.join(npz_dir, "errors.txt")
        with open(retry_path, "a") as _f:
            for p in error_paths:
                _f.write(p + "\n")
        print(f"\n  {len(error_paths)} failed game(s) recorded in {retry_path}", flush=True)

    if errors:
        print("\n  first 10 errors:", flush=True)
        for msg in errors[:10]:
            print(f"    {msg}", flush=True)

    print(
        f"\n  convert summary: ok={counts['ok']} skip={counts['skip']} "
        f"filt_games={filtered_games_total} error={counts['error']}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Index rebuild from existing files (used with --skip-convert)
# ---------------------------------------------------------------------------

def _index_game_worker(path: str) -> dict:
    """Read one per-game npz; return its path and frame count."""
    try:
        with np.load(path) as f:
            n = f["tty_chars"].shape[0]
        return {"path": path, "frames": n, "game_meta": [dict(_GAME_META_DEFAULT, length=n)]}
    except Exception as exc:
        return {"error": str(exc), "path": path}


def _build_game_index_from_scan(scan_dir: str, workers: int, index_path: str) -> None:
    """Rebuild index by scanning existing per-game npz files."""
    npz_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(scan_dir)
        for f in files
        if f.endswith(".npz") and f != "index.npz"
    ]
    total = len(npz_files)
    print(f"  scanning {total:,} game files in {scan_dir} ...", flush=True)

    accum = _new_game_accum()
    errors = 0

    with mp.Pool(workers) as pool:
        with tqdm(total=total, unit="game", desc="  index",
                  dynamic_ncols=True, file=sys.stdout, smoothing=0) as bar:
            for result in pool.imap_unordered(_index_game_worker, npz_files):
                if "error" in result:
                    errors += 1
                else:
                    _accum_game_result(accum, result)
                bar.set_postfix(ok=len(accum["gm_paths"]), err=errors)
                bar.update(1)

    if accum["gm_paths"]:
        _write_game_index(index_path, accum)


# ---------------------------------------------------------------------------
# Dataset runners
# ---------------------------------------------------------------------------

_NLD_AA_ZIPS = [f"nld-aa-dir-a{c}.zip" for c in "abcdefghijklmnop"]
_NLD_NAO_ZIPS = (
    [f"nld-nao-dir-a{c}.zip" for c in "abcdefghijklmnopqrstuvwxyz"]
    + [f"nld-nao-dir-b{c}.zip" for c in "abcdefghijklmn"]
    + ["nld-nao_xlogfiles.zip"]
)


def _run_nao_top10(args: BaseArgs) -> None:
    global _DEBUG_LOG_PATH
    raw         = args.raw_dir or args.output_dir
    zip_dir     = os.path.join(raw, "zips", "nao-top10")
    tar_path    = os.path.join(zip_dir, "nao_top10.tar")
    extract_dir = os.path.join(raw, "nao-top10")
    npz_dir     = os.path.join(args.output_dir, "nle", "nao-top10")
    index_path  = os.path.join(npz_dir, "index.npz")

    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _DEBUG_LOG_PATH = os.path.join(args.log_dir, "debug-nao-top10.log")

    print("\n─── nao-top10 ───────────────────────────────────────────────────", flush=True)
    print(f"[log]      debug → {_DEBUG_LOG_PATH}", flush=True)

    if not args.skip_download:
        os.makedirs(zip_dir, exist_ok=True)
        print(f"[download] nao_top10.tar (~11.8 GB) → {tar_path}", flush=True)
        _download(_NAO_TOP10_URL, tar_path)
    else:
        print("[download] skipped", flush=True)

    if not args.skip_extract:
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[extract]  → {extract_dir}", flush=True)
        _extract_tar(tar_path, extract_dir)
        if not args.keep_archives and os.path.exists(tar_path):
            os.remove(tar_path)
            try:
                os.rmdir(zip_dir)
            except OSError:
                pass
    else:
        print("[extract]  skipped", flush=True)

    print("[db]       n/a for nao-top10", flush=True)

    if not args.skip_convert:
        os.makedirs(npz_dir, exist_ok=True)
        print(f"[convert]  → {npz_dir}", flush=True)
        tasks = _discover_nao_top10_games(extract_dir, npz_dir, args.min_frames)
        print(f"[index]    progressive → {index_path}", flush=True)
        _run_convert_rich(
            tasks, args.workers, npz_dir,
            converter=_convert_nao_top10_game,
            max_groups=args.max_groups,
        )
    else:
        print("[convert]  skipped", flush=True)
        os.makedirs(npz_dir, exist_ok=True)
        print(f"[index]    → {index_path}", flush=True)
        _build_game_index_from_scan(npz_dir, args.workers, index_path)

    print("\nDone. Set in your experiment config:", flush=True)
    print(f"  data.dataset_dir: {npz_dir}", flush=True)


def _run_nld(dataset: str, args: BaseArgs) -> None:
    global _DEBUG_LOG_PATH
    assert dataset in ("nld-aa", "nld-nao")
    raw         = args.raw_dir or args.output_dir
    zip_dir     = os.path.join(raw, "zips", dataset)
    extract_dir = os.path.join(raw, dataset)
    db_path     = os.path.join(raw, f"{dataset}.db")
    npz_dir     = os.path.join(args.output_dir, "nle", "aa" if dataset == "nld-aa" else "nao")
    index_path  = os.path.join(npz_dir, "index.npz")

    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _DEBUG_LOG_PATH = os.path.join(args.log_dir, f"debug-{dataset}.log")
    print(f"[log]      debug → {_DEBUG_LOG_PATH}", flush=True)

    filenames = _NLD_AA_ZIPS if dataset == "nld-aa" else _NLD_NAO_ZIPS
    base_url  = _NLD_AA_BASE  if dataset == "nld-aa" else _NLD_NAO_BASE

    print(f"\n─── {dataset} ───────────────────────────────────────────────────", flush=True)

    if not args.skip_download:
        os.makedirs(zip_dir, exist_ok=True)
        print(f"[download] {len(filenames)} archives → {zip_dir}", flush=True)
        _parallel_download(base_url, filenames, zip_dir, args.workers)
    else:
        print("[download] skipped", flush=True)

    if not args.skip_extract:
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[extract]  → {extract_dir}", flush=True)
        _extract_zips(filenames, zip_dir, extract_dir, workers=args.workers)
        if not args.keep_archives:
            for _name in filenames:
                _p = os.path.join(zip_dir, _name)
                if os.path.exists(_p):
                    os.remove(_p)
            try:
                os.rmdir(zip_dir)
            except OSError:
                pass
    else:
        print("[extract]  skipped", flush=True)

    if not args.skip_db:
        print(f"[db]       → {db_path}", flush=True)
        _build_nle_db(extract_dir, db_path, dataset)
    else:
        print("[db]       skipped", flush=True)

    if not args.skip_convert:
        os.makedirs(npz_dir, exist_ok=True)
        print(f"[convert]  → {npz_dir}", flush=True)
        if dataset == "nld-aa":
            tasks    = _discover_nld_aa_games(raw, npz_dir, args.min_frames)
            converter = _convert_aa_game
        else:
            tasks    = _discover_nld_nao_games(raw, npz_dir, args.min_frames)
            converter = _convert_nld_game
        print(f"[index]    progressive → {index_path}", flush=True)
        _run_convert_rich(
            tasks, args.workers, npz_dir,
            converter=converter,
            max_groups=args.max_groups,
        )
    else:
        print("[convert]  skipped", flush=True)
        os.makedirs(npz_dir, exist_ok=True)
        print(f"[index]    → {index_path}", flush=True)
        _build_game_index_from_scan(npz_dir, args.workers, index_path)

    print("\nDone. Set in your experiment config:", flush=True)
    print(f"  data.dataset_dir: {npz_dir}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    cfg = tyro.cli(
        Union[
            Annotated[NaoTop10Args, tyro.conf.subcommand("nao-top10")],
            Annotated[NldAaArgs,    tyro.conf.subcommand("nld-aa")],
            Annotated[NldNaoArgs,   tyro.conf.subcommand("nld-nao")],
            Annotated[AllArgs,      tyro.conf.subcommand("all")],
        ]
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.raw_dir:
        os.makedirs(cfg.raw_dir, exist_ok=True)

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
