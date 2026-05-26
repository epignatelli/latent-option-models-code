"""Prepare NLE datasets for LOM pre-training.

Pipeline stages (run in order; each is individually skippable):

  download   fetch archives from remote storage
  extract    unzip / untar archives
  db         build NLE SQLite database (nld-aa / nld-nao only)
  convert    decode ttyrec → per-game .npz (nld-aa / nld-nao only)
  index      scan output directory and write index.npz

Datasets:

  nao-top10  NAO Top-10, DeepMind processed .npz, ~12 GB
             stages: download → extract → convert → index
  nld-aa     NLD-AA (Autoascend AI), 16 zip archives, ~100 GB
             stages: download → extract → db → convert → index
  nld-nao    NLD-NAO (NetHack.alt.org), 41 zip archives, ~500 GB
             stages: download → extract → db → convert → index
  all        run all three datasets in sequence

Output layout under --output-dir:

  nao-top10/nao_top10/          extracted source npz sessions (by player/session)
  nle/nao-top10/                consolidated per-player .npz files
  nle/nao-top10/index.npz       rich index (game lengths; no xlogfile metadata)
  nld-aa/                       extracted ttyrec files
  nld-aa.db                     NLE SQLite database
  nle/aa/                       converted per-game-dir .npz files (fake players)
  nle/aa/index.npz              rich index with per-game-dir and per-game metadata
  nld-nao/                      extracted ttyrec files
  nld-nao.db
  nle/nao/                      converted per-player .npz files (one file per player)
  nle/nao/index.npz             rich index with per-player and per-game metadata
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
import logging
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROWS, COLS = 24, 80

_HEX_RE   = re.compile(r"^0x[0-9a-fA-F]+$")

_NLD_AA_BASE   = "https://dl.fbaipublicfiles.com/nld/nld-aa/"
_NLD_NAO_BASE  = "https://dl.fbaipublicfiles.com/nld/nld-nao/"
_NAO_TOP10_URL = "https://storage.googleapis.com/dm_nethack/nao_top10.tar"

_xl_by_player: dict[str, list[dict]] = {}
_PROGRESS_QUEUE: "mp.Queue | None" = None
_WORKER_LOG_PATH: str = ""
_DEBUG_LOG_PATH:  str = ""
_DECODE_TIMEOUT_S: int = 300
_POOL_CONVERTER = None  # set in _run_nld before Pool creation; inherited via fork


def _worker_log(msg: str) -> None:
    """Write verbose worker output to the per-run log file (or stdout if unset)."""
    if _WORKER_LOG_PATH:
        try:
            with open(_WORKER_LOG_PATH, "a") as _f:
                _f.write(msg + "\n")
        except OSError:
            pass
    else:
        print(msg, flush=True)


def _dbg(msg: str) -> None:
    """Write one line to the debug log. Called from both main and worker processes."""
    if _DEBUG_LOG_PATH:
        try:
            with open(_DEBUG_LOG_PATH, "a") as _f:
                # Single write → O_APPEND atomicity on Linux; safe for concurrent workers.
                _f.write(f"[{time.strftime('%H:%M:%S')}][{os.getpid()}] {msg}\n")
        except OSError:
            pass


def _worker_progress(
    name: str, done: int, total: int, frames: int,
    ok: int = 0, filter_g: int = 0, err: int = 0, skip: int = 0,
    current_file: str = "",
) -> None:
    """Send a lightweight progress update to the main process bar manager."""
    if _PROGRESS_QUEUE is not None:
        try:
            _PROGRESS_QUEUE.put_nowait(
                (name, done, total, frames, ok, filter_g, err, skip, current_file)
            )
        except Exception:
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

# Maximum frames per nao-top10 chunk npz.  Players with more frames are split
# into multiple files so no single npz is too large to load during training.
_MAX_FRAMES_PER_CHUNK = 2_000_000


# --------------------------------------------------------------------------- #
# --- Config ----------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

@dataclass
class BaseArgs:
    output_dir: str = "nle_data"
    """Root directory for npz outputs and index."""
    raw_dir: str = ""
    """Directory for downloads and extraction. Defaults to output_dir if empty. Set to a fast local path (e.g. /dev/shm) to avoid NFS writes."""
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
    max_groups: int = 0
    """Maximum number of groups (players/game-dirs) to convert. 0 = no limit (process all)."""


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
    name = os.path.basename(dest)
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total or None, unit="B", unit_scale=True,
                      desc=f"  {name}", file=sys.stdout, dynamic_ncols=True) as bar:
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)  # 1 MB
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
        with tqdm(total=len(futures), unit="file", file=sys.stdout) as bar:
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

def _extract_one_zip(args: tuple) -> None:
    zip_path, dest_dir = args
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _extract_zips(filenames: list[str], zip_dir: str, dest_dir: str, workers: int = 1) -> None:
    if _is_done(dest_dir):
        print(f"  already extracted to {dest_dir} — skipping.", flush=True)
        return
    print(f"  extracting {len(filenames)} archives to {dest_dir} ({workers} workers)...", flush=True)
    tasks = [(os.path.join(zip_dir, name), dest_dir) for name in filenames]
    with tqdm(total=len(filenames), unit="zip", file=sys.stdout) as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_one_zip, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                bar.set_postfix_str(os.path.basename(futures[fut]))
                fut.result()
                bar.update(1)
    _mark_done(dest_dir)


def _extract_tar(tar_path: str, dest_dir: str) -> None:
    if _is_done(dest_dir):
        print(f"  already extracted to {dest_dir} — skipping.", flush=True)
        return
    print(f"  extracting {tar_path} to {dest_dir} ...", flush=True)
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        total = sum(m.size for m in members)
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(tar_path), file=sys.stdout) as bar:
            for member in members:
                tf.extract(member, dest_dir)
                bar.update(member.size)
    _mark_done(dest_dir)


# --------------------------------------------------------------------------- #
# --- NLE DB ----------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _build_nle_db(unzipped_dir: str, db_path: str, dataset_name: str, use_altorg: bool) -> None:
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
    if use_altorg:
        nld.add_altorg_directory(unzipped_dir, dataset_name, filename=db_path)
    else:
        nld.add_nledata_directory(unzipped_dir, dataset_name, filename=db_path)
    print(f"  [{time.strftime('%H:%M:%S')}] DB done in {(time.time()-t0)/60:.1f} min → {db_path}", flush=True)


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
    """Run _decode with a SIGALRM watchdog.  Raises TimeoutError if NLE hangs."""
    def _alarm(signum, frame):
        raise TimeoutError(
            f"_decode timed out after {_DECODE_TIMEOUT_S}s on {ttyrec_files[0]}"
        )
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(_DECODE_TIMEOUT_S)
    try:
        return _decode(ttyrec_files, ttyrec_version)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _find_existing_chunks(output_path: str) -> list[str]:
    """Return existing chunk paths for output_path, or [] if none exist."""
    _dbg(f"FIND_CHUNKS {output_path}")
    stem, ext = os.path.splitext(output_path)
    if os.path.exists(output_path):
        return [output_path]
    existing = []
    for i in range(100_000):
        cpath = f"{stem}_{i}{ext}"
        if not os.path.exists(cpath):
            break
        existing.append(cpath)
    return existing


class _ChunkWriter:
    """Accumulates decoded frames and flushes npz chunks at _MAX_FRAMES_PER_CHUNK.

    Callers feed individual games via ``add()``, then call ``finish()`` to flush
    the final chunk and collect results.  A single-chunk output is renamed from
    stem_0.npz back to stem.npz so the naming stays clean.
    """

    def __init__(self, output_path: str, id_key: str, id_dtype) -> None:
        self._output_path = output_path
        self._stem, self._ext = os.path.splitext(output_path)
        self._id_key = id_key
        self._id_dtype = id_dtype
        self._chars:   list[np.ndarray] = []
        self._colors:  list[np.ndarray] = []
        self._offsets: list[int] = [0]
        self._ids:     list = []
        self._meta:    list[dict] = []
        self._chunk_frames = 0
        self._chunk_idx = 0
        self._paths:   list[str] = []
        self._results: list[dict] = []

    def add(self, chars: np.ndarray, colors: np.ndarray, id_val, game_meta: dict) -> None:
        n = chars.shape[0]
        if self._chars and self._chunk_frames + n > _MAX_FRAMES_PER_CHUNK:
            self._flush()
        self._chars.append(chars)
        self._colors.append(colors)
        self._offsets.append(self._offsets[-1] + n)
        self._ids.append(id_val)
        self._meta.append(game_meta)
        self._chunk_frames += n

    def _flush(self) -> None:
        if not self._chars:
            return
        cpath = f"{self._stem}_{self._chunk_idx}{self._ext}"
        _dbg(f"CONCAT_START {os.path.basename(cpath)} arrays={len(self._chars)}")
        try:
            chars  = np.concatenate(self._chars)
            colors = np.concatenate(self._colors)
            _dbg(f"SAVEZ_START {os.path.basename(cpath)} shape={chars.shape}")
            np.savez_compressed(
                cpath, tty_chars=chars, tty_colors=colors,
                offsets=np.array(self._offsets, dtype=np.int64),
                **{self._id_key: np.array(self._ids, dtype=self._id_dtype)},
            )
            _dbg(f"SAVEZ_END {os.path.basename(cpath)}")
        except Exception as exc:
            _dbg(f"FLUSH_ERROR {os.path.basename(cpath)}: {exc}")
            self._results.append({"status": "error", "path": cpath,
                                   "error": f"flush failed ({self._offsets[-1]} frames): {exc}"})
            self._paths.append(cpath)
            self._chunk_idx += 1
            self._chars = []
            self._colors = []
            self._offsets = [0]
            self._ids = []
            self._meta = []
            self._chunk_frames = 0
            return
        _worker_log(
            f"  [{time.strftime('%H:%M:%S')}] [{os.getpid()}] CHUNK  {os.path.basename(cpath)}"
            f"  {self._offsets[-1]:,} fr  {len(self._offsets)-1} g"
        )
        self._results.append({"status": "ok", "path": cpath,
                               "frames": self._offsets[-1],
                               "games": len(self._offsets) - 1,
                               "game_meta": self._meta, "filtered_games": 0})
        self._paths.append(cpath)
        self._chunk_idx += 1
        self._chars = []
        self._colors = []
        self._offsets = [0]
        self._ids = []
        self._meta = []
        self._chunk_frames = 0

    def finish(self, filtered_games: int = 0) -> list[dict]:
        self._flush()
        if not self._results:
            return [{"status": "filter", "filtered_games": filtered_games}]
        self._results[0]["filtered_games"] += filtered_games
        if len(self._paths) == 1:
            os.rename(self._paths[0], self._output_path)
            self._results[0]["path"] = self._output_path
        return self._results


def _convert_player(task: tuple) -> list[dict]:
    """Decode all games for one nld-nao player into per-player npz chunk(s).

    Uses the module-level ``_xl_by_player`` dict which workers inherit from the
    main process via fork.
    """
    input_files, output_path, ttyrec_version, min_frames, player_name = task
    xl_entries = _xl_by_player.get(player_name, [])

    existing = _find_existing_chunks(output_path)
    if existing:
        results: list[dict] = []
        for cpath in existing:
            _dbg(f"LOAD_EXISTING {cpath}")
            try:
                with np.load(cpath) as f:
                    offsets = f["offsets"]
                    src_ts = f["source_timestamps"] if "source_timestamps" in f else None
            except Exception as exc:
                _dbg(f"LOAD_EXISTING_ERROR {cpath}: {exc}")
                return [{"status": "error", "path": cpath,
                         "msg": f"failed to read {cpath}: {exc}"}]
            n_games = len(offsets) - 1
            game_meta: list[dict] = []
            for i in range(n_games):
                n_frames = int(offsets[i + 1]) - int(offsets[i])
                ts = int(src_ts[i]) if src_ts is not None and i < len(src_ts) else 0
                entry = _match_xlog_entry(xl_entries, ts) if xl_entries and ts else {}
                game_meta.append(_game_meta_from_xlog(entry, n_frames, ts))
            results.append({"status": "skip", "path": cpath,
                             "frames": int(offsets[-1]), "games": n_games,
                             "game_meta": game_meta})
        return results

    _dbg(f"MAKEDIRS {os.path.dirname(output_path) or '.'}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = _ChunkWriter(output_path, "source_timestamps", np.int64)
    filtered_games = 0
    w_ok = w_filter = w_err = 0
    H = W = None
    sorted_files = sorted(input_files)
    n_files = len(sorted_files)
    pid = os.getpid()
    _worker_log(f"  [{time.strftime('%H:%M:%S')}] [{pid}] START  {player_name}  {n_files} files")
    _dbg(f"TASK_START {player_name} n_files={n_files} output={output_path}")
    _worker_progress(player_name, 0, n_files, 0)
    _last_wprint = time.time()
    _last_progress = time.time()

    for i, bz2_path in enumerate(sorted_files):
        current_file = os.path.basename(bz2_path)
        now = time.time()

        if now - _last_wprint >= 30:
            _last_wprint = now
            _worker_log(
                f"  [{time.strftime('%H:%M:%S')}] [{pid}]  READ  {player_name}"
                f"  {i+1}/{n_files}  {current_file}"
                f"  {writer._offsets[-1]:,} fr so far"
            )

        _dbg(f"OPEN_BZ2 {bz2_path}")
        file_ts = _parse_filename_ts(bz2_path)

        _dbg(f"DECODE_START {player_name} [{i+1}/{n_files}] {current_file}")
        try:
            arrays, n_frames = _decode_with_timeout([bz2_path], ttyrec_version)
        except TimeoutError as exc:
            w_err += 1
            _dbg(f"DECODE_TIMEOUT {player_name} [{i+1}/{n_files}] {current_file}: {exc}")
            _worker_progress(player_name, i + 1, n_files, writer._offsets[-1],
                             w_ok, w_filter, w_err, 0, current_file)
            continue
        except Exception as exc:
            w_err += 1
            _dbg(f"DECODE_ERROR {player_name} [{i+1}/{n_files}] {current_file}: {exc}")
            _worker_progress(player_name, i + 1, n_files, writer._offsets[-1],
                             w_ok, w_filter, w_err, 0, current_file)
            continue

        _dbg(f"DECODE_END {player_name} [{i+1}/{n_files}] {current_file} n_frames={n_frames}")

        if not arrays or n_frames < min_frames:
            w_filter += 1
            filtered_games += 1
            if now - _last_progress >= 5.0:
                _last_progress = now
                _worker_progress(player_name, i + 1, n_files, writer._offsets[-1],
                                 w_ok, w_filter, w_err, 0, current_file)
            continue

        if H is None:
            H, W = arrays["tty_chars"].shape[1], arrays["tty_chars"].shape[2]
        chars = arrays["tty_chars"].astype(np.uint8)
        assert H is not None and W is not None
        colors = (arrays["tty_colors"].astype(np.int16).clip(0, 31).astype(np.uint8)
                  if "tty_colors" in arrays
                  else np.zeros((n_frames, H, W), dtype=np.uint8))
        entry = _match_xlog_entry(xl_entries, file_ts) if xl_entries else {}
        writer.add(chars, colors, file_ts, _game_meta_from_xlog(entry, n_frames, file_ts))
        w_ok += 1

        if now - _last_progress >= 5.0:
            _last_progress = now
            _worker_progress(player_name, i + 1, n_files, writer._offsets[-1],
                             w_ok, w_filter, w_err, 0, current_file)

    _dbg(f"TASK_FINISH {player_name} w_ok={w_ok} w_filter={w_filter} w_err={w_err}")
    results = writer.finish(filtered_games)
    del writer
    return results


# --------------------------------------------------------------------------- #
# --- Discovery -------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _discover_nao_top10(extract_dir: str, output_dir: str, min_frames: int) -> list[tuple]:
    """Group DeepMind nao-top10 sessions by player username; return consolidation tasks."""
    src_dir = os.path.join(extract_dir, "nao_top10")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"nao-top10 data not found at {src_dir}")
    tasks: list[tuple] = []
    for player in sorted(os.listdir(src_dir)):
        player_dir = os.path.join(src_dir, player)
        if not os.path.isdir(player_dir):
            continue
        session_files = [
            os.path.join(player_dir, f)
            for f in os.listdir(player_dir)
            if f.endswith(".npz")
        ]
        if not session_files:
            continue
        tasks.append((session_files, os.path.join(output_dir, f"{player}.npz"), min_frames))
    return tasks


def _chunk_paths(output_path: str, n_chunks: int) -> list[str]:
    """Return the list of chunk file paths for a player.

    Single-chunk players keep the original name (no suffix).
    Multi-chunk players use stem_0.npz, stem_1.npz, …
    """
    if n_chunks == 1:
        return [output_path]
    stem, ext = os.path.splitext(output_path)
    return [f"{stem}_{i}{ext}" for i in range(n_chunks)]


def _consolidate_nao_top10_player(task: tuple) -> list[dict]:
    """Merge all nao-top10 sessions for one player into per-player npz chunk(s).

    No xlogfile is available for this dataset; game_meta contains only frame
    counts.  If the total frame count exceeds _MAX_FRAMES_PER_CHUNK the games
    are split into multiple contiguous chunks, each written as a separate npz
    (e.g. Luxidream_0.npz, Luxidream_1.npz …).  Single-chunk players keep the
    original Luxidream.npz naming (no suffix).

    Returns a list of result dicts — one per chunk written (or skipped).
    """
    session_files, output_path, min_frames = task

    # Pass 1: discover valid sessions and per-game frame counts.
    # np.concatenate needs parts + output simultaneously; pre-allocating once and
    # filling in-place halves peak RAM to just the output array + one session.
    valid: list[tuple[str, int]] = []
    game_meta_all: list[dict] = []
    H = W = None

    for npz_path in sorted(session_files):
        try:
            with np.load(npz_path) as f:
                shape = f["tty_chars"].shape
        except Exception:
            continue
        n_frames = shape[0]
        if H is None:
            H, W = shape[1], shape[2]
        if n_frames < min_frames:
            continue
        valid.append((npz_path, n_frames))
        game_meta_all.append(dict(_GAME_META_DEFAULT, length=n_frames))

    if not valid:
        return [{"status": "filter"}]

    # Group valid games into chunks of at most _MAX_FRAMES_PER_CHUNK frames each.
    chunks: list[list[int]] = []   # each inner list is indices into `valid`
    current_chunk: list[int] = []
    current_frames = 0
    for idx, (_, n_frames) in enumerate(valid):
        if current_chunk and current_frames + n_frames > _MAX_FRAMES_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = []
            current_frames = 0
        current_chunk.append(idx)
        current_frames += n_frames
    if current_chunk:
        chunks.append(current_chunk)

    n_chunks = len(chunks)
    paths = _chunk_paths(output_path, n_chunks)

    # Skip check: all expected chunk files must exist; re-do the whole player if
    # only some exist (partial previous run).
    if all(os.path.exists(p) for p in paths):
        results: list[dict] = []
        for chunk_path in paths:
            try:
                with np.load(chunk_path) as f:
                    offsets = f["offsets"]
            except Exception as exc:
                return [{"status": "error", "path": chunk_path,
                         "msg": f"failed to read {chunk_path}: {exc}"}]
            n_games = len(offsets) - 1
            gm = [
                dict(_GAME_META_DEFAULT,
                     length=int(offsets[i + 1]) - int(offsets[i]))
                for i in range(n_games)
            ]
            results.append({"status": "skip", "path": chunk_path,
                             "frames": int(offsets[-1]),
                             "games": n_games, "game_meta": gm})
        return results

    # Pass 2: write each chunk — load sessions into RAM, concatenate, savez.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = []
    for chunk_idx, game_indices in enumerate(chunks):
        chunk_path  = paths[chunk_idx]
        chunk_valid = [valid[i] for i in game_indices]
        chunk_meta  = [game_meta_all[i] for i in game_indices]
        offsets_list: list[int] = [0]
        chars_parts: list[np.ndarray] = []
        colors_parts: list[np.ndarray] = []
        assert H is not None and W is not None
        for npz_path, n_frames in chunk_valid:
            try:
                with np.load(npz_path) as f:
                    chars_parts.append(f["tty_chars"].astype(np.uint8))
                    colors_parts.append(
                        np.clip(f["tty_colors"].astype(np.int16), 0, 31).astype(np.uint8)
                        if "tty_colors" in f
                        else np.zeros((n_frames, H, W), dtype=np.uint8)
                    )
            except Exception:
                chars_parts.append(np.zeros((n_frames, H, W), dtype=np.uint8))
                colors_parts.append(np.zeros((n_frames, H, W), dtype=np.uint8))
            offsets_list.append(offsets_list[-1] + n_frames)
        np.savez_compressed(
            chunk_path,
            tty_chars=np.concatenate(chars_parts),
            tty_colors=np.concatenate(colors_parts),
            offsets=np.array(offsets_list, dtype=np.int64),
        )
        del chars_parts, colors_parts
        results.append({
            "status": "ok",
            "path": chunk_path,
            "frames": offsets_list[-1],
            "games": len(chunk_valid),
            "game_meta": chunk_meta,
        })

    return results


def _read_aa_xlogfile(game_dir: str) -> dict[str, dict]:
    """Return mapping ttyrecname → xlogfile entry for all games in game_dir."""
    _dbg(f"XLOG_LISTDIR {game_dir}")
    for fname in os.listdir(game_dir):
        if fname.endswith(".xlogfile"):
            _dbg(f"XLOG_OPEN {os.path.join(game_dir, fname)}")
            result: dict[str, dict] = {}
            with open(os.path.join(game_dir, fname), "r", errors="replace") as fh:
                for line in fh:
                    entry = _parse_xlog_line(line)
                    key = entry.get("ttyrecname", "")
                    if key:
                        result[key] = entry
            _dbg(f"XLOG_DONE {fname} entries={len(result)}")
            return result
    return {}


def _discover_nld_aa_grouped(nle_data_dir: str, output_dir: str,
                              min_frames: int) -> list[tuple]:
    """One task per game dir (fake player = one Autoascend run, ~100 games each)."""
    data_root = os.path.join(nle_data_dir, "nld-aa", "nle_data")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"nld-aa data not found at {data_root}\n"
            "Pass --nld-aa-subdir to point at the right sub-directory."
        )
    tasks: list[tuple] = []
    for gdir in sorted(os.listdir(data_root)):
        game_dir = os.path.join(data_root, gdir)
        if not os.path.isdir(game_dir):
            continue
        bz2_files = [
            os.path.join(game_dir, f)
            for f in os.listdir(game_dir)
            if f.endswith(".bz2")
        ]
        if not bz2_files:
            continue
        tasks.append((
            bz2_files,
            os.path.join(output_dir, f"{gdir}.npz"),
            3, min_frames, game_dir,
        ))
    return tasks


def _convert_aa_group(task: tuple) -> list[dict]:
    """Decode all games in one nld-aa game dir into per-group npz chunk(s).

    Xlogfile entries are matched via the ``ttyrecname`` field so metadata is
    accurate for every game.
    """
    bz2_files, output_path, ttyrec_version, min_frames, game_dir = task
    xl_by_name = _read_aa_xlogfile(game_dir)

    existing = _find_existing_chunks(output_path)
    if existing:
        results: list[dict] = []
        for cpath in existing:
            try:
                with np.load(cpath) as f:
                    offsets = f["offsets"]
                    src_ids = f["source_game_ids"] if "source_game_ids" in f else None
            except Exception as exc:
                return [{"status": "error", "path": cpath,
                         "msg": f"failed to read {cpath}: {exc}"}]
            n_games = len(offsets) - 1
            game_meta: list[dict] = []
            for i in range(n_games):
                n_frames = int(offsets[i + 1]) - int(offsets[i])
                entry = xl_by_name.get(
                    str(src_ids[i]) if src_ids is not None and i < len(src_ids) else "", {}
                )
                ts = int(entry.get("starttime", 0) or 0)
                game_meta.append(_game_meta_from_xlog(entry, n_frames, ts))
            results.append({"status": "skip", "path": cpath,
                             "frames": int(offsets[-1]), "games": n_games,
                             "game_meta": game_meta})
        return results

    bz2_sorted = sorted(bz2_files, key=lambda p: int(os.path.basename(p).split(".")[2]))
    _dbg(f"MAKEDIRS {os.path.dirname(output_path) or '.'}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = _ChunkWriter(output_path, "source_game_ids", "U64")
    filtered_games = 0
    w_ok = w_filter = w_err = 0
    H = W = None
    group_name = os.path.basename(game_dir)
    n_files = len(bz2_sorted)
    pid = os.getpid()
    _worker_log(f"  [{time.strftime('%H:%M:%S')}] [{pid}] START  {group_name}  {n_files} files")
    _dbg(f"TASK_START {group_name} n_files={n_files} output={output_path}")
    _worker_progress(group_name, 0, n_files, 0)
    _last_wprint = time.time()
    _last_progress = time.time()

    for i, bz2_path in enumerate(bz2_sorted):
        current_file = os.path.basename(bz2_path)
        now = time.time()

        if now - _last_wprint >= 30:
            _last_wprint = now
            _worker_log(
                f"  [{time.strftime('%H:%M:%S')}] [{pid}]  READ  {group_name}"
                f"  {i+1}/{n_files}  {current_file}"
                f"  {writer._offsets[-1]:,} fr so far"
            )

        _dbg(f"OPEN_BZ2 {bz2_path}")
        entry = xl_by_name.get(current_file, {})

        _dbg(f"DECODE_START {group_name} [{i+1}/{n_files}] {current_file}")
        try:
            arrays, n_frames = _decode_with_timeout([bz2_path], ttyrec_version)
        except TimeoutError as exc:
            w_err += 1
            _dbg(f"DECODE_TIMEOUT {group_name} [{i+1}/{n_files}] {current_file}: {exc}")
            _worker_progress(group_name, i + 1, n_files, writer._offsets[-1],
                             w_ok, w_filter, w_err, 0, current_file)
            continue
        except Exception as exc:
            w_err += 1
            _dbg(f"DECODE_ERROR {group_name} [{i+1}/{n_files}] {current_file}: {exc}")
            _worker_progress(group_name, i + 1, n_files, writer._offsets[-1],
                             w_ok, w_filter, w_err, 0, current_file)
            continue

        _dbg(f"DECODE_END {group_name} [{i+1}/{n_files}] {current_file} n_frames={n_frames}")

        if not arrays or n_frames < min_frames:
            w_filter += 1
            filtered_games += 1
            if now - _last_progress >= 5.0:
                _last_progress = now
                _worker_progress(group_name, i + 1, n_files, writer._offsets[-1],
                                 w_ok, w_filter, w_err, 0, current_file)
            continue

        if H is None:
            H, W = arrays["tty_chars"].shape[1], arrays["tty_chars"].shape[2]
        chars = arrays["tty_chars"].astype(np.uint8)
        assert H is not None and W is not None
        colors = (arrays["tty_colors"].astype(np.int16).clip(0, 31).astype(np.uint8)
                  if "tty_colors" in arrays
                  else np.zeros((n_frames, H, W), dtype=np.uint8))
        ts = int(entry.get("starttime", 0) or 0)
        writer.add(chars, colors, current_file, _game_meta_from_xlog(entry, n_frames, ts))
        w_ok += 1

        if now - _last_progress >= 5.0:
            _last_progress = now
            _worker_progress(group_name, i + 1, n_files, writer._offsets[-1],
                             w_ok, w_filter, w_err, 0, current_file)

    _dbg(f"TASK_FINISH {group_name} w_ok={w_ok} w_filter={w_filter} w_err={w_err}")
    results = writer.finish(filtered_games)
    del writer
    return results


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
            1, min_frames, player,
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

def _convert_wrapper(task: tuple) -> tuple:
    """Module-level wrapper so imap_unordered can pickle it.

    Returns (name, n_files, elapsed_s, result_list, exc_str_or_None).
    exc_str is the formatted traceback if the converter raised; None on success.
    """
    t0 = time.monotonic()
    name = os.path.splitext(os.path.basename(task[1]))[0]
    n_files = len(task[0])
    _dbg(f"WRAPPER_START {name} n_files={n_files}")
    try:
        results = _POOL_CONVERTER(task)  # type: ignore[operator]
        _dbg(f"WRAPPER_END {name} elapsed={time.monotonic()-t0:.1f}s")
        return name, n_files, time.monotonic() - t0, results, None
    except Exception:
        tb = traceback.format_exc()
        _dbg(f"WRAPPER_EXCEPTION {name}: {tb}")
        return name, n_files, time.monotonic() - t0, [], tb


def _setup_signal_handlers(counts: dict, wbars: dict, t0: float) -> None:
    """Install SIGTERM / SIGXCPU handlers that dump current state before exit."""
    def _handler(signum: int, frame: object) -> None:
        sig_name = {signal.SIGTERM: "SIGTERM", signal.SIGXCPU: "SIGXCPU"}.get(signum, str(signum))
        elapsed  = time.time() - t0
        active   = list(wbars.keys())
        ram_gb   = psutil.virtual_memory().used / 1024 ** 3
        msg = (
            f"[{time.strftime('%H:%M:%S')}] SIGNAL {sig_name} received  "
            f"elapsed={elapsed:.0f}s  ram={ram_gb:.1f}GB\n"
            f"  counts={counts}\n"
            f"  active_tasks={active}\n"
        )
        if _DEBUG_LOG_PATH:
            try:
                with open(_DEBUG_LOG_PATH, "a") as _f:
                    _f.write(msg)
            except OSError:
                pass
        sys.stderr.write(msg)
        sys.stderr.flush()
        # Re-raise default so the process actually dies.
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
    converter=_convert_player,
    write_index: bool = True,
    checkpoint_every: int = 500,
    max_groups: int = 0,
) -> None:
    """Convert per-player tasks with mp.Pool.imap_unordered and progressive indexing."""
    global _POOL_CONVERTER
    _POOL_CONVERTER = converter

    total = len(tasks)
    print(f"  players found: {total:,}", flush=True)

    index_path = os.path.join(npz_dir, "index.npz")

    accum, indexed_paths = _new_rich_accum(), set()
    if write_index and os.path.exists(index_path):
        accum, indexed_paths = _load_rich_accum(index_path)

    def _task_indexed(task_output_path: str) -> bool:
        if task_output_path in indexed_paths:
            return True
        stem = os.path.splitext(task_output_path)[0]
        return f"{stem}_0.npz" in indexed_paths

    pending = [t for t in tasks if not _task_indexed(t[1])]
    if max_groups > 0:
        pending = pending[:max_groups]
    total_files = sum(len(t[0]) for t in pending)
    print(f"  pending: {len(pending):,} groups  {total_files:,} files", flush=True)

    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as _f:
            for _line in _f:
                k, v = _line.split(":", 1)
                meminfo[k.strip()] = int(v.split()[0])
        commit_limit_gb = meminfo.get("CommitLimit", 0) / 1024 ** 2
        committed_gb    = meminfo.get("Committed_AS", 0) / 1024 ** 2
        print(
            f"  commit budget: {commit_limit_gb:.1f} GB limit, "
            f"{committed_gb:.1f} GB used, "
            f"{commit_limit_gb - committed_gb:.1f} GB free",
            flush=True,
        )
    except Exception:
        pass

    if not pending:
        return

    counts = {"ok": 0, "skip": 0, "filter": 0, "error": 0}
    filtered_games_total = 0
    errors: list[str] = []
    error_paths: list[str] = []
    since_ckpt = 0
    _log_interval = 60
    _t0 = time.time()
    _last_log = _t0
    files_done = 0

    n_workers = min(workers, len(pending))
    _dbg(f"POOL_START n_workers={n_workers} pending={len(pending)} total_files={total_files}")

    # Per-worker tqdm bars (positions 1..n_workers) + overall bar (position 0).
    _wbars:       dict[str, tqdm] = {}
    _wbar_slots:  list[int] = list(range(1, n_workers + 1))
    _wbar_slot_of: dict[str, int] = {}   # name → slot (for safe close)
    _wbar_lock    = threading.Lock()
    _bar_thread:  threading.Thread | None = None

    def _bar_manager() -> None:
        while True:
            try:
                msg = _PROGRESS_QUEUE.get(timeout=0.5)  # type: ignore[union-attr]
            except Exception:
                continue
            if msg is None:
                break
            name, done, total_t, frames, ok, filter_g, err, skip, current_file = msg
            with _wbar_lock:
                if name not in _wbars:
                    if not _wbar_slots:
                        continue
                    slot = _wbar_slots.pop(0)
                    _wbar_slot_of[name] = slot
                    _wbars[name] = tqdm(
                        total=total_t, initial=done,
                        desc=f"  {name[:20]}", position=slot,
                        leave=False, unit="f", file=sys.stdout,
                        ncols=120, smoothing=0.1, mininterval=60.0,
                    )
                b = _wbars[name]
                delta = done - b.n
                if delta > 0:
                    b.update(delta)
                b.set_postfix(
                    ok=ok, filt=filter_g, err=err,
                    fr=f"{frames:,}",
                    f=current_file[:18] if current_file else "",
                    refresh=True,
                )

    if _PROGRESS_QUEUE is not None:
        _bar_thread = threading.Thread(target=_bar_manager, daemon=True)
        _bar_thread.start()

    # Heartbeat: writes to debug log every 10s so we can detect a stuck main loop.
    _hb_stop = threading.Event()
    def _heartbeat() -> None:
        while not _hb_stop.wait(10.0):
            active = list(_wbars.keys())
            ram_gb = psutil.virtual_memory().used / 1024 ** 3
            _dbg(
                f"HEARTBEAT ok={counts['ok']} skip={counts['skip']} "
                f"err={counts['error']} filter={counts['filter']} "
                f"files_done={files_done}/{total_files} "
                f"active={len(active)} names={active[:8]} "
                f"ram={ram_gb:.1f}GB"
            )
    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    _setup_signal_handlers(counts, _wbars, _t0)

    with mp.Pool(processes=n_workers, maxtasksperchild=8) as pool:
        result_iter = pool.imap_unordered(_convert_wrapper, pending)

        with tqdm(total=total_files, unit="file", desc="  total",
                  ncols=120, smoothing=0.1, position=0,
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

                name, n_task_files, elapsed, result_list, exc_str = raw

                # Close the per-worker bar and return its slot.
                with _wbar_lock:
                    if name in _wbars:
                        _wbars[name].close()
                        slot = _wbar_slot_of.pop(name, None)
                        if slot is not None:
                            _wbar_slots.append(slot)
                        del _wbars[name]

                if exc_str:
                    _dbg(f"TASK_EXCEPTION {name}:\n{exc_str}")
                    counts["error"] += 1
                    errors.append(exc_str.splitlines()[-1])
                    tqdm.write(
                        f"  [{time.strftime('%H:%M:%S')}] ERR   "
                        f"  {name:<32}  {n_task_files:>5} files  {elapsed:.0f}s"
                        f"  {errors[-1]}",
                        file=sys.stdout,
                    )
                    files_done += n_task_files
                    bar.update(n_task_files)
                    bar.set_postfix(ok=counts["ok"], skip=counts["skip"],
                                    filt_g=filtered_games_total, err=counts["error"])
                    continue

                for result in result_list:
                    status = result["status"]
                    counts[status] += 1
                    if status in ("ok", "skip") and result.get("game_meta"):
                        _accum_player_result(accum, result)
                        since_ckpt += 1
                    elif status == "error":
                        errors.append(result.get("error", result.get("msg", "unknown")))
                        if result.get("path"):
                            error_paths.append(result["path"])
                    filtered_games_total += result.get("filtered_games", 0)

                total_fr   = sum(r.get("frames", 0) for r in result_list)
                total_g    = sum(r.get("games",  0) for r in result_list)
                n_chunks   = len(result_list)
                top_status = result_list[0]["status"].upper() if result_list else "UNK"
                chunk_tag  = f"×{n_chunks}" if n_chunks > 1 else "   "
                tqdm.write(
                    f"  [{time.strftime('%H:%M:%S')}] {top_status}{chunk_tag}"
                    f"  {name:<32}  {n_task_files:>5} files"
                    f"  {total_fr:>10,} fr  {total_g:>6,} g  {elapsed:>6.0f}s",
                    file=sys.stdout,
                )

                files_done += n_task_files
                bar.update(n_task_files)
                bar.set_postfix(ok=counts["ok"], skip=counts["skip"],
                                filt_g=filtered_games_total, err=counts["error"])

                now = time.time()
                if now - _last_log >= _log_interval:
                    _last_log = now
                    groups_done = counts["ok"] + counts["skip"] + counts["error"]
                    ram_gb  = psutil.virtual_memory().used  / 1024 ** 3
                    ram_tot = psutil.virtual_memory().total / 1024 ** 3
                    npz_on_disk = sum(
                        1 for e in os.scandir(npz_dir)
                        if e.name.endswith(".npz") and e.name != "index.npz"
                    )
                    tqdm.write(
                        f"\n  [{time.strftime('%H:%M:%S')}] === {files_done:,}/{total_files:,} files"
                        f"  ({groups_done}/{len(pending)} groups)"
                        f"  npz_on_disk={npz_on_disk:,}"
                        f"  ok={counts['ok']} skip={counts['skip']} err={counts['error']}"
                        f"  ram={ram_gb:.0f}/{ram_tot:.0f}GB"
                        f"  elapsed={(now - _t0)/60:.1f}min ===\n",
                        file=sys.stdout,
                    )

                if write_index and since_ckpt >= checkpoint_every and accum["pl_paths"]:
                    _dbg(f"INDEX_CHECKPOINT players={len(accum['pl_paths'])}")
                    _write_index_rich(index_path, accum)
                    since_ckpt = 0

    # Shut down background threads.
    _hb_stop.set()
    if _PROGRESS_QUEUE is not None and _bar_thread is not None:
        _PROGRESS_QUEUE.put(None)
        _bar_thread.join(timeout=2.0)

    _dbg("POOL_CLOSED writing final index")

    if write_index and accum["pl_paths"]:
        _write_index_rich(index_path, accum)

    if error_paths:
        retry_path = os.path.join(npz_dir, "errors.txt")
        with open(retry_path, "a") as _f:
            for p in error_paths:
                _f.write(p + "\n")
        print(f"\n  {len(error_paths)} failed group(s) recorded in {retry_path}", flush=True)

    if errors:
        print("\n  first 10 errors:", flush=True)
        for msg in errors[:10]:
            print(f"    {msg}", flush=True)

    print(
        f"\n  convert summary: ok={counts['ok']} skip={counts['skip']} "
        f"filt_games={filtered_games_total} error={counts['error']}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# --- Index scan (--skip-convert case) --------------------------------------- #
# --------------------------------------------------------------------------- #

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


def _build_rich_index_from_scan(
    scan_dir: str, workers: int, index_path: str,
    nle_data_dir: str | None = None,
    recursive: bool = True,
) -> None:
    """Rebuild a rich index by scanning existing per-player npz files."""
    global _xl_by_player

    if recursive:
        npz_files = [
            os.path.join(dp, f)
            for dp, _, files in os.walk(scan_dir)
            for f in files
            if f.endswith(".npz") and f != "index.npz"
        ]
    else:
        # Non-recursive: top-level flat files only (avoids source session sub-dirs).
        npz_files = [
            os.path.join(scan_dir, f)
            for f in os.listdir(scan_dir)
            if f.endswith(".npz") and f != "index.npz"
        ]
    total = len(npz_files)
    print(f"  scanning {total:,} player files in {scan_dir} ...", flush=True)

    if not _xl_by_player and nle_data_dir is not None:
        nld_nao_dir = os.path.join(nle_data_dir, "nld-nao")
        _xl_by_player = _load_xlogfiles(nld_nao_dir)

    accum = _new_rich_accum()
    errors = 0

    with mp.Pool(workers) as pool:
        with tqdm(total=total, unit="player", desc="  index", dynamic_ncols=True, file=sys.stdout) as bar:
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
    global _PROGRESS_QUEUE, _WORKER_LOG_PATH, _DEBUG_LOG_PATH
    raw         = args.raw_dir or args.output_dir
    zip_dir     = os.path.join(raw, "zips", "nao-top10")
    tar_path    = os.path.join(zip_dir, "nao_top10.tar")
    extract_dir = os.path.join(raw, "nao-top10")
    npz_dir     = os.path.join(args.output_dir, "nle", "nao-top10")
    index_path  = os.path.join(npz_dir, "index.npz")

    os.makedirs(npz_dir, exist_ok=True)
    _PROGRESS_QUEUE = mp.Queue()
    _WORKER_LOG_PATH = os.path.join(npz_dir, "workers.log")
    _DEBUG_LOG_PATH  = os.path.join(npz_dir, "debug.log")

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
        tasks = _discover_nao_top10(extract_dir, npz_dir, args.min_frames)
        print(f"[index]    progressive → {index_path}", flush=True)
        _run_convert_rich(
            tasks, args.workers, npz_dir,
            converter=_consolidate_nao_top10_player,
            write_index=not args.skip_index,
            max_groups=args.max_groups,
        )
    else:
        print("[convert]  skipped", flush=True)
        if not args.skip_index:
            os.makedirs(npz_dir, exist_ok=True)
            print(f"[index]    → {index_path}", flush=True)
            _build_rich_index_from_scan(npz_dir, args.workers, index_path)
        else:
            print("[index]    skipped", flush=True)

    print("\nDone. Set in your experiment config:", flush=True)
    print(f"  data.index_path: {index_path}", flush=True)


def _run_nld(dataset: str, args: BaseArgs) -> None:
    global _PROGRESS_QUEUE, _WORKER_LOG_PATH, _DEBUG_LOG_PATH
    assert dataset in ("nld-aa", "nld-nao")
    raw         = args.raw_dir or args.output_dir
    zip_dir     = os.path.join(raw, "zips", dataset)
    extract_dir = os.path.join(raw, dataset)
    db_path     = os.path.join(raw, f"{dataset}.db")
    _npz_subdirs = {"nld-aa": "aa", "nld-nao": "nao"}
    npz_dir     = os.path.join(args.output_dir, "nle", _npz_subdirs[dataset])
    index_path  = os.path.join(npz_dir, "index.npz")

    os.makedirs(npz_dir, exist_ok=True)
    _PROGRESS_QUEUE = mp.Queue()
    _WORKER_LOG_PATH = os.path.join(npz_dir, "workers.log")
    _DEBUG_LOG_PATH  = os.path.join(npz_dir, "debug.log")
    print(f"[log]      debug → {_DEBUG_LOG_PATH}", flush=True)

    filenames  = _nld_aa_zips() if dataset == "nld-aa" else _nld_nao_zips()
    base_url   = _NLD_AA_BASE   if dataset == "nld-aa" else _NLD_NAO_BASE
    use_altorg = dataset == "nld-nao"

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
            _remove_archives(filenames, zip_dir)
    else:
        print("[extract]  skipped", flush=True)

    if not args.skip_db:
        print(f"[db]       → {db_path}", flush=True)
        _build_nle_db(extract_dir, db_path, dataset, use_altorg)
    else:
        print("[db]       skipped", flush=True)

    if not args.skip_convert:
        os.makedirs(npz_dir, exist_ok=True)
        print(f"[convert]  → {npz_dir}", flush=True)
        if dataset == "nld-aa":
            tasks = _discover_nld_aa_grouped(raw, npz_dir, args.min_frames)
        else:
            tasks = _discover_nld_nao(raw, npz_dir, args.min_frames)
        converter = _convert_aa_group if dataset == "nld-aa" else _convert_player
        print(f"[index]    progressive → {index_path}", flush=True)
        _run_convert_rich(
            tasks, args.workers, npz_dir,
            converter=converter,
            write_index=not args.skip_index,
            max_groups=args.max_groups,
        )
    else:
        print("[convert]  skipped", flush=True)
        if not args.skip_index:
            os.makedirs(npz_dir, exist_ok=True)
            print(f"[index]    → {index_path}", flush=True)
            nle_data_dir = raw if dataset == "nld-nao" else None
            _build_rich_index_from_scan(npz_dir, args.workers, index_path, nle_data_dir)
        else:
            print("[index]    skipped", flush=True)

    print("\nDone. Set in your experiment config:", flush=True)
    print(f"  data.nle_data_dir: {raw}", flush=True)
    print(f"  data.index_path:   {index_path}", flush=True)


# --------------------------------------------------------------------------- #
# --- Entry point ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

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
