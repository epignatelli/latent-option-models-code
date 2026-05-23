"""Sweep worker counts for nld-aa conversion without memmaps.

Finds the maximum number of workers that can convert N_GROUPS groups without
running out of memory.  Each trial runs in a fresh subprocess with a temporary
output directory so results are never cached between trials.

Usage::

    python scripts/worker_sweep.py \\
        --data-dir /scratch/uceeepi/lom/datasets \\
        --n-groups 20

Output example::

    worker sweep -- nld-aa, N_GROUPS=20, --no-use-memmap
    workers=128 ... OOM (15s)
    workers=64  ... OK  (41s, 0.49 groups/s)
    workers=32  ... OK  (38s, 0.53 groups/s)
    workers=16  ... OK  (89s, 0.22 groups/s)
    workers=8   ... OK  (142s, 0.14 groups/s)
    ...
    max safe workers:  64
    fastest workers:   32  (0.53 groups/s)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import tyro


_WORKER_COUNTS = [128, 64, 32, 16, 8, 4, 2, 1]

# Exit code used by the subprocess to signal OOM (MemoryError or killed by OOM
# killer).  The OS returns 137 (128+9) when a process is killed by SIGKILL,
# which is what the Linux OOM killer sends.
_OOM_EXIT_CODES = {137}

# Substrings that appear in stderr when Python raises MemoryError.
_OOM_STDERR_MARKERS = ("memoryerror", "cannot allocate", "out of memory")


@dataclass
class SweepArgs:
    data_dir: str
    """Root directory containing nld-aa/ and nle/aa/ (the --output-dir used for prepare_data)."""
    n_groups: int = 20
    """Number of nld-aa groups to convert in each trial."""
    worker_counts: list[int] = field(default_factory=lambda: list(_WORKER_COUNTS))
    """Worker counts to try, in order."""


# --------------------------------------------------------------------------- #
# --- helpers ---------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _find_prepare_data() -> str:
    """Return an absolute path to prepare_data.py, robust to cwd changes."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "prepare_data.py")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"prepare_data.py not found next to worker_sweep.py (looked at {candidate})"
    )


def _is_oom(returncode: int, stderr: str) -> bool:
    """Classify a subprocess failure as OOM or other."""
    if returncode in _OOM_EXIT_CODES:
        return True
    stderr_lower = stderr.lower()
    return any(m in stderr_lower for m in _OOM_STDERR_MARKERS)


def _run_trial(
    prepare_data_path: str,
    data_dir: str,
    tmp_output_dir: str,
    workers: int,
    n_groups: int,
) -> tuple[bool, bool, float, str]:
    """Run one trial and return (success, oom, wall_time_s, stderr_tail).

    Returns:
        success: True if process exited 0.
        oom: True if the failure looks like an out-of-memory condition.
        wall_time_s: Elapsed wall-clock seconds.
        stderr_tail: Last 2 KB of stderr for diagnostics.
    """
    cmd = [
        sys.executable,
        prepare_data_path,
        "nld-aa",
        "--output-dir", tmp_output_dir,
        "--workers", str(workers),
        "--max-groups", str(n_groups),
        "--skip-download",
        "--skip-extract",
        "--skip-db",
        "--skip-index",
    ]

    # Point the trial at the real source data by symlinking nld-aa into the
    # temp output dir.  prepare_data expects <output_dir>/nld-aa/nle_data/...
    real_nld_aa = os.path.join(data_dir, "nld-aa")
    tmp_nld_aa  = os.path.join(tmp_output_dir, "nld-aa")
    if os.path.isdir(real_nld_aa) and not os.path.exists(tmp_nld_aa):
        os.symlink(real_nld_aa, tmp_nld_aa)

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    except Exception as exc:
        return False, False, time.monotonic() - t0, str(exc)

    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)
            print(f"    {line}", end="", flush=True)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    for line in proc.stdout:
        print(f"    {line}", end="", flush=True)

    proc.wait()
    stderr_thread.join()

    elapsed = time.monotonic() - t0
    stderr = "".join(stderr_lines)
    stderr_tail = stderr[-2048:]

    if proc.returncode == 0:
        return True, False, elapsed, stderr_tail

    oom = _is_oom(proc.returncode, stderr)
    return False, oom, elapsed, stderr_tail


# --------------------------------------------------------------------------- #
# --- main sweep ------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = tyro.cli(SweepArgs)

    prepare_data_path = _find_prepare_data()

    # Determine the filesystem for tmp dirs: use the same mount as data_dir so
    # symlinks within the tmp dir resolve correctly and no cross-device copies occur.
    tmp_root = args.data_dir
    os.makedirs(tmp_root, exist_ok=True)

    print(
        f"\nworker sweep -- nld-aa, N_GROUPS={args.n_groups}",
        flush=True,
    )
    print(f"data_dir: {args.data_dir}", flush=True)
    print(f"worker counts: {args.worker_counts}\n", flush=True)

    max_safe: int | None = None
    results: list[dict] = []

    for n_workers in args.worker_counts:
        tmp_output_dir = tempfile.mkdtemp(
            prefix=f"worker_sweep_{n_workers}w_",
            dir=tmp_root,
        )
        try:
            print(f"workers={n_workers:<3}  ...", end="  ", flush=True)
            success, oom, elapsed, stderr_tail = _run_trial(
                prepare_data_path=prepare_data_path,
                data_dir=args.data_dir,
                tmp_output_dir=tmp_output_dir,
                workers=n_workers,
                n_groups=args.n_groups,
            )
        finally:
            # Always clean up, even if the trial crashed.
            try:
                shutil.rmtree(tmp_output_dir, ignore_errors=True)
            except Exception:
                pass

        groups_per_s = args.n_groups / elapsed if elapsed > 0 else 0.0

        if success:
            status_str = f"OK  ({elapsed:.0f}s, {groups_per_s:.2f} groups/s)"
            max_safe = n_workers
        elif oom:
            status_str = f"OOM ({elapsed:.0f}s)"
        else:
            status_str = f"FAIL (rc!=0, {elapsed:.0f}s)"
            if stderr_tail:
                stderr_preview = stderr_tail.strip().splitlines()[-1][:120]
                status_str += f"\n            stderr: {stderr_preview}"

        print(status_str, flush=True)

        results.append({
            "workers": n_workers,
            "success": success,
            "oom": oom,
            "elapsed": elapsed,
            "groups_per_s": groups_per_s,
        })

        # In reverse order, OOM at N workers does NOT imply OOM at N/2 — keep going.

    print(flush=True)
    successful = [r for r in results if r["success"]]
    if successful:
        best = max(successful, key=lambda r: r["groups_per_s"])
        print(f"max safe workers:  {max(r['workers'] for r in successful)}", flush=True)
        print(f"fastest workers:   {best['workers']}  ({best['groups_per_s']:.2f} groups/s)", flush=True)
    else:
        print("no successful trials", flush=True)


if __name__ == "__main__":
    main()
