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
    workers=1  ... OK  (142s, 0.14 groups/s)
    workers=2  ... OK  (89s,  0.22 groups/s)
    workers=4  ... OOM (23s)
    max safe workers: 2
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import tyro


_WORKER_COUNTS = [1, 2, 4, 8, 16, 32, 64]

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
        "--no-use-memmap",
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return False, False, elapsed, str(exc)

    elapsed = time.monotonic() - t0
    stderr_tail = result.stderr[-2048:] if result.stderr else ""

    if result.returncode == 0:
        return True, False, elapsed, stderr_tail

    oom = _is_oom(result.returncode, result.stderr)
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
        f"\nworker sweep -- nld-aa, N_GROUPS={args.n_groups}, --no-use-memmap",
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

        if oom:
            print(
                "\n  stopping sweep: OOM at workers={}, no point testing higher counts.".format(n_workers),
                flush=True,
            )
            break

    print(flush=True)
    if max_safe is not None:
        print(f"max safe workers: {max_safe}", flush=True)
    else:
        print("max safe workers: none (all trials failed or OOMed)", flush=True)


if __name__ == "__main__":
    main()
