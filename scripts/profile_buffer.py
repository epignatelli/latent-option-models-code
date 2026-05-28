"""Profile buffer_size tradeoffs: startup time, RAM, and sample reuse ratio.

Usage:
    python -m scripts.profile_buffer                     # measure GPU speed, then sweep
    python -m scripts.profile_buffer --sizes 50 200 500  # custom sweep
    python -m scripts.profile_buffer --no-gpu            # skip GPU measurement

The script:
  1. Measures actual GPU training throughput (samp/s) with a short burst.
  2. For each buffer size, measures startup time, RAM, and unique start positions.
  3. Reports the reuse ratio using the measured GPU speed.

A reuse ratio close to 1 means the model rarely sees the same (game, t) pair
twice before that game slot is replaced. Higher ratios mean more repetition.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.optim as optim

from lom.config import EnvCfg, ModelCfg
from lom.dataset import GameBuffer, build_npz_dataloaders
from lom.models import DynamicsModel, LatentActionModel
from lom.modules import tokenise
from lom.training import reconstruction_loss, NullCtx

LOG_FILE = "/scratch/uceeepi/lom/profile_buffer.log"

_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_sh)
_fh = logging.FileHandler(LOG_FILE, mode="a")
_fh.setFormatter(_fmt)
_root.addHandler(_fh)
log = logging.getLogger(__name__)

DEFAULT_INDEX        = "/scratch/uceeepi/lom/datasets/nle/nao/index.npz"
DEFAULT_SIZES        = [50, 100, 200, 500]
DEFAULT_BATCH_SIZES  = [256, 512, 1024, 2048, 4096]
CTX                  = 4
HORIZON              = 128
REFRESH_FRAC         = 0.1
REFRESH_EVERY_S      = 60.0
SEED                 = 42

# GPU measurement settings
GPU_WARMUP_STEPS  = 20
GPU_MEASURE_STEPS = 50


def load_index(path: str):
    idx = np.load(path)
    if "player_paths" in idx:
        return idx["player_paths"].astype(str), idx["player_lengths"].astype(np.int32)
    return idx["paths"].astype(str), idx["lengths"].astype(np.int32)


def _build_models(device):
    """Build LAM + DynamicsModel for throughput measurement."""
    e = EnvCfg()
    m = ModelCfg(d_model=256, n_layers=4, n_heads=4, context_length=256,
                 latent_dim=512, num_options=256, patch_size=4)
    lam = LatentActionModel(
        vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w,
        d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
        context_length=m.context_length, latent_dim=m.latent_dim,
        codebook_size=m.num_options, horizon=1, patch_size=m.patch_size,
    ).to(device)
    dyn = DynamicsModel(
        vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w,
        d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
        context_length=m.context_length, latent_dim=m.latent_dim,
        patch_size=m.patch_size, predict_sequence=False,
    ).to(device)
    return e, lam, dyn


def measure_one_batch_size(batch_size: int, device, e, lam, dyn, ctx, dataset) -> float:
    """Run a short training burst at one batch size and return samp/s."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = optim.AdamW(list(lam.parameters()) + list(dyn.parameters()), lr=3e-4)
    data_iter = iter(loader)

    for s in range(GPU_WARMUP_STEPS + GPU_MEASURE_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        history, next_frame = batch[0].to(device), batch[1].to(device)
        with ctx:
            z, vq, _ = lam(history, next_frame)
            logits   = dyn(history, z)
            loss     = reconstruction_loss(logits, tokenise(next_frame), e.vocab_size) + vq["vq_loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if s == GPU_WARMUP_STEPS - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start  # type: ignore[possibly-undefined]
    return GPU_MEASURE_STEPS * batch_size / elapsed


def measure_gpu_throughput(index_path: str, batch_sizes: list[int]) -> dict[int, float]:
    """Sweep batch sizes and return {batch_size: samp/s}.

    Loads the dataset buffer once, then reuses it across all batch sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Measuring GPU throughput on %s  (warmup=%d  measure=%d steps per size)",
             device, GPU_WARMUP_STEPS, GPU_MEASURE_STEPS)

    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else NullCtx())
    e, lam, dyn = _build_models(device)

    log.info("Loading dataset buffer once for all batch sizes ...")
    n_steps = GPU_WARMUP_STEPS + GPU_MEASURE_STEPS
    train_ds, _ = build_npz_dataloaders(
        index_path=index_path, context_len=CTX, horizon=1,
        batch_size=1, buffer_size=10,
        steps_per_epoch=n_steps * max(batch_sizes),
        seed=SEED,
    )
    dataset = train_ds.dataset

    results = {}
    for bs in batch_sizes:
        try:
            sps = measure_one_batch_size(bs, device, e, lam, dyn, ctx, dataset)
            results[bs] = sps
            log.info("  batch=%5d  →  %7.0f samp/s  (%5.1f steps/s)", bs, sps, sps / bs)
        except torch.cuda.OutOfMemoryError:
            log.info("  batch=%5d  →  OOM, stopping sweep", bs)
            break

    if results:
        best_bs = max(results, key=results.__getitem__)
        log.info("  peak: batch=%d at %.0f samp/s", best_bs, results[best_bs])
    return results


def profile_one(paths, lengths, buffer_size: int, horizon: int) -> dict:
    t0 = time.perf_counter()
    buf = GameBuffer(
        paths, lengths,
        buffer_size=buffer_size,
        context_len=CTX,
        horizon=horizon,
        seed=SEED,
        refresh_fraction=REFRESH_FRAC,
        refresh_every=REFRESH_EVERY_S,
    )
    startup_s = time.perf_counter() - t0

    games, _ = buf._state
    game_lens   = np.array([len(g) for g in games], dtype=np.int64)
    ram_bytes   = sum(int(g.nbytes) for g in games)
    valid_lens  = np.maximum(game_lens - (CTX + horizon - 1), 0)
    total_valid = int(valid_lens.sum())
    buf.stop()

    n_refresh       = min(max(1, int(buffer_size * REFRESH_FRAC)), buffer_size)
    load_s_per_game = startup_s / buffer_size
    refresh_cycle_s = n_refresh * load_s_per_game + REFRESH_EVERY_S

    return dict(
        buffer_size     = buffer_size,
        n_games         = len(games),
        startup_s       = startup_s,
        load_s_per_game = load_s_per_game,
        ram_bytes       = ram_bytes,
        avg_game_frames = float(game_lens.mean()),
        total_valid     = total_valid,
        n_refresh       = n_refresh,
        refresh_cycle_s = refresh_cycle_s,
    )


def print_results(rows: list[dict], gpu_results: dict[int, float], horizon: int) -> None:
    log.info("")
    log.info("═" * 80)
    log.info("  GPU throughput sweep:")
    log.info("  %9s  %12s  %12s", "batch_size", "samp/s", "steps/s")
    log.info("  " + "-" * 36)
    for bs, sps in sorted(gpu_results.items()):
        log.info("  %9d  %12.0f  %12.1f", bs, sps, sps / bs)

    best_bs  = max(gpu_results, key=gpu_results.__getitem__)
    best_sps = gpu_results[best_bs]
    log.info("  → peak throughput: batch=%d  samp/s=%.0f", best_bs, best_sps)

    log.info("")
    log.info("  Buffer sweep  (horizon=%d  ctx=%d  GPU=%.0f samp/s @ batch=%d):",
             horizon, CTX, best_sps, best_bs)
    log.info("  %9s  %10s  %8s  %12s  %10s",
             "buf_size", "startup(s)", "RAM(MB)", "uniq_starts", "reuse")
    log.info("  " + "-" * 58)
    for r in rows:
        reuse = best_sps * r["refresh_cycle_s"] / max(1, r["total_valid"])
        flag  = "✓" if reuse < 5 else ("~" if reuse < 20 else "✗")
        log.info("  %9d  %10.0f  %8.0f  %12d  %8.1fx %s",
                 r["buffer_size"], r["startup_s"], r["ram_bytes"] / 1e6,
                 r["total_valid"], reuse, flag)

    log.info("")
    log.info("  Reuse ratio: times each unique start is seen per refresh cycle.")
    log.info("  < 5x = good.  5–20x = acceptable.  > 20x = increase buffer.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index",       default=DEFAULT_INDEX)
    parser.add_argument("--sizes",       type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--horizon",     type=int, default=HORIZON)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
                        help="batch sizes to sweep for GPU throughput measurement")
    parser.add_argument("--no-gpu",      action="store_true",
                        help="skip GPU measurement, use 20k samp/s placeholder")
    args = parser.parse_args()

    log.info("Loading index from %s ...", args.index)
    paths, lengths = load_index(args.index)
    log.info("  %d players  total frames: %.1fM  median: %.0f",
             len(paths), lengths.sum() / 1e6, float(np.median(lengths)))

    if args.no_gpu:
        gpu_results: dict[int, float] = {2048: 20_000.0}
        log.info("Skipping GPU measurement, using placeholder: 20000 samp/s")
    else:
        gpu_results = measure_gpu_throughput(args.index, args.batch_sizes)

    rows = []
    for bs in args.sizes:
        log.info("")
        log.info("Profiling buffer_size=%d ...", bs)
        r = profile_one(paths, lengths, bs, args.horizon)
        rows.append(r)
        log.info("  done: %.0fs startup  %.0fMB RAM  %d unique starts",
                 r["startup_s"], r["ram_bytes"] / 1e6, r["total_valid"])

    print_results(rows, gpu_results, args.horizon)


if __name__ == "__main__":
    main()
