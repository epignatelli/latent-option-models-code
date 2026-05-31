"""Profile GPU memory and training throughput.

Four model variants:

  --encoder reconstruction  — ReconstructionLOM (bidirectional, pixel reconstruction)
  --encoder latent          — LatentLOM compute-equivalent (d_model=256, n_layers=4)
  --encoder latent-medium   — LatentLOM medium (~100M, d_model=512, n_layers=8)
  --encoder latent-params   — LatentLOM param-matched to reconstruction (~152M)

  --method lam|lom controls only the horizon (1 vs --horizon).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --encoder reconstruction
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --encoder latent
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --pareto --encoder reconstruction --horizon 128
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --pareto --encoder latent --horizon 128

Each batch size runs in a fresh subprocess to avoid CUDA context corruption
from previous OOM events. Uses synthetic random data — no dataset required.
Logs to stdout and /scratch/uceeepi/lom/profile_memory.log.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
import traceback

import torch
import torch.optim as optim

from lom.config import EnvCfg, ModelCfg
from lom.models import ReconstructionLOM, LatentLOM
from lom.tokeniser import tokenise
from lom.training import NullCtx, jepa_loss, reconstruction_loss

LOG_FILE = "/scratch/uceeepi/lom/profile_memory.log"

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

DEFAULT_BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_CTX         = 4
DEFAULT_HORIZON     = 128
SEED                = 42
GPU_WARMUP_STEPS    = 20
GPU_MEASURE_STEPS   = 50


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #

def _model_cfg(context_len: int, encoder: str = "reconstruction") -> tuple[EnvCfg, ModelCfg]:
    e = EnvCfg()
    if encoder == "latent-params":
        # Parameter-matched to reconstruction (~152M): d_model=512, n_layers=12, n_heads=8 (head_dim=64)
        m = ModelCfg(d_model=512, n_layers=12, n_heads=8, context_length=context_len,
                     latent_dim=512, num_options=256, patch_size=8)
    elif encoder == "latent-medium":
        # Medium latent: ~100M backbone params; d_model=512, n_layers=8, n_heads=8
        m = ModelCfg(d_model=512, n_layers=8, n_heads=8, context_length=context_len,
                     latent_dim=512, num_options=256, patch_size=8)
    else:
        # reconstruction or latent (compute-equivalent): d_model=256, n_layers=4, n_heads=4
        m = ModelCfg(d_model=256, n_layers=4, n_heads=4, context_length=context_len,
                     latent_dim=512, num_options=256, patch_size=8)
    return e, m


def _build_models(device, method: str, encoder: str,
                  context_len: int, horizon: int) -> tuple[EnvCfg, dict]:
    e, m = _model_cfg(context_len, encoder)
    h = 1 if method == "lam" else horizon
    if encoder.startswith("latent"):
        model: ReconstructionLOM | LatentLOM = LatentLOM(
            vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w,
            n_actions=e.n_actions,
            d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
            context_length=m.context_length, horizon=h,
            latent_dim=m.latent_dim, num_options=m.num_options,
            patch_size=m.patch_size,
            ema_decay=0.996,
            vq_dropout=0.1, vq_entropy_weight=0.01,
            vq_beta=0.25, vq_reset_thresh=100, vq_ema_decay=0.99,
        ).to(device)
    else:
        model = ReconstructionLOM(
            vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w,
            n_actions=e.n_actions,
            d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
            context_length=m.context_length, horizon=h,
            latent_dim=m.latent_dim, num_options=m.num_options,
            patch_size=m.patch_size,
            vq_dropout=0.1, vq_entropy_weight=0.01,
            vq_beta=0.25, vq_reset_thresh=100, vq_ema_decay=0.99,
        ).to(device)
    return e, {"model": model}


# --------------------------------------------------------------------------- #
# Training step
# --------------------------------------------------------------------------- #

def _run_step(method: str, encoder: str, models: dict,
              batch: list, device, ctx, e) -> torch.Tensor:
    history = batch[0].to(device)
    future  = batch[1].to(device)
    out = models["model"](history, future)
    if encoder.startswith("latent"):
        return (jepa_loss(out["z_act_hat"], out["z_act_target"])
                + jepa_loss(out["z_opt_hat"], out["z_opt_target"])
                + out["vq_opt"]["vq_loss"] + out["vq_act"]["vq_loss"])
    else:
        lam_recon = reconstruction_loss(
            out["lam_logits"], tokenise(future[:, 0:1]), e.vocab_size
        )
        lom_recon = reconstruction_loss(
            out["lom_logits"], tokenise(future[:, -1:]), e.vocab_size
        )
        return lam_recon + lom_recon + out["vq_opt"]["vq_loss"] + out["vq_act"]["vq_loss"]


def _make_frame(shape: tuple, rng: torch.Generator) -> torch.Tensor:
    """Random frame with valid token ranges: char in [0,256), color in [0,32)."""
    f = torch.empty(shape, dtype=torch.uint8)
    f[..., 0] = torch.randint(0, 256, shape[:-1], generator=rng)
    f[..., 1] = torch.randint(0,  32, shape[:-1], generator=rng)
    return f


def _make_dummy_batch(batch_size: int, method: str,
                      context_len: int, horizon: int, e) -> list[torch.Tensor]:
    H, W = e.obs_h, e.obs_w
    rng = torch.Generator()
    rng.manual_seed(SEED)
    h = 1 if method == "lam" else horizon
    history = _make_frame((batch_size, context_len, H, W, 2), rng)
    future  = _make_frame((batch_size, h, H, W, 2), rng)
    return [history, future]


def _mem(device) -> float:
    return torch.cuda.memory_allocated(device) / 1e9


def _run_step_traced(method: str, encoder: str, models: dict,
                     batch: list, device, ctx, e) -> torch.Tensor:
    """Same as _run_step but logs memory at each major operation."""
    log.info("    [trace] base:                  %.2f GB", _mem(device))
    history = batch[0].to(device)
    future  = batch[1].to(device)
    log.info("    [trace] after data.to(device): %.2f GB", _mem(device))
    with ctx:
        out = models["model"](history, future)
    log.info("    [trace] after forward:         %.2f GB", _mem(device))
    with ctx:
        if encoder.startswith("latent"):
            loss = (jepa_loss(out["z_act_hat"], out["z_act_target"])
                    + jepa_loss(out["z_opt_hat"], out["z_opt_target"])
                    + out["vq_opt"]["vq_loss"] + out["vq_act"]["vq_loss"])
        else:
            lam_recon = reconstruction_loss(
                out["lam_logits"], tokenise(future[:, 0:1]), e.vocab_size
            )
            lom_recon = reconstruction_loss(
                out["lom_logits"], tokenise(future[:, -1:]), e.vocab_size
            )
            loss = lam_recon + lom_recon + out["vq_opt"]["vq_loss"] + out["vq_act"]["vq_loss"]
    log.info("    [trace] after loss:            %.2f GB", _mem(device))
    loss.backward()
    log.info("    [trace] after backward:        %.2f GB", _mem(device))
    return loss


def measure_one_batch_size(batch_size: int, method: str, encoder: str,
                           context_len: int, horizon: int,
                           compile_model: bool = False) -> float:
    """Full training loop measurement in a clean process context."""
    import gc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else NullCtx())

    e, models = _build_models(device, method, encoder, context_len, horizon)

    total_params = sum(p.numel() for p in models["model"].parameters())
    trainable_params = sum(p.numel() for p in models["model"].parameters() if p.requires_grad)
    log.info("  params  total=%9.3f M  trainable=%9.3f M",
             total_params / 1e6, trainable_params / 1e6)

    if compile_model:
        log.info("  Compiling models ...")
        models = {k: torch.compile(m, dynamic=False) for k, m in models.items()}

    batch = _make_dummy_batch(batch_size, method, context_len, horizon, e)
    # EMA encoder parameters have requires_grad=False — exclude from optimizer
    optimizer = optim.AdamW(
        [p for m in models.values() for p in m.parameters() if p.requires_grad], lr=3e-4
    )
    t_start = None

    try:
        for s in range(GPU_WARMUP_STEPS + GPU_MEASURE_STEPS):
            if s == 0 and device.type == "cuda" and not compile_model:
                log.info("  --- memory trace through step 0 ---")
                _run_step_traced(method, encoder, models, batch, device, ctx, e)
                optimizer.zero_grad(set_to_none=True)
                continue

            with ctx:
                loss = _run_step(method, encoder, models, batch, device, ctx, e)
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
    finally:
        del optimizer
        for m in models.values():
            if hasattr(m, "zero_grad"):
                m.zero_grad(set_to_none=True)
        gc.collect()


# --------------------------------------------------------------------------- #
# Subprocess worker (fresh CUDA context per batch size)
# --------------------------------------------------------------------------- #

def _worker(batch_size: int, method: str, encoder: str,
            context_len: int, horizon: int,
            compile_model: bool, q: mp.Queue) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sps = measure_one_batch_size(batch_size, method, encoder, context_len, horizon,
                                     compile_model)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        q.put(("ok", sps, peak_gb))
    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        q.put(("oom", None, peak_gb))
    except Exception:
        q.put(("error", traceback.format_exc(), 0.0))


def _spawn(batch_size: int, method: str, encoder: str,
           context_len: int, horizon: int,
           compile_model: bool = False) -> tuple[str, float | None, float]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker,
                       args=(batch_size, method, encoder, context_len, horizon,
                             compile_model, q))
    proc.start()
    proc.join()
    return q.get() if not q.empty() else ("crash", None, 0.0)


# --------------------------------------------------------------------------- #
# Batch-size sweep
# --------------------------------------------------------------------------- #

def sweep_batch_sizes(batch_sizes: list[int], method: str, encoder: str,
                      context_len: int, horizon: int,
                      compile_model: bool = False) -> dict[int, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        log.info("  VRAM: %.1f / %.1f GB free", free / 1e9, total / 1e9)
        if free < total * 0.8:
            log.warning("  GPU %.0f%% occupied — use a free GPU for accurate results",
                        100 * (1 - free / total))

    results: dict[int, float] = {}
    for bs in sorted(batch_sizes):
        log.info("  batch=%6d  ...", bs)
        outcome, value, peak_gb = _spawn(bs, method, encoder, context_len, horizon,
                                         compile_model)
        if outcome == "ok":
            results[bs] = value  # type: ignore[assignment]
            log.info("  batch=%6d  →  %7.0f samp/s  (%.1f steps/s)  peak=%.1f GB",
                     bs, value, value / bs, peak_gb)  # type: ignore[operator]
        elif outcome == "oom":
            log.info("  batch=%6d  →  OOM  (peak before OOM=%.1f GB), stopping", bs, peak_gb)
            break
        else:
            log.info("  batch=%6d  →  error: %s, stopping", bs, value)
            break

    return results


# --------------------------------------------------------------------------- #
# Pareto sweep
# --------------------------------------------------------------------------- #

def pareto_sweep(batch_sizes: list[int], context_lens: list[int],
                 method: str, encoder: str, horizon: int,
                 compile_model: bool = False) -> list[dict]:
    tokens_per_sample = lambda ctx: (ctx + (horizon if method == "lom" else 1)) * 30  # noqa: E731

    log.info("")
    log.info("═" * 70)
    log.info("  Pareto  method=%s  encoder=%s  horizon=%d  compile=%s",
             method, encoder, horizon, compile_model)
    log.info("═" * 70)

    rows = []
    for ctx in context_lens:
        toks = tokens_per_sample(ctx)
        log.info("")
        log.info("  context_len=%d  tokens/sample=%d", ctx, toks)
        results = sweep_batch_sizes(batch_sizes, method, encoder, ctx, horizon, compile_model)
        if results:
            best_bs  = max(results, key=results.__getitem__)
            best_sps = results[best_bs]
            peak_gb  = None  # peak_gb per-bs is already logged; summary uses best
        else:
            best_bs, best_sps = 0, 0.0
        rows.append({"ctx": ctx, "tokens_per_sample": toks,
                     "max_batch": best_bs, "samp_s": best_sps,
                     "all": {str(bs): sps for bs, sps in results.items()}})

    log.info("")
    log.info("  %-10s  %-14s  %-12s  %s", "ctx_len", "tokens/sample", "max_batch", "samp/s")
    log.info("  " + "-" * 52)
    for r in rows:
        log.info("  %-10d  %-14d  %-12d  %.0f",
                 r["ctx"], r["tokens_per_sample"], r["max_batch"], r["samp_s"])
    return rows


# --------------------------------------------------------------------------- #
# Horizon sweep
# --------------------------------------------------------------------------- #

def horizon_sweep(batch_sizes: list[int], horizon_lengths: list[int],
                  method: str, encoder: str, ctx: int,
                  compile_model: bool = False) -> list[dict]:
    log.info("")
    log.info("═" * 70)
    log.info("  Horizon sweep  method=%s  encoder=%s  ctx=%d  compile=%s",
             method, encoder, ctx, compile_model)
    log.info("═" * 70)

    rows = []
    for h in horizon_lengths:
        toks = (ctx + h) * 30
        log.info("")
        log.info("  horizon=%d  tokens/sample=%d", h, toks)
        results = sweep_batch_sizes(batch_sizes, method, encoder, ctx, h, compile_model)
        if results:
            best_bs  = max(results, key=results.__getitem__)
            best_sps = results[best_bs]
        else:
            best_bs, best_sps = 0, 0.0
        rows.append({"horizon": h, "tokens_per_sample": toks,
                     "max_batch": best_bs, "samp_s": best_sps,
                     "all": {str(bs): sps for bs, sps in results.items()}})

    log.info("")
    log.info("  %-10s  %-14s  %-12s  %s", "horizon", "tokens/sample", "max_batch", "samp/s")
    log.info("  " + "-" * 52)
    for r in rows:
        log.info("  %-10d  %-14d  %-12d  %.0f",
                 r["horizon"], r["tokens_per_sample"], r["max_batch"], r["samp_s"])
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method",          choices=["lam", "lom"], default="lam")
    parser.add_argument("--encoder",         choices=["reconstruction", "latent", "latent-medium", "latent-params"], default="reconstruction")
    parser.add_argument("--horizon",         type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--context-len",     type=int, default=DEFAULT_CTX)
    parser.add_argument("--batch-sizes",     type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--pareto",          action="store_true",
                        help="sweep context lengths × batch sizes for Pareto frontier")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64, 128, 256],
                        help="context lengths to sweep with --pareto")
    parser.add_argument("--horizon-sweep",   action="store_true",
                        help="sweep horizon lengths × batch sizes at fixed context length")
    parser.add_argument("--horizon-lengths", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                        help="horizon lengths to sweep with --horizon-sweep")
    parser.add_argument("--compile",         action="store_true",
                        help="apply torch.compile to all models before profiling")
    parser.add_argument("--json-out",        default=None,
                        help="write sweep results to this JSON file")
    args = parser.parse_args()

    log.info("=== profile_memory  method=%s  encoder=%s  horizon=%d  compile=%s ===",
             args.method, args.encoder, args.horizon, args.compile)

    if args.pareto:
        rows = pareto_sweep(args.batch_sizes, args.context_lengths,
                            args.method, args.encoder, args.horizon, args.compile)
        if args.json_out:
            payload = {"method": args.method, "encoder": args.encoder,
                       "horizon": args.horizon, "sweep": "ctx", "rows": rows}
            with open(args.json_out, "w") as f:
                json.dump(payload, f, indent=2)
            log.info("Pareto results written to %s", args.json_out)
    elif args.horizon_sweep:
        rows = horizon_sweep(args.batch_sizes, args.horizon_lengths,
                             args.method, args.encoder, args.context_len, args.compile)
        if args.json_out:
            payload = {"method": args.method, "encoder": args.encoder,
                       "ctx": args.context_len, "sweep": "horizon", "rows": rows}
            with open(args.json_out, "w") as f:
                json.dump(payload, f, indent=2)
            log.info("Horizon sweep results written to %s", args.json_out)
    else:
        log.info("Sweeping batch sizes  (method=%s  encoder=%s  ctx=%d  horizon=%d)",
                 args.method, args.encoder, args.context_len, args.horizon)
        results = sweep_batch_sizes(args.batch_sizes, args.method, args.encoder,
                                    args.context_len, args.horizon, args.compile)
        if results:
            best_bs = max(results, key=results.__getitem__)
            log.info("")
            log.info("  Peak: batch=%d  samp/s=%.0f", best_bs, results[best_bs])


if __name__ == "__main__":
    main()
