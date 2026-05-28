"""Profile GPU memory and training throughput.

Usage:
    # Sweep batch sizes for LAM (horizon=1):
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --model lam

    # Sweep batch sizes for full LOM (horizon=128):
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --model lom --horizon 128

    # Pareto frontier: for each context length, find max batch size:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --pareto --model lam
    CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_memory --pareto --model lom --horizon 128

Each batch size runs in a fresh subprocess to avoid CUDA context corruption
from previous OOM events. Uses synthetic random data — no dataset required.
Logs to stdout and /scratch/uceeepi/lom/profile_memory.log.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time

import torch
import torch.optim as optim

from lom.config import EnvCfg, ModelCfg
from lom.models import DynamicsModel, LatentActionModel
from lom.modules import tokenise
from lom.training import NullCtx, reconstruction_loss

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

DEFAULT_BATCH_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_CTX         = 4
DEFAULT_HORIZON     = 128
SEED                = 42
GPU_WARMUP_STEPS    = 20
GPU_MEASURE_STEPS   = 50


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #

def _build_lam_models(device, context_len: int, two_encoder: bool = False):
    e = EnvCfg()
    m = ModelCfg(d_model=256, n_layers=4, n_heads=4, context_length=context_len,
                 latent_dim=512, num_options=256, patch_size=4)
    lam = LatentActionModel(
        vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w,
        d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
        context_length=m.context_length, latent_dim=m.latent_dim,
        codebook_size=m.num_options, horizon=1, patch_size=m.patch_size,
        two_encoder=two_encoder,
    ).to(device)
    dyn = DynamicsModel(
        vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w,
        d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
        context_length=m.context_length, latent_dim=m.latent_dim,
        patch_size=m.patch_size, predict_sequence=False,
    ).to(device)
    return e, {"lam": lam, "dyn": dyn}


def _build_lom_models(device, context_len: int, horizon: int, two_encoder: bool = False):
    from lom.config import LOMModelCfg
    e = EnvCfg()
    m = LOMModelCfg(d_model=256, n_layers=4, n_heads=4, context_length=context_len,
                    latent_dim=512, num_options=256, patch_size=4)
    base = dict(vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w,
                d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
                context_length=m.context_length, latent_dim=m.latent_dim,
                patch_size=m.patch_size)
    vq = dict(codebook_size=m.num_options)
    models = {
        "option_lam":   LatentActionModel(**base, **vq, horizon=horizon,
                                          two_encoder=two_encoder).to(device),
        "action_lam":   LatentActionModel(**base, **vq, horizon=1,
                                          condition_dim=m.latent_dim,
                                          two_encoder=two_encoder).to(device),
        "lam_dynamics": DynamicsModel(**base, predict_sequence=False).to(device),
        "lom_dynamics": DynamicsModel(**base, option_dim=m.latent_dim,
                                      predict_sequence=False, horizon=horizon).to(device),
    }
    return e, models


# --------------------------------------------------------------------------- #
# Training step
# --------------------------------------------------------------------------- #

def _run_step(model_type: str, models: dict, batch: list, device, ctx, e) -> torch.Tensor:
    if model_type == "lam":
        history, next_frame = batch[0].to(device), batch[1].to(device)
        z, vq_out, _ = models["lam"](history, next_frame)
        logits = models["dyn"](history, z)
        return reconstruction_loss(logits, tokenise(next_frame), e.vocab_size) + vq_out["vq_loss"]
    else:
        history    = batch[0].to(device)
        next_frame = batch[1].to(device)
        future     = batch[2].to(device)
        sequence   = batch[3].to(device)
        z_opt, vq_opt, _ = models["option_lam"](history, sequence)
        z_act, vq_act, _ = models["action_lam"](history, next_frame, z_opt.detach())
        lam_logits = models["lam_dynamics"](history, z_act)
        lom_logits = models["lom_dynamics"](history, z_act, option_code=z_opt, horizon=1)
        lam_recon  = reconstruction_loss(lam_logits, tokenise(next_frame), e.vocab_size)
        lom_recon  = reconstruction_loss(lom_logits, tokenise(future), e.vocab_size)
        return lam_recon + lom_recon + vq_opt["vq_loss"] + vq_act["vq_loss"]


def _make_frame(shape: tuple, rng: torch.Generator) -> torch.Tensor:
    """Random frame with valid token ranges: char in [0,256), color in [0,32)."""
    f = torch.empty(shape, dtype=torch.uint8)
    f[..., 0] = torch.randint(0, 256, shape[:-1], generator=rng)
    f[..., 1] = torch.randint(0,  32, shape[:-1], generator=rng)
    return f


def _make_dummy_batch(batch_size: int, model_type: str,
                      context_len: int, horizon: int, e) -> list[torch.Tensor]:
    """Synthetic uint8 frames — same shapes as the real dataloader, no disk I/O."""
    H, W = e.obs_h, e.obs_w
    rng = torch.Generator()
    rng.manual_seed(SEED)
    history    = _make_frame((batch_size, context_len, H, W, 2), rng)
    next_frame = _make_frame((batch_size, 1, H, W, 2), rng)
    if model_type == "lom":
        future   = _make_frame((batch_size, 1, H, W, 2), rng)
        sequence = _make_frame((batch_size, horizon, H, W, 2), rng)
        return [history, next_frame, future, sequence]
    return [history, next_frame]


def measure_one_batch_size(batch_size: int, model_type: str,
                           context_len: int, horizon: int,
                           compile_model: bool = False,
                           two_encoder: bool = False) -> float:
    """Full training loop measurement in a clean process context."""
    import gc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else NullCtx())

    if model_type == "lam":
        e, models = _build_lam_models(device, context_len, two_encoder)
    else:
        e, models = _build_lom_models(device, context_len, horizon, two_encoder)

    if compile_model:
        log.info("  Compiling models ...")
        models = {k: torch.compile(m) for k, m in models.items()}

    batch = _make_dummy_batch(batch_size, model_type, context_len, horizon, e)
    optimizer = optim.AdamW([p for m in models.values() for p in m.parameters()], lr=3e-4)
    t_start = None

    try:
        for s in range(GPU_WARMUP_STEPS + GPU_MEASURE_STEPS):
            with ctx:
                loss = _run_step(model_type, models, batch, device, ctx, e)
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
            m.zero_grad(set_to_none=True)
        gc.collect()


# --------------------------------------------------------------------------- #
# Subprocess worker (fresh CUDA context per batch size)
# --------------------------------------------------------------------------- #

def _worker(batch_size: int, model_type: str,
            context_len: int, horizon: int,
            compile_model: bool, two_encoder: bool, q: mp.Queue) -> None:
    try:
        sps = measure_one_batch_size(batch_size, model_type, context_len, horizon,
                                     compile_model, two_encoder)
        q.put(("ok", sps))
    except torch.cuda.OutOfMemoryError:
        q.put(("oom", None))
    except Exception as exc:
        q.put(("error", str(exc)))


def _spawn(batch_size: int, model_type: str,
           context_len: int, horizon: int,
           compile_model: bool = False,
           two_encoder: bool = False) -> tuple[str, float | None]:
    """Run one measurement in a subprocess; return (outcome, samp/s)."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker,
                       args=(batch_size, model_type, context_len, horizon,
                             compile_model, two_encoder, q))
    proc.start()
    proc.join()
    return q.get() if not q.empty() else ("crash", None)


# --------------------------------------------------------------------------- #
# Batch-size sweep
# --------------------------------------------------------------------------- #

def sweep_batch_sizes(batch_sizes: list[int],
                      model_type: str, context_len: int, horizon: int,
                      compile_model: bool = False,
                      two_encoder: bool = False) -> dict[int, float]:
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
        outcome, value = _spawn(bs, model_type, context_len, horizon, compile_model, two_encoder)
        if outcome == "ok":
            results[bs] = value  # type: ignore[assignment]
            log.info("  batch=%6d  →  %7.0f samp/s  (%.1f steps/s)",
                     bs, value, value / bs)  # type: ignore[operator]
        elif outcome == "oom":
            log.info("  batch=%6d  →  OOM, stopping", bs)
            break
        else:
            log.info("  batch=%6d  →  error: %s, stopping", bs, value)
            break

    return results


# --------------------------------------------------------------------------- #
# Pareto sweep
# --------------------------------------------------------------------------- #

def pareto_sweep(batch_sizes: list[int], context_lens: list[int],
                 model_type: str, horizon: int,
                 compile_model: bool = False,
                 two_encoder: bool = False) -> None:
    tokens_per_sample = lambda ctx: (ctx + (horizon if model_type == "lom" else 1)) * 120  # noqa: E731

    log.info("")
    log.info("═" * 70)
    log.info("  Pareto  model=%s  horizon=%d  compile=%s  two_encoder=%s",
             model_type, horizon, compile_model, two_encoder)
    log.info("═" * 70)

    rows = []
    for ctx in context_lens:
        toks = tokens_per_sample(ctx)
        log.info("")
        log.info("  context_len=%d  tokens/sample=%d", ctx, toks)
        results = sweep_batch_sizes(batch_sizes, model_type, ctx, horizon,
                                    compile_model, two_encoder)
        if results:
            best_bs  = max(results, key=results.__getitem__)
            best_sps = results[best_bs]
        else:
            best_bs, best_sps = 0, 0.0
        rows.append((ctx, toks, best_bs, best_sps))

    log.info("")
    log.info("  %-10s  %-14s  %-12s  %s", "ctx_len", "tokens/sample", "max_batch", "samp/s")
    log.info("  " + "-" * 52)
    for ctx, toks, bs, sps in rows:
        log.info("  %-10d  %-14d  %-12d  %.0f", ctx, toks, bs, sps)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",           choices=["lam", "lom"], default="lam")
    parser.add_argument("--horizon",         type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--context-len",     type=int, default=DEFAULT_CTX)
    parser.add_argument("--batch-sizes",     type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--pareto",          action="store_true",
                        help="sweep context lengths × batch sizes for Pareto frontier")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64, 128, 256],
                        help="context lengths to sweep with --pareto")
    parser.add_argument("--compile", action="store_true",
                        help="apply torch.compile to all models before profiling")
    parser.add_argument("--two-encoder", action="store_true",
                        help="use TwoPassEncoder instead of BidirectionalEncoder")
    args = parser.parse_args()

    log.info("=== profile_memory  model=%s  horizon=%d  compile=%s  two_encoder=%s ===",
             args.model, args.horizon, args.compile, args.two_encoder)

    if args.pareto:
        pareto_sweep(args.batch_sizes, args.context_lengths, args.model, args.horizon,
                     args.compile, args.two_encoder)
    else:
        log.info("Sweeping batch sizes  (model=%s  ctx=%d  horizon=%d)",
                 args.model, args.context_len, args.horizon)
        results = sweep_batch_sizes(args.batch_sizes, args.model, args.context_len, args.horizon,
                                    args.compile, args.two_encoder)
        if results:
            best_bs = max(results, key=results.__getitem__)
            log.info("")
            log.info("  Peak: batch=%d  samp/s=%.0f", best_bs, results[best_bs])


if __name__ == "__main__":
    main()
