from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim

from .config import LAMCfg, LOMCfg
from .models import ReconstructionLOM, LatentLOM
from .dataset import build_npz_dataloaders
from .modules import DynamicsModel, EMAEncoder, LatentActionModel
from .modules import tokenise

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# --- Helpers --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def reconstruction_loss(
    logits: torch.Tensor, target: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1).long(),
    )


def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine distance between predicted and target latents, ∈ [0, 2].

    target is detached so gradients flow only through pred.
    """
    pred   = torch.nn.functional.normalize(pred,            dim=-1)
    target = torch.nn.functional.normalize(target.detach(), dim=-1)
    return (1 - (pred * target).sum(dim=-1)).mean()


def get_lr(step: int, lr: float, warmup_iters: int, max_iters: int, eta_min: float) -> float:
    if step < warmup_iters:
        return lr * step / warmup_iters
    frac = (step - warmup_iters) / (max_iters - warmup_iters)
    return eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * frac))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
    except Exception:
        return False


class NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# --------------------------------------------------------------------------- #
# --- Base Trainer ---------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class Trainer(ABC):
    """Shared scaffolding: data loading, optimiser, AMP, training loop, logging, checkpointing."""

    def __init__(self, cfg: LAMCfg | LOMCfg) -> None:
        self.cfg = cfg
        t, d, m = cfg.train, cfg.data, cfg.model

        if d.context_len > m.context_length:
            raise ValueError(f"context_len={d.context_len} exceeds context_length={m.context_length}")

        torch.manual_seed(t.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("=== %s  seed=%d  device=%s ===", self.label().upper(), t.seed, self.device)
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            log.info("GPU: %s  (%.1f GB total)",
                     torch.cuda.get_device_name(idx),
                     torch.cuda.get_device_properties(idx).total_memory / 1e9)

        if not d.dataset_dir:
            raise ValueError(
                "data.dataset_dir is required. "
                "Run scripts/prepare_data.py to generate an index.npz, then set "
                "data.dataset_dir in your experiment config."
            )
        index_path = os.path.join(d.dataset_dir, "index.npz")
        log.info("Loading dataset from %s  (context=%d  horizon=%d  buffer=%d)",
                 index_path, d.context_len, d.horizon, d.buffer_size)
        self.train_loader, self.val_loader = build_npz_dataloaders(
            index_path=index_path,
            context_len=d.context_len,
            horizon=d.horizon,
            batch_size=t.batch_size,
            buffer_size=d.buffer_size,
            val_fraction=d.val_fraction,
            steps_per_epoch=d.steps_per_epoch,
            seed=t.seed,
            return_sequence=isinstance(cfg, LOMCfg),
        )
        log.info("Dataloaders ready  (train=%d steps/epoch  val=%d  batch=%d)",
                 len(self.train_loader), len(self.val_loader), t.batch_size)

        log.info("Building models ...")
        self.models = self.build_models().to(self.device)
        n_params = sum(p.numel() for p in self.models.parameters())
        n_trainable = sum(p.numel() for p in self.models.parameters() if p.requires_grad)
        log.info("Models: %s  |  params=%s  trainable=%s",
                 ", ".join(self.models.keys()),
                 f"{n_params:,}", f"{n_trainable:,}")

        if t.compile_model:
            log.info("Compiling models with torch.compile ...")
            for key in list(self.models.keys()):
                self.models[key] = torch.compile(self.models[key])
            log.info("Compilation done.")

        decay = [p for p in self.models.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in self.models.parameters() if p.requires_grad and p.dim() < 2]
        self.optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": t.weight_decay},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=t.lr,
            betas=(t.beta1, t.beta2),
            fused=("cuda" in str(self.device) and hasattr(optim.AdamW, "fused")),
        )
        log.info("Optimiser: AdamW  lr=%.2e  wd=%.2e  warmup=%d  max_iters=%d",
                 t.lr, t.weight_decay, t.warmup_iters, t.max_iters)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        amp_dtype = dtype_map.get(t.mixed_dtype, torch.float16)
        self.ctx = (
            torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype)
            if "cpu" not in str(self.device)
            else NullCtx()
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(amp_dtype == torch.float16 and "cuda" in str(self.device))
        )
        log.info("AMP dtype=%s  grad_scaler=%s", t.mixed_dtype,
                 "enabled" if self.scaler.is_enabled() else "disabled")

        os.makedirs(t.ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(t.ckpt_dir, f"{self.label()}_pretrain.pt")
        log.info("Checkpoint path: %s", self.ckpt_path)

        self.wandb_run = None
        try:
            import wandb

            os.makedirs(cfg.wandb.dir, exist_ok=True)

            run_cfg = asdict(cfg)
            run_cfg["_git_commit"] = _git_commit()
            run_cfg["_git_dirty"]  = _git_dirty()
            run_cfg["_argv"]       = " ".join(sys.argv)
            run_cfg["_model_type"] = self.label()

            run_name = (
                f"{self.label()}"
                f"_d{os.path.basename(d.dataset_dir)}"
                f"_h{d.horizon}"
                f"_s{t.seed}"
            )

            log.info("Initialising WandB run '%s' (project=%s group=%s) ...",
                     run_name, cfg.wandb.project, cfg.wandb.group)
            self.wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                group=cfg.wandb.group,
                name=run_name,
                dir=cfg.wandb.dir,
                config=run_cfg,
                resume="allow" if t.resume else "never",
            )
            log.info("WandB run URL: %s", self.wandb_run.url)
        except Exception as exc:
            log.warning("WandB init failed: %s", exc)

        self.start_step = self.restore_checkpoint() if t.resume else 0
        log.info("Starting from step %d", self.start_step)

    def label(self) -> str:
        return "lam" if isinstance(self.cfg, LAMCfg) else "lom"

    def save_checkpoint(self, step: int) -> None:
        torch.save(
            {
                "models": self.models.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "config": asdict(self.cfg),
            },
            self.ckpt_path,
        )
        log.info("Checkpoint saved to %s", self.ckpt_path)
        if self.wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(
                name=f"{self.label()}_checkpoint",
                type="checkpoint",
                metadata={"step": step},
            )
            artifact.add_file(self.ckpt_path)
            self.wandb_run.log_artifact(artifact, aliases=["latest", f"step-{step}"])
            log.info("Checkpoint uploaded to wandb (step %d)", step)

    def restore_checkpoint(self) -> int:
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.models.load_state_dict(ckpt["models"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            log.info("Resumed from local checkpoint (step %d)", ckpt["step"])
            return ckpt["step"]
        if self.wandb_run is not None:
            try:
                import wandb
                artifact = self.wandb_run.use_artifact(f"{self.label()}_checkpoint:latest")
                artifact_dir = artifact.download()
                path = os.path.join(artifact_dir, os.path.basename(self.ckpt_path))
                ckpt = torch.load(path, map_location=self.device)
                self.models.load_state_dict(ckpt["models"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                log.info("Resumed from wandb artifact (step %d)", ckpt["step"])
                return ckpt["step"]
            except Exception as exc:
                log.warning("WandB restore failed: %s", exc)
        return 0

    @abstractmethod
    def build_models(self) -> nn.ModuleDict: ...

    @abstractmethod
    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]: ...

    @torch.no_grad()
    def eval(self) -> dict[str, float]:
        for mod in self.models.values():
            mod.eval()
        totals: dict = {}
        i = -1
        for i, batch in enumerate(self.val_loader):
            if i >= self.cfg.train.eval_iters:
                break
            batch = [x.to(self.device) for x in batch]
            with self.ctx:
                for k, v in self.step(batch).items():
                    totals[k] = totals.get(k, 0.0) + v.item()
        for mod in self.models.values():
            mod.train()
        return {k: v / (i + 1) for k, v in totals.items()}

    def train(self) -> None:
        t = self.cfg.train
        log.info("--- training start  steps=%d  log_every=%d  eval_every=%d ---",
                 t.max_iters, t.log_interval, t.eval_interval)

        for mod in self.models.values():
            mod.train()
        data_iter = iter(self.train_loader)
        t0 = time.time()

        for s in range(self.start_step, t.max_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = [x.to(self.device) for x in batch]
            lr = get_lr(s, t.lr, t.warmup_iters, t.max_iters, t.eta_min)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with self.ctx:
                loss_dict = self.step(batch)

            self.scaler.scale(loss_dict["total_loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), t.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if (s + 1) % t.log_interval == 0:
                dt = time.time() - t0
                sps = t.log_interval * t.batch_size / dt
                pct = 100.0 * (s + 1) / t.max_iters
                log.info(
                    "step %6d/%d (%4.1f%%) | loss=%.4f | %s | lr=%.2e | %.0f samp/s",
                    s + 1, t.max_iters, pct,
                    loss_dict["total_loss"].item(),
                    "  ".join(
                        f"{k}={v.item():.4f}" for k, v in loss_dict.items() if k != "total_loss"
                    ),
                    lr,
                    sps,
                )
                t0 = time.time()
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"train/{k}": v.item() for k, v in loss_dict.items()} | {"lr": lr},
                        step=s + 1,
                    )

            if (s + 1) % t.eval_interval == 0:
                log.info("  [eval] running %d val batches ...", t.eval_iters)
                val_metrics = self.eval()
                log.info("  [val]  %s", "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                if self.wandb_run:
                    self.wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=s + 1)
                self.save_checkpoint(s + 1)

        log.info("=== training complete (%d steps) ===", t.max_iters)
        if self.wandb_run:
            self.wandb_run.finish()


# --------------------------------------------------------------------------- #
# --- LAM Trainer ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class LAMTrainer(Trainer):

    def build_models(self) -> nn.ModuleDict:
        e, m = self.cfg.env, self.cfg.model
        base = dict(
            vocab_size=e.vocab_size,
            obs_h=e.obs_h,
            obs_w=e.obs_w,
            d_model=m.d_model,
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            context_length=m.context_length,
            latent_dim=m.latent_dim,
            patch_size=m.patch_size,
            dropout=m.dropout,
            bias=m.bias,
        )
        vq = dict(
            vq_dropout=m.vq_dropout,
            vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta,
            vq_reset_thresh=m.vq_reset_thresh,
            vq_ema_decay=m.vq_ema_decay,
        )
        return nn.ModuleDict(
            {
                "lam": LatentActionModel(
                    **base, codebook_size=m.num_options, horizon=1,
                    two_encoder=m.two_encoder, **vq
                ),
                "dynamics": DynamicsModel(**base, predict_sequence=False),
            }
        )

    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        history, next_frame = batch[0], batch[1]
        z, vq, _ = self.models["lam"](history, next_frame)
        logits = self.models["dynamics"](history, z)
        recon = reconstruction_loss(logits, tokenise(next_frame), self.cfg.env.vocab_size)
        total = recon + vq["vq_loss"]
        return {
            "recon": recon,
            "vq_loss": vq["vq_loss"],
            "commit_loss": vq["commit_loss"],
            "entropy": vq["entropy"],
            "total_loss": total,
        }


# --------------------------------------------------------------------------- #
# --- LOM Trainer ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class LOMTrainer(Trainer):

    def build_models(self) -> nn.ModuleDict:
        e, m, d = self.cfg.env, self.cfg.model, self.cfg.data
        base = dict(
            vocab_size=e.vocab_size,
            obs_h=e.obs_h,
            obs_w=e.obs_w,
            d_model=m.d_model,
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            context_length=m.context_length,
            latent_dim=m.latent_dim,
            patch_size=m.patch_size,
            dropout=m.dropout,
            bias=m.bias,
        )
        vq = dict(
            vq_dropout=m.vq_dropout,
            vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta,
            vq_reset_thresh=m.vq_reset_thresh,
            vq_ema_decay=m.vq_ema_decay,
        )
        return nn.ModuleDict(
            {
                "option_lam": LatentActionModel(
                    **base, codebook_size=m.num_options, horizon=d.horizon,
                    two_encoder=m.two_encoder, **vq
                ),
                "action_lam": LatentActionModel(
                    **base,
                    codebook_size=e.n_actions,
                    horizon=1,
                    condition_dim=m.latent_dim,
                    two_encoder=m.two_encoder,
                    **vq,
                ),
                "lam_dynamics": DynamicsModel(**base, predict_sequence=False),
                "lom_dynamics": DynamicsModel(
                    **base,
                    option_dim=m.latent_dim,
                    predict_sequence=m.predict_sequence,
                    horizon=d.horizon,
                ),
            }
        )

    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        history, next_frame, future_frame, sequence = batch[0], batch[1], batch[2], batch[3]
        m = self.cfg.model

        z_opt, vq_opt, _ = self.models["option_lam"](history, sequence)
        z_act, vq_act, _ = self.models["action_lam"](history, next_frame, z_opt.detach())

        lam_logits = self.models["lam_dynamics"](history, z_act)
        lam_recon = reconstruction_loss(lam_logits, tokenise(next_frame), self.cfg.env.vocab_size)

        if m.predict_sequence:
            lom_logits = self.models["lom_dynamics"](
                history,
                z_act,
                option_code=z_opt,
                horizon=self.cfg.data.horizon,
                teacher_frames=sequence,
            )
            lom_recon = reconstruction_loss(lom_logits, tokenise(sequence), self.cfg.env.vocab_size)
        else:
            lom_logits = self.models["lom_dynamics"](
                history,
                z_act,
                option_code=z_opt,
                horizon=1,
            )
            lom_recon = reconstruction_loss(lom_logits, tokenise(future_frame), self.cfg.env.vocab_size)

        vq_loss = vq_opt["vq_loss"] + vq_act["vq_loss"]
        total = lam_recon + lom_recon + vq_loss
        return {
            "lam_recon": lam_recon,
            "lom_recon": lom_recon,
            "vq_loss_option": vq_opt["vq_loss"],
            "vq_loss_action": vq_act["vq_loss"],
            "commit_loss_option": vq_opt["commit_loss"],
            "commit_loss_action": vq_act["commit_loss"],
            "entropy_option": vq_opt["entropy"],
            "entropy_action": vq_act["entropy"],
            "total_loss": total,
        }


# --------------------------------------------------------------------------- #
# --- JEPA LAM Trainer ------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class JEPALAMTrainer(Trainer):
    """LAM trainer with JEPA latent prediction loss instead of pixel reconstruction.

    DynamicsModel predicts the EMA target encoder's representation of the future
    (latent space loss). No cross-entropy over frame tokens.
    """

    def build_models(self) -> nn.ModuleDict:
        e, m = self.cfg.env, self.cfg.model
        base = dict(
            vocab_size=e.vocab_size,
            obs_h=e.obs_h,
            obs_w=e.obs_w,
            d_model=m.d_model,
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            context_length=m.context_length,
            latent_dim=m.latent_dim,
            patch_size=m.patch_size,
            dropout=m.dropout,
            bias=m.bias,
        )
        vq = dict(
            vq_dropout=m.vq_dropout,
            vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta,
            vq_reset_thresh=m.vq_reset_thresh,
            vq_ema_decay=m.vq_ema_decay,
        )
        lam = LatentActionModel(
            **base, codebook_size=m.num_options, horizon=1,
            two_encoder=True, **vq,
        )
        dynamics = DynamicsModel(
            **base,
            predict_sequence=False,
            predict_latent=True,
            target_dim=m.latent_dim,
        )
        ema_enc = EMAEncoder(lam.encoder, decay=m.ema_decay)  # type: ignore[arg-type]
        return nn.ModuleDict({"lam": lam, "dynamics": dynamics, "ema_enc": ema_enc})

    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        history, next_frame = batch[0], batch[1]
        z_q, vq, _ = self.models["lam"](history, next_frame)
        z_target = self.models["ema_enc"].encode(next_frame)
        z_hat = self.models["dynamics"](history, z_q)
        jepa = jepa_loss(z_hat, z_target)
        total = jepa + vq["vq_loss"]
        return {
            "jepa_loss": jepa,
            "vq_loss": vq["vq_loss"],
            "commit_loss": vq["commit_loss"],
            "entropy": vq["entropy"],
            "total_loss": total,
        }

    def train(self) -> None:
        """Extends base train loop with EMA update after each optimizer step."""
        t = self.cfg.train
        log.info("--- JEPA training start  steps=%d  ema_decay=%.4f ---",
                 t.max_iters, self.cfg.model.ema_decay)

        for mod in self.models.values():
            mod.train()
        # EMA encoder stays in eval mode — parameters are not trained
        self.models["ema_enc"].eval()

        data_iter = iter(self.train_loader)
        t0 = time.time()

        for s in range(self.start_step, t.max_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = [x.to(self.device) for x in batch]
            lr = get_lr(s, t.lr, t.warmup_iters, t.max_iters, t.eta_min)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with self.ctx:
                loss_dict = self.step(batch)

            self.scaler.scale(loss_dict["total_loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), t.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # EMA update: shadow the online JEPAEncoder after every optimizer step
            self.models["ema_enc"].update(self.models["lam"].encoder)

            if (s + 1) % t.log_interval == 0:
                dt = time.time() - t0
                sps = t.log_interval * t.batch_size / dt
                pct = 100.0 * (s + 1) / t.max_iters
                log.info(
                    "step %6d/%d (%4.1f%%) | loss=%.4f | %s | lr=%.2e | %.0f samp/s",
                    s + 1, t.max_iters, pct,
                    loss_dict["total_loss"].item(),
                    "  ".join(
                        f"{k}={v.item():.4f}" for k, v in loss_dict.items() if k != "total_loss"
                    ),
                    lr,
                    sps,
                )
                t0 = time.time()
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"train/{k}": v.item() for k, v in loss_dict.items()} | {"lr": lr},
                        step=s + 1,
                    )

            if (s + 1) % t.eval_interval == 0:
                log.info("  [eval] running %d val batches ...", t.eval_iters)
                val_metrics = self.eval()
                log.info("  [val]  %s", "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                if self.wandb_run:
                    self.wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=s + 1)
                self.save_checkpoint(s + 1)

        log.info("=== JEPA training complete (%d steps) ===", t.max_iters)
        if self.wandb_run:
            self.wandb_run.finish()


# --------------------------------------------------------------------------- #
# --- JEPA LOM Trainer ------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class JEPALOMTrainer(Trainer):
    """LOM trainer with JEPA latent prediction for both dynamics models.

    Architecture (4 pure encoders, no shared weights):
    - option_lam:      JEPAEncoder; (history, sequence) → z_opt via LOM VQ
    - action_lam:      JEPAEncoder; (history, next_frame) + z_opt at VQ level → z_act
    - ema_option_enc:  EMA of option_lam.encoder.future_encoder; sequence → z_opt_target
    - ema_action_enc:  EMA of action_lam.encoder.future_encoder; next_frame → z_act_target
    - lam_dynamics:    (history, z_act) → predict z_act_target
    - lom_dynamics:    (history, z_opt) → predict z_opt_target  [z_act NOT used]
    """

    def build_models(self) -> nn.ModuleDict:
        e, m, d = self.cfg.env, self.cfg.model, self.cfg.data
        base = dict(
            vocab_size=e.vocab_size,
            obs_h=e.obs_h,
            obs_w=e.obs_w,
            d_model=m.d_model,
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            context_length=m.context_length,
            latent_dim=m.latent_dim,
            patch_size=m.patch_size,
            dropout=m.dropout,
            bias=m.bias,
        )
        vq = dict(
            vq_dropout=m.vq_dropout,
            vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta,
            vq_reset_thresh=m.vq_reset_thresh,
            vq_ema_decay=m.vq_ema_decay,
        )
        option_lam = LatentActionModel(
            **base, codebook_size=m.num_options, horizon=d.horizon,
            two_encoder=True, **vq,
        )
        action_lam = LatentActionModel(
            **base, codebook_size=e.n_actions, horizon=1,
            option_code_dim=m.latent_dim, two_encoder=True, **vq,
        )
        lam_dynamics = DynamicsModel(
            **base, predict_sequence=False,
            predict_latent=True, target_dim=m.latent_dim,
        )
        lom_dynamics = DynamicsModel(
            **base, predict_sequence=False,
            predict_latent=True, target_dim=m.latent_dim,
        )
        return nn.ModuleDict({
            "option_lam":     option_lam,
            "action_lam":     action_lam,
            "lam_dynamics":   lam_dynamics,
            "lom_dynamics":   lom_dynamics,
            "ema_action_enc": EMAEncoder(action_lam.encoder, decay=m.ema_decay),  # type: ignore[arg-type]
            "ema_option_enc": EMAEncoder(option_lam.encoder, decay=m.ema_decay),  # type: ignore[arg-type]
        })

    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        history, next_frame, _, sequence = batch[0], batch[1], batch[2], batch[3]

        z_opt, vq_opt, _ = self.models["option_lam"](history, sequence)
        z_act, vq_act, _ = self.models["action_lam"](history, next_frame,
                                                      option_code=z_opt.detach())

        with torch.no_grad():
            z_act_target = self.models["ema_action_enc"].encode(next_frame)
            z_opt_target = self.models["ema_option_enc"].encode(sequence)

        z_act_hat = self.models["lam_dynamics"](history, z_act)
        z_opt_hat = self.models["lom_dynamics"](history, z_opt)  # z_act not used

        lam_jepa = jepa_loss(z_act_hat, z_act_target)
        lom_jepa = jepa_loss(z_opt_hat, z_opt_target)
        total = lam_jepa + lom_jepa + vq_opt["vq_loss"] + vq_act["vq_loss"]
        return {
            "lam_jepa_loss":      lam_jepa,
            "lom_jepa_loss":      lom_jepa,
            "vq_loss_option":     vq_opt["vq_loss"],
            "vq_loss_action":     vq_act["vq_loss"],
            "commit_loss_option": vq_opt["commit_loss"],
            "commit_loss_action": vq_act["commit_loss"],
            "entropy_option":     vq_opt["entropy"],
            "entropy_action":     vq_act["entropy"],
            "total_loss":         total,
        }

    def train(self) -> None:
        """Extends base train loop with EMA updates for both encoders."""
        t = self.cfg.train
        log.info("--- JEPA-LOM training start  steps=%d  ema_decay=%.4f ---",
                 t.max_iters, self.cfg.model.ema_decay)

        for mod in self.models.values():
            mod.train()
        self.models["ema_action_enc"].eval()
        self.models["ema_option_enc"].eval()

        data_iter = iter(self.train_loader)
        t0 = time.time()

        for s in range(self.start_step, t.max_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = [x.to(self.device) for x in batch]
            lr = get_lr(s, t.lr, t.warmup_iters, t.max_iters, t.eta_min)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with self.ctx:
                loss_dict = self.step(batch)

            self.scaler.scale(loss_dict["total_loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), t.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.models["ema_action_enc"].update(self.models["action_lam"].encoder)
            self.models["ema_option_enc"].update(self.models["option_lam"].encoder)

            if (s + 1) % t.log_interval == 0:
                dt = time.time() - t0
                sps = t.log_interval * t.batch_size / dt
                pct = 100.0 * (s + 1) / t.max_iters
                log.info(
                    "step %6d/%d (%4.1f%%) | loss=%.4f | %s | lr=%.2e | %.0f samp/s",
                    s + 1, t.max_iters, pct,
                    loss_dict["total_loss"].item(),
                    "  ".join(
                        f"{k}={v.item():.4f}" for k, v in loss_dict.items() if k != "total_loss"
                    ),
                    lr,
                    sps,
                )
                t0 = time.time()
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"train/{k}": v.item() for k, v in loss_dict.items()} | {"lr": lr},
                        step=s + 1,
                    )

            if (s + 1) % t.eval_interval == 0:
                log.info("  [eval] running %d val batches ...", t.eval_iters)
                val_metrics = self.eval()
                log.info("  [val]  %s", "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                if self.wandb_run:
                    self.wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=s + 1)
                self.save_checkpoint(s + 1)

        log.info("=== JEPA-LOM training complete (%d steps) ===", t.max_iters)
        if self.wandb_run:
            self.wandb_run.finish()


# --------------------------------------------------------------------------- #
# --- ReconstructionLOM Trainer ---------------------------------------------- #
# --------------------------------------------------------------------------- #


class ReconstructionLOMTrainer(Trainer):
    """Trainer for ReconstructionLOM (STT encoders + pixel reconstruction dynamics)."""

    def build_models(self) -> nn.ModuleDict:
        e, m, d = self.cfg.env, self.cfg.model, self.cfg.data
        model = ReconstructionLOM(
            vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w, n_actions=e.n_actions,
            d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
            context_length=m.context_length, horizon=d.horizon,
            latent_dim=m.latent_dim, num_options=m.num_options,
            patch_size=m.patch_size, dropout=m.dropout, bias=m.bias,
            predict_sequence=m.predict_sequence,
            vq_dropout=m.vq_dropout, vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta, vq_reset_thresh=m.vq_reset_thresh, vq_ema_decay=m.vq_ema_decay,
        )
        return nn.ModuleDict({"model": model})

    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        history, _, _, future = batch[0], batch[1], batch[2], batch[3]
        out = self.models["model"](history, future)
        lam_recon = reconstruction_loss(out["lam_logits"], tokenise(future[:, 0:1]), self.cfg.env.vocab_size)
        if self.cfg.model.predict_sequence:
            lom_recon = reconstruction_loss(out["lom_logits"], tokenise(future), self.cfg.env.vocab_size)
        else:
            lom_recon = reconstruction_loss(out["lom_logits"], tokenise(future[:, -1:]), self.cfg.env.vocab_size)
        vq_loss = out["vq_opt"]["vq_loss"] + out["vq_act"]["vq_loss"]
        total = lam_recon + lom_recon + vq_loss
        return {
            "lam_recon": lam_recon,
            "lom_recon": lom_recon,
            "vq_loss_option": out["vq_opt"]["vq_loss"],
            "vq_loss_action": out["vq_act"]["vq_loss"],
            "commit_loss_option": out["vq_opt"]["commit_loss"],
            "commit_loss_action": out["vq_act"]["commit_loss"],
            "entropy_option": out["vq_opt"]["entropy"],
            "entropy_action": out["vq_act"]["entropy"],
            "total_loss": total,
        }


# --------------------------------------------------------------------------- #
# --- LatentLOM Trainer ------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class LatentLOMTrainer(Trainer):
    """Trainer for LatentLOM (JEPA encoders + latent dynamics).

    Extends the base train loop with EMA updates after each optimiser step.
    """

    def build_models(self) -> nn.ModuleDict:
        e, m, d = self.cfg.env, self.cfg.model, self.cfg.data
        model = LatentLOM(
            vocab_size=e.vocab_size, obs_h=e.obs_h, obs_w=e.obs_w, n_actions=e.n_actions,
            d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
            context_length=m.context_length, horizon=d.horizon,
            latent_dim=m.latent_dim, num_options=m.num_options,
            patch_size=m.patch_size, dropout=m.dropout, bias=m.bias,
            ema_decay=m.ema_decay,
            vq_dropout=m.vq_dropout, vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta, vq_reset_thresh=m.vq_reset_thresh, vq_ema_decay=m.vq_ema_decay,
        )
        return nn.ModuleDict({"model": model})

    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        history, _, _, future = batch[0], batch[1], batch[2], batch[3]
        out = self.models["model"](history, future)
        lam_jepa = jepa_loss(out["z_act_hat"], out["z_act_target"])
        lom_jepa = jepa_loss(out["z_opt_hat"], out["z_opt_target"])
        vq_loss = out["vq_opt"]["vq_loss"] + out["vq_act"]["vq_loss"]
        total = lam_jepa + lom_jepa + vq_loss
        return {
            "lam_jepa_loss": lam_jepa,
            "lom_jepa_loss": lom_jepa,
            "vq_loss_option": out["vq_opt"]["vq_loss"],
            "vq_loss_action": out["vq_act"]["vq_loss"],
            "commit_loss_option": out["vq_opt"]["commit_loss"],
            "commit_loss_action": out["vq_act"]["commit_loss"],
            "entropy_option": out["vq_opt"]["entropy"],
            "entropy_action": out["vq_act"]["entropy"],
            "total_loss": total,
        }

    def train(self) -> None:
        t = self.cfg.train
        log.info("--- LatentLOM training start  steps=%d  ema_decay=%.4f ---",
                 t.max_iters, self.cfg.model.ema_decay)

        for mod in self.models.values():
            mod.train()
        # EMA encoders stay in eval mode — not trained by optimizer
        self.models["model"].ema_option_enc.eval()
        self.models["model"].ema_action_enc.eval()

        data_iter = iter(self.train_loader)
        t0 = time.time()

        for s in range(self.start_step, t.max_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = [x.to(self.device) for x in batch]
            lr = get_lr(s, t.lr, t.warmup_iters, t.max_iters, t.eta_min)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with self.ctx:
                loss_dict = self.step(batch)

            self.scaler.scale(loss_dict["total_loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), t.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.models["model"].update_ema()

            if (s + 1) % t.log_interval == 0:
                dt = time.time() - t0
                sps = t.log_interval * t.batch_size / dt
                pct = 100.0 * (s + 1) / t.max_iters
                log.info(
                    "step %6d/%d (%4.1f%%) | loss=%.4f | %s | lr=%.2e | %.0f samp/s",
                    s + 1, t.max_iters, pct,
                    loss_dict["total_loss"].item(),
                    "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if k != "total_loss"),
                    lr, sps,
                )
                t0 = time.time()
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"train/{k}": v.item() for k, v in loss_dict.items()} | {"lr": lr},
                        step=s + 1,
                    )

            if (s + 1) % t.eval_interval == 0:
                log.info("  [eval] running %d val batches ...", t.eval_iters)
                val_metrics = self.eval()
                log.info("  [val]  %s", "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                if self.wandb_run:
                    self.wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=s + 1)
                self.save_checkpoint(s + 1)

        log.info("=== LatentLOM training complete (%d steps) ===", t.max_iters)
        if self.wandb_run:
            self.wandb_run.finish()
