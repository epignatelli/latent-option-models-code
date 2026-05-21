from __future__ import annotations

import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim

from .config import LAMCfg, LOMCfg
from .dataset import build_dataloaders, load_nao_top10, load_nld_nao, load_nld_aa
from .models import DynamicsModel, LatentActionModel

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


def get_lr(step: int, lr: float, warmup_iters: int, max_iters: int, eta_min: float) -> float:
    if step < warmup_iters:
        return lr * step / warmup_iters
    frac = (step - warmup_iters) / (max_iters - warmup_iters)
    return eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * frac))


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

        _loaders = {"nao-top10": load_nao_top10, "nld-nao": load_nld_nao, "nld-aa": load_nld_aa}
        loader_fn = _loaders[d.dataset]
        sequences, _ = loader_fn(
            nle_data_dir=d.nle_data_dir, fallback_numpy_dir=d.fallback_numpy_dir
        )
        self.train_loader, self.val_loader = build_dataloaders(
            sequences,
            context_len=d.context_len,
            horizon=d.horizon,
            batch_size=t.batch_size,
            val_fraction=d.val_fraction,
            num_workers=d.num_workers,
            seed=t.seed,
            return_sequence=isinstance(cfg, LOMCfg),
        )

        self.models = self.build_models().to(self.device)
        if t.compile_model:
            for key in list(self.models.keys()):
                self.models[key] = torch.compile(self.models[key])

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

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        amp_dtype = dtype_map.get(t.mixed_dtype, torch.float16)
        self.ctx = (
            torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype)
            if "cpu" not in str(self.device)
            else NullCtx()
        )
        self.scaler = torch.cuda.GradScaler(
            enabled=(amp_dtype == torch.float16 and "cuda" in str(self.device))
        )

        os.makedirs(t.ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(t.ckpt_dir, f"{self.label()}_pretrain.pt")

        self.wandb_run = None
        try:
            import wandb

            os.makedirs(cfg.wandb.dir, exist_ok=True)
            self.wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                group=cfg.wandb.group,
                dir=cfg.wandb.dir,
                config=asdict(cfg),
                resume="allow" if t.resume else "never",
            )
        except Exception as exc:
            log.warning("WandB init failed: %s", exc)

        self.start_step = self.restore_checkpoint() if t.resume else 0

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
        i = 0
        for i, batch in enumerate(self.val_loader):
            if i >= self.cfg.train.eval_iters:
                break
            batch = [x.to(self.device) for x in batch]
            for k, v in self.step(batch).items():
                totals[k] = totals.get(k, 0.0) + v.item()
        for mod in self.models.values():
            mod.train()
        return {k: v / (i + 1) for k, v in totals.items()}

    def train(self) -> None:
        t = self.cfg.train
        log.info("Device: %s  |  training: %s", self.device, self.label())
        log.info("Parameters: %s", f"{sum(p.numel() for p in self.models.parameters()):,}")
        log.info("Train: %d batches/epoch  Val: %d", len(self.train_loader), len(self.val_loader))

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
                log.info(
                    "step %6d | total=%.4f | %s | lr=%.2e | %.2f s/step",
                    s + 1,
                    loss_dict["total_loss"].item(),
                    "  ".join(
                        f"{k}={v.item():.4f}" for k, v in loss_dict.items() if k != "total_loss"
                    ),
                    lr,
                    dt / t.log_interval,
                )
                t0 = time.time()
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"train/{k}": v.item() for k, v in loss_dict.items()} | {"lr": lr},
                        step=s + 1,
                    )

            if (s + 1) % t.eval_interval == 0:
                val_metrics = self.eval()
                log.info("  [val] %s", "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                if self.wandb_run:
                    self.wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=s + 1)
                self.save_checkpoint(s + 1)

        log.info("Training complete.")
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
            dropout=m.dropout,
            bias=m.bias,
        )
        vq = dict(
            vq_dropout=m.vq_dropout,
            vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta,
            vq_reset_thresh=m.vq_reset_thresh,
        )
        return nn.ModuleDict(
            {
                "lam": LatentActionModel(
                    **base, codebook_size=m.num_options, horizon=1, **vq
                ),
                "dynamics": DynamicsModel(**base, predict_sequence=False),
            }
        )

    def step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        history, next_frame = batch[0], batch[1]
        z, vq, _ = self.models["lam"](history, next_frame)
        logits = self.models["dynamics"](history, z.detach())
        recon = reconstruction_loss(logits, next_frame, self.cfg.env.vocab_size)
        total = recon + vq["vq_loss"]
        return {
            "recon": recon,
            "vq_loss": vq["vq_loss"],
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
            dropout=m.dropout,
            bias=m.bias,
        )
        vq = dict(
            vq_dropout=m.vq_dropout,
            vq_entropy_weight=m.vq_entropy_weight,
            vq_beta=m.vq_beta,
            vq_reset_thresh=m.vq_reset_thresh,
        )
        return nn.ModuleDict(
            {
                "option_lam": LatentActionModel(
                    **base, codebook_size=m.num_options, horizon=d.horizon, **vq
                ),
                "action_lam": LatentActionModel(
                    **base,
                    codebook_size=e.n_actions,
                    horizon=1,
                    condition_dim=m.latent_dim,
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

        lam_logits = self.models["lam_dynamics"](history, z_act.detach())
        lam_recon = reconstruction_loss(lam_logits, next_frame, self.cfg.env.vocab_size)

        if m.predict_sequence:
            lom_logits = self.models["lom_dynamics"](
                history,
                z_act.detach(),
                option_code=z_opt.detach(),
                horizon=self.cfg.data.horizon,
                teacher_frames=sequence,
            )
            lom_recon = reconstruction_loss(lom_logits, sequence, self.cfg.env.vocab_size)
        else:
            lom_logits = self.models["lom_dynamics"](
                history,
                z_act.detach(),
                option_code=z_opt.detach(),
                horizon=1,
            )
            lom_recon = reconstruction_loss(lom_logits, future_frame, self.cfg.env.vocab_size)

        vq_loss = vq_opt["vq_loss"] + vq_act["vq_loss"]
        total = lam_recon + lom_recon + vq_loss
        return {
            "lam_recon": lam_recon,
            "lom_recon": lom_recon,
            "vq_loss_option": vq_opt["vq_loss"],
            "vq_loss_action": vq_act["vq_loss"],
            "entropy_option": vq_opt["entropy"],
            "entropy_action": vq_act["entropy"],
            "total_loss": total,
        }
