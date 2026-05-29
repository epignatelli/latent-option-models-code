from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvCfg:
    obs_h: int = 24
    obs_w: int = 80
    vocab_size: int = 8192  # TOKEN_VOCAB = CHAR_VOCAB(256) * COLOR_VOCAB(32)
    n_actions: int = 98


@dataclass
class ModelCfg:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    context_length: int = 256
    latent_dim: int = 512
    num_options: int = 256
    vq_dropout: float = 0.1
    vq_entropy_weight: float = 0.01
    vq_beta: float = 0.25
    vq_reset_thresh: int = 100
    vq_ema_decay: float = 0.99
    two_encoder: bool = False  # True: JEPAEncoder (separate context/target passes)
    ema_decay: float = 0.996   # EMA decay for the target encoder in JEPA training
    patch_size: int = 1
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")


@dataclass
class LOMModelCfg(ModelCfg):
    predict_sequence: bool = False  # True: LOM dynamics uses teacher forcing over full sequence


@dataclass
class DataCfg:
    dataset_dir: str = ""
    context_len: int = 4
    horizon: int = 8
    val_fraction: float = 0.05
    buffer_size: int = 100
    steps_per_epoch: int = 10_000


@dataclass
class TrainCfg:
    max_iters: int = 100_000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    eta_min: float = 1e-6
    warmup_iters: int = 1_000
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 50
    compile_model: bool = True
    mixed_dtype: str = "bfloat16"
    seed: int = 42
    ckpt_dir: str = "/scratch/uceeepi/lom/checkpoints"
    resume: bool = False

    def __post_init__(self) -> None:
        if self.warmup_iters >= self.max_iters:
            raise ValueError("warmup_iters must be < max_iters")
        if self.eta_min >= self.lr:
            raise ValueError("eta_min must be < lr")


@dataclass
class WandbCfg:
    project: str = "latent-option-models"
    entity: str = "epignatelli_"
    group: str = "default"
    dir: str = "/scratch/uceeepi/lom/wandb"


@dataclass
class LAMCfg:
    env: EnvCfg = field(default_factory=EnvCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)


@dataclass
class LOMCfg:
    env: EnvCfg = field(default_factory=EnvCfg)
    model: LOMModelCfg = field(default_factory=LOMModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)
