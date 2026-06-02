"""CLI entry point for LOM pre-training.

Two required flags control what to train:

  --signal reconstruction|latent   pixel reconstruction vs latent JEPA dynamics
  --method lam|lom                 LAM (atomic, horizon=1) vs LOM (temporal, horizon from config)

Usage:
    python -m scripts.pretrain --method lam --signal latent
    python -m scripts.pretrain --method lom --signal reconstruction

    # Load an experiment config, then override individual fields:
    python -m scripts.pretrain --method lam --signal latent \\
        --config experiments/benchmark/config.yaml

    python -m scripts.pretrain --method lom --signal latent \\
        --config experiments/benchmark/config.yaml model.d_model=512

    # key=value overrides (Hydra-style) and --key value are both accepted:
    python -m scripts.pretrain --method lom --signal reconstruction \\
        train.batch_size=64 model.num_options=256
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

import tyro
import yaml

from lom.config import LOMCfg
from lom.training import ReconstructionLOMTrainer, LatentLOMTrainer


def _yaml_to_args(d: dict, prefix: str = "") -> list[str]:
    args = []
    for k, v in d.items():
        if not prefix and k == "sweep":
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            args.extend(_yaml_to_args(v, key))
        elif v is None:
            args.extend([f"--{key}", "None"])
        elif isinstance(v, bool):
            args.extend([f"--{key}", str(v).lower()])
        else:
            args.extend([f"--{key}", str(v)])
    return args


def _parse_args(argv: list[str]) -> list[str]:
    """Expand --config FILE and normalise key=value overrides to --key value."""
    config_args: list[str] = []
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            with open(argv[i + 1]) as f:
                config_args = _yaml_to_args(yaml.safe_load(f))
            i += 2
        else:
            remaining.append(argv[i])
            i += 1

    expanded: list[str] = []
    for arg in config_args + remaining:
        if re.match(r"^[a-z][a-z0-9_.]*=", arg):
            k, v = arg.split("=", 1)
            expanded.extend([f"--{k}", v])
        else:
            expanded.append(arg)

    return expanded


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", choices=["lam", "lom"], required=True)
    pre.add_argument("--signal", choices=["reconstruction", "latent"], required=True)
    known, rest = pre.parse_known_args(sys.argv[1:])

    cfg = tyro.cli(LOMCfg, args=_parse_args(rest))

    if known.method == "lam":
        cfg.data.horizon = 1
        cfg.model.num_options = 100

    trainer = (
        LatentLOMTrainer(cfg, method=known.method, signal=known.signal)
        if known.signal == "latent"
        else ReconstructionLOMTrainer(cfg, method=known.method, signal=known.signal)
    )
    trainer.train()


if __name__ == "__main__":
    main()
