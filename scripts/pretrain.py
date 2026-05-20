"""CLI entry point for LAM / LOM pre-training.

Usage:
    python -m scripts.pretrain lam [options]
    python -m scripts.pretrain lom [options]

    # Load an experiment config, then override individual fields:
    python -m scripts.pretrain lam --config experiments/temporal_abstraction/config.yaml
    python -m scripts.pretrain lom --config experiments/temporal_abstraction/config.yaml \\
        --train.batch_size 64
"""

from __future__ import annotations

import sys
from typing import Annotated, Union

import tyro
import yaml

from lom.config import LAMCfg, LOMCfg
from lom.training import LAMTrainer, LOMTrainer


def yaml_to_args(d: dict, prefix: str = "") -> list[str]:
    args = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            args.extend(yaml_to_args(v, key))
        elif v is None:
            args.extend([f"--{key}", "None"])
        elif isinstance(v, bool):
            args.extend([f"--{key}", str(v).lower()])
        else:
            args.extend([f"--{key}", str(v)])
    return args


def parse_args() -> list[str]:
    """Strip --config FILE from argv, expand its contents as CLI flags, keep the rest."""
    argv = sys.argv[1:]
    if not argv:
        return argv

    subcommand, rest = argv[0], argv[1:]

    if rest and rest[0] == "--config":
        with open(rest[1]) as f:
            d = yaml.safe_load(f)
        yaml_args = yaml_to_args(d)
        rest = yaml_args + rest[2:]

    return [subcommand] + rest


def main() -> None:
    cfg = tyro.cli(
        Union[
            Annotated[LAMCfg, tyro.conf.subcommand("lam")],
            Annotated[LOMCfg, tyro.conf.subcommand("lom")],
        ],
        args=parse_args(),
    )
    trainer = LAMTrainer(cfg) if isinstance(cfg, LAMCfg) else LOMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
