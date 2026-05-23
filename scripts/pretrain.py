"""CLI entry point for LAM / LOM pre-training.

Usage:
    python -m scripts.pretrain lam
    python -m scripts.pretrain lom

    # Override any field with dotted-path syntax (= or -- both work):
    python -m scripts.pretrain lam model.d_model=512 train.batch_size=64
    python -m scripts.pretrain lom --model.num_options=256 --data.horizon=128

    # Load an experiment config, then override individual fields:
    python -m scripts.pretrain lam --config experiments/benchmark/config.yaml
    python -m scripts.pretrain lom --config experiments/benchmark/config.yaml model.d_model=512
"""

from __future__ import annotations

import re
import sys
from typing import Annotated, Union

import tyro
import yaml

from lom.config import LAMCfg, LOMCfg
from lom.training import LAMTrainer, LOMTrainer


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
    if not argv:
        return argv

    subcommand, rest = argv[0], argv[1:]

    # Expand --config FILE into individual --key value flags
    config_args: list[str] = []
    remaining: list[str] = []
    i = 0
    while i < len(rest):
        if rest[i] == "--config" and i + 1 < len(rest):
            with open(rest[i + 1]) as f:
                config_args = _yaml_to_args(yaml.safe_load(f))
            i += 2
        else:
            remaining.append(rest[i])
            i += 1

    # Convert bare key=value overrides (Hydra-style) to --key value
    expanded: list[str] = []
    for arg in config_args + remaining:
        if re.match(r"^[a-z][a-z0-9_.]*=", arg):
            k, v = arg.split("=", 1)
            expanded.extend([f"--{k}", v])
        else:
            expanded.append(arg)

    return [subcommand] + expanded


def main() -> None:
    cfg = tyro.cli(
        Union[
            Annotated[LAMCfg, tyro.conf.subcommand("lam")],
            Annotated[LOMCfg, tyro.conf.subcommand("lom")],
        ],
        args=_parse_args(sys.argv[1:]),
    )
    trainer = LAMTrainer(cfg) if isinstance(cfg, LAMCfg) else LOMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
