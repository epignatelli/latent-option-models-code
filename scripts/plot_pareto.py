"""Plot STT vs JEPA pareto frontiers from two JSON files produced by profile_memory.

Usage:
    python -m scripts.plot_pareto \\
        /tmp/pareto_stt.json /tmp/pareto_jepa.json \\
        --out /path/to/figures/pareto.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


COLORS = {"stt": "#1f77b4", "jepa": "#ff7f0e"}
LABELS = {"stt": "STT (pixel reconstruction)", "jepa": "JEPA (latent prediction)"}
MARKERS = {"stt": "o", "jepa": "s"}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot(data: list[dict], out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for d in data:
        enc   = d["encoder"]
        rows  = [r for r in d["rows"] if r["max_batch"] > 0]
        ctxs  = [r["ctx"] for r in rows]
        batch = [r["max_batch"] for r in rows]
        sps   = [r["samp_s"] for r in rows]
        c, m, lbl = COLORS[enc], MARKERS[enc], LABELS[enc]

        axes[0].plot(ctxs, batch, color=c, marker=m, linewidth=1.8,
                     markersize=6, label=lbl)
        axes[1].plot(ctxs, sps,   color=c, marker=m, linewidth=1.8,
                     markersize=6, label=lbl)

    for ax, ylabel, title in zip(
        axes,
        ["Max batch size", "Throughput (samp/s)"],
        ["Memory frontier", "Throughput frontier"],
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.set_xlabel("Context length (frames)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("LAM: STT vs JEPA — pareto frontier  (patch\\_size=8, 85 GB GPU)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stt_json",  help="JSON from --encoder stt --json-out")
    parser.add_argument("jepa_json", help="JSON from --encoder jepa --json-out")
    parser.add_argument("--out", default="pareto.pdf",
                        help="output path for the figure (PDF or PNG)")
    args = parser.parse_args()

    stt  = load(args.stt_json)
    jepa = load(args.jepa_json)
    plot([stt, jepa], args.out)


if __name__ == "__main__":
    main()
