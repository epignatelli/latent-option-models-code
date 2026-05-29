"""Plot pareto frontiers from JSON files produced by profile_memory.

Accepts any number of JSON files; labels are derived from filenames unless
overridden with --labels.

Usage:
    python -m scripts.plot_pareto \\
        /tmp/pareto_stt.json /tmp/pareto_jepa_compute.json ... \\
        --labels "STT 155M" "JEPA-compute 22M" \\
        --title "LAM: encoder comparison" \\
        --out figures/pareto_lam.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _auto_label(path: str) -> str:
    stem = Path(path).stem
    for prefix in ("pareto_lam_", "pareto_lom_", "pareto_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem.replace("_", "-")


def plot(data: list[dict], labels: list[str], title: str, out: str,
         xaxis: str = "ctx") -> None:
    xkey    = "ctx" if xaxis == "ctx" else "horizon"
    xlabel  = "Context length (frames)" if xaxis == "ctx" else "Horizon (frames)"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, (d, lbl) in enumerate(zip(data, labels)):
        rows  = [r for r in d["rows"] if r["max_batch"] > 0]
        xs    = [r[xkey] for r in rows]
        batch = [r["max_batch"] for r in rows]
        sps   = [r["samp_s"] for r in rows]
        c, m  = PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]

        axes[0].plot(xs, batch, color=c, marker=m, linewidth=1.8, markersize=6, label=lbl)
        axes[1].plot(xs, sps,   color=c, marker=m, linewidth=1.8, markersize=6, label=lbl)

    for ax, ylabel, subtitle in zip(
        axes,
        ["Max batch size", "Throughput (samp/s)"],
        ["Memory frontier", "Throughput frontier"],
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="JSON files from profile_memory --json-out")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="display labels, one per file (default: derived from filename)")
    parser.add_argument("--title", default="Pareto frontier  (patch\\_size=8, 85 GB GPU)")
    parser.add_argument("--out", default="pareto.pdf",
                        help="output path (PDF or PNG)")
    parser.add_argument("--xaxis", choices=["ctx", "horizon"], default="ctx",
                        help="x-axis variable: ctx (default) or horizon")
    args = parser.parse_args()

    data   = [load(f) for f in args.files]
    labels = args.labels if args.labels else [_auto_label(f) for f in args.files]
    if len(labels) != len(data):
        parser.error(f"--labels count ({len(labels)}) must match file count ({len(data)})")

    plot(data, labels, args.title, args.out, xaxis=args.xaxis)


if __name__ == "__main__":
    main()
