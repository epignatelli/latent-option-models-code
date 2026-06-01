"""Plot profiling results from profile_memory --full-sweep.

Produces two figures from a directory of JSON files:

  context_length.pdf  — 1×2:  batch vs ctx  |  sps vs ctx
                               colour = method (LAM/LOM)
                               linestyle = architecture

  horizon.pdf         — 1×2:  batch vs horizon  |  sps vs horizon
                               linestyle = architecture (LOM only)

Usage:
    python -m scripts.plot_pareto --in-dir profiling_results/ --out-dir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

ENCODERS   = ["reconstruction", "latent", "latent-medium", "latent-params"]
ENC_LABELS = {"reconstruction": "Reconstruction",
              "latent":         "Latent",
              "latent-medium":  "Latent-medium",
              "latent-params":  "Latent-params"}
LINESTYLES = ["-",  "--", "-.", ":"]
MARKERS    = ["o", "s",  "^", "D"]

LAM_COLOR = "black"
LOM_COLOR = "red"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _rows(data: dict | None) -> tuple[list, list, list]:
    if data is None:
        return [], [], []
    rows  = [r for r in data["rows"] if r["max_batch"] > 0]
    xkey  = "ctx" if data["sweep"] == "ctx" else "horizon"
    xs    = [r[xkey]        for r in rows]
    batch = [r["max_batch"] for r in rows]
    sps   = [r["samp_s"]    for r in rows]
    return xs, batch, sps


def _style_ax(ax: matplotlib.axes.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _add_legend(ax: matplotlib.axes.Axes, method_labels: list[tuple[str, str]]) -> None:
    method_handles = [
        mlines.Line2D([], [], color=color, linewidth=2, label=label)
        for label, color in method_labels
    ]
    enc_handles = [
        mlines.Line2D([], [], color="black", linestyle=ls, marker=mk, markersize=4,
                      linewidth=1.5, label=ENC_LABELS[enc])
        for enc, ls, mk in zip(ENCODERS, LINESTYLES, MARKERS)
    ]
    ax.legend(handles=method_handles + enc_handles, fontsize=8,
              loc="upper right", framealpha=0.8)


def plot_context(in_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, enc in enumerate(ENCODERS):
        ls, mk = LINESTYLES[i], MARKERS[i]
        for method, color in [("lam", LAM_COLOR), ("lom", LOM_COLOR)]:
            data = load(in_dir / f"pareto_{method}_{enc}.json")
            xs, batch, sps = _rows(data)
            if not xs:
                continue
            kw = dict(color=color, linestyle=ls, linewidth=1.8, marker=mk, markersize=4)
            axes[0].plot(xs, batch, **kw)
            axes[1].plot(xs, sps,   **kw)

    _style_ax(axes[0], "Context length (frames)", "Max batch size")
    _style_ax(axes[1], "Context length (frames)", "Throughput (samp/s)")
    axes[0].set_title("Memory frontier")
    axes[1].set_title("Throughput frontier")
    _add_legend(axes[1], [("LAM", LAM_COLOR), ("LOM", LOM_COLOR)])

    fig.suptitle("Context length sweep  (patch_size=8, H100 96 GB)", fontsize=11)
    fig.tight_layout()
    out = out_dir / "context_length.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


HORIZON_CTXS = [4, 16]


def plot_horizon(in_dir: Path, out_dir: Path) -> None:
    ctxs = [c for c in HORIZON_CTXS
            if any((in_dir / f"horizon_lom_{enc}_ctx{c}.json").exists() for enc in ENCODERS)]
    if not ctxs:
        print("No horizon files found — skipping horizon plot")
        return

    fig, axes = plt.subplots(len(ctxs), 2, figsize=(10, 4 * len(ctxs)), squeeze=False)

    for row, ctx in enumerate(ctxs):
        for i, enc in enumerate(ENCODERS):
            ls, mk = LINESTYLES[i], MARKERS[i]
            data = load(in_dir / f"horizon_lom_{enc}_ctx{ctx}.json")
            xs, batch, sps = _rows(data)
            if not xs:
                continue
            kw = dict(color=LOM_COLOR, linestyle=ls, linewidth=1.8, marker=mk, markersize=4)
            axes[row, 0].plot(xs, batch, **kw)
            axes[row, 1].plot(xs, sps,   **kw)

        _style_ax(axes[row, 0], "Horizon (frames)", "Max batch size")
        _style_ax(axes[row, 1], "Horizon (frames)", "Throughput (samp/s)")
        axes[row, 0].set_title(f"ctx={ctx} — memory frontier")
        axes[row, 1].set_title(f"ctx={ctx} — throughput frontier")
        enc_handles = [
            mlines.Line2D([], [], color=LOM_COLOR, linestyle=ls, marker=mk, markersize=4,
                          linewidth=1.5, label=ENC_LABELS[enc])
            for enc, ls, mk in zip(ENCODERS, LINESTYLES, MARKERS)
        ]
        axes[row, 1].legend(handles=enc_handles, fontsize=8,
                            loc="upper right", framealpha=0.8)

    fig.suptitle("Horizon sweep  (patch_size=8, H100 96 GB)", fontsize=11)
    fig.tight_layout()
    out = out_dir / "horizon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir",  default="profiling_results",
                        help="directory containing JSON files from --full-sweep")
    parser.add_argument("--out-dir", default="figures",
                        help="directory for output PDFs")
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_context(in_dir, out_dir)
    plot_horizon(in_dir, out_dir)


if __name__ == "__main__":
    main()
