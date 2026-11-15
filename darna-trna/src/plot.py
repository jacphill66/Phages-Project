#!/usr/bin/env python3
"""
Draw the two figures: a per-position trace for one reference, and a heatmap
across all references.

Both mirror the structure of Figure 2 in the source paper. The trace puts the
read-end difference and the coverage ratio in separate stacked panels sharing
an x-axis rather than on two y-scales, because a dual-axis plot invites the
reader to compare two quantities whose units have nothing to do with each
other.

Colors are a colorblind-safe pair for the two end types, and a blue-to-red
diverging ramp with a neutral gray midpoint for the signed heatmap. The end
types are also separated by bar position and named in the legend, so identity
never rests on hue alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

C_FIVE = "#2a78d6"
C_THREE = "#eb6834"
C_RATIO = "#52514e"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8982"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#86b6ef", "#f0efec", "#ec835a", "#d03b3b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.size": 10,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load_signals(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in raw.files:
        ref, field = key.split("|")
        out.setdefault(ref, {})[field] = raw[key]
    return out


def plot_trace(
    signals: dict[str, dict[str, np.ndarray]],
    ref: str,
    out_path: Path,
    annotate_top: bool = True,
) -> int:
    """Per-position read-end difference and coverage ratio. Returns peak position."""
    if ref not in signals:
        raise SystemExit(
            f"reference {ref!r} not in signals; available: {', '.join(sorted(signals))}"
        )
    d = signals[ref]
    pos = np.arange(1, d["five_diff"].size + 1)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.18},
    )

    ax_top.bar(pos - 0.2, d["five_diff"], width=0.38, color=C_FIVE,
               label="5' ends", linewidth=0)
    ax_top.bar(pos + 0.2, d["three_diff"], width=0.38, color=C_THREE,
               label="3' ends", linewidth=0)
    ax_top.axhline(0, color=INK_MUTED, linewidth=0.6)
    ax_top.set_ylabel("delta read ends per million\n(treatment - control)")
    ax_top.set_title(f"{ref}: read-end enrichment and coverage ratio",
                     loc="left", pad=8)
    ax_top.legend(loc="upper left", ncol=2)
    ax_top.margins(y=0.22)
    ax_top.grid(axis="y", color=GRID, linewidth=0.6)
    ax_top.set_axisbelow(True)
    for side in ("top", "right"):
        ax_top.spines[side].set_visible(False)

    peak_idx = int(np.argmax(d["five_diff"] + d["three_diff"]))
    peak_pos = peak_idx + 1
    if annotate_top:
        peak_val = float((d["five_diff"] + d["three_diff"])[peak_idx])
        ax_top.annotate(
            f"position {peak_pos}",
            xy=(peak_pos, peak_val),
            xytext=(12, -4),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.6),
        )

    ax_bot.step(pos, d["cov_ratio"], where="mid", color=C_RATIO, linewidth=1.6)
    ax_bot.axhline(0, color=INK_MUTED, linewidth=0.6, linestyle=":")
    ax_bot.axvline(peak_pos + 0.5, color=INK_MUTED, linewidth=0.6, linestyle="--")
    ax_bot.set_ylabel("log2 coverage ratio")
    ax_bot.set_xlabel("position (nt from 5' end)")
    ax_bot.grid(axis="y", color=GRID, linewidth=0.6)
    ax_bot.set_axisbelow(True)
    for side in ("top", "right"):
        ax_bot.spines[side].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return peak_pos


def plot_heatmap(
    signals: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    max_pos: int | None = None,
    clip_percentile: float = 99.5,
) -> None:
    """One row per reference, columns are positions, color is the signed difference."""
    refs = sorted(signals)
    width = max_pos or max(signals[r]["total_diff"].size for r in refs)

    matrix = np.full((len(refs), width), np.nan)
    for i, ref in enumerate(refs):
        arr = signals[ref]["total_diff"]
        n = min(width, arr.size)
        matrix[i, :n] = arr[:n]

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise SystemExit("no finite values to plot")
    limit = float(np.percentile(np.abs(finite), clip_percentile)) or 1.0

    height = max(2.4, 0.18 * len(refs) + 1.4)
    fig, ax = plt.subplots(figsize=(7.6, height))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=DIVERGING,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        extent=(0.5, width + 0.5, len(refs) - 0.5, -0.5),
    )
    ax.set_yticks(range(len(refs)))
    ax.set_yticklabels(refs, fontsize=7 if len(refs) > 30 else 9)
    ax.set_xlabel("position (nt from 5' end)")
    ax.set_title("Read-end difference across references (5' + 3')",
                 loc="left", pad=8)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("delta read ends per million", fontsize=9, color=INK_SOFT)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, color=INK_MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("signals", type=Path, help="output .npz from analyze.py")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument(
        "--ref",
        help="reference to draw the trace for; defaults to the strongest signal",
    )
    ap.add_argument("--max-pos", type=int, help="truncate heatmap columns")
    args = ap.parse_args(argv)

    signals = load_signals(args.signals)
    if not signals:
        raise SystemExit("no references found in signals file")

    ref = args.ref
    if ref is None:
        ref = max(signals, key=lambda r: float(np.nanmax(signals[r]["total_diff"])))

    trace_path = args.outdir / f"trace_{ref}.png"
    peak = plot_trace(signals, ref, trace_path)
    heat_path = args.outdir / "heatmap_all_refs.png"
    plot_heatmap(signals, heat_path, max_pos=args.max_pos)

    print(f"trace    {trace_path}  (peak at position {peak})")
    print(f"heatmap  {heat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
