#!/usr/bin/env python3
"""
Draw Figure 2B/2C equivalents directly from the deposited GEO read-end table.

Two conventions, both verified against the data rather than assumed:

  countdiff is CONTROL minus DARNA, so a cleavage site is a large negative
  value. This script plots -countdiff, so gained ends point upward.

  The `end` column is genomic left/right, not molecular 5'/3'. On a
  minus-strand gene the molecule's 5' end sits at the HIGHER coordinate, so
  `end` there means 5'. The mapping follows each gene's annotated strand.
  Because the table has no strand column, reads from both strands were summed
  per position; at a tRNA locus nearly all reads come from that tRNA's own
  strand, but it is an assumption.

Coordinates must match the assembly the table was built on. The GEO record
says only "NC_000913" with no version, and the tables are NC_000913.2 --
annotating them against .3 misassigns every site. Use src/ft2gff.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from bisect import bisect_right
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

C_FIVE, C_THREE = "#2a78d6", "#eb6834"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8982"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pale_to_deep", ["#fcfcfb", "#d6e5f7", "#86b6ef", "#2a78d6", "#123d6b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 9, "legend.frameon": False,
    "font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight",
})

op = lambda p: gzip.open(p, "rt", newline="") if str(p).endswith(".gz") else open(p, newline="")


def load_trnas(gff: Path):
    feats = []
    with op(gff) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "tRNA":
                continue
            a = dict(p.split("=", 1) for p in f[8].rstrip(";\n").split(";") if "=" in p)
            feats.append((int(f[3]), int(f[4]), f[6],
                          a.get("gene") or a.get("locus_tag") or "?"))
    feats.sort()
    return feats


def collect(gff: Path, table: Path):
    """Return {tRNA: {'len':L, 'strand':s, 'five':array, 'three':array}}."""
    feats = load_trnas(gff)
    if not feats:
        raise SystemExit(f"no tRNA features in {gff}")
    starts = [f[0] for f in feats]
    span = max(b - a + 1 for a, b, *_ in feats)

    def at(pos):
        hi = bisect_right(starts, pos)
        lo = bisect_right(starts, pos - span - 1)
        for f in feats[lo:hi]:
            if f[0] <= pos <= f[1]:
                return f
        return None

    out = {}
    with op(table) as fh:
        for r in csv.DictReader(fh):
            pos = int(r["position"])
            f = at(pos)
            if f is None:
                continue
            a, b, strand, name = f
            L = b - a + 1
            slot = out.setdefault(name, {
                "len": L, "strand": strand,
                "five": np.zeros(L), "three": np.zeros(L),
            })
            off = (b - pos + 1) if strand == "-" else (pos - a + 1)
            if not 1 <= off <= L:
                continue
            is_five = (r["end"] == "start") if strand == "+" else (r["end"] == "end")
            gained = -float(r["countdiff"])
            slot["five" if is_five else "three"][off - 1] += gained
    return out


def plot_trace(data, name, out_path: Path):
    if name not in data:
        raise SystemExit(f"{name!r} not found; have: {', '.join(sorted(data))}")
    d = data[name]
    x = np.arange(1, d["len"] + 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - 0.2, d["five"], width=0.38, color=C_FIVE, label="5' ends", linewidth=0)
    ax.bar(x + 0.2, d["three"], width=0.38, color=C_THREE, label="3' ends", linewidth=0)
    ax.axhline(0, color=INK_MUTED, linewidth=0.6)
    ax.axvspan(33.5, 36.5, color="#f0efec", zorder=0)
    ax.set_xlabel("position in tRNA (nt from 5' end)")
    ax.set_ylabel("read ends gained in DARNA\n(REPM, DARNA - control)")
    ax.set_title(f"{name} ({d['strand']} strand, {d['len']} nt)", loc="left", pad=8)
    ax.legend(loc="upper left", ncol=2)
    ax.margins(y=0.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    tot = d["five"] + d["three"]
    pk = int(np.argmax(tot)) + 1
    ax.annotate(f"position {pk}", xy=(pk, float(tot[pk - 1])), xytext=(14, -2),
                textcoords="offset points", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.6))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return pk


def plot_heatmap(data, out_path: Path, top=None, max_pos=None, clip=99.0):
    order = sorted(data, key=lambda n: -(data[n]["five"] + data[n]["three"]).sum())
    if top:
        order = order[:top]
    width = max_pos or max(data[n]["len"] for n in order)

    m = np.full((len(order), width), np.nan)
    for i, n in enumerate(order):
        v = data[n]["five"] + data[n]["three"]
        k = min(width, v.size)
        m[i, :k] = v[:k]

    finite = m[np.isfinite(m)]
    vmax = float(np.percentile(finite, clip)) if finite.size else 1.0
    vmax = vmax or 1.0

    fig, ax = plt.subplots(figsize=(7.6, max(2.6, 0.17 * len(order) + 1.4)))
    im = ax.imshow(m, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax,
                   interpolation="nearest",
                   extent=(0.5, width + 0.5, len(order) - 0.5, -0.5))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6 if len(order) > 40 else 8)
    ax.set_xlabel("position in tRNA (nt from 5' end)")
    ax.set_title("Read ends gained in DARNA, across tRNAs", loc="left", pad=8)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.axvline(33.5, color=INK_MUTED, linewidth=0.5, linestyle="--")
    ax.axvline(36.5, color=INK_MUTED, linewidth=0.5, linestyle="--")

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("REPM gained (clipped at p%g)" % clip, fontsize=9, color=INK_SOFT)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=INK_MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return order


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--gff", required=True, type=Path,
                    help="annotation MATCHING the table's assembly (NC_000913.2)")
    ap.add_argument("--table", required=True, type=Path, help="deposited readends CSV")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--trna", default="trpT", help="tRNA for the trace figure")
    ap.add_argument("--top", type=int, default=40, help="tRNAs in the heatmap")
    ap.add_argument("--max-pos", type=int)
    args = ap.parse_args(argv)

    data = collect(args.gff, args.table)
    print(f"tRNAs with data: {len(data)}")

    t = args.outdir / f"deposited_trace_{args.trna}.png"
    pk = plot_trace(data, args.trna, t)
    h = args.outdir / "deposited_heatmap_tRNAs.png"
    order = plot_heatmap(data, h, top=args.top, max_pos=args.max_pos)

    print(f"trace    {t}  (peak at position {pk})")
    print(f"heatmap  {h}  ({len(order)} tRNAs)")
    print("top by total gain: " + ", ".join(order[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
