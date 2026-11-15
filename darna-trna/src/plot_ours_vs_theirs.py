#!/usr/bin/env python3
"""
Plot our per-tRNA read-end signal beside the authors' deposited values.

This is the reproduction figure: two panels on a shared axis, our pipeline
above and the deposited table below. Agreement is then visible rather than
asserted.

Conventions applied, all established empirically (see README):

  Genomic left/right from read strand. count_ends.py stores every value in
  fragment mode as a read's 5'-most base, so a forward read's lands on the
  genomic leftmost base and a reverse read's on the rightmost:
        genomic start = five[fwd] + three[fwd]
        genomic end   = five[rev] + three[rev]

  The deposited table is 0-based on starts, 1-based on ends. A deposited
  start at p is 1-based p+1; a deposited end at p is 1-based p.

  Molecular 5'/3' from gene strand. On a minus-strand gene the molecule's
  5' end is the HIGHER genomic coordinate, so genomic "end" means 5' there.

Both samples are normalised to ends-per-million within their own library
before subtracting, so the panels are comparable despite the deposited table
counting both mates where we count each fragment once.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_FIVE, C_THREE = "#2a78d6", "#eb6834"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8982"
SURFACE, GRID, BAND = "#fcfcfb", "#e4e3df", "#f0efec"

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


def genomic_ends(npz: Path):
    raw = np.load(npz)
    ref = sorted({k.split("|")[0] for k in raw.files})[0]
    g = lambda k, s: raw[f"{ref}|{k}|{s}"].astype(np.float64)
    start = g("five", "fwd") + g("three", "fwd")
    end = g("five", "rev") + g("three", "rev")
    return ref, start, end


def ours_gained(treat_npz: Path, ctrl_npz: Path):
    ref, ts, te = genomic_ends(treat_npz)
    _, cs, ce = genomic_ends(ctrl_npz)
    tf = 1e6 / (ts.sum() + te.sum())
    cf = 1e6 / (cs.sum() + ce.sum())
    return ref, ts * tf - cs * cf, te * tf - ce * cf


def theirs_gained(csv_path: Path, n: int):
    s_t = np.zeros(n); e_t = np.zeros(n)
    s_c = np.zeros(n); e_c = np.zeros(n)
    with op(csv_path) as fh:
        for r in csv.DictReader(fh):
            p = int(r["position"])
            d = float(r["Count_3204_MG"]); c = float(r["Count_3203_MG"])
            if r["end"] == "start":
                i = p
                if 0 <= i < n:
                    s_t[i] += d; s_c[i] += c
            else:
                i = p - 1
                if 0 <= i < n:
                    e_t[i] += d; e_c[i] += c
    tf = 1e6 / (s_t.sum() + e_t.sum())
    cf = 1e6 / (s_c.sum() + e_c.sum())
    return s_t * tf - s_c * cf, e_t * tf - e_c * cf


def load_trna(gff: Path, name: str):
    with op(gff) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "tRNA":
                continue
            a = dict(p.split("=", 1) for p in f[8].rstrip(";\n").split(";") if "=" in p)
            if (a.get("gene") or a.get("locus_tag")) == name:
                return int(f[3]), int(f[4]), f[6], a.get("product", "")
    raise SystemExit(f"{name!r} not found as a tRNA in {gff}")


def to_offsets(start_arr, end_arr, a, b, strand):
    L = b - a + 1
    five = np.zeros(L); three = np.zeros(L)
    for pos in range(a, b + 1):
        off = (b - pos + 1) if strand == "-" else (pos - a + 1)
        s = start_arr[pos - 1]; e = end_arr[pos - 1]
        if strand == "+":
            five[off - 1] += s; three[off - 1] += e
        else:
            five[off - 1] += e; three[off - 1] += s
    return five, three


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--treat", required=True, type=Path)
    ap.add_argument("--ctrl", required=True, type=Path)
    ap.add_argument("--theirs", required=True, type=Path)
    ap.add_argument("--gff", required=True, type=Path)
    ap.add_argument("--trna", default="trpT")
    ap.add_argument("--out", type=Path, default=Path("results/reproduction.png"))
    args = ap.parse_args(argv)

    ref, our_s, our_e = ours_gained(args.treat, args.ctrl)
    n = our_s.size
    their_s, their_e = theirs_gained(args.theirs, n)

    a, b, strand, product = load_trna(args.gff, args.trna)
    o5, o3 = to_offsets(our_s, our_e, a, b, strand)
    t5, t3 = to_offsets(their_s, their_e, a, b, strand)
    L = b - a + 1
    x = np.arange(1, L + 1)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    for ax, (f, t), label in zip(axes, ((o5, o3), (t5, t3)),
                                 ("our pipeline", "deposited (authors)")):
        ax.axvspan(33.5, 36.5, color=BAND, zorder=0)
        ax.bar(x - 0.2, f, width=0.38, color=C_FIVE, label="5' ends", linewidth=0)
        ax.bar(x + 0.2, t, width=0.38, color=C_THREE, label="3' ends", linewidth=0)
        ax.axhline(0, color=INK_MUTED, linewidth=0.6)
        ax.set_ylabel(f"{label}\nREPM gained")
        ax.margins(y=0.18)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_title(
        f"{args.trna} ({strand} strand, {L} nt) - our counts vs deposited",
        loc="left", pad=8)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_xlabel("position in tRNA (nt from 5' end)")

    po = int(np.argmax(o5 + o3)) + 1
    pt = int(np.argmax(t5 + t3)) + 1
    axes[0].annotate(f"{po}", xy=(po, float((o5 + o3)[po - 1])), xytext=(10, -2),
                     textcoords="offset points", fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.6))
    axes[1].annotate(f"{pt}", xy=(pt, float((t5 + t3)[pt - 1])), xytext=(10, -2),
                     textcoords="offset points", fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.6))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)

    both = np.concatenate([o5, o3]); oth = np.concatenate([t5, t3])
    m = (both > 0) | (oth > 0)
    r = np.corrcoef(both[m], oth[m])[0, 1] if m.sum() > 2 else float("nan")
    print(f"{args.trna}: our peak {po}, deposited peak {pt}, r = {r:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
