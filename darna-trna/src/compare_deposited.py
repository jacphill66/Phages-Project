#!/usr/bin/env python3
"""
Compare our per-position read-end counts against the authors' deposited table.

The strongest validation available: not "my figure resembles theirs" but "my
numbers agree with theirs at every position, and here is the correlation". It
works only because we aligned to the same assembly they did (NC_000913.2), so
coordinates are directly comparable.

Two conventions are resolved empirically rather than assumed.

  Which end is which. count_ends.py stores 5'/3' ends by READ strand, while
  the deposited table stores genomic left/right ("start"/"end") with strands
  summed. A forward read's 5' end and a reverse read's 3' end are both the
  genomic leftmost base, so:
        genomic start = five[fwd] + three[rev]
        genomic end   = three[fwd] + five[rev]

  Coordinate offset. Correlation is computed at shifts of -1, 0 and +1
  separately for start and end. Whichever shift maximises agreement is the
  convention the authors used -- a question their methods never answer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np


def load_ours(npz_path: Path, ref: str | None = None):
    raw = np.load(npz_path)
    refs = sorted({k.split("|")[0] for k in raw.files})
    if ref is None:
        if len(refs) != 1:
            raise SystemExit(f"several references present: {refs}; pass --ref")
        ref = refs[0]
    g = lambda kind, strand: raw[f"{ref}|{kind}|{strand}"].astype(np.float64)
    # Every value stored in fragment mode is a read's 5'-most base, so
    # whether it is the genomic leftmost or rightmost depends on the READ's
    # strand, not on which array it landed in. A forward read's 5'-most base
    # is the leftmost; a reverse read's is the rightmost.
    start = g("five", "fwd") + g("three", "fwd")
    end = g("five", "rev") + g("three", "rev")
    return ref, start, end


def load_theirs(csv_path: Path, n: int, col: str):
    op = gzip.open if str(csv_path).endswith(".gz") else open
    start = np.zeros(n + 2)
    end = np.zeros(n + 2)
    with op(csv_path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            p = int(r["position"])
            if not 0 <= p <= n + 1:
                continue
            v = float(r[col])
            (start if r["end"] == "start" else end)[p] += v
    return start, end


def corr(a: np.ndarray, b: np.ndarray) -> float:
    m = (a > 0) | (b > 0)
    if m.sum() < 2:
        return float("nan")
    x, y = a[m], b[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def best_shift(ours: np.ndarray, theirs: np.ndarray, label: str):
    n = ours.size
    print(f"\n{label}")
    results = []
    for s in (-1, 0, 1):
        lo, hi = max(0, -s), min(n, theirs.size - s)
        a = ours[lo:hi]
        b = theirs[lo + s + 1 : hi + s + 1]
        k = min(a.size, b.size)
        r = corr(a[:k], b[:k])
        ratio = (b[:k].sum() / a[:k].sum()) if a[:k].sum() else float("nan")
        results.append((r, s, ratio))
        print(f"  shift {s:+d}:  r = {r:.6f}   theirs/ours = {ratio:.3f}")
    best = max(results, key=lambda t: (-1 if np.isnan(t[0]) else t[0]))
    print(f"  -> best shift {best[1]:+d} (r = {best[0]:.6f})")
    return best[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ours", required=True, type=Path, help="our counts .npz")
    ap.add_argument("--theirs", required=True, type=Path, help="deposited CSV")
    ap.add_argument("--col", default="Count_3204_MG",
                    help="3204 = DARNA, 3203 = control")
    ap.add_argument("--ref")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    ref, our_start, our_end = load_ours(args.ours, args.ref)
    n = our_start.size
    print(f"reference {ref}, {n} positions   (their column: {args.col})")
    their_start, their_end = load_theirs(args.theirs, n, args.col)
    print(f"ours   total ends: {our_start.sum() + our_end.sum():,.0f}")
    print(f"theirs total ends: {their_start.sum() + their_end.sum():,.0f}")

    ss = best_shift(our_start, their_start, "genomic START ends")
    se = best_shift(our_end, their_end, "genomic END ends")

    print("\nTop positions by our count, with their value at the best shift:")
    print(f"{'pos':>9} {'kind':>6} {'ours':>12} {'theirs':>12} {'ratio':>7}")
    rows = []
    for kind, ours, theirs, sh in (("start", our_start, their_start, ss),
                                   ("end", our_end, their_end, se)):
        idx = np.argsort(ours)[::-1][: args.top]
        for i in idx:
            j = i + sh + 1
            t = theirs[j] if 0 <= j < theirs.size else 0.0
            rows.append((ours[i], i + 1, kind, t))
    rows.sort(reverse=True)
    for o, p, kind, t in rows[: args.top]:
        print(f"{p:>9} {kind:>6} {o:>12,.0f} {t:>12,.0f} {(t/o if o else 0):>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
