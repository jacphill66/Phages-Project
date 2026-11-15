#!/usr/bin/env python3
"""
Does our agreement with the deposited table depend on tRNA gene duplication?

Several E. coli tRNA genes are exact sequence duplicates. A read from one
matches every copy equally well, and bowtie2 reports a single best alignment,
breaking ties with a seeded pseudo-random choice. Two runs with different
seeds therefore distribute the same reads differently across copies while
preserving the family total.

If that is what drives the disagreement, single-copy tRNAs should match the
deposited counts almost exactly, duplicated ones should scatter, and summing
each family back together should restore agreement. That last part is the
real test -- it distinguishes tie-breaking from a genuine counting error.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

op = lambda p: gzip.open(p, "rt", newline="") if str(p).endswith(".gz") else open(p, newline="")


def load_genome(fa: Path) -> str:
    out = []
    with op(fa) as fh:
        for line in fh:
            if not line.startswith(">"):
                out.append(line.strip())
    return "".join(out).upper()


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
    return sorted(feats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ours", required=True, type=Path)
    ap.add_argument("--theirs", required=True, type=Path)
    ap.add_argument("--gff", required=True, type=Path)
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--col", default="Count_3204_MG")
    args = ap.parse_args(argv)

    raw = np.load(args.ours)
    ref = sorted({k.split("|")[0] for k in raw.files})[0]
    g = lambda k, s: raw[f"{ref}|{k}|{s}"].astype(np.float64)
    our_start = g("five", "fwd") + g("three", "fwd")
    our_end = g("five", "rev") + g("three", "rev")
    n = our_start.size

    their_start = np.zeros(n + 2)
    their_end = np.zeros(n + 2)
    with op(args.theirs) as fh:
        for r in csv.DictReader(fh):
            p = int(r["position"])
            if 0 <= p <= n + 1:
                (their_start if r["end"] == "start" else their_end)[p] += float(r[args.col])

    genome = load_genome(args.fasta)
    feats = load_trnas(args.gff)
    comp = str.maketrans("ACGT", "TGCA")

    by_seq = defaultdict(list)
    for a, b, st, nm in feats:
        s = genome[a - 1 : b]
        if st == "-":
            s = s.translate(comp)[::-1]
        by_seq[s].append(nm)
    fam_of = {nm: tuple(sorted(v)) for v in by_seq.values() for nm in v}

    per = {}
    for a, b, st, nm in feats:
        o = t = 0.0
        for pos in range(a, b + 1):
            o += our_start[pos - 1] + our_end[pos - 1]
            t += their_start[pos - 1] + their_end[pos]
        per[nm] = (o, t)

    rows = []
    for nm, (o, t) in per.items():
        if o < 1000:
            continue
        rows.append((len(fam_of[nm]), nm, o, t, t / o if o else float("nan")))
    rows.sort(key=lambda r: (-r[0], -r[2]))

    print(f"{'fam':>4} {'tRNA':<8} {'ours':>12} {'theirs':>12} {'ratio':>7}  members")
    for fam, nm, o, t, ratio in rows[:28]:
        mem = ",".join(fam_of[nm]) if fam > 1 else ""
        print(f"{fam:>4} {nm:<8} {o:>12,.0f} {t:>12,.0f} {ratio:>7.2f}  {mem}")

    single = [r[4] for r in rows if r[0] == 1]
    dup = [r[4] for r in rows if r[0] > 1]
    print()
    for label, v in (("single-copy", single), ("duplicated", dup)):
        if v:
            print(f"{label:<12} n={len(v):>3}  ratio mean {np.mean(v):.3f}  "
                  f"sd {np.std(v):.3f}  range {min(v):.2f}-{max(v):.2f}")

    print("\nSummed per family (tie-breaking should cancel):")
    fam_tot = defaultdict(lambda: [0.0, 0.0])
    for nm, (o, t) in per.items():
        f = fam_of[nm]
        fam_tot[f][0] += o
        fam_tot[f][1] += t
    fr = [(t / o, f, o, t) for f, (o, t) in fam_tot.items() if o >= 1000 and len(f) > 1]
    fr.sort()
    for ratio, f, o, t in fr:
        print(f"  {ratio:>6.2f}  {o:>12,.0f} {t:>12,.0f}  {','.join(f)}")
    if fr:
        v = [x[0] for x in fr]
        print(f"\n  family totals: mean {np.mean(v):.3f}  sd {np.std(v):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
