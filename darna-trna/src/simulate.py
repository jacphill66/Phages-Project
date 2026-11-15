#!/usr/bin/env python3
"""
Generate synthetic RNA-seq data with a cleavage site at a position you choose.

Why this exists
---------------
Nothing downstream of alignment will tell you if your coordinate handling is
wrong. A pipeline with a one-base shift produces a clean, confident, wrong
answer: the spike lands at 33 instead of 34 and no step errors out. The only
defence is data whose answer you already know.

This script builds that data. It emits:

  <prefix>_ref.fa              a synthetic reference sequence
  <prefix>_treat_R1.fastq.gz   treatment reads (fraction of molecules cleaved)
  <prefix>_treat_R2.fastq.gz
  <prefix>_ctrl_R1.fastq.gz    control reads (no cleavage, same everything else)
  <prefix>_ctrl_R2.fastq.gz
  <prefix>_treat.sam           ground-truth alignments, treatment
  <prefix>_ctrl.sam            ground-truth alignments, control
  <prefix>_truth.json          the parameters used, including the cut position

The FASTQ files carry adapters and so exercise trimming and alignment. The
SAM files are what a correct trim-and-align would produce, so they let you
test the counting code on its own. If counting is right on the SAM but wrong
on the FASTQ path, the bug is in trimming or alignment.

The reference is random sequence, not a real tRNA. That is deliberate: the
test is about coordinate arithmetic, which does not care what the bases are,
and shipping an invented "tRNA" sequence in a repository invites someone to
mistake it for real data later.

Model
-----
Molecules of length L are either cleaved or not. A cut "between position C and
C+1" (1-based, matching how cleavage sites are reported in the literature)
splits a molecule into a 5' product ending exactly at C and a 3' product
beginning exactly at C+1. Both products, and uncleaved molecules, are then
randomly fragmented, which preserves the cut-defined end and randomises the
other. Fragments outside the size-selection window are discarded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

ADAPTER_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
ADAPTER_R2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def make_fragments(
    ref_len: int,
    n_molecules: int,
    cut_pos: int | None,
    cleaved_fraction: float,
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """
    Return fragments as 0-based inclusive (start, end) index pairs.

    `cut_pos` is 1-based: a cut between base `cut_pos` and `cut_pos + 1`.
    """
    frags: list[tuple[int, int]] = []
    cut_idx = None if cut_pos is None else cut_pos - 1

    for _ in range(n_molecules):
        cleaved = cut_idx is not None and rng.random() < cleaved_fraction

        if cleaved:
            start = rng.randint(0, cut_idx)
            frags.append((start, cut_idx))
            end = rng.randint(cut_idx + 1, ref_len - 1)
            frags.append((cut_idx + 1, end))
        else:
            a = rng.randint(0, ref_len - 1)
            b = rng.randint(0, ref_len - 1)
            frags.append((min(a, b), max(a, b)))

    return [(s, e) for s, e in frags if min_len <= (e - s + 1) <= max_len]


def write_pair(
    frags: list[tuple[int, int]],
    ref_name: str,
    ref_seq: str,
    read_len: int,
    prefix: Path,
    label: str,
    error_rate: float,
    rng: random.Random,
) -> int:
    """Write R1/R2 FASTQ and a ground-truth SAM. Returns fragments written."""
    r1_path = prefix.with_name(f"{prefix.name}_{label}_R1.fastq.gz")
    r2_path = prefix.with_name(f"{prefix.name}_{label}_R2.fastq.gz")
    sam_path = prefix.with_name(f"{prefix.name}_{label}.sam")

    def mutate(seq: str) -> str:
        if error_rate <= 0:
            return seq
        out = []
        for base in seq:
            if rng.random() < error_rate:
                out.append(rng.choice([b for b in "ACGT" if b != base]))
            else:
                out.append(base)
        return "".join(out)

    with gzip.open(r1_path, "wt") as r1, gzip.open(r2_path, "wt") as r2, open(
        sam_path, "w"
    ) as sam:
        sam.write("@HD\tVN:1.6\tSO:unsorted\n")
        sam.write(f"@SQ\tSN:{ref_name}\tLN:{len(ref_seq)}\n")
        sam.write("@PG\tID:simulate.py\tPN:simulate.py\n")

        for i, (start, end) in enumerate(frags, start=1):
            frag_len = end - start + 1
            insert = ref_seq[start : end + 1]
            name = f"sim{i}"

            r1_seq = (insert + ADAPTER_R1)[:read_len]
            r2_seq = (revcomp(insert) + ADAPTER_R2)[:read_len]
            qual = "I" * read_len

            r1.write(f"@{name} 1:N:0:1\n{mutate(r1_seq)}\n+\n{qual}\n")
            r2.write(f"@{name} 2:N:0:1\n{mutate(r2_seq)}\n+\n{qual}\n")

            aln_len = min(read_len, frag_len)
            r1_pos = start + 1
            r2_left = max(end - aln_len + 1, start)
            r2_pos = r2_left + 1
            r1_aligned = ref_seq[start : start + aln_len]
            r2_aligned = revcomp(ref_seq[r2_left : r2_left + aln_len])
            cig = f"{aln_len}M"
            q = "I" * aln_len

            sam.write(
                f"{name}\t99\t{ref_name}\t{r1_pos}\t42\t{cig}\t=\t{r2_pos}\t"
                f"{frag_len}\t{r1_aligned}\t{q}\n"
            )
            sam.write(
                f"{name}\t147\t{ref_name}\t{r2_pos}\t42\t{cig}\t=\t{r1_pos}\t"
                f"{-frag_len}\t{r2_aligned}\t{q}\n"
            )

    return len(frags)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-o", "--prefix", default="data/sim/sim", help="output path prefix")
    ap.add_argument("--ref-len", type=int, default=76, help="reference length (nt)")
    ap.add_argument(
        "--cut-pos",
        type=int,
        default=34,
        help="1-based position; cut falls between this base and the next",
    )
    ap.add_argument("--molecules", type=int, default=200_000)
    ap.add_argument(
        "--cleaved-fraction",
        type=float,
        default=0.3,
        help="fraction of molecules cleaved in the treatment sample",
    )
    ap.add_argument("--read-len", type=int, default=75)
    ap.add_argument("--min-frag", type=int, default=17)
    ap.add_argument("--max-frag", type=int, default=100)
    ap.add_argument("--error-rate", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ref-name", default="sim_tRNA")
    args = ap.parse_args(argv)

    if not 1 <= args.cut_pos < args.ref_len:
        ap.error("--cut-pos must be inside the reference and leave room for a 3' product")

    rng = random.Random(args.seed)
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    ref_seq = random_sequence(args.ref_len, rng)
    ref_path = prefix.with_name(f"{prefix.name}_ref.fa")
    ref_path.write_text(
        f">{args.ref_name} synthetic sequence for pipeline validation, not a real tRNA\n"
        f"{ref_seq}\n"
    )

    treat_frags = make_fragments(
        args.ref_len,
        args.molecules,
        args.cut_pos,
        args.cleaved_fraction,
        args.min_frag,
        args.max_frag,
        rng,
    )
    ctrl_frags = make_fragments(
        args.ref_len,
        args.molecules,
        None,
        0.0,
        args.min_frag,
        args.max_frag,
        rng,
    )

    n_treat = write_pair(
        treat_frags, args.ref_name, ref_seq, args.read_len, prefix,
        "treat", args.error_rate, rng,
    )
    n_ctrl = write_pair(
        ctrl_frags, args.ref_name, ref_seq, args.read_len, prefix,
        "ctrl", args.error_rate, rng,
    )

    truth = {
        "ref_name": args.ref_name,
        "ref_len": args.ref_len,
        "cut_between": [args.cut_pos, args.cut_pos + 1],
        "expected_three_prime_end_at": args.cut_pos,
        "expected_five_prime_end_at": args.cut_pos + 1,
        "cleaved_fraction": args.cleaved_fraction,
        "molecules_simulated": args.molecules,
        "fragments_treat": n_treat,
        "fragments_ctrl": n_ctrl,
        "read_len": args.read_len,
        "size_selection": [args.min_frag, args.max_frag],
        "error_rate": args.error_rate,
        "seed": args.seed,
    }
    truth_path = prefix.with_name(f"{prefix.name}_truth.json")
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")

    print(f"reference         {ref_path}")
    print(f"cut between       {args.cut_pos} and {args.cut_pos + 1} (1-based)")
    print(f"  -> 3' ends expected to pile up at {args.cut_pos}")
    print(f"  -> 5' ends expected to pile up at {args.cut_pos + 1}")
    print(f"treatment         {n_treat} fragments")
    print(f"control           {n_ctrl} fragments")
    print(f"truth             {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
