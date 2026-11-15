#!/usr/bin/env python3
"""
Count read 5' and 3' ends, and per-base coverage, at every position of every
reference sequence in a SAM file.

This is the core measurement of the project. A ribonuclease that cuts at a
fixed position creates a new molecular end there in every molecule it cuts,
so a cleavage site shows up as an excess of read ends at one coordinate.

Coordinate conventions used internally
--------------------------------------
All arrays are 0-based, so array index i corresponds to SAM position i+1.
Conversion happens exactly twice: when parsing POS (subtract 1) and when
writing human-readable output (add 1). Nowhere else.

For a read aligned to the forward strand, the molecule's 5' end is the
leftmost aligned base and its 3' end is the rightmost. For a read aligned to
the reverse strand this is flipped: the 5' end is the RIGHTMOST genomic
coordinate. Getting this backwards silently relocates the signal, so it is
handled in one place (`alignment_ends`) and tested.

End-mode policy
---------------
The source paper says only that it enumerated "read ends", which is
ambiguous for paired-end data. Two interpretations are implemented:

  read      Every primary alignment contributes its own 5' and 3' end.
            Maximum data, but in a small-RNA library where both mates cover
            the same short insert, each fragment is counted twice.

  fragment  Each properly-paired fragment contributes exactly two ends: the
            5'-most base of read 1 (the fragment's 5' end) and the 5'-most
            base of read 2 (the fragment's 3' end). This assumes a library
            where read 1 begins at the insert's 5' end, which is the case
            for the Lexogen small RNA kit used in the source paper.

Report which mode you used. The choice changes absolute counts, and can
change relative ranking between references.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np

CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")

REF_CONSUMING = frozenset("MDN=X")
CLIP_OPS = frozenset("SH")

FLAG_PAIRED = 0x1
FLAG_PROPER_PAIR = 0x2
FLAG_UNMAPPED = 0x4
FLAG_REVERSE = 0x10
FLAG_READ1 = 0x40
FLAG_READ2 = 0x80
FLAG_SECONDARY = 0x100
FLAG_QCFAIL = 0x200
FLAG_DUPLICATE = 0x400
FLAG_SUPPLEMENTARY = 0x800

SKIP_FLAGS = FLAG_UNMAPPED | FLAG_SECONDARY | FLAG_QCFAIL | FLAG_SUPPLEMENTARY


def ref_span(cigar: str) -> int | None:
    """Number of reference bases the alignment covers. None if unavailable."""
    if cigar == "*":
        return None
    total = 0
    for n, op in CIGAR_RE.findall(cigar):
        if op in REF_CONSUMING:
            total += int(n)
    return total if total > 0 else None


def clip_lengths(cigar: str) -> tuple[int, int]:
    """Soft/hard-clipped base counts at the left and right ends of the read."""
    ops = CIGAR_RE.findall(cigar)
    left = right = 0
    for n, op in ops:
        if op in CLIP_OPS:
            left += int(n)
        else:
            break
    for n, op in reversed(ops):
        if op in CLIP_OPS:
            right += int(n)
        else:
            break
    return left, right


def alignment_ends(
    pos0: int, span: int, is_reverse: bool, cigar: str, extend_clipped: bool
) -> tuple[int, int]:
    """
    Return (five_prime_index, three_prime_index), 0-based and inclusive.

    If `extend_clipped` is set, clipped bases are added back onto the
    alignment boundaries. The aligner may clip a few bases rather than
    penalise mismatches, which shortens the apparent molecule; extending
    recovers the intended end at the cost of trusting bases the aligner
    rejected. Off by default.
    """
    left = pos0
    right = pos0 + span - 1
    if extend_clipped:
        cl, cr = clip_lengths(cigar)
        left -= cl
        right += cr
    return (right, left) if is_reverse else (left, right)


class RefCounter:
    """Per-position tallies for one reference sequence, split by strand."""

    def __init__(self, length: int):
        self.length = length
        self.five = np.zeros((2, length), dtype=np.int64)
        self.three = np.zeros((2, length), dtype=np.int64)
        self._cov_delta = np.zeros((2, length + 1), dtype=np.int64)

    def add_end(self, strand: int, five_idx: int, three_idx: int) -> None:
        if 0 <= five_idx < self.length:
            self.five[strand, five_idx] += 1
        if 0 <= three_idx < self.length:
            self.three[strand, three_idx] += 1

    def add_span(self, strand: int, left: int, right: int) -> None:
        lo = max(0, left)
        hi = min(self.length - 1, right)
        if lo <= hi:
            self._cov_delta[strand, lo] += 1
            self._cov_delta[strand, hi + 1] -= 1

    def coverage(self) -> np.ndarray:
        return np.cumsum(self._cov_delta[:, : self.length], axis=1)


def parse_sq(line: str) -> tuple[str, int] | None:
    """Extract (name, length) from an @SQ header line."""
    name = length = None
    for field in line.rstrip("\n").split("\t")[1:]:
        if field.startswith("SN:"):
            name = field[3:]
        elif field.startswith("LN:"):
            length = int(field[3:])
    return (name, length) if name is not None and length is not None else None


def open_text(path: str):
    if path == "-":
        return sys.stdin
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def count(
    sam_path: str,
    end_mode: str = "fragment",
    min_mapq: int = 0,
    extend_clipped: bool = False,
    require_proper_pair: bool = True,
) -> tuple[dict[str, RefCounter], dict[str, int]]:
    """Tally ends and coverage across a SAM file. Returns (counters, stats)."""
    stats = {
        "lines": 0,
        "skipped_flag": 0,
        "skipped_mapq": 0,
        "skipped_mode": 0,
        "skipped_unknown_ref": 0,
        "skipped_bad_cigar": 0,
        "counted": 0,
    }

    counters: dict[str, RefCounter] = {}

    with open_text(sam_path) as handle:
        for line in handle:
            if line.startswith("@"):
                if line.startswith("@SQ"):
                    sq = parse_sq(line)
                    if sq is not None:
                        counters[sq[0]] = RefCounter(sq[1])
                continue
            if not counters:
                raise SystemExit(
                    "No @SQ header lines found before the first alignment. "
                    "Supply a SAM that retains its header (bowtie2 emits one "
                    "by default; do not strip it)."
                )
            stats["lines"] += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            flag = int(fields[1])
            if flag & SKIP_FLAGS:
                stats["skipped_flag"] += 1
                continue
            if int(fields[4]) < min_mapq:
                stats["skipped_mapq"] += 1
                continue

            ref = fields[2]
            counter = counters.get(ref)
            if counter is None:
                stats["skipped_unknown_ref"] += 1
                continue

            cigar = fields[5]
            span = ref_span(cigar)
            if span is None:
                stats["skipped_bad_cigar"] += 1
                continue

            is_reverse = bool(flag & FLAG_REVERSE)
            strand = 1 if is_reverse else 0
            pos0 = int(fields[3]) - 1

            five_idx, three_idx = alignment_ends(
                pos0, span, is_reverse, cigar, extend_clipped
            )

            if end_mode == "read":
                counter.add_end(strand, five_idx, three_idx)
            elif end_mode == "fragment":
                paired = bool(flag & FLAG_PAIRED)
                if not paired:
                    counter.add_end(strand, five_idx, three_idx)
                else:
                    if require_proper_pair and not (flag & FLAG_PROPER_PAIR):
                        stats["skipped_mode"] += 1
                        continue
                    if flag & FLAG_READ1:
                        counter.add_end(strand, five_idx, -1)
                    elif flag & FLAG_READ2:
                        counter.add_end(strand, -1, five_idx)
                    else:
                        stats["skipped_mode"] += 1
                        continue
            else:
                raise ValueError(f"unknown end_mode {end_mode!r}")

            left = min(five_idx, three_idx)
            right = max(five_idx, three_idx)
            counter.add_span(strand, left, right)
            stats["counted"] += 1

    return counters, stats


def save_npz(counters: dict[str, RefCounter], out_path: Path) -> None:
    payload: dict[str, np.ndarray] = {}
    for ref, c in counters.items():
        cov = c.coverage()
        payload[f"{ref}|five|fwd"] = c.five[0]
        payload[f"{ref}|five|rev"] = c.five[1]
        payload[f"{ref}|three|fwd"] = c.three[0]
        payload[f"{ref}|three|rev"] = c.three[1]
        payload[f"{ref}|cov|fwd"] = cov[0]
        payload[f"{ref}|cov|rev"] = cov[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def write_tsv(counters: dict[str, RefCounter], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("ref\tpos\tstrand\tfive\tthree\tcoverage\n")
        for ref, c in sorted(counters.items()):
            cov = c.coverage()
            for strand, label in ((0, "+"), (1, "-")):
                for i in range(c.length):
                    f, t, v = c.five[strand, i], c.three[strand, i], cov[strand, i]
                    if f or t or v:
                        fh.write(f"{ref}\t{i + 1}\t{label}\t{f}\t{t}\t{v}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("sam", help="aligned SAM file ('-' for stdin, .gz accepted)")
    ap.add_argument("-o", "--out", required=True, help="output .npz path")
    ap.add_argument("--tsv", help="also write a human-readable TSV here")
    ap.add_argument(
        "--end-mode",
        choices=("fragment", "read"),
        default="fragment",
        help="which ends to count (see module docstring); default: fragment",
    )
    ap.add_argument("--min-mapq", type=int, default=0)
    ap.add_argument(
        "--extend-clipped",
        action="store_true",
        help="add soft/hard-clipped bases back onto alignment boundaries",
    )
    ap.add_argument(
        "--any-pair",
        action="store_true",
        help="in fragment mode, accept pairs not flagged as properly paired",
    )
    args = ap.parse_args(argv)

    counters, stats = count(
        args.sam,
        end_mode=args.end_mode,
        min_mapq=args.min_mapq,
        extend_clipped=args.extend_clipped,
        require_proper_pair=not args.any_pair,
    )
    save_npz(counters, Path(args.out))
    if args.tsv:
        write_tsv(counters, Path(args.tsv))

    for key, value in stats.items():
        print(f"{key}\t{value}", file=sys.stderr)
    total_ends = sum(
        int(c.five.sum() + c.three.sum()) for c in counters.values()
    )
    print(f"total_ends\t{total_ends}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
