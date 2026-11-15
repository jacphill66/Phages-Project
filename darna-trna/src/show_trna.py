#!/usr/bin/env python3
"""
Print a gene's sequence with 1-based numbering, to settle where the anticodon
sits relative to the annotation's own coordinates.

    python3 show_trna.py <genome.fa> <gff> <geneName> [expectedAnticodon]

Minus-strand genes are reverse-complemented, so position 1 is always the
molecule's 5' end -- the numbering the literature uses.
"""
import gzip, sys

fa, gff, want_gene = sys.argv[1], sys.argv[2], sys.argv[3]
want = sys.argv[4].upper() if len(sys.argv) > 4 else None
op = lambda p: gzip.open(p, "rt") if p.endswith(".gz") else open(p)

feat = None
with op(gff) as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "tRNA":
            continue
        a = dict(p.split("=", 1) for p in f[8].rstrip(";\n").split(";") if "=" in p)
        if (a.get("gene") or a.get("locus_tag")) == want_gene:
            feat = (int(f[3]), int(f[4]), f[6], a.get("product", ""))
            break
if feat is None:
    sys.exit(f"{want_gene!r} not found as a tRNA in {gff}")
start, end, strand, product = feat

chunks = []
with op(fa) as fh:
    for line in fh:
        if not line.startswith(">"):
            chunks.append(line.strip())
genome = "".join(chunks)
print(f"genome length {len(genome)}")

seq = genome[start - 1 : end].upper()
if strand == "-":
    seq = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

print(f"{want_gene}  {start}-{end}  strand {strand}  {product}  length {len(seq)}\n")
for s in range(0, len(seq), 10):
    print(f"{s+1:>4}  {seq[s:s+10]}")

print(f"\n 33-35: {seq[32:35]}")
print(f" 34-36: {seq[33:36]}")
print(f" 35-37: {seq[34:37]}")
if want:
    hits = [i + 1 for i in range(len(seq) - len(want) + 1) if seq[i:i+len(want)] == want]
    print(f"\n{want!r} starts at position(s): {hits}")
    print("Anticodon at 34 means the annotation agrees with Sprinzl numbering.")
    print("Anticodon at 33 means the annotated 5' boundary is one base late.")
