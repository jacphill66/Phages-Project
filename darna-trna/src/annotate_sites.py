import csv, gzip, sys
from bisect import bisect_right

gff, csvf, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
op = lambda p: gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")

feats = []
with op(gff) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t")
        if len(f) < 9 or f[2] not in ("tRNA","rRNA","ncRNA","tmRNA","CDS"): continue
        a = dict(p.split("=",1) for p in f[8].rstrip(";\n").split(";") if "=" in p)
        nm = a.get("gene") or a.get("locus_tag") or "?"
        if f[2] != "CDS": nm += " [%s]" % a.get("product","")
        feats.append((int(f[3]), int(f[4]), f[6], f[2], nm))
feats.sort()
starts = [x[0] for x in feats]
maxlen = max(b-a+1 for a,b,*_ in feats)

def at(pos):
    hi = bisect_right(starts, pos)
    lo = bisect_right(starts, pos - maxlen - 1)
    h = [x for x in feats[lo:hi] if x[0] <= pos <= x[1]]
    h.sort(key=lambda x: (x[3]=="CDS", x[1]-x[0]))
    return h[0] if h else None

with op(csvf) as fh:
    rows = list(csv.DictReader(fh))
rows.sort(key=lambda r: float(r["countdiff"]))

for label, sel in (("GAINED in DARNA (cleavage sites)", rows[:n]),
                   ("LOST in DARNA (intact depleted)", rows[::-1][:n])):
    print("\n%s\n%s" % (label, "-"*92))
    print("%9s %6s %10s %11s  %-32s %6s %7s" % ("position","end","DARNA","diff","feature","strand","at/len"))
    for r in sel:
        p = int(r["position"]); f = at(p)
        if f: a,b,st,_,nm = f; off = (b-p+1) if st=="-" else (p-a+1); L = b-a+1
        else: nm, st, off, L = "(none)", ".", 0, 0
        print("%9d %6s %10.0f %11.1f  %-32s %6s %4d/%-4d" % (
            p, r["end"], float(r["Count_3204_MG"]), float(r["countdiff"]), nm[:32], st, off, L))
