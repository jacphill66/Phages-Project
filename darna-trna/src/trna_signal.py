import csv, gzip, sys
from bisect import bisect_right
from collections import defaultdict

gff, csvf = sys.argv[1], sys.argv[2]
op = lambda p: gzip.open(p,"rt",newline="") if p.endswith(".gz") else open(p,newline="")

tr, allf = [], []
with op(gff) as fh:
    for line in fh:
        if line[0]=="#": continue
        f = line.split("\t")
        if len(f)<9: continue
        if f[2] not in ("tRNA","rRNA","ncRNA","tmRNA","CDS"): continue
        a = dict(p.split("=",1) for p in f[8].rstrip(";\n").split(";") if "=" in p)
        nm = a.get("gene") or a.get("locus_tag") or "?"
        rec = (int(f[3]), int(f[4]), f[6], f[2], nm)
        allf.append(rec)
        if f[2]=="tRNA": tr.append(rec)
tr.sort(); allf.sort()
ts = [x[0] for x in tr]; tmax = max(b-a+1 for a,b,*_ in tr)
as_ = [x[0] for x in allf]; amax = max(b-a+1 for a,b,*_ in allf)
print("tRNA features: %d   all features: %d" % (len(tr), len(allf)), file=sys.stderr)

def find(pos, feats, starts, mx):
    hi = bisect_right(starts,pos); lo = bisect_right(starts,pos-mx-1)
    h = [x for x in feats[lo:hi] if x[0]<=pos<=x[1]]
    h.sort(key=lambda x:(x[3]=="CDS", x[1]-x[0]))
    return h[0] if h else None

with op(csvf) as fh: rows = list(csv.DictReader(fh))

mass = defaultdict(float); hits = []
for r in rows:
    d = float(r["countdiff"])
    if d >= 0: continue
    p = int(r["position"])
    g = find(p, allf, as_, amax)
    mass[g[3] if g else "unannotated"] += -d
    t = find(p, tr, ts, tmax)
    if t:
        a,b,st,_,nm = t
        off = (b-p+1) if st=="-" else (p-a+1)
        hits.append((d, p, r["end"], nm, st, off, b-a+1,
                     float(r["Count_3203_MG"]), float(r["Count_3204_MG"]),
                     float(r["normcountratio"])))

tot = sum(mass.values())
print("\nWhere the DARNA-gained signal sits:")
for k,v in sorted(mass.items(), key=lambda kv:-kv[1]):
    print("  %-14s %12.0f  %5.1f%%" % (k, v, 100*v/tot))

print("\nTop tRNA-internal sites by difference:")
print("%9s %6s %-18s %3s %7s %10s %11s %9s" % ("position","end","tRNA","str","at/len","DARNA","diff","foldchg"))
for h in sorted(hits)[:20]:
    d,p,e,nm,st,off,L,c3,c4,ncr = h
    print("%9d %6s %-18s %3s %4d/%-3d %10.0f %11.1f %9.1f" % (p,e,nm[:18],st,off,L,c4,d,1/ncr if ncr else 0))

print("\nTop tRNA-internal sites by fold change (>=500 DARNA reads):")
big = [h for h in hits if h[8] >= 500]
for h in sorted(big, key=lambda x: -(1/x[9] if x[9] else 0))[:20]:
    d,p,e,nm,st,off,L,c3,c4,ncr = h
    print("%9d %6s %-18s %3s %4d/%-3d %10.0f %11.1f %9.1f" % (p,e,nm[:18],st,off,L,c4,d,1/ncr if ncr else 0))

print("\nOffset of DARNA-gained ends within tRNAs (signal-weighted):")
hist = defaultdict(float)
for h in hits: hist[h[5]] += -h[0]
mx = max(hist.values()) if hist else 1
for off in sorted(hist):
    if hist[off] > mx*0.01:
        print("  %3d %-50s %.0f" % (off, "#"*int(50*hist[off]/mx), hist[off]))
