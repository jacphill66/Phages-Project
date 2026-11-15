import csv, gzip, sys
from bisect import bisect_right
from collections import defaultdict

gff, csvf = sys.argv[1], sys.argv[2]
op = lambda p: gzip.open(p,"rt",newline="") if p.endswith(".gz") else open(p,newline="")

tr = []
with op(gff) as fh:
    for line in fh:
        if line[0]=="#": continue
        f = line.split("\t")
        if len(f)<9 or f[2]!="tRNA": continue
        a = dict(p.split("=",1) for p in f[8].rstrip(";\n").split(";") if "=" in p)
        tr.append((int(f[3]), int(f[4]), f[6], a.get("gene") or a.get("locus_tag") or "?",
                   a.get("product","")))
tr.sort()
byst = defaultdict(int)
for t in tr: byst[t[2]] += 1
print("tRNA genes in annotation: %d   (+: %d, -: %d)" % (len(tr), byst["+"], byst["-"]))

starts = [x[0] for x in tr]; mx = max(b-a+1 for a,b,*_ in tr)
def find(p):
    hi = bisect_right(starts,p); lo = bisect_right(starts,p-mx-1)
    h = [x for x in tr[lo:hi] if x[0]<=p<=x[1]]
    return h[0] if h else None

with op(csvf) as fh: rows = list(csv.DictReader(fh))

tot = defaultdict(float); npos = defaultdict(int); meta = {}
present = defaultdict(int)
for r in rows:
    p = int(r["position"]); t = find(p)
    if not t: continue
    nm = t[3]; meta[nm] = t
    present[nm] += 1
    d = float(r["countdiff"])
    if d < 0: tot[nm] += -d; npos[nm] += 1

print("\ntRNAs ranked by TOTAL DARNA-gained signal (paper's stated ranking):")
print("%-10s %3s %8s %12s %7s %7s" % ("tRNA","str","len","total_gain","n_pos","n_rows"))
for nm,v in sorted(tot.items(), key=lambda kv:-kv[1])[:20]:
    a,b,st,_,pr = meta[nm]
    print("%-10s %3s %8d %12.1f %7d %7d" % (nm, st, b-a+1, v, npos[nm], present[nm]))

print("\nStrand of the tRNAs with any DARNA-gained signal:")
s = defaultdict(int)
for nm in tot: s[meta[nm][2]] += 1
print("  +: %d    -: %d" % (s["+"], s["-"]))

print("\nStrand of ALL tRNAs appearing anywhere in the table (gain or not):")
s2 = defaultdict(int)
for nm in present: s2[meta[nm][2]] += 1
print("  +: %d    -: %d" % (s2["+"], s2["-"]))

print("\nThe four tRNAs the paper names:")
for want in ("trpT","thrU","tyrT","tyrV"):
    hit = [t for t in tr if t[3]==want]
    if not hit:
        print("  %-6s NOT in annotation under this name" % want); continue
    a,b,st,nm,pr = hit[0]
    print("  %-6s %s:%d-%d strand %s  %s" % (nm,"NC_000913",a,b,st,pr))
    print("         rows in table: %d   total DARNA gain: %.1f" % (present.get(nm,0), tot.get(nm,0.0)))
    sub = [r for r in rows if a <= int(r["position"]) <= b]
    for r in sorted(sub, key=lambda r: float(r["countdiff"]))[:6]:
        p = int(r["position"]); off = (b-p+1) if st=="-" else (p-a+1)
        print("           pos %d (%s) at %d/%d  ctrl %.0f  DARNA %.0f  diff %.1f" % (
            p, r["end"], off, b-a+1, float(r["Count_3203_MG"]),
            float(r["Count_3204_MG"]), float(r["countdiff"])))
