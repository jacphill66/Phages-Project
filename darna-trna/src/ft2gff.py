import re, sys, gzip
inp, out = sys.argv[1], sys.argv[2]
op = lambda p: gzip.open(p,"rt") if p.endswith(".gz") else open(p)
loc = re.compile(r"^(<?\d+)\t(>?\d+)\t(\S+)")
qual = re.compile(r"^\t\t\t(\S+)\t(.*)$")
seqid, feats, cur = "unknown", [], None
with op(inp) as fh:
    for line in fh:
        if line.startswith(">Feature"):
            m = re.search(r"(?:ref\|)?([A-Za-z0-9_.]+)", line.split(None,1)[1] if len(line.split(None,1))>1 else "")
            if m: seqid = m.group(1).rstrip("|")
            cur = None; continue
        m = loc.match(line)
        if m:
            a = int(m.group(1).lstrip("<")); b = int(m.group(2).lstrip(">"))
            strand = "+" if a <= b else "-"
            cur = {"s": min(a,b), "e": max(a,b), "st": strand, "t": m.group(3), "q": {}}
            feats.append(cur); continue
        m = qual.match(line)
        if m and cur is not None:
            cur["q"].setdefault(m.group(1), m.group(2))

last = {}
for f in feats:
    key = (f["s"], f["e"], f["st"])
    if f["t"] == "gene":
        last[key] = f["q"]
    elif key in last:
        for k, v in last[key].items():
            f["q"].setdefault(k, v)

keep = ("tRNA","rRNA","ncRNA","tmRNA","CDS","gene")
n = 0
with open(out,"w") as fh:
    fh.write("##gff-version 3\n")
    for f in feats:
        if f["t"] not in keep: continue
        nm = f["q"].get("gene") or f["q"].get("locus_tag") or "?"
        attrs = "ID=f%d;gene=%s" % (n, nm)
        if f["q"].get("locus_tag"): attrs += ";locus_tag=%s" % f["q"]["locus_tag"]
        if f["q"].get("product"):   attrs += ";product=%s" % f["q"]["product"]
        fh.write("\t".join([seqid,"ncbi_ft",f["t"],str(f["s"]),str(f["e"]),".",
                            f["st"],".",attrs])+"\n")
        n += 1
from collections import Counter
c = Counter(f["t"] for f in feats if f["t"] in keep)
print("seqid: %s" % seqid)
print("wrote %d features to %s" % (n, out))
for k,v in sorted(c.items(), key=lambda kv:-kv[1]): print("  %-10s %d" % (k,v))
