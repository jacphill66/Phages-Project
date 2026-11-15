#!/usr/bin/env python3
"""
Compare read-end counts between a treatment and a control sample.

Produces three things per position:

  REPM difference    Read Ends Per Million in treatment minus control, as in
                     the source paper. Normalising by each sample's total end
                     count makes libraries of different depth comparable.

  Coverage ratio     log2 of normalised treatment coverage over control. A
                     cut shows up as a step down on the 3' side of the site,
                     because cleaved molecules contribute coverage before the
                     cut but not after.

  Poisson p-value    One-sided tail probability of seeing at least the
                     observed treatment count, given the depth-scaled control
                     count as the expectation, with Benjamini-Hochberg FDR
                     across all tested positions. The source paper reports no
                     statistics on its sequencing figures, so this is ours
                     rather than a reproduction. It assumes positions are
                     independent, which they are not, so treat it as a ranking
                     aid rather than a hypothesis test.

A caveat worth carrying into the write-up: REPM assumes total end counts are
comparable between samples. Here one culture has an active abortive-infection
system arresting its growth and the other does not, so that assumption is
doing real work. --norm total-coverage offers an alternative that normalises
by summed coverage instead, which responds differently to a global shift in
how much RNA is fragmented.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

PSEUDO = 1.0


def _poisson_logpmf(i: int, lam: float) -> float:
    return i * math.log(lam) - lam - math.lgamma(i + 1)


def poisson_sf(k: int, lam: float) -> float:
    """
    P(X >= k) for X ~ Poisson(lam). Pure stdlib; no scipy dependency.

    When k is above the mean the upper tail is summed directly rather than
    computed as one minus the lower tail. The subtraction loses all precision
    once the lower tail rounds to 1.0, which pins every strongly enriched
    position at the same apparent p-value and destroys their ranking -- the
    exact positions we care most about ordering correctly.
    """
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0

    if k > lam:
        total = 0.0
        i = k
        while True:
            term = math.exp(_poisson_logpmf(i, lam))
            total += term
            i += 1
            if term <= total * 1e-17 or (i > k + 20 and term == 0.0):
                break
            if i > k + 10_000:
                break
        return min(1.0, total)

    lower = 0.0
    for i in range(k):
        lower += math.exp(_poisson_logpmf(i, lam))
    return max(0.0, 1.0 - lower)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values, same order as input."""
    n = pvals.size
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def load_counts(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Reshape a count_ends .npz into {ref: {five, three, cov}} (strands summed)."""
    raw = np.load(path)
    refs: dict[str, dict[str, np.ndarray]] = {}
    for key in raw.files:
        ref, kind, _strand = key.split("|")
        slot = refs.setdefault(ref, {})
        if kind in slot:
            slot[kind] = slot[kind] + raw[key]
        else:
            slot[kind] = raw[key].astype(np.float64)
    return refs


def scale_factor(refs: dict[str, dict[str, np.ndarray]], mode: str) -> float:
    if mode == "repm":
        total = sum(float(d["five"].sum() + d["three"].sum()) for d in refs.values())
    elif mode == "total-coverage":
        total = sum(float(d["cov"].sum()) for d in refs.values())
    else:
        raise ValueError(mode)
    if total <= 0:
        raise SystemExit(f"sample has zero signal under --norm {mode}")
    return 1e6 / total


def analyse(
    treat: dict[str, dict[str, np.ndarray]],
    ctrl: dict[str, dict[str, np.ndarray]],
    norm: str,
    min_count: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    t_scale = scale_factor(treat, norm)
    c_scale = scale_factor(ctrl, norm)
    # total_treat / total_ctrl. Multiplying a control count by this puts it on
    # the treatment library's depth, which is the Poisson expectation we test
    # the observed treatment count against.
    depth_ratio = c_scale / t_scale

    shared = sorted(set(treat) & set(ctrl))
    if not shared:
        raise SystemExit("treatment and control share no reference sequences")

    per_ref: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    raw_p: list[float] = []

    for ref in shared:
        t, c = treat[ref], ctrl[ref]
        n = min(t["five"].size, c["five"].size)

        five_diff = t["five"][:n] * t_scale - c["five"][:n] * c_scale
        three_diff = t["three"][:n] * t_scale - c["three"][:n] * c_scale
        total_diff = five_diff + three_diff

        cov_ratio = np.log2(
            (t["cov"][:n] * t_scale + PSEUDO) / (c["cov"][:n] * c_scale + PSEUDO)
        )

        per_ref[ref] = {
            "five_diff": five_diff,
            "three_diff": three_diff,
            "total_diff": total_diff,
            "cov_ratio": cov_ratio,
            "treat_five": t["five"][:n],
            "treat_three": t["three"][:n],
            "ctrl_five": c["five"][:n],
            "ctrl_three": c["three"][:n],
        }

        for i in range(n):
            for kind in ("five", "three"):
                obs = int(t[kind][i])
                if obs < min_count:
                    continue
                lam = max(float(c[kind][i]) * depth_ratio, 0.5)
                p = poisson_sf(obs, lam)
                raw_p.append(p)
                rows.append(
                    {
                        "ref": ref,
                        "pos": i + 1,
                        "end": kind,
                        "treat": obs,
                        "ctrl": int(c[kind][i]),
                        "expected": lam,
                        "repm_diff": float(
                            (five_diff if kind == "five" else three_diff)[i]
                        ),
                        "cov_ratio": float(cov_ratio[i]),
                        "p": p,
                    }
                )

    if rows:
        q = benjamini_hochberg(np.array(raw_p))
        for row, qv in zip(rows, q):
            row["q"] = float(qv)
    rows.sort(key=lambda r: (-r["repm_diff"], r["p"]))
    return per_ref, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--treat", required=True, type=Path, help="treatment .npz")
    ap.add_argument("--ctrl", required=True, type=Path, help="control .npz")
    ap.add_argument("--out", required=True, type=Path, help="output .npz of signals")
    ap.add_argument("--table", type=Path, help="ranked positions TSV")
    ap.add_argument(
        "--norm",
        choices=("repm", "total-coverage"),
        default="repm",
        help="normalisation; repm matches the source paper",
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="skip positions with fewer treatment ends than this when testing",
    )
    ap.add_argument("--top", type=int, default=25, help="rows to print to stdout")
    args = ap.parse_args(argv)

    treat = load_counts(args.treat)
    ctrl = load_counts(args.ctrl)
    per_ref, rows = analyse(treat, ctrl, args.norm, args.min_count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **{f"{ref}|{key}": arr for ref, d in per_ref.items() for key, arr in d.items()},
    )

    header = ["ref", "pos", "end", "treat", "ctrl", "expected", "repm_diff", "cov_ratio", "p", "q"]
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        with open(args.table, "w") as fh:
            fh.write("\t".join(header) + "\n")
            for r in rows:
                fh.write(
                    f"{r['ref']}\t{r['pos']}\t{r['end']}\t{r['treat']}\t{r['ctrl']}\t"
                    f"{r['expected']:.2f}\t{r['repm_diff']:.1f}\t{r['cov_ratio']:.3f}\t"
                    f"{r['p']:.3e}\t{r.get('q', float('nan')):.3e}\n"
                )

    print(f"{'ref':<16}{'pos':>5}{'end':>7}{'treat':>9}{'ctrl':>8}{'repm_diff':>12}{'q':>11}")
    for r in rows[: args.top]:
        print(
            f"{r['ref']:<16}{r['pos']:>5}{r['end']:>7}{r['treat']:>9}{r['ctrl']:>8}"
            f"{r['repm_diff']:>12.1f}{r.get('q', float('nan')):>11.2e}"
        )
    if not rows:
        print("(no positions passed --min-count)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
