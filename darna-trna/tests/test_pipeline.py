#!/usr/bin/env python3
"""
Validation: does the pipeline recover a cleavage site we planted ourselves?

A one-base coordinate error produces a clean, confident, wrong answer with no
exception anywhere, so the only way to catch it is to run data whose answer is
already known.

Run:  python3 -I tests/test_pipeline.py
Exit code 0 means every check passed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import count_ends  # noqa: E402
from analyze import analyse, load_counts, poisson_sf  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def test_cigar() -> None:
    check("ref_span 34M", count_ends.ref_span("34M") == 34)
    check("ref_span 30M2S ignores soft clip", count_ends.ref_span("30M2S") == 30)
    check("ref_span 10M2D10M counts deletion", count_ends.ref_span("10M2D10M") == 22)
    check("ref_span 10M2I10M ignores insertion", count_ends.ref_span("10M2I10M") == 20)
    check("ref_span unmapped", count_ends.ref_span("*") is None)
    check("clip_lengths 3S40M2S", count_ends.clip_lengths("3S40M2S") == (3, 2))


def test_ends_orientation() -> None:
    fwd = count_ends.alignment_ends(9, 20, False, "20M", False)
    rev = count_ends.alignment_ends(9, 20, True, "20M", False)
    check("forward 5' is leftmost", fwd == (9, 28), f"got {fwd}")
    check("reverse 5' is rightmost", rev == (28, 9), f"got {rev}")
    ext = count_ends.alignment_ends(9, 20, False, "2S20M3S", True)
    check("extend_clipped widens span", ext == (7, 31), f"got {ext}")


def test_one_based_conversion() -> None:
    sam = (
        "@HD\tVN:1.6\n"
        "@SQ\tSN:ref\tLN:10\n"
        "r1\t0\tref\t1\t42\t5M\t*\t0\t0\tACGTA\tIIIII\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".sam", delete=False) as fh:
        fh.write(sam)
        path = fh.name
    counters, _ = count_ends.count(path, end_mode="read")
    five = counters["ref"].five[0]
    three = counters["ref"].three[0]
    check("POS=1 maps to index 0", five[0] == 1 and five[1:].sum() == 0,
          f"five={list(five)}")
    check("5M alignment ends at index 4", three[4] == 1 and three.sum() == 1,
          f"three={list(three)}")
    cov = counters["ref"].coverage()[0]
    check("coverage spans exactly 5 bases",
          list(cov) == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], f"cov={list(cov)}")
    Path(path).unlink()


def test_poisson() -> None:
    check("poisson_sf(0, 5) == 1", abs(poisson_sf(0, 5.0) - 1.0) < 1e-12)
    check("poisson_sf(1, 1) ~ 0.6321",
          abs(poisson_sf(1, 1.0) - 0.6321205588) < 1e-6,
          f"got {poisson_sf(1, 1.0)}")
    check("large observation is significant", poisson_sf(100, 1.0) < 1e-50)


def test_recovers_planted_site(tmp: Path, cut_pos: int = 34) -> None:
    import simulate

    prefix = tmp / "sim"
    simulate.main([
        "-o", str(prefix),
        "--cut-pos", str(cut_pos),
        "--molecules", "40000",
        "--cleaved-fraction", "0.3",
        "--seed", "7",
    ])

    truth = json.loads((tmp / "sim_truth.json").read_text())
    check("truth file records the cut", truth["cut_between"] == [cut_pos, cut_pos + 1])

    for label in ("treat", "ctrl"):
        count_ends.main([
            str(tmp / f"sim_{label}.sam"),
            "-o", str(tmp / f"{label}.npz"),
        ])

    treat = load_counts(tmp / "treat.npz")
    ctrl = load_counts(tmp / "ctrl.npz")
    per_ref, rows = analyse(treat, ctrl, norm="repm", min_count=10)

    ref = truth["ref_name"]
    sig = per_ref[ref]

    three_peak = int(np.argmax(sig["three_diff"])) + 1
    five_peak = int(np.argmax(sig["five_diff"])) + 1
    check(f"3' ends peak at {cut_pos}", three_peak == cut_pos, f"got {three_peak}")
    check(f"5' ends peak at {cut_pos + 1}", five_peak == cut_pos + 1, f"got {five_peak}")

    three = sig["three_diff"]
    runner_up = np.sort(three)[-2]
    ratio = three.max() / max(runner_up, 1e-9)
    check("3' peak exceeds runner-up by >10x", ratio > 10, f"ratio {ratio:.1f}")

    top_two = {(r["pos"], r["end"]) for r in rows[:2]}
    check("planted positions rank top two",
          top_two == {(cut_pos, "three"), (cut_pos + 1, "five")},
          f"got {sorted(top_two)}")

    sig_rows = [r for r in rows if r.get("q", 1.0) < 0.01]
    check("only planted positions are significant at q<0.01",
          len(sig_rows) == 2, f"{len(sig_rows)} significant")


def test_control_has_no_site(tmp: Path) -> None:
    import simulate

    prefix = tmp / "null"
    simulate.main([
        "-o", str(prefix), "--cut-pos", "34", "--molecules", "30000",
        "--cleaved-fraction", "0.0", "--seed", "11",
    ])
    for label in ("treat", "ctrl"):
        count_ends.main([
            str(tmp / f"null_{label}.sam"), "-o", str(tmp / f"null_{label}.npz"),
        ])
    a = load_counts(tmp / "null_treat.npz")
    b = load_counts(tmp / "null_ctrl.npz")
    _, rows = analyse(a, b, norm="repm", min_count=10)
    hits = [r for r in rows if r.get("q", 1.0) < 0.01]
    check("no significant sites when nothing was cut",
          len(hits) == 0,
          f"{len(hits)} false positives, top: {hits[:2]}")


def test_shifted_site_tracks(tmp: Path) -> None:
    import simulate

    found = []
    for cut in (20, 50):
        prefix = tmp / f"shift{cut}"
        simulate.main([
            "-o", str(prefix), "--cut-pos", str(cut), "--molecules", "30000",
            "--cleaved-fraction", "0.3", "--seed", "3",
        ])
        for label in ("treat", "ctrl"):
            count_ends.main([
                str(tmp / f"shift{cut}_{label}.sam"),
                "-o", str(tmp / f"shift{cut}_{label}.npz"),
            ])
        per_ref, _ = analyse(
            load_counts(tmp / f"shift{cut}_treat.npz"),
            load_counts(tmp / f"shift{cut}_ctrl.npz"),
            norm="repm", min_count=10,
        )
        sig = per_ref["sim_tRNA"]
        found.append(int(np.argmax(sig["three_diff"])) + 1)
    check("peak follows the planted position", found == [20, 50], f"got {found}")


def main() -> int:
    test_cigar()
    test_ends_orientation()
    test_one_based_conversion()
    test_poisson()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_recovers_planted_site(tmp)
        test_control_has_no_site(tmp)
        test_shifted_site_tracks(tmp)

    width = max(len(name) for _, name, _ in results)
    failed = 0
    for status, name, detail in results:
        suffix = f"   {detail}" if detail and status == FAIL else ""
        print(f"[{status}] {name:<{width}}{suffix}")
        failed += status == FAIL

    print()
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
