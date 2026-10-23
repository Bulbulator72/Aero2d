"""Сравнение калибровочных прогонов (benchmarks/calib/*.jsonl) на общих точках."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "calib", "*.jsonl")))
sets = {}
for f in files:
    pts = {}
    for ln in open(f):
        d = json.loads(ln)
        if "cl" not in d:
            continue
        for a, ce, de, c, cd, ok in zip(d["alpha"], d["cl_exp"], d["cd_exp"], d["cl"], d["cd"], d["conv"]):
            if ok and np.isfinite(c) and np.isfinite(cd):
                pts[(d["name"], round(d["re"]), round(a, 2))] = (ce, de, c, cd)
    sets[os.path.basename(f).replace(".jsonl", "")] = pts
common = set.intersection(*[set(p) for p in sets.values()]) if sets else set()
print(f"общих точек: {len(common)}")
print(f"{'набор':<10} {'точек':>6} {'ΔCl ср':>8} {'|ΔCl|':>7} {'Cd мед':>8} {'|Cd| мед':>9} {'Cd 1e5':>8} {'Cd 2e5':>8}")
for k, pts in sets.items():
    P = [pts[c] for c in common]
    if not P:
        continue
    P = np.array(P)
    dcl = P[:, 2] - P[:, 0]
    rcd = P[:, 3] / P[:, 1] - 1
    re = np.array([c[1] for c in common])
    lo, hi = re < 1.5e5, re >= 1.5e5
    print(f"{k:<10} {len(pts):6d} {dcl.mean():+8.3f} {np.abs(dcl).mean():7.3f} {np.median(rcd):+8.1%} {np.median(np.abs(rcd)):9.1%}"
          f" {np.median(rcd[lo]):+8.1%} {np.median(rcd[hi]):+8.1%}")
