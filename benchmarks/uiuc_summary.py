"""Сводка массовой сверки (uiuc_sweep.jsonl): ошибки Cl и Cd против продувок.

Запуск: python benchmarks/uiuc_summary.py [файл.jsonl ...] [--ncrit 9]

Учитываются точки «до срыва»: от α = −2° до (α при Cl_max эксперимента − 1°).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(paths, ncrit=None):
    rows = []
    for p in paths:
        for ln in open(p):
            d = json.loads(ln)
            if "cl" not in d:
                continue
            if ncrit is not None and abs(d["ncrit"] - ncrit) > 1e-6:
                continue
            rows.append(d)
    return rows


def points(rows, key_cl="cl", key_cd="cd"):
    out = []
    for d in rows:
        a = np.array(d["alpha"])
        cle, cde = np.array(d["cl_exp"]), np.array(d["cd_exp"])
        if key_cl not in d:
            continue
        cl = np.array(d[key_cl], float)
        cd = np.array(d[key_cd], float)
        conv = np.array(d["conv"]) if key_cl == "cl" else np.isfinite(cl)
        amax = a[np.argmax(cle)]
        pre = (a >= -2.0) & (a <= amax - 1.0)
        for i in np.where(pre)[0]:
            out.append(dict(name=d["name"], re=d["re"], alpha=a[i], cle=cle[i], cde=cde[i],
                            cl=cl[i], cd=cd[i], conv=bool(conv[i])))
    return out


def bucket(re):
    for b in (6e4, 1e5, 1.5e5, 2e5, 3e5, 4e5, 5e5):
        if abs(re / b - 1) < 0.2:
            return b
    return re


def report(pts, title):
    print(f"\n{title}")
    print("   Re      точек  сошлось   ΔCl ср.  |ΔCl| ср.  ΔCd/Cd медиана  |ΔCd/Cd| медиана")
    by = {}
    for p in pts:
        by.setdefault(bucket(p["re"]), []).append(p)
    allok = []
    for b in sorted(by):
        P = by[b]
        ok = [p for p in P if p["conv"] and np.isfinite(p["cl"]) and np.isfinite(p["cd"])]
        allok += ok
        if not ok:
            print(f"  {b:8.0f}  {len(P):5d}  0")
            continue
        dcl = np.array([p["cl"] - p["cle"] for p in ok])
        rcd = np.array([p["cd"] / p["cde"] - 1 for p in ok])
        print(f"  {b:8.0f}  {len(P):5d}  {len(ok) / len(P):6.0%}   {dcl.mean():+.3f}    {np.abs(dcl).mean():.3f}"
              f"      {np.median(rcd):+6.1%}          {np.median(np.abs(rcd)):5.1%}")
    if allok:
        dcl = np.array([p["cl"] - p["cle"] for p in allok])
        rcd = np.array([p["cd"] / p["cde"] - 1 for p in allok])
        print(f"  всего     {len(pts):5d}  {len(allok) / len(pts):6.0%}   {dcl.mean():+.3f}    {np.abs(dcl).mean():.3f}"
              f"      {np.median(rcd):+6.1%}          {np.median(np.abs(rcd)):5.1%}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ncrit = None
    if "--ncrit" in sys.argv:
        ncrit = float(sys.argv[sys.argv.index("--ncrit") + 1])
        args = [a for a in args if a != str(sys.argv[sys.argv.index("--ncrit") + 1])]
    paths = args or [os.path.join(HERE, "uiuc_sweep.jsonl")]
    rows = load(paths, ncrit)
    print(f"{len(rows)} поляр, {len({r['name'] for r in rows})} профилей")
    report(points(rows), "aero2d против эксперимента (до срыва)")
    if any("cl_xf" in r for r in rows):
        report(points(rows, "cl_xf", "cd_xf"), "XFOIL против эксперимента (до срыва)")


if __name__ == "__main__":
    main()
