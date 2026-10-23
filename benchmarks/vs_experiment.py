"""Сравнение aero2d и XFOIL с продувками UIUC (E387, Re 6e4…4.6e5).

Запуск: python benchmarks/vs_experiment.py [Re ...]   (по умолчанию все Re)
Результат — таблица и benchmarks/vs_experiment_<Re>.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from aero2d.geometry import load_airfoil, repanel   # noqa: E402
from aero2d.viscous import ViscousSolver             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "uiuc")


def read_uiuc_polar(path):
    """Файл *_drag.dat формата UIUC → {Re: array[[α, Cl, Cd], ...]}."""
    lines = open(path, encoding="latin-1").read().splitlines()
    out, i = {}, 0
    while i < len(lines):
        if lines[i].strip().startswith("Average Reynolds"):
            re = float(lines[i + 1])
            n = int(lines[i + 3])
            rows = [list(map(float, lines[i + 5 + k].split()[:3])) for k in range(n)]
            out[re] = np.array(rows)
            i += 5 + n
        else:
            i += 1
    return out


def main():
    data = read_uiuc_polar(os.path.join(DATA, "e387_drag.dat"))
    want = [float(a) for a in sys.argv[1:]]
    af = load_airfoil(os.path.join(DATA, "e387_coord.dat"))
    afx = repanel(af, 60)
    try:
        from benchmarks.xfoil_ref import xfoil_polar
    except Exception:
        xfoil_polar = None
    for re, tab in sorted(data.items()):
        if want and not any(abs(re / w - 1) < 0.1 for w in want):
            continue
        alphas = tab[:, 0]
        t = time.time()
        v = ViscousSolver(af, re, ncrit=9)
        pol = v.polar(alphas)
        tm = time.time() - t
        xf = xfoil_polar("e387", alphas, re, coords=(afx.x, afx.y)) if xfoil_polar else [dict(cl=np.nan, cd=np.nan)] * len(alphas)
        rows = []
        print(f"E387  Re={re:.0f}  ({tm:.0f} с)")
        print("    α     Cl_эксп Cl_aero Cl_xf   Cd_эксп  Cd_aero  Cd_xf    сошлось")
        for (a, cle, cde), p, x in zip(tab, pol, xf):
            rows.append(dict(alpha=a, cl_exp=cle, cd_exp=cde, cl=p.cl, cd=p.cd, conv=bool(p.converged),
                             xtr_u=p.xtr_u, xtr_l=p.xtr_l, cl_xf=x["cl"], cd_xf=x["cd"]))
            print(f"  {a:6.2f}  {cle:6.3f}  {p.cl:6.3f}  {x['cl']:6.3f}   {cde:.4f}   {p.cd:.4f}   {x['cd']:.4f}   "
                  f"{'да' if p.converged else 'НЕТ'}")
        sys.stdout.flush()
        with open(os.path.join(HERE, f"vs_experiment_{re:.0f}.json"), "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
