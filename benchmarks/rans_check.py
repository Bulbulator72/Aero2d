"""Проверка RANS (Спаларт–Аллмарас):

1. NACA 0012, M 0,15, Re 6·10⁶ — эталоны NASA Turbulence Modeling Resource
   (CFL3D / FUN3D на сетках до 7·10⁶ ячеек, SA):
       α = 0:  Cd 0,00819 / 0,00812
       α = 10: Cl 1,0909 / 1,0983, Cd 0,01231 / 0,01242
       α = 15: Cl 1,5461, Cd 0,02124 (CFL3D)
2. GA(W)-1 + закрылок Фаулера 29 % (NASA CR-2443): Re 2,2·10⁶, M 0,21,
   δf = 30° и 40°, α до Cl_max.

Запуск: python benchmarks/rans_check.py [naca | gaw30 | gaw40 | all] [--iters 400]
Результат дописывается в benchmarks/rans_check.txt. Время: NACA — минуты на
угол, механизация (40 тыс. ячеек) — 10–60 мин на угол в зависимости от ПК.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

TMR = {0.0: (0.0, 0.00819), 10.0: (1.0909, 0.01231), 15.0: (1.5461, 0.02124)}


def log(line, out):
    print(line, flush=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(line + "\n")


VERB = False


def naca(iters, out):
    from aero2d.geometry import naca4, repanel
    from aero2d.rans import rans_polar
    af = repanel(naca4("0012", 200), 130)
    t0 = time.time()

    def cb(k, n, it, rn, r):
        if r is not None:
            ref = TMR[r.alpha]
            log(f"NACA 0012  α {r.alpha:5.1f}  Cl {r.cl:.4f} (TMR {ref[0]:.4f})  Cd {r.cd:.5f} "
                f"(TMR {ref[1]:.5f})  Cdp {r.cdp:.5f} Cdf {r.cdf:.5f}  итераций {len(r.residual)} "
                f"{'сошлось' if r.converged else 'НЕ сошлось'}  {time.time() - t0:.0f} с", out)
        return False
    res, mesh = rans_polar([af], [0.0, 10.0, 15.0], 6e6, mach=0.15, callback=cb, max_iter=iters,
                           verbose=VERB)
    log(f"   сетка {mesh.n_cells} ячеек", out)


def gaw(defl, iters, out):
    import gaw1_fowler as G
    from aero2d.rans import mesh_for, rans_polar
    els = G.configuration(defl, n_main=180, n_flap=100)
    e = G.EXP_CL[defl]
    alphas = [0.0, 5.0, 10.0, 12.0, 14.0]
    mesh = mesh_for(els, 7.5, G.RE)
    log(f"GA(W)-1 + Фаулер δf = {defl}°: сетка {mesh.n_cells} ячеек", out)
    t0 = time.time()

    def cb(k, n, it, rn, r):
        if r is not None:
            ce = np.interp(r.alpha, e[:, 0], e[:, 1], right=np.nan)
            log(f"  α {r.alpha:5.1f}  Cl {r.cl:.3f} (опыт {ce:.3f})  Cd {r.cd:.4f}  Cm {r.cm:.3f}  "
                f"осн. {r.elements[0]['cl']:.3f} закр. {r.elements[1]['cl']:.3f}  "
                f"{'сошлось' if r.converged else 'НЕ сошлось'}  {time.time() - t0:.0f} с", out)
        return False
    rans_polar(els, alphas, G.RE, mach=G.MACH, mesh=mesh, xref=(0.25, 0.0), callback=cb, max_iter=iters,
               verbose=VERB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="all", choices=["naca", "gaw30", "gaw40", "all"])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("-v", "--verbose", action="store_true", help="печатать ход итераций")
    a = ap.parse_args()
    global VERB
    VERB = a.verbose
    out = os.path.join(HERE, "rans_check.txt")
    log(f"--- {time.strftime('%Y-%m-%d %H:%M')}, {os.cpu_count()} логических ядер", out)
    if a.case in ("naca", "all"):
        naca(a.iters, out)
    if a.case in ("gaw30", "all"):
        gaw(30, a.iters, out)
    if a.case in ("gaw40", "all"):
        gaw(40, a.iters, out)


if __name__ == "__main__":
    main()
