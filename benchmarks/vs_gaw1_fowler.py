"""Сверка вязкого многоэлементного расчёта с продувками GA(W)-1 + 29 % закрылок
Фаулера (NASA CR-2443, рис. 7: Re = 2.2·10⁶, M = 0.21, переход на основном
элементе зафиксирован на 5 % хорды).

Запуск: python benchmarks/vs_gaw1_fowler.py [--defl 10 20 30 40] [--workers 4]
Результат: benchmarks/vs_gaw1_fowler.txt и .png
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import gaw1_fowler as G   # noqa: E402


def run(task):
    np.seterr(all="ignore")
    d, alphas, ncrit = task
    from aero2d.multi import MultiPanelSolver
    from aero2d.mviscous import MultiViscousSolver, default_transition
    t0 = time.time()
    if d == 0:
        from aero2d.geometry import Airfoil
        from aero2d.viscous import ViscousSolver
        x = np.concatenate([G._X[::-1], G._X[1:]])
        y = np.concatenate([G._ZU[::-1], G._ZL[1:]])
        vs = ViscousSolver(Airfoil(x, y, "GA(W)-1"), G.RE, ncrit=ncrit, mach=G.MACH, xtr=(G.XTR, G.XTR))
        res = vs.polar(alphas)
        return dict(d=d, alpha=list(alphas), cl=[r.cl for r in res], cd=[r.cd for r in res],
                    conv=[bool(r.converged) for r in res], cl_inv=[np.nan] * len(alphas), time=time.time() - t0)
    els = G.configuration(d, n_main=200, n_flap=100)
    inv = MultiPanelSolver(els, chord=1.0)
    ms = MultiViscousSolver(els, G.RE, ncrit=ncrit, mach=G.MACH, xtr=default_transition(els, (G.XTR, G.XTR)),
                            n_panels=[180, 100])
    res = ms.polar(alphas)
    return dict(d=d, alpha=list(alphas), cl=[r.cl for r in res], cd=[r.cd for r in res],
                cl_main=[r.elements[0]["cl"] for r in res], cl_flap=[r.elements[1]["cl"] for r in res],
                conv=[bool(r.converged) for r in res],
                cl_inv=[inv.solve(a, G.MACH).cl for a in alphas], time=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defl", type=int, nargs="+", default=[0, 10, 20, 30, 40])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ncrit", type=float, default=9.0)
    args = ap.parse_args()
    tasks = []
    for d in args.defl:
        e = G.EXP_CL[d]
        alphas = np.round(e[:, 0], 2)
        if d:
            alphas = alphas[alphas < 12]              # до срыва
        tasks.append((d, alphas, args.ncrit))
    with mp.get_context("fork").Pool(args.workers) as pool:
        out = pool.map(run, tasks)
    lines = ["GA(W)-1 + 29% Fowler (NASA CR-2443), Re 2.2e6, M 0.21, переход 5% (осн. элемент)",
             f"Ñкрит = {args.ncrit}",
             " δf    α      Cl эксп  Cl расч  Cl невязк  ΔCl     Cd расч  сошлось"]
    allerr = []
    for o in out:
        e = G.EXP_CL[o["d"]]
        for i, a in enumerate(o["alpha"]):
            cle = e[np.argmin(np.abs(e[:, 0] - a)), 1]
            dcl = o["cl"][i] - cle
            if o["conv"][i]:
                allerr.append((o["d"], dcl))
            lines.append(f"{o['d']:3d}  {a:6.2f}   {cle:6.3f}   {o['cl'][i]:6.3f}   {o['cl_inv'][i]:7.3f}  "
                         f"{dcl:+6.3f}   {o['cd'][i]:7.4f}   {'да' if o['conv'][i] else 'НЕТ'}")
        lines.append(f"     ({o['time']:.0f} с)")
    if allerr:
        d_ = np.array([x[1] for x in allerr])
        lines.append(f"сошлось точек: {len(d_)}, ΔCl ср. {d_.mean():+.3f}, |ΔCl| ср. {np.abs(d_).mean():.3f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(HERE, "vs_gaw1_fowler.txt"), "w").write(txt + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    for k, o in enumerate(out):
        c = f"C{k}"
        e = G.EXP_CL[o["d"]]
        ax.plot(e[:, 0], e[:, 1], "o", color=c, mfc="none", label=f"эксп. δf={o['d']}°")
        cl = np.where(o["conv"], o["cl"], np.nan)
        ax.plot(o["alpha"], cl, "-", color=c, label="aero2d вязк.")
        if o["d"]:
            ax.plot(o["alpha"], o["cl_inv"], ":", color=c, lw=0.8)
    ax.set_xlabel("α, °")
    ax.set_ylabel("Cl")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("GA(W)-1 + закрылок Фаулера 29 %, Re 2.2e6 (точки — CR-2443, пунктир — невязкий)")
    fig.savefig(os.path.join(HERE, "vs_gaw1_fowler.png"), dpi=110)


if __name__ == "__main__":
    main()
