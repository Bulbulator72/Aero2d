"""Руль 30 % хорды на NACA 0009: производные по α и δ против продувок NACA
(Ames, ARR L-301, 1941, NACA 4×6 ft вертикальная труба: Re_эфф ≈ 2,76·10⁶ при
коэффициенте турбулентности 1,93, т. е. Re опыта ≈ 1,43·10⁶ в турбулентном
потоке — расчёт при Re = 1,43·10⁶ и малом Ñкрит; щель герметична; Ch не
исправлены на влияние стенок, cl исправлены). Значения: ch_α = (∂ch/∂α)δ,
ch_δ = (∂ch/∂δ)α — по тексту отчёта (раздел «Parameters»): ch_α от формы
носка почти не зависит (−0,0068 простой руль, −0,0060 с компенсацией 0,20c_р),
ch_δ зависит (−0,0120 простой, −0,0088 с компенсацией и тупым носком).

Запуск: python benchmarks/vs_naca_flap.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from aero2d.geometry import naca4, repanel             # noqa: E402
from aero2d.controls import deflect, hinge_point, hinge_moment   # noqa: E402
from aero2d.panel import PanelSolver                   # noqa: E402
from aero2d.viscous import ViscousSolver               # noqa: E402

EXP = {  # производные на градус
    0.0: dict(cl_a=0.094, tau=0.57, ch_a=-0.0068, ch_d=-0.0120),
    0.2: dict(cl_a=0.094, tau=0.57, ch_a=-0.0060, ch_d=-0.0088),
}
RE = 1.43e6
XH = 0.7


def point(af, method, alpha, delta, balance, ncrit=9.0):
    h = hinge_point(af, XH)
    geo = repanel(deflect(af, XH, delta), 120, keep_frame=True) if delta else af
    if method == "inv":
        r = PanelSolver(geo).solve(alpha)
    else:
        r = ViscousSolver(geo, RE, ncrit=ncrit).solve(alpha)
    return r.cl, hinge_moment(r.x, r.y, r.cp, h, XH, delta, balance)


def derivs(method, balance, ncrit=9.0, d=4.0):
    af = naca4("0009", 161)
    clp, chp = point(af, method, d, 0.0, balance, ncrit)
    clm, chm = point(af, method, -d, 0.0, balance, ncrit)
    cdp, hdp = point(af, method, 0.0, d, balance, ncrit)
    cdm, hdm = point(af, method, 0.0, -d, balance, ncrit)
    cl_a = (clp - clm) / (2 * d)
    cl_d = (cdp - cdm) / (2 * d)
    return dict(cl_a=cl_a, tau=cl_d / cl_a, ch_a=(chp - chm) / (2 * d), ch_d=(hdp - hdm) / (2 * d))


def main():
    np.seterr(all="ignore")
    print("NACA 0009, руль 0,30c, щель герметична; производные на градус, α, δ = ±4°")
    print(f"{'компенсация':<12} {'источник':<22} {'cl_α':>7} {'τ=cl_δ/cl_α':>12} {'ch_α':>9} {'ch_δ':>9}")
    for bal, e in EXP.items():
        rows = [("эксперимент NACA", e), ("aero2d невязкий", derivs("inv", bal)),
                ("aero2d вязкий Ñ=9", derivs("visc", bal, 9.0)),
                ("aero2d вязкий Ñ=4", derivs("visc", bal, 4.0)),
                ("aero2d вязкий Ñ=2", derivs("visc", bal, 2.0))]
        for nm, r in rows:
            print(f"{bal:<12g} {nm:<22} {r['cl_a']:7.4f} {r['tau']:12.3f} {r['ch_a']:9.4f} {r['ch_d']:9.4f}")


if __name__ == "__main__":
    main()
