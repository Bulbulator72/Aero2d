"""Сравнение вязкого расчёта aero2d с XFOIL (поляры).

Запуск: python benchmarks/viscous_vs_xfoil.py [--quick]
Нужен установленный xfoil (только для проверки). Результат — viscous_vs_xfoil.json
и таблица в консоли.
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

from aero2d.geometry import naca4                    # noqa: E402
from aero2d.viscous import ViscousSolver             # noqa: E402
from benchmarks.xfoil_ref import xfoil_polar         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    quick = "--quick" in sys.argv
    foils = ["0012", "2412", "4415"] if not quick else ["2412"]
    res = [1e5, 1e6, 1e7]
    alphas = np.arange(-4, 12.1, 2.0)
    out = []
    for f in foils:
        af = naca4(f, 200)
        afx = naca4(f, 120)
        for re in res:
            t = time.time()
            v = ViscousSolver(af, re)
            pol = v.polar(alphas, callback=lambda i, n, r: print(f"    {i}/{n} α={r.alpha:g} "
                                                                  f"{'ok' if r.converged else 'нет'} "
                                                                  f"{time.time() - t:.0f}s", flush=True))
            tm = time.time() - t
            xf = xfoil_polar("n" + f, alphas, re, coords=(afx.x, afx.y))
            for p, x in zip(pol, xf):
                out.append(dict(foil=f, re=re, alpha=float(p.alpha), conv=bool(p.converged),
                                cl=p.cl, cd=p.cd, cm=p.cm, xtr_u=p.xtr_u, xtr_l=p.xtr_l,
                                xf_cl=x["cl"], xf_cd=x["cd"], xf_cm=x["cm"],
                                xf_xtr_u=x["xtr_u"], xf_xtr_l=x["xtr_l"]))
            print(f"NACA {f}  Re={re:.0e}  ({tm:.0f} с на поляру)")
            print("   α    Cl     Cl_xf    Cd       Cd_xf    dCd%   xtr_u  xf     сошлось")
            for r in out[-len(alphas):]:
                dcd = 100 * (r["cd"] / r["xf_cd"] - 1) if np.isfinite(r["xf_cd"]) else np.nan
                print(f"  {r['alpha']:4.0f} {r['cl']:7.4f} {r['xf_cl']:7.4f}  {r['cd']:.5f}  {r['xf_cd']:.5f} "
                      f"{dcd:6.1f}  {r['xtr_u']:.3f}  {r['xf_xtr_u']:.3f}  {'да' if r['conv'] else 'НЕТ'}")
            sys.stdout.flush()
    with open(os.path.join(HERE, "viscous_vs_xfoil.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
