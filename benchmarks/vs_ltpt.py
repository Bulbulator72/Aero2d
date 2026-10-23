"""E387: aero2d против NASA LTPT (McGhee и др., 1988) и UIUC.

Запуск: python benchmarks/vs_ltpt.py [Ñкрит ...]   (по умолчанию 9 и 11)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
warnings.filterwarnings("ignore")

from aero2d.geometry import load_airfoil      # noqa: E402
from aero2d.viscous import polar_robust       # noqa: E402


def main():
    nc_list = [float(a) for a in sys.argv[1:]] or [9.0, 11.0]
    lt = json.load(open(os.path.join(HERE, "data", "ltpt", "e387_ltpt.json")))
    af = load_airfoil(os.path.join(HERE, "data", "uiuc", "e387_coord.dat"))
    np.seterr(all="ignore")
    for re_s, rows in lt["polars"].items():
        rows = np.array(rows)
        re = float(re_s)
        res = {nc: polar_robust(af, re, rows[:, 0], ncrit=nc) for nc in nc_list}
        print(f"\nE387 Re = {re:.0f}: LTPT | " + " | ".join(f"aero2d Ñ={nc:g}" for nc in nc_list))
        print("   α     Cl     Cd    " + "   ".join("  Cl     Cd   " for _ in nc_list))
        for k, (a, cl, cd, cm) in enumerate(rows):
            line = f"{a:6.2f} {cl:6.3f} {cd:.4f}  "
            for nc in nc_list:
                r = res[nc][k]
                line += f"  {r.cl:6.3f} {r.cd:.4f}{'' if r.converged else '*'}"
            print(line)
    print("\n* — не сошлось")


if __name__ == "__main__":
    main()
