"""Эталонные расчёты XFOIL (только для проверки aero2d; в программу не входит).

XFOIL из пакета Debian/Ubuntu падает после каждой сошедшейся точки
(SIGFPE в модуле вывода), поэтому каждая точка — отдельный запуск; результат
берётся из файла поляры, который успевает записаться.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np


def xfoil_point(airfoil: str, alpha: float, re: float, mach: float = 0.0, ncrit: float = 9.0,
                coords=None, iters: int = 300, xtr=(1.0, 1.0), npan: int = 160):
    """airfoil — 'NACA 2412' или имя при coords=(x, y). Возвращает dict или None
    (не сошлось). Результат разбирается из текстового вывода XFOIL."""
    import re as _re
    with tempfile.TemporaryDirectory() as d:
        if coords is not None:
            f = os.path.join(d, "af.dat")
            np.savetxt(f, np.column_stack(coords), header=airfoil, comments="")
            load = f"LOAD {f}\n"
        else:
            load = f"{airfoil}\n"
        cmd = (f"PLOP\nG F\n\n{load}PANE\nOPER\nVPAR\nN {ncrit}\n"
               f"XTR {xtr[0]} {xtr[1]}\n\nVISC {re:g}\nMACH {mach}\nITER {iters}\n"
               f"ALFA {alpha}\n\nQUIT\n")
        try:
            out = subprocess.run(["xfoil"], input=cmd, capture_output=True, text=True,
                                 timeout=60, cwd=d).stdout
        except subprocess.TimeoutExpired:
            return None
    if "VISCAL:  Convergence failed" in out:
        return None
    num = r"(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)"
    cl = _re.findall(r"a =\s*" + num + r"\s+CL =\s*" + num, out)
    cd = _re.findall(r"Cm =\s*" + num + r"\s+CD =\s*" + num + r"\s+=>\s+CDf =\s*" + num
                     + r"\s+CDp =\s*" + num, out)
    t1 = _re.findall(r"Side 1.*transition at x/c =\s*" + num, out)
    t2 = _re.findall(r"Side 2.*transition at x/c =\s*" + num, out)
    if not cl or not cd:
        return None
    a, c = cl[-1], cd[-1]
    return dict(alpha=float(a[0]), cl=float(a[1]), cm=float(c[0]), cd=float(c[1]),
                cdf=float(c[2]), cdp=float(c[3]),
                xtr_u=float(t1[-1]) if t1 else np.nan, xtr_l=float(t2[-1]) if t2 else np.nan)


def xfoil_polar(airfoil, alphas, re, **kw):
    out = []
    for a in alphas:
        r = xfoil_point(airfoil, a, re, **kw)
        out.append(r if r else dict(alpha=a, cl=np.nan, cd=np.nan, cdp=np.nan, cm=np.nan,
                                    xtr_u=np.nan, xtr_l=np.nan))
    return out


if __name__ == "__main__":
    for re in (1e5, 1e6, 1e7):
        print(re, xfoil_point("NACA 2412", 4.0, re))
