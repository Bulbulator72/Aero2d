"""Распределения пограничного слоя из XFOIL (xfoil-python с добавленными
get_nbl/get_bl; путь к сборке — переменная XFOIL_PY или папка по умолчанию)."""
import ctypes
import os
import sys

import numpy as np

_DEF = "/tmp/claude-0/-home-claude/0704b23f-1f69-5fd2-b8ea-72811cd4380f/scratchpad/xfsrc/xfoil-python/build/lib.linux-x86_64-cpython-312"
sys.path.insert(0, os.environ.get("XFOIL_PY", _DEF))
from xfoil import XFoil            # noqa: E402
from xfoil.model import Airfoil    # noqa: E402


def xfoil_bl(x, y, alpha, re, ncrit=9.0, iters=200, npan=None):
    xf = XFoil()
    xf.print = False
    xf.airfoil = Airfoil(np.asarray(x, float), np.asarray(y, float))
    if npan:
        xf.repanel(n_nodes=npan)
    else:
        xf.repanel()
    xf.Re, xf.n_crit, xf.max_iter = re, ncrit, iters
    cl, cd, cm, cp = xf.a(alpha)
    lib = xf._lib
    lib.get_nbl.restype = ctypes.c_int
    out = dict(cl=cl, cd=cd, cm=cm)
    for side, name in ((1, "upper"), (2, "lower")):
        n = lib.get_nbl(ctypes.byref(ctypes.c_int(side)))
        arrs = [np.zeros(n, np.float32) for _ in range(5)]
        ptrs = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in arrs]
        lib.get_bl(ctypes.byref(ctypes.c_int(side)), ctypes.byref(ctypes.c_int(n)), *ptrs)
        xx, ue, ds, th, ct = [a.astype(float) for a in arrs]
        out[name] = dict(x=xx, ue=ue, dstar=ds, theta=th, ctau=ct, H=ds / np.maximum(th, 1e-12))
    return out
