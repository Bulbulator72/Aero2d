"""Проверка RANS на компьютере пользователя: скорость и правильность.

Запуск: python benchmarks/verify_rans.py [quick | full]
  quick (по умолчанию, ~10–30 мин): NACA 0012, M 0,15, Re 6·10⁶, α = 0° и 10°,
         предобусловливатель ILU(0) и ILU(3) — сравнение времени и с эталоном NASA.
  full  (около часа): то же + GA(W)-1 с закрылком Фаулера 30°, α = 0…12° через 2°
         (7 углов): старт на грубой сетке, затем все углы параллельно на ядрах —
         сравнение с продувкой NASA CR-2443 и время на этом ПК.
  rc    (1–2 часа): модель SA-RC (поправка на кривизну) — NACA 0012 α = 0° и 10° против
         эталона NASA для SA-RC и механизация GA(W)-1 с закрылком 40° (α = 0, 5, 10°),
         где SA расходится с опытом.
  aft   (1–2 часа): модель перехода SA-AFT (Ñкрит = 9): профиль E387 при Re 2·10⁵ и 3·10⁵,
         α = 0, 2, 4, 6° — против продувки NASA LTPT (TM-4062), вязкого метода aero2d
         (точки перехода) и полностью турбулентного SA.
Отчёт: benchmarks/verify_report.txt — его и нужно прислать.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "verify_report.txt")


def log(s=""):
    print(s, flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def header():
    import numpy as np
    from aero2d.parallel import cpu_count
    log(f"=== aero2d verify_rans {time.strftime('%Y-%m-%d %H:%M')} ===")
    log(f"ОС: {platform.platform()}; процессор: {platform.processor() or '?'}; логических ядер: {cpu_count()}")
    try:
        import jax
        import gmsh
        import scipy
        log(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
            f"jax {jax.__version__}, gmsh {gmsh.__version__}")
    except Exception as e:                                       # noqa: BLE001
        log(f"ошибка импорта: {e}")
    try:
        import psutil
        log(f"память: {psutil.virtual_memory().total / 1e9:.0f} ГБ")
    except ImportError:
        pass


def naca():
    import numpy as np
    from aero2d.geometry import naca4, repanel
    from aero2d.rans import RansSolver
    from aero2d.rans_mesh import build_mesh, quality
    TMR = {0.0: (0.0, 0.00819), 10.0: (1.0909, 0.01231)}
    af = repanel(naca4("0012", 200), 130)
    t = time.time()
    mesh = build_mesh([af], re=6e6, far_radius=50)
    log(f"\n--- NACA 0012, M 0,15, Re 6e6: сетка {quality(mesh)['cells']} ячеек ({time.time() - t:.1f} с)")
    log("эталон NASA TMR (CFL3D): α=0: Cd 0,00819; α=10: Cl 1,0909, Cd 0,01231")
    for fill in (3, 0):
        s = RansSolver(mesh, 0.15, 6e6)
        s.prec_fill = fill
        Q = None
        for a in (0.0, 10.0):
            t = time.time()
            r = s.solve(a, Q0=Q, max_iter=400)
            ref = TMR[a]
            log(f"ILU({fill})  α {a:4.1f}: Cl {r.cl:.4f} (эталон {ref[0]:.4f})  Cd {r.cd:.5f} (эталон {ref[1]:.5f})  "
                f"итераций {r.iterations}, {r.status}, {time.time() - t:.0f} с")
            Q = r.state


def gaw():
    import numpy as np
    import gaw1_fowler as G
    from aero2d.rans import Convergence, mesh_for, rans_polar
    from aero2d.parallel import auto_workers
    els = G.configuration(30, n_main=180, n_flap=100)
    mesh = mesh_for(els, 7.5, G.RE)
    e = G.EXP_CL[30]
    w = auto_workers()
    log(f"\n--- GA(W)-1 + закрылок Фаулера 30°, Re 2,2e6, M 0,21: сетка {mesh.n_cells} ячеек, процессов до {w}")
    t0 = time.time()

    def cb(k, n, it, rn, r):
        if r is not None:
            ce = np.interp(r.alpha, e[:, 0], e[:, 1], right=np.nan)
            log(f"  α {r.alpha:5.1f}: Cl {r.cl:.3f} (опыт {ce:.3f}), {r.status}, {time.time() - t0:.0f} с от начала")
        return False
    res, _ = rans_polar(els, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], G.RE, mach=G.MACH, mesh=mesh,
                        xref=(0.25, 0.0), callback=cb, max_iter=400, conv=Convergence(), workers=w,
                        coarse_start=True)
    log("  прежние результаты: α 0 / 5 / 10 / 12° — Cl 2,397 / 3,021 / 3,414 / 3,395; "
        "прошлый запуск этой проверки (4 угла, без грубого старта) — 57 мин")
    for r in res:
        if r is not None:
            log(f"  итог α {r.alpha:5.1f}: Cl {r.cl:.3f} Cd {r.cd:.4f} осн. {r.elements[0]['cl']:.3f} "
                f"закр. {r.elements[1]['cl']:.3f}, итераций {r.iterations}, {r.status}")
    log(f"  всего {time.time() - t0:.0f} с")


def rc():
    import numpy as np
    import gaw1_fowler as G
    from aero2d.geometry import naca4, repanel
    from aero2d.parallel import auto_workers
    from aero2d.rans import Convergence, RansSolver, mesh_for, rans_polar
    from aero2d.rans_mesh import build_mesh
    REF = {0.0: (0.0, 0.00793), 10.0: (1.0947, 0.01179)}
    af = repanel(naca4("0012", 200), 130)
    mesh = build_mesh([af], re=6e6, far_radius=50)
    log(f"\n--- SA-RC, NACA 0012, M 0,15, Re 6e6: сетка {mesh.n_cells} ячеек")
    log("эталон NASA TMR для SA-RC (CFL3D): α=0: Cd 0,00793; α=10: Cl 1,0947, Cd 0,01179 "
        "(для SA: Cd 0,00819; Cl 1,0909, Cd 0,01231)")
    s = RansSolver(mesh, 0.15, 6e6)
    s.turb_model = "sa-rc"
    Q = None
    for a in (0.0, 10.0):
        t = time.time()
        r = s.solve(a, Q0=Q, max_iter=400)
        log(f"SA-RC α {a:4.1f}: Cl {r.cl:.4f} (эталон {REF[a][0]:.4f})  Cd {r.cd:.5f} (эталон {REF[a][1]:.5f})  "
            f"итераций {r.iterations}, {r.status}, {time.time() - t:.0f} с")
        Q = r.state
    els = G.configuration(40, n_main=180, n_flap=100)
    m40 = mesh_for(els, 5.0, G.RE)
    e = G.EXP_CL[40]
    log(f"\n--- SA-RC, GA(W)-1 + закрылок Фаулера 40°: сетка {m40.n_cells} ячеек")
    log("  SA (прежний расчёт): Cl 2,649 / 3,274 / 3,736 при α = 0 / 5 / 10°; опыт 2,917 / 3,383 / 3,502")
    t0 = time.time()

    def cb(k, n, it, rn, r):
        if r is not None:
            ce = np.interp(r.alpha, e[:, 0], e[:, 1], right=np.nan)
            log(f"  α {r.alpha:5.1f}: Cl {r.cl:.3f} (опыт {ce:.3f}), {r.status}, {time.time() - t0:.0f} с от начала")
        return False
    res, _ = rans_polar(els, [0.0, 5.0, 10.0], G.RE, mach=G.MACH, mesh=m40, xref=(0.25, 0.0), callback=cb,
                        max_iter=400, conv=Convergence(), workers=auto_workers(), coarse_start=True,
                        solver_opts={"turb_model": "sa-rc"})
    for r in res:
        if r is not None and hasattr(r, "elements"):
            log(f"  итог α {r.alpha:5.1f}: Cl {r.cl:.3f} Cd {r.cd:.4f} осн. {r.elements[0]['cl']:.3f} "
                f"закр. {r.elements[1]['cl']:.3f}, итераций {r.iterations}, {r.status}")


def aft():
    import json
    import numpy as np
    from aero2d.geometry import load_airfoil, repanel
    from aero2d.parallel import auto_workers
    from aero2d.rans import Convergence, mesh_for, rans_polar
    from aero2d.viscous import ViscousSolver
    lt = json.load(open(os.path.join(HERE, "data", "ltpt", "e387_ltpt.json")))
    af = load_airfoil(os.path.join(HERE, "data", "uiuc", "e387_coord.dat"))
    af = repanel(af, 180)
    alphas = [0.0, 2.0, 4.0, 6.0]
    for re in (2e5, 3e5):
        rows = np.array(lt["polars"][str(int(re))])
        mesh = mesh_for([af], 3.0, re)
        log(f"\n--- SA-AFT, E387, Re {re:.0f}, M 0,1: сетка {mesh.n_cells} ячеек")
        vs = ViscousSolver(af, re, ncrit=9.0, mach=0.1)
        ref = {}
        for a in alphas:
            r = vs.solve(a)
            ref[a] = (r.cl, r.cd, r.xtr_u, r.xtr_l)
        t0 = time.time()
        out = {}
        for tm in ("sa-aft", "sa"):
            def cb(k, n, it, rn, r, tm=tm):
                if r is not None:
                    log(f"  {tm} α {r.alpha:4.1f}: Cl {r.cl:.3f} Cd {r.cd:.4f}, {r.status}, {time.time() - t0:.0f} с")
                return False
            res, _ = rans_polar([af], alphas, re, mach=0.1, mesh=mesh, callback=cb, max_iter=500,
                                conv=Convergence(), workers=auto_workers(), coarse_start=False,
                                solver_opts={"turb_model": tm, "ncrit": 9.0})
            out[tm] = res
        from aero2d.core import _rans_xtr
        from scipy.spatial import cKDTree
        log("   α  | LTPT Cl  Cd    | вязк. Cl  Cd     x_tr верх/низ | SA-AFT Cl  Cd     x_tr верх/низ | SA Cl  Cd")
        for i, a in enumerate(alphas):
            e = np.interp(a, rows[:, 0], rows[:, 1]), np.interp(a, rows[:, 0], rows[:, 2])
            v = ref[a]
            ra, rs = out["sa-aft"][i], out["sa"][i]
            xt = (np.nan, np.nan)
            if ra is not None and ra.elements[0].get("gam") is not None:
                el = ra.elements[0]
                _, j = cKDTree(np.column_stack([af.x, af.y])).query(np.column_stack([el["x"], el["y"]]))
                o = np.argsort(j, kind="stable")
                xt = _rans_xtr(np.asarray(af.x), j[o], el["x"][o], el["gam"][o])
            f = lambda r: f"{r.cl:.3f} {r.cd:.4f}" if r is not None else "—"
            log(f" {a:4.1f} | {e[0]:.3f} {e[1]:.4f} | {v[0]:.3f} {v[1]:.4f}  {v[2]:.2f}/{v[3]:.2f} | "
                f"{f(ra)}  {xt[0]:.2f}/{xt[1]:.2f} | {f(rs)}")
        log(f"  всего {time.time() - t0:.0f} с")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    header()
    t = time.time()
    jobs = (("rc", rc),) if mode == "rc" else (("aft", aft),) if mode == "aft" else (("naca", naca), ("gaw", gaw))
    for name, fn in jobs:
        if name == "gaw" and mode != "full":
            continue
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            log("ОШИБКА:\n" + traceback.format_exc())
    log(f"\nготово за {(time.time() - t) / 60:.1f} мин. Пришлите файл {OUT}")


if __name__ == "__main__":
    main()
