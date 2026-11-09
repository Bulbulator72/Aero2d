"""Поляра NLF(1)-0416 (M 0,1, Re 4·10⁶, Tu 0,15 %) во всём диапазоне углов до срыва: вязкий
метод и RANS «SA + переход по вязкому методу» против эксперимента NASA TP-1861 (рис. 11d).
Батник 21. Отчёт — benchmarks/nlf_polar_report.txt: каждый угол записывается сразу, как досчитан.

Углы считаются по очереди: 0°, затем вверх до срыва, затем отрицательные; каждый — от решения
соседнего (как поляра в окне). Состояние сохраняется после каждого угла (nlf_polar_state.pkl):
повторный запуск продолжит со следующего угла. Аргумент — углы через запятую
(по умолчанию −4,0,4,8,10,12,13,14,15,16)."""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
REPORT = os.path.join(HERE, "nlf_polar_report.txt")
STATE = os.path.join(HERE, "nlf_polar_state.pkl")


def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def main():
    import aero2d
    from aero2d.geometry import load_airfoil, repanel
    from aero2d.rans import (Convergence, RansSolver, lam_xtr_at, mesh_for, ncrit_from_tu,
                             viscous_transition_table)
    from aero2d.viscous import ViscousSolver
    al = [float(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                             "-4,0,4,8,10,12,13,14,15,16".split(","))]
    re, mach, tu = 4e6, 0.1, 0.15
    nc = ncrit_from_tu(tu)
    exp = json.load(open(os.path.join(HERE, "data", "nlf0416", "exp_re4e6.json")))
    ea, ecl = np.array(exp["alpha_cl"]).T
    ecc, ecd = np.array(exp["cl_cd"]).T
    af = repanel(load_airfoil(os.path.join(HERE, "data", "nlf0416", "nlf0416_coord.dat")), 160)
    # порядок: 0°, вверх, затем вниз — каждый угол от соседнего
    order = sorted([a for a in al if a >= 0]) + sorted([a for a in al if a < 0], reverse=True)
    if os.path.exists(STATE):
        d = pickle.load(open(STATE, "rb"))
        log(f"\n=== продолжение: готово углов {len(d['done'])} из {len(order)}")
    else:
        log(f"=== Поляра NLF(1)-0416; aero2d {aero2d.__version__}; {time.strftime('%Y-%m-%d %H:%M')}; "
            f"ядер {os.cpu_count()}")
        log(f"M {mach}, Re {re:.0e}, Tu {tu} % (Ñкрит {nc:.2f}); эксперимент: {exp['source']}")
        t0 = time.time()
        vs = ViscousSolver(af, re, ncrit=nc, mach=mach)
        visc = {}
        for a in order:
            try:
                r = vs.solve(a)
                visc[a] = (r.cl, r.cd, r.xtr_u, r.xtr_l, bool(getattr(r, "converged", True)))
            except Exception:              # noqa: BLE001
                visc[a] = (np.nan, np.nan, np.nan, np.nan, False)
        table = viscous_transition_table(af, order, re, nc, mach)
        log(f"вязкий метод: {time.time() - t0:.0f} с; точки перехода для RANS — при углах "
            + ", ".join(f"{a:g}" for a in sorted(table)))
        mesh = mesh_for([af], 0.0, re)
        log(f"сетка: {mesh.n_cells} ячеек")
        log("\n   α  | эксп. Cl | вязкий Cl  Cd      x_tr     | RANS Cl  Cd      x_tr      эксп. Cd* | итер., мин, статус")
        d = dict(mesh=mesh, table=table, visc=visc, done={}, Q=None, a_prev=None)
        pickle.dump(d, open(STATE, "wb"))
    s = RansSolver(d["mesh"], mach, re, chord=1.0, xref=(0.25, 0.0))
    s.lam_xtr_table = d["table"] or None
    s.tu = tu
    for a in order:
        if a in d["done"]:
            continue
        # отрицательные углы — от решения при 0°
        q0, a0 = (d["done"][0.0][-1], 0.0) if (a < 0 and 0.0 in d["done"]) else (d["Q"], d["a_prev"])
        t = time.time()
        print(f"--- α = {a:g}° (от {'однородного потока' if q0 is None else f'решения при {a0:g}°'})", flush=True)

        def cb(it, rn, a=a):
            if it % 20 == 0:
                print(f"     α {a:g}°: итерация {it}, невязка {rn:.2e}  ({(time.time() - t) / 60:.0f} мин)", flush=True)
            return False
        if q0 is None:
            # первый угол — со старта на грубой сетке (как в окне программы): в разы быстрее
            from aero2d.rans import rans_polar

            def cb2(k, n, it, rn, res_, a=a):
                return cb(it, rn) if res_ is None else False
            rr, _ = rans_polar([af], [a], re, mach=mach, mesh=d["mesh"], max_iter=400, conv=Convergence(),
                               workers=1, coarse_start=True, callback=cb2,
                               solver_opts={"lam_xtr_table": d["table"] or None, "tu": tu})
            r = rr[0]
        else:
            r = s.solve(a, Q0=q0, alpha0=a0, max_iter=400, conv=Convergence(), callback=cb)
        xu, xl = lam_xtr_at(d["table"], a) if d["table"] else (np.nan, np.nan)
        d["done"][a] = (r.cl, r.cd, xu, xl, r.iterations, (time.time() - t) / 60, r.status, r.state)
        d["Q"], d["a_prev"] = r.state, a
        pickle.dump(d, open(STATE + ".tmp", "wb"))
        os.replace(STATE + ".tmp", STATE)
        v = d["visc"][a]
        ec = float(np.interp(a, ea, ecl)) if ea.min() <= a <= ea.max() else np.nan
        ed = float(np.interp(r.cl, ecc, ecd)) if ecc.min() <= r.cl <= ecc.max() else np.nan
        log(f" {a:5.1f}| {ec:6.3f}   | {v[0]:6.3f}  {v[1]:.5f}  {v[2]:.2f}/{v[3]:.2f}"
            f"{'' if v[4] else ' (не сошлось)'} | {r.cl:6.3f}  {r.cd:.5f}  {xu:.2f}/{xl:.2f}   {ed:.4f}  "
            f"| {r.iterations}, {(time.time() - t) / 60:.0f} мин, {r.status}")
    cl = {a: v[0] for a, v in d["done"].items()}
    k = max(cl, key=cl.get)
    vc = {a: v[0] for a, v in d["visc"].items() if np.isfinite(v[0])}
    kv = max(vc, key=vc.get) if vc else np.nan
    log(f"* Cd эксперимента — при Cl расчёта RANS (из поляры Cd(Cl)).")
    log(f"Cl_max: эксперимент {exp['clmax']:.2f} при {exp['alpha_clmax']:.0f}°; RANS {cl[k]:.3f} при {k:g}°; "
        f"вязкий {vc.get(kv, np.nan):.3f} при {kv:g}°")
    log("Готово.")


if __name__ == "__main__":
    main()
