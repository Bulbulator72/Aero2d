"""Поляра NLF(1)-0416 (M 0,1, Re 4·10⁶, Tu 0,15 %) во всём диапазоне углов до срыва: вязкий
метод и RANS «SA + переход по вязкому методу» против эксперимента NASA TP-1861 (рис. 11d).
Батник 21. Отчёт — benchmarks/nlf_polar_report.txt. Аргумент — углы через запятую
(по умолчанию −4,0,4,8,10,12,13,14,15,16)."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
REPORT = os.path.join(HERE, "nlf_polar_report.txt")


def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def main():
    import aero2d
    from aero2d.core import run_rans, run_viscous
    from aero2d.geometry import load_airfoil, repanel
    from aero2d.parallel import auto_workers
    from aero2d.rans import Convergence, ncrit_from_tu
    al = [float(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                             "-4,0,4,8,10,12,13,14,15,16".split(","))]
    re, mach, tu = 4e6, 0.1, 0.15
    exp = json.load(open(os.path.join(HERE, "data", "nlf0416", "exp_re4e6.json")))
    ea, ecl = np.array(exp["alpha_cl"]).T
    ecc, ecd = np.array(exp["cl_cd"]).T
    af = repanel(load_airfoil(os.path.join(HERE, "data", "nlf0416", "nlf0416_coord.dat")), 160)
    log(f"=== Поляра NLF(1)-0416; aero2d {aero2d.__version__}; {time.strftime('%Y-%m-%d %H:%M')}; "
        f"ядер {os.cpu_count()}")
    log(f"M {mach}, Re {re:.0e}, Tu {tu} % (Ñкрит {ncrit_from_tu(tu):.2f}); эксперимент: {exp['source']}")
    t0 = time.time()
    pv = run_viscous(af, al, mach, re, ncrit=ncrit_from_tu(tu))
    log(f"вязкий метод: {time.time() - t0:.0f} с")
    t0 = time.time()
    w = auto_workers()
    pr = run_rans(af, al, mach, re, max_iter=400, conv=Convergence(), workers=w,
                  solver_opts={"xtr_source": "viscous", "tu": tu}, coarse_start=True)
    log(f"RANS (SA + переход по вязкому методу), процессов {w}: {(time.time() - t0) / 60:.0f} мин")
    log("сырые данные: α " + " ".join(f"{a:g}" for a in al))
    log("   RANS Cl " + " ".join(f"{v:.4f}" for v in pr.cl) + " | Cd " + " ".join(f"{v:.5f}" for v in pr.cd))
    log("   вязк Cl " + " ".join(f"{v:.4f}" for v in pv.cl) + " | Cd " + " ".join(f"{v:.5f}" for v in pv.cd))
    log("\n   α  | эксп. Cl  Cd*    | вязкий Cl  Cd      x_tr     | RANS Cl  Cd      x_tr      статус")
    for i, a in enumerate(al):
        ec = float(np.interp(a, ea, ecl)) if ea.min() <= a <= ea.max() else np.nan
        rc = pr.cl[i]
        ed = float(np.interp(rc, ecc, ecd)) if np.isfinite(rc) and ecc.min() <= rc <= ecc.max() else np.nan
        st = (pr.notes_per_point[i] if hasattr(pr, "notes_per_point") else "")
        ok = "" if (pr.ok is None or pr.ok[i]) else "не сошлось"
        log(f" {a:5.1f}| {ec:6.3f}  {ed:.4f} | {pv.cl[i]:6.3f}  {pv.cd[i]:.5f}  {pv.xtr_u[i]:.2f}/{pv.xtr_l[i]:.2f} | "
            f"{rc:6.3f}  {pr.cd[i]:.5f}  {pr.xtr_u[i]:.2f}/{pr.xtr_l[i]:.2f}  {ok}{st}")
    log("* Cd эксперимента — при Cl расчёта RANS (из поляры Cd(Cl)).")
    k = np.nanargmax(pr.cl)
    log(f"Cl_max: эксперимент {exp['clmax']:.2f} при {exp['alpha_clmax']:.0f}°; RANS {pr.cl[k]:.3f} при {al[k]:g}°; "
        f"вязкий {np.nanmax(pv.cl):.3f} при {al[int(np.nanargmax(pv.cl))]:g}°")
    log("Готово.")


if __name__ == "__main__":
    main()
