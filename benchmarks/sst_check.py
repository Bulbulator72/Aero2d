"""Проверка модели k-ω SST (вариант NASA SST-V с поддерживающими членами) по эталонам NASA TMR
(tmbwg.github.io/turbmodels). Батник 22. Отчёт — benchmarks/sst_report.txt, каждый случай пишется
сразу, как досчитан.

1. Пластина (M 0,2, Re 5·10⁶ на ед. длины): Cf при x = 0,2 и 0,5 — SA и SST
   (CFL3D: SA 0,00342 / 0,00298; SST-V 0,00342 / 0,00297).
2. NACA 0012 (геометрия TMR, M 0,15, Re 6·10⁶): α = 0, 10, 15° — SST и SA
   (CFL3D, сетка 897×257: SST Cl 1,0778 / 1,5068, Cd 0,00809 / 0,01236 / 0,02219;
   SA Cl 1,0909 / 1,5461, Cd 0,00819 / 0,01231 / 0,02124).
3. NACA 4412 (M 0,09, Re 1,52·10⁶, α = 13,87°, отрыв у задней кромки): SA и SST
   (CFL3D 897×257: SA Cl 1,721, Cd 0,0286 (Cdp 0,02156, Cdf 0,00704), отрыв x/c 0,787;
   SST Cl 1,616, Cd 0,0311 (Cdp 0,0242, Cdf 0,00687), отрыв 0,772).
Аргумент — номера случаев через запятую (по умолчанию 1,2,3)."""
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
REPORT = os.path.join(HERE, "sst_report.txt")


def log(s=""):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def progress(tag):
    t0 = time.time()

    def cb(k, n, it, rn, res_):
        if res_ is None and it % 20 == 0:
            print(f"     {tag}: итерация {it}, невязка {rn:.2e}  ({(time.time() - t0) / 60:.0f} мин)", flush=True)
        return False
    return cb


def plate():
    from aft_check import plate_mesh
    from aero2d.rans import Convergence, RansSolver
    re = 5e6
    m = plate_mesh(re)
    log(f"\n[1] Пластина, M 0,2, Re 5e6/ед. длины, ячеек {m.n_cells}")
    log("   модель | Cf(0,2)  эталон | Cf(0,5)  эталон | итер., мин, статус")
    ref = {"sa": (0.00342, 0.00298), "sst": (0.00342, 0.00297)}
    Q = None
    for tm in ("sa", "sst"):
        t = time.time()
        s = RansSolver(m, 0.2, re, vortex_bc=False)
        s.limiter_zone = "off"
        s.turb_model = tm
        r = s.solve(0.0, Q0=Q, alpha0=0.0 if Q is not None else None, max_iter=400, conv=Convergence())
        w = np.asarray(s.prim(s.jnp.asarray(r.state)))
        xw = m.centroid[s.cw, 0]
        cf = s.mu_inf * w[s.cw, 1] / s.dn_w / (0.5 * 0.2 ** 2)
        o = np.argsort(xw)
        c2, c5 = (float(np.interp(x, xw[o], cf[o])) for x in (0.2, 0.5))
        log(f"   {tm:6s} | {c2:.5f} {ref[tm][0]:.5f} | {c5:.5f} {ref[tm][1]:.5f} | "
            f"{r.iterations}, {(time.time() - t) / 60:.0f}, {r.status}")
        Q = r.state


def sep_x(r, chord=1.0):
    """Точка отрыва на верхней стороне: первая смена знака Cf за x/c 0,3 (по граням стенки)."""
    e = r.elements[0]
    x, y, cf = (np.asarray(e[k]) for k in ("x", "y", "cf"))
    xs = []
    # верх — грани с наибольшим y при данном x: делим по средней линии y_mid(x)
    xb = np.linspace(0, 1, 101)
    ym = np.array([0.5 * (y[(abs(x - a) < 0.02)].max() + y[(abs(x - a) < 0.02)].min())
                   if np.any(abs(x - a) < 0.02) else 0.0 for a in xb])
    up = y > np.interp(x, xb, ym)
    k = np.where(up)[0]
    k = k[np.argsort(x[k])]
    xu, cu = x[k], cf[k]
    for i in range(len(xu) - 1):
        if xu[i] > 0.3 * chord and cu[i] > 0 >= cu[i + 1]:
            xs.append(xu[i] + (xu[i + 1] - xu[i]) * cu[i] / (cu[i] - cu[i + 1]))
            break
    return xs[0] if xs else float("nan")


def airfoil(name, af, re, mach, alphas, refs, sep=False, split=None):
    from aero2d.rans import Convergence, mesh_for, rans_polar
    mesh = mesh_for([af], max(alphas), re)
    log(f"\n[{name}] M {mach}, Re {re:.3g}, сетка {mesh.n_cells} ячеек")
    log("   модель   α   |  Cl     эталон |  Cd       эталон" + (" | отрыв x/c эталон" if sep else "")
        + (" | Cdp     эталон  | Cdf     эталон" if split else "") + " | итер., мин, статус")
    for tm in ("sst", "sa"):
        t = time.time()
        rr, _ = rans_polar([af], alphas, re, mach=mach, mesh=mesh, max_iter=500, conv=Convergence(), workers=1,
                           coarse_start=True, callback=progress(tm), solver_opts={"turb_model": tm})
        dt = (time.time() - t) / 60
        for a, r in zip(alphas, rr):
            rc, rd, rs = refs[tm].get(a, (np.nan, np.nan, np.nan))
            line = f"   {tm:6s} {a:5.2f} | {r.cl:6.4f} {rc:6.4f} | {r.cd:.5f}  {rd:.5f}"
            if sep:
                line += f" | {sep_x(r):.3f}     {rs:.3f}"
            if split:
                sp, sf = split[tm]
                line += f" | {r.cdp:.5f} {sp:.5f} | {r.cdf:.5f} {sf:.5f}"
            log(line + f" | {r.iterations}, {dt:.0f} (все углы), {r.status}")


def main():
    import aero2d
    from aero2d.geometry import naca4, repanel
    cases = [int(c) for c in (sys.argv[1].split(",") if len(sys.argv) > 1 else "1,2,3".split(","))]
    log(f"=== Проверка k-ω SST по NASA TMR; aero2d {aero2d.__version__}; {time.strftime('%Y-%m-%d %H:%M')}; "
        f"ядер {os.cpu_count()}")
    for c in cases:
        try:
            if c == 1:
                plate()
            elif c == 2:
                af = repanel(naca4("0012", 200, scaled_te=True), 130)
                airfoil("2] NACA 0012 (TMR", af, 6e6, 0.15, [0.0, 10.0, 15.0],
                        {"sst": {0.0: (0.0, 0.00809, 0), 10.0: (1.0778, 0.01236, 0), 15.0: (1.5068, 0.02219, 0)},
                         "sa": {0.0: (0.0, 0.00819, 0), 10.0: (1.0909, 0.01231, 0), 15.0: (1.5461, 0.02124, 0)}})
            elif c == 3:
                af = repanel(naca4("4412", 200), 130)
                airfoil("3] NACA 4412 (TMR, отрыв у кромки", af, 1.52e6, 0.09, [13.87],
                        {"sst": {13.87: (1.616, 0.0311, 0.772)}, "sa": {13.87: (1.721, 0.0286, 0.787)}}, sep=True,
                        split={"sst": (0.0242, 0.00687), "sa": (0.02156, 0.00704)})
        except Exception:                                  # noqa: BLE001
            log(f"случай {c}: ошибка\n" + traceback.format_exc())
    log("Готово.")


if __name__ == "__main__":
    main()
