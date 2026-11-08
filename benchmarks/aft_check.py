"""Проверка модели перехода SA-AFT2019b (батник 18_RANS_transition_check.bat).

1) Пластина Шубауэра–Клебанова (Tu 0,0408 % → Ñкрит 10,3) — сверка с FUN3D и OVERFLOW
   (Hildebrand et al., AIAA AVIATION 2023, https://ntrs.nasa.gov/citations/20230004109):
   ламинарное Cf при Re_x 3·10⁶ — 0,000381; начало перехода Re_x ≈ 4,8–5,0·10⁶;
   турбулентное Cf при Re_x 7·10⁶ — 0,00299.
2) NLF(1)-0416, M 0,1, Re 4·10⁶, α 0°, Tu 0,15 % — сверка с экспериментом (NASA TP-1861:
   переход x/c ≈ 0,43 сверху, ≈ 0,60 снизу, c_l ≈ 0,45) и вязким методом aero2d.

Расчёт профиля сохраняется каждые 20 итераций (aft_check_state.pkl): если окно закрыть,
повторный запуск продолжит с места остановки. Отчёт — benchmarks/aft_report.txt.
Аргументы: plate — только пластина; nlf — только профиль; nlf_tr — профиль на сетке со
сгущением у точек перехода (по вязкому методу; сохранение aft_check_state_tr.pkl); без
аргументов — пластина и профиль."""
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
REPORT = os.path.join(HERE, "aft_report.txt")
STATE = os.path.join(HERE, "aft_check_state.pkl")


def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def geom(n, h0, L):
    from scipy.optimize import brentq
    N = n - 1
    r = brentq(lambda r: h0 * (r ** N - 1) / (r - 1) - L, 1.0 + 1e-9, 3.0)
    return np.r_[0.0, np.cumsum(h0 * r ** np.arange(N))]


def plate_mesh(re, nx_pl=241, nx_up=41, ny=97, H=0.5, yplus=0.5, dx_le=2e-4):
    """Прямоугольная сетка: x от −0,5 до 1 (пластина 0…1 на нижней границе), y от 0 до H."""
    from aero2d.rans_mesh import _connectivity
    cf = 0.026 * re ** (-1 / 7)
    h0 = yplus / (re * np.sqrt(cf / 2))
    x = np.r_[-geom(nx_up, dx_le, 0.5)[::-1][:-1], geom(nx_pl, dx_le, 1.0)]
    y = geom(ny, h0, H)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    ni, nj = len(x), len(y)
    ids = np.arange(ni * nj).reshape(nj, ni)
    cells = [np.array([ids[j, i], ids[j, i + 1], ids[j + 1, i + 1], ids[j + 1, i]])
             for j in range(nj - 1) for i in range(ni - 1)]
    key = lambda a, b: (min(a, b), max(a, b))
    wall, far = {}, set()
    for i in range(ni - 1):
        k = key(ids[0, i], ids[0, i + 1])
        if i >= nx_up - 1:
            wall[k] = 0
        else:
            far.add(k)
        far.add(key(ids[nj - 1, i], ids[nj - 1, i + 1]))
    for j in range(nj - 1):
        far.add(key(ids[j, 0], ids[j + 1, 0]))
        far.add(key(ids[j, ni - 1], ids[j + 1, ni - 1]))
    return _connectivity(nodes, cells, wall, far, None)


def plate():
    from aero2d.rans import Convergence, RansSolver, ncrit_from_tu
    re = 8e6                                   # длина пластины 1 → Re_x до 8·10⁶
    t0 = time.time()
    m = plate_mesh(re)
    lam = RansSolver(m, 0.15, re, vortex_bc=False, turbulent=False)
    lam.limiter_zone = "off"
    r = lam.solve(0.0, max_iter=200, conv=Convergence())
    Q = r.state
    rho = Q[:, 0]
    Q7 = np.column_stack([Q[:, :4], 0.1 * rho, 0 * rho, 0 * rho])
    s = RansSolver(m, 0.15, re, vortex_bc=False)
    s.limiter_zone = "off"
    s.turb_model = "sa-aft"
    s.tu = 0.0408
    r = s.solve(0.0, Q0=Q7, alpha0=0.0, max_iter=900, conv=Convergence())
    w = np.asarray(s.prim(s.jnp.asarray(r.state)))
    xw = m.centroid[s.cw, 0]
    cf = s.mu_inf * w[s.cw, 1] / s.dn_w / (0.5 * 0.15 ** 2)
    o = np.argsort(xw)
    rx, cf = re * xw[o], cf[o]
    k = (rx > 1e6) & (rx < 7.9e6)
    log(f"\n[пластина Шубауэра–Клебанова] Ñкрит {ncrit_from_tu(0.0408):.2f}, ячеек {m.n_cells}: "
        f"{r.status}, итераций {r.iterations}, {time.time() - t0:.0f} с")
    log("   Re_x      Cf        FUN3D/OVERFLOW")
    ref = {3e6: "0.000381", 5e6: "(переходная зона)", 7e6: "0.00299"}
    for v in (1e6, 2e6, 3e6, 4e6, 4.5e6, 5e6, 5.5e6, 6e6, 7e6):
        log(f"   {v:.1e}   {np.interp(v, rx, cf):.5f}   {ref.get(v, '')}")
    log(f"   начало перехода (минимум Cf) Re_x = {rx[k][np.argmin(cf[k])]:.2e}   (FUN3D/OVERFLOW ≈ 4,8–5,0e6)")


def nlf(refine=False):
    from scipy.spatial import cKDTree
    from aero2d.core import _rans_xtr
    from aero2d.geometry import load_airfoil, repanel
    from aero2d.rans import Convergence, RansSolver, mesh_for, ncrit_from_tu
    from aero2d.viscous import ViscousSolver
    a, re, tu, mach = 0.0, 4e6, 0.15, 0.1
    af = repanel(load_airfoil(os.path.join(HERE, "data", "nlf0416", "nlf0416_coord.dat")), 160)
    nc = ncrit_from_tu(tu)
    state = STATE.replace(".pkl", "_tr.pkl") if refine else STATE
    title = "NLF(1)-0416, сетка со сгущением у точек перехода" if refine else "NLF(1)-0416"
    if os.path.exists(state):
        d = pickle.load(open(state, "rb"))
        log(f"\n[{title}] продолжение с итерации {d['it']}")
    else:
        vs = ViscousSolver(af, re, ncrit=nc, mach=mach).solve(a)
        log(f"\n[{title}] M {mach}, Re {re:.0e}, α {a}°, Tu {tu} % (Ñкрит {nc:.2f})")
        log(f"   вязкий метод: Cl {vs.cl:.4f}  Cd {vs.cd:.5f}  переход {vs.xtr_u:.3f} / {vs.xtr_l:.3f}")
        log("   эксперимент (TP-1861): Cl ≈ 0,45, переход ≈ 0,43 / 0,60")
        from aero2d.rans import transition_mesh_options
        from aero2d.rans_mesh import MeshOptions
        mo = transition_mesh_options(MeshOptions(transition_refine=None if refine else False),
                                     {"turb_model": "sa-aft", "tu": tu}, [a], mach)
        d = dict(mesh=mesh_for([af], a, re, opts=mo), Q=None, it=0, cfl=None, t=0.0)
        z = getattr(d["mesh"], "tr_zones", None)
        log(f"   сетка: {d['mesh'].n_cells} ячеек" + (f"; сгущение (шаг {mo.tr_ds:g} хорды): верх "
            f"{z['upper'][0]:.2f}–{z['upper'][1]:.2f}, низ {z['lower'][0]:.2f}–{z['lower'][1]:.2f}" if z else ""))
    m = d["mesh"]
    s = RansSolver(m, mach, re)
    s.turb_model = "sa-aft"
    s.tu = tu
    s.aft_state = dict(d.get("aft_state", {}))
    def xtr(fr):
        el = fr["elements"][0]
        if el.get("gam") is None:
            return np.nan, np.nan
        _, j = cKDTree(np.column_stack([af.x, af.y])).query(np.column_stack([el["x"], el["y"]]))
        o = np.argsort(j, kind="stable")
        return _rans_xtr(np.asarray(af.x), j[o], el["x"][o], el["gam"][o])

    def steady(fh, n=10):
        """Силы установились: за последние n кусков (20·n итераций) Cl меняется < 0,0005, Cd < 0,00002."""
        if len(fh) < n:
            return False
        cl = [f[1] for f in fh[-n:]]
        cd = [f[2] for f in fh[-n:]]
        return max(cl) - min(cl) < 5e-4 and max(cd) - min(cd) < 2e-5

    d.setdefault("fh", [])
    status = ""
    print("   итерация: Cl, Cd, невязка (во сколько раз упала от наибольшей), CFL, переход верх/низ", flush=True)
    while d["it"] < int(os.environ.get("AFT_IT", "3000")):
        t = time.time()
        r = s.solve(a, Q0=d["Q"], alpha0=a if d["Q"] is not None else None, max_iter=20, conv=Convergence(),
                    cfl0=d["cfl"])
        d["Q"], d["it"], d["t"] = r.state, d["it"] + r.iterations, d["t"] + time.time() - t
        hc = np.atleast_1d((r.history or {}).get("cfl", [50.0]))
        d["cfl"] = float(hc[-1]) if len(hc) else 50.0
        d["aft_state"] = dict(s.aft_state)
        rn = np.atleast_1d((r.history or {}).get("rn", [np.nan]))
        rn = float(rn[-1]) if len(rn) else np.nan
        fr = s.forces(r.state, a)
        d["fh"].append((d["it"], fr["cl"], fr["cd"], rn))
        pickle.dump(d, open(state + ".tmp", "wb"))
        os.replace(state + ".tmp", state)
        rmax = max(f[3] for f in d["fh"] if np.isfinite(f[3])) if any(np.isfinite(f[3]) for f in d["fh"]) else np.nan
        xu, xl = xtr(fr)
        ok = steady(d["fh"])
        print(f"   {d['it']}: Cl {fr['cl']:.4f}  Cd {fr['cd']:.5f}  невязка {rn:.2e} (÷{rmax / rn:.0f})  "
              f"CFL {d['cfl']:.0f}  переход {xu:.3f}/{xl:.3f}  ({d['t'] / 60:.0f} мин)"
              + ("  — силы установились" if ok else ""), flush=True)
        status = r.status
        if status.startswith("сошлось"):
            break
        if ok:
            status = "силы установились (за 200 итераций Cl < 0,0005, Cd < 0,00002) — расчёт остановлен"
            break
    fr = s.forces(d["Q"], a)
    xt = xtr(fr)
    log(f"   SA-AFT: Cl {fr['cl']:.4f}  Cd {fr['cd']:.5f}  переход {xt[0]:.3f} / {xt[1]:.3f}; "
        f"{status}, итераций {d['it']}, {d['t'] / 60:.0f} мин")


def nlf_vtr():
    """NLF(1)-0416 в режиме «SA + переход по вязкому методу» (батник 20)."""
    from aero2d.core import run_rans
    from aero2d.geometry import load_airfoil, repanel
    from aero2d.rans import Convergence
    a, re, tu, mach = 0.0, 4e6, 0.15, 0.1
    af = repanel(load_airfoil(os.path.join(HERE, "data", "nlf0416", "nlf0416_coord.dat")), 160)
    t0 = time.time()
    p = run_rans(af, [a], mach, re, max_iter=400, conv=Convergence(),
                 solver_opts={"xtr_source": "viscous", "tu": tu}, coarse_start=True)
    log(f"\n[NLF(1)-0416, SA + переход по вязкому методу] M {mach}, Re {re:.0e}, α {a}°, Tu {tu} %")
    log("   эксперимент (TP-1861): Cl ≈ 0,45, переход ≈ 0,43 / 0,60; вязкий метод: Cl 0,4707, Cd 0,00517")
    log(f"   RANS: Cl {p.cl[0]:.4f}  Cd {p.cd[0]:.5f}  переход {p.xtr_u[0]:.3f} / {p.xtr_l[0]:.3f}; "
        f"{(time.time() - t0) / 60:.0f} мин")


def main():
    import aero2d
    what = sys.argv[1:] or ["plate", "nlf"]
    log(f"=== Проверка модели перехода SA-AFT; aero2d {aero2d.__version__}; {time.strftime('%Y-%m-%d %H:%M')}; "
        f"ядер {os.cpu_count()}")
    if "plate" in what:
        plate()
    if "nlf" in what:
        nlf()
    if "nlf_tr" in what:
        nlf(refine=True)
    if "nlf_vtr" in what:
        nlf_vtr()
    log("\nГотово.")


if __name__ == "__main__":
    main()
