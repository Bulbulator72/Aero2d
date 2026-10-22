"""Эталонный набор aero2d (этап 0/1).

Запуск:  python benchmarks/run_benchmarks.py [--quick] [--fp64]

1. Панельный метод против точных решений Кармана–Треффца (сходимость).
2. Эйлер, дозвук: NACA 0012, M = 0.5, α = 2° против панелей + Кармана–Цяня.
3. Эйлер, трансзвук: NACA 0012, M = 0.8, α = 1.25° (AGARD AR-211,
   Vassberg & Jameson 2010: Cl ≈ 0.351, Cd ≈ 0.0227) и M = 0.85, α = 1°.
4. FP32 против FP64 на одной сетке (если доступна двойная точность).
5. Пакетный режим: время на один случай при разных размерах пакета.

Результаты: benchmarks/results.json и benchmarks/results.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aero2d.geometry import KarmanTrefftz, Joukowski, naca4   # noqa: E402
from aero2d.panel import PanelSolver                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

REF_TRANSONIC = {
    (0.80, 1.25): dict(cl=0.351, cd=0.0227, src="Vassberg & Jameson (2010), AGARD AR-211"),
}


def bench_panel():
    rows = []
    for g in [Joukowski(0.1, 0.0), KarmanTrefftz(0.1, 0.05, 10), KarmanTrefftz(0.08, 0.04, 15)]:
        for n in (81, 161, 321):
            t = time.perf_counter()
            s = PanelSolver(g.airfoil(n))
            r = s.solve(4.0)
            dt = time.perf_counter() - t
            rows.append(dict(case=g.airfoil(5).name, n=n, cl=r.cl, cl_exact=g.cl(4.0),
                             err_pct=100 * (r.cl / g.cl(4.0) - 1), time_s=dt))
    return rows


def bench_euler(quick: bool, fp64: bool):
    import jax
    if fp64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from aero2d.grid import ogrid
    from aero2d.euler import EulerSolver
    from aero2d.implicit import ImplicitEuler

    nside, nj = (65, 49) if quick else (97, 65)
    n_iter = 80 if quick else 100
    grid = ogrid(naca4("0012", nside), nj=nj, ds0=2e-3 if quick else 1.5e-3, radius=40.0)
    machs = np.array([0.5, 0.8, 0.85])
    alphas = np.array([2.0, 1.25, 1.0])
    out = {"grid": list(grid.shape), "cases": []}
    results = {}
    for dtype, tag in ([(jnp.float32, "fp32")] + ([(jnp.float64, "fp64")] if fp64 else [])):
        s = ImplicitEuler(EulerSolver(grid, dtype=dtype))
        t = time.perf_counter()
        r = s.solve(machs, alphas, n_iter=n_iter, verbose=False, tol=1e-6)
        el = time.perf_counter() - t
        results[tag] = r
        out[f"time_{tag}_s"] = el
        out[f"newton_its_{tag}"] = s.iterations
        out[f"log10_res_drop_{tag}"] = [float(v) for v in np.log10(r.res_hist[-1] / r.res_hist[0])]
    ps = PanelSolver(naca4("0012", 161))
    r = results["fp32"]
    for k, (m, a) in enumerate(zip(machs, alphas)):
        row = dict(mach=float(m), alpha=float(a), cl=float(r.cl[k]), cd=float(r.cd[k]),
                   cm=float(r.cm[k]))
        if m < 0.7:
            row["cl_panel_KT"] = float(ps.solve(a, m).cl)
        if (round(m, 2), round(a, 2)) in REF_TRANSONIC:
            row["ref"] = REF_TRANSONIC[(round(m, 2), round(a, 2))]
        if "fp64" in results:
            row["cl_fp64"] = float(results["fp64"].cl[k])
            row["cd_fp64"] = float(results["fp64"].cd[k])
        out["cases"].append(row)
    return out


def write_md(res, path):
    L = ["# Результаты эталонных тестов aero2d", ""]
    L += ["## Панельный метод против точного решения (α = 4°)", "",
          "| профиль | N | Cl | Cl точн. | ошибка, % |", "|---|---|---|---|---|"]
    for r in res["panel"]:
        L.append(f"| {r['case']} | {r['n']} | {r['cl']:.5f} | {r['cl_exact']:.5f} | {r['err_pct']:+.4f} |")
    if "euler" in res:
        e = res["euler"]
        L += ["", f"## Эйлер, NACA 0012, сетка {e['grid'][0]}×{e['grid'][1]} узлов", "",
              "| M | α | Cl | Cd | Cm | сравнение |", "|---|---|---|---|---|---|"]
        for c in e["cases"]:
            cmp = []
            if "cl_panel_KT" in c:
                cmp.append(f"панели+КЦ Cl={c['cl_panel_KT']:.4f}")
            if "ref" in c:
                cmp.append(f"эталон Cl={c['ref']['cl']}, Cd={c['ref']['cd']}")
            if "cl_fp64" in c:
                cmp.append(f"FP64 Cl={c['cl_fp64']:.4f}, Cd={c['cd_fp64']:.5f}")
            L.append(f"| {c['mach']} | {c['alpha']} | {c['cl']:.4f} | {c['cd']:.5f} | {c['cm']:.4f} | {'; '.join(cmp)} |")
        L.append("")
        L.append("Время (CPU этой машины, все случаи пакетом): " +
                 ", ".join(f"{k[5:-2]} {v:.0f} с" for k, v in e.items() if k.startswith("time_")))
        for k, v in e.items():
            if k.startswith("log10_res_drop_"):
                L.append(f"Падение невязки ({k[15:]}), порядков: " + ", ".join(f"{x:.1f}" for x in v)
                         + f"; итераций Ньютона: {e['newton_its_' + k[15:]]}")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--no-euler", action="store_true")
    a = ap.parse_args()
    res = {"panel": bench_panel()}
    if not a.no_euler:
        res["euler"] = bench_euler(a.quick, a.fp64)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    write_md(res, os.path.join(HERE, "results.md"))
    print(open(os.path.join(HERE, "results.md")).read())


if __name__ == "__main__":
    main()
