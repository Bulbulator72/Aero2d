"""Главный тест прототипа на вашей машине: пропускная способность пакетного
решателя на GPU и пригодность одинарной точности.

Запуск (Linux или WSL2 с CUDA):
    pip install -U "jax[cuda12]"
    python benchmarks/gpu_throughput.py            # FP32, пакеты 1…128
    python benchmarks/gpu_throughput.py --fp64     # то же в FP64 (для сравнения)

Что измеряется:
  * время одной итерации Ньютона на один случай при разном размере пакета —
    видно, с какого размера пакета GPU «насыщается»;
  * время полной поляры NACA 0012 (M = 0.7, α = −4…12°, 17 точек одним пакетом);
  * расхождение FP32 и FP64 по Cl, Cd (при запуске с --fp64).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--nside", type=int, default=129, help="точек на сторону профиля")
    ap.add_argument("--nj", type=int, default=97, help="узлов по нормали")
    ap.add_argument("--batches", default="1,4,16,32,64,128")
    a = ap.parse_args()

    import jax
    if a.fp64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from aero2d.geometry import naca4
    from aero2d.grid import ogrid
    from aero2d.euler import EulerSolver
    from aero2d.implicit import ImplicitEuler

    print("Устройства JAX:", jax.devices())
    dtype = jnp.float64 if a.fp64 else jnp.float32
    g = ogrid(naca4("0012", a.nside), nj=a.nj, ds0=1e-3, radius=40.0)
    print(f"Сетка {g.shape[0]}×{g.shape[1]} узлов, {(g.shape[0]) * (g.shape[1] - 1)} ячеек, {dtype.__name__}")
    s = ImplicitEuler(EulerSolver(g, dtype=dtype))

    # --- пропускная способность ----------------------------------------
    print("\nПакет | с/итерация | мс/итерация на случай")
    for B in [int(b) for b in a.batches.split(",")]:
        machs = np.linspace(0.3, 0.75, B)
        alphas = np.linspace(-2, 6, B)
        try:
            s.solve(machs, alphas, n_iter=2, verbose=False)        # компиляция
            t = time.perf_counter()
            s.solve(machs, alphas, n_iter=5, verbose=False)
            dt = (time.perf_counter() - t) / 5
            print(f"{B:5d} | {dt:10.3f} | {1000 * dt / B:10.1f}")
        except Exception as e:  # обычно — нехватка памяти GPU
            print(f"{B:5d} | ошибка: {type(e).__name__}: {str(e)[:80]}")
            break

    # --- полная поляра ---------------------------------------------------
    alphas = np.arange(-4, 12.1, 1.0)
    machs = np.full_like(alphas, 0.7)
    t = time.perf_counter()
    r = s.solve(machs, alphas, n_iter=120, verbose=False, tol=1e-6)
    el = time.perf_counter() - t
    print(f"\nПоляра M=0.7, {len(alphas)} точек: {el:.1f} с ({s.iterations} итераций Ньютона, с компиляцией)")
    print(" α     Cl       Cd       Cm     log10(res)")
    drop = np.log10(r.res_hist[-1] / r.res_hist[0])
    for k in range(len(alphas)):
        print(f"{alphas[k]:4.0f} {r.cl[k]:8.4f} {r.cd[k]:8.5f} {r.cm[k]:8.4f}   {drop[k]:6.1f}")
    out = os.path.join(os.path.dirname(__file__), f"polar_M07_{'fp64' if a.fp64 else 'fp32'}.csv")
    np.savetxt(out, np.column_stack([alphas, r.cl, r.cd, r.cm, drop]), delimiter=",",
               header="alpha,cl,cd,cm,log10_res_drop", comments="", fmt="%.6g")
    print("Сохранено:", out)
    other = out.replace("fp64", "fp32") if a.fp64 else out.replace("fp32", "fp64")
    if os.path.exists(other):
        o = np.loadtxt(other, delimiter=",", skiprows=1)
        print(f"Макс. расхождение FP32/FP64: ΔCl = {np.max(np.abs(o[:, 1] - r.cl)):.2e}, "
              f"ΔCd = {np.max(np.abs(o[:, 2] - r.cd)):.2e}")


if __name__ == "__main__":
    main()
