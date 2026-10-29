"""Точка входа собранной программы (PyInstaller): окно aero2d.

freeze_support() обязателен: параллельные поляры запускают процессы методом
spawn, и в собранной программе каждый процесс — это снова aero2d.exe.

aero2d.exe --selftest — проверка собранной программы без окна: панельный и
вязкий методы (поляра на двух процессах), сетка gmsh, JAX; отчёт — в файл
aero2d_selftest.txt рядом с aero2d.exe (окна консоли у программы нет)."""
import multiprocessing
import os
import sys


def selftest(path):
    import time
    import traceback
    lines = []

    def log(msg):
        lines.append(msg)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    ok = True
    log(f"aero2d selftest, Python {sys.version.split()[0]}, {sys.platform}, frozen={getattr(sys, 'frozen', False)}")
    steps = []

    def step(name):
        def deco(f):
            steps.append((name, f))
            return f
        return deco

    @step("профили из набора")
    def _():
        from aero2d import core
        return f"{len(core.library())} шт."

    @step("панельный метод")
    def _():
        from aero2d import core
        from aero2d.geometry import naca4
        p = core.run_panel(naca4("2412", 120), [0.0, 4.0], 0.0)
        return f"Cl = {p.cl.round(3).tolist()}"

    @step("вязкий метод, поляра на 2 процессах")
    def _():
        from aero2d import core
        from aero2d.geometry import naca4
        p = core.run_viscous_batch(naca4("2412", 160), [0.0, 2.0, 4.0, 6.0], 0.0, [1e6], workers=2)[0]
        assert p.ok.all(), "не все точки сошлись"
        return f"Cl = {p.cl.round(3).tolist()}"

    @step("JAX")
    def _():
        import jax
        import jax.numpy as jnp
        return f"{jax.__version__}, {jax.devices()[0]}, sum = {float(jnp.arange(10.0).sum()):g}"

    @step("сетка gmsh")
    def _():
        from aero2d.geometry import naca4, repanel
        from aero2d.rans import mesh_for
        from aero2d.rans_mesh import MeshOptions
        m = mesh_for([repanel(naca4("0012", 120), 50)], 0.0, 1e6,
                     opts=MeshOptions(far_radius=15, far_size=5.0, near_size=0.02))
        return f"{m.n_cells} ячеек"

    for name, f in steps:
        t = time.time()
        try:
            log(f"OK    {name}: {f()} ({time.time() - t:.1f} с)")
        except Exception:                                    # noqa: BLE001
            ok = False
            log(f"ОШИБКА {name}:\n{traceback.format_exc()}")
    log("ИТОГ: всё работает" if ok else "ИТОГ: есть ошибки — пришлите этот файл")
    return 0 if ok else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv:
        base = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
        sys.exit(selftest(os.path.join(base, "aero2d_selftest.txt")))
    from aero2d.gui import main
    main()
