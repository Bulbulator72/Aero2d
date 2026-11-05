"""Поиск причины заниженного Cd NACA 0012 при α = 10° (M 0,15, Re 6e6, SA).

Эталон NASA TMR (CFL3D, 897×257): Cl 1,0909, Cd 0,01231 (давл. 0,00612, трение ≈0,0062).
У нас на сетке проверки: Cl 1,088, Cd 0,0109 — трение совпадает, давление ниже на 0,0013.
Уже исключены: размер области (200 хорд), число точек на профиле (260),
недосходимость (невязка 8e-7, силы не меняются), форма профиля (профиль TMR).

Каждый вариант меняет одно: вязкие потоки и источник SA по неограниченным
градиентам, второй порядок переноса ν̃, без ограничителя, более частая сетка
поперёк слоя. Отчёт: benchmarks/cd_study_report.txt (дописывается после каждого
варианта — можно прислать и недосчитанный).

Запуск: python benchmarks/rans_cd_study.py [имена вариантов через пробел]"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cd_study_report.txt")
XS = (0.005, 0.02, 0.1, 0.3, 0.6, 0.9, 0.98)
REF_UP = (-5.506, -4.001, -1.962, -0.976, -0.404, 0.015, 0.151)
REF_LO = (0.135, 0.996, 0.618, 0.243, 0.120, 0.113, 0.154)
REF_CF = (0.0256, 0.01509, 0.00700, 0.00402, 0.00235, 0.00091, 0.00027)

# имя: (описание, параметры решателя, параметры сетки)
VARIANTS = {
    "base": ("как в проверке (ограничитель у кромок)", {}, {}),
    "rawgrad": ("вязкие потоки и источник SA по неограниченным градиентам",
                {"visc_raw_grad": True, "sa_raw_grad": True}, {}),
    "turb2": ("ν̃ переносится вторым порядком", {"turb_order": 2}, {}),
    "nolim": ("без ограничителя", {"limiter_zone": "off"}, {}),
    "ratio11": ("сетка: рост слоя 1,1 вместо 1,2", {}, {"ratio": 1.1}),
    "yplus05": ("сетка: y+ 0,5 вместо 1", {}, {"yplus": 0.5}),
    # серия по y+ без ограничителя (так расчёт сходится лучше всего): куда идёт решение
    # при измельчении у стенки — к эталону или от него
    "yp2": ("серия y+: 2, без ограничителя", {"limiter_zone": "off"}, {"yplus": 2.0}),
    "yp1": ("серия y+: 1, без ограничителя", {"limiter_zone": "off"}, {"yplus": 1.0}),
    "yp05": ("серия y+: 0,5, без ограничителя", {"limiter_zone": "off"}, {"yplus": 0.5}),
    "yp025": ("серия y+: 0,25, без ограничителя", {"limiter_zone": "off"}, {"yplus": 0.25}),
}
SERIES = ("yp2", "yp1", "yp05", "yp025")


def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def run(name):
    from aero2d.geometry import naca4, repanel
    from aero2d.rans import Convergence, RansSolver, rans_polar
    from aero2d.rans_mesh import build_mesh
    desc, sopts, mopts = VARIANTS[name]
    af = repanel(naca4("0012", 200), 130)
    t = time.time()
    mesh = build_mesh([af], re=6e6, far_radius=50, **mopts)
    opts = {"limiter_zone": "te"}
    opts.update(sopts)
    res, _ = rans_polar([af], [10.0], 6e6, mach=0.15, mesh=mesh, max_iter=400, conv=Convergence(), workers=1,
                        coarse_start=True, solver_opts=opts)
    r = res[0]
    log(f"\n[{name}] {desc}: ячеек {mesh.n_cells}")
    log(f"   Cl {r.cl:.4f}  Cd {r.cd:.5f}  давл. {r.cdp:.5f}  трение {r.cdf:.5f}  итераций {r.iterations}, "
        f"{r.status}, {time.time() - t:.0f} с")
    s = RansSolver(mesh, 0.15, 6e6)
    for k, v in opts.items():
        setattr(s, k, v)
    e = s.forces(r.state, 10.0)["elements"][0]
    x, y, cp, cf = e["x"], e["y"], e["cp"], e["cf"]
    up = y > 0
    ou, ol = np.argsort(x[up]), np.argsort(x[~up])
    log("   x       Cp верх (TMR)      Cp низ (TMR)      Cf верх (TMR)")
    for i, xx in enumerate(XS):
        log("   %.3f  %7.3f (%7.3f)  %6.3f (%6.3f)  %.5f (%.5f)" % (
            xx, np.interp(xx, x[up][ou], cp[up][ou]), REF_UP[i], np.interp(xx, x[~up][ol], cp[~up][ol]), REF_LO[i],
            abs(np.interp(xx, x[up][ou], cf[up][ou])), REF_CF[i]))


def main():
    # без аргументов — серия по y+ (первые 6 вариантов уже посчитаны 2026-11-05)
    names = [a for a in sys.argv[1:] if a in VARIANTS] or list(SERIES)
    import aero2d
    log(f"=== Поиск причины низкого Cd, NACA 0012 α 10°; aero2d {aero2d.__version__}; "
        f"{time.strftime('%Y-%m-%d %H:%M')}; ядер {os.cpu_count()}")
    log("эталон TMR: Cl 1,0909  Cd 0,01231  давл. 0,00612  трение ≈0,0062")
    for n in names:
        try:
            run(n)
        except Exception as ex:          # один упавший вариант не должен останавливать остальные
            log(f"\n[{n}] ошибка: {type(ex).__name__}: {ex}")
    log("\nГотово.")


if __name__ == "__main__":
    main()
