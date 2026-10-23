"""Сценарии расчёта для интерфейса: профиль → поляра (панели, панели +
пограничный слой, Эйлер)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .geometry import Airfoil, naca4, load_airfoil
from .panel import PanelSolver, close_trailing_edge, critical_cp

# Сетки для уравнений Эйлера: (точек на сторону профиля, узлов по нормали, первый шаг)
GRIDS = {
    "грубая (быстро)": (65, 49, 2e-3),
    "средняя": (97, 65, 1.5e-3),
    "мелкая (медленно)": (129, 97, 1e-3),
}


@dataclass
class Polar:
    label: str
    method: str
    mach: float
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray            # NaN для панелей (невязкий расчёт без сопротивления)
    cm: np.ndarray
    ok: np.ndarray            # сошлось / в пределах применимости
    cp_x: list = field(default_factory=list)
    cp: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    re: float = float("nan")
    cdf: np.ndarray | None = None          # сопротивление трения (вязкий расчёт)
    xtr_u: np.ndarray | None = None        # точки перехода
    xtr_l: np.ndarray | None = None
    bl: list = field(default_factory=list)  # распределения пограничного слоя по углам
    cp_inv: list = field(default_factory=list)


def make_airfoil(source: str, value: str, n_side: int = 100) -> Airfoil:
    """source = 'naca' (value — 4 цифры) или 'file' (value — путь)."""
    if source == "naca":
        code = value.strip().upper().replace("NACA", "").strip()
        if len(code) != 4 or not code.isdigit():
            raise ValueError("Код NACA должен состоять из 4 цифр, например 2412")
        if int(code[2:]) == 0:
            raise ValueError("Толщина профиля не может быть нулевой")
        return naca4(code, n_side)
    af = load_airfoil(value, n_side)
    return close_trailing_edge(af)


def alpha_range(a0: float, a1: float, da: float) -> np.ndarray:
    if da <= 0:
        raise ValueError("Шаг по углу атаки должен быть положительным")
    if a1 < a0:
        a0, a1 = a1, a0
    n = int(np.floor((a1 - a0) / da + 1e-9)) + 1
    if n > 200:
        raise ValueError("Слишком много углов атаки (больше 200)")
    return a0 + da * np.arange(n)


def run_panel(af: Airfoil, alphas, mach: float) -> Polar:
    if mach >= 1:
        raise ValueError("Панельный метод с поправкой Кармана–Цяня применим только при M < 1")
    s = PanelSolver(af)
    cl, cm, ok, xs, cps = [], [], [], [], []
    cpstar = critical_cp(mach)
    for a in alphas:
        r = s.solve(float(a), mach)
        cl.append(r.cl); cm.append(r.cm)
        ok.append(bool(np.min(r.cp) > cpstar))
        xs.append(r.x); cps.append(r.cp)
    ok = np.array(ok)
    notes = []
    if mach > 0 and not ok.all():
        bad = np.asarray(alphas)[~ok]
        notes.append(f"При α = {bad.min():g}…{bad.max():g}° на профиле есть сверхзвуковая зона "
                     f"(Cp < Cp* = {cpstar:.2f}): панели+КЦ там неприменимы, нужен Эйлер.")
    notes.append("Панельный метод невязкий: сопротивление и срыв не рассчитываются.")
    return Polar(f"{af.name}, панели, M={mach:g}", "panel", mach, np.asarray(alphas, float),
                 np.array(cl), np.full(len(cl), np.nan), np.array(cm), ok, xs, cps, notes)


def run_euler(af: Airfoil, alphas, mach: float, grid: str = "грубая (быстро)",
              n_iter: int = 100, callback=None) -> Polar:
    """Все углы атаки считаются одним пакетом. callback(text, fraction) -> True для остановки."""
    import jax.numpy as jnp
    from .geometry import repanel
    from .grid import ogrid
    from .euler import EulerSolver
    from .implicit import ImplicitEuler

    if mach < 0.2:
        raise ValueError("Решатель Эйлера пока не работает при M < 0,2 (нет предобусловливания "
                         "для малых чисел Маха). Используйте панельный метод.")
    nside, nj, ds0 = GRIDS[grid]
    if callback:
        callback("Построение сетки…", 0.0)
    g = ogrid(repanel(af, nside), nj=nj, ds0=ds0, radius=40.0)
    solver = ImplicitEuler(EulerSolver(g, dtype=jnp.float32))
    alphas = np.asarray(alphas, float)

    def cb(it, n, logres, cl):
        if callback is None:
            return False
        worst = np.nanmax(logres) if np.any(np.isfinite(logres)) else 0.0
        return callback(f"Итерация {it}/{n}: невязка упала на {-worst:.1f} порядка "
                        f"(худший случай); первая итерация включает компиляцию", it / n)

    if callback:
        callback("Компиляция и первая итерация (может занять минуту)…", 0.0)
    r = solver.solve(np.full_like(alphas, mach), alphas, n_iter=n_iter, tol=1e-6,
                     verbose=False, callback=cb)
    xw = np.asarray(solver.b.m_np.xw)
    ok = np.asarray(r.converged, bool)
    notes = []
    if not ok.all():
        notes.append(f"Не сошлись: α = {', '.join(f'{a:g}' for a in alphas[~ok])}°. "
                     "Их значения ненадёжны (обычно трансзвук с сильными скачками или большие α). "
                     "Попробуйте более мелкую сетку или больше итераций.")
    notes.append("Эйлер невязкий: Cd — только волновое сопротивление (скачки уплотнения), "
                 "трения и срыва нет.")
    return Polar(f"{af.name}, Эйлер ({grid.split(' ')[0]}), M={mach:g}", "euler", mach, alphas,
                 r.cl, r.cd, r.cm, ok, [xw] * len(alphas), [c for c in r.cp], notes)


def run_viscous(af: Airfoil, alphas, mach: float, re: float, ncrit: float = 9.0,
                xtr=(1.0, 1.0), callback=None) -> Polar:
    """Панели + интегральный пограничный слой (как XFOIL). callback(text, fraction)
    -> True для остановки."""
    from .viscous import polar_robust
    if not 1e3 <= re <= 1e9:
        raise ValueError("Число Рейнольдса должно быть в диапазоне 1e3…1e9")
    if mach >= 0.8:
        raise ValueError("Вязкий панельный метод применим при M < 0,8 (сжимаемость — поправкой "
                         "Кармана–Цяня); для трансзвука используйте Эйлер.")
    alphas = np.asarray(alphas, float)
    stop = {"flag": False}

    def cb(i, n, r):
        if callback:
            stop["flag"] = bool(callback(f"Точка {i}/{n}: α = {r.alpha:g}° — "
                                         f"{'сошлась' if r.converged else 'НЕ сошлась'}", i / n))
        if stop["flag"]:
            raise _Stop()

    if callback:
        callback("Первая точка (маршевое начальное приближение)…", 0.0)
    try:
        res = polar_robust(af, re, alphas, ncrit=ncrit, mach=mach, xtr=xtr, callback=cb)
    except _Stop:
        res = None
    if res is None or any(r is None for r in res):
        raise RuntimeError("Расчёт остановлен")
    ok = np.array([r.converged for r in res])
    cpstar = critical_cp(mach)
    sup = np.array([np.min(r.cp) < cpstar for r in res]) if mach > 0 else np.zeros(len(res), bool)
    notes = []
    if not ok.all():
        notes.append(f"Не сошлись: α = {', '.join(f'{a:g}' for a in alphas[~ok])}°. Обычно это "
                     "срыв (вязко-невязкий метод его описывает только до начала) или сильные "
                     "отрывные пузыри при малых Re.")
    if sup.any():
        notes.append(f"При α = {', '.join(f'{a:g}' for a in alphas[sup])}° на профиле есть "
                     "сверхзвуковая зона: поправка Кармана–Цяня там неточна.")
    notes.append(f"Re = {re:.3g}, Ñкрит = {ncrit:g} (9 — типичная аэродинамическая труба; "
                 "меньше — более турбулентный поток). Cd — по Сквайру–Янгу в конце следа, "
                 "Cdf — трение. Срыв на больших углах описывается приближённо.")
    label = f"{af.name}, вязкий, Re={re:.3g}, M={mach:g}"
    return Polar(label, "viscous", mach, alphas, np.array([r.cl for r in res]),
                 np.array([r.cd for r in res]), np.array([r.cm for r in res]), ok & ~sup,
                 [r.x for r in res], [r.cp for r in res], notes, re=re,
                 cdf=np.array([r.cdf for r in res]),
                 xtr_u=np.array([r.xtr_u for r in res]), xtr_l=np.array([r.xtr_l for r in res]),
                 bl=[r.side for r in res], cp_inv=[r.cp_inv for r in res])


class _Stop(Exception):
    pass


def save_csv(polars, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["расчёт", "метод", "M", "Re", "alpha", "Cl", "Cd", "Cdf", "Cm",
                    "xtr_верх", "xtr_низ", "надёжно"])
        f6 = lambda v: "" if v is None or not np.isfinite(v) else f"{v:.6f}"
        for p in polars:
            for k in range(len(p.alpha)):
                w.writerow([p.label, p.method, f"{p.mach:g}",
                            "" if not np.isfinite(p.re) else f"{p.re:.4g}", f"{p.alpha[k]:g}",
                            f6(p.cl[k]), f6(p.cd[k]), f6(None if p.cdf is None else p.cdf[k]),
                            f6(p.cm[k]), f6(None if p.xtr_u is None else p.xtr_u[k]),
                            f6(None if p.xtr_l is None else p.xtr_l[k]),
                            "да" if p.ok[k] else "нет"])
