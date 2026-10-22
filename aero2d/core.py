"""Сценарии расчёта для интерфейса: профиль → поляра (панели или Эйлер)."""
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


def save_csv(polars, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["расчёт", "метод", "M", "alpha", "Cl", "Cd", "Cm", "надёжно"])
        for p in polars:
            for k in range(len(p.alpha)):
                cd = "" if np.isnan(p.cd[k]) else f"{p.cd[k]:.6f}"
                w.writerow([p.label, p.method, f"{p.mach:g}", f"{p.alpha[k]:g}",
                            f"{p.cl[k]:.6f}", cd, f"{p.cm[k]:.6f}", "да" if p.ok[k] else "нет"])
