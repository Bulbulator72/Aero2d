"""Многоэлементный профиль (предкрылок, основной элемент, закрылки):
невязкий панельный метод в той же формулировке, что и PanelSolver (функция
тока, линейная завихренность, условие Кутта и экстраполяция на острой кромке
каждого элемента). Каждому элементу — своя постоянная функции тока.

Это первый этап расчёта механизации: распределение давления, нагрузка на
элементы, влияние щелей и углов отклонения. Вязкие эффекты (пограничные слои,
слияние следов с пограничным слоем следующего элемента, отрыв на закрылке)
здесь не учитываются — Cl получается завышенным, особенно на больших углах
отклонения закрылка.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Airfoil, repanel
from .panel import psi_influence, panel_geometry, karman_tsien, close_trailing_edge


@dataclass
class MultiResult:
    alpha: float
    mach: float
    cl: float
    cd_p: float
    cm: float
    elements: list = field(default_factory=list)   # по элементам: x, y, cp, ue, cl, cd, cm


class MultiPanelSolver:
    """elements — список контуров (обход как у профиля: задняя кромка → верх →
    передняя кромка → низ → задняя кромка). chord — опорная хорда для
    коэффициентов, xref — точка момента (по умолчанию 1/4 опорной хорды от
    самой передней точки)."""

    def __init__(self, elements, chord: float | None = None, xref=None):
        els = []
        for e in elements:
            x, y = np.asarray(e.x, float), np.asarray(e.y, float)
            if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-9:
                e = close_trailing_edge(e)
                x, y = np.asarray(e.x, float), np.asarray(e.y, float)
            area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
            if area < 0:
                x, y = x[::-1].copy(), y[::-1].copy()
            els.append((x, y))
        self.els = els
        xs = np.concatenate([e[0] for e in els])
        self.chord = chord if chord else float(xs.max() - xs.min())
        x0 = float(xs.min())
        self.xref = xref if xref is not None else (x0 + 0.25 * self.chord, 0.0)
        sizes = [len(x) for x, _ in els]                  # N_e + 1 узлов
        offs = np.concatenate([[0], np.cumsum(sizes)])
        E = len(els)
        nunk = offs[-1] + E
        A = np.zeros((nunk, nunk))
        bx = np.zeros(nunk)
        by = np.zeros(nunk)
        row = 0
        for e, (x, y) in enumerate(els):
            N = len(x) - 1
            for f, (xf, yf) in enumerate(els):
                A[row:row + N, offs[f]:offs[f + 1]] = psi_influence(xf, yf, x[:-1], y[:-1])
            A[row:row + N, offs[-1] + e] = -1.0
            bx[row:row + N] = -y[:-1]
            by[row:row + N] = x[:-1]
            row += N
            o = offs[e]
            A[row, o] = A[row, o + N] = 1.0                       # Кутта
            row += 1
            A[row, [o, o + 1, o + 2]] = [1.0, -2.0, 1.0]          # острая кромка
            A[row, [o + N, o + N - 1, o + N - 2]] += [-1.0, 2.0, -1.0]
            row += 1
        assert row == nunk
        self.cond = np.linalg.cond(A)
        lu = np.linalg.inv(A)
        self.offs = offs
        self.g_x = (lu @ bx)[:offs[-1]]
        self.g_y = (lu @ by)[:offs[-1]]

    def gamma(self, alpha_deg):
        a = np.radians(alpha_deg)
        g = np.cos(a) * self.g_x + np.sin(a) * self.g_y
        return [g[self.offs[e]:self.offs[e + 1]] for e in range(len(self.els))]

    def solve(self, alpha_deg: float, mach: float = 0.0) -> MultiResult:
        a = np.radians(alpha_deg)
        ca, sa = np.cos(a), np.sin(a)
        c = self.chord
        xr, yr = self.xref
        out, FX, FY, M = [], 0.0, 0.0, 0.0
        for (x, y), g in zip(self.els, self.gamma(alpha_deg)):
            cp0 = 1 - g**2
            cp = karman_tsien(cp0, mach)
            L, _, _, nx, ny = panel_geometry(x, y)
            ca_, cb_ = cp[:-1], cp[1:]
            cpm = 0.5 * (ca_ + cb_)
            fx = -np.sum(cpm * nx * L) / c
            fy = -np.sum(cpm * ny * L) / c
            ra = (x[:-1] - xr) * ny - (y[:-1] - yr) * nx
            rb = (x[1:] - xr) * ny - (y[1:] - yr) * nx
            m = np.sum(L * (ca_ * ra / 3 + (ca_ * rb + cb_ * ra) / 6 + cb_ * rb / 3)) / c**2
            FX += fx
            FY += fy
            M += m
            out.append(dict(x=x, y=y, cp=cp, ue=g, cl=fy * ca - fx * sa, cd=fx * ca + fy * sa, cm=m))
        return MultiResult(alpha_deg, mach, FY * ca - FX * sa, FX * ca + FY * sa, M, out)


# ---------------------------------------------------------------------------
# геометрия механизации
# ---------------------------------------------------------------------------
def split_flap(af: Airfoil, x_cut: float = 0.75, deflection: float = 20.0, gap: float = 0.015,
               overlap: float = 0.0, n_side: int = 80):
    """Простой однощелевой закрылок из одного профиля (для оценок и примеров).

    Основная часть — исходный профиль до x_cut; нижняя поверхность на участке
    [x_cut − 0.12, x_cut] плавно поднимается к верхней (ниша под закрылок),
    кромка острая. Закрылок — участок профиля от x_cut − 0.03 до задней кромки
    со скруглённым носком; он отклоняется вниз на deflection градусов вокруг
    носка и ставится под кромку основной части: перекрытие overlap (доли
    хорды, > 0 — носок закрылка впереди кромки) и зазор gap — наименьшее
    расстояние от кромки основной части до закрылка.
    Возвращает [основная часть, закрылок]."""
    from matplotlib.path import Path
    x, y = np.asarray(af.x, float), np.asarray(af.y, float)
    ile = int(np.argmin(x))
    xu, yu = x[:ile + 1][::-1], y[:ile + 1][::-1]
    xl, yl = x[ile:], y[ile:]
    fu = lambda t: np.interp(t, xu, yu)
    fl = lambda t: np.interp(t, xl, yl)
    cove = 0.12
    # --- основная часть
    t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 400))) * x_cut
    up = fu(t)
    w = np.clip((t - (x_cut - cove)) / cove, 0, 1)
    w = 3 * w**2 - 2 * w**3
    lo = fl(t) * (1 - w) + (up - 0.002 * (1 - w)) * w
    lo[-1] = up[-1]
    main = Airfoil(np.concatenate([t[::-1], t[1:]]), np.concatenate([up[::-1], lo[1:]]),
                   af.name + " — основная часть")
    te = np.array([t[-1], up[-1]])
    # --- закрылок
    xf0 = x_cut - 0.03
    tf = xf0 + 0.5 * (1 - np.cos(np.linspace(0, np.pi, 300))) * (1 - xf0)
    yc = 0.5 * (fu(tf) + fl(tf))
    th = np.maximum(fu(tf) - fl(tf), 0)
    rr = np.clip((tf - xf0) / 0.05, 0, 1)
    th = th * np.sqrt(np.clip(1 - (1 - rr) ** 2, 0, 1))
    fx = np.concatenate([tf[::-1], tf[1:]])
    fy = np.concatenate([(yc + 0.5 * th)[::-1], (yc - 0.5 * th)[1:]])
    d = np.radians(deflection)
    px, py = xf0, yc[0]
    rx, ry = fx - px, fy - py
    fx, fy = px + rx * np.cos(d) + ry * np.sin(d), py - rx * np.sin(d) + ry * np.cos(d)
    # носок закрылка — на overlap впереди кромки основной части
    fx = fx + (te[0] - overlap) - fx.min()
    main_path = Path(np.column_stack([main.x, main.y]))
    dy = te[1] - fy[np.argmin(fx)] - gap
    for _ in range(60):
        gy = fy + dy
        dist = np.min(np.hypot(fx - te[0], gy - te[1]))
        hit = main_path.contains_points(np.column_stack([fx, gy])).any()
        if hit:
            dy -= 0.002
            continue
        if abs(dist - gap) < 2e-4:
            break
        dy += 0.6 * (dist - gap) if dist > gap else -0.6 * (gap - dist)
    flap = Airfoil(fx, fy + dy, af.name + f" — закрылок {deflection:g}°")
    return [repanel(main, n_side, keep_frame=True),
            repanel(flap, max(n_side // 2, 40), keep_frame=True)]


def load_multi(path: str):
    """Файл с несколькими элементами: блоки координат, разделённые пустой
    строкой, строкой с текстом или строкой «999 999» (формат MSES)."""
    blocks, cur = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            parts = ln.replace(",", " ").split()
            vals = None
            if len(parts) >= 2:
                try:
                    vals = (float(parts[0]), float(parts[1]))
                except ValueError:
                    vals = None
            if vals is None or abs(vals[0]) >= 998:
                if len(cur) >= 10:
                    blocks.append(np.array(cur))
                cur = []
                continue
            cur.append(vals)
    if len(cur) >= 10:
        blocks.append(np.array(cur))
    if not blocks:
        raise ValueError("В файле не найдено ни одного контура")
    return [Airfoil(b[:, 0], b[:, 1], f"элемент {i + 1}") for i, b in enumerate(blocks)]
