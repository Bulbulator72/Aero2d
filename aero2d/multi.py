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
def _surfaces(af: Airfoil):
    x, y = np.asarray(af.x, float), np.asarray(af.y, float)
    ile = int(np.argmin(x))
    xu, yu = x[:ile + 1][::-1], y[:ile + 1][::-1]
    xl, yl = x[ile:], y[ile:]
    return (lambda t: np.interp(t, xu, yu)), (lambda t: np.interp(t, xl, yl))


def _cos_pts(a, b, n=300):
    return a + 0.5 * (1 - np.cos(np.linspace(0, np.pi, n))) * (b - a)


def _smooth(w):
    w = np.clip(w, 0, 1)
    return 3 * w**2 - 2 * w**3


def _contour(t, up, lo):
    """Замкнутый контур: задняя кромка → верх → нос → низ → задняя кромка."""
    return np.concatenate([t[::-1], t[1:]]), np.concatenate([up[::-1], lo[1:]])


def _segment(fu, fl, x0, x1, round_le=True, te_taper=0.0, close_te=True):
    """Элемент из участка профиля [x0, x1] (средняя линия + толщина): скруглённый
    нос (если x0 > 0) и острая кромка; te_taper > 0 — толщина плавно сходит
    к нулю на последней доле te_taper длины (для дефлектора)."""
    t = _cos_pts(x0, x1)
    yc = 0.5 * (fu(t) + fl(t))
    th = np.maximum(fu(t) - fl(t), 0)
    if round_le and x0 > 1e-6:
        r = min(0.05, 0.25 * (x1 - x0))
        rr = np.clip((t - x0) / r, 0, 1)
        th = th * np.sqrt(np.clip(1 - (1 - rr) ** 2, 0, 1))
    if te_taper > 0:
        L = te_taper * (x1 - x0)
        th = th * (1 - _smooth((t - (x1 - L)) / L))
    if close_te:
        th[-1] = 0.0
    return _contour(t, yc + 0.5 * th, yc - 0.5 * th)


def _cove_fairing(t, up, lo, xa, xb, wedge=8.0):
    """Нижняя поверхность основной части над нишей закрылка: кубика Эрмита от
    начала ниши (касательно к низу профиля) к кромке козырька (угол клина
    кромки — wedge градусов). Реальная ниша занята застойной зоной с примерно
    такой границей; тонкий «язык» над нишей (верх минус 0,2 %) делает
    вязкий расчёт плохо обусловленным."""
    lo = lo.copy()
    m = t >= xa
    if m.sum() < 3:
        return lo
    ia = int(np.argmax(m))
    h = 1e-3
    s_lo = (np.interp(xa + h, t, lo) - np.interp(xa - h, t, lo)) / (2 * h)
    s_up = (np.interp(xb, t, up) - np.interp(xb - 2 * h, t, up)) / (2 * h)
    s_te = np.tan(np.arctan(s_up) + np.radians(wedge))
    p0, p1 = np.array([xa, lo[ia]]), np.array([xb, np.interp(xb, t, up)])
    L = xb - xa
    u = (t[m] - xa) / L
    h00, h10, h01, h11 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u, -2*u**3 + 3*u**2, u**3 - u**2
    lo[m] = h00 * p0[1] + h10 * L * s_lo + h01 * p1[1] + h11 * L * s_te
    lo[m] = np.minimum(lo[m], up[m] - 0.004 * (1 - u) ** 0.5)
    lo[-1] = up[-1]
    return lo


def _rotate(x, y, px, py, deg):
    """Поворот по часовой стрелке на deg градусов (положительный угол — задняя
    часть элемента вниз, если точка поворота спереди)."""
    d = np.radians(deg)
    rx, ry = x - px, y - py
    return px + rx * np.cos(d) + ry * np.sin(d), py - rx * np.sin(d) + ry * np.cos(d)


def _intersects(xa, ya, xb, yb):
    from matplotlib.path import Path
    pa, pb = Path(np.column_stack([xa, ya])), Path(np.column_stack([xb, yb]))
    return pb.contains_points(np.column_stack([xa, ya])).any() or \
        pa.contains_points(np.column_stack([xb, yb])).any()


def _place_behind(ex, ey, bodies, te, gap, overlap):
    """Элемент (закрылок, дефлектор) за задней кромкой te: носок на overlap
    впереди кромки, затем опускаем, пока не будет зазора gap и пересечений."""
    ex = ex + (te[0] - overlap) - ex.min()
    ey = ey + te[1] - ey[np.argmin(ex)] + 0.02
    best = None
    for dy in np.arange(0.0, -0.4, -0.0005):
        gy = ey + dy
        if any(_intersects(ex, gy, bx, by) for bx, by in bodies):
            continue
        dist = np.min(np.hypot(ex - te[0], gy - te[1]))
        if dist >= gap:
            best = dy
            break
    return ex, ey + (best if best is not None else -0.05)


def _place_ahead(ex, ey, main, gap, overlap):
    """Предкрылок: его задняя кромка — на overlap позади носка основной части,
    поднимаем, пока не будет зазора gap от кромки предкрылка до основной части."""
    mx, my = main
    ite = int(np.argmax(ex))                     # задняя кромка предкрылка
    ex = ex + (mx.min() + overlap) - ex[ite]
    xte = ex[ite]
    near = np.abs(mx - xte) < 0.02
    ytop = my[near].max() if near.any() else my.max()
    ey = ey + (ytop - ey[ite])
    for dy in np.arange(0.0, 0.4, 0.0005):
        gy = ey + dy
        if _intersects(ex, gy, mx, my):
            continue
        if np.min(np.hypot(mx - xte, my - gy[ite])) >= gap:
            return ex, gy
    return ex, ey + 0.05


FLAP_TYPES = {
    "нет": None,
    "однощелевой": "slotted",
    "Фаулера": "fowler",
    "двухщелевой": "double",
    "руль/элерон (поворотный)": "control",
}


def highlift(af: Airfoil, flap: dict | None = None, slat: dict | None = None, n_side: int = 80):
    """Механизация из одного профиля (хорда 1). Все размеры — доли хорды,
    углы — градусы (закрылок/дефлектор — вниз, предкрылок — носком вниз).

    flap = dict(type="slotted"|"fowler"|"double", chord=0.30, deflection=30,
                gap=0.015, overlap=0.0,
                # только для двухщелевого — дефлектор (передний элемент):
                vane_chord=0.12, vane_deflection=15, vane_gap=0.015, vane_overlap=0.0)
      slotted — щель у носка закрылка, ниша под ним короткая;
      fowler  — закрылок убран в глубокую нишу на всю свою хорду и выдвигается
                назад (overlap обычно ≈ 0…0,02 — увеличение хорды);
      double  — дефлектор + основной закрылок, каждый со своими зазором,
                перекрытием (относительно кромки впереди стоящего элемента) и углом.
    slat = dict(chord=0.15, deflection=20, gap=0.02, overlap=0.01)
    Возвращает список элементов (спереди назад)."""
    fu, fl = _surfaces(af)
    names = []
    # --- основная часть
    x_le_main = 0.0
    slat_el = None
    if slat:
        cs = slat.get("chord", 0.15)
        x_le_main = 0.35 * cs
    x_cut = 1.0 - (flap or {}).get("chord", 0.0) if flap else 1.0
    if flap:
        ftype = flap.get("type", "slotted")
        cf = flap["chord"]
        # Фаулер: верхняя обшивка (козырёк) нависает над убранным закрылком почти
        # до задней кромки (как у GA(W)-1: козырёк до 0,96 при закрылке 0,29),
        # ниша закрыта обтекателем от носка убранного закрылка до кромки
        # козырька; однощелевой/двухщелевой — короткий козырёк, обтекатель
        # ниши длиной 0,85 хорды закрылка перед ним
        x_lip = x_cut + 0.85 * cf if ftype == "fowler" else x_cut
        xa = x_cut if ftype == "fowler" else x_cut - 0.85 * cf
        t = _cos_pts(x_le_main, x_lip, 400)
        if slat:
            mx, my = _segment(fu, fl, x_le_main, x_lip, round_le=True, close_te=False)
            up = np.interp(t, *_upper_of(mx, my))
            lo = np.interp(t, *_lower_of(mx, my))
        else:
            up, lo = fu(t), fl(t)
        lo = _cove_fairing(t, up, lo, xa, x_lip)
        mx, my = _contour(t, up, lo)
    else:
        mx, my = _segment(fu, fl, x_le_main, 1.0, round_le=bool(slat))
    els = [Airfoil(mx, my, af.name + " — основная часть")]
    bodies = [(mx, my)]
    te = np.array([mx[0], my[0]])
    # --- закрылки
    if flap:
        ftype = flap.get("type", "slotted")
        if ftype == "double":
            vc = flap.get("vane_chord", 0.4 * flap["chord"])
            vx, vy = _segment(fu, fl, x_cut - 0.03, x_cut + vc, te_taper=0.6)
            vx, vy = _rotate(vx, vy, vx.min(), vy[np.argmin(vx)], flap.get("vane_deflection", 15.0))
            vx, vy = _place_behind(vx, vy, bodies, te, flap.get("vane_gap", 0.015),
                                   flap.get("vane_overlap", 0.0))
            els.append(Airfoil(vx, vy, af.name + f" — дефлектор {flap.get('vane_deflection', 15):g}°"))
            bodies.append((vx, vy))
            te = np.array([vx[0], vy[0]])
            fx, fy = _segment(fu, fl, x_cut + vc - 0.02, 1.0)
        else:
            fx, fy = _segment(fu, fl, x_cut - 0.03, 1.0)
        fx, fy = _rotate(fx, fy, fx.min(), fy[np.argmin(fx)], flap.get("deflection", 20.0))
        fx, fy = _place_behind(fx, fy, bodies, te, flap.get("gap", 0.015), flap.get("overlap", 0.0))
        nm = {"slotted": "закрылок", "fowler": "закрылок Фаулера", "double": "закрылок"}[ftype]
        els.append(Airfoil(fx, fy, af.name + f" — {nm} {flap.get('deflection', 20):g}°"))
    # --- предкрылок
    if slat:
        cs = slat.get("chord", 0.15)
        t = _cos_pts(0.0, cs, 300)
        up, lo = fu(t), fl(t)
        w = _smooth((t - 0.3 * cs) / (0.7 * cs))
        lo = lo * (1 - w) + (up - 0.002 * (1 - w)) * w
        lo[-1] = up[-1]
        sx, sy = _contour(t, up, lo)
        sx, sy = _rotate(sx, sy, sx.max(), sy[0], -slat.get("deflection", 20.0))
        sx, sy = _place_ahead(sx, sy, (mx, my), slat.get("gap", 0.02), slat.get("overlap", 0.01))
        els.insert(0, Airfoil(sx, sy, af.name + f" — предкрылок {slat.get('deflection', 20):g}°"))
    out = []
    for e in els:
        nn = n_side if "основная" in e.name else max(n_side // 2, 40)
        r = repanel(e, nn, keep_frame=True)
        out.append(Airfoil(r.x, r.y, e.name))
    return out


def _upper_of(x, y):
    i = int(np.argmin(x))
    return x[:i + 1][::-1], y[:i + 1][::-1]


def _lower_of(x, y):
    i = int(np.argmin(x))
    return x[i:], y[i:]


def split_flap(af: Airfoil, x_cut: float = 0.75, deflection: float = 20.0, gap: float = 0.015,
               overlap: float = 0.0, n_side: int = 80):
    """Однощелевой закрылок (совместимость): закрылок хордой 1 − x_cut."""
    return highlift(af, flap=dict(type="slotted", chord=1 - x_cut, deflection=deflection,
                                  gap=gap, overlap=overlap), n_side=n_side)


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
