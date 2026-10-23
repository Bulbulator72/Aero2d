"""Рули и элероны (органы управления оперения): отклонение задней части
профиля вокруг оси шарнира одним контуром (щель герметична) и шарнирный
момент с учётом осевой компенсации.

Геометрия — как команда FLAP в XFOIL: часть контура за осью поворачивается;
на сжатой стороне контуры обрезаются по точке пересечения, на растянутой —
соединяются дугой окружности с центром на оси.

Шарнирный момент (на единицу размаха, положителен, если стремится отклонить
руль задней кромкой вниз — в сторону положительного δ):
    Ch = M_ш / (q∞ · c_р²),   c_р — хорда руля от оси до задней кромки.
Он равен интегралу давления по обшивке руля за осью плюс вклад осевой
компенсации: носок руля длиной b = k·c_р впереди оси находится в нише
неподвижной части; при герметичной щели у носка его верхняя сторона
нагружена давлением у верхней кромки ниши, нижняя — у нижней. Отсюда
    ΔCh = (Cp_н − Cp_в) · b² cosδ / (2 c_р²),
где Cp_в, Cp_н — давление на верхней и нижней обшивке у оси. Это обычная
инженерная модель компенсации (равномерное давление в нише).
"""
from __future__ import annotations

import numpy as np

from .geometry import Airfoil


def _split(af: Airfoil):
    x, y = np.asarray(af.x, float), np.asarray(af.y, float)
    i = int(np.argmin(x))
    return (x[:i + 1][::-1], y[:i + 1][::-1]), (x[i:], y[i:])      # верх, низ: от носка к кромке


def hinge_point(af: Airfoil, x_hinge: float, y_rel: float = 0.5):
    """Ось шарнира на хорде x_hinge; по высоте — доля y_rel между нижней
    (0) и верхней (1) обшивкой (0,5 — середина толщины)."""
    (xu, yu), (xl, yl) = _split(af)
    yu_h, yl_h = np.interp(x_hinge, xu, yu), np.interp(x_hinge, xl, yl)
    return np.array([x_hinge, yl_h + y_rel * (yu_h - yl_h)])


def _rot(x, y, h, deg):
    """Поворот по часовой стрелке (δ > 0 — задняя кромка вниз)."""
    d = np.radians(deg)
    rx, ry = x - h[0], y - h[1]
    return h[0] + rx * np.cos(d) + ry * np.sin(d), h[1] - rx * np.sin(d) + ry * np.cos(d)


def _seg_intersection(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-14:
        return None
    w = q1 - p1
    t = (w[0] * d2[1] - w[1] * d2[0]) / den
    u = (w[0] * d1[1] - w[1] * d1[0]) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t, u
    return None


def _join_side(xs, ys, h, deg, upper):
    """Одна сторона (от носка к кромке): неподвижная часть до оси + повёрнутая."""
    k = np.searchsorted(xs, h[0])
    fx, fy = xs[:k], ys[:k]
    rx, ry = _rot(xs[k:], ys[k:], h, deg)
    # точка обшивки на оси
    y0 = np.interp(h[0], xs, ys)
    fx, fy = np.append(fx, h[0]), np.append(fy, y0)
    rx0, ry0 = _rot(np.array([h[0]]), np.array([y0]), h, deg)
    rx, ry = np.concatenate([rx0, rx]), np.concatenate([ry0, ry])
    # сжатая сторона: ищем пересечение неподвижной и повёрнутой обшивок
    P = np.column_stack([fx, fy])
    Q = np.column_stack([rx, ry])
    for i in range(len(P) - 2, max(len(P) - 60, -1), -1):
        for j in range(0, min(len(Q) - 1, 60)):
            r = _seg_intersection(P[i], P[i + 1], Q[j], Q[j + 1])
            if r is not None and (i < len(P) - 2 or j > 0):
                pt = P[i] + r[0] * (P[i + 1] - P[i])
                return np.concatenate([fx[:i + 1], [pt[0]], rx[j + 1:]]), \
                    np.concatenate([fy[:i + 1], [pt[1]], ry[j + 1:]])
    # растянутая сторона: дуга с центром на оси
    a0 = np.arctan2(fy[-1] - h[1], fx[-1] - h[0])
    a1 = np.arctan2(ry[0] - h[1], rx[0] - h[0])
    r = np.hypot(fx[-1] - h[0], fy[-1] - h[1])
    n = max(int(abs(a1 - a0) / np.radians(3)), 2)
    aa = np.linspace(a0, a1, n + 1)[1:-1]
    return np.concatenate([fx, h[0] + r * np.cos(aa), rx]), np.concatenate([fy, h[1] + r * np.sin(aa), ry])


def deflect(af: Airfoil, x_hinge: float, deflection: float, y_rel: float = 0.5) -> Airfoil:
    """Профиль с отклонённым рулём (одним контуром)."""
    h = hinge_point(af, x_hinge, y_rel)
    (xu, yu), (xl, yl) = _split(af)
    ux, uy = _join_side(xu, yu, h, deflection, True)
    lx, ly = _join_side(xl, yl, h, deflection, False)
    x = np.concatenate([ux[::-1], lx[1:]])
    y = np.concatenate([uy[::-1], ly[1:]])
    x[0] = x[-1] = 0.5 * (x[0] + x[-1])
    y[0] = y[-1] = 0.5 * (y[0] + y[-1])
    return Airfoil(x, y, af.name + f", руль {deflection:g}°")


def hinge_moment(x, y, cp, hinge, x_hinge, deflection, balance=0.0):
    """Ch по распределению давления в узлах контура (обход как у профиля).
    hinge — точка оси, x_hinge — положение оси по хорде (для хорды руля
    c_р = 1 − x_hinge), balance — осевая компенсация b/c_р."""
    x, y, cp = (np.asarray(v, float) for v in (x, y, cp))
    cr = 1.0 - x_hinge
    xm, ym = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
    dx, dy = np.diff(x), np.diff(y)
    cpm = 0.5 * (cp[:-1] + cp[1:])
    # внешняя нормаль при обходе против часовой стрелки: (dy, −dx)
    fx, fy = -cpm * dy, cpm * dx
    # панели руля: за осью (в системе повёрнутого руля)
    d = np.radians(deflection)
    rx, ry = xm - hinge[0], ym - hinge[1]
    along = rx * np.cos(d) - ry * np.sin(d)          # координата вдоль хорды руля
    on = along > 1e-9
    mz = np.sum((rx * fy - ry * fx)[on])             # против часовой стрелки
    ch = -mz / cr**2                                  # + — задняя кромка вниз
    if balance > 0:
        i = int(np.argmin(x))
        xu, yu, cu = x[:i + 1][::-1], y[:i + 1][::-1], cp[:i + 1][::-1]
        xl, yl, cl_ = x[i:], y[i:], cp[i:]
        xe = hinge[0] - 0.02 * cr                     # давление у кромок ниши (перед осью)
        cpu, cpl = np.interp(xe, xu, cu), np.interp(xe, xl, cl_)
        ch += (cpl - cpu) * (balance * cr) ** 2 * np.cos(d) / (2 * cr**2)
    return float(ch)


# ---------------------------------------------------------------------------
# Руль с открытой щелью: два элемента (неподвижная часть с нишей + руль)
#
# ЭКСПЕРИМЕНТАЛЬНО, в окно программы не выведено. В невязкой постановке
# результат определяется тем, на какой губе ниши ставится условие Кутта, а
# это решает вязкое течение в щели: при компенсации 0,20c_р переключение губы
# меняет Cl на ±0,06 уже при α = δ = 0, а производные зависят от зазора не
# так, как в опыте (NACA ARR L-301: зазор слегка уменьшает cl_α и τ и почти
# не меняет ch). Для открытой щели нужен вязкий многоэлементный расчёт.
# ---------------------------------------------------------------------------
def _skin(af: Airfoil):
    (xu, yu), (xl, yl) = _split(af)
    return (lambda t: np.interp(t, xu, yu)), (lambda t: np.interp(t, xl, yl))


def gapped(af: Airfoil, x_hinge: float, deflection: float, balance: float = 0.0, gap: float = 0.005,
           lip: str = "upper", delta_max: float | None = None, n: int = 90):
    """Руль с открытой щелью: [неподвижная часть, руль].

    Носок руля — полуэллипс с центром на оси: вперёд на b = balance·c_р (но не
    меньше местной полутолщины), по высоте — до обшивки. Ниша неподвижной части
    повторяет огибающую носка при отклонениях ±delta_max, отстоящую на gap;
    кромки ниши («губы») — там, где ниша выходит на обшивку. Условие Кутта для
    неподвижной части ставится на той губе, через которую выходит поток из щели:
    lip="upper" (нагрузка вверх, руль вниз) или "lower"."""
    fu, fl = _skin(af)
    h = hinge_point(af, x_hinge)
    cr = 1.0 - x_hinge
    if delta_max is None:                    # ниша — под фактический ход руля
        delta_max = max(abs(deflection) + 5.0, 15.0)
    ru, rl = fu(x_hinge) - h[1], h[1] - fl(x_hinge)
    b = max(balance * cr, 0.5 * (ru + rl))
    # --- носок руля (в собственной системе, до поворота): от верха через носик к низу
    ph = np.linspace(0.5 * np.pi, 1.5 * np.pi, 61)
    nx = h[0] + b * np.cos(ph)
    ny = h[1] + np.where(np.sin(ph) > 0, ru, rl) * np.sin(ph)
    # --- огибающая носка при повороте ±delta_max (в полярных координатах вокруг оси)
    angs = np.linspace(0.40 * np.pi, 1.60 * np.pi, 361)
    renv = np.zeros_like(angs)
    for th in np.radians(np.linspace(-delta_max, delta_max, 25)):
        rx, ry = _rot(nx, ny, h, np.degrees(th))
        a = np.mod(np.arctan2(ry - h[1], rx - h[0]), 2 * np.pi)
        r = np.hypot(rx - h[0], ry - h[1])
        o = np.argsort(a)
        renv = np.maximum(renv, np.interp(angs, a[o], r[o], left=0, right=0))
    rc = renv + gap
    # --- губы: где ниша выходит на обшивку
    def lip_x(skin, upper):
        xs = np.linspace(x_hinge - 3 * b - 0.05, x_hinge, 2000)
        ys = skin(xs)
        a = np.mod(np.arctan2(ys - h[1], xs - h[0]), 2 * np.pi)
        d = np.hypot(xs - h[0], ys - h[1])
        f = d - np.interp(a, angs, rc)
        k = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
        return xs[k[0]] if len(k) else x_hinge - b - gap
    xlu, xll = lip_x(fu, True), lip_x(fl, False)
    alu = np.mod(np.arctan2(fu(xlu) - h[1], xlu - h[0]), 2 * np.pi)
    all_ = np.mod(np.arctan2(fl(xll) - h[1], xll - h[0]), 2 * np.pi)
    # --- неподвижная часть: губа (верх) → верх → носок профиля → низ → губа (низ) → ниша → губа (верх)
    t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    tu, tl = t * xlu, t * xll
    up_x, up_y = tu[::-1], fu(tu[::-1])
    lo_x, lo_y = tl[1:], fl(tl[1:])
    ca = np.linspace(all_, alu, 50)[1:-1]                  # от нижней губы через перёд к верхней
    cr_ = np.interp(ca, angs, rc)
    cx, cy = h[0] + cr_ * np.cos(ca), h[1] + cr_ * np.sin(ca)
    fx = np.concatenate([up_x, lo_x, cx, [up_x[0]]])
    fy = np.concatenate([up_y, lo_y, cy, [up_y[0]]])
    if lip == "lower":                                      # кромка Кутта — на нижней губе
        k = len(up_x) + len(lo_x) - 1
        fx = np.concatenate([fx[k:-1], fx[:k + 1]])
        fy = np.concatenate([fy[k:-1], fy[:k + 1]])
    fixed = Airfoil(fx, fy, af.name + " — неподвижная часть")
    # --- руль: кромка → верх → носок → низ → кромка, затем поворот
    tr = x_hinge + 0.5 * (1 - np.cos(np.linspace(0, np.pi, n))) * cr
    rx = np.concatenate([tr[::-1], nx[1:-1], tr])
    ry = np.concatenate([fu(tr[::-1]), ny[1:-1], fl(tr)])
    ry[0] = ry[-1] = 0.5 * (ry[0] + ry[-1])
    rx, ry = _rot(rx, ry, h, deflection)
    rudder = Airfoil(rx, ry, af.name + f" — руль {deflection:g}°")
    return [fixed, rudder], h


def hinge_moment_element(x, y, cp, hinge, x_hinge):
    """Ch по полному контуру руля (включая носок впереди оси)."""
    x, y, cp = (np.asarray(v, float) for v in (x, y, cp))
    xm, ym = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
    dx, dy = np.diff(x), np.diff(y)
    cpm = 0.5 * (cp[:-1] + cp[1:])
    fx, fy = -cpm * dy, cpm * dx
    mz = np.sum((xm - hinge[0]) * fy - (ym - hinge[1]) * fx)
    return float(-mz / (1.0 - x_hinge) ** 2)


def solve_gapped(af: Airfoil, alpha: float, x_hinge: float, deflection: float, balance: float = 0.0,
                 gap: float = 0.005, tau: float = 0.6, width: float = 2.0, mach: float = 0.0):
    """Невязкий расчёт руля с открытой щелью. Поток через щель выходит через
    верхнюю губу при нагрузке вверх и через нижнюю — при нагрузке вниз;
    около нулевой нагрузки (|α + τδ| ≲ width) решения с условием Кутта на
    верхней и нижней губе смешиваются плавно, чтобы симметричный профиль при
    α = δ = 0 давал ноль. Возвращает (Cl, Cm, Ch, элементы верхнего варианта)."""
    from .multi import MultiPanelSolver
    load = alpha + tau * deflection
    w = 0.5 + 0.5 * np.tanh(load / width)
    out = []
    for lip in ("upper", "lower"):
        els, h = gapped(af, x_hinge, deflection, balance, gap, lip)
        r = MultiPanelSolver(els, chord=1.0, xref=(0.25, 0.0)).solve(alpha, mach)
        e = r.elements[1]
        out.append((r.cl, r.cm, hinge_moment_element(e["x"], e["y"], e["cp"], h, x_hinge), r))
    cl, cm, ch = (w * out[0][i] + (1 - w) * out[1][i] for i in range(3))
    return cl, cm, ch, out[0][3] if w >= 0.5 else out[1][3]
