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
