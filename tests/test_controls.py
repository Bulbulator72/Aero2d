"""Рули: геометрия отклонения и шарнирный момент против теории тонкого профиля
(Глауэрт) и проверка модифицированных профилей NACA."""
import numpy as np

from aero2d.geometry import naca4, naca_sym, repanel
from aero2d.controls import deflect, hinge_point, hinge_moment
from aero2d.panel import PanelSolver


def glauert(alpha, delta, xh):
    a, d = np.radians(alpha), np.radians(delta)
    th_h = np.arccos(1 - 2 * xh)
    A0 = a + d * (np.pi - th_h) / np.pi
    th = np.linspace(1e-4, np.pi - 1e-6, 20001)
    dcp = 4 * A0 * (1 + np.cos(th)) / np.sin(th)
    A1 = 0.0
    for n in range(1, 400):
        An = 2 * d * np.sin(n * th_h) / (n * np.pi)
        A1 = An if n == 1 else A1
        dcp += 4 * An * np.sin(n * th)
    x = 0.5 * (1 - np.cos(th))
    m = x > xh
    return np.pi * (2 * A0 + A1), -np.trapezoid(dcp[m] * (x[m] - xh), x[m]) / (1 - xh) ** 2


def test_hinge_moment_thin_airfoil_limit():
    af = naca4("0003", 301)
    for xh in (0.7, 0.8):
        h = hinge_point(af, xh)
        for al, de in ((2.0, 0.0), (0.0, 5.0), (2.0, 5.0)):
            r = PanelSolver(repanel(deflect(af, xh, de), 200, keep_frame=True)).solve(al)
            cl, ch = glauert(al, de, xh)
            assert abs(r.cl / cl - 1) < 0.04
            assert abs(hinge_moment(r.x, r.y, r.cp, h, xh, de) / ch - 1) < 0.06


def test_balance_reduces_hinge_moment():
    af = naca4("0012", 201)
    h = hinge_point(af, 0.7)
    r = PanelSolver(repanel(deflect(af, 0.7, 10.0), 150, keep_frame=True)).solve(2.0)
    c0 = hinge_moment(r.x, r.y, r.cp, h, 0.7, 10.0, 0.0)
    c25 = hinge_moment(r.x, r.y, r.cp, h, 0.7, 10.0, 0.25)
    assert c0 < c25 < 0


def test_modified_naca_ordinates():
    """NACA 0012-64: табличные ординаты (Abbott & von Doenhoff)."""
    a = naca_sym("0012-64", 300)
    i = int(np.argmin(a.x))
    xu, yu = a.x[:i + 1][::-1], a.y[:i + 1][::-1]
    for x, y in ((0.025, 0.02453), (0.3, 0.05827), (0.5, 0.05827), (0.7, 0.0448)):
        assert abs(np.interp(x, xu, yu) - y) < 4e-4
