"""Верификация панельного метода по точным решениям Кармана–Треффца."""
import numpy as np
import pytest

from aero2d.geometry import KarmanTrefftz, Joukowski, naca4, cst
from aero2d.panel import PanelSolver, karman_tsien


CASES = [Joukowski(0.1, 0.0), Joukowski(0.1, 0.05),
         KarmanTrefftz(0.1, 0.05, 10), KarmanTrefftz(0.08, 0.04, 15)]


@pytest.mark.parametrize("g", CASES)
def test_cl_exact(g):
    s = PanelSolver(g.airfoil(161))
    for a in (-4.0, 0.0, 4.0, 8.0):
        assert abs(s.solve(a).cl - g.cl(a)) < 3e-4 * max(1, abs(g.cl(a)))


@pytest.mark.parametrize("g", CASES)
def test_second_order(g):
    """Ошибка Cl убывает примерно в 4 раза при удвоении числа панелей."""
    e = [abs(PanelSolver(g.airfoil(n)).solve(4.0).cl - g.cl(4.0)) for n in (81, 161, 321)]
    assert e[0] / e[1] > 3.2 and e[1] / e[2] > 3.2


def test_cp_exact():
    g = KarmanTrefftz(0.1, 0.05, 10)
    r = PanelSolver(g.airfoil(321)).solve(4.0)
    th = g.theta_of_point(r.x, r.y)
    m = (r.x > 0.002) & (r.x < 0.99)
    assert np.max(np.abs(r.cp - g.cp(th, 4.0))[m]) < 5e-3


def test_naca_reference():
    """Невязкие значения NACA 2412 (совпадают с XFOIL inviscid с точностью ~1%)."""
    s = PanelSolver(naca4("2412", 121))
    r = s.solve(0.0)
    assert abs(r.cl - 0.259) < 0.005
    assert abs(r.cm + 0.055) < 0.003


def test_dalembert():
    r = PanelSolver(naca4("4415", 161)).solve(6.0)
    assert abs(r.cd_p) < 1e-3


def test_karman_tsien_limits():
    assert karman_tsien(-0.5, 0.0) == -0.5
    # для малых Cp поправка близка к Прандтлю–Глауэрту
    assert abs(karman_tsien(-0.01, 0.5) / (-0.01 / np.sqrt(0.75)) - 1) < 0.01


def test_cst_symmetric():
    af = cst([0.17, 0.15, 0.15, 0.15], [-0.17, -0.15, -0.15, -0.15])
    assert abs(PanelSolver(af).solve(0.0).cl) < 1e-6
