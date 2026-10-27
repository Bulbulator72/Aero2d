"""Вязко-невязкий метод: замыкающие соотношения, влияние источников, сходимость
и сравнение с опорными значениями XFOIL 6.99 (рассчитаны заранее; XFOIL для
тестов не нужен)."""
import numpy as np
import pytest

from aero2d import blclosure as bc
from aero2d.geometry import naca4
from aero2d.panel import source_influence, psi_influence, vortex_velocity_influence
from aero2d.viscous import ViscousSolver


def test_blasius_closures():
    """Блазиус: H = 2.59 → Cf·Reθ = 0.441 (корреляция Drela даёт 0.428 —
    погрешность самой аппроксимации, так же в XFOIL), 2CD/H*·Reθ ≈ 0.22, H* ≈ 1.573."""
    hk, rt = np.array([2.591]), np.array([1000.0])
    assert abs(bc.cfl(hk, rt)[0] * rt[0] - 0.441) < 0.02
    assert abs(bc.dil(hk, rt)[0] * rt[0] - 0.2206) < 0.003
    assert abs(bc.hsl(hk)[0] - 1.573) < 0.005


def test_turbulent_flat_plate_cf():
    """Турбулентная пластина, Reθ = 1e4, H = 1.4: Cf ≈ 0.0028 (около Людвига–Тиллмана)."""
    cf = bc.cft(np.array([1.4]), np.array([1e4]), 0.0)[0]
    lt = 0.246 * 10 ** (-0.678 * 1.4) * 1e4 ** -0.268
    assert abs(cf / lt - 1) < 0.15


def test_source_velocity_matches_stream_function():
    af = naca4("2412", 60)
    xp, yp = np.array([0.3, 1.5, -0.2]), np.array([0.3, -0.2, 0.05])
    h = 1e-6
    _, U, V = source_influence(af.x, af.y, xp, yp)
    dpy = (source_influence(af.x, af.y, xp, yp + h)[0] - source_influence(af.x, af.y, xp, yp - h)[0]) / (2 * h)
    dpx = (source_influence(af.x, af.y, xp + h, yp)[0] - source_influence(af.x, af.y, xp - h, yp)[0]) / (2 * h)
    assert np.allclose(dpy, U, atol=1e-7) and np.allclose(-dpx, V, atol=1e-7)
    Uv, Vv = vortex_velocity_influence(af.x, af.y, xp, yp)
    dpy = (psi_influence(af.x, af.y, xp, yp + h) - psi_influence(af.x, af.y, xp, yp - h)) / (2 * h)
    assert np.allclose(dpy, Uv, atol=1e-6)


@pytest.fixture(scope="module")
def n0012():
    return ViscousSolver(naca4("0012", 200), 1e6)


def test_symmetric_alpha0(n0012):
    """NACA 0012, Re = 1e6, α = 0: XFOIL 6.99 — Cd = 0.00533, x_tr = 0.687."""
    r = n0012.solve(0.0)
    assert r.converged
    assert abs(r.cl) < 1e-4
    assert abs(r.cd / 0.00533 - 1) < 0.04
    assert abs(r.xtr_u - 0.687) < 0.02 and abs(r.xtr_u - r.xtr_l) < 1e-6


def test_lifting_alpha4(n0012):
    """NACA 0012, Re = 1e6, α = 4: XFOIL — Cl = 0.417, Cd = 0.00725, x_tr = 0.258 / 0.957."""
    r = n0012.solve(4.0)
    assert r.converged
    assert abs(r.cl / 0.4173 - 1) < 0.02
    assert abs(r.cd / 0.00725 - 1) < 0.05
    assert abs(r.xtr_u - 0.258) < 0.02 and abs(r.xtr_l - 0.957) < 0.02


def test_forced_transition():
    """Принудительный переход у передней кромки: XFOIL — Cd = 0.01086, Cdf = 0.00908."""
    v = ViscousSolver(naca4("0012", 200), 1e6, xtr=(0.05, 0.05))
    r = v.solve(0.0)
    assert r.converged
    assert abs(r.cd / 0.01086 - 1) < 0.03
    assert abs(r.cdf / 0.00908 - 1) < 0.05


def test_continuation_polar():
    v = ViscousSolver(naca4("2412", 200), 3e6)
    pol = v.polar([0.0, 2.0, 4.0])
    assert all(p.converged for p in pol)
    cl = [p.cl for p in pol]
    assert cl[0] < cl[1] < cl[2]


def test_polar_point_equals_single_point():
    """Канонический путь продолжения: точка поляры совпадает с той же точкой,
    посчитанной отдельно (раньше у 4415 при α = 4° расхождение было 17 %)."""
    af = naca4("2412", 200)
    pol = ViscousSolver(af, 1e6).polar([-1.0, 1.0, 3.0])
    one = ViscousSolver(af, 1e6).polar([3.0])[0]
    assert pol[2].converged and one.converged
    assert abs(pol[2].cl - one.cl) < 1e-9 and abs(pol[2].cd - one.cd) < 1e-12
