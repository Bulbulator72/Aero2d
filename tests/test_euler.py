"""Проверки решателя Эйлера на грубой сетке (быстро, ~1–2 мин на CPU)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from aero2d.geometry import naca4  # noqa: E402
from aero2d.grid import ogrid  # noqa: E402
from aero2d.euler import EulerSolver, hllc_flux, prim_to_cons, cons_to_prim  # noqa: E402
from aero2d.implicit import ImplicitEuler  # noqa: E402
from aero2d.panel import PanelSolver  # noqa: E402


def test_hllc_consistency():
    """F(W, W) равен физическому потоку."""
    W = jnp.array([[1.2], [0.3], [-0.1], [0.9]])
    S = jnp.array([[0.6], [0.8]])
    F = hllc_flux(W, W, S)
    r, u, v, p = W[:, 0]
    q = u * 0.6 + v * 0.8
    E = p / 0.4 + 0.5 * r * (u * u + v * v)
    ref = jnp.array([r * q, r * u * q + p * 0.6, r * v * q + p * 0.8, (E + p) * q])
    assert np.allclose(F[:, 0], ref, rtol=1e-6)


def test_prim_cons_roundtrip():
    W = jnp.array([1.1, 0.4, -0.2, 0.7])
    assert np.allclose(cons_to_prim(prim_to_cons(W)), W, rtol=1e-6)


def test_freestream_preserved():
    """Однородный поток — точное стационарное решение (проверка метрик)."""
    g = ogrid(naca4("0012", 33), nj=25)
    s = EulerSolver(g, vortex_correction=False)
    w = s.freestream(jnp.float32(0.5), jnp.float32(3.0))
    U = s.init_state(w)
    R, _ = s.residual(U, w, 0.0)
    # вдали от тела невязка ~ машинный ноль (у стенки поток не однороден из-за непротекания)
    assert float(jnp.max(jnp.abs(R[:, :, 3:]) / s.vol[None, :, 3:])) < 1e-3


@pytest.fixture(scope="module")
def subsonic():
    g = ogrid(naca4("0012", 49), nj=41, ds0=3e-3)
    s = ImplicitEuler(EulerSolver(g))
    r = s.solve([0.5, 0.5], [0.0, 2.0], n_iter=40, verbose=False, tol=1e-5)
    return r


def test_symmetric_zero_lift(subsonic):
    assert abs(subsonic.cl[0]) < 2e-3


def test_subsonic_vs_panel(subsonic):
    """Дозвук: Cl близок к панелям с поправкой Прандтля–Глауэрта (на грубой
    сетке — в пределах нескольких процентов), сопротивление мало."""
    cl_pg = PanelSolver(naca4("0012", 121)).solve(2.0).cl / np.sqrt(1 - 0.5**2)
    assert abs(subsonic.cl[1] / cl_pg - 1) < 0.04
    assert abs(subsonic.cd[1]) < 3e-3


def test_converged(subsonic):
    drop = np.log10(subsonic.res_hist[-1] / subsonic.res_hist[0])
    assert np.all(drop < -3)
