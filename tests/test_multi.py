"""Многоэлементный панельный метод."""
import numpy as np

from aero2d.geometry import naca4, Airfoil
from aero2d.panel import PanelSolver
from aero2d.multi import MultiPanelSolver, split_flap
from aero2d.field import compute_field


def test_single_element_matches_panel():
    af = naca4("2412", 121)
    s = PanelSolver(af).solve(4.0)
    m = MultiPanelSolver([af], chord=1.0, xref=(0.25, 0.0)).solve(4.0)
    assert abs(s.cl - m.cl) < 1e-10 and abs(s.cm - m.cm) < 1e-10


def test_far_elements_do_not_interact():
    af = naca4("0012", 101)
    far = Airfoil(af.x, af.y + 500.0)
    r = MultiPanelSolver([af, far], chord=1.0).solve(5.0)
    single = PanelSolver(af).solve(5.0).cl
    for e in r.elements:
        assert abs(e["cl"] / single - 1) < 2e-3


def test_flap_grid_convergence_and_trend():
    cls = [MultiPanelSolver(split_flap(naca4("2412", 201), 0.75, 30.0, 0.015, 0.0, n_side=n),
                            chord=1.0).solve(4.0).cl for n in (60, 120)]
    assert abs(cls[0] / cls[1] - 1) < 2e-3
    c0 = MultiPanelSolver(split_flap(naca4("2412", 201), 0.75, 0.0), chord=1.0).solve(4.0).cl
    c20 = MultiPanelSolver(split_flap(naca4("2412", 201), 0.75, 20.0), chord=1.0).solve(4.0).cl
    assert c20 > c0 + 1.0


def test_field_far_from_body_is_freestream():
    af = naca4("0012", 81)
    ctx = dict(method="panel", airfoil=af, mach=0.0, alpha=np.array([0.0]))
    f = compute_field(ctx, 0, box=(5.0, 6.0, 5.0, 6.0), n=10)
    assert np.allclose(f["speed"], 1.0, atol=0.02)
