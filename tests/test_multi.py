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


def test_highlift_types_valid_and_converged():
    from aero2d.multi import highlift, _intersects
    af = naca4("4415", 201)
    cfgs = [dict(flap=dict(type="slotted", chord=0.25, deflection=30, gap=0.015, overlap=0.02)),
            dict(flap=dict(type="fowler", chord=0.30, deflection=30, gap=0.02, overlap=0.0)),
            dict(flap=dict(type="double", chord=0.35, deflection=45, gap=0.012, overlap=0.0,
                           vane_chord=0.12, vane_deflection=20, vane_gap=0.015, vane_overlap=0.0)),
            dict(slat=dict(chord=0.15, deflection=25, gap=0.02, overlap=0.01),
                 flap=dict(type="fowler", chord=0.30, deflection=30, gap=0.02, overlap=0.0))]
    base = PanelSolver(af).solve(8.0).cl
    for c in cfgs:
        els = highlift(af, n_side=70, **c)
        for i in range(len(els)):
            for j in range(i + 1, len(els)):
                assert not _intersects(els[i].x, els[i].y, els[j].x, els[j].y)
        r1 = MultiPanelSolver(els, chord=1.0).solve(8.0).cl
        r2 = MultiPanelSolver(highlift(af, n_side=120, **c), chord=1.0).solve(8.0).cl
        assert abs(r1 / r2 - 1) < 5e-3
        assert r1 > base + 0.5
