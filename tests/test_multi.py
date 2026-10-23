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


def test_multi_viscous_single_matches_viscous():
    """Один элемент в многоэлементном вязком решателе = обычный вязкий расчёт."""
    import warnings
    warnings.filterwarnings("ignore")
    from aero2d.geometry import naca4
    from aero2d.viscous import ViscousSolver
    from aero2d.mviscous import MultiViscousSolver
    af = naca4("2412", 120)
    r1 = ViscousSolver(af, 3e6, n_panels=160).solve(4.0)
    r2 = MultiViscousSolver([af], 3e6, n_panels=160).solve(4.0)
    assert r1.converged and r2.converged
    assert abs(r1.cl - r2.cl) < 1e-4 and abs(r1.cd - r2.cd) < 1e-5


def test_unwrapped_source_psi_single_valued():
    """Функция тока источников с разрезом за концом следа однозначна вокруг тела."""
    from aero2d.geometry import naca4
    from aero2d.sources import lin_source, unwrapped_angles
    af = naca4("0012", 40)
    x, y = af.x, af.y
    wx, wy = np.linspace(1.0, 2, 10), np.zeros(10)
    rng = np.random.default_rng(1)
    qa, qw = rng.random(len(x)), rng.random(10) + 1
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, 4001)
    cx, cy = 0.5 + 0.8 * np.cos(t), 0.35 * np.sin(t)
    px, py = np.concatenate([x, wx[1:]]), np.concatenate([y, wy[1:]])
    phi = unwrapped_angles(px, py, cx, cy)
    Pa, Xa, Ya = lin_source(x, y, cx, cy, phi=phi[:, :len(x)])
    Pw, Xw, Yw = lin_source(wx, wy, cx, cy, wake_sheet=True, phi=phi[:, len(x) - 1:])
    p = Pa @ qa + Pw @ qw
    U, V = Ya @ qa + Yw @ qw, -(Xa @ qa + Xw @ qw)
    dp = np.sum(0.5 * (U[1:] + U[:-1]) * np.diff(cy) - 0.5 * (V[1:] + V[:-1]) * np.diff(cx))
    assert abs((p[-1] - p[0]) - dp) < 1e-6


def test_merge_residual_reduces_to_interval():
    """Отрезок со слиянием следа при нулевой доле следа совпадает с обычным."""
    from aero2d.geometry import naca4
    from aero2d.mviscous import MultiViscousSolver
    ms = MultiViscousSolver([naca4("0012", 60)], 1e6, n_panels=100)
    A = np.array([2e-4, 3e-4, 10.0, 0.05, 1.2, 0.30])
    B = np.array([2.2e-4, 3.4e-4, 10.5, 0.05, 1.15, 0.31])
    W = np.array([1e-3, 1.3e-3, 10.0, 0.04, 1.3, 0.5])
    V = np.concatenate([A, B, W])[:, None].astype(complex)
    ex = (np.array([0.3]), np.array([0.31]), np.array([False]), np.array([0.0]))
    r0 = ms._interval_res(np.concatenate([A, B])[:, None].astype(complex), *ex)
    r1 = ms._merge_res(V, *ex, np.array([0.0]))
    assert np.allclose(r0, r1)
    r2 = ms._merge_res(V, *ex, np.array([0.5]))
    assert not np.allclose(r0, r2)
