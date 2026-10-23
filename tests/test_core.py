"""Сценарии интерфейса: подготовка профиля, поляры, экспорт."""
import numpy as np
import pytest

from aero2d import core
from aero2d.geometry import naca4, repanel, normalize, Airfoil
from aero2d.panel import PanelSolver


def test_alpha_range():
    assert np.allclose(core.alpha_range(-2, 2, 1), [-2, -1, 0, 1, 2])
    assert np.allclose(core.alpha_range(4, 0, 2), [0, 2, 4])
    with pytest.raises(ValueError):
        core.alpha_range(0, 1, 0)


def test_naca_input_validation():
    assert core.make_airfoil("naca", "NACA 0012").name == "NACA 0012"
    for bad in ("241", "24a2", "2400"):
        with pytest.raises(ValueError):
            core.make_airfoil("naca", bad)


def test_file_roundtrip(tmp_path):
    """Профиль из файла (мало точек, обратный обход, формат Lednicer) даёт
    тот же Cl, что и аналитический."""
    ref = PanelSolver(naca4("2412", 100)).solve(4).cl
    a = naca4("2412", 35)
    p1 = tmp_path / "selig.dat"
    p1.write_text("NACA 2412\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in zip(a.x, a.y)))
    p2 = tmp_path / "reversed.dat"
    p2.write_text("\n".join(f"{x:.6f} {y:.6f}" for x, y in zip(a.x[::-1], a.y[::-1])))
    n = 35
    up = zip(a.x[:n][::-1], a.y[:n][::-1])
    lo = zip(a.x[n - 1:], a.y[n - 1:])
    p3 = tmp_path / "lednicer.dat"
    p3.write_text("NACA 2412\n35. 35.\n\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in up)
                  + "\n\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in lo))
    for p in (p1, p2, p3):
        af = core.make_airfoil("file", str(p))
        assert abs(PanelSolver(af).solve(4).cl / ref - 1) < 2e-3, p.name


def test_normalize_scaled_rotated():
    a = naca4("0012", 60)
    t = np.radians(10)
    x = 2 * (a.x * np.cos(t) - a.y * np.sin(t)) + 5
    y = 2 * (a.x * np.sin(t) + a.y * np.cos(t)) - 1
    af = repanel(normalize(Airfoil(x, y)), 100)
    assert abs(PanelSolver(af).solve(0).cl) < 1e-3
    assert abs(af.x.min()) < 1e-3 and abs(af.x.max() - 1) < 1e-3


def test_run_panel_and_csv(tmp_path):
    p = core.run_panel(naca4("2412", 100), core.alpha_range(0, 8, 2), 0.5)
    assert np.all(np.diff(p.cl) > 0) and np.all(np.isnan(p.cd))
    assert not p.ok[-1]          # при α = 8°, M = 0.5 есть сверхзвуковая зона
    f = tmp_path / "p.csv"
    core.save_csv([p], f)
    assert len(f.read_text(encoding="utf-8-sig").splitlines()) == 6


def test_euler_low_mach_rejected():
    with pytest.raises(ValueError):
        core.run_euler(naca4("0012", 60), [0.0], 0.1)


def test_smooth_airfoil_removes_noise_keeps_shape():
    import numpy as np
    from aero2d.geometry import naca4, smooth_airfoil, Airfoil
    af = naca4("2412", 81)
    rng = np.random.default_rng(0)
    noisy = Airfoil(af.x, af.y + rng.normal(0, 3e-4, len(af.y)) * (af.x > 0.002))
    sm = smooth_airfoil(noisy)
    ref = smooth_airfoil(af)

    def upper(a):
        i = int(np.argmin(a.x))
        return np.interp(np.linspace(0.01, 0.95, 200), a.x[:i + 1][::-1], a.y[:i + 1][::-1])
    true = upper(af)
    assert np.max(np.abs(upper(ref) - true)) < 3e-4          # гладкий контур почти не меняется
    assert np.std(upper(sm) - true) < 0.6 * 3e-4             # шум уменьшен


def test_parallel_polar_matches_sequential():
    """Поляра, разбитая на куски по α и посчитанная в двух процессах, совпадает
    с последовательной."""
    import numpy as np
    from aero2d import core
    af = core.make_airfoil("naca", "0012")
    al = np.arange(0.0, 8.0, 1.0)
    p1 = core.run_viscous(af, al, 0.0, 1e6, workers=1)
    p2 = core.run_viscous(af, al, 0.0, 1e6, workers=2)
    assert p1.ok.all() and p2.ok.all()
    assert np.abs(p1.cl - p2.cl).max() < 1e-4 and np.abs(p1.cd - p2.cd).max() < 1e-5


def test_resolve_workers():
    from aero2d.parallel import resolve_workers, cpu_count, auto_workers
    assert resolve_workers("авто") == auto_workers() >= 1
    assert resolve_workers("1") == 1
    assert resolve_workers(10 ** 6) == cpu_count()
