"""RANS: сетка, блочный ILU, конвейер расчёта (короткий прогон)."""
import importlib.util

import numpy as np
import pytest

from aero2d.bilu import BlockILU0
from aero2d.geometry import naca4, repanel

gmsh_missing = importlib.util.find_spec("gmsh") is None


def test_block_ilu_exact_for_block_tridiagonal():
    # для блочно-трёхдиагональной матрицы ILU(0) = точное LU
    rng = np.random.default_rng(0)
    n, k = 30, 5
    rows, cols, blocks = [], [], []
    for i in range(n):
        for j in (i - 1, i, i + 1):
            if 0 <= j < n:
                rows.append(i); cols.append(j)
                b = rng.normal(size=(k, k)) * 0.3
                if i == j:
                    b += 4 * np.eye(k)
                blocks.append(b)
    A = np.zeros((n * k, n * k))
    for i, j, b in zip(rows, cols, blocks):
        A[i * k:(i + 1) * k, j * k:(j + 1) * k] = b
    ilu = BlockILU0(n, rows, cols, k=k, perm=np.arange(n)).factor(blocks)
    x = rng.normal(size=n * k)
    assert np.allclose(ilu.solve(A @ x), x, atol=1e-10)


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_mesh_cells_closed():
    from aero2d.rans_mesh import build_mesh
    m = build_mesh([repanel(naca4("0012", 120), 60)], re=1e6, far_radius=20)
    # сумма n·S по граням каждой ячейки = 0
    acc = np.zeros((m.n_cells, 2))
    l, r = m.face_cells.T
    ns = m.face_normal * m.face_area[:, None]
    np.add.at(acc, l, ns)
    ok = r >= 0
    np.add.at(acc, r[ok], -ns[ok])
    assert np.abs(acc).max() < 1e-9
    assert (m.face_tag == 1).sum() > 50 and m.volume.min() > 0


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_run_rans_pipeline_and_stop():
    from aero2d import core
    af = repanel(naca4("0012", 120), 60)
    calls = []

    def cb(text, frac):
        calls.append(text)
        return len(calls) > 4                 # остановить на первых итерациях
    p = core.run_rans(af, [0.0, 2.0], 0.15, 1e6, cb, max_iter=20)
    assert p.method == "rans" and len(p.alpha) == 2
    assert np.isfinite(p.cl[0]) and abs(p.cl[0]) < 0.05
    assert any("остановлен" in t for t in p.notes)


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_mesh_in_worker_thread():
    # окно программы строит сетку в фоновом потоке — gmsh не должен ставить обработчик сигналов
    import threading
    from aero2d.rans_mesh import build_mesh
    out = {}

    def job():
        try:
            out["n"] = build_mesh([repanel(naca4("0012", 120), 60)], re=1e6, far_radius=20).n_cells
        except Exception as e:                       # noqa: BLE001
            out["err"] = repr(e)
    t = threading.Thread(target=job)
    t.start()
    t.join()
    assert "err" not in out and out["n"] > 1000


def test_polar_order_and_convergence_defaults():
    from aero2d.rans import Convergence, _seq_order
    a = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    seq = _seq_order(a)
    assert a[seq[0]] == 0.0 and list(a[seq]) == [0, 2, 4, 8, -2, -4]
    c = Convergence()
    assert c.orders == 4.5 and c.hold == 20 and c.stall_ok


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_mesh_options():
    from aero2d.rans import mesh_for
    from aero2d.rans_mesh import MeshOptions
    af = naca4("0012", 121)
    m1 = mesh_for([af], 0.0, 1e6, opts=MeshOptions(far_radius=20))
    m2 = mesh_for([af], 0.0, 1e6, opts=MeshOptions(far_radius=20, n_surface=60, near_size=0.008))
    assert m2.n_cells < m1.n_cells and (m2.face_tag == 1).sum() < (m1.face_tag == 1).sum()


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_continue_rans_appends_history():
    from aero2d import core
    from aero2d.rans_mesh import MeshOptions
    af = repanel(naca4("0012", 120), 60)
    p = core.run_rans(af, [0.0], 0.15, 1e6, None, max_iter=8, mesh_opts=MeshOptions(far_radius=20))
    q = core.continue_rans(p, None, 6)
    H = q.field_ctx["history"][0]
    assert len(H["it"]) == 14 and H["it"][-1] == 14 and q.label.endswith("(досчитано)")
    assert len(p.field_ctx["history"][0]["it"]) == 8          # прежняя поляра не изменилась


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_rollback_keeps_best_state():
    from aero2d.rans import RansSolver
    from aero2d.rans_mesh import build_mesh
    # сетка без роста ячейки следа: на ней невязка в начале растёт и откат гарантирован
    m = build_mesh([repanel(naca4("0012", 120), 60)], re=1e6, far_radius=20, wake_h1_growth=0.0)
    s = RansSolver(m, 0.15, 1e6)
    s.rollback_factor, s.max_rollbacks = 1.05, 2          # искусственно частые откаты
    r = s.solve(0.0, max_iter=40)
    assert r.history["events"] and "откат" in r.history["events"][0][1]
    assert np.all(np.isfinite(r.state)) and len(r.history["res_cell"]) == m.n_cells


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_transfer_state_between_meshes():
    from aero2d.rans import RansSolver, coarse_options, mesh_for, transfer_state
    af = naca4("0012", 121)
    fine = mesh_for([af], 0.0, 1e6)
    coarse = mesh_for([af], 0.0, 1e6, opts=coarse_options([af]))
    assert coarse.n_cells < 0.8 * fine.n_cells
    s = RansSolver(coarse, 0.15, 1e6)
    Q = np.asarray(s.cons(s.jnp.asarray(np.tile(s.freestream(3.0), (coarse.n_cells, 1)))))
    Qf = transfer_state(coarse, Q, fine)
    assert Qf.shape == (fine.n_cells, 5) and np.all(np.isfinite(Qf)) and np.all(Qf[:, 0] > 0)


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_sa_rc_factor_near_one_in_boundary_layer():
    from aero2d.rans import RansSolver
    from aero2d.rans_mesh import build_mesh
    m = build_mesh([repanel(naca4("0012", 120), 60)], re=1e6, far_radius=20)
    s = RansSolver(m, 0.15, 1e6)
    r = s.solve(2.0, max_iter=60)
    s.turb_model = "sa-rc"
    w = s.prim(s.jnp.asarray(r.state))
    f = np.asarray(s.fr1(w, s.gradients(w, s._wall_values(w))))
    k = (m.wall_dist < 1e-3) & (m.centroid[:, 0] > 0.05) & (m.centroid[:, 0] < 0.95)
    assert 0.95 < np.median(f[k]) < 1.05 and f.min() >= 0 and f.max() <= 1.25


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_sa_aft_transition_model_basics():
    from aero2d.rans import RansSolver, wall_direction
    from aero2d.rans_mesh import build_mesh
    m = build_mesh([repanel(naca4("0012", 120), 60)], re=1e6, far_radius=20)
    d, g = wall_direction(m)
    assert np.allclose(np.hypot(g[:, 0], g[:, 1]), 1.0) and np.all(d <= m.wall_dist + 1e-9)
    s = RansSolver(m, 0.15, 1e6)
    s.turb_model = "sa-aft"
    assert s.nv == 7 and len(s.freestream(2.0)) == 7 and s.freestream(2.0)[4] == 0.1
    r = s.solve(2.0, max_iter=6)
    assert r.state.shape == (m.n_cells, 7) and np.all(np.isfinite(r.state))
    # в набегающем потоке ñ = 0, γ̃ = 0; в ламинарном слое у стенки γ̃ < 0 (γ → 1/c3)
    assert r.elements[0]["gam"] is not None
    s2 = RansSolver(m, 0.15, 1e6)                           # тёплый старт SA → SA-AFT и обратно
    s2.turb_model = "sa"
    r2 = s2.solve(2.0, Q0=r.state, max_iter=2)
    assert r2.state.shape == (m.n_cells, 5)


def test_rans_xtr_from_indicator():
    from aero2d.core import _rans_xtr
    xc = np.array([1.0, 0.5, 0.0, 0.5, 1.0])            # верх (з.к. → носок), низ
    x = np.linspace(0, 1, 11)
    jn = np.r_[np.zeros(11, int), np.full(11, 4)]
    xs = np.r_[x, x]
    ind = np.r_[(x > 0.42).astype(float), np.zeros(11)]   # верх: турбулентный с 0,5; низ ламинарный
    u, l_ = _rans_xtr(xc, jn, xs, ind)
    assert abs(u - 0.5) < 1e-12 and l_ == 1.0


def test_nested_dissection_lu_exact():
    import scipy.sparse as sp
    from aero2d.bilu import NestedLU
    rng = np.random.default_rng(1)
    n, k = 60, 3
    rows = list(range(n)) + list(range(n - 1)) + list(range(1, n)) + [0, n - 1]
    cols = list(range(n)) + list(range(1, n)) + list(range(n - 1)) + [n - 1, 0]
    blocks = [rng.normal(size=(k, k)) * 0.3 + (4 * np.eye(k) if r == c else 0) for r, c in zip(rows, cols)]
    A = sp.lil_matrix((n * k, n * k))
    for r, c, b in zip(rows, cols, blocks):
        A[r * k:(r + 1) * k, c * k:(c + 1) * k] = b
    A = A.tocsr()
    lu = NestedLU(n, np.array(rows), np.array(cols), k=k).factor(A)
    x = rng.normal(size=n * k)
    assert np.allclose(lu.solve(A @ x), x, atol=1e-10)


def test_gmres_left_right():
    from aero2d.krylov import gmres
    rng = np.random.default_rng(2)
    A = np.eye(40) * 3 + rng.normal(size=(40, 40)) * 0.2
    b = rng.normal(size=40)
    D = np.diag(1 / np.diag(A))
    for right in (False, True):
        x, info = gmres(lambda v: A @ v, b, prec=lambda v: D @ v, rtol=1e-10, maxiter=40, right=right)
        assert info == 0 and np.linalg.norm(A @ x - b) < 1e-8 * np.linalg.norm(b)


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_unsteady_and_gif(tmp_path):
    """Нестационарный RANS (BDF2) на грубой сетке: несколько шагов разгона из покоя
    и сборка GIF."""
    from aero2d.rans import RansSolver
    from aero2d.rans_mesh import build_mesh
    from aero2d.animate import make_gif
    af = repanel(naca4("0012", 120), 40)
    m = build_mesh([af], re=1e6, far_radius=15, far_size=5.0, ratio=1.4, slot_size=0.02, growth=0.4)
    s = RansSolver(m, 0.2, 1e6)
    run = s.solve_unsteady(5.0, None, t_end=0.1, steps_per_chord=20, sub_iter=3, frame_every=1)
    assert len(run["t"]) == 2 and np.all(np.isfinite(run["cl"])) and len(run["frames"]) == 3
    path = tmp_path / "a.gif"
    assert make_gif(m, [af], run, str(path), n=80, dpi=40) == 3
    assert path.stat().st_size > 1000


@pytest.mark.skipif(gmsh_missing, reason="нет gmsh")
def test_mesh_does_not_depend_on_alpha():
    """Сетка механизации не зависит от угла расчёта: поляра и отдельная точка
    считаются на одной сетке (раньше сгущение следа строилось при среднем угле поляры)."""
    from aero2d.multi import highlift
    from aero2d.rans import mesh_for
    from aero2d.rans_mesh import MeshOptions
    els = highlift(naca4("2412", 120), flap=dict(type="slotted", chord=0.3, deflection=20.0), n_side=50)
    o = MeshOptions(far_radius=15, n_surface=50, near_size=0.01, wake_size=0.02)
    assert mesh_for(els, 0.0, 1e6, opts=o).n_cells == mesh_for(els, 12.0, 1e6, opts=o).n_cells


def test_ncrit_from_tu():
    from aero2d.rans import ncrit_from_tu
    assert abs(ncrit_from_tu(0.07) - 9.0) < 0.01          # Мэк: Tu 0,07 % ≈ Ñкрит 9
    assert abs(ncrit_from_tu(1.0) - 2.75) < 0.01        # с ограничителем Дрелы τ = 2,5·tanh(Tu/2,5)
    assert abs(ncrit_from_tu(0.0408) - 10.3) < 0.01      # Шубауэр–Клебанов (Hildebrand 2023)
    assert ncrit_from_tu(0.1) > ncrit_from_tu(0.3)


def test_trip_mask_sides():
    """Принудительный переход SA-AFT: зона — у стенки, за x_tr своей стороны."""
    from aero2d.geometry import naca4, repanel
    from aero2d.rans import RansSolver
    from aero2d.rans_mesh import build_mesh
    m = build_mesh([repanel(naca4("2412", 120), 60)], re=1e6, far_radius=15, far_size=5.0, ratio=1.4,
                   slot_size=0.02, growth=0.4)
    s = RansSolver(m, 0.15, 1e6)
    s.turb_model = "sa-aft"
    assert not s.trip_mask().any()                         # xtr = 1 — свободный переход
    s.xtr_u, s.xtr_l = 0.3, 0.6
    k = s.trip_mask()
    c = m.centroid
    assert k.any()
    assert c[k & (c[:, 1] > 0.03), 0].min() > 0.28        # верх — за 0,3
    assert c[k & (c[:, 1] < -0.03), 0].min() > 0.55       # низ — за 0,6
    assert s.wall_d[k].max() < 0.06
    s.turb_model = "sa"
    assert not s.trip_mask().any()                         # без модели перехода — нет


def test_refine_surface_transition_zone():
    """Сгущение у точки перехода: в зоне шаг ≤ ds, рост шага от края зоны плавный, носок — узел."""
    from aero2d.geometry import naca4, repanel
    from aero2d.rans import refine_surface
    af = repanel(naca4("0012", 200), 120)
    z = {"upper": (0.3, 0.5), "lower": (0.5, 0.7)}
    r = refine_surface(af, z, 0.001)
    d = np.hypot(np.diff(r.x), np.diff(r.y))
    xm = 0.5 * (r.x[1:] + r.x[:-1])
    ile = int(np.argmin(r.x))
    assert abs(r.x[ile] - af.x.min()) < 1e-9
    up = np.arange(len(d)) < ile
    assert d[up & (xm > 0.32) & (xm < 0.48)].max() < 1.05e-3
    assert d[~up & (xm > 0.52) & (xm < 0.68)].max() < 1.05e-3
    assert d[up & (xm > 0.6) & (xm < 0.9)].min() > 2e-3          # вне зоны — прежний шаг
    mid = (xm[1:] > 0.05) & (xm[1:] < 0.95)
    q = np.maximum(d[1:] / d[:-1], d[:-1] / d[1:])
    assert q[mid].max() < 1.3


def test_laminar_mask_viscous_transition():
    """Переход по вязкому методу: ламинарная область — у стенки до точки перехода своей стороны."""
    from aero2d.geometry import naca4, repanel
    from aero2d.rans import RansSolver, lam_xtr_at
    from aero2d.rans_mesh import build_mesh
    m = build_mesh([repanel(naca4("2412", 120), 60)], re=1e6, far_radius=15, far_size=5.0, ratio=1.4,
                   slot_size=0.02, growth=0.4)
    s = RansSolver(m, 0.15, 1e6)
    assert not s.laminar_mask().any()
    s.lam_xtr = (0.3, 0.6)
    k = s.laminar_mask()
    c = m.centroid
    assert k.any()
    assert c[k & (c[:, 1] > 0.03), 0].max() < 0.32         # верх — до 0,3
    assert c[k & (c[:, 1] < -0.03), 0].max() < 0.62        # низ — до 0,6
    assert lam_xtr_at({0.0: (0.4, 0.6), 4.0: (0.2, 0.8)}, 2.0) == (0.30000000000000004, 0.7)
    assert lam_xtr_at({0.0: (0.4, 0.6)}, 10.0) == (0.4, 0.6)


def test_sst_laminar_region_jacobian():
    """k-ω SST с ламинарным участком (переход по вязкому методу): невязка и якобиан строятся
    (маска порождения k передаётся в якобиан поячеечно), в ламинарных ячейках порождения k нет."""
    import numpy as np
    from aero2d.geometry import naca4, repanel
    from aero2d.rans import RansSolver
    from aero2d.rans_mesh import build_mesh
    m = build_mesh([repanel(naca4("2412", 120), 60)], re=1e6, far_radius=15, far_size=5.0, ratio=1.4,
                   slot_size=0.02, growth=0.4)
    s = RansSolver(m, 0.15, 1e6)
    s.turb_model = "sst"
    s.lam_xtr = (0.3, 0.6)
    winf = s.freestream(2.0)
    Q = np.asarray(s.cons(s.jnp.asarray(np.tile(winf, (s.nc, 1)))))
    F = s._fns()
    R = np.asarray(F["res1"](s.jnp.asarray(Q), s.jnp.asarray(winf), s.jnp.asarray(0.0)))
    assert R.shape == (s.nc, 6) and np.all(np.isfinite(R))
    A = s.jac1(s.jnp.asarray(Q), s.jnp.asarray(winf), s.jnp.asarray(0.0), np.ones(s.nc))
    assert A.shape == (6 * s.nc, 6 * s.nc)


def test_seq_from_seeds():
    """Добавление углов к поляре: порядок — от ближайшего посчитанного угла, каждый следующий от соседа."""
    import numpy as np
    from aero2d.rans import _seq_from
    al = np.array([13.0, 12.0, 14.0, 15.0, 11.0])
    assert [al[i] for i in _seq_from(al, [0.0, 2.0, 10.0])] == [11.0, 12.0, 13.0, 14.0, 15.0]


def test_merge_and_drop_polars():
    import numpy as np
    from aero2d.core import Polar, drop_unconverged, merge_polars
    def mk(a, ok):
        a = np.array(a, float)
        return Polar("P", "rans", 0.15, a, 0.1 * a, 0.01 + 0 * a, 0 * a, np.array(ok, bool),
                     cp_x=[np.r_[x] for x in a], cp=[np.r_[x] for x in a],
                     field_ctx=dict(method="rans", alpha=a, states=[f"S{x:g}" for x in a], history=[None] * len(a)))
    m = merge_polars(mk([0, 2, 4, 6], [1, 1, 0, 1]), mk([5, 6, 7], [1, 1, 1]))
    assert list(m.alpha) == [0, 2, 4, 5, 6, 7]
    assert m.field_ctx["states"] == ["S0", "S2", "S4", "S5", "S6", "S7"]
    assert [c[0] for c in m.cp] == [0, 2, 4, 5, 6, 7]
    d = drop_unconverged(m)
    assert list(d.alpha) == [0, 2, 5, 6, 7] and d.field_ctx["states"] == ["S0", "S2", "S5", "S6", "S7"]


def test_rans_polar_seeds_extend():
    """Добавить углы к поляре: без грубой сетки, первый новый угол стартует от решения прежней поляры."""
    import numpy as np
    from aero2d.geometry import naca4, repanel
    from aero2d import rans as R
    from aero2d.rans_mesh import build_mesh
    m = build_mesh([repanel(naca4("2412", 120), 60)], re=1e6, far_radius=15, far_size=5.0, ratio=1.4,
                   slot_size=0.02, growth=0.4)
    r0, _ = R.rans_polar([repanel(naca4("2412", 120), 60)], [0.0], 1e6, mesh=m, max_iter=15, coarse_start=False)
    calls = []
    orig = R.RansSolver.solve

    def spy(self, alpha, Q0=None, *a, **k):
        calls.append((float(alpha), Q0 is not None, k.get("alpha0")))
        return orig(self, alpha, Q0, *a, **k)
    R.RansSolver.solve = spy
    try:
        r1, _ = R.rans_polar([repanel(naca4("2412", 120), 60)], [2.0, 1.0], 1e6, mesh=m, max_iter=5,
                             seeds=[(0.0, r0[0].state)])
    finally:
        R.RansSolver.solve = orig
    assert calls[0] == (1.0, True, 0.0)            # сначала ближайший к посчитанному 0°, от его решения
    # 2° — от ближайшего сошедшегося: 1° (если сошёлся за 5 итераций) или прежний 0°
    assert calls[1][0] == 2.0 and calls[1][1] and calls[1][2] in (0.0, 1.0)
    assert all(r is not None for r in r1)
