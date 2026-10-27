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
    assert c.orders == 3 and c.hold == 20 and c.stall_ok


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
    m = build_mesh([repanel(naca4("0012", 120), 60)], re=1e6, far_radius=20)
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
