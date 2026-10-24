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
