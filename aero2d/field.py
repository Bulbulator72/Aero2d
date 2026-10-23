"""Поле течения вокруг профиля: скорость, давление, линии тока.

Скорость в точках поля — сумма набегающего потока, вихревого слоя на контуре
(γ панельного решения) и, для вязкого расчёта, источников, моделирующих
вытеснение пограничным слоем и следом (σ = d(Ue·δ*)/ds, как в самом
вязко-невязком методе). Так поле согласовано с рассчитанными Cl и Cp.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path

from .panel import PanelSolver, vortex_velocity_influence, karman_tsien

REGIONS = {
    "Весь профиль": (-0.45, 1.6, -0.55, 0.55),
    "Механизация (щель закрылка)": (0.55, 1.15, -0.25, 0.12),
    "Передняя кромка": (-0.08, 0.22, -0.12, 0.12),
    "Задняя кромка и след": (0.7, 1.35, -0.18, 0.14),
}


def _grid(box, n):
    x0, x1, y0, y1 = box
    nx = n
    ny = max(int(n * (y1 - y0) / (x1 - x0)), 20)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    return np.meshgrid(xs, ys)


def compute_field(ctx: dict, k: int, box=None, n: int = 220) -> dict:
    """ctx — описание расчёта (Polar.field_ctx), k — номер точки поляры.
    Возвращает сетку X, Y, скорости U, V (в долях V∞), Cp и маску тела."""
    box = box or REGIONS["Весь профиль"]
    X, Y = _grid(box, n)
    xp, yp = X.ravel(), Y.ravel()
    alpha = float(ctx["alpha"][k])
    a = np.radians(alpha)
    af = ctx.get("airfoil")
    extra = {}
    if ctx["method"] == "viscous":
        from .viscous import ViscousSolver
        from .sources import lin_source
        npn = ctx["n_panels"][k]
        v = ViscousSolver(af, ctx["re"], ncrit=ctx["ncrit"], mach=ctx["mach"], xtr=ctx["xtr"], n_panels=npn)
        with np.errstate(all="ignore"):
            r = v.solve(alpha, init=ctx["states"][k], homotopy=False, max_iter=15)
            g_inv = v.ps.gamma(alpha)
            v._g_cur = None
            v._build(alpha, g_inv)
            Xs = v._from_state(r.state)
            Xs = v._relocate(Xs, alpha, g_inv)
        m = Xs[:, 1]
        g = g_inv + v.dg_dm @ m
        sgm = np.where(np.arange(v.N + 1) <= v.k, -1.0, 1.0)
        qa = np.zeros(v.N + 1)
        nodes = np.where(v.st_of_node >= 0)[0]
        qa[nodes] = sgm[nodes] * m[v.st_of_node[nodes]]
        qw = m[v.st_wk]
        x, y = v.x, v.y
        Uv, Vv = vortex_velocity_influence(x, y, xp, yp)
        _, Pxa, Pya = lin_source(x, y, xp, yp)
        _, Pxw, Pyw = lin_source(v.wx, v.wy, xp, yp, wake_sheet=True)
        U = np.cos(a) + Uv @ g + Pya @ qa + Pyw @ qw
        V = np.sin(a) + Vv @ g - (Pxa @ qa + Pxw @ qw)
        extra["wake"] = (v.wx, v.wy)
        # граница вытеснения: контур + δ* по нормали
        disp = []
        for nm, sts in (("upper", v.st_up), ("lower", v.st_lo)):
            nd = v.node[sts]
            ds = Xs[sts, 1] / np.maximum(np.abs(v.ue_inv[sts] + v.D[sts] @ m), 1e-6)
            nx_, ny_ = _node_normals(x, y)
            disp.append((x[nd] + ds * nx_[nd], y[nd] + ds * ny_[nd]))
        extra["disp"] = disp
        extra["converged"] = bool(r.converged)
    elif ctx["method"] == "multi":
        from .multi import MultiPanelSolver
        ms = MultiPanelSolver(ctx["elements"], chord=ctx.get("chord"))
        U = np.full(xp.shape, np.cos(a))
        V = np.full(xp.shape, np.sin(a))
        inside = np.zeros(xp.shape, bool)
        bodies = []
        for (x, y), g in zip(ms.els, ms.gamma(alpha)):
            Uv, Vv = vortex_velocity_influence(x, y, xp, yp)
            U += Uv @ g
            V += Vv @ g
            inside |= Path(np.column_stack([x, y])).contains_points(np.column_stack([xp, yp]))
            bodies.append((x, y))
        extra["bodies"] = bodies
        extra["converged"] = True
        x, y = bodies[0]
    else:
        ps = PanelSolver(af)
        g = ps.gamma(alpha)
        x, y = ps.x, ps.y
        Uv, Vv = vortex_velocity_influence(x, y, xp, yp)
        U = np.cos(a) + Uv @ g
        V = np.sin(a) + Vv @ g
        extra["converged"] = True
    if ctx["method"] != "multi":
        inside = Path(np.column_stack([x, y])).contains_points(np.column_stack([xp, yp]))
    q2 = U**2 + V**2
    cp0 = 1.0 - q2
    cp = karman_tsien(cp0, ctx.get("mach", 0.0)) if ctx.get("mach", 0.0) > 0 else cp0
    shape = X.shape
    U, V, cp = (np.where(inside, np.nan, arr).reshape(shape) for arr in (U, V, cp))
    extra.setdefault("bodies", [(x, y)])
    return dict(X=X, Y=Y, U=U, V=V, cp=cp, speed=np.sqrt(U**2 + V**2), alpha=alpha,
                body=(x, y), **extra)


def _node_normals(x, y):
    """Внешние нормали в узлах контура (обход: верх от задней кромки к передней,
    затем низ — внешняя нормаль слева от направления обхода… с учётом знака)."""
    tx = np.gradient(x)
    ty = np.gradient(y)
    t = np.hypot(tx, ty)
    tx, ty = tx / t, ty / t
    # обход против часовой стрелки — внешняя нормаль справа от касательной
    return ty, -tx
