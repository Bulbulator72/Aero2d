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


def turb_values(Q, mach, re, chord=1.0):
    """Турбулентные величины в ячейках по состоянию RANS (обе модели):
    mut — вихревая вязкость μt/μ∞; для k-ω SST ещё k — k/V∞², omega_ln — ln(ω·c/V∞).
    SA: μt = ρν̃·fv1; SST: μt = ρk/ω (без ограничителя Брэдшоу — он влияет только в
    пограничном слое с сильным градиентом давления)."""
    from .rans import GAM as G, SUTH, CV1
    Q = np.asarray(Q, float)
    rho = Q[:, 0]
    T = G * (G - 1) * (Q[:, 3] - 0.5 * (Q[:, 1] ** 2 + Q[:, 2] ** 2) / rho) / rho
    mu = T ** 1.5 * (1 + SUTH) / (T + SUTH)                 # μ/μ∞
    if Q.shape[1] == 6:
        kh = np.maximum(Q[:, 4] / rho, 0.0)
        wt = Q[:, 5] / rho
        k = kh * 1e-5                                       # k/V∞² (k_ref = 1e−5·V∞²)
        mut = rho * k / np.exp(wt) * re / chord             # ρk/ω / μ∞, μ∞ = V∞c/Re
        return dict(mut=mut, k=k, omega_ln=wt)
    chi = np.maximum(Q[:, 4] / rho, 0.0)
    X = rho * chi / mu
    return dict(mut=rho * chi * X ** 3 / (X ** 3 + CV1 ** 3))


def _grid(box, n):
    x0, x1, y0, y1 = box
    nx = n
    ny = max(int(n * (y1 - y0) / (x1 - x0)), 20)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    return np.meshgrid(xs, ys)


def compute_field(ctx: dict, k: int, box=None, n: int = 320) -> dict:
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
    elif ctx["method"] == "multi_viscous":
        from .mviscous import MultiViscousSolver
        from .sources import lin_source
        ms = MultiViscousSolver(ctx["elements"], ctx["re"], ncrit=ctx["ncrit"], mach=ctx["mach"],
                                xtr=ctx["xtr"], chord=ctx.get("chord") or 1.0, n_panels=ctx["n_panels"])
        with np.errstate(all="ignore"):
            r = ms.solve(alpha, init=ctx["states"][k], homotopy=False, max_iter=15)
            g_inv = ms.mp.gamma(alpha)
            ms._g_cur = None
            ms._build(alpha, g_inv)
            Xs = ms._from_state(r.state)
            Xs = ms._relocate(Xs, alpha, g_inv)
        m = Xs[:, 1]
        g = g_inv + ms.dg_dm @ m
        q = np.zeros(ms.offs[-1])
        nodes = np.where(ms.st_of_node >= 0)[0]
        q[nodes] = ms.sgm[nodes] * m[ms.st_of_node[nodes]]
        U = np.full(xp.shape, np.cos(a))
        V = np.full(xp.shape, np.sin(a))
        inside = np.zeros(xp.shape, bool)
        bodies, disp, wakes = [], [], []
        ue = ms.ue_inv + ms.D @ m
        for e, (x, y) in enumerate(ms.els):
            sl = slice(ms.offs[e], ms.offs[e + 1])
            Uv, Vv = vortex_velocity_influence(x, y, xp, yp)
            _, Px, Py = lin_source(x, y, xp, yp)
            wx, wy = ms.wakes[e]
            _, Pxw, Pyw = lin_source(wx, wy, xp, yp, wake_sheet=True)
            qw = m[ms.st_wk_el[e]]
            U += Uv @ g[sl] + Py @ q[sl] + Pyw @ qw
            V += Vv @ g[sl] - (Px @ q[sl] + Pxw @ qw)
            inside |= Path(np.column_stack([x, y])).contains_points(np.column_stack([xp, yp]))
            bodies.append((x, y))
            wakes.append((wx, wy))
            nx_, ny_ = _node_normals(x, y)
            for sts in (ms.st_up_el[e], ms.st_lo_el[e]):
                nd = ms.node[sts] - ms.offs[e]
                ds = Xs[sts, 1] / np.maximum(np.abs(ue[sts]), 1e-6)
                disp.append((x[nd] + ds * nx_[nd], y[nd] + ds * ny_[nd]))
        extra.update(bodies=bodies, disp=disp, wakes=wakes, converged=bool(r.converged))
        x, y = bodies[0]
    elif ctx["method"] == "rans":
        from .rans_mesh import NodeInterp
        mesh, Q = ctx["mesh"], ctx["states"][k]
        if Q is None:
            raise ValueError("для этого угла RANS не посчитан")
        GAM = 1.4
        Q = np.asarray(Q)
        rho = Q[:, 0]
        u, v = Q[:, 1] / rho, Q[:, 2] / rho
        p = (GAM - 1) * (Q[:, 3] - 0.5 * rho * (u**2 + v**2))
        M = ctx["mach"]
        ip = ctx.get("_interp")
        if ip is None:
            ip = ctx["_interp"] = NodeInterp(mesh)
        tv = turb_values(Q, M, ctx.get("re", 1e6), ctx.get("chord") or 1.0)
        sst = "omega_ln" in tv
        vals = np.column_stack([u / M, v / M, (p - 1 / GAM) / (0.5 * M**2),
                                np.sqrt(u**2 + v**2) / np.sqrt(GAM * p / rho),
                                tv["k"] if sst else Q[:, 4] / rho, tv["mut"]]
                               + ([tv["omega_ln"]] if sst else []))
        # на стенке: прилипание (u = v = 0, M = 0), ν̃ = k = μt = 0; давление и ω — из соседних ячеек
        wall_bc = {0: 0.0, 1: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        f = ip(vals, xp, yp, wall=wall_bc)
        # заливка — по самой расчётной сетке (значения в узлах, линейно в треугольниках): гладкий
        # контур тел и вся детализация слоя и следа; растр — только для линий тока
        Nv = ip.W @ vals
        for j_, v_ in wall_bc.items():
            Nv[ip.wall, j_] = v_
        x0b, x1b, y0b, y1b = box
        dx_, dy_ = 0.05 * (x1b - x0b), 0.05 * (y1b - y0b)
        P = ip.nodes[ip.tris]
        sel = ((P[..., 0].max(1) >= x0b - dx_) & (P[..., 0].min(1) <= x1b + dx_)
               & (P[..., 1].max(1) >= y0b - dy_) & (P[..., 1].min(1) <= y1b + dy_))
        nodes_ = {"cp": Nv[:, 2], "speed": np.hypot(Nv[:, 0], Nv[:, 1]), "mach_local": Nv[:, 3],
                  "mut": Nv[:, 5]}
        if sst:
            nodes_.update(k_turb=Nv[:, 4], omega=np.exp(Nv[:, 6]))
        else:
            nodes_["nut"] = Nv[:, 4]
        extra.update(tri=(ip.nodes[:, 0], ip.nodes[:, 1], ip.tris[sel]), node=nodes_)
        U, V = f[:, 0], f[:, 1]
        from .rans_mesh import wall_loops
        inside = ~np.isfinite(U)                       # вне расчётной области — внутри тел
        bodies = ctx.get("_walls")
        if bodies is None:
            bodies = ctx["_walls"] = wall_loops(mesh)    # тела — по стенкам самой сетки
        extra.update(bodies=bodies, converged=True, mach_local=f[:, 3].reshape(X.shape),
                     mut=f[:, 5].reshape(X.shape), cp_rans=f[:, 2])
        if sst:
            extra.update(k_turb=f[:, 4].reshape(X.shape), omega=np.exp(f[:, 6]).reshape(X.shape))
        else:
            extra["nut"] = f[:, 4].reshape(X.shape)
        H = (ctx.get("history") or [None] * (k + 1))[k]
        if H is not None and "res_cell" in H:
            ci = ip.cell_at(xp, yp)
            rc = np.where(ci >= 0, np.asarray(H["res_cell"], float)[np.maximum(ci, 0)], np.nan)
            extra["res_cell"] = rc.reshape(X.shape)
            extra["worst_xy"] = mesh.centroid[np.asarray(H.get("worst", []), int)]
        x, y = bodies[0]
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
    if ctx["method"] not in ("multi", "multi_viscous", "rans"):
        inside = Path(np.column_stack([x, y])).contains_points(np.column_stack([xp, yp]))
    q2 = U**2 + V**2
    cp0 = 1.0 - q2
    cp = karman_tsien(cp0, ctx.get("mach", 0.0)) if ctx.get("mach", 0.0) > 0 else cp0
    if "cp_rans" in extra:                     # RANS: давление из решения (с потерями в следе)
        cp = extra.pop("cp_rans")
    shape = X.shape
    speed = np.sqrt(U**2 + V**2)
    # для заливки: внутри тел — значения ближайшей точки снаружи, иначе по
    # контуру остаётся «лесенка» пустых ячеек; тело потом закрашивается сверху
    ins = inside.reshape(shape)
    if ins.any():
        from scipy.ndimage import distance_transform_edt
        iy, ix = distance_transform_edt(ins, return_distances=False, return_indices=True)
        cp_f = cp.reshape(shape)[iy, ix]
        sp_f = speed.reshape(shape)[iy, ix]
        # остальные величины (M, μt, k, ω, …) — так же: иначе у контура белая «лесенка»
        for k_ in [k_ for k_, v_ in extra.items() if isinstance(v_, np.ndarray) and v_.shape == shape]:
            extra[k_] = np.where(ins | ~np.isfinite(extra[k_]), extra[k_][iy, ix], extra[k_])
    else:
        cp_f, sp_f = cp.reshape(shape), speed.reshape(shape)
    U, V = (np.where(inside, np.nan, arr).reshape(shape) for arr in (U, V))
    cp, speed = cp_f, sp_f
    extra.setdefault("bodies", [(x, y)])
    return dict(X=X, Y=Y, U=U, V=V, cp=cp, speed=speed, alpha=alpha,
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
