"""Анимация поля течения по времени (GIF) по кадрам нестационарного расчёта RANS.

Кадры — состояния Q в центрах ячеек (RansSolver.solve_unsteady). Каждый кадр
интерполируется на прямоугольную сетку (одна триангуляция центров на все
кадры), рисуется заливкой выбранной величины с телами профиля и графиком
Cl(t) с бегущей отметкой; кадры собираются в GIF средствами Pillow (входит в
зависимости matplotlib — новых пакетов не нужно).
"""
from __future__ import annotations

import io

import numpy as np

GAM = 1.4

QUANTITIES = {
    "Завихренность ωc/V∞": "vorticity",
    "Скорость |V|/V∞": "speed",
    "Давление Cp": "cp",
    "Число Маха": "mach",
    "Турбулентная вязкость ν̃/ν∞": "nut",
}


def _grid(box, n):
    x0, x1, y0, y1 = box
    ny = max(int(n * (y1 - y0) / (x1 - x0)), 20)
    return np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, ny))


def frame_fields(mesh, Q, mach, X, Y, tri=None):
    """Поля кадра на сетке X, Y: u, v (в долях V∞), Cp, M, ν̃/ν∞, ω·c/V∞.
    tri — rans_mesh.NodeInterp этой сетки (интерполяция по самой расчётной сетке,
    с прилипанием на стенке; строится один раз на все кадры)."""
    from .rans_mesh import NodeInterp
    if tri is None:
        tri = NodeInterp(mesh)
    Q = np.asarray(Q, float)
    rho = Q[:, 0]
    u, v = Q[:, 1] / rho, Q[:, 2] / rho
    p = (GAM - 1) * (Q[:, 3] - 0.5 * rho * (u ** 2 + v ** 2))
    vals = np.column_stack([u / mach, v / mach, (p - 1 / GAM) / (0.5 * mach ** 2),
                            np.sqrt(u ** 2 + v ** 2) / np.sqrt(GAM * p / rho), Q[:, 4] / rho])
    f = tri(vals, X.ravel(), Y.ravel(), wall={0: 0.0, 1: 0.0, 3: 0.0, 4: 0.0}).reshape(X.shape + (5,))
    U, V = f[..., 0], f[..., 1]
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    om = np.gradient(V, dx, axis=1) - np.gradient(U, dy, axis=0)
    return dict(u=U, v=V, cp=f[..., 2], mach=f[..., 3], nut=f[..., 4], speed=np.hypot(U, V), vorticity=om)


def make_gif(mesh, elements, run, path, quantity="vorticity", box=(-0.5, 2.5, -0.9, 0.9), n=320,
             fps=12, dpi=90, vlim=None, streamlines=False, title="", progress=None):
    """GIF по результату RansSolver.solve_unsteady (run). quantity — ключ из
    QUANTITIES.values(). vlim — пределы шкалы (по умолчанию — по всем кадрам,
    одинаковые для всего ролика). progress(k, n) → True — прервать.
    Возвращает число кадров."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.path import Path
    from PIL import Image
    from .rans_mesh import NodeInterp

    mach = float(run.get("mach", 0.15)) if isinstance(run, dict) else 0.15
    frames = run["frames"]
    X, Y = _grid(box, n)
    from .rans_mesh import wall_loops
    tri = NodeInterp(mesh)
    ti, ok, _ = tri.locate(X.ravel(), Y.ravel())
    inside = ~ok.reshape(X.shape)                      # вне расчётной области — внутри тел
    walls = wall_loops(mesh)
    fields = []
    for k, (t, Q) in enumerate(frames):
        f = frame_fields(mesh, Q, mach, X, Y, tri)
        z = np.where(inside, np.nan, f[quantity])
        fields.append((t, z, f))
        if progress and progress(k, 2 * len(frames)):
            return 0
    # пределы шкалы — без тонкого слоя у стенок (там завихренность на порядки больше,
    # чем в отрыве и следе, и вихри стали бы неразличимы)
    from scipy.ndimage import distance_transform_edt
    dpx = (box[1] - box[0]) / max(X.shape[1] - 1, 1)
    far = distance_transform_edt(~inside) * dpx > 0.02 if inside.any() else np.ones(X.shape, bool)
    allz = np.concatenate([np.ravel(z[np.isfinite(z) & far]) for _, z, _ in fields])
    if vlim is None:
        if quantity == "vorticity":
            a = float(np.percentile(np.abs(allz), 99.0)) or 1.0
            vlim = (-a, a)
        elif quantity == "cp":
            vlim = (float(np.percentile(allz, 1.0)), float(np.percentile(allz, 99.5)))
        else:
            vlim = (float(np.percentile(allz, 0.5)), float(np.percentile(allz, 99.5)))
    cmap = {"vorticity": "RdBu_r", "cp": "RdBu_r", "speed": "viridis", "mach": "viridis",
            "nut": "magma"}.get(quantity, "viridis")
    label = {v: k for k, v in QUANTITIES.items()}.get(quantity, quantity)
    t_all = np.asarray(run["t"])
    cl_all = np.asarray(run["cl"])
    images = []
    for k, (t, z, f) in enumerate(fields):
        fig = Figure(figsize=(8.0, 5.4), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.10, 0.30, 0.77, 0.62])
        cax = fig.add_axes([0.89, 0.30, 0.02, 0.62])
        im = ax.imshow(z, origin="lower", extent=box, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                       interpolation="bilinear", aspect="equal")
        fig.colorbar(im, cax=cax)
        if streamlines:
            try:
                uu = np.where(inside, np.nan, f["u"])
                vv = np.where(inside, np.nan, f["v"])
                ax.streamplot(X[0], Y[:, 0], uu, vv, density=1.0, color="k", linewidth=0.4, arrowsize=0.5)
            except Exception:                                   # noqa: BLE001 — линии тока не обязательны
                pass
        for wx_, wy_ in walls:
            ax.fill(wx_, wy_, color="#888", zorder=3)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        cl_t = np.interp(t, t_all, cl_all) if len(t_all) else np.nan
        ax.set_title(f"{title}  t·V∞/c = {t:.2f}   Cl = {cl_t:.3f}\n{label}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax2 = fig.add_axes([0.10, 0.08, 0.77, 0.15])
        if len(t_all):
            ax2.plot(t_all, cl_all, color="#1f5aa6", lw=1.2)
            ax2.axvline(t, color="#c0392b", lw=1.0)
            ax2.plot([t], [cl_t], "o", color="#c0392b", ms=4)
            ax2.set_xlim(0, max(float(t_all[-1]), 1e-6))
        ax2.set_xlabel("t·V∞/c", fontsize=8)
        ax2.set_ylabel("Cl", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(alpha=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=255))
        if progress and progress(len(frames) + k, 2 * len(frames)):
            return 0
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / max(fps, 1)), loop=0,
                   optimize=False, disposal=2)
    return len(images)
