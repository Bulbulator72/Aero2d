"""Слой источников с линейным распределением на полупанелях (порт схемы XFOIL:
PSILIN/PSWLIN, Drela 1989).

Интенсивность источников задаётся через «массу» q в узлах слоя: σ = dq/ds.
На каждой панели [jo, jp] σ линейна на двух половинах; значения в узлах —
центральные разности q, в середине панели — разность по самой панели. Слой
получается непрерывным, поэтому скорость в узлах конечна, а быстрые
(«шахматные») моды q правильно влияют на скорость.

Функция возвращает чувствительности функции тока и её градиента к q:
  Psi[p, j] = ∂ψ(p)/∂q_j,   Px, Py — ∂²ψ/∂x∂q, ∂²ψ/∂y∂q (U = Py, V = −Px).
"""
from __future__ import annotations

import numpy as np

QOPI = 0.25 / np.pi


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cut_angle(dx, dy, cut=0.0):
    """Полярный угол вектора (dx, dy) с разрезом вдоль направления cut (рад):
    значения в (cut, cut + 2π]."""
    return cut + np.mod(np.arctan2(dy, dx) - cut - 1e-300, 2 * np.pi)


def unwrapped_angles(path_x, path_y, xp, yp, cut=0.0):
    """Полярные углы точек (xp, yp) относительно узлов пути (тело + его след,
    путь заканчивается в конце следа), непрерывные вдоль пути. Разрез — луч
    из конца следа в направлении cut (по умолчанию +x, вниз по потоку); он не
    должен пересекать тела."""
    dx = xp[:, None] - path_x[None, :]
    dy = yp[:, None] - path_y[None, :]
    raw = np.arctan2(dy, dx)
    phi = np.empty_like(raw)
    phi[:, -1] = cut_angle(dx[:, -1], dy[:, -1], cut)
    for j in range(raw.shape[1] - 2, -1, -1):
        phi[:, j] = phi[:, j + 1] + _wrap(raw[:, j] - raw[:, j + 1])
    return phi


def lin_source(xs, ys, xp, yp, self_idx=None, on_sheet=False, wake_sheet=False, phi=None):
    """xs, ys — узлы слоя (ns); xp, yp — точки (np). self_idx[p] — индекс узла
    слоя, совпадающего с точкой p (или -1). on_sheet — точки лежат на этом же
    слое (тогда SGN = 1, как в XFOIL). wake_sheet — слой следа (ветвь арктангенса
    как в PSWLIN).
    phi (np, ns) — заданные полярные углы точек относительно узлов слоя
    (непрерывные вдоль слоя): функция тока многозначна, и для точек на другом
    теле разрез нужно провести так, чтобы он это тело не пересекал (см.
    unwrapped_angles)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xp, yp = np.asarray(xp, float), np.asarray(yp, float)
    ns, npt = len(xs), len(xp)
    if self_idx is None:
        self_idx = -np.ones(npt, int)
    Psi = np.zeros((npt, ns))
    Px = np.zeros((npt, ns))
    Py = np.zeros((npt, ns))
    shift_sign = -1.0 if wake_sheet else 1.0
    for jo in range(ns - 1):
        jp = jo + 1
        jm = jo - 1 if jo > 0 else jo
        jq = jp + 1 if jp < ns - 1 else jp
        dso = np.hypot(xs[jp] - xs[jo], ys[jp] - ys[jo])
        if dso == 0:
            continue
        dsio = 1.0 / dso
        sx, sy = (xs[jp] - xs[jo]) * dsio, (ys[jp] - ys[jo]) * dsio
        apan = np.arctan2(sx, -sy)
        rx1, ry1 = xp - xs[jo], yp - ys[jo]
        rx2, ry2 = xp - xs[jp], yp - ys[jp]
        x1 = sx * rx1 + sy * ry1
        x2 = sx * rx2 + sy * ry2
        yy = sx * ry1 - sy * rx1
        rs1 = rx1 * rx1 + ry1 * ry1
        rs2 = rx2 * rx2 + ry2 * ry2
        sgn = np.ones(npt) if on_sheet else np.where(yy >= 0, 1.0, -1.0)
        ok1 = (self_idx != jo) & (rs1 > 0)
        ok2 = (self_idx != jp) & (rs2 > 0)
        g1 = np.where(ok1, np.log(np.where(ok1, rs1, 1.0)), 0.0)
        g2 = np.where(ok2, np.log(np.where(ok2, rs2, 1.0)), 0.0)
        t1 = np.where(ok1, np.arctan2(sgn * x1, sgn * yy) + shift_sign * (0.5 - 0.5 * sgn) * np.pi, 0.0)
        t2 = np.where(ok2, np.arctan2(sgn * x2, sgn * yy) + shift_sign * (0.5 - 0.5 * sgn) * np.pi, 0.0)
        x0 = 0.5 * (x1 + x2)
        rs0 = x0 * x0 + yy * yy
        g0 = np.log(rs0)
        t0 = np.arctan2(sgn * x0, sgn * yy) + shift_sign * (0.5 - 0.5 * sgn) * np.pi
        if phi is not None:
            # t − apan ≡ −φ (φ — полярный угол точки относительно узла)
            ph1 = phi[:, jo]
            ph0r = np.arctan2(yp - 0.5 * (ys[jo] + ys[jp]), xp - 0.5 * (xs[jo] + xs[jp]))
            ph0 = ph1 + _wrap(ph0r - ph1)
            ph2 = phi[:, jp]
            t1, t0, t2 = apan - ph1, apan - ph0, apan - ph2
        # производные локальных координат по координатам точки:
        # для направления n: x1i = x2i = s·n, yyi = s × n
        dirs = ((1.0, 0.0), (0.0, 1.0))

        dsm = np.hypot(xs[jp] - xs[jm], ys[jp] - ys[jm])
        dsim = 1.0 / dsm
        dsp = np.hypot(xs[jq] - xs[jo], ys[jq] - ys[jo])
        dsip = 1.0 / dsp

        # ---- полупанель 1–0
        dxinv = 1.0 / (x1 - x0)
        psum = x0 * (t0 - apan) - x1 * (t1 - apan) + 0.5 * yy * (g1 - g0)
        pdif = ((x1 + x0) * psum + rs1 * (t1 - apan) - rs0 * (t0 - apan) + (x0 - x1) * yy) * dxinv
        psx1 = -(t1 - apan)
        psx0 = t0 - apan
        psyy = 0.5 * (g1 - g0)
        pdx1 = ((x1 + x0) * psx1 + psum + 2.0 * x1 * (t1 - apan) - pdif) * dxinv
        pdx0 = ((x1 + x0) * psx0 + psum - 2.0 * x0 * (t0 - apan) + pdif) * dxinv
        pdyy = ((x1 + x0) * psyy + 2.0 * (x0 - x1 + yy * (t1 - t0))) * dxinv
        cm_ = QOPI * (-psum * dsim + pdif * dsim)
        co_ = QOPI * (-psum * dsio - pdif * dsio)
        cp_ = QOPI * (psum * (dsio + dsim) + pdif * (dsio - dsim))
        Psi[:, jm] += cm_
        Psi[:, jo] += co_
        Psi[:, jp] += cp_
        for k, (nx, ny) in enumerate(dirs):
            xi_ = sx * nx + sy * ny
            yyi = sx * ny - sy * nx
            psni = psx1 * xi_ + psx0 * xi_ + psyy * yyi
            pdni = pdx1 * xi_ + pdx0 * xi_ + pdyy * yyi
            T = Px if k == 0 else Py
            T[:, jm] += QOPI * (-psni * dsim + pdni * dsim)
            T[:, jo] += QOPI * (-psni * dsio - pdni * dsio)
            T[:, jp] += QOPI * (psni * (dsio + dsim) + pdni * (dsio - dsim))

        # ---- полупанель 0–2
        dxinv = 1.0 / (x0 - x2)
        psum = x2 * (t2 - apan) - x0 * (t0 - apan) + 0.5 * yy * (g0 - g2)
        pdif = ((x0 + x2) * psum + rs0 * (t0 - apan) - rs2 * (t2 - apan) + (x2 - x0) * yy) * dxinv
        psx0 = -(t0 - apan)
        psx2 = t2 - apan
        psyy = 0.5 * (g0 - g2)
        pdx0 = ((x0 + x2) * psx0 + psum + 2.0 * x0 * (t0 - apan) - pdif) * dxinv
        pdx2 = ((x0 + x2) * psx2 + psum - 2.0 * x2 * (t2 - apan) + pdif) * dxinv
        pdyy = ((x0 + x2) * psyy + 2.0 * (x2 - x0 + yy * (t0 - t2))) * dxinv
        Psi[:, jo] += QOPI * (-psum * (dsip + dsio) - pdif * (dsip - dsio))
        Psi[:, jp] += QOPI * (psum * dsio - pdif * dsio)
        Psi[:, jq] += QOPI * (psum * dsip + pdif * dsip)
        for k, (nx, ny) in enumerate(dirs):
            xi_ = sx * nx + sy * ny
            yyi = sx * ny - sy * nx
            psni = psx0 * xi_ + psx2 * xi_ + psyy * yyi
            pdni = pdx0 * xi_ + pdx2 * xi_ + pdyy * yyi
            T = Px if k == 0 else Py
            T[:, jo] += QOPI * (-psni * (dsip + dsio) - pdni * (dsip - dsio))
            T[:, jp] += QOPI * (psni * dsio - pdni * dsio)
            T[:, jq] += QOPI * (psni * dsip + pdni * dsip)
    return Psi, Px, Py
