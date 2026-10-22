"""Генерация структурированной O-сетки вокруг профиля.

Метод: гиперболический (маршевый) рост слоёв по сглаженным нормалям с
геометрическим нарастанием шага и сглаживанием вдоль слоя. Индекс i идёт вдоль
контура против часовой стрелки (периодический), j — от стенки наружу.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .geometry import Airfoil


@dataclass
class OGrid:
    x: np.ndarray   # (ni, nj) узлы
    y: np.ndarray

    @property
    def shape(self):
        return self.x.shape

    def cell_areas(self) -> np.ndarray:
        x, y = self.x, self.y
        x1, y1 = np.roll(x, -1, 0), np.roll(y, -1, 0)
        # диагонали ячейки (i,j)-(i+1,j+1) и (i+1,j)-(i,j+1)
        d1x, d1y = x1[:, 1:] - x[:, :-1], y1[:, 1:] - y[:, :-1]
        d2x, d2y = x[:, 1:] - x1[:, :-1], y[:, 1:] - y1[:, :-1]
        # обход (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1) идёт по часовой стрелке
        return -0.5 * (d1x * d2y - d1y * d2x)


def _growth_ratio(ds0, total, n):
    """Отношение r: ds0 * (r^n - 1)/(r - 1) = total."""
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        s = ds0 * (r**n - 1) / (r - 1)
        lo, hi = (r, hi) if s < total else (lo, r)
    return 0.5 * (lo + hi)


def _smooth_periodic(a, eps, passes):
    for _ in range(passes):
        a = a + eps * (np.roll(a, 1, 0) + np.roll(a, -1, 0) - 2 * a)
    return a


def ogrid(af: Airfoil, nj: int = 64, ds0: float = 2e-3, radius: float = 40.0,
          normal_smoothing: int = 20) -> OGrid:
    pts = af.coords()[:-1].astype(float)          # без повтора задней кромки
    ni = len(pts)
    center = np.array([0.5, 0.0])
    r = _growth_ratio(ds0, radius, nj - 1)
    d = ds0 * r ** np.arange(nj - 1)
    X = np.zeros((ni, nj, 2))
    X[:, 0] = pts
    for j in range(1, nj):
        cur = X[:, j - 1]
        t = np.roll(cur, -1, 0) - np.roll(cur, 1, 0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n = np.column_stack([t[:, 1], -t[:, 0]])
        # сглаживание нормалей сильнее у стенки, где слои тонкие
        n = _smooth_periodic(n, 0.25, normal_smoothing if j > 1 else 2)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        new = cur + d[j - 1] * n
        # сглаживание положения точек вдоль слоя (растёт к внешней границе)
        frac = j / (nj - 1)
        new = _smooth_periodic(new, 0.1 + 0.3 * frac, 2 + int(8 * frac))
        X[:, j] = new
    # внешний слой — точная окружность (полярные углы сохраняются)
    rel = X[:, -1] - center
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    Rmean = np.mean(np.linalg.norm(rel, axis=1))
    X[:, -1] = center + Rmean * np.column_stack([np.cos(ang), np.sin(ang)])
    g = OGrid(X[..., 0], X[..., 1])
    a = g.cell_areas()
    if a.min() <= 0:
        raise RuntimeError(f"Сетка с вывернутыми ячейками: min area {a.min():.3e}")
    return g
