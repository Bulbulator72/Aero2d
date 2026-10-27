"""GMRES с левым предобусловливанием (для неявного решателя RANS).

scipy.sparse.linalg.gmres хранит базис Крылова столбцами и на сетках в
десятки тысяч ячеек тратит на служебные операции больше, чем на сами
произведения якобиана на вектор. Здесь базис — список непрерывных векторов,
ортогонализация — классический Грам–Шмидт с повтором (CGS2: две матрично-
векторные операции BLAS вместо k скалярных произведений).
"""
from __future__ import annotations

import numpy as np


def gmres(matvec, b, prec=None, x0=None, rtol=1e-2, maxiter=60, callback=None, right=False):
    """Решение A x = b. matvec(v) = A v, prec(v) ≈ A⁻¹ v. Левое предобусловливание
    (по умолчанию) минимизирует ‖M⁻¹(b − A x)‖, правое (right=True) — истинную
    невязку ‖b − A x‖ (решается A M⁻¹ y = b, x = M⁻¹ y): при плохо обусловленном M
    левое может «сойтись» при большой истинной невязке. Без перезапусков:
    maxiter — размер базиса. callback(r) — относительная невязка на каждой итерации.
    Возвращает (x, info): info = 0 — достигнута точность rtol, иначе число итераций."""
    if right and prec is not None:
        if x0 is not None:
            r0 = b - matvec(x0)
            y, info = gmres(lambda v: matvec(prec(v)), r0, None, None, rtol, maxiter, callback)
            return x0 + prec(y), info
        y, info = gmres(lambda v: matvec(prec(v)), b, None, None, rtol, maxiter, callback)
        return prec(y), info
    P = prec if prec is not None else (lambda v: v)
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    r = P(b - matvec(x)) if x0 is not None else P(b)
    beta = float(np.linalg.norm(r))
    if beta == 0.0 or not np.isfinite(beta):
        return x, 0
    m = int(maxiter)
    V = np.empty((m + 1, n))
    V[0] = r / beta
    H = np.zeros((m + 1, m))
    cs, sn = np.zeros(m), np.zeros(m)
    g = np.zeros(m + 1)
    g[0] = beta
    k_done = 0
    info = m
    for j in range(m):
        w = P(matvec(V[j]))
        h = V[:j + 1] @ w
        w -= h @ V[:j + 1]
        h2 = V[:j + 1] @ w                      # повторная ортогонализация (CGS2)
        w -= h2 @ V[:j + 1]
        h += h2
        hn = float(np.linalg.norm(w))
        H[:j + 1, j] = h
        H[j + 1, j] = hn
        # вращения Гивенса
        for i in range(j):
            t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = t
        d = np.hypot(H[j, j], H[j + 1, j])
        if d == 0.0:
            k_done = j
            break
        cs[j], sn[j] = H[j, j] / d, H[j + 1, j] / d
        H[j, j] = d
        H[j + 1, j] = 0.0
        g[j + 1] = -sn[j] * g[j]
        g[j] = cs[j] * g[j]
        k_done = j + 1
        rel = abs(g[j + 1]) / beta
        if callback is not None:
            callback(rel)
        if not np.isfinite(rel):
            break
        if rel <= rtol:
            info = 0
            break
        if hn == 0.0:
            info = 0
            break
        V[j + 1] = w / hn
    k = k_done
    if k == 0:
        return x, info
    y = np.linalg.solve(np.triu(H[:k, :k]), g[:k]) if np.all(np.diag(H[:k, :k]) != 0) else \
        np.linalg.lstsq(H[:k, :k], g[:k], rcond=None)[0]
    x += y @ V[:k]
    return x, info
