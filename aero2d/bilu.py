"""Блочное неполное LU-разложение ILU(0) для разреженных матриц с блоками
k×k (предобусловливатель для неявного решателя RANS).

Строки обрабатываются «уровнями»: внутри уровня строки не зависят друг от
друга, поэтому разложение и треугольные решения векторизуются средствами
NumPy (число уровней ~ √N при упорядочении Катхилла–Макки).
"""
from __future__ import annotations

import numpy as np


class BlockILU0:
    def __init__(self, n, rows, cols, k=5, perm=None):
        """Структура: n блочных строк, ненулевые блоки (rows[e], cols[e]) —
        без повторов, диагональ обязательна. perm — новый порядок строк
        (perm[новый] = старый); по умолчанию — обратный Катхилл–Макки."""
        import scipy.sparse as sp
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        rows, cols = np.asarray(rows), np.asarray(cols)
        if perm is None:
            G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
            perm = reverse_cuthill_mckee((G + G.T).tocsr(), symmetric_mode=True)
        inv = np.empty(n, int)
        inv[perm] = np.arange(n)
        self.n, self.k, self.perm, self.iperm = n, k, perm, inv
        r, c = inv[rows], inv[cols]
        order = np.lexsort((c, r))
        self.src = order                          # исходный номер блока для каждой позиции
        r, c = r[order], c[order]
        self.r, self.c = r, c
        nnz = len(r)
        key = r.astype(np.int64) * n + c
        lookup = {int(kk): e for e, kk in enumerate(key)}
        self.diag = np.array([lookup[i * n + i] for i in range(n)])
        low = np.where(c < r)[0]
        up = np.where(c > r)[0]
        # уровни для нижнего треугольника
        lev = np.zeros(n, int)
        start = np.searchsorted(r, np.arange(n))
        end = np.searchsorted(r, np.arange(n), side="right")
        for i in range(n):
            cc = c[start[i]:end[i]]
            lo = cc[cc < i]
            if len(lo):
                lev[i] = lev[lo].max() + 1
        levu = np.zeros(n, int)
        for i in range(n - 1, -1, -1):
            cc = c[start[i]:end[i]]
            hi = cc[cc > i]
            if len(hi):
                levu[i] = levu[hi].max() + 1
        self.lev, self.levu = lev, levu
        # ранг нижнего элемента в строке
        rank = np.zeros(nnz, int)
        for i in range(n):
            ee = np.arange(start[i], end[i])
            ee = ee[c[ee] < i]
            rank[ee] = np.arange(len(ee))
        # пары обновлений для каждого нижнего e=(i,k): (i,j) −= (i,k)(k,j), j > k
        pairs_e, pairs_ij, pairs_kj = [], [], []
        for e in low:
            i, kk = r[e], c[e]
            for e2 in range(start[kk], end[kk]):
                j = c[e2]
                if j > kk:
                    eij = lookup.get(int(i * n + j))
                    if eij is not None:
                        pairs_e.append(e); pairs_ij.append(eij); pairs_kj.append(e2)
        pairs_e, pairs_ij, pairs_kj = (np.array(a, int) for a in (pairs_e, pairs_ij, pairs_kj))
        # расписание разложения: (уровень, ранг)
        self.fact = []
        L = lev.max() + 1
        maxr = rank[low].max() + 1 if len(low) else 0
        lev_e = lev[r]
        for l in range(L):
            steps = []
            for p in range(maxr):
                E = low[(lev_e[low] == l) & (rank[low] == p)]
                if len(E) == 0:
                    continue
                m = np.isin(pairs_e, E)
                steps.append((E, pairs_e[m], pairs_ij[m], pairs_kj[m]))
            rws = np.where(lev == l)[0]
            self.fact.append((steps, rws))
        # расписание решения
        self.fwd = []
        for l in range(L):
            rws = np.where(lev == l)[0]
            E = low[lev_e[low] == l]
            self.fwd.append((rws, E))
        self.bwd = []
        levu_e = levu[r]
        for l in range(levu.max() + 1):
            rws = np.where(levu == l)[0]
            E = up[levu_e[up] == l]
            self.bwd.append((rws, E))
        self.B = None

    def factor(self, blocks):
        """blocks — блоки в исходном порядке (как rows/cols конструктора)."""
        B = np.array(blocks, float)[self.src]
        invU = np.zeros((self.n, self.k, self.k))
        for steps, rws in self.fact:
            for E, pe, pij, pkj in steps:
                B[E] = np.einsum("eab,ebc->eac", B[E], invU[self.c[E]])
                if len(pe):
                    upd = np.einsum("eab,ebc->eac", B[pe], B[pkj])
                    np.subtract.at(B, pij, upd)
            invU[rws] = np.linalg.inv(B[self.diag[rws]])
        self.B, self.invU = B, invU
        return self

    def solve(self, b):
        """b — вектор (n·k,) в исходной нумерации."""
        k = self.k
        y = np.asarray(b, float).reshape(-1, k)[self.perm].copy()
        B, r, c = self.B, self.r, self.c
        for rws, E in self.fwd:
            if len(E):
                acc = np.einsum("eab,eb->ea", B[E], y[c[E]])
                np.subtract.at(y, r[E], acc)
        x = y
        for rws, E in self.bwd:
            if len(E):
                acc = np.einsum("eab,eb->ea", B[E], x[c[E]])
                np.subtract.at(x, r[E], acc)
            x[rws] = np.einsum("eab,eb->ea", self.invU[rws], x[rws])
        out = np.empty_like(x)
        out[self.perm] = x
        return out.ravel()
