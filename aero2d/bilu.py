"""Блочное неполное LU-разложение ILU(0) для разреженных матриц с блоками
k×k (предобусловливатель для неявного решателя RANS).

Строки обрабатываются «уровнями»: внутри уровня строки не зависят друг от
друга, поэтому разложение и треугольные решения векторизуются средствами
NumPy (число уровней ~ √N при упорядочении Катхилла–Макки).
"""
from __future__ import annotations

import numpy as np


class BlockILU0:
    def __init__(self, n, rows, cols, k=5, perm=None, fill=0):
        """Структура: n блочных строк, ненулевые блоки (rows[e], cols[e]) —
        без повторов, диагональ обязательна. perm — новый порядок строк
        (perm[новый] = старый); по умолчанию — обратный Катхилл–Макки.
        fill — уровень заполнения: 0 — ILU(0); 1, 2 — ILU(k): добавляются
        блоки, которые появились бы при исключении (больше памяти, сильнее
        предобусловливатель)."""
        import scipy.sparse as sp
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        rows, cols = np.asarray(rows), np.asarray(cols)
        if perm is None:
            G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
            perm = reverse_cuthill_mckee((G + G.T).tocsr(), symmetric_mode=True)
        inv = np.empty(n, int)
        inv[perm] = np.arange(n)
        self.n, self.k, self.perm, self.iperm = n, k, perm, inv
        r0, c0 = inv[rows], inv[cols]
        # шаблон с заполнением уровня fill (в новой нумерации)
        P = sp.csr_matrix((np.ones(len(r0)), (r0, c0)), shape=(n, n))
        for _ in range(int(fill)):
            Lp = sp.tril(P, -1).tocsr()
            Up = sp.triu(P, 1).tocsr()
            P = (P + Lp @ Up).tocsr()
        P = P.tocoo()
        r, c = P.row.astype(np.int64), P.col.astype(np.int64)
        order = np.lexsort((c, r))
        r, c = r[order], c[order]
        key = r * n + c
        # где в шаблоне стоят исходные блоки
        self.src_pos = np.searchsorted(key, r0.astype(np.int64) * n + c0)
        self.nnz_orig = len(r0)
        self.fill = int(fill)
        self.r, self.c = r, c
        nnz = len(r)
        self.diag = np.searchsorted(key, np.arange(n, dtype=np.int64) * (n + 1))
        low = np.where(c < r)[0]
        up = np.where(c > r)[0]
        start = np.searchsorted(r, np.arange(n))
        end = np.searchsorted(r, np.arange(n), side="right")
        # уровни для нижнего и верхнего треугольника (строки уровня независимы)
        lev = np.zeros(n, int)
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
        # ранг нижнего элемента в строке (номер по возрастанию столбца)
        rank = np.zeros(nnz, int)
        nlow_row = np.bincount(r[low], minlength=n)
        first_low = start                                    # нижние идут первыми в строке
        rank[low] = low - first_low[r[low]]
        # пары обновлений для каждого нижнего e=(i,k): (i,j) −= (i,k)(k,j), j > k — векторно
        nup = end - start - nlow_row - 1                     # верхних в строке (без диагонали)
        up_first = self.diag + 1                             # верхние — сразу после диагонали
        cnt = nup[c[low]]
        tot = int(cnt.sum())
        E = np.repeat(low, cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        kj = np.repeat(up_first[c[low]], cnt) + offs
        kij = r[E].astype(np.int64) * n + c[kj]
        loc = np.searchsorted(key, kij)
        loc = np.minimum(loc, nnz - 1)
        ok = key[loc] == kij
        pairs_e, pairs_ij, pairs_kj = E[ok], loc[ok], kj[ok]
        self.n_pairs = len(pairs_e)
        # расписание разложения: (уровень, ранг) — группировка сортировкой
        self.fact = []
        L = lev.max() + 1
        maxr = int(rank[low].max()) + 1 if len(low) else 1
        lev_e = lev[r]
        gl = lev_e[low] * maxr + rank[low]
        o_l = np.argsort(gl, kind="stable")
        gl_s, low_s = gl[o_l], low[o_l]
        gp = lev_e[pairs_e] * maxr + rank[pairs_e]
        o_p = np.argsort(gp, kind="stable")
        gp_s = gp[o_p]
        pe, pij, pkj = pairs_e[o_p], pairs_ij[o_p], pairs_kj[o_p]
        rows_by_lev = np.argsort(lev, kind="stable")
        lev_bounds = np.searchsorted(lev[rows_by_lev], np.arange(L + 1))
        for l_ in range(L):
            steps = []
            for p_ in range(maxr):
                g = l_ * maxr + p_
                a, b = np.searchsorted(gl_s, g), np.searchsorted(gl_s, g, side="right")
                if a == b:
                    continue
                a2, b2 = np.searchsorted(gp_s, g), np.searchsorted(gp_s, g, side="right")
                steps.append((low_s[a:b], pe[a2:b2], pij[a2:b2], pkj[a2:b2]))
            self.fact.append((steps, rows_by_lev[lev_bounds[l_]:lev_bounds[l_ + 1]]))
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
        B = np.zeros((len(self.r), self.k, self.k))
        B[self.src_pos] = np.asarray(blocks, float)
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
