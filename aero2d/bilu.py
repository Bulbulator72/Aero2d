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
        # расписание решения: внутри уровня элементы отсортированы по строке —
        # суммы по строкам через np.add.reduceat (быстрее np.subtract.at)
        def sched(levels, cand, n_lev):
            out = []
            le = levels[r[cand]]
            o = np.lexsort((r[cand], le))
            cs, ls = cand[o], le[o]
            bnd = np.searchsorted(ls, np.arange(n_lev + 1))
            for l in range(n_lev):
                rws = np.where(levels == l)[0]
                E = cs[bnd[l]:bnd[l + 1]]
                if len(E):
                    rr = r[E]
                    st = np.r_[0, np.where(np.diff(rr) != 0)[0] + 1]
                    out.append((rws, E, st, rr[st]))
                else:
                    out.append((rws, E, None, None))
            return out
        self.fwd = sched(lev, low, L)
        self.bwd = sched(levu, up, levu.max() + 1)
        self.B = None

    def factor(self, blocks):
        """blocks — блоки в исходном порядке (как rows/cols конструктора)."""
        B = np.zeros((len(self.r), self.k, self.k))
        B[self.src_pos] = np.asarray(blocks, float)
        invU = np.zeros((self.n, self.k, self.k))
        for steps, rws in self.fact:
            for E, pe, pij, pkj in steps:
                B[E] = np.matmul(B[E], invU[self.c[E]])
                if len(pe):
                    # внутри шага у каждой строки один исключаемый элемент — позиции
                    # (i, j) не повторяются, можно вычитать без np.subtract.at
                    B[pij] -= np.matmul(B[pe], B[pkj])
            invU[rws] = np.linalg.inv(B[self.diag[rws]])
        self.B, self.invU = B, invU
        return self

    def solve(self, b):
        """b — вектор (n·k,) в исходной нумерации."""
        k = self.k
        y = np.asarray(b, float).reshape(-1, k)[self.perm].copy()
        B, r, c = self.B, self.r, self.c
        for rws, E, st, ru in self.fwd:
            if len(E):
                acc = np.matmul(B[E], y[c[E]][:, :, None])[:, :, 0]
                y[ru] -= np.add.reduceat(acc, st, axis=0)
        x = y
        for rws, E, st, ru in self.bwd:
            if len(E):
                acc = np.matmul(B[E], x[c[E]][:, :, None])[:, :, 0]
                x[ru] -= np.add.reduceat(acc, st, axis=0)
            x[rws] = np.matmul(self.invU[rws], x[rws][:, :, None])[:, :, 0]
        out = np.empty_like(x)
        out[self.perm] = x
        return out.ravel()



def nested_dissection(n, rows, cols, leaf=200):
    """Упорядочение вложенными сечениями графа (rows, cols) с n вершинами:
    рекурсивное деление уровнями обхода в ширину от псевдопериферийной вершины,
    разделитель — средний уровень, нумеруется после обеих частей. Для точного LU
    двумерных сеток заполнение в разы меньше, чем у COLAMD/MMD (NACA 0012, 14 тыс.
    ячеек, блоки 5×5: разложение SuperLU за 2 с вместо десятков минут)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order, connected_components, shortest_path
    G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    G = ((G + G.T) > 0).astype(float).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()

    def rec(idx, out):
        if len(idx) <= leaf:
            out.append(idx)
            return
        sub = G[idx][:, idx]
        ncomp, lab = connected_components(sub, directed=False)
        if ncomp > 1:
            for k in range(ncomp):
                rec(idx[lab == k], out)
            return
        start = 0
        for _ in range(2):
            order = breadth_first_order(sub, start, directed=False, return_predecessors=False)
            start = order[-1]
        d = shortest_path(sub, method="D", unweighted=True, indices=start)
        m = np.floor(np.median(d))
        a_, b_ = idx[d < m], idx[d > m]
        if len(a_) == 0 or len(b_) == 0:
            out.append(idx)
            return
        rec(a_, out)
        rec(b_, out)
        out.append(idx[d == m])
    out = []
    rec(np.arange(n), out)
    return np.concatenate(out)


class NestedLU:
    """Точное разреженное LU (SuperLU) блочной матрицы n×n блоков k×k в порядке
    вложенных сечений (nested_dissection) — предобусловливатель «lu»."""

    def __init__(self, n, rows, cols, k=5):
        perm = nested_dissection(n, rows, cols)
        self.bperm = (perm[:, None] * k + np.arange(k)[None]).ravel()
        self.inv = np.empty_like(self.bperm)
        self.inv[self.bperm] = np.arange(len(self.bperm))
        self.lu = None

    def factor(self, A):
        """A — scipy.sparse (n·k × n·k) в исходной нумерации."""
        import scipy.sparse.linalg as spla
        Ap = A.tocsr()[self.bperm][:, self.bperm].tocsc()
        self.lu = spla.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        return self

    def solve(self, b):
        return self.lu.solve(np.asarray(b, float)[self.bperm])[self.inv]


class JaxTriSolve:
    """Прямой и обратный ход ILU (BlockILU0) одним скомпилированным циклом JAX
    (lax.scan по уровням; уровни дополнены до одинакового размера фиктивными
    элементами — нулевой блок и строка n). В numpy каждый уровень — несколько
    вызовов с накладными расходами Python; на сетке 35 тыс. ячеек с ILU(3)
    (667 уровней) это 0,2 с на применение, здесь — миллисекунды."""

    def __init__(self, ilu: BlockILU0):
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        self.ilu = ilu
        n, k, nnz = ilu.n, ilu.k, len(ilu.r)
        r, c = ilu.r, ilu.c

        def pad(groups, width_of, fill):
            P = max(1, max(width_of(g) for g in groups))
            return P, fill
        P = max(1, max(len(E) for _, E, _, _ in ilu.fwd))
        FE = np.full((len(ilu.fwd), P), nnz, np.int32)
        FR = np.full((len(ilu.fwd), P), n, np.int32)
        FC = np.full((len(ilu.fwd), P), n, np.int32)
        for l, (_, E, _, _) in enumerate(ilu.fwd):
            FE[l, :len(E)], FR[l, :len(E)], FC[l, :len(E)] = E, r[E], c[E]
        P = max(1, max(len(E) for _, E, _, _ in ilu.bwd))
        Pr = max(1, max(len(rw) for rw, _, _, _ in ilu.bwd))
        nb = len(ilu.bwd)
        BE = np.full((nb, P), nnz, np.int32)
        BR = np.full((nb, P), n, np.int32)
        BC = np.full((nb, P), n, np.int32)
        BW = np.full((nb, Pr), n, np.int32)
        for l, (rw, E, _, _) in enumerate(ilu.bwd):
            BE[l, :len(E)], BR[l, :len(E)], BC[l, :len(E)] = E, r[E], c[E]
            BW[l, :len(rw)] = rw
        self.idx = tuple(jnp.asarray(a) for a in (FE, FR, FC, BE, BR, BC, BW))
        perm = jnp.asarray(ilu.perm)

        def solve(B, invU, b, idx):
            FE, FR, FC, BE, BR, BC, BW = idx
            y = jnp.zeros((n + 1, k), b.dtype).at[:n].set(b.reshape(n, k)[perm])

            def fwd(y, ix):
                e, rr, cc = ix
                return y.at[rr].add(-jnp.einsum("eab,eb->ea", B[e], y[cc])), None
            y, _ = jax.lax.scan(fwd, y, (FE, FR, FC))

            def bwd(x, ix):
                e, rr, cc, rw = ix
                x = x.at[rr].add(-jnp.einsum("eab,eb->ea", B[e], x[cc]))
                return x.at[rw].set(jnp.einsum("eab,eb->ea", invU[rw], x[rw])), None
            x, _ = jax.lax.scan(bwd, y, (BE, BR, BC, BW))
            out = jnp.zeros((n, k), b.dtype).at[perm].set(x[:n])
            return out.reshape(-1)
        self._solve = jax.jit(solve)
        self._B = None

    def factor(self, blocks):
        """Разложение (numpy, BlockILU0.factor) и перенос множителей в JAX."""
        import jax.numpy as jnp
        self.ilu.factor(blocks)
        k = self.ilu.k
        self._B = jnp.asarray(np.concatenate([self.ilu.B, np.zeros((1, k, k))]))
        self._invU = jnp.asarray(np.concatenate([self.ilu.invU, np.zeros((1, k, k))]))
        self.invU = self.ilu.invU
        return self

    def solve(self, b):
        return np.array(self._solve(self._B, self._invU, np.asarray(b, float), self.idx))
