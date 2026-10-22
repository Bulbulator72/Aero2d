"""Неявный решатель: псевдонестационарный метод Ньютона–Крылова.

На каждом шаге решается
    (V/Δτ + ∂R/∂U) ΔU = −R(U),
где R — невязка второго порядка (из EulerSolver), её якобиан действует на вектор
точно и без матрицы — через jax.jvp (автоматическое дифференцирование).
Система решается GMRES с предобусловливателем: якобиан первого порядка
(блоки 4×4 по граням считает jax.jacfwd), обращаемый линейным методом Якоби —
блочная прогонка вдоль линий j (от стенки наружу, все линии i параллельно),
связь по i — итерациями. Шаг Δτ растёт по мере падения невязки (SER).

Почему так:
  * все операции — пакетные тензорные (vmap по случаям, параллельно по линиям):
    хорошо ложится на GPU;
  * линейная прогонка вдоль нормали к стенке — та же основа, что понадобится
    для RANS с ячейками y+ ~ 1 и удлинением 10^4;
  * точный якобиан через jvp даёт квадратичную сходимость вблизи решения.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from .euler import (EulerSolver, EulerResult, cons_to_prim, prim_to_cons, hllc_flux,
                    farfield_ghost, GAMMA)


def _flux_c(UL, UR, S):
    return hllc_flux(cons_to_prim(UL), cons_to_prim(UR), S)


def _face_jac(UL, UR, S):
    """Блоки ∂F/∂UL, ∂F/∂UR для массива граней. UL, UR: (4, n), S: (2, n)."""
    f = lambda a, b, s: _flux_c(a, b, s)
    jac = jax.vmap(jax.jacfwd(f, argnums=(0, 1)), in_axes=(1, 1, 1))
    AL, AR = jac(UL, UR, S)          # (n, 4, 4)
    return AL, AR


def _wall_jac(U0, S):
    def f(u, s):
        nx, ny = s / jnp.sqrt(s[0] ** 2 + s[1] ** 2)
        w = cons_to_prim(u)
        vn = w[1] * nx + w[2] * ny
        g = jnp.stack([w[0], w[1] - 2 * vn * nx, w[2] - 2 * vn * ny, w[3]])
        return hllc_flux(g, w, s)
    return jax.vmap(jax.jacfwd(f), in_axes=(1, 1))(U0, S)


def _far_jac(Ui, S, winf, xf, yf, circ):
    def f(u, s, x, y):
        w = cons_to_prim(u)
        wb = farfield_ghost(w, s, winf, x, y, circ)
        return hllc_flux(w, wb, s)
    return jax.vmap(jax.jacfwd(f), in_axes=(1, 1, 0, 0))(Ui, S, xf, yf)


def _bmv(A, x):
    """Блочное умножение: A (..., 4, 4), x (..., 4)."""
    return jnp.einsum("...ab,...b->...a", A, x)


class ImplicitEuler:
    def __init__(self, base: EulerSolver, cfl0: float = 10.0, cfl_max: float = 300.0,
                 gmres_restart: int = 50, gmres_tol: float = 0.01, jacobi_sweeps: int = 4,
                 max_update: float = 0.2):
        """cfl_max ≈ 300: при больших значениях в трансзвуке (M ≈ 0.85) Ньютон
        может «застрять» на нефизичном положении скачков; в дозвуке сходимость
        почти не зависит от cfl_max выше ~100."""
        self.b = base
        self.cfl0, self.cfl_max = cfl0, cfl_max
        self.restart, self.gtol = gmres_restart, gmres_tol
        self.sweeps = jacobi_sweeps
        self.max_update = max_update

    # ------------------------------------------------------------------
    def _assemble(self, U, winf, circ, dtau_inv):
        """Блоки якобиана первого порядка + диагональ V/Δτ.
        Возвращает D, Bim, Bip, Bjm, Bjp формы (ni, nc, 4, 4)."""
        b = self.b
        ni, nc = b.ni, b.nc
        dt = U.dtype
        # i-грани: грань k между ячейками k-1 и k
        UL = jnp.roll(U, 1, axis=1).reshape(4, -1)
        UR = U.reshape(4, -1)
        AL, AR = _face_jac(UL, UR, b.Si.reshape(2, -1))
        AL, AR = AL.reshape(ni, nc, 4, 4), AR.reshape(ni, nc, 4, 4)
        D = jnp.roll(AL, -1, axis=0) - AR
        Bip = jnp.roll(AR, -1, axis=0)
        Bim = -AL
        # внутренние j-грани m = 1..nc-1 между j=m-1 и m
        UL = U[:, :, :-1].reshape(4, -1)
        UR = U[:, :, 1:].reshape(4, -1)
        AL, AR = _face_jac(UL, UR, b.Sj[:, :, 1:-1].reshape(2, -1))
        AL, AR = AL.reshape(ni, nc - 1, 4, 4), AR.reshape(ni, nc - 1, 4, 4)
        z = jnp.zeros((ni, 1, 4, 4), dt)
        D = D + jnp.concatenate([AL, z], 1) - jnp.concatenate([z, AR], 1)
        Bjp = jnp.concatenate([AR, z], 1)
        Bjm = jnp.concatenate([z, -AL], 1)
        # стенка (j = 0) и внешняя граница (j = nc-1)
        Aw = _wall_jac(U[:, :, 0], b.Sj[:, :, 0])
        Af = _far_jac(U[:, :, -1], b.Sj[:, :, -1], winf, b.xf, b.yf, circ * b.vortex)
        D = D.at[:, 0].add(-Aw).at[:, -1].add(Af)
        eye = jnp.eye(4, dtype=dt)
        D = D + (b.vol * dtau_inv)[:, :, None, None] * eye
        return D, Bim, Bip, Bjm, Bjp

    @staticmethod
    def _factor_lines(D, Bjm, Bjp):
        """Блочная прогонка вдоль j: прямой ход (факторизация) для всех i сразу."""
        Dt = jnp.moveaxis(D, 1, 0)       # (nc, ni, 4, 4)
        Am = jnp.moveaxis(Bjm, 1, 0)
        Cp = jnp.moveaxis(Bjp, 1, 0)

        def fwd(cprev, x):
            Dj, Aj, Cj = x
            den = Dj - Aj @ cprev
            inv = jnp.linalg.inv(den)
            c = inv @ Cj
            return c, (inv, c)

        c0 = jnp.zeros_like(Dt[0])
        _, (inv, c) = jax.lax.scan(fwd, c0, (Dt, Am, Cp))
        return inv, c, Am                 # (nc, ni, 4, 4)

    @staticmethod
    def _solve_lines(fac, r):
        inv, c, Am = fac
        rt = jnp.transpose(r, (2, 1, 0))  # (nc, ni, 4)

        def fwd(dprev, x):
            invj, Aj, rj = x
            d = _bmv(invj, rj - _bmv(Aj, dprev))
            return d, d

        _, d = jax.lax.scan(fwd, jnp.zeros_like(rt[0]), (inv, Am, rt))

        def bwd(xnext, x):
            dj, cj = x
            xj = dj - _bmv(cj, xnext)
            return xj, xj

        _, xs = jax.lax.scan(bwd, jnp.zeros_like(rt[0]), (d, c), reverse=True)
        return jnp.transpose(xs, (2, 1, 0))  # (4, ni, nc)

    def _precond(self, fac, Bim, Bip, r):
        """Линейный Якоби: связь по i переносится в правую часть."""
        x = self._solve_lines(fac, r)
        for _ in range(self.sweeps - 1):
            xb = jnp.transpose(x, (1, 2, 0))       # (ni, nc, 4)
            off = _bmv(Bim, jnp.roll(xb, 1, 0)) + _bmv(Bip, jnp.roll(xb, -1, 0))
            x = self._solve_lines(fac, r - jnp.transpose(off, (2, 0, 1)))
        return x

    # ------------------------------------------------------------------
    def _circ(self, U, winf):
        """Циркуляция Γ = ½ Cl V∞ c из текущего решения (входит в невязку,
        поэтому связь «подъёмная сила ↔ поправка на вихрь» попадает в якобиан)."""
        cl = self.b.forces(U, winf)[0]
        return 0.5 * cl * jnp.sqrt(winf[1] ** 2 + winf[2] ** 2)

    def _res(self, U, winf):
        return self.b.residual(U, winf, self._circ(U, winf))

    def step(self, U, winf, circ, cfl):
        b = self.b
        circ = self._circ(U, winf)
        R, W = self._res(U, winf)
        dt_loc = b.local_dt(W) / b.cfl * cfl            # Δτ при заданном CFL
        dtau_inv = 1.0 / dt_loc
        D, Bim, Bip, Bjm, Bjp = self._assemble(U, winf, circ, dtau_inv)
        fac = self._factor_lines(D, Bjm, Bjp)
        diag = (b.vol * dtau_inv)[None]

        def matvec(v):
            _, jv = jax.jvp(lambda u: self._res(u, winf)[0], (U,), (v,))
            return diag * v + jv

        M = lambda r: self._precond(fac, Bim, Bip, r)
        dU, _ = gmres(matvec, -R, x0=M(-R), M=M, tol=self.gtol, atol=0.0,
                      restart=self.restart, maxiter=1, solve_method="batched")
        # ограничение шага по положительности плотности и давления
        Wn = cons_to_prim(U + dU)
        rel = jnp.maximum(jnp.max(jnp.abs(Wn[0] - W[0]) / W[0]), jnp.max(jnp.abs(Wn[3] - W[3]) / W[3]))
        omega = jnp.minimum(1.0, self.max_update / (rel + 1e-30))
        U = U + omega * dU
        rnorm = jnp.sqrt(jnp.mean((R[0] / b.vol) ** 2))
        return U, rnorm, omega

    def solve(self, machs, alphas, n_iter=200, tol=1e-8, verbose=True, U0=None, callback=None):
        """Пакетный расчёт. Случай, разошедшийся на шаге, откатывается и
        повторяется с меньшим CFL; после 5 неудач он «замораживается» и
        помечается как несошедшийся (остальные случаи пакета это не затрагивает).

        callback(it, n_iter, log10_res, cl) вызывается после каждой итерации;
        если он возвращает True, расчёт прерывается (результат — текущее состояние)."""
        b = self.b
        machs = jnp.asarray(machs, b.dtype)
        alphas = jnp.asarray(alphas, b.dtype)
        winf = jax.vmap(b.freestream)(machs, alphas)
        U = jax.vmap(b.init_state)(winf) if U0 is None else U0
        nb = len(machs)
        circ = jnp.zeros(nb, b.dtype)
        cfl = np.full(nb, self.cfl0)
        step = jax.jit(jax.vmap(self.step))
        forces = jax.jit(jax.vmap(b.forces))
        hist, clh = [], []
        r0 = None
        fails = np.zeros(nb, int)
        frozen = np.zeros(nb, bool)
        self.iterations = 0
        for it in range(n_iter):
            Unew, rn, om = step(U, winf, circ, jnp.asarray(cfl, b.dtype))
            rn, om = np.asarray(rn), np.asarray(om)
            finite = np.isfinite(rn) & np.asarray(jnp.all(jnp.isfinite(Unew), axis=(1, 2, 3)))
            bad = ~finite | frozen
            fails += ~finite
            frozen |= fails >= 5
            keep = jnp.asarray(bad)[:, None, None, None]
            U = jnp.where(keep, U, Unew)
            rn = np.where(finite, rn, np.nan)
            if r0 is None:
                r0 = np.where(np.isfinite(rn), np.maximum(rn, 1e-30), 1.0)
            cl = np.asarray(forces(U, winf)[0])
            # SER: CFL растёт обратно пропорционально невязке
            ser = self.cfl0 * (r0 / np.where(np.isfinite(rn), np.maximum(rn, 1e-30), r0))
            cfl = np.clip(ser, self.cfl0, self.cfl_max)
            cfl = np.where(om < 1.0, np.maximum(cfl * 0.5, 1.0), cfl)
            cfl = np.where(~finite, 1.0, cfl)
            hist.append(rn)
            clh.append(cl)
            self.iterations = it + 1
            if verbose and (it % 5 == 0 or it == n_iter - 1):
                print(f"  it {it + 1:4d}  log10(res/res0) {np.array2string(np.log10(rn / r0), precision=2)}"
                      f"  CFL {np.array2string(cfl, precision=0)}  Cl {np.array2string(cl, precision=4)}")
            done = (rn / r0 < tol) | frozen
            if callback is not None and callback(it + 1, n_iter, np.log10(rn / r0), cl):
                break
            if np.all(done):
                break
        self.cl_history = np.array(clh)
        self.frozen = frozen
        hist = np.array(hist)
        cl, cd, cm, cp = forces(U, winf)
        res = EulerResult(np.asarray(machs), np.asarray(alphas), np.asarray(cl), np.asarray(cd),
                          np.asarray(cm), np.asarray(cp), hist, U)
        last = np.array([h[np.isfinite(h)][-1] if np.any(np.isfinite(h)) else np.nan for h in hist.T])
        res.converged = (~frozen) & (np.log10(last / r0) < -3)
        return res
