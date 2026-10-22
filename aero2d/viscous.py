"""Вязко-невязкий расчёт профиля: панельный метод + интегральный пограничный
слой со строгой связью (как XFOIL, Drela 1989).

Схема:
  * невязкое течение — панели с линейной завихренностью (panel.py);
  * влияние вытеснения — источники на контуре и следе, интенсивность
    σ = d(Ue δ*)/dξ; скорость на границе слоя Ue = Ue_inv + D·m, m = Ue δ*;
  * пограничный слой — интегральные уравнения импульса и кинетической
    энергии, ламинарный переход по eᴺ (Ñ), турбулентный — уравнение
    запаздывания для sqrt(Cτ); след — турбулентный;
  * все уравнения (слой + связь) решаются одновременно методом Ньютона;
    якобиан — комплексным шагом (точно, без ручного дифференцирования).

Неизвестные на станции: θ, m = Ue δ*, Ñ, S = sqrt(Cτ).
Обезразмеривание: хорда 1, V∞ = 1, ν = 1/Re.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import blclosure as bc
from .geometry import Airfoil
from .panel import (PanelSolver, panel_geometry, source_influence,
                    vortex_velocity_influence, karman_tsien)

CSTEP = 1e-30          # комплексный шаг


@dataclass
class ViscousResult:
    alpha: float
    re: float
    cl: float
    cd: float
    cdf: float
    cdp: float
    cm: float
    xtr_u: float
    xtr_l: float
    converged: bool
    iterations: int
    x: np.ndarray            # узлы профиля
    y: np.ndarray
    cp: np.ndarray           # вязкий Cp в узлах
    cp_inv: np.ndarray
    ue: np.ndarray           # скорость на границе слоя в узлах (со знаком обхода)
    side: dict = field(default_factory=dict)   # распределения по сторонам и следу
    state: object = None     # для продолжения по α


def _hiemenz_like(ue_slope, re):
    return np.sqrt(0.075 / (re * max(ue_slope, 1e-6)))


class ViscousSolver:
    def __init__(self, airfoil: Airfoil, re: float, ncrit: float = 9.0, mach: float = 0.0,
                 xtr=(1.0, 1.0), wake_length: float = 1.0, n_wake: int | None = None,
                 n_panels: int = 160):
        from .geometry import repanel
        airfoil = repanel(airfoil, n_panels // 2 + 1, spacing="viscous")
        self.ps = PanelSolver(airfoil)
        self.af = airfoil
        self.re = float(re)
        self.ncrit = float(ncrit)
        self.mach = float(mach)
        self.xtr = tuple(float(v) for v in xtr)
        x, y = self.ps.x, self.ps.y
        self.x, self.y = x, y
        self.N = self.ps.N
        self.s = np.concatenate([[0.0], np.cumsum(self.ps.L)])     # длина дуги узлов
        self.wake_length = wake_length
        self.n_wake = n_wake or max(self.N // 8 + 2, 12)
        # влияние источников на контуре в узлах (не зависит от α)
        self.Bpsi_air, _, _ = source_influence(x, y, x[:-1], y[:-1])

    # ------------------------------------------------------------------
    # невязкие величины, след, матрица влияния
    # ------------------------------------------------------------------
    def _wake(self, g):
        """Линия следа по линиям тока невязкого течения от задней кромки."""
        x, y, L = self.x, self.y, self.ps.L
        n = self.n_wake
        ds0 = 0.5 * (L[0] + L[-1])
        # геометрическая прогрессия шага до длины следа
        lo, hi = 1.0, 1.5
        for _ in range(100):
            r = 0.5 * (lo + hi)
            tot = ds0 * (r ** (n - 1) - 1) / (r - 1)
            lo, hi = (r, hi) if tot < self.wake_length else (lo, r)
        ds = ds0 * r ** np.arange(n - 1)
        # направление — биссектриса задней кромки
        t1 = np.array([x[0] - x[1], y[0] - y[1]])
        t2 = np.array([x[-1] - x[-2], y[-1] - y[-2]])
        d = t1 / np.linalg.norm(t1) + t2 / np.linalg.norm(t2)
        d /= np.linalg.norm(d)
        a = np.radians(self._alpha)
        pts = [np.array([x[0], y[0]])]
        for j in range(n - 1):
            p = pts[-1]
            if j == 0:
                v = d
            else:
                U, V = vortex_velocity_influence(x, y, np.array([p[0]]), np.array([p[1]]))
                v = np.array([U[0] @ g + np.cos(a), V[0] @ g + np.sin(a)])
                v /= np.linalg.norm(v)
                # полушаг (RK2)
                q = p + 0.5 * ds[j] * v
                U, V = vortex_velocity_influence(x, y, np.array([q[0]]), np.array([q[1]]))
                v = np.array([U[0] @ g + np.cos(a), V[0] @ g + np.sin(a)])
                v /= np.linalg.norm(v)
            pts.append(p + ds[j] * v)
        pts = np.array(pts)
        return pts[:, 0], pts[:, 1]

    def _topology(self, g):
        """Точка торможения и станции слоя по текущему распределению γ."""
        N = self.N
        # верх: γ > 0 (узлы 0..k), низ: γ < 0
        idx = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
        if len(idx) == 0:
            raise RuntimeError("Не найдена точка торможения")
        k = idx[np.argmin(np.abs(idx - N // 2))]
        f = g[k] / (g[k] - g[k + 1])
        s_st = self.s[k] + f * self.ps.L[k]
        up = np.arange(k, -1, -1)            # от торможения к задней кромке
        lo = np.arange(k + 1, N + 1)
        # узел, почти совпадающий с точкой торможения, исключается из слоя
        # (m = 0 в нём); иначе первая станция вырождается (ξ → 0, Ue → 0)
        if f < 0.1:
            up = up[1:]
        elif f > 0.9:
            lo = lo[1:]
        return k, s_st, up, lo

    def _build(self, alpha, g_inv):
        """Всё, что зависит от α и топологии: след, станции, ue_inv, D."""
        self._alpha = alpha
        N = self.N
        x, y = self.x, self.y
        k, s_st, up, lo = self._topology(g_inv if self._g_cur is None else self._g_cur)
        self.k, self.s_st = k, s_st
        self.sig = (k, len(up), len(lo))
        if getattr(self, "_wake_alpha", None) != alpha:
            self.wx, self.wy = self._wake(g_inv)
            self._wake_alpha = alpha
            self._wake_cache = None
        wx, wy = self.wx, self.wy
        nw = len(wx)
        nu, nl = len(up), len(lo)
        Ns = nu + nl + nw
        self.nu, self.nl, self.nw, self.Ns = nu, nl, nw, Ns
        self.up, self.lo = up, lo
        # знак Ue на узлах: верх +γ, низ −γ
        sg_node = np.where(np.arange(N + 1) <= k, 1.0, -1.0)
        self.sg_node = sg_node
        # станция → узел (для профиля), -1 для следа
        node = np.concatenate([up, lo, -np.ones(nw, int)])
        self.node = node
        self.is_wake = np.concatenate([np.zeros(nu + nl, bool), np.ones(nw, bool)])
        self.st_up = np.arange(nu)
        self.st_lo = nu + np.arange(nl)
        self.st_wk = nu + nl + np.arange(nw)
        # ξ
        xi = np.empty(Ns)
        xi[self.st_up] = s_st - self.s[up]
        xi[self.st_lo] = self.s[lo] - s_st
        wl = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(wx), np.diff(wy)))])
        xi[self.st_wk] = xi[self.st_lo[-1]] + wl
        self.xi = np.maximum(xi, 1e-9)
        self.xc = np.concatenate([x[up], x[lo], wx])        # для принудительного перехода
        # --- источники: σ (панели профиля + следа) = S @ m(станции)
        npa, npw = N, nw - 1
        S = np.zeros((npa + npw, Ns))
        st_of_node = -np.ones(N + 1, int)
        st_of_node[up] = self.st_up
        st_of_node[lo] = self.st_lo
        self.st_of_node = st_of_node
        sgm = np.where(np.arange(N + 1) <= k, -1.0, 1.0)     # геометрический знак m
        for p in range(N):
            if st_of_node[p + 1] >= 0:
                S[p, st_of_node[p + 1]] += sgm[p + 1] / self.ps.L[p]
            if st_of_node[p] >= 0:
                S[p, st_of_node[p]] -= sgm[p] / self.ps.L[p]
        Lw = np.hypot(np.diff(wx), np.diff(wy))
        for j in range(npw):
            S[npa + j, self.st_wk[j + 1]] += 1.0 / Lw[j]
            S[npa + j, self.st_wk[j]] -= 1.0 / Lw[j]
        # функция тока источников следа в узлах профиля
        if self._wake_cache is None:
            Bpsi_w, _, _ = source_influence(wx, wy, x[:-1], y[:-1])
            pairs = [(j, j - 1 if j > 0 else None, j if j < npw else None) for j in range(1, nw)]
            # скорости в узлах следа (кроме нулевого) от всех источников и вихрей
            _, Ua, Va = source_influence(x, y, wx[1:], wy[1:])
            pairs_local = [(j - 1, a, b) for (j, a, b) in pairs]
            _, Uw, Vw = source_influence(wx, wy, wx[1:], wy[1:], self_pairs=pairs_local)
            Uv, Vv = vortex_velocity_influence(x, y, wx[1:], wy[1:])
            # касательная следа в узлах
            tw = np.zeros((nw, 2))
            dw = np.column_stack([np.diff(wx), np.diff(wy)]) / Lw[:, None]
            tw[0] = dw[0]
            tw[-1] = dw[-1]
            tw[1:-1] = dw[:-1] + dw[1:]
            tw /= np.linalg.norm(tw, axis=1, keepdims=True)
            self._wake_cache = (Bpsi_w, Ua, Va, Uw, Vw, Uv, Vv, tw)
        Bpsi_w, Ua, Va, Uw, Vw, Uv, Vv, tw = self._wake_cache
        Bpsi = np.hstack([self.Bpsi_air, Bpsi_w])            # N x (npa+npw)
        Ainv = self.ps.Ainv
        dg_dsig = -Ainv[:N + 1, :N] @ Bpsi                    # (N+1) x npanels
        dg_dm = dg_dsig @ S                                   # (N+1) x Ns
        self.dg_dm = dg_dm
        D = np.zeros((Ns, Ns))
        D[:nu + nl] = (sg_node[node[:nu + nl]])[:, None] * dg_dm[node[:nu + nl]]
        # след: скорость от вихрей (через dγ) и от всех источников
        Uall = np.hstack([Ua, Uw]) @ S + Uv @ dg_dm
        Vall = np.hstack([Va, Vw]) @ S + Vv @ dg_dm
        D[self.st_wk[1:]] = tw[1:, 0:1] * Uall + tw[1:, 1:2] * Vall
        D[self.st_wk[0]] = 0.5 * (D[self.st_up[-1]] + D[self.st_lo[-1]])
        self.D = D * getattr(self, "couple", 1.0)
        # невязкая скорость на станциях
        a = np.radians(alpha)
        ue0 = np.empty(Ns)
        ue0[:nu + nl] = sg_node[node[:nu + nl]] * g_inv[node[:nu + nl]]
        uw = tw[1:, 0] * (Uv @ g_inv + np.cos(a)) + tw[1:, 1] * (Vv @ g_inv + np.sin(a))
        ue0[self.st_wk[1:]] = uw
        ue0[self.st_wk[0]] = 0.5 * (ue0[self.st_up[-1]] + ue0[self.st_lo[-1]])
        self.ue_inv = ue0
        # ξ = ±(s − s_st): знак производной по положению точки торможения
        self.xi_sign = np.where(np.arange(Ns) < nu, 1.0, -1.0)
        self._set_dsst(g_inv if self._g_cur is None else self._g_cur)

    def _set_dsst(self, g):
        """ds_st/dm: точка торможения — нуль линейной интерполяции γ между узлами k, k+1."""
        k = self.k
        gk, gk1 = g[k], g[k + 1]
        d = gk - gk1
        self.dsst_dm = self.ps.L[k] * (-gk1 * self.dg_dm[k] + gk * self.dg_dm[k + 1]) / d**2

    # ------------------------------------------------------------------
    # начальное приближение
    # ------------------------------------------------------------------
    def _initial(self):
        re, Ns = self.re, self.Ns
        X = np.zeros((Ns, 4))
        ue = np.maximum(self.ue_inv, 1e-3)
        for side, st in (("u", self.st_up), ("l", self.st_lo)):
            xi = self.xi[st]
            u = ue[st]
            xtr_f = self.xtr[0] if side == "u" else self.xtr[1]
            th = np.zeros(len(st))
            h = np.zeros(len(st))
            n = np.zeros(len(st))
            turb = False
            integ = u[0] ** 5 * xi[0] / 6.0           # ∫ ue^5 dξ (линейный старт)
            th[0] = np.sqrt(0.45 / re * integ / u[0] ** 6)
            h[0] = 2.24
            for i in range(1, len(st)):
                dx = xi[i] - xi[i - 1]
                if not turb:
                    integ += 0.5 * (u[i] ** 5 + u[i - 1] ** 5) * dx
                    th[i] = np.sqrt(0.45 / re * integ / u[i] ** 6)
                    lam = th[i] ** 2 * re * (u[i] - u[i - 1]) / dx
                    if lam < -0.09 or self.xc[st[i]] >= xtr_f:
                        turb = True
                        h[i] = 2.5
                        n[i] = self.ncrit + 1.0
                        continue
                    h[i] = (2.61 - 3.75 * lam + 5.24 * lam**2) if lam >= 0 else 2.088 + 0.0731 / (lam + 0.14)
                    ax = np.real(bc.dampl(np.array(h[i]), np.array(th[i]), np.array(max(re * u[i] * th[i], 1.0))))
                    n[i] = n[i - 1] + ax * dx
                    if n[i] >= self.ncrit:
                        turb = True
                else:
                    hh = 1.5
                    rt = max(re * u[i] * th[i - 1], 10.0)
                    cf = float(np.real(bc.cft(np.array(hh), np.array(rt), 0.0)))
                    dlnu = np.log(u[i] / u[i - 1])
                    # неявная схема: θ_i (1 + (H+2) Δln u) = θ_{i-1} + cf/2 dx
                    th[i] = (th[i - 1] + 0.5 * cf * dx) / max(1 + (hh + 2) * dlnu, 0.3)
                    h[i] = hh
                    n[i] = max(n[i - 1], self.ncrit + 1.0) + 20.0 * dx
            ds = h * th
            X[st, 0] = th
            X[st, 1] = u * ds
            X[st, 2] = n
            X[st, 3] = 0.03
        # след
        tu, tl = self.st_up[-1], self.st_lo[-1]
        th0 = X[tu, 0] + X[tl, 0]
        ds0 = X[tu, 1] / ue[tu] + X[tl, 1] / ue[tl]
        wl = self.xi[self.st_wk] - self.xi[self.st_wk[0]]
        h0 = ds0 / th0
        # θ в следе почти постоянна, H плавно спадает к 1; m непрерывна на кромке
        for j, sti in enumerate(self.st_wk):
            hh = 1.0 + (h0 - 1.0) * np.exp(-wl[j] / 0.3)
            uu = max(ue[sti], ue[self.st_wk[0]])
            X[sti] = [th0, uu * hh * th0, self.ncrit + 1.0, 0.03]
        # S — равновесное значение
        st = self._stations(X, self.ue_inv)
        turb = self._turb_flags(X)
        X[:, 3] = np.where(turb, np.real(st["cq"]), np.real(bc.s_transition(st["hk"], st["cq"])))
        return X

    # ------------------------------------------------------------------
    # маршевое начальное приближение (как MRCHUE в XFOIL)
    # ------------------------------------------------------------------
    @staticmethod
    def _local_newton(fun, v, unk, extra_eq=None, iters=25):
        """Решает fun(v) = 0 по компонентам v[unk]. fun векторизована по
        столбцам: принимает (nv, k) и возвращает (4, k), поэтому базовая точка и
        все комплексные возмущения считаются за один вызов."""
        v = v.astype(float).copy()
        nq = len(unk)

        def F(Vm):
            r = fun(Vm)
            if extra_eq is not None:
                r = np.vstack([r, extra_eq(Vm)[None, :]])
            return r

        r0 = None
        for _ in range(iters):
            Vm = np.repeat(v[:, None].astype(complex), nq + 1, axis=1)
            hs = np.maximum(np.abs(v[unk]), 1e-12) * CSTEP
            for q, iq in enumerate(unk):
                Vm[iq, q + 1] += 1j * hs[q]
            Fm = F(Vm)
            r0 = np.real(Fm[:, 0])
            if not np.all(np.isfinite(r0)):
                return v, False
            Jm = np.imag(Fm[:, 1:]) / hs[None, :]
            try:
                dv = np.linalg.solve(Jm, -r0)
            except np.linalg.LinAlgError:
                return v, False
            rl = 1.0
            for q, iq in enumerate(unk):
                if iq % 6 in (0, 1, 3, 4) and v[iq] != 0:
                    rr = dv[q] / abs(v[iq])
                    if rr < -0.5:
                        rl = min(rl, -0.5 / rr)
                    if rr > 1.5:
                        rl = min(rl, 1.5 / rr)
            v[unk] += rl * dv
            if np.max(np.abs(dv) / np.maximum(np.abs(v[unk]), 1e-10)) < 1e-9:
                return v, True
        return v, bool(r0 is not None and np.max(np.abs(r0)) < 1e-6)

    def _march(self):
        """Пограничный слой при заданной невязкой скорости; где слой отрывается
        (H больше предела) — обратный режим: H задано, Ue ищется.
        Вектор отрезка: [θ,m,Ñ,S,ue,ξ]_a + [θ,m,Ñ,S,ue,ξ]_b (индексы 0–5, 6–11)."""
        X0 = self._initial()
        X = X0.copy()
        ue = np.maximum(self.ue_inv.copy(), 1e-4)
        hmax_l, hmax_t = 3.8, 2.5
        TH, M, N_, S_, U = 6, 7, 8, 9, 10           # индексы станции b

        def solve_b(fint, v, hmax):
            vd, ok = self._local_newton(fint, v, [TH, M, N_, S_])
            hk = vd[M] / vd[U] / vd[TH]
            if ok and 1.0 < hk <= hmax:
                return vd
            eq = lambda Vm, hm=hmax: Vm[M] / Vm[U] / Vm[TH] - hm
            v = v.copy()
            v[M] = hmax * v[TH] * v[U]
            vi, ok2 = self._local_newton(fint, v, [TH, M, N_, S_, U], extra_eq=eq)
            return vi if ok2 else vd

        for side, st in (("u", self.st_up), ("l", self.st_lo)):
            xtr_f = self.xtr[0] if side == "u" else self.xtr[1]
            i0 = st[0]
            v = np.concatenate([X[i0], [ue[i0], self.xi[i0]]])
            v, ok = self._local_newton(lambda Vm: self._sim_res(Vm), v, [0, 1, 2, 3])
            X[i0] = v[:4]
            for a, b in zip(st[:-1], st[1:]):
                extra = (self.xc[[a]], self.xc[[b]], np.array([False]), np.array([xtr_f]))
                fint = lambda Vm, extra=extra: self._interval_res(Vm, *extra)
                guess = X[a].copy()
                guess[2] = max(X[a, 2], X0[b, 2]) if X[a, 2] < self.ncrit else X[a, 2]
                guess[1] = ue[b] * (X[a, 1] / ue[a])
                v = np.concatenate([X[a], [ue[a], self.xi[a]], guess, [ue[b], self.xi[b]]])
                turb_b = guess[2] >= self.ncrit or self.xc[b] >= xtr_f
                v = solve_b(fint, v, hmax_t if turb_b else hmax_l)
                X[b] = v[TH:TH + 4]
                ue[b] = v[U]
        # след
        tu, tl, w0 = self.st_up[-1], self.st_lo[-1], self.st_wk[0]
        ue[w0] = 0.5 * (ue[tu] + ue[tl])
        th0 = X[tu, 0] + X[tl, 0]
        ds0 = X[tu, 1] / ue[tu] + X[tl, 1] / ue[tl]
        s0 = (X[tu, 3] * X[tu, 0] + X[tl, 3] * X[tl, 0]) / th0
        X[w0] = [th0, ue[w0] * ds0, self.ncrit + 1.0, s0]
        wk = np.ones(1, bool)
        for a, b in zip(self.st_wk[:-1], self.st_wk[1:]):
            extra = (self.xc[[a]], self.xc[[b]], wk, np.array([1.0]))
            fint = lambda Vm, extra=extra: self._interval_res(Vm, *extra)
            guess = X[a].copy()
            guess[1] = ue[b] * X[a, 1] / ue[a]
            v = np.concatenate([X[a], [ue[a], self.xi[a]], guess, [ue[b], self.xi[b]]])
            v = solve_b(fint, v, hmax_t)
            X[b] = v[TH:TH + 4]
            ue[b] = v[U]
        self._ue_march = ue
        return X

    # ------------------------------------------------------------------
    # невязки
    # ------------------------------------------------------------------
    def _turb_flags(self, X):
        n = np.real(X[:, 2])
        turb = (n >= self.ncrit) | self.is_wake
        xtr = np.where(np.isin(np.arange(self.Ns), self.st_up), self.xtr[0], self.xtr[1])
        turb |= (self.xc >= xtr) & ~self.is_wake
        return turb

    def _stations(self, X, ue, turb=None):
        if turb is None:
            turb = self._turb_flags(X)
        th, m, s = X[:, 0], X[:, 1], X[:, 3]
        ds = m / ue
        msq = self.mach**2 + 0 * th
        return bc.station(th, ds, ue, s, self.re, msq, turb, self.is_wake)

    def _segment(self, A, B, xa, xb, wake, turb):
        """Невязки импульса и формпараметра на отрезке a→b (массивы станций A, B —
        словари closure), плюс уравнение запаздывания (для turb)."""
        ulog = np.log(B["ue"] / A["ue"])
        tlog = np.log(B["th"] / A["th"])
        xlog = np.log(xb / xa)
        hlog = np.log(B["hs"] / A["hs"])
        ha = 0.5 * (A["h"] + B["h"])
        msq = self.mach**2
        # Cf в середине отрезка
        thm, dsm, uem = 0.5 * (A["th"] + B["th"]), 0.5 * (A["ds"] + B["ds"]), 0.5 * (A["ue"] + B["ue"])
        mid = bc.station(thm, dsm, uem, 0.5 * (A["s"] + B["s"]), self.re, msq + 0 * thm,
                         turb, wake)
        xa_t, xb_t = xa / A["th"], xb / B["th"]
        cfx = 0.5 * mid["cf"] * (0.5 * (xa + xb)) / thm + 0.25 * (A["cf"] * xa_t + B["cf"] * xb_t)
        rez_t = tlog + (ha + 2.0 - msq) * ulog - xlog * 0.5 * cfx
        # формпараметр с противопоточным весом
        hl = np.log(B["hk"] / A["hk"])
        hlsq = bc._min(hl**2, 15.0 + 0 * hl)
        upw = 1.0 - 0.5 * np.exp(-hlsq * 5.0 / B["hk"] ** 2)
        dix = (1 - upw) * A["di"] * xa_t + upw * B["di"] * xb_t
        cfx2 = (1 - upw) * A["cf"] * xa_t + upw * B["cf"] * xb_t
        hsa, hca = 0.5 * (A["hs"] + B["hs"]), 0.5 * (A["hc"] + B["hc"])
        btmp = 2.0 * hca / hsa + 1.0 - ha
        rez_h = hlog + btmp * ulog + xlog * (0.5 * cfx2 - dix)
        # запаздывание
        ald = np.where(wake, bc.DLCON, 1.0)
        hka = 0.5 * (A["hk"] + B["hk"])
        rta = 0.5 * (A["rt"] + B["rt"])
        da = 0.5 * (A["ds"] + B["ds"])
        cfa = 0.5 * (A["cf"] + B["cf"])
        gcc = np.where(wake, 0.0, bc.GCCON)
        hkc = bc._max(hka - 1.0 - gcc / rta, 0.01 + 0 * hka)
        hr = hkc / (bc.GACON * ald * hka)
        uq = (0.5 * cfa - hr**2) / (bc.GBCON * da)
        usa = 0.5 * (A["us"] + B["us"])
        scc = bc.SCCON * 1.333 / (1.0 + usa)
        sa = (1 - upw) * A["s"] + upw * B["s"]
        cqa = (1 - upw) * A["cq"] + upw * B["cq"]
        dea = 0.5 * (A["de"] + B["de"])
        dxi = xb - xa
        slog = np.log(B["s"] / A["s"])
        rez_c = scc * (cqa - sa * ald) * dxi - dea * 2.0 * slog + dea * 2.0 * (uq * dxi - ulog)
        return rez_t, rez_h, rez_c

    def _interval_res5(self, V, xa, xb, xca, xcb, wake, xtr_f):
        """То же, что _interval_res, но ξ передаются отдельно (для марша)."""
        V6 = np.concatenate([V[:5], xa.reshape(1, -1).astype(V.dtype), V[5:10],
                             xb.reshape(1, -1).astype(V.dtype)])
        return self._interval_res(V6, xca, xcb, wake, xtr_f)

    def _interval_res(self, V, xca, xcb, wake, xtr_f):
        """Невязки для набора отрезков. V: (12, n) — [θ,m,Ñ,S,ue,ξ] в начале и
        конце. ξ — тоже вход: от него зависят все невязки, а сама ξ зависит от
        положения точки торможения. Возвращает (4, n)."""
        th1, m1, n1, s1, u1, xa, th2, m2, n2, s2, u2, xb = V
        nc = self.ncrit
        msq = self.mach**2 + 0 * th1
        n1r, n2r = np.real(n1), np.real(n2)
        F = np.zeros(1, bool)
        A_l = bc.station(th1, m1 / u1, u1, s1, self.re, msq, F, wake)
        B_l = bc.station(th2, m2 / u2, u2, s2, self.re, msq, F, wake)
        # тип отрезка (по вещественным частям). Переходный — если ламинарная
        # экстраполяция Ñ на конец отрезка достигает Ñкрит; решение не зависит
        # от неизвестной Ñ₂, поэтому области «ламинарный»/«переходный» не
        # перекрываются (иначе возможны два решения).
        phi1 = np.real(n1 + 0.5 * (A_l["ax"] + B_l["ax"]) * (xb - xa))
        forced = (xcb >= xtr_f) & ~wake
        turb_start = wake | (n1r >= nc) | ((xca >= xtr_f) & ~wake)
        lam_end = ~turb_start & ~forced & (phi1 < nc)
        trans = ~lam_end & ~turb_start
        A_t = bc.station(th1, m1 / u1, u1, s1, self.re, msq, ~F, wake)
        B_t = bc.station(th2, m2 / u2, u2, s2, self.re, msq, ~F, wake)
        # --- ламинарный отрезок
        rt_l, rh_l, _ = self._segment(A_l, B_l, xa, xb, wake, F)
        ra_l = n2 - n1 - 0.5 * (A_l["ax"] + B_l["ax"]) * (xb - xa)
        rs_l = s2 - bc.s_transition(B_l["hk"], B_l["cq"])
        # --- турбулентный отрезок
        rt_t, rh_t, rc_t = self._segment(A_t, B_t, xa, xb, wake, ~F)
        # за переходом Ñ продолжает медленно расти: если точка перехода сдвинется
        # вниз по потоку, она окажется в следующем отрезке, а не «выключит»
        # турбулентность во всей области за ней
        ra_t = np.where(wake, n2 - n1, n2 - n1 - 20.0 * (xb - xa))
        # --- переходный отрезок: точка перехода внутри
        # доля ламинарного участка по Ñ и по принудительному x
        dn = n2 - n1
        dn = np.where(np.abs(np.real(dn)) < 1e-12, 1e-12 + 0 * dn, dn)
        fn = (nc - n1) / dn
        fn = np.where((n2r >= nc) & (phi1 >= nc), fn, 1.0 + 0 * fn)
        fx = np.where(forced, (xtr_f - xca) / np.where(np.abs(xcb - xca) < 1e-12, 1e-12, xcb - xca), 1.0)
        f = bc._min(fn, fx + 0 * fn)
        f = bc._max(bc._min(f, 1.0 + 0 * f), 0.0 + 0 * f)
        ft = f          # f → 1 и f → 0 непрерывно переходят в ламинарный / турбулентный отрезок
        tht = th1 + ft * (th2 - th1)
        dst = m1 / u1 + ft * (m2 / u2 - m1 / u1)
        uet = u1 + ft * (u2 - u1)
        xt = xa + ft * (xb - xa)
        T_l = bc.station(tht, dst, uet, s1, self.re, msq, F, wake)
        T_t0 = bc.station(tht, dst, uet, s1, self.re, msq, ~F, wake)
        st_tr = bc.s_transition(T_t0["hk"], T_t0["cq"])
        T_t = bc.station(tht, dst, uet, st_tr, self.re, msq, ~F, wake)
        r1t, r1h, _ = self._segment(A_l, T_l, xa, xt, wake, F)
        r2t, r2h, r2c = self._segment(T_t, B_t, xt, xb, wake, ~F)
        # Ñ: при естественном переходе — до точки перехода (по ней же и Ñ = Ñкрит),
        # при принудительном — обычное уравнение роста по всему отрезку
        ra_tr = np.where(forced & (phi1 < nc),
                         n2 - n1 - 0.5 * (A_l["ax"] + B_l["ax"]) * (xb - xa),
                         n2 - n1 - 0.5 * (A_l["ax"] + T_l["ax"]) * (xb - xa))
        R = np.empty((4,) + th1.shape, dtype=np.result_type(th1, float))
        R[0] = np.where(lam_end, rt_l, np.where(trans, r1t + r2t, rt_t))
        R[1] = np.where(lam_end, rh_l, np.where(trans, r1h + r2h, rh_t))
        R[2] = np.where(lam_end, ra_l, np.where(trans, ra_tr, ra_t))
        R[3] = np.where(lam_end, rs_l, np.where(trans, r2c, rc_t))
        self._last_types = (lam_end, trans)
        self._last_ft = np.real(ft)
        return R

    def _sim_res5(self, V, xi):
        V6 = np.concatenate([V[:5], xi.reshape(1, -1).astype(V.dtype)])
        return self._sim_res(V6)

    def _sim_res(self, V):
        """Первая станция у точки торможения (автомодельное решение)."""
        th, m, n, s, ue, xi = V
        msq = self.mach**2 + 0 * th
        F = np.zeros(1, bool)
        st = bc.station(th, m / ue, ue, s, self.re, msq, F, F)
        xot = xi / st["th"]
        r0 = st["h"] + 2.0 - msq - 0.5 * st["cf"] * xot
        btmp = 2.0 * st["hc"] / st["hs"] + 1.0 - st["h"]
        r1 = btmp + (0.5 * st["cf"] * xot - st["di"] * xot)
        r2 = n
        r3 = s - bc.s_transition(st["hk"], st["cq"])
        return np.array([r0, r1, r2, r3])

    def _wake0_res(self, V):
        thu, mu, nu, su, uu, _, thl, ml, nl, sl, ul, _, thw, mw, nw, sw, uw, _ = V
        r0 = thw - thu - thl
        r1 = mw / uw - mu / uu - ml / ul
        r2 = nw - self.ncrit - 1.0
        r3 = sw - (su * thu + sl * thl) / (thu + thl)
        return np.array([r0, r1, r2, r3])

    # ------------------------------------------------------------------
    def _assemble(self, X, jac=True):
        """Невязка R (Ns,4) и якобиан J (4Ns x 4Ns) с учётом Ue = Ue_inv + D m
        и зависимости ξ от положения точки торможения (тоже функция m).
        jac=False — только невязка."""
        Ns = self.Ns
        ue = self.ue_inv + self.D @ X[:, 1]
        self._ue = ue
        R = np.zeros((Ns, 4))
        J = np.zeros((Ns, 4, Ns, 4))
        Jue = np.zeros((Ns, 4, Ns))          # dR/due по станциям
        Jxi = np.zeros((Ns, 4, Ns))          # dR/dξ по станциям
        XU = np.column_stack([X, ue, self.xi])   # (Ns, 6)
        NV = 6

        def cs(fun, deps, extra):
            """Комплексный шаг по всем входам deps (n, k) станций."""
            n, kdep = deps.shape
            V0 = XU[deps].transpose(1, 2, 0).reshape(kdep * NV, n).astype(complex)
            base = np.real(fun(V0, *extra))
            if not jac:
                return base, None
            dR = np.zeros((4, kdep * NV, n))
            scale = np.maximum(np.abs(np.real(V0)), 1e-12)
            for q in range(kdep * NV):
                Vq = V0.copy()
                Vq[q] = Vq[q] + 1j * CSTEP * scale[q]
                dR[:, q, :] = np.imag(fun(Vq, *extra)) / (CSTEP * scale[q])
            return base, dR

        def scatter(rows, deps, dR):
            if dR is None:
                return
            for c in range(deps.shape[1]):
                J[rows, :, deps[:, c], :] += dR[:, c * NV:c * NV + 4, :].transpose(2, 0, 1)
                Jue[rows, :, deps[:, c]] += dR[:, c * NV + 4, :].T
                Jxi[rows, :, deps[:, c]] += dR[:, c * NV + 5, :].T

        # первые станции сторон
        rows = np.array([self.st_up[0], self.st_lo[0]])
        deps = rows[:, None]
        base, dR = cs(lambda V: self._sim_res(V), deps, ())
        R[rows] = base.T
        scatter(rows, deps, dR)
        # отрезки
        rows = np.concatenate([self.st_up[1:], self.st_lo[1:], self.st_wk[1:]])
        prev = rows - 1
        deps = np.column_stack([prev, rows])
        wake = self.is_wake[rows]
        xtr_f = np.where(np.isin(rows, self.st_up), self.xtr[0], self.xtr[1])
        extra = (self.xc[prev], self.xc[rows], wake, xtr_f)
        base, dR = cs(lambda V, *e: self._interval_res(V, *e), deps, extra)
        R[rows] = base.T
        scatter(rows, deps, dR)
        # первая станция следа
        w0 = self.st_wk[0]
        deps = np.array([[self.st_up[-1], self.st_lo[-1], w0]])
        base, dR = cs(lambda V: self._wake0_res(V), deps, ())
        R[w0] = base[:, 0]
        scatter(np.array([w0]), deps, dR)
        if not jac:
            return R, None
        # связь: dR/dm += dR/due · D + Σ dR/dξ · dξ/ds_st · ds_st/dm
        J[:, :, :, 1] += np.einsum("iak,kj->iaj", Jue, self.D)
        jx = np.einsum("iak,k->ia", Jxi, self.xi_sign)
        J[:, :, :, 1] += jx[:, :, None] * self.dsst_dm[None, None, :]
        return R, J.reshape(4 * Ns, 4 * Ns)

    # ------------------------------------------------------------------
    def solve(self, alpha: float, init=None, max_iter: int = 60, tol: float = 1e-5,
              verbose: bool = False) -> ViscousResult:
        g_inv = self.ps.gamma(alpha)
        self._g_cur = None
        self._build(alpha, g_inv)
        if init is not None and init[0] == (self.k, self.Ns):
            X = init[1].copy()
        else:
            X = self._march()
        X = self._relocate(X, alpha, g_inv)
        converged = False
        hist, damp = [], 1.0
        it = 0
        for it in range(1, max_iter + 1):
            R, J = self._assemble(X)
            try:
                dX = np.linalg.solve(J, -R.ravel()).reshape(self.Ns, 4)
            except np.linalg.LinAlgError:
                break
            # ограничение шага (как в XFOIL: относительные изменения в [-0.5, 1.5])
            rl = 1.0
            why = ""
            turb = self._turb_flags(X)
            dlo, dhi = getattr(self, "dlim", (-0.5, 1.5))
            for c, lo_, hi_ in ((0, dlo, dhi), (1, dlo, dhi), (3, dlo, dhi)):
                ref = np.maximum(np.abs(X[:, c]), 1e-12 if c != 1 else 1e-9)
                rr = dX[:, c] / ref
                if c == 3:
                    rr = np.where(turb, rr, 0.0)
                if rr.min() < lo_ and lo_ / rr.min() < rl:
                    rl = lo_ / rr.min(); why = f"var{c} st{rr.argmin()} {rr.min():.2g}"
                if rr.max() > hi_ and hi_ / rr.max() < rl:
                    rl = hi_ / rr.max(); why = f"var{c} st{rr.argmax()} {rr.max():.2g}"
            dn = np.abs(dX[:, 2]).max()
            dnmax = getattr(self, "dnmax", 5.0)
            if dn > dnmax:
                rl = min(rl, dnmax / dn)
            ue_new = self.ue_inv + self.D @ (X[:, 1] + rl * dX[:, 1])
            ue_old = self._ue
            du = (ue_new - ue_old) / np.maximum(np.abs(ue_old), 1e-3)
            if np.abs(du).max() > 0.5:
                rl = min(rl, 0.5 / np.abs(du).max())
            # разрыв циклов: если размер шага перестал уменьшаться (Ньютон «скачет»
            # через излом замыкающих соотношений), шаги постепенно демпфируются
            if (len(hist) >= 8 and max(hist[-6:]) < 0.3
                    and min(hist[-3:]) > 0.7 * min(hist[-6:-3])):
                damp = max(damp * 0.6, 0.1)
            elif len(hist) >= 2 and hist[-1] < 0.7 * hist[-2]:
                damp = min(1.0, damp * 2.0)
            rl *= damp
            if damp < 1:
                why += f" damp {damp:.2f}"
            X = X + rl * dX
            X[:, 0] = np.maximum(X[:, 0], 1e-10)
            X[:, 3] = np.maximum(X[:, 3], 1e-6)
            rms = np.sqrt(np.mean((dX[:, [0, 1, 3]] / np.maximum(np.abs(X[:, [0, 1, 3]]), 1e-9)) ** 2))
            hist.append(rms)
            if verbose:
                ir = np.unravel_index(np.abs(R).argmax(), R.shape)
                print(f"   it {it:3d} rms {rms:.2e} rlx {rl:.3f} {why}  |R| {np.abs(R).max():.2e} at st{ir[0]} eq{ir[1]}  min ue {self._ue.min():.3f}")
                if getattr(self, "debug_tr", False):
                    rows_ = np.concatenate([self.st_up[1:], self.st_lo[1:], self.st_wk[1:]])
                    lam_, tr_ = self._last_types
                    sl = getattr(self, "dbg_slice", slice(53, 61))
                    hh = (X[sl, 1] - rl * dX[sl, 1]) / self._ue[sl] / (X[sl, 0] - rl * dX[sl, 0])
                    print("      trans at", rows_[tr_], "ft", np.round(self._last_ft[tr_], 3), "H", np.round(hh, 2),
                          " n53-60", np.round(X[sl, 2] - rl * dX[sl, 2], 3), " dn", np.round(dX[sl, 2], 3))
            # перенос точки торможения
            sig = self.sig
            X = self._relocate(X, alpha, g_inv)
            if self.sig != sig:
                continue
            if rms < tol:
                converged = True
                break
        return self._result(alpha, X, converged, it, g_inv)

    def _relocate(self, X, alpha, g_inv):
        """Перенос точки торможения по текущему вязкому γ. Возвращает X
        (возможно, на новой системе станций)."""
        for _ in range(6):
            g = g_inv + self.dg_dm @ X[:, 1]
            k, s_st, up, lo = self._topology(g)
            if (k, len(up), len(lo)) != self.sig:
                if getattr(self, "verbose_remap", False):
                    print(f"      remap {self.sig} -> {(k, len(up), len(lo))}")
                X = self._remap(X, g, alpha, g_inv)
                continue
            # та же топология — только сдвиг ξ
            self.s_st = s_st
            self._set_dsst(g)
            self.xi[self.st_up] = np.maximum(s_st - self.s[self.up], 1e-9)
            self.xi[self.st_lo] = np.maximum(self.s[self.lo] - s_st, 1e-9)
            wl = self.xi[self.st_wk] - self.xi[self.st_wk[0]]
            self.xi[self.st_wk] = self.xi[self.st_lo[-1]] + wl
            break
        return X

    def _remap(self, X, g, alpha, g_inv):
        """Точка торможения перешла через узел: перестроить станции."""
        old_node, old_wk = self.node.copy(), self.st_wk.copy()
        Xnode = np.zeros((self.N + 1, 4))
        Xnode[old_node[old_node >= 0]] = X[:self.nu + self.nl]
        Xw = X[old_wk]
        self._g_cur = g
        self._build(alpha, g_inv)
        Xn = np.zeros((self.Ns, 4))
        Xn[:self.nu + self.nl] = Xnode[self.node[:self.nu + self.nl]]
        for st in (self.st_up[0], self.st_lo[0]):
            if Xn[st, 0] <= 0:                        # узел только что вошёл в слой
                nb = st + 1 if st == self.st_up[0] else st + 1
                Xn[st] = Xn[nb] if nb < self.Ns else Xn[st]
                Xn[st, 1] *= 0.5
        Xn[self.st_wk] = Xw
        return Xn

    def _result(self, alpha, X, converged, it, g_inv):
        Ns = self.Ns
        ue = self.ue_inv + self.D @ X[:, 1]
        g = g_inv + self.dg_dm @ X[:, 1]
        cp0 = 1 - g**2
        cp = karman_tsien(cp0, self.mach)
        ps = self.ps
        a = np.radians(alpha)
        L, nx, ny = ps.L, ps.nx, ps.ny
        cpm = 0.5 * (cp[:-1] + cp[1:])
        fx, fy = -np.sum(cpm * nx * L), -np.sum(cpm * ny * L)
        cl = fy * np.cos(a) - fx * np.sin(a)
        x, y = self.x, self.y
        ra = (x[:-1] - 0.25) * ny - y[:-1] * nx
        rb = (x[1:] - 0.25) * ny - y[1:] * nx
        ca_, cb_ = cp[:-1], cp[1:]
        cm = np.sum(L * (ca_ * ra / 3 + (ca_ * rb + cb_ * ra) / 6 + cb_ * rb / 3))
        st = self._stations(X, ue)
        last = self.st_wk[-1]
        h_end = np.real(st["h"][last])
        cd = 2.0 * X[last, 0] * ue[last] ** ((h_end + 5.0) / 2.0)
        # трение
        cf = np.real(st["cf"])
        tau = cf * ue**2                       # τ_w / q∞ = Cf (Ue/V∞)²
        cdf = 0.0
        for sts, sgn in ((self.st_up, -1.0), (self.st_lo, 1.0)):
            nodes = self.node[sts]
            for i in range(1, len(sts)):
                na, nb = nodes[i - 1], nodes[i]
                p = min(na, nb)
                tdir = sgn * np.array([ps.tx[p], ps.ty[p]])
                proj = tdir[0] * np.cos(a) + tdir[1] * np.sin(a)
                cdf += 0.5 * (tau[sts[i - 1]] + tau[sts[i]]) * L[p] * proj
        # точки перехода
        xtr = []
        for sts, xt_f in ((self.st_up, self.xtr[0]), (self.st_lo, self.xtr[1])):
            n = X[sts, 2]
            idx = np.where((n >= self.ncrit) | (self.xc[sts] >= xt_f))[0]
            if len(idx) == 0 or idx[0] == 0:
                xtr.append(1.0 if len(idx) == 0 else float(self.xc[sts[0]]))
            else:
                i = idx[0]
                fn = (self.ncrit - n[i - 1]) / max(n[i] - n[i - 1], 1e-12) if n[i] >= self.ncrit else 1.0
                fx = (xt_f - self.xc[sts[i - 1]]) / max(self.xc[sts[i]] - self.xc[sts[i - 1]], 1e-12) \
                    if self.xc[sts[i]] >= xt_f else 1.0
                f = min(max(min(fn, fx), 0), 1)
                xtr.append(float(self.xc[sts[i - 1]] + f * (self.xc[sts[i]] - self.xc[sts[i - 1]])))
        side = {}
        for name, sts in (("upper", self.st_up), ("lower", self.st_lo), ("wake", self.st_wk)):
            side[name] = dict(x=self.xc[sts], xi=self.xi[sts], theta=X[sts, 0], dstar=X[sts, 1] / ue[sts],
                              H=np.real(st["h"][sts]), cf=cf[sts], ue=ue[sts], n=X[sts, 2],
                              ctau=X[sts, 3] ** 2)
        side["wake_xy"] = (self.wx, self.wy)
        res = ViscousResult(alpha, self.re, float(cl), float(cd), float(cdf), float(cd - cdf),
                            float(cm), xtr[0], xtr[1], converged, it, x, y, cp, 1 - g_inv**2,
                            g, side, ((self.k, self.Ns), X))
        return res

    def polar(self, alphas, verbose=False):
        """Расчёт по углам с продолжением: начинается от ближайшего к нулю угла."""
        alphas = np.asarray(alphas, float)
        order = np.argsort(np.abs(alphas))
        out = [None] * len(alphas)
        # две ветки: вверх и вниз от минимального |α|
        a0 = order[0]
        res0 = self.solve(alphas[a0], verbose=verbose)
        out[a0] = res0
        for direction in (1, -1):
            init = res0.state if res0.converged else None
            idx = [i for i in np.argsort(alphas * direction) if direction * (alphas[i] - alphas[a0]) > 0]
            for i in idx:
                r = self.solve(alphas[i], init=init, verbose=verbose)
                if not r.converged and init is not None:
                    r2 = self.solve(alphas[i], init=None, verbose=verbose)
                    if r2.converged:
                        r = r2
                out[i] = r
                init = r.state if r.converged else init
        return out
