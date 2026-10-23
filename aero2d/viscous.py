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
        if n_panels:
            airfoil = repanel(airfoil, n_panels // 2 + 1, spacing="viscous", keep_frame=True)
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
        # первая точка следа — на 1e-4 хорды за кромкой (как в XFOIL): слой
        # источников следа не касается узлов профиля
        pts = [np.array([x[0], y[0]]) + getattr(self, "wake_offset", 0.0) * d]
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

    def _topology(self, g, prev=None):
        """Точка торможения и станции слоя по текущему распределению γ.
        prev — текущая сигнатура (k, nu, nl) для гистерезиса исключения узла."""
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
        was_up = was_lo = False
        if prev is not None and prev[0] == k:
            was_up = prev[1] == k                  # узел k уже исключён сверху
            was_lo = prev[2] == N - k - 1          # узел k+1 уже исключён снизу
        if f < (0.15 if was_up else 0.05):
            up = up[1:]
        elif f > (0.85 if was_lo else 0.95):
            lo = lo[1:]
        return k, s_st, up, lo

    def _influence_const(self, S):
        """Постоянные по панелям источники (первоначальная схема)."""
        N, Ns = self.N, self.Ns
        x, y, wx, wy = self.x, self.y, self.wx, self.wy
        nw, npw = len(wx), len(wx) - 1
        Lw = np.hypot(np.diff(wx), np.diff(wy))
        if self._wake_cache is None:
            Bpsi_w, _, _ = source_influence(wx, wy, x[:-1], y[:-1])
            pairs = [(j, j - 1 if j > 0 else None, j if j < npw else None) for j in range(1, nw)]
            _, Ua, Va = source_influence(x, y, wx[1:], wy[1:])
            pairs_local = [(j - 1, a, b) for (j, a, b) in pairs]
            _, Uw, Vw = source_influence(wx, wy, wx[1:], wy[1:], self_pairs=pairs_local)
            Uv, Vv = vortex_velocity_influence(x, y, wx[1:], wy[1:])
            tw = self._wake_tangent(wx, wy)
            self._wake_cache = (Bpsi_w, Ua, Va, Uw, Vw, Uv, Vv, tw)
        Bpsi_w, Ua, Va, Uw, Vw, Uv, Vv, tw = self._wake_cache
        Bpsi = np.hstack([self.Bpsi_air, Bpsi_w])
        dg_dm = (-self.ps.Ainv[:N + 1, :N] @ Bpsi) @ S
        D = np.zeros((Ns, Ns))
        nb = self.nu + self.nl
        D[:nb] = (self.sg_node[self.node[:nb]])[:, None] * dg_dm[self.node[:nb]]
        Uall = np.hstack([Ua, Uw]) @ S + Uv @ dg_dm
        Vall = np.hstack([Va, Vw]) @ S + Vv @ dg_dm
        D[self.st_wk[1:]] = tw[1:, 0:1] * Uall + tw[1:, 1:2] * Vall
        D[self.st_wk[0]] = 0.5 * (D[self.st_up[-1]] + D[self.st_lo[-1]])
        return D, dg_dm, tw, Uv, Vv

    @staticmethod
    def _wake_tangent(wx, wy):
        Lw = np.hypot(np.diff(wx), np.diff(wy))
        dw = np.column_stack([np.diff(wx), np.diff(wy)]) / Lw[:, None]
        tw = np.zeros((len(wx), 2))
        tw[0], tw[-1] = dw[0], dw[-1]
        tw[1:-1] = dw[:-1] + dw[1:]
        return tw / np.linalg.norm(tw, axis=1, keepdims=True)

    def _influence_linear(self, S, sgm, st_of_node):
        """Источники с линейным распределением на полупанелях (как в XFOIL):
        «масса» q в узлах, σ = dq/ds; на профиле q = ±m (знак — по стороне),
        в следе q = m."""
        from .sources import lin_source
        N, Ns = self.N, self.Ns
        x, y, wx, wy = self.x, self.y, self.wx, self.wy
        nw = len(wx)
        if self._wake_cache is None:
            if not hasattr(self, "_air_psi"):
                self._air_psi = lin_source(x, y, x[:-1], y[:-1], self_idx=np.arange(N),
                                           on_sheet=True)[0]
            Pw, _, _ = lin_source(wx, wy, x[:-1], y[:-1], wake_sheet=True)
            _, Pxa, Pya = lin_source(x, y, wx[1:], wy[1:])
            _, Pxw, Pyw = lin_source(wx, wy, wx[1:], wy[1:], self_idx=np.arange(1, nw),
                                     on_sheet=True, wake_sheet=True)
            Uv, Vv = vortex_velocity_influence(x, y, wx[1:], wy[1:])
            tw = self._wake_tangent(wx, wy)
            self._wake_cache = (Pw, Pxa, Pya, Pxw, Pyw, Uv, Vv, tw)
        Pw, Pxa, Pya, Pxw, Pyw, Uv, Vv, tw = self._wake_cache
        Qa = np.zeros((N + 1, Ns))
        nodes = np.where(st_of_node >= 0)[0]
        Qa[nodes, st_of_node[nodes]] = sgm[nodes]
        Qw = np.zeros((nw, Ns))
        Qw[np.arange(nw), self.st_wk] = 1.0
        dpsi_dm = self._air_psi @ Qa + Pw @ Qw                 # N x Ns
        dg_dm = -self.ps.Ainv[:N + 1, :N] @ dpsi_dm
        D = np.zeros((Ns, Ns))
        nb = self.nu + self.nl
        D[:nb] = (self.sg_node[self.node[:nb]])[:, None] * dg_dm[self.node[:nb]]
        Uall = Pya @ Qa + Pyw @ Qw + Uv @ dg_dm
        Vall = -(Pxa @ Qa + Pxw @ Qw) + Vv @ dg_dm
        D[self.st_wk[1:]] = tw[1:, 0:1] * Uall + tw[1:, 1:2] * Vall
        D[self.st_wk[0]] = 0.5 * (D[self.st_up[-1]] + D[self.st_lo[-1]])
        return D, dg_dm, tw, Uv, Vv

    def _build(self, alpha, g_inv):
        """Всё, что зависит от α и топологии: след, станции, ue_inv, D."""
        self._alpha = alpha
        N = self.N
        x, y = self.x, self.y
        k, s_st, up, lo = self._topology(g_inv if self._g_cur is None else self._g_cur,
                                         getattr(self, "sig", None) if self._g_cur is not None else None)
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
        if getattr(self, "source_model", "linear") == "linear":
            D, dg_dm, tw, Uv, Vv = self._influence_linear(S, sgm, st_of_node)
        else:
            D, dg_dm, tw, Uv, Vv = self._influence_const(S)
        self.dg_dm = dg_dm
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

        def solve_b(fint, v, xtr_f=None, xcb=None):
            vd, ok = self._local_newton(fint, v, [TH, M, N_, S_])
            if not np.all(np.isfinite(vd)):
                vd, ok = v.copy(), False
            hk = vd[M] / vd[U] / vd[TH]
            if xtr_f is None:
                hmax = hmax_t                      # след
                turb = True
            else:
                turb = vd[N_] >= self.ncrit or xcb >= xtr_f
                hmax = hmax_t if turb else hmax_l
            if ok and 1.0 < hk <= hmax:
                return vd
            # обратный режим: целевой H меняется плавно (как MRCHUE в XFOIL):
            # в ламинарном отрыве растёт, в турбулентном — убывает к пределу
            h1 = v[1] / v[4] / v[0]
            dxt = (v[11] - v[5]) / v[0]
            htarg = h1 + 0.03 * dxt if not turb else h1 - 0.15 * dxt
            hm = max(htarg, hmax)
            if xtr_f is None:
                hm = max(min(h1, hmax), 1.02)
            eq = lambda Vm, hm=hm: Vm[M] / Vm[U] / Vm[TH] - hm
            v = v.copy()
            v[M] = hm * v[TH] * v[U]
            vi, ok2 = self._local_newton(fint, v, [TH, M, N_, S_, U], extra_eq=eq)
            if ok2 and np.all(np.isfinite(vi)) and vi[TH] > 0 and vi[U] > 0:
                return vi
            return vd if np.all(np.isfinite(vd)) and vd[TH] > 0 else v

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
                v = solve_b(fint, v, xtr_f, self.xc[b])
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
            v = solve_b(fint, v)
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
        uc, msq, ree = self._edge(ue)
        return bc.station(th, ds, uc, s, ree, msq, turb, self.is_wake)

    def _segment(self, A, B, xa, xb, wake, turb):
        """Невязки импульса и формпараметра на отрезке a→b (массивы станций A, B —
        словари closure), плюс уравнение запаздывания (для turb)."""
        ulog = np.log(B["ue"] / A["ue"])
        tlog = np.log(B["th"] / A["th"])
        xlog = np.log(xb / xa)
        hlog = np.log(B["hs"] / A["hs"])
        ha = 0.5 * (A["h"] + B["h"])
        msq = 0.5 * (A["msq"] + B["msq"])
        # Cf в середине отрезка
        thm, dsm, uem = 0.5 * (A["th"] + B["th"]), 0.5 * (A["ds"] + B["ds"]), 0.5 * (A["ue"] + B["ue"])
        mid = bc.station(thm, dsm, uem, 0.5 * (A["s"] + B["s"]), 0.5 * (A["re"] + B["re"]),
                         msq + 0 * thm, turb, wake)
        xa_t, xb_t = xa / A["th"], xb / B["th"]
        cfx = 0.5 * mid["cf"] * (0.5 * (xa + xb)) / thm + 0.25 * (A["cf"] * xa_t + B["cf"] * xb_t)
        rez_t = tlog + (ha + 2.0 - msq) * ulog - xlog * 0.5 * cfx
        # формпараметр с противопоточным весом (как BLDIF в XFOIL: по изменению
        # ln(Hk − 1); в следе — слабее)
        hl = np.log((B["hk"] - 1.0) / (A["hk"] - 1.0))
        hlsq = bc._min(hl**2, 15.0 + 0 * hl)
        hdcon = np.where(wake, 1.0, 5.0) / B["hk"] ** 2
        upw = 1.0 - 0.5 * np.exp(-hlsq * hdcon)
        dix = (1 - upw) * A["di"] * xa_t + upw * B["di"] * xb_t
        cfx2 = (1 - upw) * A["cf"] * xa_t + upw * B["cf"] * xb_t
        hsa, hca = 0.5 * (A["hs"] + B["hs"]), 0.5 * (A["hc"] + B["hc"])
        btmp = 2.0 * hca / hsa + 1.0 - ha
        rez_h = hlog + btmp * ulog + xlog * (0.5 * cfx2 - dix)
        # запаздывание
        ald = np.where(wake, bc.DLCON, 1.0)
        hka = (1 - upw) * A["hk"] + upw * B["hk"]            # как в XFOIL (BLDIF)
        rta = 0.5 * (A["rt"] + B["rt"])
        da = 0.5 * (A["ds"] + B["ds"])
        cfa = (1 - upw) * A["cf"] + upw * B["cf"]
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
        th1, m1, n1, s1, u1i, xa, th2, m2, n2, s2, u2i, xb = V
        nc = self.ncrit
        d1, d2 = m1 / u1i, m2 / u2i                       # δ* = m / Ue (несжимаемая)
        u1, msq1, re1 = self._edge(u1i)
        u2, msq2, re2 = self._edge(u2i)
        m1, m2 = u1 * d1, u2 * d2                         # дальше — со сжимаемой Ue
        msq = 0.5 * (msq1 + msq2)
        self._re_seg = 0.5 * (re1 + re2)
        n1r, n2r = np.real(n1), np.real(n2)
        F = np.zeros(1, bool)
        A_l = bc.station(th1, m1 / u1, u1, s1, re1, msq1, F, wake)
        B_l = bc.station(th2, m2 / u2, u2, s2, re2, msq2, F, wake)
        # тип отрезка (по вещественным частям). Переходный — если ламинарная
        # экстраполяция Ñ на конец отрезка достигает Ñкрит; решение не зависит
        # от неизвестной Ñ₂, поэтому области «ламинарный»/«переходный» не
        # перекрываются (иначе возможны два решения).
        phi1 = np.real(n1 + bc.axset(A_l["ax"], B_l["ax"], th1, th2, n1, nc + 0 * n1, nc) * (xb - xa))
        forced = (xcb >= xtr_f) & ~wake
        turb_start = wake | (n1r >= nc) | ((xca >= xtr_f) & ~wake)
        lam_end = ~turb_start & ~forced & (phi1 < nc)
        trans = ~lam_end & ~turb_start
        A_t = bc.station(th1, m1 / u1, u1, s1, re1, msq1, ~F, wake)
        B_t = bc.station(th2, m2 / u2, u2, s2, re2, msq2, ~F, wake)
        # --- ламинарный отрезок
        rt_l, rh_l, _ = self._segment(A_l, B_l, xa, xb, wake, F)
        ra_l = n2 - n1 - bc.axset(A_l["ax"], B_l["ax"], th1, th2, n1, n2, nc) * (xb - xa)
        rs_l = s2 - bc.s_transition(B_l["hk"], B_l["cq"])
        # --- турбулентный отрезок
        rt_t, rh_t, rc_t = self._segment(A_t, B_t, xa, xb, wake, ~F)
        # за переходом Ñ продолжает медленно расти: если точка перехода сдвинется
        # вниз по потоку, она окажется в следующем отрезке, а не «выключит»
        # турбулентность во всей области за ней
        ra_t = np.where(wake, n2 - n1, n2 - n1 - 20.0 * (xb - xa))
        # --- переходный отрезок: точка перехода внутри
        # доля ламинарного участка по Ñ и по принудительному x
        # естественный переход (как TRCHEK2 в XFOIL): точка перехода — где
        # Ñкрит = Ñ₁ + ½(ax_A + ax_T)·(x_t − x_a), ax_T — в самой точке перехода.
        # Решается простыми итерациями от линейной оценки; при f → 1 совпадает с
        # условием ламинарного отрезка (phi1 = Ñкрит), при f → 0 — с турбулентным.
        # От Ñ₂ точка перехода не зависит (иначе возможен цикл переключений).
        dx_ = xb - xa
        shape = np.shape(th1)

        def bcast(x):
            return np.broadcast_to(np.asarray(x), shape)

        def clip01(v):
            return bc._max(bc._min(v, 1.0 + 0 * v), 0.0 + 0 * v)

        def tstate(ft, P):
            a1, a2, d1_, d2_, v1, v2, r1_, r2_ = P
            return (a1 + ft * (a2 - a1), d1_ + ft * (d2_ - d1_), v1 + ft * (v2 - v1),
                    r1_ + ft * (r2_ - r1_))

        natural = phi1 >= nc
        fn = np.ones(shape, dtype=np.result_type(th1, float))
        sel = np.where(bcast(trans & natural))[0] if len(shape) else np.array([], int)
        if sel.size:
            # точка перехода только для переходных отрезков: Ньютон по
            # вещественным частям (производная по f — разностью), затем одна
            # поправка в комплексной арифметике — она даёт правильные
            # производные f по неизвестным (теорема о неявной функции)
            Pc = tuple(bcast(q)[sel] for q in (th1, th2, m1 / u1, m2 / u2, u1, u2, re1, re2))
            Pr = tuple(np.real(q) for q in Pc)
            s1c, msc, axac, n1c, dxc = (bcast(q)[sel] for q in (s1, msq + 0 * th1, A_l["ax"], n1, dx_))
            wk = bcast(wake)[sel]
            phic = bcast(phi1)[sel]

            def gfun(ft, cplx):
                P = Pc if cplx else Pr
                tht, dst, uet, ret = tstate(ft, P)
                if cplx:
                    s_, ms_, axa, n1_, dx2, t1_ = s1c, msc, axac, n1c, dxc, Pc[0]
                else:
                    s_, ms_, axa, n1_, dx2, t1_ = (np.real(q) for q in (s1c, msc, axac, n1c, dxc, Pc[0]))
                axt = bc.station(tht, dst, uet, s_, ret, ms_, F, wk)["ax"]
                return n1_ + bc.axset(axa, axt, t1_, tht, n1_, nc + 0 * n1_, nc) * ft * dx2 - nc

            n1re = np.real(n1c)
            den = np.where(np.abs(phic - n1re) < 1e-12, 1e-12, phic - n1re)
            fr = np.clip((nc - n1re) / den, 0.0, 1.0)
            for _ in range(8):
                g0 = gfun(fr, False)
                gp = (gfun(fr + 1e-6, False) - g0) / 1e-6
                gp = np.where(np.abs(gp) < 1e-9, 1e-9, gp)
                fr_new = np.clip(fr - g0 / gp, 0.0, 1.0)
                done_ = np.max(np.abs(fr_new - fr)) < 1e-10
                fr = fr_new
                if done_:
                    break
            g0 = gfun(fr, False)
            gp = (gfun(fr + 1e-6, False) - g0) / 1e-6
            gp = np.where(np.abs(gp) < 1e-9, 1e-9, gp)
            if np.iscomplexobj(fn):
                fc = fr - (gfun(fr + 0j, True) - g0) / gp
                fc = np.where((fr <= 0.0) | (fr >= 1.0), fr + 0j, fc)
            else:
                fc = fr
            fn[sel] = fc
        fx = np.where(forced, (xtr_f - xca) / np.where(np.abs(xcb - xca) < 1e-12, 1e-12, xcb - xca), 1.0)
        use_nat = natural & (np.real(fn) <= fx)
        f = np.where(use_nat, fn, fx + 0 * fn)
        ft = clip01(f)    # f → 1 и f → 0 непрерывно переходят в ламинарный / турбулентный отрезок
        tht, dst, uet, ret = tstate(ft, (th1, th2, m1 / u1, m2 / u2, u1, u2, re1, re2))
        xt = xa + ft * (xb - xa)
        T_l = bc.station(tht, dst, uet, s1, ret, msq, F, wake)
        T_t0 = bc.station(tht, dst, uet, s1, ret, msq, ~F, wake)
        st_tr = bc.s_transition(T_t0["hk"], T_t0["cq"])
        T_t = bc.station(tht, dst, uet, st_tr, ret, msq, ~F, wake)
        r1t, r1h, _ = self._segment(A_l, T_l, xa, xt, wake, F)
        r2t, r2h, r2c = self._segment(T_t, B_t, xt, xb, wake, ~F)
        # Ñ: при естественном переходе — до точки перехода (по ней же и Ñ = Ñкрит),
        # при принудительном — обычное уравнение роста по всему отрезку
        ra_tr = np.where(use_nat,
                         n2 - nc - 20.0 * (1.0 - ft) * dx_,
                         n2 - n1 - bc.axset(A_l["ax"], B_l["ax"], th1, th2, n1, n2, nc) * dx_)
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
        th, m, n, s, uei, xi = V
        F = np.zeros(1, bool)
        ue, msq, ree = self._edge(uei)
        st = bc.station(th, m / uei, ue, s, ree, msq, F, F)
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
    def _edge(self, u):
        """Сжимаемость (как в XFOIL): несжимаемая Ue панельного метода →
        сжимаемая по Карману–Цяню; местный Me², отношение ρ/μ на границе слоя
        (изоэнтропия + Сазерленд). Возвращает (ue_c, Me², Re_eff)."""
        M = self.mach
        if M <= 0:
            return u, 0.0 * u, self.re + 0.0 * u
        g = 1.4
        b = np.sqrt(1 - M**2)
        tk = M**2 / (1 + b) ** 2
        uc = u * (1 - tk) / (1 - tk * u**2)
        hst = (g - 1) * M**2 / (1 + 0.5 * (g - 1) * M**2)
        tr = 1 - 0.5 * uc**2 * hst                      # T_e / T_0 (в единицах T0 через q∞=1)
        msq = uc**2 * hst / ((g - 1) * tr)
        rho = tr ** (1 / (g - 1))
        hv = 0.35                                       # Сазерленд: S/T0
        mu = np.sqrt(tr) ** 3 * (1 + hv) / (tr + hv)
        # нормировка на набегающий поток
        tinf = 1 - 0.5 * hst
        rinf = tinf ** (1 / (g - 1))
        muinf = np.sqrt(tinf) ** 3 * (1 + hv) / (tinf + hv)
        return uc, msq, self.re * (rho / rinf) / (mu / muinf)

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
            Q = kdep * NV
            scale = np.maximum(np.abs(np.real(V0)), 1e-12)
            # все возмущения — одним векторным вызовом: блок q столбцов
            # содержит мнимое возмущение q-го входа
            Vb = np.tile(V0, (1, Q))
            for q in range(Q):
                Vb[q, q * n:(q + 1) * n] += 1j * CSTEP * scale[q]
            eb = tuple(np.tile(np.asarray(e), Q) if np.ndim(e) else e for e in extra)
            types = getattr(self, "_last_types", None), getattr(self, "_last_ft", None)
            out = np.imag(fun(Vb, *eb)).reshape(4, Q, n)
            self._last_types, self._last_ft = types
            dR = out / (CSTEP * scale)[None, :, :]
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
              verbose: bool = False, homotopy: bool = True) -> ViscousResult:
        """Решение для одного угла атаки. Если Ньютон не сошёлся при малом Re,
        включается продолжение по числу Рейнольдса: решение при большем Re
        (где отрывы слабее) и пошаговое уменьшение Re до заданного."""
        with np.errstate(all="ignore"):
            return self._solve_h(alpha, init, max_iter, tol, verbose, homotopy)

    def _solve_h(self, alpha, init, max_iter, tol, verbose, homotopy):
        r = self._solve(alpha, init, max_iter, tol, verbose)
        if r.converged or not homotopy or self.re > 2e6:
            return r
        re_target = self.re
        seq = []
        re_k = re_target
        while re_k < min(30 * re_target, 3e6):
            re_k *= 2.0
            seq.append(re_k)
        seq = seq[::-1]
        try:
            best, best_re = None, None
            vh = getattr(self, "verbose_homotopy", False)
            # опорное решение — при наибольшем Re, где сходится
            for re_k in seq:
                self.re = re_k
                rk = self._solve(alpha, init, max_iter, tol, verbose)
                if vh:
                    print(f"      Re={re_k:.3g}: {'ok' if rk.converged else 'нет'} it={rk.iterations}")
                if rk.converged:
                    best, best_re = rk, re_k
                    break
            if best is None:
                return r
            # спуск к заданному Re с адаптивным шагом: при неудаче шаг дробится
            ratio, tries = 2.0, 0
            while best_re > re_target * 1.0001 and tries < 8:
                tries += 1
                re_k = max(best_re / ratio, re_target)
                self.re = re_k
                rk = self._solve(alpha, self._scale_state(best.state, best_re / re_k), max_iter, tol, verbose)
                if vh:
                    print(f"      Re={re_k:.3g}: {'ok' if rk.converged else 'нет'} it={rk.iterations} "
                          f"Cl={rk.cl:.4f} Cd={rk.cd:.5f} xtr={rk.xtr_u:.3f}/{rk.xtr_l:.3f}")
                if rk.converged:
                    best, best_re = rk, re_k
                    ratio = min(2.0, ratio ** 1.5)
                else:
                    ratio = ratio ** 0.5
                    if ratio < 1.08:
                        break
            if best_re <= re_target * 1.0001:
                return best
            return r
        finally:
            self.re = re_target

    @staticmethod
    def _scale_state(state, re_ratio):
        """Перенос состояния на Re/re_ratio: толщины растут примерно как Re^0.4."""
        f = re_ratio ** 0.4
        st = {k: v.copy() for k, v in state.items()}
        for k in ("node", "wake"):
            st[k][:, 0] *= f
            st[k][:, 1] *= f
        return st

    def _solve(self, alpha, init, max_iter, tol, verbose) -> ViscousResult:
        g_inv = self.ps.gamma(alpha)
        self._g_cur = None
        self._build(alpha, g_inv)
        if init is not None:
            X = self._from_state(init)
        else:
            X = self._march()
        X = self._relocate(X, alpha, g_inv)
        converged = False
        hist, damp, rhist = [], 1.0, []
        it = 0
        best_r, best_it = np.inf, 0
        for it in range(1, max_iter + 1):
            R, J = self._assemble(X)
            rmax_ = float(np.abs(R).max())
            if not np.isfinite(rmax_):
                break
            # застой: за 20 итераций невязка не уменьшилась вдвое — дальше не
            # тратить время (сработают продолжение по α или по Re)
            if rmax_ < 0.5 * best_r:
                best_r, best_it = rmax_, it
            elif it - best_it > 20 and rmax_ > 1e-3:
                break
            try:
                dX = np.linalg.solve(J, -R.ravel()).reshape(self.Ns, 4)
            except np.linalg.LinAlgError:
                break
            # ограничение шага (как в XFOIL: относительные изменения в [-0.5, 1.5])
            rl = 1.0
            why = ""
            turb = self._turb_flags(X)
            # ограничение шага (как UPDATE в XFOIL): относительные изменения θ, δ*,
            # Ue и sqrt(Cτ) — в пределах [-0.5, 1.5]; Ñ — не больше 5 за шаг
            dlo, dhi = getattr(self, "dlim", (-0.5, 1.5))
            ue_old = self._ue
            due = self.D @ dX[:, 1]
            dstar = X[:, 1] / ue_old
            ddstar = dX[:, 1] / ue_old - X[:, 1] * due / ue_old**2      # линеаризация δ* = m/Ue
            rels = (("θ", dX[:, 0] / np.maximum(np.abs(X[:, 0]), 1e-12)),
                    ("δ*", ddstar / np.maximum(np.abs(dstar), np.abs(X[:, 0]))),
                    ("Ue", due / np.maximum(np.abs(ue_old), 0.02)),
                    ("S", np.where(turb, dX[:, 3] / np.maximum(np.abs(X[:, 3]), 0.01), 0.0)))
            for nm, rr in rels:
                if rr.min() < dlo and dlo / rr.min() < rl:
                    rl = dlo / rr.min(); why = f"{nm} st{rr.argmin()} {rr.min():.2g}"
                if rr.max() > dhi and dhi / rr.max() < rl:
                    rl = dhi / rr.max(); why = f"{nm} st{rr.argmax()} {rr.max():.2g}"
            dn = np.abs(dX[:, 2]).max()
            dnmax = getattr(self, "dnmax", 5.0)
            if dn > dnmax:
                rl = min(rl, dnmax / dn)
            # разрыв циклов: если размер шага перестал уменьшаться (Ньютон «скачет»
            # через излом замыкающих соотношений), шаги постепенно демпфируются
            # невязка уменьшилась так, как обещал (демпфированный) шаг Ньютона —
            # направление верное, демпфирование ослабляется
            rmax = float(np.abs(R).max())
            good = (len(rhist) >= 1 and rhist[-1][1] < 1
                    and rmax <= (1.0 - 0.5 * rhist[-1][1]) * rhist[-1][0])
            rhist.append((rmax, 1.0))
            if good:
                damp = min(1.0, damp * 2.0)
            elif (len(hist) >= 8 and max(hist[-6:]) < 0.3
                    and min(hist[-3:]) > 0.7 * min(hist[-6:-3])):
                damp = max(damp * 0.6, 0.1)
            elif len(hist) >= 2 and hist[-1] < 0.7 * hist[-2]:
                damp = min(1.0, damp * 2.0)
            rl *= damp
            rhist[-1] = (rhist[-1][0], rl)
            if damp < 1:
                why += f" damp {damp:.2f}"
            # вблизи решения — поиск вдоль направления Ньютона: шаг, не уменьшающий
            # ‖R‖, укорачивается (сильная нелинейность в ламинарном пузыре)
            r2 = float(np.linalg.norm(R))
            ue_keep = self._ue
            for ls in range(5):
                X1 = self._apply_step(X, rl * dX)
                if np.abs(R).max() > 0.5 or ls == 4:
                    break
                saved = (self.s_st, self.xi.copy(), self.dsst_dm.copy())
                if not self._shift_xi(X1, g_inv):
                    self.s_st, self.xi, self.dsst_dm = saved
                    break
                R1 = self._assemble(X1, jac=False)
                R1 = R1[0] if isinstance(R1, tuple) else R1
                self.s_st, self.xi, self.dsst_dm = saved
                if np.all(np.isfinite(R1)) and np.linalg.norm(R1) < r2:
                    break
                rl *= 0.35
                why += " ls"
            self._ue = ue_keep
            X = X1
            rms = np.sqrt(np.mean(np.concatenate([rels[0][1], rels[1][1], rels[2][1]]) ** 2))
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

    def _shift_xi(self, X, g_inv):
        """Сдвиг ξ по новому положению точки торможения без смены топологии
        (False — если топология изменилась бы)."""
        g = g_inv + self.dg_dm @ X[:, 1]
        try:
            k, s_st, up, lo = self._topology(g, self.sig)
        except RuntimeError:
            return False
        if (k, len(up), len(lo)) != self.sig:
            return False
        self.s_st = s_st
        self._set_dsst(g)
        self.xi = self.xi.copy()
        self.xi[self.st_up] = np.maximum(s_st - self.s[self.up], 1e-9)
        self.xi[self.st_lo] = np.maximum(self.s[self.lo] - s_st, 1e-9)
        wl = self.xi[self.st_wk] - self.xi[self.st_wk[0]]
        self.xi[self.st_wk] = self.xi[self.st_lo[-1]] + wl
        return True

    def _apply_step(self, X, dX):
        X = X + dX
        X[:, 0] = np.maximum(X[:, 0], 1e-10)
        X[:, 1] = np.maximum(X[:, 1], 1e-12)
        X[:, 3] = np.maximum(X[:, 3], 1e-6)
        return self._hk_limit(X)

    def _hk_limit(self, X):
        """Как DSLIM в XFOIL: после шага Ньютона H = δ*/θ не меньше 1.02 на
        профиле и 1.00005 в следе (иначе замыкания вырождаются: H → 0 даёт
        бесконечную диссипацию). Ue зависит от m, поэтому — пара проходов."""
        hlim = np.where(self.is_wake, 1.00005, 1.02)
        for _ in range(3):
            ue = np.maximum(np.abs(self.ue_inv + self.D @ X[:, 1]), 1e-6)
            mmin = hlim * X[:, 0] * ue
            low = X[:, 1] < mmin
            if not low.any():
                break
            X[low, 1] = mmin[low]
        return X

    def _to_state(self, X):
        """Состояние по узлам профиля и следу (не зависит от точки торможения)."""
        Xnode = np.full((self.N + 1, 4), np.nan)
        Xnode[self.node[:self.nu + self.nl]] = X[:self.nu + self.nl]
        return dict(node=Xnode, wake=X[self.st_wk].copy())

    def _from_state(self, state):
        Xn = state["node"]
        X = np.zeros((self.Ns, 4))
        X[:self.nu + self.nl] = Xn[self.node[:self.nu + self.nl]]
        nw = min(len(state["wake"]), self.nw)
        X[self.st_wk[:nw]] = state["wake"][:nw]
        # узлы, которых не было в слое (точка торможения сместилась): ближайший
        # сосед по той же стороне (от торможения — вниз по потоку)
        for side in (self.st_up, self.st_lo):
            ok = np.all(np.isfinite(X[side]), axis=1) & (X[side, 0] > 0)
            if ok.any() and not ok.all():
                idx = np.where(ok)[0]
                for j in np.where(~ok)[0]:
                    X[side[j]] = X[side[idx[np.argmin(np.abs(idx - j))]]]
        bad = ~np.all(np.isfinite(X), axis=1) | (X[:, 0] <= 0)
        if bad.any():
            X[bad] = self._initial()[bad]
        return X

    def _relocate(self, X, alpha, g_inv):
        """Перенос точки торможения по текущему вязкому γ. Возвращает X
        (возможно, на новой системе станций)."""
        for _ in range(6):
            g = g_inv + self.dg_dm @ X[:, 1]
            try:
                k, s_st, up, lo = self._topology(g, self.sig)
            except RuntimeError:
                return X
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
        state = self._to_state(X)
        self._g_cur = g
        self._build(alpha, g_inv)
        return self._from_state(state)

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
        ue_end = np.real(self._edge(ue[last])[0])          # сжимаемая скорость в конце следа
        cd = 2.0 * X[last, 0] * ue_end ** ((h_end + 5.0) / 2.0)
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
                              H=np.real(st["h"][sts]), cf=cf[sts], cf_inf=cf[sts] * ue[sts] ** 2,
                              ue=ue[sts], n=X[sts, 2], ctau=X[sts, 3] ** 2)
        side["wake_xy"] = (self.wx, self.wy)
        res = ViscousResult(alpha, self.re, float(cl), float(cd), float(cdf), float(cd - cdf),
                            float(cm), xtr[0], xtr[1], converged, it, x, y, cp, 1 - g_inv**2,
                            g, side, self._to_state(X))
        return res

    def polar(self, alphas, verbose=False, callback=None, max_iter=50):
        """Поляра с продолжением по углу атаки. Опорная точка — первая сошедшаяся
        в порядке |α| (обычно α ≈ 0); от неё — вверх и вниз, каждая точка
        стартует с решения соседней. Несошедшаяся точка повторяется через
        промежуточный угол и с маршевого приближения."""
        alphas = np.asarray(alphas, float)
        n = len(alphas)
        out = [None] * n
        done = [0]

        def report(i, r):
            out[i] = r
            done[0] += 1
            if callback:
                callback(done[0], n, r)

        anchor = None
        for i in np.argsort(np.abs(alphas)):
            r = self.solve(alphas[i], verbose=verbose, max_iter=max_iter)
            report(i, r)
            if r.converged:
                anchor = i
                break
        if anchor is None:
            for i in range(n):
                if out[i] is None:
                    report(i, self.solve(alphas[i], verbose=verbose, max_iter=max_iter))
            return out
        def advance(a_t, r0, depth=0):
            """Переход от сошедшегося решения r0 к углу a_t; при неудаче — через
            промежуточные углы (деление шага пополам, до 3 уровней)."""
            r = self.solve(a_t, init=r0.state, verbose=verbose, max_iter=max_iter, homotopy=False)
            if r.converged or depth >= 3:
                return r
            rm = advance(0.5 * (a_t + r0.alpha), r0, depth + 1)
            if not rm.converged:
                return r
            r2 = advance(a_t, rm, depth + 1)
            return r2 if r2.converged else r

        for direction in (1, -1):
            last_ok = out[anchor]
            idx = [i for i in np.argsort(alphas * direction)
                   if direction * (alphas[i] - alphas[anchor]) > 0]
            for i in idx:
                prev = out[i]
                r = advance(alphas[i], last_ok)
                if prev is not None:
                    out[i] = r
                else:
                    report(i, r)
                if r.converged:
                    last_ok = r
        # повторная попытка для несошедшихся точек — от соседей с другой стороны
        order = np.argsort(alphas)
        for pos, i in enumerate(order):
            if out[i] is not None and out[i].converged:
                continue
            for nb in (order[pos + 1] if pos + 1 < n else None, order[pos - 1] if pos > 0 else None):
                if nb is not None and out[nb] is not None and out[nb].converged:
                    r = advance(alphas[i], out[nb], 1)
                    if r.converged:
                        out[i] = r
                        break
            else:
                # последнее средство — холодный старт с продолжением по Re, но только
                # для «дыр» внутри сошедшегося диапазона (за срывом не тратим время)
                okn = [out[j] is not None and out[j].converged for j in order]
                inner = any(okn[:pos]) and any(okn[pos + 1:])
                if inner:
                    r = self.solve(alphas[i], init=None, verbose=verbose, max_iter=max_iter)
                    if r.converged:
                        out[i] = r
        return out


def polar_robust(airfoil, re, alphas, ncrit=9.0, mach=0.0, xtr=(1.0, 1.0), callback=None,
                 n_panels=160, alt_panels=(170, 140), max_iter=50):
    """Поляра с запасным вариантом: одиночные несошедшиеся точки (соседи по α
    сошлись) и случай, когда не сошлось ничего, пересчитываются на слегка
    другой панельной сетке. Иногда дискретная система вблизи точки торможения
    или перехода почти вырождена, и сдвиг узлов это устраняет."""
    alphas = np.asarray(alphas, float)
    solver = ViscousSolver(airfoil, re, ncrit=ncrit, mach=mach, xtr=xtr, n_panels=n_panels)
    res = solver.polar(alphas, callback=callback, max_iter=max_iter)
    for r in res:
        r.n_panels = n_panels
    order = np.argsort(alphas)
    ok = np.array([r.converged for r in res])
    if ok.all():
        return res
    retry = []
    for pos, i in enumerate(order):
        if ok[i]:
            continue
        left = pos > 0 and ok[order[pos - 1]]
        right = pos + 1 < len(order) and ok[order[pos + 1]]
        if (left and right) or not ok.any():
            retry.append(i)
    for npn in alt_panels:
        if not retry:
            break
        s2 = ViscousSolver(airfoil, re, ncrit=ncrit, mach=mach, xtr=xtr, n_panels=npn)
        still = []
        for i in retry:
            r = s2.solve(alphas[i], max_iter=max_iter)
            if r.converged:
                r.n_panels = npn
                res[i] = r
            else:
                still.append(i)
        retry = still
    return res
