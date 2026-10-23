"""Вязкий расчёт многоэлементного профиля (предкрылок, основной элемент,
закрылки): обобщение вязко-невязкого метода viscous.py на несколько тел.

Невязкая часть — панели с линейной завихренностью на каждом теле (как
multi.py): своя постоянная функции тока и своё условие Кутта на каждом
элементе. У каждого элемента — свой пограничный слой (верх и низ от своей
точки торможения) и свой след, проведённый по линии тока от его задней
кромки до общего сечения за профилем. Вытеснение моделируется источниками
на всех поверхностях и следах; все уравнения (слои + связь) решаются одним
методом Ньютона, как в одноэлементном случае.

Упрощение относительно MSES: след переднего элемента не сливается с
пограничным слоем следующего («confluent boundary layer» не моделируется) —
он идёт над закрылком как отдельный слой со своей скоростью на границе. При
щелях порядка 1–3 % хорды это даёт разумные Cl и Cp; сопротивление и
максимум подъёмной силы на больших отклонениях закрылка — с меньшей
точностью.

Функция тока источников многозначна; для влияния одного тела (и его следа)
на другое разрез проводится по лучу из конца следа вниз по потоку
(sources.unwrapped_angles), иначе он пересёк бы соседнее тело.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import blclosure as bc
from .geometry import Airfoil, repanel
from .panel import psi_influence, vortex_velocity_influence, panel_geometry, karman_tsien
from .sources import lin_source, unwrapped_angles
from .viscous import ViscousSolver


@dataclass
class MultiViscousResult:
    alpha: float
    re: float
    cl: float
    cd: float
    cdf: float
    cdp: float
    cm: float
    converged: bool
    iterations: int
    elements: list = field(default_factory=list)   # по элементам: x, y, cp, cp_inv, cl, xtr_u, xtr_l, side
    state: object = None

    @property
    def xtr_u(self):
        return self.elements[0]["xtr_u"] if self.elements else 1.0

    @property
    def xtr_l(self):
        return self.elements[0]["xtr_l"] if self.elements else 1.0


def default_transition(elements, xtr_main=(1.0, 1.0)):
    """Переход по умолчанию для механизации: на основном (самом длинном)
    элементе — заданный xtr_main (доли его хорды; ≥ 1 — свободный); на
    элементах за ним (закрылки) верхняя поверхность турбулентна почти от
    носка (2 % их хорды): она лежит под турбулентным следом и струёй из
    щели, ламинарный участок с отрывным пузырём там нефизичен и делает
    расчёт неустойчивым. Предкрылок и нижние поверхности — свободный переход."""
    chords = [float(np.max(e.x) - np.min(e.x)) for e in elements]
    im = int(np.argmax(chords))
    x0m = float(np.min(elements[im].x))
    out = []
    for i, e in enumerate(elements):
        x0 = float(np.min(e.x))
        if i == im:
            out.append(tuple(x0 + v * chords[i] if v < 1 else np.inf for v in xtr_main))
        elif x0 > x0m + 0.5 * chords[im]:
            out.append((x0 + 0.02 * chords[i], np.inf))
        else:
            out.append((np.inf, np.inf))
    return out


class _MultiPanels:
    """Невязкая система нескольких тел (та же, что MultiPanelSolver) с
    сохранённой обратной матрицей — нужна матрица влияния источников на γ."""

    def __init__(self, els):
        self.els = els
        sizes = [len(x) for x, _ in els]
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        E = len(els)
        nunk = offs[-1] + E
        A = np.zeros((nunk, nunk))
        bx = np.zeros(nunk)
        by = np.zeros(nunk)
        psi_rows = []
        row = 0
        for e, (x, y) in enumerate(els):
            N = len(x) - 1
            for f, (xf, yf) in enumerate(els):
                A[row:row + N, offs[f]:offs[f + 1]] = psi_influence(xf, yf, x[:-1], y[:-1])
            A[row:row + N, offs[-1] + e] = -1.0
            bx[row:row + N] = -y[:-1]
            by[row:row + N] = x[:-1]
            psi_rows.append(np.arange(row, row + N))
            row += N
            o = offs[e]
            A[row, o] = A[row, o + N] = 1.0
            row += 1
            A[row, [o, o + 1, o + 2]] = [1.0, -2.0, 1.0]
            A[row, [o + N, o + N - 1, o + N - 2]] += [-1.0, 2.0, -1.0]
            row += 1
        self.Ainv = np.linalg.inv(A)
        self.offs = offs
        self.psi_rows = np.concatenate(psi_rows)
        self.ng = offs[-1]
        self.g_x = (self.Ainv @ bx)[:self.ng]
        self.g_y = (self.Ainv @ by)[:self.ng]

    def gamma(self, alpha_deg):
        a = np.radians(alpha_deg)
        return np.cos(a) * self.g_x + np.sin(a) * self.g_y


class MultiViscousSolver(ViscousSolver):
    """elements — список Airfoil (обход как у профиля), координаты в общей
    системе; chord — опорная хорда (по умолчанию 1); xtr — для каждого
    элемента пара (x_верх, x_низ) принудительного перехода в координате x
    (по умолчанию свободный переход); wake_length — длина следа за самой
    дальней задней кромкой, в опорных хордах."""

    def __init__(self, elements, re: float, ncrit: float = 9.0, mach: float = 0.0, xtr=None,
                 chord: float = 1.0, xref=None, n_panels=None, wake_length: float = 1.0,
                 n_wake: int | None = None):
        els = []
        for i, e in enumerate(elements):
            if n_panels:
                npn = n_panels[i] if np.ndim(n_panels) else n_panels
                e = repanel(e, npn // 2 + 1, spacing="viscous", keep_frame=True)
            x, y = np.asarray(e.x, float).copy(), np.asarray(e.y, float).copy()
            area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
            if area < 0:
                x, y = x[::-1].copy(), y[::-1].copy()
            if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-9:
                x[0] = x[-1] = 0.5 * (x[0] + x[-1])
                y[0] = y[-1] = 0.5 * (y[0] + y[-1])
            els.append((x, y))
        self.els = els
        self.E = len(els)
        self.mp = _MultiPanels(els)
        self.ps = self.mp
        self.offs = self.mp.offs
        self.re = float(re)
        self.ncrit = float(ncrit)
        self.mach = float(mach)
        self.chord = float(chord)
        xs = np.concatenate([x for x, _ in els])
        self.xref = xref if xref is not None else (xs.min() + 0.25 * self.chord, 0.0)
        if xtr is None:
            xtr = [(np.inf, np.inf)] * self.E
        elif np.ndim(xtr) == 1:
            xtr = [tuple(xtr)] * self.E
        self.xtr_el = [tuple(float(v) for v in t) for t in xtr]
        self.xtr = self.xtr_el[0]
        self.geom = []
        for x, y in els:
            L, tx, ty, nx, ny = panel_geometry(x, y)
            self.geom.append(dict(L=L, tx=tx, ty=ty, nx=nx, ny=ny,
                                  s=np.concatenate([[0.0], np.cumsum(L)]), N=len(L)))
        self.wake_length = wake_length
        self.n_wake = n_wake
        self._g_cur = None
        self._wake_alpha = None
        # влияние источников каждого тела на свои узлы (не зависит от α)
        self._self_psi = [lin_source(x, y, x[:-1], y[:-1], self_idx=np.arange(len(x) - 1), on_sheet=True)[0]
                          for x, y in els]

    # ------------------------------------------------------------------
    def _wakes(self, g):
        """Линии следа от задних кромок по линиям тока невязкого течения."""
        a = np.radians(self._alpha)
        x_far = max(x[0] for x, _ in self.els) + self.wake_length * self.chord
        out = []

        def vel(p):
            U, V = np.cos(a), np.sin(a)
            for e, (x, y) in enumerate(self.els):
                ge = g[self.offs[e]:self.offs[e + 1]]
                Uv, Vv = vortex_velocity_influence(x, y, np.array([p[0]]), np.array([p[1]]))
                U += Uv[0] @ ge
                V += Vv[0] @ ge
            v = np.array([U, V])
            return v / np.linalg.norm(v)

        for e, (x, y) in enumerate(self.els):
            G = self.geom[e]
            length = max(x_far - x[0], 0.3 * self.chord)
            n = self.n_wake or max(G["N"] // 8 + 2, 12)
            if e < self.E - 1:
                n = int(n * 1.3)
            ds0 = 0.5 * (G["L"][0] + G["L"][-1])
            lo, hi = 1.0, 1.6
            for _ in range(100):
                r = 0.5 * (lo + hi)
                tot = ds0 * (r ** (n - 1) - 1) / (r - 1)
                lo, hi = (r, hi) if tot < length else (lo, r)
            ds = ds0 * r ** np.arange(n - 1)
            t1 = np.array([x[0] - x[1], y[0] - y[1]])
            t2 = np.array([x[-1] - x[-2], y[-1] - y[-2]])
            d = t1 / np.linalg.norm(t1) + t2 / np.linalg.norm(t2)
            d /= np.linalg.norm(d)
            pts = [np.array([x[0], y[0]])]
            for j in range(n - 1):
                p = pts[-1]
                if j == 0:
                    v = d
                else:
                    # шаг дробится на части (в щели линии тока сильно искривлены)
                    sub = 4 if ds[j] > 0.02 * self.chord else 1
                    q = p.copy()
                    h = ds[j] / sub
                    for _ in range(sub):
                        v1 = vel(q)
                        v2 = vel(q + 0.5 * h * v1)
                        q = q + h * v2
                    pts.append(q)
                    continue
                pts.append(p + ds[j] * v)
            pts = np.array(pts)
            out.append((pts[:, 0], pts[:, 1]))
        return out

    def _topology_el(self, e, g, prev=None):
        G = self.geom[e]
        N = G["N"]
        ge = g[self.offs[e]:self.offs[e + 1]]
        idx = np.where((ge[:-1] > 0) & (ge[1:] <= 0))[0]
        if len(idx) == 0:
            raise RuntimeError(f"Не найдена точка торможения на элементе {e + 1}")
        k = idx[np.argmin(np.abs(idx - N // 2))]
        f = ge[k] / (ge[k] - ge[k + 1])
        s_st = G["s"][k] + f * G["L"][k]
        up = np.arange(k, -1, -1)
        lo = np.arange(k + 1, N + 1)
        was_up = was_lo = False
        if prev is not None and prev[0] == k:
            was_up = prev[1] == k
            was_lo = prev[2] == N - k - 1
        if f < (0.15 if was_up else 0.05):
            up = up[1:]
        elif f > (0.85 if was_lo else 0.95):
            lo = lo[1:]
        return k, s_st, up, lo

    def _topology_all(self, g, prev=None):
        out = []
        for e in range(self.E):
            out.append(self._topology_el(e, g, prev[e] if prev is not None else None))
        return out

    def _build(self, alpha, g_inv):
        self._alpha = alpha
        topo = self._topology_all(g_inv if self._g_cur is None else self._g_cur,
                                  getattr(self, "sig", None) if self._g_cur is not None else None)
        self.sig = tuple((k, len(up), len(lo)) for k, _, up, lo in topo)
        if self._wake_alpha != alpha:
            self.wakes = self._wakes(g_inv)
            self._wake_alpha = alpha
            self._wake_cache = None
        E = self.E
        # нумерация станций: элемент e — верх, низ, след
        st_up, st_lo, st_wk, node_g, el_of, xc, xi_sign, xtr_st = [], [], [], [], [], [], [], []
        is_wake = []
        o = 0
        self.k_el, self.sst_el = [], []
        for e in range(E):
            k, s_st, up, lo = topo[e]
            nw = len(self.wakes[e][0])
            nu, nl = len(up), len(lo)
            st_up.append(o + np.arange(nu))
            st_lo.append(o + nu + np.arange(nl))
            st_wk.append(o + nu + nl + np.arange(nw))
            node_g.append(np.concatenate([self.offs[e] + up, self.offs[e] + lo, -np.ones(nw, int)]))
            el_of.append(np.full(nu + nl + nw, e))
            x, y = self.els[e]
            xc.append(np.concatenate([x[up], x[lo], self.wakes[e][0]]))
            xi_sign.append(np.concatenate([np.ones(nu), -np.ones(nl + nw)]))
            xtr_st.append(np.concatenate([np.full(nu, self.xtr_el[e][0]), np.full(nl, self.xtr_el[e][1]),
                                          np.full(nw, 1.0)]))
            is_wake.append(np.concatenate([np.zeros(nu + nl, bool), np.ones(nw, bool)]))
            self.k_el.append(k)
            self.sst_el.append(s_st)
            o += nu + nl + nw
        self.Ns = Ns = o
        self.st_up_el, self.st_lo_el, self.st_wk_el = st_up, st_lo, st_wk
        self.up_el = [t[2] for t in topo]
        self.lo_el = [t[3] for t in topo]
        self.node = np.concatenate(node_g)
        self.el_of = np.concatenate(el_of)
        self.xc = np.concatenate(xc)
        self.xi_sign = np.concatenate(xi_sign)
        self.xtr_st = np.concatenate(xtr_st)
        self.is_wake = np.concatenate(is_wake)
        self.is_up = np.zeros(Ns, bool)
        for e in range(E):
            self.is_up[st_up[e]] = True
        # совместимость с методами базового класса (марш по одному элементу)
        self.st_up, self.st_lo, self.st_wk = st_up[0], st_lo[0], st_wk[0]
        self._set_xi()
        # --- «массы» источников: q в узлах тел и следов через m станций
        ng = self.offs[-1]
        sgn_node = np.zeros(ng)               # знак Ue в узле: верх +γ, низ −γ
        sgm = np.zeros(ng)                    # геометрический знак m в узле
        st_of_node = -np.ones(ng, int)
        for e in range(E):
            n0 = self.offs[e]
            Ne = self.geom[e]["N"]
            idx = np.arange(Ne + 1)
            sgn_node[n0 + idx] = np.where(idx <= self.k_el[e], 1.0, -1.0)
            sgm[n0 + idx] = np.where(idx <= self.k_el[e], -1.0, 1.0)
            st_of_node[n0 + self.up_el[e]] = st_up[e]
            st_of_node[n0 + self.lo_el[e]] = st_lo[e]
        self.sg_node, self.st_of_node, self.sgm = sgn_node, st_of_node, sgm
        Qa = np.zeros((ng, Ns))
        nodes = np.where(st_of_node >= 0)[0]
        Qa[nodes, st_of_node[nodes]] = sgm[nodes]
        Qw = []
        for e in range(E):
            q = np.zeros((len(self.wakes[e][0]), Ns))
            q[np.arange(len(st_wk[e])), st_wk[e]] = 1.0
            Qw.append(q)
        self._influence(Qa, Qw)
        # невязкая скорость на станциях
        a = np.radians(alpha)
        ue0 = np.empty(Ns)
        surf = ~self.is_wake
        ue0[surf] = sgn_node[self.node[surf]] * g_inv[self.node[surf]]
        for e in range(E):
            Uv, Vv, tw = self._wcache["UVv"][e], self._wcache["UVv2"][e], self._wcache["tw"][e]
            uw = tw[1:, 0] * (Uv @ g_inv + np.cos(a)) + tw[1:, 1] * (Vv @ g_inv + np.sin(a))
            ue0[st_wk[e][1:]] = uw
            ue0[st_wk[e][0]] = 0.5 * (ue0[st_up[e][-1]] + ue0[st_lo[e][-1]])
        self.ue_inv = ue0
        self._set_dsst(g_inv if self._g_cur is None else self._g_cur)

    def _set_xi(self):
        E = self.E
        xi = np.empty(self.Ns)
        for e in range(E):
            s = self.geom[e]["s"]
            s_st = self.sst_el[e]
            xi[self.st_up_el[e]] = s_st - s[self.up_el[e]]
            xi[self.st_lo_el[e]] = s[self.lo_el[e]] - s_st
            wx, wy = self.wakes[e]
            wl = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(wx), np.diff(wy)))])
            xi[self.st_wk_el[e]] = xi[self.st_lo_el[e][-1]] + wl
        self.xi = np.maximum(xi, 1e-9)

    def _influence(self, Qa, Qw):
        """D = dUe/dm (Ns x Ns) и dγ/dm."""
        E, offs = self.E, self.offs
        a = np.radians(self._alpha)
        if self._wake_cache is None:
            c = dict(psi_air={}, psi_w={}, Pxw={}, Pyw={}, UVv=[], UVv2=[], tw=[])
            paths = []
            for e in range(E):
                x, y = self.els[e]
                wx, wy = self.wakes[e]
                paths.append((np.concatenate([x, wx[1:]]), np.concatenate([y, wy[1:]])))
            for f in range(E):          # тело/след-источник
                xs, ys = self.els[f]
                wx, wy = self.wakes[f]
                nN = len(xs)
                for e in range(E):      # тело-приёмник (функция тока в узлах)
                    xe, ye = self.els[e]
                    if e == f:
                        c["psi_air"][f, e] = self._self_psi[f]
                        c["psi_w"][f, e] = lin_source(wx, wy, xe[:-1], ye[:-1], wake_sheet=True)[0]
                    else:
                        phi = unwrapped_angles(*paths[f], xe[:-1], ye[:-1])
                        c["psi_air"][f, e] = lin_source(xs, ys, xe[:-1], ye[:-1], phi=phi[:, :nN])[0]
                        c["psi_w"][f, e] = lin_source(wx, wy, xe[:-1], ye[:-1], wake_sheet=True,
                                                      phi=phi[:, nN - 1:])[0]
                for e in range(E):      # скорость в точках следа e
                    pxw, pyw = self.wakes[e][0][1:], self.wakes[e][1][1:]
                    _, Px, Py = lin_source(xs, ys, pxw, pyw)
                    c["Pxw"][("air", f, e)] = (Px, Py)
                    if e == f:
                        _, Px, Py = lin_source(wx, wy, pxw, pyw, self_idx=np.arange(1, len(wx)),
                                               on_sheet=True, wake_sheet=True)
                    else:
                        _, Px, Py = lin_source(wx, wy, pxw, pyw, wake_sheet=True)
                    c["Pxw"][("wk", f, e)] = (Px, Py)
            for e in range(E):
                pxw, pyw = self.wakes[e][0][1:], self.wakes[e][1][1:]
                Uv = np.zeros((len(pxw), offs[-1]))
                Vv = np.zeros((len(pxw), offs[-1]))
                for f in range(E):
                    xs, ys = self.els[f]
                    u, v = vortex_velocity_influence(xs, ys, pxw, pyw)
                    Uv[:, offs[f]:offs[f + 1]] = u
                    Vv[:, offs[f]:offs[f + 1]] = v
                c["UVv"].append(Uv)
                c["UVv2"].append(Vv)
                c["tw"].append(self._wake_tangent(*self.wakes[e]))
            self._wake_cache = c
        c = self._wake_cache
        self._wcache = c
        Ns = Qa.shape[1]
        # ψ в узлах тел (строки ψ-уравнений) от источников
        dpsi = []
        for e in range(E):
            acc = np.zeros((len(self.els[e][0]) - 1, Ns))
            for f in range(E):
                acc += c["psi_air"][f, e] @ Qa[offs[f]:offs[f + 1]] + c["psi_w"][f, e] @ Qw[f]
            dpsi.append(acc)
        dpsi = np.vstack(dpsi)
        Ainv = self.mp.Ainv
        dg_dm = -Ainv[:offs[-1]][:, self.mp.psi_rows] @ dpsi
        D = np.zeros((Ns, Ns))
        surf = np.where(~self.is_wake)[0]
        D[surf] = self.sg_node[self.node[surf]][:, None] * dg_dm[self.node[surf]]
        for e in range(E):
            Ux = np.zeros((len(self.wakes[e][0]) - 1, Ns))
            Vy = np.zeros_like(Ux)
            for f in range(E):
                Px, Py = c["Pxw"][("air", f, e)]
                Ux += Py @ Qa[offs[f]:offs[f + 1]]
                Vy -= Px @ Qa[offs[f]:offs[f + 1]]
                Px, Py = c["Pxw"][("wk", f, e)]
                Ux += Py @ Qw[f]
                Vy -= Px @ Qw[f]
            Ux += c["UVv"][e] @ dg_dm
            Vy += c["UVv2"][e] @ dg_dm
            tw = c["tw"][e]
            D[self.st_wk_el[e][1:]] = tw[1:, 0:1] * Ux + tw[1:, 1:2] * Vy
            D[self.st_wk_el[e][0]] = 0.5 * (D[self.st_up_el[e][-1]] + D[self.st_lo_el[e][-1]])
        cr = getattr(self, "cross", 1.0)       # < 1 — ослабленное влияние вытеснения
        if cr != 1.0:                          # одного элемента на другие (_solve_h)
            el_node = np.repeat(np.arange(E), np.diff(offs))
            same_g = el_node[:, None] == self.el_of[None, :]
            dg_dm = np.where(same_g, dg_dm, cr * dg_dm)
            same_s = self.el_of[:, None] == self.el_of[None, :]
            D = np.where(same_s, D, cr * D)
        self.dg_dm = dg_dm
        self.D = D

    def _set_dsst(self, g):
        """ds_st/dm для каждого элемента: матрица (E, Ns)."""
        rows = []
        for e in range(self.E):
            k = self.k_el[e]
            i, j = self.offs[e] + k, self.offs[e] + k + 1
            gk, gk1 = g[i], g[j]
            d = gk - gk1
            rows.append(self.geom[e]["L"][k] * (-gk1 * self.dg_dm[i] + gk * self.dg_dm[j]) / d**2)
        self.dsst_mat = np.array(rows)
        # для базового класса: dξ/ds_st по станциям собирается в _assemble
        self.dsst_dm = self.dsst_mat

    # ------------------------------------------------------------------
    def _turb_flags(self, X):
        n = np.real(X[:, 2])
        turb = (n >= self.ncrit) | self.is_wake
        turb |= (self.xc >= self.xtr_st) & ~self.is_wake
        return turb

    def _march(self):
        """Маршевое начальное приближение с несколькими внешними итерациями:
        вытеснение слоёв и следа основного элемента заметно меняет обтекание
        носка закрылка (точка торможения смещается), поэтому марш повторяется
        по скорости Ue_inv + ½·D·m с перенесённой точкой торможения."""
        X = self._march_once()
        if getattr(self, "_in_march", False) or self.E < 2:
            return X
        self._in_march = True
        try:
            alpha = self._alpha
            g_inv = self.mp.gamma(alpha)
            ue_prev = None
            for ip in range(getattr(self, "march_passes", 1)):
                X = self._relocate(X, alpha, g_inv)
                ue_keep = self.ue_inv
                try:
                    ue_new = ue_keep + self.D @ X[:, 1]
                    w = getattr(self, "march_relax", 0.5)
                    ue_t = ue_new if ue_prev is None or len(ue_prev) != len(ue_new) else \
                        (1 - w) * ue_prev + w * ue_new
                    if ue_prev is None:
                        ue_t = ue_keep + w * (ue_new - ue_keep)
                    ue_prev = ue_t
                    if getattr(self, "verbose_march", False):
                        print(f"   march pass {ip}: max|ue_visc - ue_t| = {np.abs(ue_new - ue_t).max():.3f}")
                    self.ue_inv = np.maximum(ue_t, 1e-3)
                    X2 = self._march_once()
                finally:
                    self.ue_inv = ue_keep
                if not np.all(np.isfinite(X2)):
                    if getattr(self, "verbose_march", False):
                        bad = np.where(~np.all(np.isfinite(X2), axis=1))[0]
                        print(f"   march pass {ip}: NaN at stations {bad[:10]} (of {len(bad)})")
                    break
                X = X2
            X = self._relocate(X, alpha, g_inv)
            X = self._fix_stagnation(X)
        finally:
            self._in_march = False
        return X

    def _fix_stagnation(self, X, k_int=4):
        """Пересчёт станций у точек торможения по текущей скорости
        Ue = Ue_inv + D·m: после переноса точки торможения туда попадают
        значения соседних узлов, несовместные с автомодельным решением (H < 1,
        огромные невязки), и Ньютон с ограничением шага от них не уходит."""
        X = X.copy()
        ue = np.maximum(self.ue_inv + self.D @ X[:, 1], 1e-3)
        for e in range(self.E):
            for st in (self.st_up_el[e], self.st_lo_el[e]):
                i0 = st[0]
                v = np.concatenate([X[i0], [ue[i0], self.xi[i0]]])
                th0 = 0.29 * np.sqrt(self.xi[i0] / (self.re * ue[i0]))
                v[0], v[1], v[2], v[3] = th0, 2.2 * th0 * ue[i0], 0.0, max(v[3], 1e-4)
                v, ok = self._local_newton(lambda Vm: self._sim_res(Vm), v, [0, 1, 2, 3])
                if not ok or not np.all(np.isfinite(v)):
                    continue
                X[i0] = v[:4]
                xtr_f = self.xtr_st[i0]
                for a, b in zip(st[:k_int], st[1:k_int + 1]):
                    extra = (self.xc[[a]], self.xc[[b]], np.array([False]), np.array([xtr_f]))
                    fint = lambda Vm, extra=extra: self._interval_res(Vm, *extra)
                    g = X[a].copy()
                    g[1] = X[a, 1] / ue[a] * ue[b]
                    if X[b, 2] < self.ncrit:
                        g[2] = X[a, 2]
                    V = np.concatenate([X[a], [ue[a], self.xi[a]], g, [ue[b], self.xi[b]]])
                    vb, ok = self._local_newton(fint, V, [6, 7, 8, 9])
                    if not ok or not np.all(np.isfinite(vb)) or vb[6] <= 0:
                        break
                    X[b] = vb[6:10]
        return X

    def _march_once(self):
        X = np.zeros((self.Ns, 4))
        ue = np.zeros(self.Ns)
        keep = (self.st_up, self.st_lo, self.st_wk, self.xtr)
        try:
            for e in range(self.E):
                self.st_up, self.st_lo, self.st_wk = self.st_up_el[e], self.st_lo_el[e], self.st_wk_el[e]
                self.xtr = self.xtr_el[e]
                Xe = ViscousSolver._march(self)
                rows = np.concatenate([self.st_up, self.st_lo, self.st_wk])
                X[rows] = Xe[rows]
                ue[rows] = self._ue_march[rows]
        finally:
            self.st_up, self.st_lo, self.st_wk, self.xtr = keep
        self._ue_march = ue
        return X

    def _initial(self):
        # базовый _initial работает с текущими self.st_up/st_lo/st_wk/xtr
        return ViscousSolver._initial(self)

    def _assemble(self, X, jac=True):
        Ns = self.Ns
        ue = self.ue_inv + self.D @ X[:, 1]
        self._ue = ue
        R = np.zeros((Ns, 4))
        J = np.zeros((Ns, 4, Ns, 4)) if jac else None
        Jue = np.zeros((Ns, 4, Ns)) if jac else None
        Jxi = np.zeros((Ns, 4, Ns)) if jac else None
        XU = np.column_stack([X, ue, self.xi])
        NV = 6
        CSTEP = 1e-30

        def cs(fun, deps, extra):
            n, kdep = deps.shape
            V0 = XU[deps].transpose(1, 2, 0).reshape(kdep * NV, n).astype(complex)
            base = np.real(fun(V0, *extra))
            if not jac:
                return base, None
            Q = kdep * NV
            scale = np.maximum(np.abs(np.real(V0)), 1e-12)
            Vb = np.tile(V0, (1, Q))
            for q in range(Q):
                Vb[q, q * n:(q + 1) * n] += 1j * CSTEP * scale[q]
            eb = tuple(np.tile(np.asarray(e), Q) if np.ndim(e) else e for e in extra)
            types = getattr(self, "_last_types", None), getattr(self, "_last_ft", None)
            out = np.imag(fun(Vb, *eb)).reshape(4, Q, n)
            self._last_types, self._last_ft = types
            return base, out / (CSTEP * scale)[None, :, :]

        def scatter(rows, deps, dR):
            if dR is None:
                return
            for c in range(deps.shape[1]):
                J[rows, :, deps[:, c], :] += dR[:, c * NV:c * NV + 4, :].transpose(2, 0, 1)
                Jue[rows, :, deps[:, c]] += dR[:, c * NV + 4, :].T
                Jxi[rows, :, deps[:, c]] += dR[:, c * NV + 5, :].T

        E = self.E
        rows = np.array([s[0] for s in self.st_up_el] + [s[0] for s in self.st_lo_el])
        base, dR = cs(lambda V: self._sim_res(V), rows[:, None], ())
        R[rows] = base.T
        scatter(rows, rows[:, None], dR)
        rows = np.concatenate([np.concatenate([self.st_up_el[e][1:], self.st_lo_el[e][1:], self.st_wk_el[e][1:]])
                               for e in range(E)])
        prev = rows - 1
        deps = np.column_stack([prev, rows])
        extra = (self.xc[prev], self.xc[rows], self.is_wake[rows], self.xtr_st[rows])
        base, dR = cs(lambda V, *ex: self._interval_res(V, *ex), deps, extra)
        R[rows] = base.T
        scatter(rows, deps, dR)
        w0 = np.array([self.st_wk_el[e][0] for e in range(E)])
        deps = np.array([[self.st_up_el[e][-1], self.st_lo_el[e][-1], self.st_wk_el[e][0]] for e in range(E)])
        base, dR = cs(lambda V: self._wake0_res(V), deps, ())
        R[w0] = base.T
        scatter(w0, deps, dR)
        if not jac:
            return R, None
        J[:, :, :, 1] += np.einsum("iak,kj->iaj", Jue, self.D)
        # ξ станций элемента e зависит от положения его точки торможения
        XiS = np.zeros((Ns, E))
        XiS[np.arange(Ns), self.el_of] = self.xi_sign
        jx = np.einsum("iak,ke->iae", Jxi, XiS)
        J[:, :, :, 1] += np.einsum("iae,ej->iaj", jx, self.dsst_dm)
        return R, J.reshape(4 * Ns, 4 * Ns)

    # ------------------------------------------------------------------
    def _shift_xi(self, X, g_inv):
        g = g_inv + self.dg_dm @ X[:, 1]
        try:
            topo = self._topology_all(g, self.sig)
        except RuntimeError:
            return False
        if tuple((k, len(up), len(lo)) for k, _, up, lo in topo) != self.sig:
            return False
        self.sst_el = [t[1] for t in topo]
        self._set_dsst(g)
        self._set_xi()
        return True

    def _relocate(self, X, alpha, g_inv):
        for _ in range(6):
            g = g_inv + self.dg_dm @ X[:, 1]
            try:
                topo = self._topology_all(g, self.sig)
            except RuntimeError:
                return X
            sig = tuple((k, len(up), len(lo)) for k, _, up, lo in topo)
            if sig != self.sig:
                X = self._remap(X, g, alpha, g_inv)
                continue
            self.sst_el = [t[1] for t in topo]
            self._set_dsst(g)
            self._set_xi()
            break
        return X

    def polar(self, alphas, verbose=False, callback=None, max_iter=50):
        """Поляра с продолжением по α (как у одиночного профиля); точки, не
        сошедшиеся от соседей, повторяются с холодного старта с продолжением по
        связи элементов и по Re, а от новой сошедшейся точки — снова соседи."""
        out = ViscousSolver.polar(self, alphas, verbose=verbose, callback=callback, max_iter=max_iter)
        alphas = np.asarray(alphas, float)
        order = list(np.argsort(alphas))
        fails = 0
        for _ in range(2):
            bad = [i for i in order if not out[i].converged]
            if not bad:
                break
            changed = False
            for i in bad:
                if out[i].converged or fails >= getattr(self, "max_cold_fails", 2):
                    continue
                r = self.solve(alphas[i], verbose=verbose, max_iter=max_iter)
                if not r.converged:
                    fails += 1                        # дорогие попытки — не больше двух неудач
                    continue
                out[i] = r
                changed = True
                pos = order.index(i)
                for step in (1, -1):                  # от новой точки — к соседям
                    last, j = r, pos + step
                    while 0 <= j < len(order) and not out[order[j]].converged:
                        rj = self.solve(alphas[order[j]], init=last.state, verbose=verbose,
                                        max_iter=max_iter, homotopy=False)
                        if not rj.converged:
                            break
                        out[order[j]] = last = rj
                        j += step
            if not changed:
                break
        return out

    def _solve_h(self, alpha, init, max_iter, tol, verbose, homotopy):
        """Если Ньютон не сошёлся — продолжение по числу Рейнольдса: решение
        при большем Re (слои тоньше, связь слабее, точки торможения смещаются
        меньше) и спуск к заданному Re с адаптивным шагом. Для механизации
        это основной способ получить первую точку поляры."""
        r = self._solve(alpha, init, max_iter, tol, verbose)
        if r.converged or not homotopy:
            return r
        # 1) продолжение по взаимному влиянию элементов: сначала каждый слой
        #    взаимодействует только со своим элементом (как одиночный профиль в
        #    невязком поле остальных), затем связь между элементами включается
        if init is None:
            try:
                st = None
                for c in getattr(self, "cross_steps", (0.0, 0.5, 1.0)):
                    self.cross = c
                    rc = self._solve(alpha, st, max_iter, tol, verbose)
                    if not rc.converged:
                        break
                    st = rc.state
                else:
                    return rc
            finally:
                self.cross = 1.0
        # 2) продолжение по числу Рейнольдса
        re_t = self.re
        try:
            best = None
            for f in getattr(self, "re_factors", (4.0, 16.0)):
                self.re = re_t * f
                rk = self._solve(alpha, None, max_iter, tol, verbose)
                if rk.converged:
                    best, best_re = rk, self.re
                    break
            if best is None:
                return r
            ratio, tries = 2.0, 0
            while best_re > re_t * 1.0001 and tries < 10:
                tries += 1
                re_k = max(best_re / ratio, re_t)
                self.re = re_k
                rk = self._solve(alpha, self._scale_state(best.state, best_re / re_k), max_iter, tol, verbose)
                if rk.converged:
                    best, best_re = rk, re_k
                    ratio = min(2.0, ratio ** 1.5)
                else:
                    ratio = ratio ** 0.5
                    if ratio < 1.08:
                        break
            return best if best_re <= re_t * 1.0001 else r
        finally:
            self.re = re_t

    # шаг ξ при поиске вдоль направления Ньютона в базовом _solve сохраняет
    # (s_st, xi, dsst_dm) — здесь s_st заменён списком
    @property
    def s_st(self):
        return list(self.sst_el)

    @s_st.setter
    def s_st(self, v):
        self.sst_el = list(v)

    def _to_state(self, X):
        Xn = np.full((self.offs[-1], 4), np.nan)
        surf = ~self.is_wake
        Xn[self.node[surf]] = X[surf]
        return dict(node=Xn, wake=[X[s].copy() for s in self.st_wk_el])

    def _from_state(self, state):
        Xn = state["node"]
        X = np.zeros((self.Ns, 4))
        surf = ~self.is_wake
        X[surf] = Xn[self.node[surf]]
        for e in range(self.E):
            w = state["wake"][e]
            nw = min(len(w), len(self.st_wk_el[e]))
            X[self.st_wk_el[e][:nw]] = w[:nw]
        for e in range(self.E):
            for side in (self.st_up_el[e], self.st_lo_el[e]):
                ok = np.all(np.isfinite(X[side]), axis=1) & (X[side, 0] > 0)
                if ok.any() and not ok.all():
                    idx = np.where(ok)[0]
                    for j in np.where(~ok)[0]:
                        X[side[j]] = X[side[idx[np.argmin(np.abs(idx - j))]]]
        bad = ~np.all(np.isfinite(X), axis=1) | (X[:, 0] <= 0)
        if bad.any():
            X[bad] = self._march()[bad]
        if self.E > 1:
            X = self._fix_stagnation(X)
        return X

    @staticmethod
    def _scale_state(state, re_ratio):
        f = re_ratio ** 0.4
        st = dict(node=state["node"].copy(), wake=[w.copy() for w in state["wake"]])
        st["node"][:, :2] *= f
        for w in st["wake"]:
            w[:, :2] *= f
        return st

    # ------------------------------------------------------------------
    def _result(self, alpha, X, converged, it, g_inv):
        ue = self.ue_inv + self.D @ X[:, 1]
        g = g_inv + self.dg_dm @ X[:, 1]
        a = np.radians(alpha)
        c = self.chord
        xr, yr = self.xref
        st = self._stations(X, ue)
        cf = np.real(st["cf"])
        FX = FY = M = 0.0
        cd = cdf = 0.0
        els = []
        for e, (x, y) in enumerate(self.els):
            G = self.geom[e]
            ge = g[self.offs[e]:self.offs[e + 1]]
            cp = karman_tsien(1 - ge**2, self.mach)
            L, nx, ny = G["L"], G["nx"], G["ny"]
            ca_, cb_ = cp[:-1], cp[1:]
            cpm = 0.5 * (ca_ + cb_)
            fx, fy = -np.sum(cpm * nx * L) / c, -np.sum(cpm * ny * L) / c
            ra = (x[:-1] - xr) * ny - (y[:-1] - yr) * nx
            rb = (x[1:] - xr) * ny - (y[1:] - yr) * nx
            m = np.sum(L * (ca_ * ra / 3 + (ca_ * rb + cb_ * ra) / 6 + cb_ * rb / 3)) / c**2
            FX += fx
            FY += fy
            M += m
            last = self.st_wk_el[e][-1]
            h_end = np.real(st["h"][last])
            ue_end = np.real(self._edge(ue[last])[0])
            cd_e = 2.0 * X[last, 0] * ue_end ** ((h_end + 5.0) / 2.0) / c
            cd += cd_e
            tau = cf * ue**2
            cdf_e = 0.0
            for sts, sgn in ((self.st_up_el[e], -1.0), (self.st_lo_el[e], 1.0)):
                nodes = self.node[sts] - self.offs[e]
                for i in range(1, len(sts)):
                    p = min(nodes[i - 1], nodes[i])
                    proj = sgn * (G["tx"][p] * np.cos(a) + G["ty"][p] * np.sin(a))
                    cdf_e += 0.5 * (tau[sts[i - 1]] + tau[sts[i]]) * L[p] * proj / c
            cdf += cdf_e
            xtr = []
            for sts in (self.st_up_el[e], self.st_lo_el[e]):
                n = X[sts, 2]
                xt_f = self.xtr_st[sts[0]]
                idx = np.where((n >= self.ncrit) | (self.xc[sts] >= xt_f))[0]
                if len(idx) == 0 or idx[0] == 0:
                    xtr.append(np.nan if len(idx) == 0 else float(self.xc[sts[0]]))
                else:
                    i = idx[0]
                    fn = (self.ncrit - n[i - 1]) / max(n[i] - n[i - 1], 1e-12) if n[i] >= self.ncrit else 1.0
                    fx_ = (xt_f - self.xc[sts[i - 1]]) / max(self.xc[sts[i]] - self.xc[sts[i - 1]], 1e-12) \
                        if self.xc[sts[i]] >= xt_f else 1.0
                    f = min(max(min(fn, fx_), 0), 1)
                    xtr.append(float(self.xc[sts[i - 1]] + f * (self.xc[sts[i]] - self.xc[sts[i - 1]])))
            side = {}
            for name, sts in (("upper", self.st_up_el[e]), ("lower", self.st_lo_el[e]), ("wake", self.st_wk_el[e])):
                side[name] = dict(x=self.xc[sts], xi=self.xi[sts], theta=X[sts, 0], dstar=X[sts, 1] / ue[sts],
                                  H=np.real(st["h"][sts]), cf=cf[sts], cf_inf=cf[sts] * ue[sts] ** 2, ue=ue[sts], n=X[sts, 2],
                                  ctau=X[sts, 3] ** 2)
            side["wake_xy"] = self.wakes[e]
            gi = g_inv[self.offs[e]:self.offs[e + 1]]
            els.append(dict(x=x, y=y, cp=cp, cp_inv=karman_tsien(1 - gi**2, self.mach), ue=ge,
                            cl=fy * np.cos(a) - fx * np.sin(a), cd=cd_e, cm=m,
                            xtr_u=xtr[0], xtr_l=xtr[1], side=side))
        cl = FY * np.cos(a) - FX * np.sin(a)
        return MultiViscousResult(alpha, self.re, float(cl), float(cd), float(cdf), float(cd - cdf),
                                  float(M), converged, it, els, self._to_state(X))
