"""RANS (Spalart–Allmaras) на неструктурированной гибридной сетке.

Сжимаемые осреднённые уравнения Навье–Стокса + однопараметрическая модель
турбулентности Spalart–Allmaras в варианте SA-neg без ft2 (как на NASA
Turbulence Modeling Resource). Метод конечных объёмов, значения в центрах
ячеек:
  * невязкие потоки — схема Роу с поправкой для малых чисел Маха (LM-Roe,
    Rieper 2011): при M ≈ 0,2 без неё численная вязкость искажает течение;
  * второй порядок — линейная реконструкция по градиентам (взвешенный МНК);
  * вязкие потоки — градиент на грани = среднее градиентов ячеек + поправка
    по линии центров;
  * стенка — прилипание, теплоизолированная; внешняя граница —
    характеристическая (поток Роу к набегающему потоку) с поправкой на
    точечный вихрь (циркуляция по текущей подъёмной силе);
  * установление — псевдовременной метод Ньютона–Крылова: GMRES с точным
    произведением якобиана на вектор (jax.jvp) и предобусловливателем ILU от
    якобиана первого порядка (собирается автоматическим дифференцированием).

Обезразмеривание: ρ∞ = 1, скорость звука a∞ = 1, p∞ = 1/γ, T∞ = 1, длина —
хорда; μ∞ = M/Re. Переменная модели турбулентности — χ = ν̃/ν∞ (неизвестная
ρχ), чтобы все неизвестные были порядка единицы.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np

GAM = 1.4
PR, PRT = 0.72, 0.9
SUTH = 110.4 / 288.15
# Spalart–Allmaras
CB1, SIG, CB2, KAP = 0.1355, 2.0 / 3.0, 0.622, 0.41
CW1 = CB1 / KAP**2 + (1 + CB2) / SIG
CW2, CW3, CV1, CV2, CV3, CN1 = 0.3, 2.0, 7.1, 0.7, 0.9, 16.0


@dataclass
class RansResult:
    alpha: float
    cl: float
    cd: float
    cdp: float
    cdf: float
    cm: float
    converged: bool
    iterations: int
    residual: list = field(default_factory=list)
    elements: list = field(default_factory=list)    # по элементам: x, cp, cf (по граням стенки)
    state: object = None
    status: str = ""                                # «сошлось», «не сошлось», «принято вручную», …
    history: dict = field(default_factory=dict)     # невязки по уравнениям, Cl, Cd по итерациям


@dataclass
class Convergence:
    """Критерий перехода к следующему углу атаки.

    Угол считается сошедшимся, когда последние `hold` итераций подряд
    невязка не выше 10^(−orders) от опорной (наибольшей за расчёт на этой
    сетке) и за эти итерации Cl меняется меньше чем на cl_tol (доля от |Cl|),
    а Cd — меньше чем на cd_tol (абсолютно).
    stall_ok: если невязка упала хотя бы на 2 порядка и перестала падать
    («застряла» в нескольких ячейках), а силы стоят max(hold, 30) итераций, —
    тоже считать сошедшимся."""
    orders: float = 3.0
    hold: int = 20
    cl_tol: float = 2e-4
    cd_tol: float = 2e-5
    stall_ok: bool = True


class RansControl:
    """Связь с окном программы во время расчёта: monitor(info) вызывается
    на каждой итерации (info — словарь: alpha, it, res — невязки 5 уравнений,
    rn, cl, cd, cfl, rref); skip=True — принять текущий угол и перейти к
    следующему; stop=True — прекратить расчёт поляры."""

    def __init__(self, monitor=None):
        self.monitor = monitor
        self.skip = False
        self.stop = False


def _jnp():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


class RansSolver:
    def __init__(self, mesh, mach: float, re: float, chord: float = 1.0, xref=(0.25, 0.0),
                 chi_inf: float = 3.0, order: int = 2, vortex_bc: bool = True, turbulent: bool = True):
        jax, jnp = _jnp()
        self.jax, self.jnp = jax, jnp
        self.m = mesh
        self.mach, self.re, self.chord = float(mach), float(re), float(chord)
        self.mu_inf = self.mach * chord / self.re
        self.nu_inf = self.mu_inf                    # ρ∞ = 1
        self.xref = xref
        self.chi_inf = chi_inf
        self.order = order
        self.vortex_bc = vortex_bc
        self.turbulent = turbulent
        self.limiter_k = 5.0
        self.cfl_mode = "grow"
        self.cfl_growth = 1.25
        self.conv = Convergence()      # критерий перехода к следующему углу
        self.sa_raw_grad = False       # True: источник SA по неограниченным градиентам (на механизации хуже сходится)
        self.visc_raw_grad = False     # True: вязкие потоки по неограниченным градиентам
        self.freeze_limiter = True     # заморозка ограничителя при застое невязки
        self.right_prec = False        # True: правое предобусловливание GMRES (точнее линейные решения, но на практике медленнее)
        self.fd_matvec = False         # True: матрица-вектор разностью (в ~3 раза быстрее JVP, но неточно на изломах ограничителя)
        self.zmin = 0.0
        if not turbulent:
            self.chi_inf = 0.0
        self.gamma_vortex = 0.0
        self._setup()

    # ------------------------------------------------------------------
    def _setup(self):
        jnp = self.jnp
        m = self.m
        nc = m.n_cells
        self.nc = nc
        fi = np.where(m.face_tag == 0)[0]
        fw = np.where(m.face_tag == 1)[0]
        ff = np.where(m.face_tag == 2)[0]
        L, R = m.face_cells[fi, 0], m.face_cells[fi, 1]
        xc = m.centroid
        # внутренние грани
        self.iL, self.iR = L, R
        self.n_i = m.face_normal[fi]
        self.A_i = m.face_area[fi]
        self.rL = m.face_center[fi] - xc[L]
        self.rR = m.face_center[fi] - xc[R]
        d = xc[R] - xc[L]
        self.d_i = d
        self.dist_i = np.hypot(d[:, 0], d[:, 1])
        # стенки
        cw = m.face_cells[fw, 0]
        self.cw = cw
        self.n_w = m.face_normal[fw]
        self.A_w = m.face_area[fw]
        self.r_w = m.face_center[fw] - xc[cw]
        self.dn_w = np.maximum(np.einsum("ij,ij->i", self.r_w, self.n_w), 1e-14)
        self.body_w = m.face_body[fw]
        self.fc_w = m.face_center[fw]
        # внешняя граница
        cf = m.face_cells[ff, 0]
        self.cf = cf
        self.n_f = m.face_normal[ff]
        self.A_f = m.face_area[ff]
        self.r_f = m.face_center[ff] - xc[cf]
        self.fc_f = m.face_center[ff]
        # МНК: соседи по внутренним граням и стенкам
        w_i = 1.0 / self.dist_i**2
        Mxx = np.zeros(nc); Mxy = np.zeros(nc); Myy = np.zeros(nc)
        for idx in (L, R):
            np.add.at(Mxx, idx, w_i * d[:, 0] ** 2)
            np.add.at(Mxy, idx, w_i * d[:, 0] * d[:, 1])
            np.add.at(Myy, idx, w_i * d[:, 1] ** 2)
        dw = self.r_w
        ww = 1.0 / np.maximum(np.sum(dw**2, 1), 1e-30)
        np.add.at(Mxx, cw, ww * dw[:, 0] ** 2)
        np.add.at(Mxy, cw, ww * dw[:, 0] * dw[:, 1])
        np.add.at(Myy, cw, ww * dw[:, 1] ** 2)
        det = Mxx * Myy - Mxy**2
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        self.Minv = np.stack([np.stack([Myy / det, -Mxy / det], -1), np.stack([-Mxy / det, Mxx / det], -1)], -2)
        self.w_i, self.w_w = w_i, ww
        self.vol = m.volume
        self.wall_d = np.maximum(m.wall_dist, 1e-12)
        # в JAX
        J = lambda a: jnp.asarray(a)
        self.J = dict(iL=J(L), iR=J(R), n_i=J(self.n_i), A_i=J(self.A_i), rL=J(self.rL), rR=J(self.rR),
                      d_i=J(d), dist_i=J(self.dist_i), w_i=J(w_i), cw=J(cw), n_w=J(self.n_w), A_w=J(self.A_w),
                      r_w=J(self.r_w), dn_w=J(self.dn_w), w_w=J(ww), cf=J(cf), n_f=J(self.n_f), A_f=J(self.A_f),
                      r_f=J(self.r_f), fc_f=J(self.fc_f), Minv=J(self.Minv), vol=J(self.vol), wd=J(self.wall_d))
        self._res_jit = None

    # ------------------------------------------------------------------
    # физика
    # ------------------------------------------------------------------
    def prim(self, Q):
        """Q (n,5): ρ, ρu, ρv, ρE, ρχ → w: ρ, u, v, p, χ."""
        jnp = self.jnp
        r = Q[:, 0]
        u, v = Q[:, 1] / r, Q[:, 2] / r
        p = (GAM - 1) * (Q[:, 3] - 0.5 * r * (u**2 + v**2))
        return jnp.stack([r, u, v, p, Q[:, 4] / r], 1)

    def cons(self, w):
        jnp = self.jnp
        r, u, v, p, chi = (w[:, k] for k in range(5))
        return jnp.stack([r, r * u, r * v, p / (GAM - 1) + 0.5 * r * (u**2 + v**2), r * chi], 1)

    def freestream(self, alpha):
        a = np.radians(alpha)
        return np.array([1.0, self.mach * np.cos(a), self.mach * np.sin(a), 1.0 / GAM, self.chi_inf])

    def mu_lam(self, T):
        return self.mu_inf * T**1.5 * (1 + SUTH) / (T + SUTH)

    def roe(self, wL, wR, n):
        jnp = self.jnp
        nx, ny = n[:, 0], n[:, 1]
        rL, uL, vL, pL, cL = (wL[:, k] for k in range(5))
        rR, uR, vR, pR, cR = (wR[:, k] for k in range(5))
        unL, unR = uL * nx + vL * ny, uR * nx + vR * ny
        HL = GAM / (GAM - 1) * pL / rL + 0.5 * (uL**2 + vL**2)
        HR = GAM / (GAM - 1) * pR / rR + 0.5 * (uR**2 + vR**2)
        FL = jnp.stack([rL * unL, rL * uL * unL + pL * nx, rL * vL * unL + pL * ny, rL * HL * unL], 1)
        FR = jnp.stack([rR * unR, rR * uR * unR + pR * nx, rR * vR * unR + pR * ny, rR * HR * unR], 1)
        sL, sR = jnp.sqrt(rL), jnp.sqrt(rR)
        rh = sL * sR
        uh = (sL * uL + sR * uR) / (sL + sR)
        vh = (sL * vL + sR * vR) / (sL + sR)
        Hh = (sL * HL + sR * HR) / (sL + sR)
        q2 = uh**2 + vh**2
        ah = jnp.sqrt(jnp.maximum((GAM - 1) * (Hh - 0.5 * q2), 1e-12))
        unh = uh * nx + vh * ny
        tx, ty = -ny, nx
        uth = uh * tx + vh * ty
        dr, dp = rR - rL, pR - pL
        dun = unR - unL
        dut = (uR - uL) * tx + (vR - vL) * ty
        aL = jnp.sqrt(GAM * pL / rL); aR = jnp.sqrt(GAM * pR / rR)
        z = jnp.minimum(1.0, jnp.maximum(jnp.sqrt(uL**2 + vL**2) / aL, jnp.sqrt(uR**2 + vR**2) / aR))
        z = jnp.maximum(z, self.zmin)
        a1 = (dp - rh * ah * z * dun) / (2 * ah**2)
        a2 = dr - dp / ah**2
        a3 = (dp + rh * ah * z * dun) / (2 * ah**2)
        at = rh * dut

        def fix(l):
            dl = 0.05 * ah
            al = jnp.abs(l)
            return jnp.where(al < dl, (l**2 + dl**2) / (2 * dl), al)
        l1, l2, l3 = fix(unh - ah), jnp.abs(unh), fix(unh + ah)
        d0 = l1 * a1 + l2 * a2 + l3 * a3
        d1 = l1 * a1 * (uh - ah * nx) + l2 * (a2 * uh + at * tx) + l3 * a3 * (uh + ah * nx)
        d2 = l1 * a1 * (vh - ah * ny) + l2 * (a2 * vh + at * ty) + l3 * a3 * (vh + ah * ny)
        d3 = l1 * a1 * (Hh - ah * unh) + l2 * (a2 * 0.5 * q2 + at * uth) + l3 * a3 * (Hh + ah * unh)
        F = 0.5 * (FL + FR) - 0.5 * jnp.stack([d0, d1, d2, d3], 1)
        mf = F[:, 0]
        F5 = 0.5 * mf * (cL + cR) - 0.5 * jnp.abs(mf) * (cR - cL)
        return jnp.concatenate([F, F5[:, None]], 1)

    def sa_terms(self, rho, chi, T):
        """μ, μt и вспомогательные величины SA (χ = ν̃/ν∞)."""
        jnp = self.jnp
        mu = self.mu_lam(T)
        nu_t = chi * self.nu_inf                       # ν̃
        X = rho * nu_t / mu                            # ν̃/ν
        fv1 = X**3 / (X**3 + CV1**3)
        mut = jnp.where(chi > 0, rho * nu_t * fv1, 0.0)
        fn = jnp.where(chi >= 0, 1.0, (CN1 + X**3) / (CN1 - X**3))
        return mu, mut, X, fv1, fn

    def visc_flux(self, wf, gu, gv, gT, gc, n):
        """Вязкий поток через грань: wf — значения на грани, g* — градиенты (n,2)."""
        jnp = self.jnp
        rho, u, v, p, chi = (wf[:, k] for k in range(5))
        T = GAM * p / rho
        mu, mut, X, fv1, fn = self.sa_terms(rho, chi, T)
        me = mu + mut
        div = gu[:, 0] + gv[:, 1]
        txx = me * (2 * gu[:, 0] - 2.0 / 3.0 * div)
        tyy = me * (2 * gv[:, 1] - 2.0 / 3.0 * div)
        txy = me * (gu[:, 1] + gv[:, 0])
        nx, ny = n[:, 0], n[:, 1]
        fx = txx * nx + txy * ny
        fy = txy * nx + tyy * ny
        k = (mu / PR + mut / PRT) / (GAM - 1)
        fe = u * fx + v * fy + k * (gT[:, 0] * nx + gT[:, 1] * ny)
        fs = (mu + rho * chi * self.nu_inf * fn) / SIG * (gc[:, 0] * nx + gc[:, 1] * ny)
        return jnp.stack([jnp.zeros_like(fx), fx, fy, fe, fs], 1)

    def sa_source(self, w, g, wd):
        """Источник SA на единицу объёма в переменных ρχ: ρ(P − D)/ν∞ + (cb2/σ)ρ ν∞|∇χ|²."""
        jnp = self.jnp
        rho, u, v, p, chi = (w[:, k] for k in range(5))
        T = GAM * p / rho
        mu, mut, X, fv1, fn = self.sa_terms(rho, chi, T)
        Om = jnp.abs(g[:, 2, 0] - g[:, 1, 1])
        nut = chi * self.nu_inf
        fv2 = 1 - X / (1 + X * fv1)
        Sbar = nut * fv2 / (KAP**2 * wd**2)
        St = jnp.where(Sbar >= -CV2 * Om, Om + Sbar,
                       Om + Om * (CV2**2 * Om + CV3 * Sbar) / ((CV3 - 2 * CV2) * Om - Sbar + 1e-30))
        St = jnp.maximum(St, 1e-16)
        r = jnp.minimum(nut / (St * KAP**2 * wd**2), 10.0)
        gg = r + CW2 * (r**6 - r)
        fw = gg * ((1 + CW3**6) / (gg**6 + CW3**6)) ** (1.0 / 6.0)
        prod = jnp.where(chi >= 0, CB1 * St * chi, CB1 * Om * chi)
        dest = jnp.where(chi >= 0, CW1 * fw * nut * chi / wd**2, -CW1 * nut * chi / wd**2)
        cross = CB2 / SIG * self.nu_inf * (g[:, 4, 0] ** 2 + g[:, 4, 1] ** 2)
        return rho * (prod - dest + cross)

    # ------------------------------------------------------------------
    def gradients(self, w, wall_vals):
        """Градиенты МНК всех 5 примитивных переменных: (nc, 5, 2)."""
        jnp = self.jnp
        J = self.J
        dphi = w[J["iR"]] - w[J["iL"]]                 # (nfi,5)
        contrib = (J["w_i"][:, None, None] * dphi[:, :, None] * J["d_i"][:, None, :])
        rhs = jnp.zeros((self.nc, 5, 2))
        rhs = rhs.at[J["iL"]].add(contrib).at[J["iR"]].add(contrib)
        dw = wall_vals - w[J["cw"]]
        rhs = rhs.at[J["cw"]].add(J["w_w"][:, None, None] * dw[:, :, None] * J["r_w"][:, None, :])
        return jnp.einsum("cij,ckj->cki", J["Minv"], rhs)

    def _wall_values(self, w):
        jnp = self.jnp
        wc = w[self.J["cw"]]
        z = jnp.zeros_like(wc[:, 0])
        return jnp.stack([wc[:, 0], z, z, wc[:, 3], z], 1)

    def _far_state(self, winf, gam):
        """Состояние на внешней границе: набегающий поток + точечный вихрь
        циркуляции gam (с поправкой Прандтля–Глауэрта)."""
        jnp = self.jnp
        J = self.J
        x = J["fc_f"][:, 0] - self.xref[0]
        y = J["fc_f"][:, 1] - self.xref[1]
        M = self.mach
        b = np.sqrt(1 - M**2)
        a = self.alpha_rad
        xr = x * np.cos(a) + y * np.sin(a)
        yr = -x * np.sin(a) + y * np.cos(a)
        th = jnp.arctan2(yr, xr)
        r = jnp.sqrt(xr**2 + yr**2)
        fac = gam * b / (2 * np.pi * r * (1 - M**2 * jnp.sin(th) ** 2))
        du_r, dv_r = fac * jnp.sin(th), -fac * jnp.cos(th)        # в повёрнутых осях
        du = du_r * np.cos(a) - dv_r * np.sin(a)
        dv = du_r * np.sin(a) + dv_r * np.cos(a)
        U = winf[1] + du
        V = winf[2] + dv
        # изоэнтропически: p, ρ по скорости
        q2 = U**2 + V**2
        T = 1 + 0.5 * (GAM - 1) * (M**2 - q2)
        rho = T ** (1 / (GAM - 1))
        p = rho * T / GAM
        return jnp.stack([rho, U, V, p, jnp.full_like(rho, winf[4])], 1)

    def residual(self, Q, winf, gam=0.0, order=None, phi_frozen=None):
        """Невязка R (nc,5): сумма потоков наружу минус источник·объём."""
        jnp = self.jnp
        J = self.J
        order = self.order if order is None else order
        w = self.prim(Q)
        wv = self._wall_values(w)
        g = self.gradients(w, wv)
        gr = g                                   # неограниченные — для вязких членов и источника SA
        iL, iR = J["iL"], J["iR"]
        if order >= 2 and self.limiter_k > 0:
            phi = self.venkat(w, g, wv) if phi_frozen is None else phi_frozen
            g = g * phi[:, :, None]
        if order >= 2:
            g = g.at[J["cf"]].set(0.0)            # у внешней границы — первый порядок
        if order >= 2:
            wL = w[iL] + jnp.einsum("fkj,fj->fk", g[iL], J["rL"])
            wR = w[iR] + jnp.einsum("fkj,fj->fk", g[iR], J["rR"])
            # положительность ρ, p
            wL = jnp.where((wL[:, [0]] > 0.2 * w[iL][:, [0]]) & (wL[:, [3]] > 0.2 * w[iL][:, [3]]), wL, w[iL])
            wR = jnp.where((wR[:, [0]] > 0.2 * w[iR][:, [0]]) & (wR[:, [3]] > 0.2 * w[iR][:, [3]]), wR, w[iR])
        else:
            wL, wR = w[iL], w[iR]
        Fc = self.roe(wL, wR, J["n_i"])
        # вязкие: градиент на грани
        wf = 0.5 * (w[iL] + w[iR])
        T = GAM * w[:, 3] / w[:, 0]
        gvv = gr if self.visc_raw_grad else g
        gT = self._grad_T(w, gvv)
        e = J["d_i"] / J["dist_i"][:, None]

        def fgrad(phiL, phiR, gl, gr):
            ga = 0.5 * (gl + gr)
            corr = (phiR - phiL) / J["dist_i"] - jnp.sum(ga * e, 1)
            return ga + corr[:, None] * e
        gu = fgrad(w[iL, 1], w[iR, 1], gvv[iL, 1], gvv[iR, 1])
        gv = fgrad(w[iL, 2], w[iR, 2], gvv[iL, 2], gvv[iR, 2])
        gTf = fgrad(T[iL], T[iR], gT[iL], gT[iR])
        gc = fgrad(w[iL, 4], w[iR, 4], gvv[iL, 4], gvv[iR, 4])
        Fv = self.visc_flux(wf, gu, gv, gTf, gc, J["n_i"])
        F = (Fc - Fv) * J["A_i"][:, None]
        Rs = jnp.zeros((self.nc, 5)).at[iL].add(F).at[iR].add(-F)
        # стенка
        cw = J["cw"]
        ww = w[cw]
        pw = ww[:, 3] + (jnp.sum(g[cw, 3] * J["r_w"], 1) if order >= 2 else 0.0)
        pw = jnp.where(pw > 0.2 * ww[:, 3], pw, ww[:, 3])
        n = J["n_w"]
        Fcw = jnp.stack([jnp.zeros_like(pw), pw * n[:, 0], pw * n[:, 1], jnp.zeros_like(pw), jnp.zeros_like(pw)], 1)
        dn = J["dn_w"]

        def wgrad(phiw, phii, gi):
            corr = (phiw - phii) / dn - jnp.sum(gi * n, 1)
            return gi + corr[:, None] * n
        z = jnp.zeros_like(pw)
        gu_w = wgrad(z, ww[:, 1], gvv[cw, 1])
        gv_w = wgrad(z, ww[:, 2], gvv[cw, 2])
        gc_w = wgrad(z, ww[:, 4], gvv[cw, 4])
        wfw = jnp.stack([ww[:, 0], z, z, ww[:, 3], z], 1)
        gT_w = gT[cw] - jnp.sum(gT[cw] * n, 1)[:, None] * n      # теплоизолированная
        Fvw = self.visc_flux(wfw, gu_w, gv_w, gT_w, gc_w, n)
        Fw = (Fcw - Fvw) * J["A_w"][:, None]
        Rs = Rs.at[cw].add(Fw)
        # внешняя граница
        cf = J["cf"]
        wi = w[cf] + (jnp.einsum("fkj,fj->fk", g[cf], J["r_f"]) if order >= 2 else 0.0)
        wi = jnp.where((wi[:, [0]] > 0.2 * w[cf][:, [0]]) & (wi[:, [3]] > 0.2 * w[cf][:, [3]]), wi, w[cf])
        wg = self._far_state(winf, gam)
        Ff = self.roe(wi, wg, J["n_f"]) * J["A_f"][:, None]
        Rs = Rs.at[cf].add(Ff)
        # источник SA
        if not self.turbulent:                 # ламинарный расчёт: ρχ ≡ 0
            return Rs.at[:, 4].set(Q[:, 4] * J["vol"])
        S = self.sa_source(w, gr if self.sa_raw_grad else g, J["wd"])
        Rs = Rs.at[:, 4].add(-S * J["vol"])
        return Rs

    def venkat(self, w, g, wv):
        """Ограничитель Венкатакришнана (по всем 5 примитивным переменным):
        в гладком течении почти не действует (ε² = (K·h)³), но гасит
        выбросы реконструкции на вырожденных ячейках «веера» у задних кромок."""
        jnp = self.jnp
        J = self.J
        iL, iR, cw = J["iL"], J["iR"], J["cw"]
        big = 1e300
        wmax = w.at[iL].max(w[iR]).at[iR].max(w[iL]).at[cw].max(wv)
        wmin = w.at[iL].min(w[iR]).at[iR].min(w[iL]).at[cw].min(wv)
        dmax, dmin = wmax - w, wmin - w
        h = jnp.sqrt(J["vol"])
        # характерный масштаб переменных для ε
        scale = jnp.asarray([1.0, self.mach, self.mach, 1.0 / GAM, max(self.chi_inf, 1.0)])
        eps2 = ((self.limiter_k * h) ** 3)[:, None] * scale[None, :] ** 2

        def psi(d2, i):
            d1 = jnp.where(d2 > 0, dmax[i], dmin[i])
            num = (d1**2 + eps2[i]) * d2 + 2 * d2**2 * d1
            den = d2 * (d1**2 + 2 * d2**2 + d1 * d2 + eps2[i]) + 1e-300
            return jnp.where(jnp.abs(d2) < 1e-14, 1.0, jnp.clip(num / den, 0.0, 1.0))
        pL = psi(jnp.einsum("fkj,fj->fk", g[iL], J["rL"]), iL)
        pR = psi(jnp.einsum("fkj,fj->fk", g[iR], J["rR"]), iR)
        lim = jnp.ones_like(w).at[iL].min(pL).at[iR].min(pR)
        return lim

    def _grad_T(self, w, g):
        # T = γ p/ρ: ∇T = γ(∇p/ρ − p∇ρ/ρ²)
        rho, p = w[:, 0], w[:, 3]
        return GAM * (g[:, 3] / rho[:, None] - (p / rho**2)[:, None] * g[:, 0])

    # ------------------------------------------------------------------
    # якобиан первого порядка (предобусловливатель)
    # ------------------------------------------------------------------
    def _jac1_parts(self, Q, winf, gam):
        jax, jnp = self.jax, self.jnp
        J = self.J

        def face_int(qL, qR, n, d, dist):
            wL, wR = self.prim(qL[None])[0], self.prim(qR[None])[0]
            Fc = self.roe(wL[None], wR[None], n[None])[0]
            e = d / dist
            TL, TR = GAM * wL[3] / wL[0], GAM * wR[3] / wR[0]
            g2 = lambda a, b: ((b - a) / dist * e)[None]
            Fv = self.visc_flux(0.5 * (wL + wR)[None], g2(wL[1], wR[1]), g2(wL[2], wR[2]), g2(TL, TR),
                                g2(wL[4], wR[4]), n[None])[0]
            return Fc - Fv

        def face_wall(qi, n, dn):
            wi = self.prim(qi[None])[0]
            p = wi[3]
            Fc = jnp.stack([0.0 * p, p * n[0], p * n[1], 0.0 * p, 0.0 * p])
            gw = lambda phi: (-phi / dn * n)[None]
            wf = jnp.stack([wi[0], 0.0 * p, 0.0 * p, p, 0.0 * p])
            Fv = self.visc_flux(wf[None], gw(wi[1]), gw(wi[2]), jnp.zeros((1, 2)), gw(wi[4]), n[None])[0]
            return Fc - Fv

        def face_far(qi, wg, n):
            wi = self.prim(qi[None])[0]
            return self.roe(wi[None], wg[None], n[None])[0]

        iL, iR = J["iL"], J["iR"]
        jl, jr = jax.vmap(jax.jacfwd(face_int, argnums=(0, 1)))(Q[iL], Q[iR], J["n_i"], J["d_i"], J["dist_i"])
        jw = jax.vmap(jax.jacfwd(face_wall))(Q[J["cw"]], J["n_w"], J["dn_w"])
        wg = self._far_state(winf, gam)
        jf = jax.vmap(jax.jacfwd(face_far))(Q[J["cf"]], wg, J["n_f"])
        # источник: ∂S/∂(ρχ) при замороженных градиентах
        w = self.prim(Q)
        g = self.gradients(w, self._wall_values(w))

        def src(q, gi, wd):
            return self.sa_source(self.prim(q[None]), gi[None], wd[None])[0]
        js = jax.vmap(jax.grad(src))(Q, g, J["wd"])            # (nc,5)
        return jl, jr, jw, jf, js

    def jac1(self, Q, winf, gam, dt):
        """Разреженная матрица V/Δt + J₁ (блоки 5×5), scipy CSR."""
        import scipy.sparse as sp
        if not hasattr(self, "_jac1_fn"):
            self._jac1_fn = self.jax.jit(self._jac1_parts)
        jl, jr, jw, jf, js = (np.asarray(a) for a in self._jac1_fn(Q, winf, gam))
        nc = self.nc
        A_i, A_w, A_f = self.A_i[:, None, None], self.A_w[:, None, None], self.A_f[:, None, None]
        jl, jr, jw, jf = jl * A_i, jr * A_i, jw * A_w, jf * A_f
        rows, cols, blocks = [], [], []
        # диагональ
        D = np.zeros((nc, 5, 5))
        np.add.at(D, self.iL, jl)
        np.add.at(D, self.iR, -jr)
        np.add.at(D, self.cw, jw)
        np.add.at(D, self.cf, jf)
        D[:, 4, 4] += self.vol * np.maximum(-js[:, 4], 0.0)
        if not self.turbulent:
            D[:, 4, :] = 0.0
            D[:, 4, 4] = self.vol
            jl[:, 4, :] = 0; jr[:, 4, :] = 0
        D += (self.vol / dt)[:, None, None] * np.eye(5)[None]
        rows.append(np.arange(nc)); cols.append(np.arange(nc)); blocks.append(D)
        rows.append(self.iL); cols.append(self.iR); blocks.append(jr)
        rows.append(self.iR); cols.append(self.iL); blocks.append(-jl)
        r = np.concatenate(rows); c = np.concatenate(cols); b = np.concatenate(blocks)
        if getattr(self, "_want_blocks", False):
            return r, c, b
        ii = (r[:, None, None] * 5 + np.arange(5)[None, :, None]) + 0 * np.arange(5)[None, None, :]
        jj = (c[:, None, None] * 5 + np.arange(5)[None, None, :]) + 0 * np.arange(5)[None, :, None]
        return sp.csr_matrix((b.ravel(), (ii.ravel(), jj.ravel())), shape=(5 * nc, 5 * nc))

    # ------------------------------------------------------------------
    def local_dt(self, Q, cfl):
        w = np.asarray(self.prim(self.jnp.asarray(Q)))
        a = np.sqrt(GAM * w[:, 3] / w[:, 0])
        lam = np.zeros(self.nc)
        un = np.abs(0.5 * ((w[self.iL, 1] + w[self.iR, 1]) * self.n_i[:, 0] +
                           (w[self.iL, 2] + w[self.iR, 2]) * self.n_i[:, 1]))
        aa = 0.5 * (a[self.iL] + a[self.iR])
        s = (un + aa) * self.A_i
        np.add.at(lam, self.iL, s)
        np.add.at(lam, self.iR, s)
        np.add.at(lam, self.cf, (np.abs(w[self.cf, 1] * self.n_f[:, 0] + w[self.cf, 2] * self.n_f[:, 1]) + a[self.cf]) * self.A_f)
        np.add.at(lam, self.cw, a[self.cw] * self.A_w)
        T = GAM * w[:, 3] / w[:, 0]
        mu = self.mu_inf * T**1.5 * (1 + SUTH) / (T + SUTH)
        nut = np.maximum(w[:, 4], 0) * self.nu_inf
        lv = np.zeros(self.nc)
        sv = self.A_i**2
        np.add.at(lv, self.iL, sv)
        np.add.at(lv, self.iR, sv)
        np.add.at(lv, self.cw, self.A_w**2)
        lam += 4.0 * (mu / w[:, 0] + nut) * lv / self.vol
        return cfl * self.vol / lam

    def forces(self, Q, alpha):
        jnp = self.jnp
        w = np.asarray(self.prim(jnp.asarray(Q)))
        g = np.asarray(self.gradients(self.jnp.asarray(w), self._wall_values(self.jnp.asarray(w))))
        cw, n, A, dn = self.cw, self.n_w, self.A_w, self.dn_w
        ww = w[cw]
        pw = ww[:, 3] + np.sum(g[cw, 3] * self.r_w, 1)
        T = GAM * ww[:, 3] / ww[:, 0]
        mu = self.mu_inf * T**1.5 * (1 + SUTH) / (T + SUTH)
        # касательное напряжение по разности «центр − стенка»
        ut = ww[:, 1] * (-n[:, 1]) + ww[:, 2] * n[:, 0]
        tau = mu * ut / dn                       # вдоль t = (−ny, nx)
        q = 0.5 * self.mach**2
        fx = (pw - 1 / GAM) * n[:, 0] * A - (-tau * (-n[:, 1])) * A
        fy = (pw - 1 / GAM) * n[:, 1] * A - (-tau * n[:, 0]) * A
        # сила на тело: давление действует по нормали грани (в тело), трение — по направлению течения у стенки
        fxp = (pw - 1 / GAM) * n[:, 0] * A
        fyp = (pw - 1 / GAM) * n[:, 1] * A
        fxv = tau * (-n[:, 1]) * A
        fyv = tau * n[:, 0] * A
        a = np.radians(alpha)
        c = self.chord
        Lp = (-fxp * np.sin(a) + fyp * np.cos(a)).sum() / (q * c)
        Dp = (fxp * np.cos(a) + fyp * np.sin(a)).sum() / (q * c)
        Lv = (-fxv * np.sin(a) + fyv * np.cos(a)).sum() / (q * c)
        Dv = (fxv * np.cos(a) + fyv * np.sin(a)).sum() / (q * c)
        xm = self.fc_w[:, 0] - self.xref[0]
        ym = self.fc_w[:, 1] - self.xref[1]
        cm = -((xm * (fyp + fyv) - ym * (fxp + fxv)).sum()) / (q * c**2)
        cp = (pw - 1 / GAM) / q
        cf = tau / q
        per = []
        for b in range(int(self.body_w.max()) + 1):
            k = self.body_w == b
            per.append(dict(x=self.fc_w[k, 0], y=self.fc_w[k, 1], cp=cp[k], cf=cf[k],
                            cl=float(((-(fxp + fxv) * np.sin(a) + (fyp + fyv) * np.cos(a))[k]).sum() / (q * c)),
                            cd=float((((fxp + fxv) * np.cos(a) + (fyp + fyv) * np.sin(a))[k]).sum() / (q * c))))
        return dict(cl=Lp + Lv, cd=Dp + Dv, cdp=Dp, cdf=Dv, cm=cm, cp=cp, cf=cf, elements=per)

    # ------------------------------------------------------------------
    def solve(self, alpha, Q0=None, max_iter=300, tol=1e-7, cfl0=5.0, cfl_max=1e6, first_order_iters=30,
              verbose=False, callback=None, gmres_tol=0.01, gmres_iter=60, conv=None, control=None):
        """Установление до выполнения критерия conv (Convergence; по умолчанию —
        self.conv) или до max_iter итераций Ньютона. control — RansControl."""
        crit = conv or self.conv
        import scipy.sparse.linalg as spla
        jax, jnp = self.jax, self.jnp
        winf = self.freestream(alpha)
        self.alpha_rad = np.radians(alpha)
        if Q0 is None:
            Q = np.asarray(self.cons(jnp.asarray(np.tile(winf, (self.nc, 1)))))
            # у стенки — затухание скорости (мягкий старт)
            f = np.clip(self.wall_d / 0.02, 0, 1)[:, None]
            w0 = np.tile(winf, (self.nc, 1)); w0[:, 1:3] *= f
            Q = np.asarray(self.cons(jnp.asarray(w0)))
        else:
            Q = np.array(Q0, float)
        res1 = jax.jit(lambda q, wi, gm: self.residual(q, wi, gm, order=1))
        res2 = jax.jit(lambda q, wi, gm, ph: self.residual(q, wi, gm, order=2, phi_frozen=ph))
        jvp1 = jax.jit(lambda q, v, wi, gm: jax.jvp(lambda qq: self.residual(qq, wi, gm, order=1), (q,), (v,))[1])
        jvp2 = jax.jit(lambda q, v, wi, gm, ph: jax.jvp(
            lambda qq: self.residual(qq, wi, gm, order=2, phi_frozen=ph), (q,), (v,))[1])
        phi_fn = jax.jit(lambda q: self.venkat(self.prim(q), self.gradients(self.prim(q), self._wall_values(self.prim(q))),
                                               self._wall_values(self.prim(q))))
        phi = None                                    # «замороженный» ограничитель (при застое невязки)
        from .bilu import BlockILU0
        if not hasattr(self, "_bilu"):
            self._want_blocks = True
            r_, c_, _ = self.jac1(jnp.asarray(np.tile(np.asarray(self.cons(jnp.asarray(winf[None]))), (self.nc, 1))),
                                  jnp.asarray(winf), jnp.asarray(0.0), np.ones(self.nc))
            self._want_blocks = False
            self._bilu = BlockILU0(self.nc, r_, c_)
        if not hasattr(self, "_jac1_fn") or getattr(self, "_jac_alpha", None) != alpha:
            self._jac1_fn = jax.jit(self._jac1_parts)
            self._jac_alpha = alpha
        hist = []
        r0 = None
        cfl = cfl0
        self._last_ok, self._last_lin = True, 0.0
        fh = []
        H = dict(it=[], res=[], rn=[], cl=[], cd=[], cfl=[])
        conv = False
        status = ""
        it = 0
        for it in range(1, max_iter + 1):
            if control is not None and (control.skip or control.stop):
                status = "принято вручную" if control.skip else "остановлено"
                control.skip = False
                break
            order = 1 if it <= first_order_iters and Q0 is None else self.order
            if order == 1:
                resf, jvpf = res1, jvp1
            else:
                resf = lambda q, wi, gm_: res2(q, wi, gm_, phi)                     # noqa: E731
                jvpf = lambda q, v, wi, gm_: jvp2(q, v, wi, gm_, phi)              # noqa: E731
            Qj = jnp.asarray(Q)
            winf_j = jnp.asarray(winf)
            gm = jnp.asarray(self.gamma_vortex if self.vortex_bc else 0.0)
            R = np.asarray(resf(Qj, winf_j, gm))
            rn = float(np.sqrt(np.mean((R / self.vol[:, None]) ** 2, 0)).sum())
            if not np.isfinite(rn):
                status = "расходится"
                break
            if r0 is None or (order == 2 and len(hist) and hist[-1][1] == 1):
                r0 = rn
            hist.append((rn, order))
            fr_ = self.forces(Q, alpha)
            req = np.sqrt(np.mean((R / self.vol[:, None]) ** 2, 0))
            for k_, v_ in (("it", it), ("res", req), ("rn", rn), ("cl", fr_["cl"]), ("cd", fr_["cd"]),
                           ("cfl", cfl)):
                H[k_].append(v_)
            if order == 2:
                self._rref = max(getattr(self, "_rref", 0.0), rn)
            if control is not None and control.monitor is not None:
                control.monitor(dict(alpha=alpha, it=it, res=req, rn=rn, cl=fr_["cl"], cd=fr_["cd"], cfl=cfl,
                                     rref=getattr(self, "_rref", rn), order=order))
            if order == 2 and rn < tol * r0:
                conv = True
                break
            # критерий сходимости (Convergence). Опорная невязка — наибольшая за все
            # расчёты на этом решателе (у тёплого старта своя невязка мала).
            if order == 2:
                fh.append((fr_["cl"], fr_["cd"]))
                o2 = [h[0] for h in hist if h[1] == 2]
                rref = self._rref
                stalled = len(o2) > 60 and min(o2[-30:]) > 0.5 * min(o2[:-30])

                def steady(win, tol_cl, tol_cd):
                    if len(fh) <= win:
                        return False
                    cls = [f[0] for f in fh[-win - 1:]]
                    cds = [f[1] for f in fh[-win - 1:]]
                    return (max(cls) - min(cls) < tol_cl * max(0.2, abs(fh[-1][0]))
                            and max(cds) - min(cds) < tol_cd)
                hold = max(1, int(crit.hold))
                lvl = 10.0 ** (-crit.orders) * rref
                if len(o2) >= hold and max(o2[-hold:]) <= lvl and steady(hold, crit.cl_tol, crit.cd_tol):
                    conv, status = True, "сошлось"
                    break
                if (crit.stall_ok and len(o2) > 40 and stalled and rn < 1e-2 * rref
                        and steady(max(hold, 30), crit.cl_tol, crit.cd_tol)):
                    conv, status = True, "сошлось (невязка застряла, силы установились)"
                    break
                # застой невязки при почти установившихся силах — заморозить ограничитель
                # («дребезг» ограничителя в нескольких ячейках не даёт невязке падать)
                if (self.freeze_limiter and phi is None and self.limiter_k > 0 and stalled
                        and steady(30, 0.005, 1e-3)):
                    phi = phi_fn(Qj)
                    if verbose:
                        print(f"  it {it:4d}: невязка застряла на {rn:.2e} — ограничитель заморожен", flush=True)
            # CFL: рост по уменьшению невязки (SER)
            if len(hist) >= 2 and hist[-2][1] == order:
                if self.cfl_mode == "ser":
                    cfl = min(cfl_max, max(cfl0, cfl * min(2.0, (hist[-2][0] / rn) ** 1.0)))
                else:                                   # рост, пока шаги удаются
                    ok_ = getattr(self, "_last_ok", True)
                    lin_ = getattr(self, "_last_lin", 0.0)
                    grow = (self.cfl_growth if rn < 1.2 * hist[-2][0] and ok_ and lin_ < 0.25 else
                            1.0 if rn < 3.0 * hist[-2][0] and ok_ else 0.5)
                    cfl = min(cfl_max, max(cfl0, cfl * grow))
            dt = self.local_dt(Q, cfl)
            self._want_blocks = True
            _, _, blocks = self.jac1(Qj, winf_j, gm, dt)
            self._want_blocks = False
            ilu = self._bilu.factor(blocks)
            if not np.all(np.isfinite(ilu.invU)):
                cfl = max(cfl0, cfl / 10)
                continue
            N5 = 5 * self.nc
            M = spla.LinearOperator((N5, N5), ilu.solve)
            Vdt = np.repeat(self.vol / dt, 5)

            qn = float(np.linalg.norm(Q))
            Rf = R.ravel()

            def mv(x):
                if not self.fd_matvec:
                    jv = np.asarray(jvpf(Qj, jnp.asarray(x.reshape(-1, 5)), winf_j, gm)).ravel()
                    return jv + Vdt * x
                # производная по направлению разностью (одна невязка вместо JVP)
                xn = float(np.linalg.norm(x))
                if xn == 0.0:
                    return np.zeros_like(x)
                eps = 1e-12 * (1.0 + qn) / xn          # мал: изломы ограничителя
                Rp = np.asarray(resf(jnp.asarray(Q + eps * x.reshape(-1, 5)), winf_j, gm)).ravel()
                return (Rp - Rf) / eps + Vdt * x
            gh = []
            if self.right_prec:
                # правое предобусловливание: GMRES минимизирует истинную линейную невязку
                Aop = spla.LinearOperator((N5, N5), lambda y: mv(ilu.solve(y)))
                y, info = spla.gmres(Aop, -R.ravel(), rtol=gmres_tol, maxiter=1, restart=gmres_iter,
                                     callback=lambda rr: gh.append(rr), callback_type="pr_norm")
                dq = ilu.solve(y)
            else:
                Aop = spla.LinearOperator((N5, N5), mv)
                dq, info = spla.gmres(Aop, -R.ravel(), M=M, rtol=gmres_tol, maxiter=1, restart=gmres_iter,
                                      callback=lambda rr: gh.append(rr), callback_type="pr_norm")
            lin = float(np.linalg.norm(mv(dq) + R.ravel()) / max(np.linalg.norm(R), 1e-300))
            gh.append(lin)
            self._last_ok = bool(np.isfinite(lin) and lin < 0.5)
            self._last_lin = lin
            if not np.isfinite(lin) or lin > 0.9:      # линейная задача не решилась — уменьшить шаг
                cfl = max(cfl0, cfl / 4)
                continue
            dq = dq.reshape(-1, 5)
            # ограничение шага: ρ и p не меньше 20 % от текущих
            w = np.asarray(self.prim(Qj))
            Qn = Q + dq
            wn = np.asarray(self.prim(jnp.asarray(Qn)))
            rel = np.maximum(np.max(-(wn[:, 0] - w[:, 0]) / w[:, 0]), np.max(-(wn[:, 3] - w[:, 3]) / w[:, 3]))
            lam = 1.0 if rel < 0.2 else 0.2 / rel
            if lam < 1.0:
                cfl = max(cfl0, cfl * 0.5)
            Q = Q + lam * dq
            # циркуляция для вихревой поправки
            if self.vortex_bc and it % 5 == 0:
                fr = self.forces(Q, alpha)
                self.gamma_vortex = 0.5 * fr["cl"] * self.mach * self.chord
            if verbose and (it % 10 == 0 or it < 5):
                fr = self.forces(Q, alpha)
                if verbose > 1:
                    req = np.sqrt(np.mean((R / self.vol[:, None]) ** 2, 0))
                    wq = np.asarray(self.prim(jnp.asarray(Q)))
                    print("     eq res", np.array2string(req, precision=2), "chi max %.1f min %.2f" % (wq[:, 4].max(), wq[:, 4].min()),
                          "worst cell", int(np.argmax(np.abs(R[:, 4]) / self.vol)), "wd %.2e" % self.wall_d[int(np.argmax(np.abs(R[:, 4]) / self.vol))])
                print(f"  it {it:4d} ord {order} res {rn:.3e} cfl {cfl:.2e} lam {lam:.2f} gmres {len(gh)}:{gh[-1] if gh else 0:.1e} "
                      f"Cl {fr['cl']:.4f} Cd {fr['cd']:.5f}", flush=True)
            if callback and callback(it, rn):
                break
        fr = self.forces(Q, alpha)
        if not status:
            status = "сошлось" if conv else ("расходится" if hist and not np.isfinite(hist[-1][0])
                                             else "не сошлось за отведённые итерации")
        H = {k_: np.array(v_) for k_, v_ in H.items()}
        H["rref"] = float(getattr(self, "_rref", 0.0) or (max(H["rn"]) if len(H["rn"]) else 1.0))
        return RansResult(alpha, fr["cl"], fr["cd"], fr["cdp"], fr["cdf"], fr["cm"], conv, it,
                          [h[0] for h in hist], fr["elements"], Q, status, H)


def wake_lines(elements, alpha, re, chord=1.0, length=0.6):
    """Линии следа за задними кромками по невязкому многоэлементному решению
    (для сгущения сетки RANS); пусто, если построить не удалось."""
    if len(elements) < 2:
        return None
    try:
        from .mviscous import MultiViscousSolver
        ms = MultiViscousSolver(elements, re)
        ms._alpha = float(alpha)
        wk = ms._wakes(ms.mp.gamma(ms._alpha))
        xe = max(float(e.x.max()) for e in elements) + length * chord
        return [(wx[wx < xe], wy[wx < xe]) for wx, wy in wk]
    except Exception:                                  # noqa: BLE001
        return None


def mesh_for(elements, alpha, re, chord=1.0, fine=False, opts=None):
    """Сетка RANS по параметрам opts (MeshOptions; по умолчанию — автоматические).
    Для механизации — сгущение в щелях и по линиям следа невязкого решения при угле alpha."""
    from .rans_mesh import MeshOptions, build_mesh
    from .geometry import repanel
    o = opts or MeshOptions()
    multi = len(elements) > 1
    els = elements
    if o.n_surface and o.n_surface > 10:
        cmax = max(float(np.ptp(e.x)) for e in elements)
        els = [repanel(e, max(20, int(round(o.n_surface * float(np.ptp(e.x)) / cmax))), keep_frame=True)
               for e in elements]
    near = o.near_size or ((0.002 if fine else 0.003) if multi else 0.004)
    kw = dict(yplus=o.yplus, ratio=o.ratio, bl_thickness=(o.bl_thickness or None), far_radius=o.far_radius,
              far_size=o.far_size, slot_size=near * chord, growth=o.growth)
    if multi and o.wake_refine:
        kw.update(wakes=wake_lines(els, alpha, re, chord, length=o.wake_length),
                  wake_size=(0.006 if fine else o.wake_size))
    return build_mesh(els, chord=chord, re=re, **kw)


def _seq_order(alphas):
    """Порядок счёта: от угла, ближайшего к нулю, в обе стороны."""
    order = np.argsort(alphas, kind="stable")
    i0 = int(np.argmin(np.abs(alphas[order])))
    return list(order[i0:]) + list(order[:i0][::-1])


def _solve_sequence(solver, alphas, idx, states, out, max_iter, verbose, conv, control, on_iter, on_done):
    """Углы idx по порядку, каждый — от ближайшего посчитанного (states)."""
    for i in idx:
        Q = None
        if states:
            j = min(states, key=lambda jj: abs(alphas[jj] - alphas[i]))
            Q = states[j]
        r = solver.solve(float(alphas[i]), Q0=Q, max_iter=max_iter, verbose=verbose,
                         callback=(lambda it, rn: on_iter(i, it, rn)), conv=conv, control=control)
        out[i] = r
        if r.converged or r.status == "принято вручную" or (not states and Q is None):
            states[i] = r.state
        if on_done(i, r) or (control is not None and control.stop):
            return True
    return False


class _ProcControl(RansControl):
    """RansControl внутри процесса-исполнителя: монитор — в общую очередь,
    «следующий угол» и «стоп» — через общие флаги."""

    def __init__(self, queue, skip_gen, stop_ev):
        super().__init__(monitor=lambda d: queue.put(("iter", d)))
        self._q, self._gen, self._stop = queue, skip_gen, stop_ev
        self._seen = skip_gen.value

    @property
    def skip(self):
        g = self._gen.value
        if g != self._seen:
            self._seen = g
            return True
        return False

    @skip.setter
    def skip(self, v):
        pass

    @property
    def stop(self):
        return self._stop.is_set()

    @stop.setter
    def stop(self, v):
        pass


def _rans_chunk(args):
    """Исполнитель: часть поляры в отдельном процессе."""
    import warnings
    warnings.filterwarnings("ignore")
    (mesh, mach, re, chord, xref, alphas, idx, Q0, rref, max_iter, conv, queue, skip_gen, stop_ev) = args
    solver = RansSolver(mesh, mach, re, chord=chord, xref=xref)
    solver._rref = rref
    ctl = _ProcControl(queue, skip_gen, stop_ev)
    out = [None] * len(alphas)
    st = {}
    for i in idx:
        # каждый следующий угол — от ближайшего посчитанного в этом куске, первый — от Q0
        Q = st[min(st, key=lambda jj: abs(alphas[jj] - alphas[i]))] if st else Q0
        r = solver.solve(float(alphas[i]), Q0=Q, max_iter=max_iter, conv=conv, control=ctl)
        out[i] = r
        if r.converged or r.status == "принято вручную":
            st[i] = r.state
        queue.put(("done", i, r.alpha, r.cl, r.status))
        if ctl.stop:
            break
    return [(i, out[i]) for i in idx if out[i] is not None]


def rans_polar(elements, alphas, re, mach=0.15, chord=1.0, xref=None, mesh=None, callback=None,
               max_iter=250, verbose=False, conv=None, control=None, workers=1):
    """Поляра RANS для одно- или многоэлементного профиля: сетка строится один
    раз (сгущение по линиям следа невязкого решения при среднем угле), углы
    атаки считаются от ближайшего к нулю в обе стороны, каждый — от решения
    ближайшего посчитанного.
    workers > 1: первый угол считается здесь, остальные делятся на непрерывные
    куски и считаются одновременно в отдельных процессах (каждый кусок стартует
    от решения первого угла).
    callback(k, n, it, res, result_or_None) -> True для остановки.
    conv — критерий сходимости (Convergence), control — RansControl (монитор
    итераций, «следующий угол», остановка)."""
    alphas = np.asarray(alphas, float)
    if mesh is None:
        mesh = mesh_for(elements, float(np.median(alphas)), re, chord)
    xs = np.concatenate([e.x for e in elements])
    xr = xref if xref is not None else (float(xs.min()) + 0.25 * chord, 0.0)
    solver = RansSolver(mesh, mach, re, chord=chord, xref=xr)
    seq = _seq_order(alphas)
    n = len(alphas)
    out = [None] * n
    states = {}
    count = [0]

    def on_iter(i, it, rn):
        return bool(callback and callback(count[0], n, it, rn, None))

    def on_done(i, r):
        count[0] += 1
        return bool(callback and callback(count[0], n, 0, 0.0, r))
    workers = int(max(1, min(workers, n - 1)))
    if workers <= 1:
        _solve_sequence(solver, alphas, seq, states, out, max_iter, verbose, conv, control, on_iter, on_done)
        return out, mesh
    # --- первый угол здесь, остальные — параллельно
    stop = _solve_sequence(solver, alphas, seq[:1], states, out, max_iter, verbose, conv, control, on_iter, on_done)
    if stop:
        return out, mesh
    i0 = seq[0]
    Q0 = states.get(i0, out[i0].state)
    rest = seq[1:]
    up = [i for i in rest if alphas[i] >= alphas[i0]]
    dn = [i for i in rest if alphas[i] < alphas[i0]]
    # куски пропорционально числу углов в каждую сторону
    w_up = int(round(workers * len(up) / len(rest))) if up else 0
    w_up = min(max(w_up, 1 if up else 0), len(up))
    w_dn = min(max(workers - w_up, 1 if dn else 0), len(dn))
    chunks = [list(c) for c in np.array_split(up, w_up) if len(c)] if up else []
    chunks += [list(c) for c in np.array_split(dn, w_dn) if len(c)] if dn else []
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    from .parallel import cpu_count
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    queue, skip_gen, stop_ev = mgr.Queue(), mgr.Value("i", 0), mgr.Event()
    rref = getattr(solver, "_rref", 0.0)
    # потоки XLA: на каждый процесс — своя доля ядер
    keep = os.environ.get("XLA_FLAGS")
    per = max(1, cpu_count() // len(chunks))
    if per <= 2:
        os.environ["XLA_FLAGS"] = ((keep or "") + " --xla_cpu_multi_thread_eigen=false").strip()
    try:
        with ProcessPoolExecutor(len(chunks), mp_context=ctx) as ex:
            futs = [ex.submit(_rans_chunk, (mesh, mach, re, chord, xr, alphas, c, Q0, rref, max_iter, conv,
                                            queue, skip_gen, stop_ev)) for c in chunks]
            pending = set(futs)
            last_skip = False
            while pending:
                done, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)
                for f in done:
                    for i, r in f.result():
                        out[i] = r
                while True:
                    try:
                        msg = queue.get_nowait()
                    except Exception:                       # noqa: BLE001
                        break
                    if msg[0] == "iter":
                        d = msg[1]
                        if control is not None and control.monitor is not None:
                            control.monitor(d)
                        if callback and callback(count[0], n, d["it"], d["rn"], None):
                            stop_ev.set()
                    elif msg[0] == "done":
                        count[0] += 1
                        if callback:
                            class _R:                       # лёгкая сводка для строки состояния
                                pass
                            rr = _R()
                            rr.alpha, rr.cl, rr.status = msg[2], msg[3], msg[4]
                            if callback(count[0], n, 0, 0.0, rr):
                                stop_ev.set()
                if control is not None:
                    if control.stop:
                        stop_ev.set()
                    if control.skip and not last_skip:
                        skip_gen.value += 1
                        control.skip = False
                    last_skip = False
    finally:
        if keep is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = keep
        mgr.shutdown()
    return out, mesh
