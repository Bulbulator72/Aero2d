"""Пакетный решатель двумерных уравнений Эйлера на JAX.

Конечные объёмы на структурированной O-сетке (центры ячеек):
  * поток HLLC, реконструкция MUSCL по примитивным переменным с ограничителем
    ван Албады;
  * стенка — зеркальные фиктивные ячейки, внешняя граница — характеристические
    условия по инвариантам Римана с поправкой на точечный вихрь;
  * установление по локальному шагу, 4-стадийная схема Рунге–Кутты (Джеймсон)
    с неявным сглаживанием невязки (БПФ по периодическому направлению,
    прогонка по радиальному);
  * пакетирование: все случаи (M, alpha) на одной сетке считаются
    одновременно (jax.vmap); точность — float32 или float64.

Обезразмеривание: rho∞ = 1, a∞ = 1, p∞ = 1/gamma, |V∞| = M.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .grid import OGrid

GAMMA = 1.4


# ---------------------------------------------------------------------------
# Метрики (считаются в float64 в NumPy, затем приводятся к рабочей точности)
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    vol: np.ndarray      # (ni, nc)            площади ячеек
    Si: np.ndarray       # (2, ni, nc)         нормаль грани между ячейками i-1 и i (к росту i)
    Sj: np.ndarray       # (2, ni, nc+1)       нормаль грани j (к росту j), j=0 — стенка
    xc: np.ndarray       # (ni, nc)            центры ячеек
    yc: np.ndarray
    xw: np.ndarray       # (ni,) середины граней стенки
    yw: np.ndarray


def compute_metrics(g: OGrid) -> Metrics:
    x, y = g.x.astype(np.float64), g.y.astype(np.float64)
    vol = g.cell_areas()
    # i-грань: ребро (i,j)→(i,j+1); нормаль (−dy, dx) смотрит к росту i
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    Si = np.stack([-dy, dx])
    # j-грань: ребро (i,j)→(i+1,j); нормаль (dy, −dx) смотрит к росту j (от тела)
    x1, y1 = np.roll(x, -1, 0), np.roll(y, -1, 0)
    dx = x1 - x
    dy = y1 - y
    Sj = np.stack([dy, -dx])
    xc = 0.25 * (x[:, :-1] + x[:, 1:] + x1[:, :-1] + x1[:, 1:])
    yc = 0.25 * (y[:, :-1] + y[:, 1:] + y1[:, :-1] + y1[:, 1:])
    xw = 0.5 * (x[:, 0] + x1[:, 0])
    yw = 0.5 * (y[:, 0] + y1[:, 0])
    return Metrics(vol, Si, Sj, xc, yc, xw, yw)


# ---------------------------------------------------------------------------
# Термодинамика и поток HLLC
# ---------------------------------------------------------------------------
def cons_to_prim(U):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (GAMMA - 1) * (U[3] - 0.5 * rho * (u * u + v * v))
    return jnp.stack([rho, u, v, p])


def prim_to_cons(W):
    rho, u, v, p = W
    return jnp.stack([rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v)])


def hllc_flux(WL, WR, S):
    """Поток через грань с вектором площади S (2, ...) — уже умноженный на длину."""
    area = jnp.sqrt(S[0] ** 2 + S[1] ** 2)
    nx, ny = S[0] / area, S[1] / area
    rL, uL, vL, pL = WL
    rR, uR, vR, pR = WR
    qL = uL * nx + vL * ny
    qR = uR * nx + vR * ny
    cL = jnp.sqrt(GAMMA * pL / rL)
    cR = jnp.sqrt(GAMMA * pR / rR)
    EL = pL / (GAMMA - 1) + 0.5 * rL * (uL**2 + vL**2)
    ER = pR / (GAMMA - 1) + 0.5 * rR * (uR**2 + vR**2)
    # оценки скоростей волн (Эйнфельдт, через средние по Роу)
    sl, sr = jnp.sqrt(rL), jnp.sqrt(rR)
    qt = (sl * qL + sr * qR) / (sl + sr)
    HL = (EL + pL) / rL
    HR = (ER + pR) / rR
    Ht = (sl * HL + sr * HR) / (sl + sr)
    ut = (sl * uL + sr * uR) / (sl + sr)
    vt = (sl * vL + sr * vR) / (sl + sr)
    ct = jnp.sqrt(jnp.maximum((GAMMA - 1) * (Ht - 0.5 * (ut**2 + vt**2)), 1e-12))
    SL = jnp.minimum(qL - cL, qt - ct)
    SR = jnp.maximum(qR + cR, qt + ct)
    SM = (pR - pL + rL * qL * (SL - qL) - rR * qR * (SR - qR)) / (rL * (SL - qL) - rR * (SR - qR))

    def F(r, u, v, p, E, q):
        return jnp.stack([r * q, r * u * q + p * nx, r * v * q + p * ny, (E + p) * q])

    def Ustar(r, u, v, p, E, q, Sk):
        f = r * (Sk - q) / (Sk - SM)
        return jnp.stack([f,
                          f * (u + (SM - q) * nx),
                          f * (v + (SM - q) * ny),
                          f * (E / r + (SM - q) * (SM + p / (r * (Sk - q))))])

    FL = F(rL, uL, vL, pL, EL, qL)
    FR = F(rR, uR, vR, pR, ER, qR)
    UL = jnp.stack([rL, rL * uL, rL * vL, EL])
    UR = jnp.stack([rR, rR * uR, rR * vR, ER])
    FsL = FL + SL * (Ustar(rL, uL, vL, pL, EL, qL, SL) - UL)
    FsR = FR + SR * (Ustar(rR, uR, vR, pR, ER, qR, SR) - UR)
    flux = jnp.where(SL >= 0, FL, jnp.where(SM >= 0, FsL, jnp.where(SR > 0, FsR, FR)))
    return flux * area


LIMITER = "smooth"


def van_albada(a, b, eps):
    """Ограничитель наклона ван Албады.
    "smooth" (по умолчанию) — дифференцируемая форма с порогом eps: нужна
    неявному методу Ньютона (TVD-переключатель даёт «дребезг» и останавливает
    сходимость); "van_albada" — TVD-форма; "minmod" — самый диссипативный."""
    eps = eps + 1e-20
    if LIMITER == "minmod":
        return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)
    if LIMITER == "smooth":
        return ((b * b + eps) * a + (a * a + eps) * b) / (a * a + b * b + 2 * eps)
    return jnp.where(a * b > 0, ((b * b + eps) * a + (a * a + eps) * b) / (a * a + b * b + 2 * eps), 0.0)


# ---------------------------------------------------------------------------
# Граничные условия
# ---------------------------------------------------------------------------
def wall_ghost(W, Sj0):
    """Зеркальные фиктивные ячейки (2 слоя) для непротекания."""
    area = jnp.sqrt(Sj0[0] ** 2 + Sj0[1] ** 2)
    nx, ny = Sj0[0] / area, Sj0[1] / area

    def mirror(Wc):
        r, u, v, p = Wc
        vn = u * nx + v * ny
        return jnp.stack([r, u - 2 * vn * nx, v - 2 * vn * ny, p])

    return mirror(W[:, :, 1]), mirror(W[:, :, 0])   # (j=-2, j=-1)


def farfield_ghost(Wi, Sjf, winf, xf, yf, circ):
    """Характеристическое условие по инвариантам Римана.
    winf — примитивы набегающего потока, circ — циркуляция для поправки на
    точечный вихрь (центр в (0.25, 0))."""
    area = jnp.sqrt(Sjf[0] ** 2 + Sjf[1] ** 2)
    nx, ny = Sjf[0] / area, Sjf[1] / area
    rinf, uinf, vinf, pinf = winf
    minf2 = (uinf**2 + vinf**2) / (GAMMA * pinf / rinf)
    # поправка на вихрь (с учётом сжимаемости, Usab/Thomas–Salas)
    # (только для дозвукового набегающего потока; Γ > 0 — по часовой стрелке)
    dxv, dyv = xf - 0.25, yf
    alpha = jnp.arctan2(vinf, uinf)
    phi = jnp.arctan2(dyv, dxv)
    rr = jnp.sqrt(dxv**2 + dyv**2)
    beta = jnp.sqrt(jnp.maximum(1 - minf2, 1e-6))
    fac = jnp.where(minf2 < 1, circ * beta / (2 * jnp.pi * rr * (1 - minf2 * jnp.sin(phi - alpha) ** 2)), 0.0)
    ub = uinf + fac * jnp.sin(phi)
    vb = vinf - fac * jnp.cos(phi)
    # давление/плотность — из постоянства полной энтальпии и энтропии
    q2inf = uinf**2 + vinf**2
    q2b = ub**2 + vb**2
    cinf2 = GAMMA * pinf / rinf
    cb2 = jnp.maximum(cinf2 + 0.5 * (GAMMA - 1) * (q2inf - q2b), 1e-6)
    pb = pinf * (cb2 / cinf2) ** (GAMMA / (GAMMA - 1))
    rb = rinf * (cb2 / cinf2) ** (1 / (GAMMA - 1))
    # характеристики
    r, u, v, p = Wi
    ci = jnp.sqrt(GAMMA * p / r)
    cb = jnp.sqrt(GAMMA * pb / rb)
    qni = u * nx + v * ny
    qnb = ub * nx + vb * ny
    Rp = qni + 2 * ci / (GAMMA - 1)
    Rm = qnb - 2 * cb / (GAMMA - 1)
    qn = 0.5 * (Rp + Rm)
    c = 0.25 * (GAMMA - 1) * (Rp - Rm)
    outflow = qn > 0
    # касательная скорость и энтропия — с той стороны, откуда приходит информация
    ut = jnp.where(outflow, u - qni * nx, ub - qnb * nx)
    vt = jnp.where(outflow, v - qni * ny, vb - qnb * ny)
    s = jnp.where(outflow, p / r**GAMMA, pb / rb**GAMMA)
    rho = (c * c / (GAMMA * s)) ** (1 / (GAMMA - 1))
    pres = rho * c * c / GAMMA
    Wb = jnp.stack([rho, ut + qn * nx, vt + qn * ny, pres])
    # при сверхзвуковом втекании/истечении — полностью внешние/внутренние данные
    qnorm_inf = uinf * nx + vinf * ny
    sup_in = (qnorm_inf < 0) & (jnp.abs(qnorm_inf) > jnp.sqrt(cinf2))
    sup_out = (qni > 0) & (qni > ci)
    Wb = jnp.where(sup_in, jnp.stack([rb, ub, vb, pb]), Wb)
    Wb = jnp.where(sup_out, Wi, Wb)
    return Wb


# ---------------------------------------------------------------------------
# Невязка
# ---------------------------------------------------------------------------
def _recon(Wm, W0, Wp, Wpp, eps):
    """Значения слева/справа на грани между W0 и Wp (MUSCL, ван Албада)."""
    sL = van_albada(W0 - Wm, Wp - W0, eps)
    sR = van_albada(Wp - W0, Wpp - Wp, eps)
    return W0 + 0.5 * sL, Wp - 0.5 * sR


class EulerSolver:
    def __init__(self, grid: OGrid, dtype=jnp.float32, cfl: float = 2.5,
                 irs_eps: float = 0.6, order: int = 2, vortex_correction: bool = True,
                 mg_levels: int = 1, lim_k: float = 0.02):
        if dtype == jnp.float64:
            jax.config.update("jax_enable_x64", True)
        self.dtype = dtype
        self.grid = grid
        m = compute_metrics(grid)
        self.m_np = m
        c = lambda a: jnp.asarray(a, dtype)
        self.vol, self.Si, self.Sj = c(m.vol), c(m.Si), c(m.Sj)
        self.xf, self.yf = c(grid.x[:, -1]), c(grid.y[:, -1])
        # середины внешних граней
        self.xf = c(0.5 * (grid.x[:, -1] + np.roll(grid.x[:, -1], -1)))
        self.yf = c(0.5 * (grid.y[:, -1] + np.roll(grid.y[:, -1], -1)))
        self.xw, self.yw = c(m.xw), c(m.yw)
        self.cfl, self.irs_eps, self.order, self.lim_k = cfl, irs_eps, order, lim_k
        self.vortex = vortex_correction
        self.ni, self.nc = m.vol.shape
        # неявное сглаживание невязки: знаменатель для БПФ по i
        k = np.arange(self.ni)
        self.irs_den_i = c(1 + 2 * irs_eps * (1 - np.cos(2 * np.pi * k / self.ni)))
        # многосеточный метод (FAS, ЭКСПЕРИМЕНТАЛЬНО: при M < 0.6 пока неустойчив;
        # для серийных расчётов используйте implicit.ImplicitEuler)
        self.coarse = None
        if mg_levels > 1:
            if self.ni % 2 or self.nc % 2:
                raise ValueError("Для многосеточного метода число ячеек по i и j должно быть чётным")
            cg = OGrid(grid.x[::2, ::2], grid.y[::2, ::2])
            self.coarse = EulerSolver(cg, dtype, cfl, irs_eps, order=1,
                                      vortex_correction=vortex_correction,
                                      mg_levels=mg_levels - 1)

    # --- свободный поток -------------------------------------------------
    def freestream(self, mach, alpha_deg):
        a = jnp.deg2rad(alpha_deg)
        return jnp.stack([jnp.ones_like(mach), mach * jnp.cos(a), mach * jnp.sin(a),
                          jnp.ones_like(mach) / GAMMA]).astype(self.dtype)

    # --- невязка для одного случая -------------------------------------
    def residual(self, U, winf, circ):
        W = cons_to_prim(U)                             # (4, ni, nc)
        mach = jnp.sqrt(winf[1] ** 2 + winf[2] ** 2)
        scale = jnp.stack([winf[0], mach, mach, winf[3]])  # характерные величины переменных
        eps = ((self.lim_k * scale) ** 2)[:, None, None]
        g2, g1 = wall_ghost(W, self.Sj[:, :, 0])
        Wf = farfield_ghost(W[:, :, -1], self.Sj[:, :, -1], winf, self.xf, self.yf,
                            circ * self.vortex)
        # расширенный массив по j: [g2, g1, W..., Wf, Wf]
        We = jnp.concatenate([g2[:, :, None], g1[:, :, None], W, Wf[:, :, None], Wf[:, :, None]], axis=2)
        # --- потоки по j (грани j=0..nc) ---
        if self.order == 2:
            WL, WR = _recon(We[:, :, 0:-3], We[:, :, 1:-2], We[:, :, 2:-1], We[:, :, 3:], eps)
        else:
            WL, WR = We[:, :, 1:-2], We[:, :, 2:-1]
        WL = WL.at[0].set(jnp.maximum(WL[0], 1e-6)).at[3].set(jnp.maximum(WL[3], 1e-6))
        WR = WR.at[0].set(jnp.maximum(WR[0], 1e-6)).at[3].set(jnp.maximum(WR[3], 1e-6))
        Fj = hllc_flux(WL, WR, self.Sj)                  # (4, ni, nc+1)
        # --- потоки по i (периодично; грань i — между i-1 и i) ---
        Wm2, Wm1, Wp1 = jnp.roll(W, 2, 1), jnp.roll(W, 1, 1), jnp.roll(W, -1, 1)
        if self.order == 2:
            WL, WR = _recon(Wm2, Wm1, W, Wp1, eps)
        else:
            WL, WR = Wm1, W
        WL = WL.at[0].set(jnp.maximum(WL[0], 1e-6)).at[3].set(jnp.maximum(WL[3], 1e-6))
        WR = WR.at[0].set(jnp.maximum(WR[0], 1e-6)).at[3].set(jnp.maximum(WR[3], 1e-6))
        Fi = hllc_flux(WL, WR, self.Si)                  # (4, ni, nc)
        R = (jnp.roll(Fi, -1, 1) - Fi) + (Fj[:, :, 1:] - Fj[:, :, :-1])
        return R, W

    def local_dt(self, W):
        r, u, v, p = W
        c = jnp.sqrt(GAMMA * p / r)
        Si_c = 0.5 * (self.Si + jnp.roll(self.Si, -1, 1))
        Sj_c = 0.5 * (self.Sj[:, :, 1:] + self.Sj[:, :, :-1])
        li = jnp.abs(u * Si_c[0] + v * Si_c[1]) + c * jnp.sqrt(Si_c[0] ** 2 + Si_c[1] ** 2)
        lj = jnp.abs(u * Sj_c[0] + v * Sj_c[1]) + c * jnp.sqrt(Sj_c[0] ** 2 + Sj_c[1] ** 2)
        return self.cfl * self.vol / (li + lj)

    def smooth(self, R):
        if self.irs_eps <= 0:
            return R
        # по i — периодическая трёхдиагональная система с постоянными коэффициентами → БПФ
        Rh = jnp.fft.fft(R, axis=1)
        R = jnp.real(jnp.fft.ifft(Rh / self.irs_den_i[None, :, None], axis=1)).astype(self.dtype)
        # по j — прогонка (−eps, 1+2eps, −eps)
        e = self.irs_eps
        n = self.nc
        dl = jnp.full((n,), -e, self.dtype).at[0].set(0.0)
        du = jnp.full((n,), -e, self.dtype).at[-1].set(0.0)
        d = jnp.full((n,), 1 + 2 * e, self.dtype).at[0].set(1 + e).at[-1].set(1 + e)
        Rt = jnp.moveaxis(R, 2, 0).reshape(n, -1)
        sol = jax.lax.linalg.tridiagonal_solve(dl, d, du, Rt)
        return jnp.moveaxis(sol.reshape(n, 4, self.ni), 0, 2)

    # --- силы -----------------------------------------------------------
    def wall_pressure(self, W):
        return 1.5 * W[3, :, 0] - 0.5 * W[3, :, 1]

    def forces(self, U, winf):
        W = cons_to_prim(U)
        pw = self.wall_pressure(W)
        S = self.Sj[:, :, 0]                       # нормаль из тела в поток × длина
        q = 0.5 * (winf[1] ** 2 + winf[2] ** 2)   # rho∞ = 1
        cp = (pw - winf[3]) / q
        fx = -jnp.sum(cp * S[0])
        fy = -jnp.sum(cp * S[1])
        mach = jnp.sqrt(2 * q)
        ca, sa = winf[1] / mach, winf[2] / mach
        cl = fy * ca - fx * sa
        cd = fx * ca + fy * sa
        cm = jnp.sum(cp * ((self.xw - 0.25) * S[1] - self.yw * S[0]))
        return cl, cd, cm, cp

    # --- шаг установления (одна итерация RK) ---------------------------
    def _iterate(self, U, winf, circ, P=None):
        """Одна итерация 4-стадийной схемы RK; P — вынуждающий член FAS."""
        U0 = U
        W = cons_to_prim(U)
        dt = self.local_dt(W)
        for a in (0.25, 1 / 3, 0.5, 1.0):
            R, _ = self.residual(U, winf, circ)
            if P is not None:
                R = R + P
            R = self.smooth(R)
            U = U0 - a * (dt / self.vol)[None] * R
        res = jnp.sqrt(jnp.mean(((U[0] - U0[0]) / dt) ** 2))
        return U, res

    # --- многосеточный цикл (FAS, V-цикл) -------------------------------
    @staticmethod
    def _sum4(a):
        return a[..., 0::2, 0::2] + a[..., 1::2, 0::2] + a[..., 0::2, 1::2] + a[..., 1::2, 1::2]

    @staticmethod
    def _prolong(d):
        """Билинейная интерполяция поправки с грубой сетки (веса 3/4, 1/4)."""
        # по i — периодично
        e = 0.75 * d + 0.25 * jnp.roll(d, 1, axis=1)
        o = 0.75 * d + 0.25 * jnp.roll(d, -1, axis=1)
        d = jnp.stack([e, o], axis=2).reshape(d.shape[0], -1, d.shape[2])
        # по j — со «снятием» значения на границах
        dm = jnp.concatenate([d[:, :, :1], d[:, :, :-1]], axis=2)
        dp = jnp.concatenate([d[:, :, 1:], d[:, :, -1:]], axis=2)
        e = 0.75 * d + 0.25 * dm
        o = 0.75 * d + 0.25 * dp
        return jnp.stack([e, o], axis=3).reshape(d.shape[0], d.shape[1], -1)

    def _cycle(self, U, winf, circ, P=None):
        U, res = self._iterate(U, winf, circ, P)
        if self.coarse is not None:
            cs = self.coarse
            R, _ = self.residual(U, winf, circ)
            if P is not None:
                R = R + P
            Uc0 = self._sum4(U * self.vol[None]) / self._sum4(self.vol)[None]
            Rc, _ = cs.residual(Uc0, winf, circ)
            Pc = self._sum4(R) - Rc
            Uc, _ = cs._cycle(Uc0, winf, circ, Pc)
            dU = self._prolong(Uc - Uc0)
            U = U + dU
            U, _ = self._iterate(U, winf, circ, P)
        return U, res

    def init_state(self, winf):
        return jnp.broadcast_to(prim_to_cons(winf[:, None, None]), (4, self.ni, self.nc)).astype(self.dtype)

    def solve(self, machs, alphas, n_iter=4000, chunk=200, tol=1e-7, verbose=True, U0=None,
              tol_cl=2e-5, tol_cd=2e-6):
        """Пакетный расчёт. machs, alphas — одномерные массивы одинаковой длины.
        Остановка: невязка упала в 1/tol раз ИЛИ за последний блок итераций
        Cl и Cd всех случаев изменились меньше tol_cl / tol_cd."""
        machs = jnp.asarray(machs, self.dtype)
        alphas = jnp.asarray(alphas, self.dtype)
        winf = jax.vmap(self.freestream)(machs, alphas)          # (B, 4)
        U = jax.vmap(self.init_state)(winf) if U0 is None else U0
        circ = jnp.zeros(machs.shape, self.dtype)

        def single_chunk(U, winf, circ):
            def body(k, carry):
                U, _, circ = carry
                U, res = self._cycle(U, winf, circ)
                cl, _, _, _ = self.forces(U, winf)
                mach = jnp.sqrt(winf[1] ** 2 + winf[2] ** 2)
                circ = 0.5 * cl * mach            # Γ = ½ Cl c V∞, c = 1
                return U, res, circ
            return jax.lax.fori_loop(0, chunk, body, (U, jnp.zeros((), self.dtype), circ))

        step = jax.jit(jax.vmap(single_chunk))
        clcd = jax.jit(jax.vmap(lambda U, w: self.forces(U, w)[:2]))
        hist, clh, cdh = [], [], []
        res0 = None
        self.iterations = 0
        for it in range(0, n_iter, chunk):
            U, res, circ = step(U, winf, circ)
            self.iterations = it + chunk
            res = np.asarray(res)
            if res0 is None:
                res0 = res
            cl, cd = (np.asarray(v) for v in clcd(U, winf))
            hist.append(res); clh.append(cl); cdh.append(cd)
            if verbose:
                print(f"  iter {it + chunk:6d}  log10(res/res0) max {np.max(np.log10(res / res0)):+.2f}"
                      f"  Cl {np.array2string(cl, precision=4)}")
            if not np.all(np.isfinite(res)):
                break
            if np.all(res / res0 < tol):
                break
            if len(clh) > 2 and np.all(np.abs(clh[-1] - clh[-2]) < tol_cl) \
                    and np.all(np.abs(cdh[-1] - cdh[-2]) < tol_cd):
                break
        self.cl_history = np.array(clh)
        if not np.all(np.isfinite(np.asarray(U))):
            raise FloatingPointError("Решение разошлось (NaN)")
        cl, cd, cm, cp = jax.vmap(self.forces)(U, winf)
        return EulerResult(np.asarray(machs), np.asarray(alphas), np.asarray(cl), np.asarray(cd),
                           np.asarray(cm), np.asarray(cp), np.array(hist), U)


@dataclass
class EulerResult:
    mach: np.ndarray
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    cp: np.ndarray        # (B, ni) на гранях стенки
    res_hist: np.ndarray  # (n_chunks, B)
    U: jnp.ndarray
    converged: np.ndarray | None = None   # признак сходимости по каждому случаю
