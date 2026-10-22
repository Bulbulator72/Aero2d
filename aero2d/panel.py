"""Панельный метод в формулировке через функцию тока (как в XFOIL):
линейно распределённая завихренность на плоских панелях, условие
psi(узел) = Psi0 на всех узлах, условие Кутта и экстраполяция вихря на
острой задней кромке. Порядок точности — второй (проверено на точных
решениях Кармана–Треффца). Сжимаемость — поправка Кармана–Цяня.

Обход контура: задняя кромка → верх → передняя кромка → низ → задняя кромка.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .geometry import Airfoil


@dataclass
class PanelResult:
    alpha: float
    mach: float
    x: np.ndarray         # узлы
    y: np.ndarray
    cp: np.ndarray        # в узлах, с поправкой на сжимаемость
    cp_inc: np.ndarray    # несжимаемый
    ue: np.ndarray        # касательная скорость в узлах (несжимаемая, U∞=1), со знаком обхода
    cl: float             # по интегралу давления
    cl_gamma: float       # по циркуляции (Жуковский), несжимаемый
    cm: float             # относительно 1/4 хорды, кабрирование > 0
    cd_p: float           # интеграл давления (в невязком потоке должен быть ≈ 0)


def panel_geometry(x, y):
    dx, dy = np.diff(x), np.diff(y)
    length = np.hypot(dx, dy)
    tx, ty = dx / length, dy / length
    nx, ny = ty, -tx                # наружная нормаль при обходе против часовой стрелки
    return length, tx, ty, nx, ny


def _r2lnr(r2):
    """r^2 ln r с безопасным пределом в нуле."""
    safe = np.where(r2 > 0, r2, 1.0)
    return np.where(r2 > 0, 0.5 * r2 * np.log(safe), 0.0)


def _lnr(r2):
    safe = np.where(r2 > 0, r2, 1.0)
    return np.where(r2 > 0, 0.5 * np.log(safe), 0.0)


def psi_influence(x, y, xp, yp):
    """Функция тока в точках (xp, yp) от единичных узловых вихрей
    (линейное распределение по панелям). Возвращает матрицу n_pts x (N+1)."""
    L, tx, ty, _, _ = panel_geometry(x, y)
    N = len(L)
    rx = xp[:, None] - x[None, :-1]
    ry = yp[:, None] - y[None, :-1]
    xl = rx * tx + ry * ty
    yl = -rx * ty + ry * tx
    r1 = xl**2 + yl**2
    r2 = (xl - L) ** 2 + yl**2
    dth = np.arctan2(yl, xl - L) - np.arctan2(yl, xl)
    F0 = xl * _lnr(r1) - (xl - L) * _lnr(r2) - L + yl * dth        # ∫ ln r ds
    F1 = xl * F0 - (0.5 * (_r2lnr(r1) - _r2lnr(r2)) - (r1 - r2) / 4)  # ∫ s ln r ds
    k = 1.0 / (2 * np.pi)
    P = np.zeros((len(xp), N + 1))
    P[:, :-1] += k * (F0 - F1 / L)
    P[:, 1:] += k * F1 / L
    return P


def karman_tsien(cp0, mach):
    if mach <= 0:
        return cp0
    b = np.sqrt(1 - mach**2)
    return cp0 / (b + mach**2 / (1 + b) * cp0 / 2)


def critical_cp(mach, gamma=1.4):
    """Cp, при котором местная скорость звуковая."""
    if mach <= 0:
        return -np.inf
    return 2 / (gamma * mach**2) * (((2 + (gamma - 1) * mach**2) / (gamma + 1)) ** (gamma / (gamma - 1)) - 1)


def vortex_velocity_influence(x, y, xp, yp):
    """Скорость (U, V) в точках (xp, yp) от единичных узловых вихрей с линейным
    распределением по панелям (та же модель, что в psi_influence).
    Возвращает две матрицы n_pts x (N+1). Точки не должны лежать на панелях."""
    L, tx, ty, _, _ = panel_geometry(x, y)
    N = len(L)
    rx = xp[:, None] - x[None, :-1]
    ry = yp[:, None] - y[None, :-1]
    xl = rx * tx + ry * ty
    yl = -rx * ty + ry * tx
    r1 = xl**2 + yl**2
    r2 = (xl - L) ** 2 + yl**2
    dth = np.arctan2(yl, xl - L) - np.arctan2(yl, xl)
    lnr = 0.5 * np.log(r1 / r2)
    I0, I1 = dth, xl * dth - yl * lnr
    J0, J1 = lnr, xl * lnr - L + yl * dth
    k = 1.0 / (2 * np.pi)
    ua, ub = k * (I0 - I1 / L), k * I1 / L
    va, vb = -k * (J0 - J1 / L), -k * J1 / L
    U = np.zeros((len(xp), N + 1))
    V = np.zeros((len(xp), N + 1))
    U[:, :-1] += ua * tx - va * ty
    U[:, 1:] += ub * tx - vb * ty
    V[:, :-1] += ua * ty + va * tx
    V[:, 1:] += ub * ty + vb * tx
    return U, V


def source_influence(x, y, xp, yp, self_pairs=None):
    """Влияние постоянных по панелям источников единичной интенсивности
    (панели — отрезки ломаной x, y) в точках (xp, yp).
    Возвращает (Psi, U, V), каждая n_pts x N.

    Функция тока источника многозначна; разрез каждой панели направлен по её
    «правой» нормали (для обхода тела против часовой стрелки — наружу), так что
    внутри тела и на его узлах функция тока однозначна.
    self_pairs — список (точка, панель_до, панель_после) для точек, лежащих на
    стыке двух панелей: логарифмическая особенность скорости в них заменяется
    регуляризованным значением."""
    L, tx, ty, _, _ = panel_geometry(x, y)
    rx = xp[:, None] - x[None, :-1]
    ry = yp[:, None] - y[None, :-1]
    xl = rx * tx + ry * ty
    yl = -rx * ty + ry * tx
    r1 = xl**2 + yl**2
    r2 = (xl - L) ** 2 + yl**2
    k = 1.0 / (2 * np.pi)
    w1, w2 = -xl, L - xl
    with np.errstate(divide="ignore", invalid="ignore"):
        F = lambda w, r: w * np.arctan2(w, yl) - yl * _lnr(r)
        Psi = k * (F(w2, r2) - F(w1, r1))
        lnr = 0.5 * np.log(np.where(r1 > 0, r1, 1.0) / np.where(r2 > 0, r2, 1.0))
    dth = np.arctan2(yl, xl - L) - np.arctan2(yl, xl)
    ul, vl = k * lnr, k * dth
    if self_pairs is not None:
        # Точка на стыке панелей: скорость кусочно-постоянного слоя источников
        # здесь логарифмически сингулярна, (σ_a − σ_b)·ln ε / 2π. Для гладкого
        # слоя этот член конечен; ε выбрано так, чтобы реакция на самую быструю
        # («шахматную») моду m совпадала с непрерывным слоем: u = π m / (2 Δs),
        # что даёт ln(Δs/ε) = π²/4. Без этого члена шахматная мода не
        # подавляется связью и след «разваливается» (H < 1).
        c = np.exp(-np.pi**2 / 4)
        for ip, ja, jb in self_pairs:
            for j in (ja, jb):
                if j is not None:
                    ul[ip, j] = 0.0
                    vl[ip, j] = 0.0
            if ja is not None and jb is not None:
                eps = c * np.sqrt(L[ja] * L[jb])
                ul[ip, ja] = k * np.log(L[ja] / eps)
                ul[ip, jb] = -k * np.log(L[jb] / eps)
            elif ja is not None:
                ul[ip, ja] = k * np.log(1.0 / c)
            elif jb is not None:
                ul[ip, jb] = -k * np.log(1.0 / c)
    U = ul * tx - vl * ty
    V = ul * ty + vl * tx
    return Psi, U, V


class PanelSolver:
    """Матрица собирается и факторизуется один раз; любой угол атаки —
    суперпозиция двух базовых решений (поток вдоль x и вдоль y)."""

    def __init__(self, airfoil: Airfoil):
        x, y = np.asarray(airfoil.x, float), np.asarray(airfoil.y, float)
        gap = np.hypot(x[0] - x[-1], y[0] - y[-1])
        if gap > 1e-9:
            raise ValueError("Пока поддерживается только замкнутая задняя кромка "
                             f"(зазор {gap:.2e}); используйте close_trailing_edge().")
        self.af = airfoil
        self.x, self.y = x, y
        self.L, self.tx, self.ty, self.nx, self.ny = panel_geometry(x, y)
        N = len(self.L)               # панелей; узлов N+1, узел N совпадает с 0
        P = psi_influence(x, y, x[:-1], y[:-1])
        A = np.zeros((N + 2, N + 2))
        A[:N, :N + 1] = P
        A[:N, N + 1] = -1.0           # неизвестная Psi0
        A[N, 0] = A[N, N] = 1.0       # Кутта
        A[N + 1, [0, 1, 2]] = [1.0, -2.0, 1.0]       # экстраполяция на острой кромке
        A[N + 1, [N, N - 1, N - 2]] += [-1.0, 2.0, -1.0]
        self.cond = np.linalg.cond(A)
        Ainv = np.linalg.inv(A)
        self.Ainv = Ainv
        self.N = N
        b_x = np.zeros(N + 2); b_x[:N] = -y[:-1]     # psi∞ = y cos a - x sin a
        b_y = np.zeros(N + 2); b_y[:N] = x[:-1]
        self.g_x = (Ainv @ b_x)[:N + 1]
        self.g_y = (Ainv @ b_y)[:N + 1]

    def gamma(self, alpha_deg):
        a = np.radians(alpha_deg)
        return np.cos(a) * self.g_x + np.sin(a) * self.g_y

    def solve(self, alpha_deg: float, mach: float = 0.0) -> PanelResult:
        a = np.radians(alpha_deg)
        ca, sa = np.cos(a), np.sin(a)
        g = self.gamma(alpha_deg)
        ue = g                       # внутри тела поток покоится → скорость снаружи = gamma
        cp0 = 1 - ue**2
        cp = karman_tsien(cp0, mach)
        L, nx, ny = self.L, self.nx, self.ny
        ca_, cb_ = cp[:-1], cp[1:]
        cpm = 0.5 * (ca_ + cb_)
        fx = -np.sum(cpm * nx * L)
        fy = -np.sum(cpm * ny * L)
        # момент: точный интеграл произведения двух линейных функций
        x, y = self.x, self.y
        ra = (x[:-1] - 0.25) * ny - y[:-1] * nx
        rb = (x[1:] - 0.25) * ny - y[1:] * nx
        cm = np.sum(L * (ca_ * ra / 3 + (ca_ * rb + cb_ * ra) / 6 + cb_ * rb / 3))
        cl = fy * ca - fx * sa
        cd = fx * ca + fy * sa
        circ = np.sum(0.5 * (g[:-1] + g[1:]) * L)
        return PanelResult(alpha_deg, mach, x, y, cp, cp0, ue, cl, 2 * circ, cm, cd)

    def polar(self, alphas, mach: float = 0.0):
        return [self.solve(a, mach) for a in alphas]


def close_trailing_edge(af: Airfoil, blend: float = 0.1) -> Airfoil:
    """Замыкает заднюю кромку, плавно убирая зазор на последних `blend` хорды."""
    x, y = af.x.copy(), af.y.copy()
    dy = y[0] - y[-1]
    if abs(dy) < 1e-12:
        return af
    ile = np.argmin(x)
    w = np.clip((x - (1 - blend)) / blend, 0, 1) ** 2
    y[:ile] -= 0.5 * dy * w[:ile]
    y[ile:] += 0.5 * dy * w[ile:]
    y[0] = y[-1] = 0.5 * (y[0] + y[-1])
    x[-1] = x[0]
    return Airfoil(x, y, af.name + " (closed TE)")
