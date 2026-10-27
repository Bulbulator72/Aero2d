"""Геометрия профилей.

Соглашение об обходе контура (как в XFOIL): от задней кромки по верхней
поверхности к передней кромке и далее по нижней поверхности обратно к задней
кромке. Первая и последняя точки — задняя кромка (для острой кромки совпадают).
Хорда нормирована на 1, передняя кромка в (0, 0), задняя в (1, 0).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from math import comb


@dataclass
class Airfoil:
    x: np.ndarray
    y: np.ndarray
    name: str = "airfoil"

    @property
    def n(self) -> int:
        return len(self.x)

    def coords(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    @staticmethod
    def from_file(path: str) -> "Airfoil":
        """Читает формат Selig (имя в первой строке, далее x y)."""
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        name = lines[0]
        pts = np.array([[float(v) for v in ln.split()[:2]] for ln in lines[1:]])
        return Airfoil(pts[:, 0], pts[:, 1], name)


# ---------------------------------------------------------------------------
# NACA
# ---------------------------------------------------------------------------
def _cosine_spacing(n: int) -> np.ndarray:
    beta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


def naca4(code: str, n_side: int = 100, closed_te: bool = True) -> Airfoil:
    """NACA 4-значный профиль. n_side — число точек на каждой стороне."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    xc = _cosine_spacing(n_side)
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                  + 0.2843 * xc**3 + a4 * xc**4)
    yc = np.zeros_like(xc)
    dyc = np.zeros_like(xc)
    if m > 0 and p > 0:
        f = xc < p
        yc[f] = m / p**2 * (2 * p * xc[f] - xc[f] ** 2)
        dyc[f] = 2 * m / p**2 * (p - xc[f])
        b = ~f
        yc[b] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xc[b] - xc[b] ** 2)
        dyc[b] = 2 * m / (1 - p) ** 2 * (p - xc[b])
    th = np.arctan(dyc)
    xu, yu = xc - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = xc + yt * np.sin(th), yc - yt * np.cos(th)
    x = np.concatenate([xu[::-1], xl[1:]])
    y = np.concatenate([yu[::-1], yl[1:]])
    if closed_te:
        y[0] = y[-1] = 0.5 * (y[0] + y[-1])
    return Airfoil(x, y, f"NACA {code}")


def _naca_mod_thickness(x, t, le_index, xmax):
    """Толщина модифицированного 4-значного профиля NACA 00tt-IT (Stack, 1939;
    формулы — NASA TM-4741, Ladson): I — индекс радиуса носка, T = 10·xmax —
    положение наибольшей толщины. Возвращает полутолщину."""
    m = xmax
    d1 = float(np.interp(m, [0.2, 0.3, 0.4, 0.5, 0.6], [0.200, 0.234, 0.315, 0.465, 0.700]))
    d0 = 0.002
    e = 1 - m
    # хвостовая часть: y(m) = 0.1, y'(m) = 0
    d2 = (0.294 - 2 * d1 * e) / e**2
    d3 = (d1 * e - 0.196) / e**3
    chi = le_index / 6.0 if le_index < 9 else 10.3933
    a0 = 0.296904 * chi
    # носовая часть: y(m) = 0.1, y'(m) = 0, кривизна в m как у хвостовой
    ypp = 2 * d2 + 6 * d3 * e
    A = np.array([[m, m**2, m**3], [1, 2 * m, 3 * m**2], [0, 2, 6 * m]])
    rhs = np.array([0.1 - a0 * np.sqrt(m), -0.5 * a0 / np.sqrt(m), ypp + 0.25 * a0 * m**-1.5])
    a1, a2, a3 = np.linalg.solve(A, rhs)
    yf = a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3
    xa = 1 - x
    ya = d0 + d1 * xa + d2 * xa**2 + d3 * xa**3
    return (t / 0.2) * np.where(x < m, yf, ya)


def naca_sym(code: str, n_side: int = 100) -> Airfoil:
    """Симметричные профили оперения: «0012-64» (модифицированный 4-значный:
    толщина 12 %, индекс носка 6, наибольшая толщина на 40 %) и «16-012»
    (серия 16: толщина = модифицированный 4-значный с I = 4, T = 5).
    Задняя кромка замыкается."""
    c = code.upper().replace("NACA", "").replace(" ", "")
    if c.startswith("16-"):
        t, le, xm = int(c[3:]) / 100.0, 4, 0.5
    else:
        base, it = c.split("-")
        t, le, xm = int(base[2:]) / 100.0, int(it[0]), int(it[1]) / 10.0
    xc = _cosine_spacing(n_side)
    yt = _naca_mod_thickness(xc, t, le, xm)
    yt = yt - yt[-1] * np.clip((xc - 0.8) / 0.2, 0, 1) ** 2   # острая кромка (меняет только x > 0,8)
    x = np.concatenate([xc[::-1], xc[1:]])
    y = np.concatenate([yt[::-1], -yt[1:]])
    return Airfoil(x, y, f"NACA {c}")


# ---------------------------------------------------------------------------
# CST (Kulfan)
# ---------------------------------------------------------------------------
def _bernstein(n: int, x: np.ndarray) -> np.ndarray:
    return np.array([comb(n, i) * x**i * (1 - x) ** (n - i) for i in range(n + 1)])


def cst(au, al, n_side: int = 100, te_thickness: float = 0.0,
        n1: float = 0.5, n2: float = 1.0) -> Airfoil:
    """Профиль по параметризации CST. au, al — весовые коэффициенты
    верхней и нижней поверхностей (al обычно отрицательные)."""
    au, al = np.asarray(au, float), np.asarray(al, float)
    xc = _cosine_spacing(n_side)
    c = xc**n1 * (1 - xc) ** n2
    yu = c * (au @ _bernstein(len(au) - 1, xc)) + xc * te_thickness / 2
    yl = c * (al @ _bernstein(len(al) - 1, xc)) - xc * te_thickness / 2
    x = np.concatenate([xc[::-1], xc[1:]])
    y = np.concatenate([yu[::-1], yl[1:]])
    return Airfoil(x, y, "CST")


# ---------------------------------------------------------------------------
# Профили Кармана–Треффца / Жуковского с точным решением потенциального обтекания
# ---------------------------------------------------------------------------
class KarmanTrefftz:
    """Окружность радиуса R с центром mu в плоскости zeta, проходящая через
    zeta = c, отображается в профиль с углом задней кромки tau:
        (z - n c)/(z + n c) = ((zeta - c)/(zeta + c))^n,  n = 2 - tau/pi.
    При tau = 0 — профиль Жуковского (острая кромка-«клюв»).

    eps  — параметр толщины (сдвиг центра влево, в долях c),
    delta — параметр кривизны (сдвиг центра вверх, в долях c).
    Координаты, Cp и Cl отнесены к хорде профиля; угол атаки — от хорды.
    """

    def __init__(self, eps=0.1, delta=0.0, tau_deg=0.0, c=1.0):
        self.eps, self.delta, self.tau_deg, self.c = eps, delta, tau_deg, c
        self.nexp = 2.0 - np.radians(tau_deg) / np.pi
        self.mu = complex(-eps * c, delta * c)
        self.R = abs(c - self.mu)
        self.theta_te = np.angle(c - self.mu)
        th = self.theta_te + np.linspace(0, 2 * np.pi, 200001)
        z = self._z(th)
        self.z_te = self.nexp * c
        i = np.argmax(np.abs(z - self.z_te))
        self.z_le = z[i]
        self.chord = abs(self.z_te - self.z_le)
        self.chord_angle = np.angle(self.z_te - self.z_le)
        self._th_fine = th
        self._z_fine = self._normalize(z)

    def _zeta(self, th):
        return self.mu + self.R * np.exp(1j * th)

    def _w(self, zeta):
        return (zeta - self.c) / (zeta + self.c)

    def _z(self, th):
        wn = self._w(self._zeta(th)) ** self.nexp
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.nexp * self.c * (1 + wn) / (1 - wn)

    def _dzdzeta(self, zeta):
        n, c = self.nexp, self.c
        w = self._w(zeta)
        dw = 2 * c / (zeta + c) ** 2
        return n * c * 2 * n * w ** (n - 1) * dw / (1 - w**n) ** 2

    def _normalize(self, z):
        """Передняя кромка в (0,0), хорда 1 вдоль оси x."""
        return (z - self.z_le) * np.exp(-1j * self.chord_angle) / self.chord

    def airfoil(self, n: int = 201) -> Airfoil:
        th = self.theta_te + np.linspace(0, 2 * np.pi, n)
        z = self._normalize(self._z(th))
        z[0] = z[-1] = self._normalize(self.z_te)
        return Airfoil(z.real.copy(), z.imag.copy(),
                       f"Karman-Trefftz eps={self.eps} delta={self.delta} tau={self.tau_deg}")

    def theta_of_point(self, x, y):
        """Параметр окружности для точек, лежащих (почти) на контуре."""
        p = np.asarray(x) + 1j * np.asarray(y)
        return np.array([self._th_fine[np.argmin(np.abs(self._z_fine - pi))]
                         for pi in np.atleast_1d(p)])

    def _alpha_plane(self, alpha_deg):
        return np.radians(alpha_deg) + self.chord_angle

    def cp(self, th, alpha_deg: float):
        """Точный несжимаемый Cp на контуре."""
        a = self._alpha_plane(alpha_deg)
        zeta = self._zeta(th)
        s = zeta - self.mu
        gam = 4 * np.pi * self.R * np.sin(a - self.theta_te)   # U = 1
        dw = np.exp(-1j * a) - self.R**2 * np.exp(1j * a) / s**2 + 1j * gam / (2 * np.pi * s)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.abs(dw / self._dzdzeta(zeta))
        return 1 - q**2

    def cl(self, alpha_deg: float) -> float:
        a = self._alpha_plane(alpha_deg)
        return 2 * 4 * np.pi * self.R * np.sin(a - self.theta_te) / self.chord


def Joukowski(eps=0.1, delta=0.0, c=1.0) -> KarmanTrefftz:
    return KarmanTrefftz(eps, delta, 0.0, c)


# ---------------------------------------------------------------------------
# Подготовка произвольного контура (из файла) к расчёту
# ---------------------------------------------------------------------------
def normalize(af: Airfoil) -> Airfoil:
    """Приводит контур к соглашению пакета: обход задняя кромка → верх →
    передняя кромка → низ → задняя кромка, хорда 1, передняя кромка в (0,0)."""
    x, y = np.asarray(af.x, float), np.asarray(af.y, float)
    # формат Lednicer: две половины, каждая от передней к задней кромке
    if x[0] < 0.5 and np.any(np.diff(x) < -0.5):
        k = int(np.argmax(np.diff(x) < -0.5)) + 1
        up, lo = slice(0, k), slice(k, None)
        x = np.concatenate([x[up][::-1], x[lo][1:] if np.isclose(x[lo][0], x[up][0]) else x[lo]])
        y = np.concatenate([af.y[up][::-1], af.y[lo][1:] if np.isclose(af.x[lo][0], af.x[up][0]) else af.y[lo]])
    # уже нормированные данные (типичные файлы профилей) не трогаем: иначе
    # у профилей с кривизной поворот на угол «геометрической» хорды сдвинет
    # отсчёт угла атаки относительно справочных данных
    already = (abs(x.min()) < 2e-3 and abs(x.max() - 1) < 2e-3
               and abs(0.5 * (y[0] + y[-1])) < 2e-3)
    if not already:
        x, y = _rescale(x, y)
    # обход против часовой стрелки (площадь > 0)
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if area < 0:
        x, y = x[::-1], y[::-1]
    return Airfoil(x, y, af.name)


def _rescale(x, y):
    """Хорда — от задней кромки до самой удалённой от неё точки."""
    xte = 0.5 * np.array([x[0] + x[-1], y[0] + y[-1]])
    d = np.hypot(x - xte[0], y - xte[1])
    ile = int(np.argmax(d))
    le = np.array([x[ile], y[ile]])
    ch = xte - le
    c = np.hypot(*ch)
    ang = np.arctan2(ch[1], ch[0])
    ca, sa = np.cos(-ang), np.sin(-ang)
    xs, ys = (x - le[0]) / c, (y - le[1]) / c
    return xs * ca - ys * sa, xs * sa + ys * ca


def _corners(x, y, angle=30.0):
    """Номера внутренних точек контура, где направление обхода поворачивает больше
    чем на angle градусов (углы ниши закрылка, излом носка дефлектора и т. п.)."""
    d = np.column_stack([np.diff(x), np.diff(y)])
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
    c = np.clip(np.sum(d[:-1] * d[1:], 1), -1.0, 1.0)
    return np.where(np.degrees(np.arccos(c)) > angle)[0] + 1


class _PiecewiseSpline:
    """Сплайн контура по длине дуги, разрезанный в изломах: на каждом гладком
    участке — свой кубический сплайн. Один сплайн через острый угол даёт выбросы
    и петли (у задней кромки основного элемента, в углах ниши закрылка)."""

    def __init__(self, s, x, y, cuts):
        from scipy.interpolate import CubicSpline, make_interp_spline
        self.bounds = np.concatenate([[0], np.asarray(cuts, int), [len(s) - 1]])
        self.sb = s[self.bounds]
        self.parts = []
        for a, b in zip(self.bounds[:-1], self.bounds[1:]):
            ss, xx, yy = s[a:b + 1], x[a:b + 1], y[a:b + 1]
            if len(ss) >= 4:
                self.parts.append((CubicSpline(ss, xx), CubicSpline(ss, yy)))
            else:
                k = len(ss) - 1
                self.parts.append((make_interp_spline(ss, xx, k=k), make_interp_spline(ss, yy, k=k)))

    def __call__(self, st):
        st = np.asarray(st, float)
        j = np.clip(np.searchsorted(self.sb, st, side="right") - 1, 0, len(self.parts) - 1)
        x, y = np.empty_like(st), np.empty_like(st)
        for k, (fx, fy) in enumerate(self.parts):
            m = j == k
            if m.any():
                x[m], y[m] = fx(st[m]), fy(st[m])
        return x, y


def repanel(af: Airfoil, n_side: int = 100, spacing: str = "cosine", keep_frame: bool = False,
            corner_angle: float = 30.0) -> Airfoil:
    """Перераспределяет точки контура: сплайн по длине дуги, n_side точек на
    сторону. spacing="cosine" — сгущение к обеим кромкам; "viscous" — сильное
    сгущение к передней кромке и умеренное к задней (панели у задней кромки
    около 1 % хорды, как в XFOIL): для вязкого расчёта слишком мелкие панели
    у задней кромки делают задачу жёсткой.
    Изломы контура (поворот больше corner_angle градусов: углы ниши, острый
    носок) сохраняются: сплайн разрезается в них, точка излома остаётся узлом."""
    x, y = np.asarray(af.x, float), np.asarray(af.y, float)
    keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(y)) > 1e-10])
    x, y = x[keep], y[keep]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    cuts = _corners(x, y, corner_angle) if corner_angle else np.array([], int)
    sp = _PiecewiseSpline(s, x, y, cuts)
    # передняя кромка — минимум x на сплайне
    sf = np.linspace(0, s[-1], 20001)
    sle = sf[np.argmin(sp(sf)[0])]
    if spacing == "viscous":
        u = np.linspace(0, 1, n_side)               # 0 — передняя, 1 — задняя кромка
        w = 0.55
        g = w * (1 - np.cos(0.5 * np.pi * u)) + (1 - w) * 0.5 * (1 - np.cos(np.pi * u))
        s1 = sle * (1 - g[::-1])
        s2 = sle + (s[-1] - sle) * g
    else:
        b = np.linspace(0, np.pi, n_side)
        s1 = sle * 0.5 * (1 - np.cos(b))
        s2 = sle + (s[-1] - sle) * 0.5 * (1 - np.cos(b))
    st = np.concatenate([s1, s2[1:]])
    # изломы — точно в узлах (ближайший узел переносится в излом)
    for sc in s[cuts]:
        k = int(np.argmin(np.abs(st[1:-1] - sc))) + 1
        st[k] = sc
    st = np.sort(st)
    xo, yo = sp(st)
    out = Airfoil(xo, yo, af.name)
    if keep_frame:            # элементы механизации: без нормировки и поворота
        area = 0.5 * np.sum(out.x[:-1] * out.y[1:] - out.x[1:] * out.y[:-1])
        if area < 0:
            out = Airfoil(out.x[::-1].copy(), out.y[::-1].copy(), af.name)
    else:
        out = normalize(out)
    gap = np.hypot(out.x[0] - out.x[-1], out.y[0] - out.y[-1])
    if gap <= 2e-3 * max(float(np.ptp(out.x)), 1e-12):
        # почти замкнутый контур — концы сводятся в одну точку
        out.x[0] = out.x[-1] = 0.5 * (out.x[0] + out.x[-1])
        out.y[0] = out.y[-1] = 0.5 * (out.y[0] + out.y[-1])
    else:
        # заметный зазор (тупая кромка, конец ниши) — концы не сдвигаются, контур
        # замыкается отрезком: раньше они сводились в середину, и форма у кромки менялась
        out = Airfoil(np.r_[out.x, out.x[0]], np.r_[out.y, out.y[0]], out.name)
    return out


def smooth_airfoil(af: Airfoil, strength: float = 1.0) -> Airfoil:
    """Сглаживание измеренных/оцифрованных координат (шум ~0,1 мм делает
    расчёт пограничного слоя у передней кромки неустойчивым). Каждая сторона —
    сглаживающий сплайн y(u), u = sqrt(x − x_пк) (так передняя кромка гладкая),
    параметр сглаживания — по обобщённой перекрёстной проверке (GCV),
    strength > 1 — сильнее. Контур предварительно нормирован (normalize)."""
    from scipy.interpolate import make_smoothing_spline
    af = normalize(af)
    x, y = np.asarray(af.x, float), np.asarray(af.y, float)
    ile = int(np.argmin(x))
    xle, yle = x[ile], y[ile]
    out = []
    for xs, ys in ((x[:ile + 1][::-1], y[:ile + 1][::-1]), (x[ile:], y[ile:])):
        u = np.sqrt(np.maximum(xs - xle, 0.0))
        keep = np.concatenate([[True], np.diff(u) > 1e-9])
        u, ys = u[keep], ys[keep]
        spl = make_smoothing_spline(u, ys)
        if strength != 1.0:
            lam0 = getattr(spl, "lam", None)
            if lam0 is not None:
                spl = make_smoothing_spline(u, ys, lam=lam0 * strength)
        ysm = spl(u)
        ysm[0] = yle
        out.append((xle + u**2, ysm))
    (xu, yu), (xl, yl) = out
    xx = np.concatenate([xu[::-1], xl[1:]])
    yy = np.concatenate([yu[::-1], yl[1:]])
    return Airfoil(xx, yy, af.name)


def load_airfoil(path: str, n_side: int = 100, smooth: bool = False) -> Airfoil:
    """Чтение файла координат (Selig или Lednicer, с заголовком или без),
    нормировка и перепанелизация."""
    rows, name = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            parts = ln.replace(",", " ").split()
            try:
                v = [float(p) for p in parts[:2]]
                if len(v) == 2:
                    rows.append(v)
                    continue
            except ValueError:
                pass
            if name is None and ln.strip():
                name = ln.strip()
    pts = np.array(rows)
    # строка Lednicer «число точек верх / низ» (значения > 1)
    if len(pts) and pts[0, 0] > 1.5 and pts[0, 1] > 1.5:
        pts = pts[1:]
    if len(pts) < 10:
        raise ValueError("В файле слишком мало точек контура")
    af = Airfoil(pts[:, 0], pts[:, 1], name or "airfoil")
    af = smooth_airfoil(af) if smooth else normalize(af)
    return repanel(af, n_side)
