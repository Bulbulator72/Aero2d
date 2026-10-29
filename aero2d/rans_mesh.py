"""Гибридная сетка для RANS вокруг одно- и многоэлементного профиля (gmsh).

У стенок — слой четырёхугольников (первый шаг по нормали ~ y⁺ ≈ 1, рост
ratio, «веер» на острых задних кромках), снаружи — треугольники, размер
которых растёт от поверхности к внешней границе (окружность радиуса R хорд).
Сгущение в следах — по расстоянию до линий следа (если заданы).

Результат — словарь массивов (см. Mesh), независимый от gmsh: узлы,
ячейки (3 или 4 вершины), грани с соседями и нормалями, метки границ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    nodes: np.ndarray          # (Nn, 2)
    cells: list                # списки вершин (против часовой стрелки)
    centroid: np.ndarray       # (Nc, 2)
    volume: np.ndarray         # (Nc,)
    face_cells: np.ndarray     # (Nf, 2): левая, правая ячейка (−1 для граничной)
    face_nodes: np.ndarray     # (Nf, 2)
    face_normal: np.ndarray    # (Nf, 2) — единичная нормаль от левой к правой
    face_area: np.ndarray      # (Nf,)
    face_center: np.ndarray    # (Nf, 2)
    face_tag: np.ndarray       # (Nf,) 0 — внутренняя, 1 — стенка, 2 — внешняя граница
    face_body: np.ndarray      # (Nf,) номер элемента для стенок, −1 иначе
    wall_dist: np.ndarray      # (Nc,) расстояние до ближайшей стенки

    @property
    def n_cells(self):
        return len(self.volume)


@dataclass
class MeshOptions:
    """Параметры сетки RANS (значение 0 — подобрать автоматически).

    yplus        — y⁺ первой ячейки у стенки (1 — для модели SA без пристеночных функций);
    ratio        — рост высоты ячеек в слое четырёхугольников;
    bl_thickness — толщина слоя четырёхугольников, доли хорды (авто: 0,01, но
                   не больше 0,3 наименьшей щели между элементами);
    n_surface    — точек на сторону основного элемента (у остальных — пропорционально
                   хорде; 0 — как в исходном контуре);
    near_size    — размер треугольников у стенок и в щелях, доли хорды
                   (авто: 0,004 — один профиль, 0,003 — механизация);
    growth       — рост размера треугольников с расстоянием от стенки (доля расстояния);
    wake_refine  — сгущение по линиям следа (для механизации);
    wake_alphas  — углы атаки, при которых строятся линии следа для сгущения (сетка
                   от угла расчёта не зависит: поляра и отдельная точка считаются
                   на одной и той же сетке);
    wake_size    — размер ячеек на линии следа; wake_length — длина сгущения за кромкой;
    far_radius   — радиус внешней границы, хорд; far_size — размер ячеек на ней, хорд."""
    yplus: float = 1.0
    ratio: float = 1.2
    bl_thickness: float = 0.0
    n_surface: int = 0
    near_size: float = 0.0
    growth: float = 0.25
    wake_refine: bool = True
    wake_size: float = 0.008
    wake_length: float = 0.6
    wake_alphas: tuple = (0.0, 6.0, 12.0)
    far_radius: float = 50.0
    far_size: float = 8.0


def build_mesh(elements, chord=1.0, re=1e6, yplus=1.0, ratio=1.2, bl_thickness=None,
               far_radius=50.0, far_size=8.0, surf_ratio=1.0, wakes=None, wake_size=0.01,
               slot_size=None, verbose=False, growth=0.25):
    """elements — список Airfoil (обход: ЗК → верх → носок → низ → ЗК), в общей
    системе координат; Re — по хорде chord. Первый шаг у стенки — по оценке
    y⁺ = yplus для турбулентной пластины (Cf ≈ 0,026 Re^(−1/7)).
    wakes — список ломаных (x, y) линий следа для сгущения."""
    import gmsh
    cf = 0.026 * re ** (-1 / 7)
    h0 = yplus * chord / (re * np.sqrt(cf / 2))
    # толщина слоя четырёхугольников: не больше трети наименьшей щели
    gaps = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            a, b = elements[i], elements[j]
            d = np.hypot(a.x[:, None] - b.x[None, :], a.y[:, None] - b.y[None, :]).min()
            gaps.append(d)
    min_gap = min(gaps) if gaps else np.inf
    if bl_thickness is None:
        # 0,01c: веер у острой задней кромки тем шире, чем толще слой, а последний ряд
        # слоя тем выше — при 0,02c у кромки перепад объёмов соседних ячеек был ~30
        bl_thickness = min(0.01 * chord, 0.3 * min_gap)
    # interruptible=False: не ставить обработчик Ctrl+C — иначе gmsh падает,
    # когда сетка строится не в главном потоке (расчёт из окна программы)
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:                                   # старые версии gmsh
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.NumThreads", 1)        # воспроизводимая сетка
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.model.add("aero2d")
        geo = gmsh.model.geo
        loops, wall_curves, fan_pts, body_curves, surf_pts, seg_len = [], [], [], [], [], []
        for e, af in enumerate(elements):
            x, y = np.asarray(af.x, float), np.asarray(af.y, float)
            if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-12:
                x, y = x[:-1], y[:-1]
            # шаг по поверхности — по соседним точкам контура
            ds = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
            lc = surf_ratio * np.maximum(np.r_[ds[-1], ds[:-1]], ds)
            pts = [geo.addPoint(xi, yi, 0, li) for xi, yi, li in zip(x, y, lc)]
            surf_pts += list(zip(pts, np.maximum(np.r_[ds[-1], ds[:-1]], ds)))
            curves = []
            for k in range(len(pts)):
                c = geo.addLine(pts[k], pts[(k + 1) % len(pts)])
                curves.append(c)
            seg_len += list(ds)
            loops.append(geo.addCurveLoop(curves))
            wall_curves += curves
            body_curves.append(curves)
            fan_pts.append(pts[0])                      # задняя кромка
        far = []
        n_far = 64
        cpt = geo.addPoint(0.5 * chord, 0, 0, far_size * chord)
        fpts = [geo.addPoint(0.5 * chord + far_radius * chord * np.cos(t), far_radius * chord * np.sin(t), 0,
                             far_size * chord) for t in np.linspace(0, 2 * np.pi, n_far, endpoint=False)]
        for k in range(n_far):
            far.append(geo.addCircleArc(fpts[k], cpt, fpts[(k + 1) % n_far]))
        floop = geo.addCurveLoop(far)
        surf = geo.addPlaneSurface([floop] + loops)
        geo.synchronize()
        smin = 0.004 * chord if slot_size is None else slot_size

        def wall_divisions():
            """Отрезки контура длиннее размера треугольников у стенки делятся: ширина
            ячеек слоя вдоль поверхности ≈ размеру треугольников над слоем."""
            for c, n in zip(wall_curves, seg_n):
                gmsh.model.mesh.setTransfiniteCurve(c, n)
        seg_n = [max(2, int(np.ceil(L / smin)) + 1) for L in seg_len]
        wall_divisions()
        # слой четырёхугольников
        fbl = gmsh.model.mesh.field.add("BoundaryLayer")
        gmsh.model.mesh.field.setNumbers(fbl, "CurvesList", wall_curves)
        gmsh.model.mesh.field.setNumbers(fbl, "FanPointsList", fan_pts)
        gmsh.model.mesh.field.setNumber(fbl, "Size", h0)
        gmsh.model.mesh.field.setNumber(fbl, "Ratio", ratio)
        gmsh.model.mesh.field.setNumber(fbl, "Thickness", bl_thickness)
        gmsh.model.mesh.field.setNumber(fbl, "Quads", 1)
        gmsh.model.mesh.field.setAsBoundaryLayer(fbl)
        # размер треугольников растёт от КРАЯ слоя четырёхугольников (а не от стенки:
        # иначе на краю слоя треугольник сразу в 2–3 раза больше последнего ряда слоя),
        # а на самом краю равен шагу слоя: высоте последнего ряда и ширине ячеек вдоль
        # поверхности (у кромок она мала — крупные треугольники над узкими ячейками
        # слоя и над веером у задней кромки давали перепад объёмов в десятки раз)
        h_last, tot, h = h0, 0.0, h0
        while tot + h < bl_thickness and h < bl_thickness:
            tot += h
            h_last, h = h, h * ratio
        s_far = far_size * chord

        def ramp(dist_field, s0):
            """Размер s0 на краю слоя, далее s0 + growth·(d − толщина слоя), до s_far."""
            f = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(f, "SizeMin", s0)
            gmsh.model.mesh.field.setNumber(f, "SizeMax", s_far)
            gmsh.model.mesh.field.setNumber(f, "DistMin", bl_thickness)
            gmsh.model.mesh.field.setNumber(f, "DistMax", bl_thickness + (s_far - s0) / max(growth, 1e-3))
            return f
        fields = []
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", wall_curves)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 20)
        # размер на краю слоя — ширина ячеек слоя вдоль поверхности, но не меньше высоты
        # его последнего ряда (подобрано по Volume change, Quality и неортогональности
        # на одиночном профиле и механизации; см. README)
        match = lambda w: max(w, 1.2 * h_last)
        s0 = float(min(smin, match(smin)))
        fields.append(ramp(fd, s0))
        # мельче — по точкам контура, сгруппированным по шагу (уровни через ×1,5)
        s_edge = np.array([match(min(d_, smin)) for _, d_ in surf_pts])
        lev = np.floor(np.log(np.maximum(s_edge, 1e-12) / s0) / np.log(1.5))
        for L in np.unique(lev[lev < 0]):
            sel = [t for (t, _), l_ in zip(surf_pts, lev) if l_ == L]
            fp = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fp, "PointsList", sel)
            fields.append(ramp(fp, s0 * 1.5 ** L))
        if wakes:
            wpts = []
            for wx, wy in wakes:
                for xx, yy in zip(wx, wy):
                    wpts.append(geo.addPoint(xx, yy, 0, wake_size * chord))
            geo.synchronize()
            # повторная синхронизация сбрасывает разбиение стенки — задать заново
            # (иначе каждый отрезок контура делился ещё на несколько)
            wall_divisions()
            fw = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fw, "PointsList", wpts)
            fwm = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(fwm, "F", f"{wake_size * chord} + 0.3*F{fw}")
            fields.append(fwm)
        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)            # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.MeshSizeMax", far_size * chord)
        gmsh.model.mesh.generate(2)
        # --- извлечение
        ntags, ncoord, _ = gmsh.model.mesh.getNodes()
        nodes = ncoord.reshape(-1, 3)[:, :2]
        idx = np.full(int(ntags.max()) + 1, -1)
        idx[ntags.astype(int)] = np.arange(len(ntags))
        cells = []
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        for et, en in zip(etypes, enodes):
            nv = {2: 3, 3: 4}[int(et)]
            cells += [list(idx[en.reshape(-1, nv)[k].astype(int)]) for k in range(len(en) // nv)]
        # граничные рёбра
        wall_edges = {}
        for b, cs in enumerate(body_curves):
            for c in cs:
                _, _, en = gmsh.model.mesh.getElements(1, c)
                for ed in en[0].reshape(-1, 2):
                    wall_edges[tuple(sorted(idx[ed.astype(int)]))] = b
        far_edges = set()
        for c in far:
            _, _, en = gmsh.model.mesh.getElements(1, c)
            for ed in en[0].reshape(-1, 2):
                far_edges.add(tuple(sorted(idx[ed.astype(int)])))
    finally:
        gmsh.finalize()
    return _connectivity(nodes, cells, wall_edges, far_edges, elements)


def _connectivity(nodes, cells, wall_edges, far_edges, elements):
    # ориентация против часовой стрелки
    cells2 = []
    cen = np.zeros((len(cells), 2))
    vol = np.zeros(len(cells))
    for k, c in enumerate(cells):
        p = nodes[c]
        a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        if a < 0:
            c = c[::-1]
            p = p[::-1]
            a = -a
        cells2.append(c)
        # центр масс многоугольника
        cr = p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]
        cen[k] = [np.sum((p[:, 0] + np.roll(p[:, 0], -1)) * cr), np.sum((p[:, 1] + np.roll(p[:, 1], -1)) * cr)]
        cen[k] /= 6 * a
        vol[k] = a
    edges = {}
    for k, c in enumerate(cells2):
        n = len(c)
        for i in range(n):
            a, b = c[i], c[(i + 1) % n]
            key = (min(a, b), max(a, b))
            if key in edges:
                edges[key][1] = k
            else:
                edges[key] = [k, -1, a, b]          # a→b — обход левой ячейки (против часовой)
    nf = len(edges)
    fc = np.zeros((nf, 2), int)
    fn = np.zeros((nf, 2), int)
    tag = np.zeros(nf, int)
    body = -np.ones(nf, int)
    for i, (key, (l, r, a, b)) in enumerate(edges.items()):
        fc[i] = (l, r)
        fn[i] = (a, b)
        if r < 0:
            if key in wall_edges:
                tag[i], body[i] = 1, wall_edges[key]
            elif key in far_edges:
                tag[i] = 2
            else:
                raise RuntimeError("граничное ребро без метки")
    pa, pb = nodes[fn[:, 0]], nodes[fn[:, 1]]
    t = pb - pa
    area = np.hypot(t[:, 0], t[:, 1])
    normal = np.column_stack([t[:, 1], -t[:, 0]]) / area[:, None]     # наружу из левой ячейки
    fcen = 0.5 * (pa + pb)
    # расстояние до стенки — до отрезков стенок
    from scipy.spatial import cKDTree
    wf = np.where(tag == 1)[0]
    sub = np.linspace(0, 1, 9)
    wp = (pa[wf, None, :] + sub[None, :, None] * t[wf, None, :]).reshape(-1, 2)
    d, _ = cKDTree(wp).query(cen)
    return Mesh(nodes, cells2, cen, vol, fc, fn, normal, area, fcen, tag, body, d)


def quality(mesh: Mesh) -> dict:
    """Простые показатели: число ячеек, четырёхугольников, мин. объём, макс.
    отношение соседних объёмов."""
    l, r = mesh.face_cells.T
    m = r >= 0
    vr = mesh.volume[l[m]] / mesh.volume[r[m]]
    return dict(cells=mesh.n_cells, quads=sum(len(c) == 4 for c in mesh.cells),
                vmin=float(mesh.volume.min()), vol_ratio_max=float(np.max(np.maximum(vr, 1 / vr))),
                wall_faces=int((mesh.face_tag == 1).sum()))


# показатели качества (как в ICEM CFD: Determinant 2x2, Quality, Min angle, Volume change)
QUALITY_METRICS = {
    "det": ("Determinant 2x2", "отношение наименьшего определителя якобиана в углах ячейки к "
            "наибольшему: 1 — параллелограмм, ≤ 0 — вывернутая; у треугольника всегда 1"),
    "quality": ("Quality", "наименьший нормированный якобиан (синус угла; у треугольника — "
                "делённый на sin 60°): 1 — прямоугольник / равносторонний треугольник"),
    "angle": ("Min angle, °", "наименьший внутренний угол ячейки"),
    "vol_ratio": ("Volume change", "наибольшее отношение объёмов ячейки и её соседей (≥ 1)"),
    "nonortho": ("Non-orthogonality, °", "наибольший угол между нормалью грани и линией центров ячеек"),
}


def cell_quality(mesh: Mesh) -> dict:
    """Показатели качества каждой ячейки (массивы длины n_cells), см. QUALITY_METRICS."""
    n = mesh.n_cells
    det = np.ones(n)
    qual = np.ones(n)
    amin = np.full(n, 180.0)
    for k in (3, 4):
        idx = np.array([i for i, c in enumerate(mesh.cells) if len(c) == k], int)
        if not len(idx):
            continue
        P = mesh.nodes[np.array([mesh.cells[i] for i in idx])]           # (m,k,2)
        e_next = np.roll(P, -1, axis=1) - P
        e_prev = np.roll(P, 1, axis=1) - P
        J = e_next[..., 0] * e_prev[..., 1] - e_next[..., 1] * e_prev[..., 0]   # (m,k)
        ln = np.linalg.norm(e_next, axis=2) * np.linalg.norm(e_prev, axis=2)
        s = J / np.maximum(ln, 1e-300)                                     # синус угла в вершине
        cosang = np.sum(e_next * e_prev, axis=2) / np.maximum(ln, 1e-300)
        ang = np.degrees(np.arctan2(s, cosang))                            # < 0 — вывернутый угол
        amin[idx] = ang.min(1)
        if k == 4:
            jmax = np.max(J, 1)
            det[idx] = np.where(jmax > 0, J.min(1) / np.where(jmax > 0, jmax, 1.0), -1.0)
            qual[idx] = s.min(1)
        else:
            qual[idx] = s.min(1) / np.sin(np.pi / 3)
            det[idx] = np.where(J.min(1) > 0, 1.0, -1.0)
    l, r = mesh.face_cells.T
    m = r >= 0
    vr = mesh.volume[l[m]] / mesh.volume[r[m]]
    vr = np.maximum(vr, 1.0 / vr)
    vol_ratio = np.ones(n)
    np.maximum.at(vol_ratio, l[m], vr)
    np.maximum.at(vol_ratio, r[m], vr)
    d = mesh.centroid[r[m]] - mesh.centroid[l[m]]
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
    no = np.degrees(np.arccos(np.clip(np.abs(np.sum(d * mesh.face_normal[m], 1)), 0.0, 1.0)))
    nonortho = np.zeros(n)
    np.maximum.at(nonortho, l[m], no)
    np.maximum.at(nonortho, r[m], no)
    return dict(det=det, quality=np.clip(qual, -1.0, 1.0), angle=amin, vol_ratio=vol_ratio, nonortho=nonortho)


def quality_report(mesh: Mesh, cq: dict | None = None) -> str:
    """Сводка как гистограмма Quality в ICEM: число ячеек по интервалам и худшие значения."""
    cq = cq or cell_quality(mesh)
    out = []
    bins = [-1.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]
    for key in ("det", "quality"):
        name = QUALITY_METRICS[key][0]
        h, _ = np.histogram(cq[key], bins=bins)
        parts = [f"{a:g}…{min(b, 1):g}: {c}" for a, b, c in zip(bins[:-1], bins[1:], h) if c]
        out.append(f"{name}: мин. {cq[key].min():.3f}; " + ", ".join(parts))
    out.append(f"Min angle: мин. {cq['angle'].min():.1f}°, ячеек < 10°: {(cq['angle'] < 10).sum()}")
    out.append(f"Volume change: макс. {cq['vol_ratio'].max():.0f}, ячеек > 10: {(cq['vol_ratio'] > 10).sum()}, "
               f"> 100: {(cq['vol_ratio'] > 100).sum()}")
    out.append(f"Non-orthogonality: макс. {cq['nonortho'].max():.0f}°, ячеек > 70°: {(cq['nonortho'] > 70).sum()}")
    return "\n".join(out)


class NodeInterp:
    """Интерполяция величин из центров ячеек в произвольные точки по самой
    расчётной сетке: значения переносятся в узлы (среднее по соседним ячейкам с
    весом объёма; на стенке — заданные значения, например скорость 0), затем —
    линейно внутри треугольников сетки (четырёхугольники делятся на два).
    В отличие от триангуляции Делоне по центрам ячеек треугольники не
    перекидываются через тело (тонкая задняя кромка, ниша закрылка), а у стенки
    выполняется условие прилипания — без ложных полос вдоль контура."""

    def __init__(self, mesh: Mesh):
        import scipy.sparse as sp
        from matplotlib.tri import Triangulation
        tris, tcell = [], []
        for i, c in enumerate(mesh.cells):
            if len(c) == 3:
                tris.append(c); tcell.append(i)
            else:
                tris += [[c[0], c[1], c[2]], [c[0], c[2], c[3]]]; tcell += [i, i]
        self.tris = np.asarray(tris, int)
        self.tcell = np.asarray(tcell, int)
        self.tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], self.tris)
        self.finder = self.tri.get_trifinder()
        rows = np.concatenate([np.asarray(c, int) for c in mesh.cells])
        cols = np.repeat(np.arange(mesh.n_cells), [len(c) for c in mesh.cells])
        w = mesh.volume[cols]
        W = sp.csr_matrix((w, (rows, cols)), shape=(len(mesh.nodes), mesh.n_cells))
        s = np.asarray(W.sum(1)).ravel()
        self.W = sp.diags(1.0 / np.maximum(s, 1e-300)) @ W
        self.wall = np.zeros(len(mesh.nodes), bool)
        self.wall[mesh.face_nodes[mesh.face_tag == 1].ravel()] = True
        self.nodes = mesh.nodes
        self._pts = None

    def locate(self, x, y):
        """Треугольник и барицентрические координаты точек (запоминаются для тех же точек)."""
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        key = (x.size, float(x.sum()), float(y.sum()))
        if self._pts is not None and self._pts[0] == key:
            return self._pts[1:]
        ti = self.finder(x, y)
        ok = ti >= 0
        bc = np.zeros((x.size, 3))
        P = self.nodes[self.tris[np.where(ok, ti, 0)]]                    # (n,3,2)
        a, b, c = P[:, 0], P[:, 1], P[:, 2]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        l1 = ((x - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (y - a[:, 1])) / det
        l2 = ((b[:, 0] - a[:, 0]) * (y - a[:, 1]) - (x - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        bc[:, 0], bc[:, 1], bc[:, 2] = 1 - l1 - l2, l1, l2
        self._pts = (key, ti, ok, bc)
        return ti, ok, bc

    def __call__(self, cell_vals, x, y, wall=None):
        """cell_vals (n_cells, k) → (n_точек, k), вне сетки (внутри тел) — NaN.
        wall — {номер столбца: значение на стенке} (например {0: 0.0, 1: 0.0} — прилипание)."""
        V = np.asarray(cell_vals, float)
        V = V[:, None] if V.ndim == 1 else V
        Nv = self.W @ V
        for j, val in (wall or {}).items():
            Nv[self.wall, j] = val
        ti, ok, bc = self.locate(x, y)
        tn = self.tris[np.where(ok, ti, 0)]
        out = np.einsum("pj,pjk->pk", bc, Nv[tn])
        out[~ok] = np.nan
        return out

    def node_vorticity(self, un, vn):
        """Завихренность в узлах по узловым скоростям: градиент линейного поля в
        каждом треугольнике, затем среднее по треугольникам узла с весом площади.
        Дифференцирование уже интерполированного поля давало постоянное значение
        в каждом треугольнике — «мозаику» и шум вне слоя."""
        P = self.nodes[self.tris]
        x, y = P[..., 0], P[..., 1]
        a2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        bx = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / a2[:, None]
        by = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / a2[:, None]
        om = np.sum(bx * vn[self.tris], 1) - np.sum(by * un[self.tris], 1)
        w = np.abs(a2)
        num = np.bincount(self.tris.ravel(), np.repeat(om * w, 3), len(self.nodes))
        den = np.bincount(self.tris.ravel(), np.repeat(w, 3), len(self.nodes))
        return num / np.maximum(den, 1e-300)

    def interp_nodes(self, nv, x, y):
        """Линейная интерполяция узловых значений nv в точки (вне сетки — NaN)."""
        ti, ok, bc = self.locate(x, y)
        out = np.einsum("pj,pj->p", bc, nv[self.tris[np.where(ok, ti, 0)]])
        out[~ok] = np.nan
        return out

    def cell_at(self, x, y):
        """Номер ячейки, в которой лежит точка (−1 — вне сетки): для кусочно-постоянных величин."""
        ti, ok, _ = self.locate(x, y)
        return np.where(ok, self.tcell[np.where(ok, ti, 0)], -1)


def wall_loops(mesh: Mesh):
    """Контуры тел по граням стенки самой сетки (замкнутые ломаные, по телам) — для
    рисования: совпадают с границей расчётной области точно."""
    out = []
    wf = np.where(mesh.face_tag == 1)[0]
    for b in np.unique(mesh.face_body[wf]):
        edges = mesh.face_nodes[wf[mesh.face_body[wf] == b]]
        nxt = {}
        for a, c in edges:
            nxt.setdefault(a, []).append(c)
            nxt.setdefault(c, []).append(a)
        start = int(edges[0, 0])
        loop, prev, cur = [start], None, start
        for _ in range(len(edges)):
            cand = [n for n in nxt[cur] if n != prev]
            if not cand:
                break
            prev, cur = cur, cand[0]
            if cur == start:
                break
            loop.append(cur)
        P = mesh.nodes[loop + [start]]
        out.append((P[:, 0], P[:, 1]))
    return out
