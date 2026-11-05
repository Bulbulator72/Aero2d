"""Сетка RANS с С-блоком у каждого элемента (как С-сетка в ICEM CFD).

Острая кромка со свободным следом: блок продолжается вдоль линии следа на ~1 хорду
(как С-сетка Mavriplis / NASA): к концу разреза ячейки квадратные. Кромка в щели (разрез
упирается в соседний элемент) и тупая кромка — короткий разрез и полоса за торцом
(_cblock_slot). Размер треугольников на границе блока — по площади прилегающих ячеек.

Около стенки — структурированный блок четырёхугольников: путь «низ разреза за
задней кромкой → контур профиля → верх разреза», слои выдавливаются по
сглаженным нормалям (каждый слой — от предыдущего). У острой задней кромки
слой не сворачивается веером в точку, а продолжается по разрезу (линии следа):
высота первой ячейки на разрезе плавно растёт от y⁺ = 1 у кромки до
равномерного деления толщины слоя к концу разреза, шаг вдоль разреза
продолжает шаг контура у кромки. Снаружи блоков — треугольники gmsh, их
размер на границе блока равен шагу крайнего ряда блока. Узлы на границе общие.

Если блок вывернулся (вогнутые углы, тонкие щели) или задел соседний элемент —
исключение BlockError, и build_mesh строит прежнюю сетку gmsh со слоем
BoundaryLayer (веер у кромки)."""
from __future__ import annotations

import numpy as np


class BlockError(RuntimeError):
    pass


def _subdivide(x, y, smin, grade=1.25, closed=True):
    """Замкнутый контур (без повтора первой точки): отрезки длиннее smin делятся, а шаги
    соседних отрезков после деления различаются не больше чем в grade раз (длинный
    отрезок рядом с коротким дробится) — иначе скачут ширины столбцов слоя."""
    xs, ys = [], []
    n = len(x)
    Lk = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
    if not closed:
        Lk = Lk[:-1]
    T = np.minimum(Lk, np.broadcast_to(np.asarray(smin, float)[..., None] if np.ndim(smin) == 0
                                       else np.asarray(smin, float)[:len(Lk)], Lk.shape))
    for _ in range(200):
        nb_ = np.minimum(np.roll(T, 1), np.roll(T, -1))
        if not closed:
            nb_[0], nb_[-1] = T[1] if len(T) > 1 else T[0], T[-2] if len(T) > 1 else T[-1]
        T2 = np.minimum(T, grade * nb_)
        if np.allclose(T2, T):
            break
        T = T2
    for k in range(n if closed else n - 1):
        a, b = np.array([x[k], y[k]]), np.array([x[(k + 1) % n], y[(k + 1) % n]])
        m = max(1, int(np.ceil(Lk[k] / T[k] - 1e-9)))
        for j in range(m):
            p = a + (b - a) * j / m
            xs.append(p[0])
            ys.append(p[1])
    if not closed:
        xs.append(x[-1])
        ys.append(y[-1])
    return np.array(xs), np.array(ys)


def _coarsen_te(x, y, h):
    """Шаг контура у задней кромки (первая точка) не меньше h: точки, идущие от кромки чаще
    h, заменяются равномерными с шагом ≈ h — до места, где исходный шаг сам больше h.
    Иначе столбцы слоя у кромки узкие и высокие, к их верху липнут крошечные треугольники."""
    P = np.column_stack([x, y])
    n = len(P)
    seg = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)        # seg[k]: P[k] → P[k+1]
    out_up, out_lo = [], []
    for side in (1, -1):
        # вдоль стороны от кромки: индексы 0,1,2,… (верх) или 0,n-1,n-2,… (низ)
        ids = [0]
        k = 0
        acc = 0.0
        while True:
            k2 = (k + side) % n
            ln = seg[k] if side == 1 else seg[k2]
            acc += ln
            ids.append(k2)
            k = k2
            if ln >= h or len(ids) > n // 3:     # первый отрезок не короче h — тоже в участок
                break
        if len(ids) < 3:
            (out_up if side == 1 else out_lo).append(None)
            continue
        m = max(1, int(round(acc / h)))
        pts = P[ids]
        sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        tgt = np.linspace(0.0, sl[-1], m + 1)[1:-1]
        new = np.column_stack([np.interp(tgt, sl, pts[:, 0]), np.interp(tgt, sl, pts[:, 1])])
        (out_up if side == 1 else out_lo).append((ids, new))
    keep = np.ones(n, bool)
    up, lo = out_up[0], out_lo[0]
    for r in (up, lo):
        if r is not None:
            keep[r[0][1:-1]] = False
    parts = [P[:1]]
    if up is not None:
        parts.append(up[1])
    mid = P[1:][keep[1:]]
    parts.append(mid)
    if lo is not None:
        # низ: точки шли от кромки назад по обходу — в контур вставляются в обратном порядке
        parts.insert(len(parts), lo[1][::-1])
    Q = np.vstack(parts)
    return Q[:, 0], Q[:, 1]


def _find_base(x, y):
    """Тупая задняя кромка: торец — короткий участок контура между двумя углами у
    кромки. Контур (против часовой) → (открытая ломаная от верхнего угла торца через
    носок к нижнему, торец не входит) или None, если кромка острая."""
    P = np.column_stack([x, y])
    n = len(P)
    a_ = np.roll(P, 1, axis=0) - P
    b_ = np.roll(P, -1, axis=0) - P
    cosang = np.sum(a_ * b_, 1) / np.maximum(np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1), 1e-300)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))          # внутренний угол ломаной в узле
    ch = float(np.ptp(x))
    te = np.where((x > x.max() - 0.03 * ch) & (ang < 135.0))[0]  # углы у кромки (поворот > 45°)
    seg = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)
    best = None
    for i in te:
        for j in te:
            if i == j:
                continue
            # участок от i до j по обходу (против часовой: низ → торец → верх)
            k, ln, ok = i, 0.0, True
            while k != j:
                ln += seg[k]
                k = (k + 1) % n
                if ln > 0.05 * ch:
                    ok = False
                    break
            chord_ij = float(np.hypot(*(P[i] - P[j])))
            if ok and chord_ij > 1e-5 * ch and ln < 1.3 * chord_ij and (best is None or ln < best[2]):
                best = (i, j, ln)
    if best is None:
        return None
    i, j, _ = best
    idx = [(j + k) % n for k in range((i - j) % n + 1)]          # от верхнего угла j до нижнего i
    return P[idx]


def _coarsen_start(P, h):
    """Открытая ломаная: у начала шаг не меньше h (как _coarsen_te для одной стороны)."""
    seg = np.hypot(*np.diff(P, axis=0).T)
    k, acc = 0, 0.0
    while k < len(seg) and k < len(P) // 3:
        acc += seg[k]
        k += 1
        if seg[k - 1] >= h:
            break
    if k < 2:
        return P
    m = max(1, int(round(acc / h)))
    sl = np.r_[0.0, np.cumsum(seg[:k])]
    tgt = np.linspace(0.0, acc, m + 1)[1:-1]
    new = np.column_stack([np.interp(tgt, sl, P[:k + 1, 0]), np.interp(tgt, sl, P[:k + 1, 1])])
    return np.vstack([P[:1], new, P[k:]])


def _seg_dist(p, segs):
    """Расстояние от точек p (n,2) до отрезков segs (m,2,2)."""
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = np.clip(np.einsum("nmk,mk->nm", p[:, None, :] - a[None], ab) / np.maximum((ab ** 2).sum(1), 1e-300),
                0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.min(np.hypot(*(p[:, None, :] - q).transpose(2, 0, 1)), axis=1)


def _ratio_for(h1, t, n):
    """q: h1·(qⁿ − 1)/(q − 1) = t (сумма n слоёв с первым h1 равна t)."""
    if abs(h1 * n - t) < 1e-12 * t:
        return 1.0
    lo, hi = (1e-6, 1.0) if h1 * n > t else (1.0, 3.0)
    f = lambda q: h1 * (n if abs(q - 1) < 1e-12 else (q ** n - 1) / (q - 1)) - t
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def c_block(x, y, h0, ratio, t, smin, wake_len, obstacles=(), wake_size=0.008, flow_dir=(1.0, 0.0),
            smooth=None):
    """См. _c_block; smooth=None — сглаживание нормалей подбирается: самое слабое, при
    котором блок не выворачивается (у кромки над разрезом — вогнутый угол)."""
    xx, yy = np.asarray(x, float), np.asarray(y, float)
    if np.hypot(xx[0] - xx[-1], yy[0] - yy[-1]) < 1e-12:
        xx, yy = xx[:-1], yy[:-1]
    if 0.5 * np.sum(xx * np.roll(yy, -1) - np.roll(xx, -1) * yy) < 0:
        xx, yy = np.r_[xx[:1], xx[1:][::-1]], np.r_[yy[:1], yy[1:][::-1]]
    Pb = _find_base(xx, yy)
    if wake_len <= 0 or (Pb is not None and np.hypot(*(Pb[0] - Pb[-1])) > 2e-4 * float(np.ptp(xx))):
        # wake_len ≤ 0 — без блока следа (сгущение по следу выключено): короткий разрез
        # тупая кромка: разрезы от обоих углов торца и полоса за торцом (как блок за торцом в
        # PyAero) — O-обход углов торца с 5–17 точками на торце выворачивался
        from . import _cblock_slot
        b = _cblock_slot.c_block(x, y, h0, ratio, t, smin, 1.5 * t, obstacles)
        b["out_size"] = _matched_sizes(b["nodes"], b["quads"], b["outline"])
        return b
    err = None
    for sm in ((smooth,) if smooth is not None else (10, 30, 60, 120, 250)):
        try:
            b = _c_block(x, y, h0, ratio, t, smin, wake_len, obstacles, wake_size, flow_dir, sm)
            if b.get("short"):
                # кромка в щели: длинный блок следа не помещается — короткий разрез (как у
                # NASA: след переднего элемента над закрылком разрешается сеткой вокруг закрылка)
                from . import _cblock_slot
                b = _cblock_slot.c_block(x, y, h0, ratio, t, smin, 1.5 * t, obstacles)
                b["out_size"] = _matched_sizes(b["nodes"], b["quads"], b["outline"])
                return b
            b["smooth"] = sm
            return b
        except BlockError as e:
            err = e
            if "вывернулся" not in str(e):
                raise
    raise err


def _layers_z(h1, T, n):
    """Высоты узлов столбцов: n слоёв, первый h1, сумма T (геометрическая прогрессия)."""
    q = np.array([_ratio_for(h, Ti, n) for h, Ti in zip(h1, T)])
    j = np.arange(n + 1)
    qm = np.where(np.abs(q - 1) < 1e-12, 2.0, q)
    return np.where(np.abs(q - 1)[:, None] < 1e-12, h1[:, None] * j[None],
                    h1[:, None] * (qm[:, None] ** j[None] - 1) / (qm - 1)[:, None])


def _thick_for_top(h1, top, n):
    """Толщина n слоёв с первым h1 и последним top."""
    q = (np.maximum(top, h1) / h1) ** (1.0 / max(n - 1, 1))
    return np.where(np.abs(q - 1) < 1e-9, n * h1, h1 * (q ** n - 1) / np.where(np.abs(q - 1) < 1e-9, 1, q - 1))


def _march(path, z, smooth):
    """Слои по сглаженным нормалям (справа по обходу), каждый — от предыдущего."""
    layers = [path.copy()]
    for k in range(1, z.shape[1]):
        P = layers[-1]
        tg = np.empty_like(P)
        tg[1:-1] = P[2:] - P[:-2]
        tg[0], tg[-1] = P[1] - P[0], P[-1] - P[-2]
        nr = np.column_stack([tg[:, 1], -tg[:, 0]])
        nr /= np.maximum(np.linalg.norm(nr, axis=1), 1e-300)[:, None]
        for _ in range(smooth):
            sm = nr.copy()
            sm[1:-1] = nr[:-2] + 2 * nr[1:-1] + nr[2:]
            nr = sm / np.maximum(np.linalg.norm(sm, axis=1), 1e-300)[:, None]
        layers.append(P + nr * (z[:, k] - z[:, k - 1])[:, None])
    return layers


def _check(nodes, quads, what):
    P = nodes[quads]
    e1 = np.roll(P, -1, axis=1) - P
    e0 = P - np.roll(P, 1, axis=1)
    cr = e0[..., 0] * e1[..., 1] - e0[..., 1] * e1[..., 0]
    sgn = np.sign(np.median(cr))
    bad = np.any(cr * sgn <= 0, axis=1)
    if bad.any():
        c = P[bad].mean(1)
        raise BlockError(f"блок вывернулся ({what}): {bad.sum()} ячеек, например у ({c[0, 0]:.4f}; {c[0, 1]:.4f})")


def _c_block(x, y, h0, ratio, t, smin, wake_len, obstacles=(), wake_size=0.008, flow_dir=(1.0, 0.0),
             smooth=10, wake_growth=1.12, te_w=None):
    """С-блок четырёхугольников вокруг одного элемента и вдоль его следа (как С-сетка с
    разрезом в ICEM / гиперболическая С-сетка Mavriplis): путь «верх разреза → контур →
    низ разреза», слои — по сглаженным нормалям. Разрез — линия следа от острой задней
    кромки: сначала по биссектрисе кромки, затем по направлению потока flow_dir; длина до
    wake_len, но разрез укорачивается, если блок подходит к другим элементам. Вдоль следа
    шаг растёт не быстрее wake_growth до wake_size, первая ячейка — от высоты у стенки до
    шага следа: к концу разреза ячейки квадратные, и треугольники примыкают без скачка.

    x, y — замкнутый контур, первая точка — задняя кромка (без повтора в конце).
    Возвращает dict: nodes, quads, wall (рёбра стенки), outline (контур блока, индексы
    узлов), out_size (размер треугольников у точек outline)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-12:
        x, y = x[:-1], y[:-1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:                                   # против часовой: снаружи — справа по обходу
        x, y = np.r_[x[:1], x[1:][::-1]], np.r_[y[:1], y[1:][::-1]]
    c_el = float(np.ptp(x))
    n = int(np.ceil(np.log(1 + t * (ratio - 1) / h0) / np.log(ratio)))
    t_min = 1.5 * h0 * n
    t = max(t, t_min)
    obst = np.asarray(obstacles) if len(obstacles) else None
    ht = float(h0 * _ratio_for(h0, t, n) ** (n - 1))           # верхний ряд у стенки
    blunt = False                                  # тупые кромки — в c_block (_cblock_slot)
    # шаг у кромки — не крупнее половины шага блока: при 1,2·ht шаг выходил то 0,0042 (и
    # делился пополам по cap до 0,0021), то 0,0037 (не делился) — от округления числа слоёв;
    # у кромки оставался столбец вдвое шире, и при y⁺ 0,5 расчёт не сходился (NACA 0012,
    # α 10°). Мельче 1,2·ht не берём: узкие высокие столбцы у кромки перекашиваются
    cap_v = float(np.clip(4.0 * ht, 0.4 * smin, smin))
    x, y = _coarsen_te(x, y, te_w or min(1.2 * ht, 0.5 * cap_v))
    loop = np.column_stack([x, y])
    fixed = np.zeros(len(loop), bool)
    t1, t2 = loop[1] - loop[0], loop[-1] - loop[0]
    u = -(t1 / np.linalg.norm(t1) + t2 / np.linalg.norm(t2))
    w_te = 0.5 * (np.linalg.norm(t1) + np.linalg.norm(t2))
    if np.linalg.norm(u) < 1e-9:
        raise BlockError("не определить направление разреза за кромкой")
    u /= np.linalg.norm(u)
    seg = np.hypot(*np.diff(np.vstack([loop, loop[:1]]), axis=0).T)
    cap = np.full(len(loop), float(np.clip(4.0 * ht, 0.4 * smin, smin)))
    if obst is not None:
        # в щели на треугольники остаётся полоса ≈ 0,4 зазора — шаг блока не крупнее её
        mid = 0.5 * (loop + np.roll(loop, -1, axis=0))
        cap = np.minimum(cap, np.maximum(0.4 * _seg_dist(mid, obst), 0.1 * smin))
    if te_w:
        cap[0], cap[-1] = min(cap[0], te_w), min(cap[-1], te_w)
    cap = np.where(fixed, np.maximum(seg, 1e-12), cap)
    sx, sy = _subdivide(loop[:, 0], loop[:, 1], cap)
    S = np.column_stack([sx, sy])
    M = len(S)
    # толщина у стенки: тоньше, где шаг вдоль поверхности мелкий (верхний ряд ≤ 1,5 шага)
    wS = np.hypot(*np.diff(np.vstack([S, S[:1]]), axis=0).T)
    wS = np.minimum(wS, np.roll(wS, 1))
    Tg = np.geomspace(t_min, t, 60)
    Hg = np.array([h0 * _ratio_for(h0, Ti, n) ** (n - 1) for Ti in Tg])
    TS = np.interp(1.5 * wS, Hg, Tg, left=t_min, right=t)
    for _ in range(M):
        T2 = np.minimum(TS, 1.2 * np.minimum(np.roll(TS, 1), np.roll(TS, -1)))
        if np.allclose(T2, TS):
            break
        TS = T2
    # и по длине дуги: толщина меняется не круче 0,2 на единицу длины — при мелком шаге на носке
    # (≈ 10⁻⁴ c) правило «×1,2 между соседями» давало узкий клин (с треугольниками VC ≈ 45 внутри,
    # расчёт не сходился); теперь тонкое место — плавная ямка
    seg_S = np.hypot(*np.diff(np.vstack([S, S[:1]]), axis=0).T)
    for _ in range(3):
        for i in range(1, M):
            TS[i] = min(TS[i], TS[i - 1] + 0.2 * seg_S[i - 1])
        TS[0] = min(TS[0], TS[-1] + 0.2 * seg_S[-1])
        for i in range(M - 2, -1, -1):
            TS[i] = min(TS[i], TS[i + 1] + 0.2 * seg_S[i])
        TS[-1] = min(TS[-1], TS[0] + 0.2 * seg_S[-1])
    T0 = float(TS[0])
    h_top0 = float(h0 * _ratio_for(h0, T0, n) ** (n - 1))
    # --- линия следа: биссектриса кромки, затем — направление потока
    f = np.asarray(flow_dir, float)
    f /= np.linalg.norm(f)
    turn = 0.25 * c_el                                   # длина плавного поворота линии следа

    def centerline(L, step):
        k = max(2, int(np.ceil(L / step)))
        pts, p, d = [S[0]], S[0].copy(), 0.0
        for _ in range(k):
            b = min(d / turn, 1.0)
            b = b * b * (3 - 2 * b)
            dv = (1 - b) * u + b * f
            p = p + dv / np.linalg.norm(dv) * (L / k)
            d += L / k
            pts.append(p)
        P_ = np.array(pts)
        sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P_, axis=0).T))]
        return P_, sl

    # длина разреза: до места, где блок (толщиной T0) подошёл бы к соседнему телу
    P_c, sl = centerline(float(wake_len), min(w_te, 0.002 * c_el))
    clr = _seg_dist(P_c, obst) if obst is not None else np.full(len(P_c), np.inf)
    # до соседнего тела: свой блок + его блок (≤ T0) + полоса треугольников ⇒ ≥ 3·T0
    bad = np.where(clr < 3.0 * T0)[0]
    L_try = float(wake_len) if not len(bad) else 0.8 * float(sl[bad[0]])
    clr_l = lambda dd: np.interp(dd, sl, clr)
    # к концу разреза — квадратные ячейки шага s_e: не крупнее wake_size, ~0,1 длины и
    # места до соседа; блок не тоньше, чем у стенки (T0)
    # и блок следа (≈ n·s_e) не толще 0,3 радиуса поворота линии следа — иначе на вогнутой
    # стороне изгиба слои сжимаются
    s_e = float(max(min(wake_size, max(w_te, 0.1 * L_try), float(np.min(clr_l(np.linspace(0, L_try, 50)))) / (3.0 * n),
                        0.3 * turn / n), T0 / n))
    need = (max(w_te, s_e) - min(w_te, s_e)) / (1 - 1 / wake_growth) + 4 * s_e
    if L_try < need and te_w is None and w_te > s_e:
        # короткий разрез (щель): шаг у кромки мельче, чтобы вдоль разреза он рос плавно
        w_fit = s_e + max(L_try - 4 * s_e, 0.0) * (1 - 1 / wake_growth)
        return dict(short=max(w_fit, s_e))
    if L_try < 3 * max(min(w_te, s_e), h0):
        raise BlockError("нет места для разреза за кромкой")
    d, s_ = [], w_te
    while not d or d[-1] < L_try:
        d.append((d[-1] if d else 0.0) + s_)
        s_ = min(s_ * wake_growth, s_e) if s_ < s_e else max(s_ / wake_growth, s_e)
    d = np.array(d) * (L_try / d[-1])
    sk = np.diff(np.r_[0.0, d])
    xi = d / L_try
    # первая ячейка растёт от h0 к s_e не быстрее 1,8 на столбец (на коротком разрезе в щели
    # прежний рост «по доле длины» давал скачки высоты соседних столбцов в разы)
    h1w = np.minimum(h0 * (s_e / h0) ** xi, h0 * 1.8 ** (np.arange(len(d)) + 1))
    top = np.maximum(sk, h_top0 * (1 - xi) + s_e * xi)
    Tw = np.maximum(_thick_for_top(h1w, top, n), T0 * (1 - xi))
    Tw = np.minimum(Tw, np.maximum(clr_l(d) / 3.0, T0))
    W = np.column_stack([np.interp(d, sl, P_c[:, 0]), np.interp(d, sl, P_c[:, 1])])
    K = len(W)
    # --- путь: верх разреза (от конца к кромке), контур, кромка ещё раз, низ разреза
    path = np.vstack([W[::-1], S, S[:1], W])
    base = np.r_[np.arange(K)[::-1] + M, np.arange(M), 0, np.arange(K) + M]
    Np = len(path)
    h1 = np.r_[h1w[::-1], np.full(M + 1, h0), h1w]
    Tp = np.r_[Tw[::-1], TS, TS[:1], Tw]
    # верхний ряд каждого столбца следа не выше его ширины (иначе у контура блока узкие высокие
    # ячейки, и треугольники на их коротких рёбрах в разы мельче), толщины соседних столбцов
    # различаются не больше чем в 1,2 раза — без ступенек на стыке стенки и следа
    wp = np.hypot(*np.diff(path, axis=0).T)
    wcol = np.minimum(np.r_[wp[0], wp], np.r_[wp, wp[-1]])
    Tcap = np.maximum(_thick_for_top(h1, 1.0 * wcol, n), 1.05 * n * h1)
    in_wake = np.r_[np.ones(K, bool), np.zeros(M + 1, bool), np.ones(K, bool)]
    Tp = np.where(in_wake, np.minimum(Tp, Tcap), Tp)          # у стенки толщина — по TS
    for _ in range(Np):
        T2 = np.minimum(Tp, 1.2 * np.minimum(np.r_[Tp[0], Tp[:-1]], np.r_[Tp[1:], Tp[-1]]))
        if np.allclose(T2, Tp):
            break
        Tp = T2
    z = _layers_z(h1, np.maximum(Tp, 1.05 * n * h1), n)
    layers = _march(path, z, smooth)
    nb = M + K
    nodes = np.vstack([S, W] + layers[1:])
    idx = np.empty((n + 1, Np), int)
    idx[0] = base
    for k in range(1, n + 1):
        idx[k] = nb + (k - 1) * Np + np.arange(Np)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
    _check(nodes, quads, "у стенки и в следе")
    wall = [(base[K + i], base[K + i + 1]) for i in range(M)]
    outline = list(idx[n]) + list(idx[n - 1:0:-1, -1]) + [idx[0, -1]] + list(idx[1:n, 0])
    out_size = _matched_sizes(nodes, quads, outline, mean=True)
    return dict(nodes=nodes, quads=quads, wall=wall, outline=outline, out_size=out_size,
                layers=n, wake_len=L_try, blunt=blunt)


def _matched_sizes(nodes, quads, outline, mean=False):
    """Размер треугольников у каждой точки контура блока — по площади прилегающих к контуру
    ячеек блока: треугольник площадью ≈ 0,433·s² той же площади (Volume change ≈ 1 на стыке)."""
    P = nodes[quads]
    area = 0.5 * np.abs(np.sum(P[:, :, 0] * np.roll(P[:, :, 1], -1, 1) - np.roll(P[:, :, 0], -1, 1) * P[:, :, 1], 1))
    edge_cell = {}
    for c, q in enumerate(quads):
        for a, b in zip(q, np.roll(q, -1)):
            edge_cell[(min(a, b), max(a, b))] = c
    no = len(outline)
    ea = np.zeros(no)
    for k in range(no):
        a, b = outline[k], outline[(k + 1) % no]
        c = edge_cell.get((min(a, b), max(a, b)))
        ea[k] = area[c] if c is not None else np.nan
    ea = np.where(np.isfinite(ea), ea, np.nanmedian(ea))
    s_ = np.sqrt(ea / 0.433)
    s_ = 0.5 * (s_ + np.roll(s_, 1))
    Pq = nodes[outline]
    el = np.hypot(*(np.roll(Pq, -1, axis=0) - Pq).T)
    # среднее геометрическое размера «по площади» и длины ребра контура: у плоских ячеек
    # верхнего ряда размер по площади заметно меньше ребра — треугольник на таком ребре
    # вытянут (Quality ≈ 0,43); по среднему — Quality ≥ 0,59 при Volume change ≤ 5
    # (mean=True — для длинного блока следа; у коротких блоков в щелях — по площади: там
    # важнее Volume change на стыке с мелкими ячейками щели)
    if mean:
        s_ = np.sqrt(s_ * 0.5 * (el + np.roll(el, 1)))
    return np.minimum(s_, 1.3 * np.maximum(el, np.roll(el, 1)))


def _crosses(A, B):
    """Пересекаются ли замкнутые ломаные A и B (или одна внутри другой)."""
    from matplotlib.path import Path
    if Path(B).contains_points(A[:1])[0] or Path(A).contains_points(B[:1])[0]:
        return True
    a0, a1 = A, np.roll(A, -1, axis=0)
    b0, b1 = B, np.roll(B, -1, axis=0)
    lo = np.minimum(a0, a1); hi = np.maximum(a0, a1)
    blo = np.minimum(b0, b1); bhi = np.maximum(b0, b1)
    ii, jj = np.nonzero((lo[:, None, 0] <= bhi[None, :, 0]) & (hi[:, None, 0] >= blo[None, :, 0])
                        & (lo[:, None, 1] <= bhi[None, :, 1]) & (hi[:, None, 1] >= blo[None, :, 1]))
    if not len(ii):
        return False
    p, r = a0[ii], a1[ii] - a0[ii]
    q, s_ = b0[jj], b1[jj] - b0[jj]
    cr = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    den = cr(r, s_)
    ok = np.abs(den) > 1e-300
    tt = np.where(ok, cr(q - p, s_) / np.where(ok, den, 1), -1)
    uu = np.where(ok, cr(q - p, r) / np.where(ok, den, 1), -1)
    return bool(np.any(ok & (tt > 1e-9) & (tt < 1 - 1e-9) & (uu > 1e-9) & (uu < 1 - 1e-9)))


def build_cblock_mesh(elements, chord, h0, ratio, t, far_radius, far_size, wakes, wake_size, smin, growth,
                      wake_len=1.0, flow_alpha=6.0, verbose=False):
    # wake_len — длина блока следа в хордах (0 — без блока следа, только короткий разрез)
    """Сетка: С-блоки у элементов + треугольники gmsh снаружи (см. описание модуля)."""
    import gmsh
    from .rans_mesh import _connectivity
    polys = []
    for e in elements:
        x, y = np.asarray(e.x, float), np.asarray(e.y, float)
        polys.append(np.stack([np.column_stack([x, y]), np.column_stack([np.roll(x, -1), np.roll(y, -1)])], 1))
    fd = (np.cos(np.radians(flow_alpha)), np.sin(np.radians(flow_alpha)))
    scale = [1.0] * len(elements)
    for attempt in range(6):
        blocks = []
        for i, e in enumerate(elements):
            obst = np.concatenate([p for k, p in enumerate(polys) if k != i]) if len(elements) > 1 else []
            blocks.append(c_block(e.x, e.y, h0, ratio, t, smin, scale[i] * wake_len * chord, obst,
                                  wake_size * chord, fd))
        # контуры блоков не должны пересекаться: блок следа, задевший соседа, — короче
        outl = [b["nodes"][b["outline"]] for b in blocks]
        clash = set()
        for i in range(len(outl)):
            for k in range(i + 1, len(outl)):
                segs = np.stack([outl[k], np.roll(outl[k], -1, axis=0)], 1)
                if _seg_dist(outl[i], segs).min() < 0.05 * t + 1e-6 * chord or _crosses(outl[i], outl[k]):
                    clash |= {i, k}
        if not clash:
            break
        for i in clash:
            scale[i] *= 0.6
    else:
        raise BlockError("блоки соседних элементов касаются друг друга")
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.model.add("aero2d_c")
        geo = gmsh.model.geo
        loops, out_curves, out_pts = [], [], []
        for b in blocks:
            P = b["nodes"][b["outline"]]
            pts = [geo.addPoint(px, py, 0, s) for (px, py), s in zip(P, b["out_size"])]
            cs = [geo.addLine(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
            loops.append(geo.addCurveLoop(cs))
            out_curves += cs
            out_pts.append(list(zip(pts, b["out_size"])))
        R = far_radius * chord
        n_far = 64
        cpt = geo.addPoint(0.5 * chord, 0, 0, far_size * chord)
        fpts = [geo.addPoint(0.5 * chord + R * np.cos(a), R * np.sin(a), 0, far_size * chord)
                for a in np.linspace(0, 2 * np.pi, n_far, endpoint=False)]
        far = [geo.addCircleArc(fpts[k], cpt, fpts[(k + 1) % n_far]) for k in range(n_far)]
        geo.addPlaneSurface([geo.addCurveLoop(far)] + loops)
        geo.synchronize()
        for c in out_curves:
            gmsh.model.mesh.setTransfiniteCurve(c, 2)
        # размер: на границе блока — шаг его крайнего ряда, дальше растёт на growth·расстояние
        s_far = far_size * chord
        fields = []

        def ramp(dist_field, s0):
            f = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(f, "SizeMin", s0)
            gmsh.model.mesh.field.setNumber(f, "SizeMax", s_far)
            gmsh.model.mesh.field.setNumber(f, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(f, "DistMax", (s_far - s0) / max(growth, 1e-3))
            return f
        allp = [ps for b in out_pts for ps in b]
        sz = np.array([s for _, s in allp])
        lev = np.floor(np.log(np.maximum(sz, 1e-12) / smin) / np.log(1.4))
        # уровни и крупнее smin: торец длинного блока следа — ячейки шага следа (иначе там
        # треугольники мельче четырёхугольников в разы)
        for L_ in np.unique(lev):
            sel = [p for (p, _), l_ in zip(allp, lev) if l_ == L_]
            fp = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fp, "PointsList", sel)
            fields.append(ramp(fp, smin * 1.4 ** L_))
        if wakes:
            wl = []
            for wx, wy in wakes:
                for xx, yy in zip(wx, wy):
                    wl.append(gmsh.model.geo.addPoint(xx, yy, 0, wake_size * chord))
            geo.synchronize()
            for c in out_curves:
                gmsh.model.mesh.setTransfiniteCurve(c, 2)
            fw = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fw, "PointsList", wl)
            fwm = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(fwm, "F", f"{wake_size * chord} + 0.3*F{fw}")
            fields.append(fwm)
        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 10)      # меньше треугольников с углом < 30°
        gmsh.option.setNumber("Mesh.MeshSizeMax", s_far)
        gmsh.model.mesh.generate(2)
        ntags, ncoord, _ = gmsh.model.mesh.getNodes()
        gnodes = ncoord.reshape(-1, 3)[:, :2]
        gidx = np.full(int(ntags.max()) + 1, -1)
        gidx[ntags.astype(int)] = np.arange(len(ntags))
        tris = []
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        for et, en in zip(etypes, enodes):
            if int(et) != 2:
                raise BlockError("gmsh построил не треугольники")
            tris.append(gidx[en.reshape(-1, 3).astype(int)])
        tris = np.concatenate(tris)
        far_e = []
        for c in far:
            _, _, en = gmsh.model.mesh.getElements(1, c)
            far_e.append(gidx[en[0].reshape(-1, 2).astype(int)])
        far_e = np.concatenate(far_e)
    finally:
        gmsh.finalize()
    # --- сборка: узлы блоков, затем узлы gmsh (совпадающие с границами блоков — общие)
    from scipy.spatial import cKDTree
    nodes, cells, wall_edges, off = [], [], {}, 0
    for bi, b in enumerate(blocks):
        nodes.append(b["nodes"])
        cells += [list(q + off) for q in b["quads"]]
        for a, c in b["wall"]:
            wall_edges[(min(a, c) + off, max(a, c) + off)] = bi
        off += len(b["nodes"])
    bn = np.vstack(nodes)
    out_ids = np.concatenate([np.array(b["outline"]) + o for b, o in
                              zip(blocks, np.cumsum([0] + [len(b["nodes"]) for b in blocks[:-1]]))])
    tree = cKDTree(bn[out_ids])
    dmin, which = tree.query(gnodes)
    tol = 1e-9 * chord
    gmap = np.where(dmin < tol, out_ids[which], -1)
    new = gmap < 0
    gmap[new] = off + np.arange(new.sum())
    if (~new).sum() != len(out_ids):
        raise BlockError("узлы на границе блока не совпали с треугольной сеткой")
    allnodes = np.vstack([bn, gnodes[new]])
    cells += [list(gmap[tr]) for tr in tris]
    far_edges = {tuple(sorted(gmap[e])) for e in far_e}
    return _connectivity(allnodes, cells, wall_edges, far_edges, elements)
