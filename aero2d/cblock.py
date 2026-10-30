"""Сетка RANS с С-блоком у каждого элемента (как С-сетка в ICEM CFD).

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
        last = ids[-1]
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


def c_block(x, y, h0, ratio, t, smin, wake_len, obstacles=(), clearance=0.0, wake_growth=1.15,
            smooth=None):
    """См. _c_block; smooth=None — сглаживание нормалей подбирается: самое слабое, при
    котором блок не выворачивается (у острой кромки над разрезом — вогнутый угол)."""
    err = None
    for sm in ((smooth,) if smooth is not None else (10, 30, 60, 120, 250)):
        try:
            b = _c_block(x, y, h0, ratio, t, smin, wake_len, obstacles, clearance, wake_growth, sm)
            b["smooth"] = sm
            return b
        except BlockError as e:
            err = e
            if "вывернулся" not in str(e):
                raise
    raise err


def _c_block(x, y, h0, ratio, t, smin, wake_len, obstacles=(), clearance=0.0, wake_growth=1.15,
             smooth=3, te_min=None):
    """Структурированный С-блок вокруг одного элемента.

    x, y — замкнутый контур, первая точка — задняя кромка (без повтора в конце).
    obstacles — отрезки (m,2,2) других тел: разрез укорачивается, чтобы до них
    оставалось не меньше clearance. Возвращает dict: nodes (n,2), quads (k,4),
    wall (рёбра стенки, пары узлов), outline (замкнутый контур блока, индексы узлов),
    out_size (размер треугольников у каждой точки outline)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-12:
        x, y = x[:-1], y[:-1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:                                   # против часовой: снаружи — справа по обходу
        x, y = np.r_[x[:1], x[1:][::-1]], np.r_[y[:1], y[1:][::-1]]
    Pb = _find_base(x, y)
    # тупая задняя кромка — полоса за торцом; торец тоньше 0,02 % хорды (численный след
    # построения контура) — как острая кромка
    if Pb is not None and np.hypot(*(Pb[0] - Pb[-1])) > 2e-4 * float(np.ptp(x)):
        return _c_block_blunt(Pb, h0, ratio, t, smin, wake_len, obstacles, wake_growth, smooth)
    # --- толщина слоя t (по самой узкой щели), число слоёв n у всех столбцов одно
    n = int(np.ceil(np.log(1 + t * (ratio - 1) / h0) / np.log(ratio)))
    t_min = 1.5 * h0 * n
    obst = np.asarray(obstacles) if len(obstacles) else None

    def thickness(P):
        # одна толщина на весь блок (t уже ограничена самой узкой щелью): переменная
        # толщина давала свои перепады на стыках толстых и тонких участков
        return np.full(len(P), max(t, t_min))

    def h_top(T):
        T = np.atleast_1d(T)
        qq = np.array([_ratio_for(h0, Ti, n) for Ti in T])
        return h0 * qq ** (n - 1)
    P0 = np.column_stack([x, y])
    T0 = thickness(P0)
    x, y = _coarsen_te(x, y, te_min if te_min else 1.2 * float(h_top(T0[0])[0]))
    P0 = np.column_stack([x, y])
    T0 = thickness(P0)
    # шаг вдоль поверхности не больше 2,5 высоты верхнего ряда (ячейки у границы блока
    # близки к квадратным — иначе к их широкому верху примыкают треугольники крупнее)
    cap = np.clip(4.0 * h_top(np.minimum(T0, np.roll(T0, -1))), 0.4 * smin, smin)
    if obst is not None:
        # в щели между блоками на треугольники остаётся полоса ≈ 0,4 зазора — шаг блока
        # не крупнее её (иначе к широким верхним ячейкам примыкают мелкие треугольники)
        mid = 0.5 * (P0 + np.roll(P0, -1, axis=0))
        cap = np.minimum(cap, np.maximum(0.4 * _seg_dist(mid, obst), 0.1 * smin))
    sx, sy = _subdivide(x, y, cap)
    S = np.column_stack([sx, sy])
    M = len(S)
    TS = thickness(S)
    # где шаг вдоль поверхности мелкий (носок, кромка), слой тоньше: верхний ряд не выше
    # 1,5 шага — иначе узкие высокие ячейки у границы блока и крошечные треугольники над ними;
    # толщины соседних столбцов различаются не больше чем в 1,2 раза
    wS = np.hypot(*np.diff(np.vstack([S, S[:1]]), axis=0).T)
    wS = np.minimum(wS, np.roll(wS, 1))
    Tg = np.geomspace(t_min, max(t, t_min), 60)
    Hg = h_top(Tg)
    TS = np.minimum(TS, np.interp(1.5 * wS, Hg, Tg, left=t_min, right=t))
    for _ in range(len(S)):
        T2 = np.minimum(TS, 1.2 * np.minimum(np.roll(TS, 1), np.roll(TS, -1)))
        if np.allclose(T2, TS):
            break
        TS = T2
    # --- разрез за задней кромкой
    t1, t2 = S[1] - S[0], S[-1] - S[0]
    u = -(t1 / np.linalg.norm(t1) + t2 / np.linalg.norm(t2))
    if np.linalg.norm(u) < 1e-9:
        raise BlockError("задняя кромка не острая")
    u /= np.linalg.norm(u)
    w0 = 0.5 * (np.linalg.norm(t1) + np.linalg.norm(t2))
    T_te = float(TS[0])
    L = 30 * T_te                                # не длиннее; ниже — по соседним телам
    if obst is not None:
        dd = np.linspace(0.0, L, 200)[1:]
        cl = _seg_dist(S[0] + dd[:, None] * u, obst)
        bad = np.where(cl < 1.6 * T_te)[0]       # свой слой + слой соседа (≤ 0,3 зазора) + запас
        if len(bad):
            L = dd[bad[0]] * 0.8
    # шаг вдоль разреза: от шага контура у кромки к высоте ряда в конце разреза (там ячейки
    # почти квадратные — к торцу блока примыкают треугольники того же размера); соседние
    # шаги различаются не больше чем в wake_growth раз, длина разреза — под это
    h_end = T_te / n
    L = min(L, max(wake_len, (w0 - h_end) / (1 - 1 / wake_growth) + 4 * h_end))
    if L < 3 * max(w0, h_end):
        raise BlockError("нет места для разреза за кромкой")
    d, s_ = [], w0
    while not d or d[-1] < L:
        d.append((d[-1] if d else 0.0) + s_)
        s_ = max(s_ / wake_growth, h_end) if s_ > h_end else min(s_ * wake_growth, h_end)
    d = np.array(d) * (L / d[-1])
    W = S[0] + d[:, None] * u
    K = len(W)
    # --- путь: верх разреза (от конца к кромке), контур, кромка ещё раз, низ разреза
    path = np.vstack([W[::-1], S, S[:1], W])
    base = np.r_[np.arange(K)[::-1] + M, np.arange(M), 0, np.arange(K) + M]   # узлы слоя 0
    dist = np.r_[d[::-1], np.zeros(M + 1), d]                                 # расстояние по разрезу
    Np = len(path)
    # --- слои: одинаковое число n у всех столбцов, высота первой ячейки на разрезе растёт
    h1 = h0 * (h_end / h0) ** np.clip(dist / L, 0.0, 1.0)
    Tp = np.r_[np.full(K, T_te), TS, TS[:1], np.full(K, T_te)]
    q = np.array([_ratio_for(h, Ti, n) for h, Ti in zip(h1, Tp)])
    j = np.arange(n + 1)
    z = np.where(np.abs(q - 1)[:, None] < 1e-12, h1[:, None] * j[None],
                 h1[:, None] * (q[:, None] ** j[None] - 1) / np.where(np.abs(q - 1) < 1e-12, 1, q - 1)[:, None])
    # --- выдавливание слоёв по сглаженным нормалям
    layers = [path.copy()]
    for k in range(1, n + 1):
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
    # --- узлы и ячейки
    nb = M + K
    nodes = np.vstack([S, W] + [Lr for Lr in layers[1:]])
    idx = np.empty((n + 1, Np), int)
    idx[0] = base
    for k in range(1, n + 1):
        idx[k] = nb + (k - 1) * Np + np.arange(Np)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
    # проверка: все четырёхугольники выпуклые и одной ориентации
    P = nodes[quads]
    e1 = np.roll(P, -1, axis=1) - P
    e0 = P - np.roll(P, 1, axis=1)
    cr = e0[..., 0] * e1[..., 1] - e0[..., 1] * e1[..., 0]
    sgn = np.sign(np.median(cr))
    if not np.all(cr * sgn > 0):
        badq = np.where(np.any(cr * sgn <= 0, axis=1))[0]
        c = P[badq].mean(1)
        lay = badq // (Np - 1)
        raise BlockError(f"блок вывернулся: {len(badq)} ячеек, например у ({c[0, 0]:.4f}; {c[0, 1]:.4f}), "
                         f"слои {lay.min()}–{lay.max()}")
    wall = [(base[K + i], base[K + i + 1]) for i in range(M)]       # контур, включая S_{M-1} → S_0
    outline = list(idx[n]) + list(idx[n - 1:0:-1, -1]) + [idx[0, -1]] + list(idx[1:n, 0])
    # размер треугольников у границы блока: шаг вдоль крайнего ряда, но не меньше его высоты
    top = layers[-1]
    ds = np.hypot(*np.diff(top, axis=0).T)
    w_top = np.r_[ds[0], np.maximum(ds[:-1], ds[1:]), ds[-1]]
    h_top = z[:, n] - z[:, n - 1]
    size_top = np.maximum(w_top, 1.2 * h_top)
    hh = np.diff(z, axis=1)
    size_end_lo = [hh[-1, k] for k in range(n - 1, 0, -1)] + [hh[-1, 0]]
    size_end_up = [hh[0, k] for k in range(1, n)]
    out_size = list(size_top) + size_end_lo + size_end_up
    return dict(nodes=nodes, quads=quads, wall=wall, outline=outline, out_size=np.array(out_size),
                layers=n, wake_len=L)


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


def _layers_z(h1, T, n):
    q = np.array([_ratio_for(h, Ti, n) for h, Ti in zip(h1, T)])
    j = np.arange(n + 1)
    qm = np.where(np.abs(q - 1) < 1e-12, 1.0, q)
    return np.where(np.abs(q - 1)[:, None] < 1e-12, h1[:, None] * j[None],
                    h1[:, None] * (qm[:, None] ** j[None] - 1) / np.where(np.abs(q - 1) < 1e-12, 1, q - 1)[:, None])


def _c_block_blunt(Pb, h0, ratio, t, smin, wake_len, obstacles=(), wake_growth=1.15, smooth=10):
    """С-блок для тупой задней кромки: разрезы от верхнего и нижнего углов торца по
    потоку, С-блок «верхний разрез → контур → нижний разрез» и структурированная полоса
    между разрезами за торцом (торец — стенка). Ряды полосы сгущаются к разрезам так же,
    как первые ячейки С-блока по ту сторону разреза."""
    n = int(np.ceil(np.log(1 + t * (ratio - 1) / h0) / np.log(ratio)))
    t_min = 1.5 * h0 * n
    t = max(t, t_min)
    obst = np.asarray(obstacles) if len(obstacles) else None

    def h_top(T):
        T = np.atleast_1d(T)
        return h0 * np.array([_ratio_for(h0, Ti, n) for Ti in T]) ** (n - 1)
    ht = float(h_top(t)[0])
    P = _coarsen_start(Pb, 1.2 * ht)
    P = _coarsen_start(P[::-1], 1.2 * ht)[::-1]
    cap = np.full(len(P) - 1, float(np.clip(4.0 * ht, 0.4 * smin, smin)))
    if obst is not None:
        mid = 0.5 * (P[1:] + P[:-1])
        cap = np.minimum(cap, np.maximum(0.4 * _seg_dist(mid, obst), 0.1 * smin))
    sx, sy = _subdivide(P[:, 0], P[:, 1], cap, closed=False)
    S = np.column_stack([sx, sy])
    M = len(S)
    Pu, Pl = S[0], S[-1]
    tb = float(np.hypot(*(Pu - Pl)))
    nh = max(1, int(np.ceil(np.log(1 + 0.5 * tb * (ratio - 1) / h0) / np.log(ratio))))
    mb = 2 * nh
    # толщина: тоньше, где шаг вдоль поверхности мелкий (верхний ряд не выше 1,5 шага), и у
    # торца — такая, чтобы в конце разреза ряды С-блока совпали с рядами полосы (tb/mb)
    seg = np.hypot(*np.diff(S, axis=0).T)
    wS = np.minimum(np.r_[seg[0], seg], np.r_[seg, seg[-1]])
    Tg = np.geomspace(t_min, t, 60)
    TS = np.interp(1.5 * wS, h_top(Tg), Tg, left=t_min, right=t)
    T_end = float(np.clip(n * tb / mb, t_min, t))
    TS[0], TS[-1] = min(TS[0], T_end), min(TS[-1], T_end)
    for _ in range(M):
        nb_ = np.minimum(np.r_[TS[0], TS[:-1]], np.r_[TS[1:], TS[-1]])
        T2 = np.minimum(TS, 1.2 * nb_)
        if np.allclose(T2, TS):
            break
        TS = T2
    tu, tl = S[0] - S[1], S[-1] - S[-2]
    u = tu / np.linalg.norm(tu) + tl / np.linalg.norm(tl)
    if np.linalg.norm(u) < 1e-9:
        raise BlockError("не определить направление разреза за торцом")
    u /= np.linalg.norm(u)
    w0 = 0.5 * (np.linalg.norm(tu) + np.linalg.norm(tl))
    T_te = float(min(TS[0], TS[-1]))
    h_cut = min(T_te / n, tb / mb)
    L = 30 * T_te
    if obst is not None:
        dd = np.linspace(0.0, L, 200)[1:]
        cl = np.minimum(_seg_dist(Pu + dd[:, None] * u, obst), _seg_dist(Pl + dd[:, None] * u, obst))
        bad = np.where(cl < 1.6 * T_te)[0]
        if len(bad):
            L = dd[bad[0]] * 0.8
    L = min(L, max(wake_len, (w0 - h_cut) / (1 - 1 / wake_growth) + 4 * h_cut))
    if L < 3 * max(w0, h_cut):
        raise BlockError("нет места для разреза за торцом")
    d, s_ = [], w0
    while not d or d[-1] < L:
        d.append((d[-1] if d else 0.0) + s_)
        s_ = max(s_ / wake_growth, h_cut) if s_ > h_cut else min(s_ * wake_growth, h_cut)
    d = np.array(d) * (L / d[-1])
    K = len(d)
    Wu, Wl = Pu + d[:, None] * u, Pl + d[:, None] * u
    # --- С-блок
    path = np.vstack([Wu[::-1], S, Wl])
    Np = len(path)
    base = np.r_[(M + np.arange(K))[::-1], np.arange(M), M + K + np.arange(K)]
    dist = np.r_[d[::-1], np.zeros(M), d]
    h1 = h0 * (h_cut / h0) ** np.clip(dist / L, 0.0, 1.0)
    Tp = np.r_[np.full(K, TS[0]), TS, np.full(K, TS[-1])]
    z = _layers_z(h1, Tp, n)
    layers = _march(path, z, smooth)
    # --- полоса за торцом: столбцы k = 0 (торец) … K, ряды j = 0 (верхний разрез) … mb
    dk = np.r_[0.0, d]
    h1s = h0 * (h_cut / h0) ** np.clip(dk / L, 0.0, 1.0)
    zs = _layers_z(h1s, np.full(K + 1, 0.5 * tb), nh)                 # (K+1, nh+1) от разреза
    A = np.vstack([Pu[None], Wu])
    B = np.vstack([Pl[None], Wl])
    e = (B - A) / tb
    strip = []
    for j in range(1, mb):
        strip.append(A + e * zs[:, j:j + 1] if j <= nh else B - e * zs[:, mb - j:mb - j + 1])
    strip = np.stack(strip, 1)                                         # (K+1, mb-1, 2)
    off_s = M + 2 * K
    off_c = off_s + (K + 1) * (mb - 1)

    def G(k, j):
        if j == 0:
            return 0 if k == 0 else M + k - 1
        if j == mb:
            return M - 1 if k == 0 else M + K + k - 1
        return off_s + k * (mb - 1) + (j - 1)
    sq = np.array([[G(k, j), G(k + 1, j), G(k + 1, j + 1), G(k, j + 1)] for k in range(K) for j in range(mb)])
    idx = np.empty((n + 1, Np), int)
    idx[0] = base
    for k in range(1, n + 1):
        idx[k] = off_c + (k - 1) * Np + np.arange(Np)
    cq = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
    nodes = np.vstack([S, Wu, Wl, strip.reshape(-1, 2)] + layers[1:])
    _check(nodes, cq, "у стенки")
    _check(nodes, sq, "за торцом")
    wall = [(i, i + 1) for i in range(M - 1)] + [(G(0, j), G(0, j + 1)) for j in range(mb)]
    end_strip = [G(K, j) for j in range(mb - 1, 0, -1)]
    outline = list(idx[n]) + list(idx[n - 1:0:-1, -1]) + [idx[0, -1]] + end_strip + [idx[0, 0]] + list(idx[1:n, 0])
    top = layers[-1]
    ds_ = np.hypot(*np.diff(top, axis=0).T)
    w_top = np.r_[ds_[0], np.maximum(ds_[:-1], ds_[1:]), ds_[-1]]
    hh = np.diff(z, axis=1)
    size_top = np.maximum(w_top, 1.2 * hh[:, -1])
    end_pts = nodes[[idx[0, -1]] + end_strip + [idx[0, 0]]]
    es = np.hypot(*np.diff(end_pts, axis=0).T)
    size_strip = list(np.maximum(es[:-1], es[1:]))
    out_size = (list(size_top) + [hh[-1, k] for k in range(n - 1, 0, -1)] + [max(hh[-1, 0], es[0])]
                + size_strip + [max(hh[0, 0], es[-1])] + [hh[0, k] for k in range(1, n)])
    return dict(nodes=nodes, quads=np.vstack([cq, sq]), wall=wall, outline=outline,
                out_size=np.array(out_size), layers=n, wake_len=L, blunt=tb)


def build_cblock_mesh(elements, chord, h0, ratio, t, far_radius, far_size, wakes, wake_size, smin, growth,
                      cut_len=1.5, verbose=False):
    """Сетка: С-блоки у элементов + треугольники gmsh снаружи (см. описание модуля)."""
    import gmsh
    from .rans_mesh import _connectivity
    polys = []
    for e in elements:
        x, y = np.asarray(e.x, float), np.asarray(e.y, float)
        polys.append(np.stack([np.column_stack([x, y]), np.column_stack([np.roll(x, -1), np.roll(y, -1)])], 1))
    blocks = []
    for i, e in enumerate(elements):
        obst = np.concatenate([p for k, p in enumerate(polys) if k != i]) if len(elements) > 1 else []
        blocks.append(c_block(e.x, e.y, h0, ratio, t, smin, cut_len * t, obst))
    # контуры блоков не должны пересекаться
    outl = [b["nodes"][b["outline"]] for b in blocks]
    for i in range(len(outl)):
        for k in range(i + 1, len(outl)):
            segs = np.stack([outl[k], np.roll(outl[k], -1, axis=0)], 1)
            if _seg_dist(outl[i], segs).min() < 1e-6 * chord:
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
        lev = np.minimum(lev, 0)
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
