"""Сценарии расчёта для интерфейса: профиль → поляра (панели, панели +
пограничный слой, Эйлер)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .geometry import Airfoil, naca4, load_airfoil
from .panel import PanelSolver, close_trailing_edge, critical_cp

# Сетки для уравнений Эйлера: (точек на сторону профиля, узлов по нормали, первый шаг)
GRIDS = {
    "грубая (быстро)": (65, 49, 2e-3),
    "средняя": (97, 65, 1.5e-3),
    "мелкая (медленно)": (129, 97, 1e-3),
}


@dataclass
class Polar:
    label: str
    method: str
    mach: float
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray            # NaN для панелей (невязкий расчёт без сопротивления)
    cm: np.ndarray
    ok: np.ndarray            # сошлось / в пределах применимости
    cp_x: list = field(default_factory=list)
    cp: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    re: float = float("nan")
    cdf: np.ndarray | None = None          # сопротивление трения (вязкий расчёт)
    xtr_u: np.ndarray | None = None        # точки перехода
    xtr_l: np.ndarray | None = None
    bl: list = field(default_factory=list)  # распределения пограничного слоя по углам
    cp_inv: list = field(default_factory=list)
    field_ctx: dict | None = None          # данные для расчёта поля течения (field.py)
    ch: np.ndarray | None = None           # шарнирный момент руля (controls.py)


def make_airfoil(source: str, value: str, n_side: int = 100, smooth: bool = False) -> Airfoil:
    """source = 'naca' (value — 4 цифры) или 'file' (value — путь). smooth —
    сгладить контур (для измеренных координат)."""
    if source == "lib":
        return close_trailing_edge(load_airfoil(value, n_side))
    if source == "naca" and "-" in value:
        from .geometry import naca_sym
        try:
            return naca_sym(value, n_side)
        except (ValueError, IndexError):
            raise ValueError("Симметричный профиль оперения: «0012-64» (модифицированный 4-значный: "
                             "толщина, индекс носка, положение наибольшей толщины) или «16-012»") from None
    if source == "naca":
        code = value.strip().upper().replace("NACA", "").strip()
        if len(code) != 4 or not code.isdigit():
            raise ValueError("Код NACA должен состоять из 4 цифр, например 2412")
        if int(code[2:]) == 0:
            raise ValueError("Толщина профиля не может быть нулевой")
        return naca4(code, n_side)
    af = load_airfoil(value, n_side, smooth=smooth)
    return close_trailing_edge(af)


def library():
    """Встроенный набор профилей (симметричные 6-й серии для оперения и др.):
    список (название, путь)."""
    import glob
    import os
    out = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "airfoils", "*.dat"))):
        with open(f, encoding="utf-8", errors="replace") as fh:
            name = fh.readline().strip().replace(" AIRFOIL", "").replace("NASA/LANGLEY ", "NACA ")
        out.append((name, f))
    return out


def alpha_range(a0: float, a1: float, da: float) -> np.ndarray:
    if da <= 0:
        raise ValueError("Шаг по углу атаки должен быть положительным")
    if a1 < a0:
        a0, a1 = a1, a0
    n = int(np.floor((a1 - a0) / da + 1e-9)) + 1
    if n > 200:
        raise ValueError("Слишком много углов атаки (больше 200)")
    return a0 + da * np.arange(n)


def _hinge(control, xs, ys, cps):
    from .controls import hinge_moment
    if not control:
        return None
    return np.array([hinge_moment(x, y, cp, control["hinge"], control["x_hinge"],
                                  control["deflection"], control.get("balance", 0.0))
                     for x, y, cp in zip(xs, ys, cps)])


def _gap_note(control):
    g = (control or {}).get("gap", 0.0)
    if not g:
        return []
    from .controls import gap_factors
    ka, kd = gap_factors(g)
    return [f"Щель у носка руля открыта: {g:g}c. Учтена эмпирической поправкой по продувкам NACA "
            f"(ARR L-301, NACA 0009, руль 0,30c, компенсация 0,20c_р): наклон cl(α) ×{ka:.3f}, "
            f"приращение cl от руля ×{kd:.3f}; ch пересчитан при той же связи ch(cl), что при "
            "герметичной щели (по опыту NACA она от щели почти не зависит). Для этого дополнительно "
            "посчитана поляра без отклонения руля. Cm не исправлен. Поправка проверена на щелях "
            "0,001–0,010c; для других профилей и рулей — оценка."]


def _control_note(control):
    if not control:
        return []
    return _gap_note(control) + [f"Руль: ось на x/c = {control['x_hinge']:g} (хорда руля {1 - control['x_hinge']:g}), "
            f"δ = {control['deflection']:g}°, осевая компенсация {100 * control.get('balance', 0):g} % "
            "хорды руля. Ch = M_ш/(q∞·c_р²), c_р — хорда руля за осью; Ch > 0 — момент отклоняет "
            "руль задней кромкой вниз. Расчёт выполнен с герметичной щелью"
            + (" (открытая — поправкой, см. выше)" if control.get("gap") else "") +
            ", давление в нише компенсации — равным давлению у кромок ниши (инженерная модель)."]


def run_panel(af: Airfoil, alphas, mach: float, control=None) -> Polar:
    if mach >= 1:
        raise ValueError("Панельный метод с поправкой Кармана–Цяня применим только при M < 1")
    s = PanelSolver(af)
    cl, cm, ok, xs, cps = [], [], [], [], []
    cpstar = critical_cp(mach)
    for a in alphas:
        r = s.solve(float(a), mach)
        cl.append(r.cl); cm.append(r.cm)
        ok.append(bool(np.min(r.cp) > cpstar))
        xs.append(r.x); cps.append(r.cp)
    ok = np.array(ok)
    notes = []
    if mach > 0 and not ok.all():
        bad = np.asarray(alphas)[~ok]
        notes.append(f"При α = {bad.min():g}…{bad.max():g}° на профиле есть сверхзвуковая зона "
                     f"(Cp < Cp* = {cpstar:.2f}): панели+КЦ там неприменимы, нужен Эйлер.")
    notes.append("Панельный метод невязкий: сопротивление и срыв не рассчитываются.")
    notes += _control_note(control)
    ys = [s.y for _ in alphas]
    cl, ch = np.array(cl), _hinge(control, xs, ys, cps)
    if control and control.get("gap") and control.get("base") is not None:
        from .controls import apply_gap
        sb = PanelSolver(control["base"])
        cl, ch = apply_gap(cl, ch, [sb.solve(float(a), mach).cl for a in alphas], control["gap"])
    return Polar(f"{af.name}, панели, M={mach:g}", "panel", mach, np.asarray(alphas, float),
                 cl, np.full(len(cl), np.nan), np.array(cm), ok, xs, cps, notes,
                 ch=ch,
                 field_ctx=dict(method="panel", airfoil=af, mach=mach, alpha=np.asarray(alphas, float)))


def run_multi(elements, alphas, mach: float, name: str = "многоэлементный", chord=None) -> Polar:
    """Невязкий расчёт многоэлементного профиля (предкрылок, закрылки)."""
    from .multi import MultiPanelSolver
    if mach >= 1:
        raise ValueError("Панельный метод с поправкой Кармана–Цяня применим только при M < 1")
    ms = MultiPanelSolver(elements, chord=chord)
    cpstar = critical_cp(mach)
    cl, cm, ok, xs, cps, per = [], [], [], [], [], []
    for a in alphas:
        r = ms.solve(float(a), mach)
        cl.append(r.cl); cm.append(r.cm)
        ok.append(bool(min(np.min(e["cp"]) for e in r.elements) > cpstar))
        # элементы — через NaN, чтобы на графике Cp они рисовались раздельно
        xs.append(np.concatenate([np.append(e["x"], np.nan) for e in r.elements]))
        cps.append(np.concatenate([np.append(e["cp"], np.nan) for e in r.elements]))
        per.append([e["cl"] for e in r.elements])
    per = np.array(per)
    notes = [f"Многоэлементный профиль, {len(elements)} элемента(ов); опорная хорда {ms.chord:.3f} "
             "(для закрылка из профиля — хорда исходного профиля, для файла — габарит). Расчёт НЕВЯЗКИЙ: пограничные слои, следы элементов "
             "и отрыв на закрылке не учитываются, поэтому Cl завышен (обычно на 10–30 % при "
             "отклонённом закрылке), сопротивление не рассчитывается. Это оценка распределения "
             "нагрузки и влияния щели/угла; вязкий расчёт механизации — следующий этап.",
             "Cl по элементам при каждом α: " + "; ".join(
                 f"{a:g}°: " + ", ".join(f"{c:.3f}" for c in row) for a, row in zip(alphas, per))]
    if mach > 0 and not all(ok):
        notes.append("Есть точки со сверхзвуковой зоной (Cp < Cp*): поправка Кармана–Цяня там неточна.")
    return Polar(f"{name}, панели, M={mach:g}", "multi", mach, np.asarray(alphas, float), np.array(cl),
                 np.full(len(cl), np.nan), np.array(cm), np.array(ok), xs, cps, notes,
                 field_ctx=dict(method="multi", elements=elements, mach=mach, chord=chord,
                                alpha=np.asarray(alphas, float)))


def run_euler(af: Airfoil, alphas, mach: float, grid: str = "грубая (быстро)",
              n_iter: int = 100, callback=None) -> Polar:
    """Все углы атаки считаются одним пакетом. callback(text, fraction) -> True для остановки."""
    import jax.numpy as jnp
    from .geometry import repanel
    from .grid import ogrid
    from .euler import EulerSolver
    from .implicit import ImplicitEuler

    if mach < 0.2:
        raise ValueError("Решатель Эйлера пока не работает при M < 0,2 (нет предобусловливания "
                         "для малых чисел Маха). Используйте панельный метод.")
    nside, nj, ds0 = GRIDS[grid]
    if callback:
        callback("Построение сетки…", 0.0)
    g = ogrid(repanel(af, nside), nj=nj, ds0=ds0, radius=40.0)
    solver = ImplicitEuler(EulerSolver(g, dtype=jnp.float32))
    alphas = np.asarray(alphas, float)

    def cb(it, n, logres, cl):
        if callback is None:
            return False
        worst = np.nanmax(logres) if np.any(np.isfinite(logres)) else 0.0
        return callback(f"Итерация {it}/{n}: невязка упала на {-worst:.1f} порядка "
                        f"(худший случай); первая итерация включает компиляцию", it / n)

    if callback:
        callback("Компиляция и первая итерация (может занять минуту)…", 0.0)
    r = solver.solve(np.full_like(alphas, mach), alphas, n_iter=n_iter, tol=1e-6,
                     verbose=False, callback=cb)
    xw = np.asarray(solver.b.m_np.xw)
    ok = np.asarray(r.converged, bool)
    notes = []
    if not ok.all():
        notes.append(f"Не сошлись: α = {', '.join(f'{a:g}' for a in alphas[~ok])}°. "
                     "Их значения ненадёжны (обычно трансзвук с сильными скачками или большие α). "
                     "Попробуйте более мелкую сетку или больше итераций.")
    notes.append("Эйлер невязкий: Cd — только волновое сопротивление (скачки уплотнения), "
                 "трения и срыва нет.")
    return Polar(f"{af.name}, Эйлер ({grid.split(' ')[0]}), M={mach:g}", "euler", mach, alphas,
                 r.cl, r.cd, r.cm, ok, [xw] * len(alphas), [c for c in r.cp], notes)


def run_viscous(af: Airfoil, alphas, mach: float, re: float, ncrit: float = 9.0,
                xtr=(1.0, 1.0), callback=None, control=None, workers=None) -> Polar:
    """Панели + интегральный пограничный слой (как XFOIL). callback(text, fraction)
    -> True для остановки. workers — число процессов (None — авто)."""
    return run_viscous_batch(af, alphas, mach, [re], ncrit, xtr, callback, control, workers)[0]


def _check_viscous(re_list, mach):
    for re in re_list:
        if not 1e3 <= re <= 1e9:
            raise ValueError("Число Рейнольдса должно быть в диапазоне 1e3…1e9")
    if mach >= 0.8:
        raise ValueError("Вязкий панельный метод применим при M < 0,8 (сжимаемость — поправкой "
                         "Кармана–Цяня); для трансзвука используйте Эйлер.")


def _run_parallel(kind, make_payload, alphas, re_list, workers, min_len, callback):
    """Общая часть: куски по α × числа Re → процессы; сборка результатов."""
    from .parallel import resolve_workers, split_alphas, run_tasks
    alphas = np.asarray(alphas, float)
    nw = resolve_workers(workers)
    per_re = max(1, nw // len(re_list))
    chunks = split_alphas(alphas, per_re, min_len)
    tasks = []
    for ir, re in enumerate(re_list):
        for ic, idx in enumerate(chunks):
            tasks.append((kind, make_payload(re, alphas[idx]), (ir, ic)))
    total = len(alphas) * len(re_list)
    done = {"n": 0}

    def on_done(key, res):
        done["n"] += len(res)
        nconv = sum(bool(r.converged) for r in res)
        if callback:
            return bool(callback(f"Готово {done['n']}/{total} точек ({min(nw, len(tasks))} процесс(ов) "
                                 f"из {nw}); в последнем куске сошлось {nconv}/{len(res)}",
                                 done["n"] / total))
        return False

    if callback:
        callback(f"Расчёт на {min(nw, len(tasks))} процессах (ядер для расчёта: {nw})…", 0.0)
    out = run_tasks(tasks, nw, on_done)
    if len(out) < len(tasks):
        raise RuntimeError("Расчёт остановлен")
    results = []
    for ir in range(len(re_list)):
        res = [None] * len(alphas)
        for ic, idx in enumerate(chunks):
            for j, r in zip(idx, out[(ir, ic)]):
                res[j] = r
        results.append(res)
    return alphas, results, min(nw, len(tasks))


def run_viscous_batch(af: Airfoil, alphas, mach: float, re_list, ncrit: float = 9.0,
                      xtr=(1.0, 1.0), callback=None, control=None, workers=None):
    """Поляры для нескольких Re сразу, параллельно на ядрах процессора."""
    re_list = [float(r) for r in np.atleast_1d(re_list)]
    _check_viscous(re_list, mach)
    alphas, results, nproc = _run_parallel(
        "single", lambda re, al: (af, re, al, ncrit, mach, tuple(xtr)), alphas, re_list, workers, 4, callback)
    bases = [None] * len(re_list)
    if control and control.get("gap") and control.get("base") is not None:
        base = control["base"]
        if callback:
            callback("Открытая щель: поляра без отклонения руля…", 0.0)
        _, rb, _ = _run_parallel("single", lambda re, al: (base, re, al, ncrit, mach, tuple(xtr)),
                                 alphas, re_list, workers, 4, callback)
        bases = [np.array([r.cl if r.converged else np.nan for r in res]) for res in rb]
    return [_viscous_polar(af, alphas, res, mach, re, ncrit, xtr, control, nproc, cb)
            for re, res, cb in zip(re_list, results, bases)]


def _viscous_polar(af, alphas, res, mach, re, ncrit, xtr, control, nproc, cl_base=None):
    ok = np.array([r.converged for r in res])
    cpstar = critical_cp(mach)
    sup = np.array([np.min(r.cp) < cpstar for r in res]) if mach > 0 else np.zeros(len(res), bool)
    notes = []
    if not ok.all():
        notes.append(f"Не сошлись: α = {', '.join(f'{a:g}' for a in alphas[~ok])}°. Обычно это "
                     "срыв (вязко-невязкий метод его описывает только до начала) или сильные "
                     "отрывные пузыри при малых Re. Если не сходятся и умеренные углы, а "
                     "координаты измеренные или оцифрованные, — включите «сгладить контур»: "
                     "шум координат у передней кромки делает расчёт слоя неустойчивым.")
    if sup.any():
        notes.append(f"При α = {', '.join(f'{a:g}' for a in alphas[sup])}° на профиле есть "
                     "сверхзвуковая зона: поправка Кармана–Цяня там неточна.")
    notes.append(f"Re = {re:.3g}, Ñкрит = {ncrit:g}. Ñкрит задаёт возмущённость потока: 9 — "
                 "обычная труба, 10–11 — малотурбулентная труба или спокойная атмосфера (при "
                 "сверке с продувками UIUC и NASA LTPT сопротивление при Re 1e5–3e5 ближе всего "
                 "при 10–11), 4–7 — турбулентный поток. Cd — по Сквайру–Янгу в конце следа, "
                 "Cdf — трение. Срыв на больших углах описывается приближённо.")
    notes.append(f"Считалось параллельно: {nproc} процесс(ов); поляра разбита на куски по α, "
                 "внутри куска — продолжение от соседней точки.")
    notes += _control_note(control)
    label = f"{af.name}, вязкий, Re={re:.3g}, M={mach:g}"
    cl = np.array([r.cl for r in res])
    ch = _hinge(control, [r.x for r in res], [r.y for r in res], [r.cp for r in res])
    if cl_base is not None:
        from .controls import apply_gap
        cl, ch = apply_gap(cl, ch, cl_base, control["gap"])
        ok = ok & np.isfinite(cl)
        label += f", щель {control['gap']:g}c"
    return Polar(label, "viscous", mach, alphas, cl,
                 np.array([r.cd for r in res]), np.array([r.cm for r in res]), ok & ~sup,
                 [r.x for r in res], [r.cp for r in res], notes, re=re,
                 cdf=np.array([r.cdf for r in res]),
                 xtr_u=np.array([r.xtr_u for r in res]), xtr_l=np.array([r.xtr_l for r in res]),
                 bl=[r.side for r in res], cp_inv=[r.cp_inv for r in res],
                 ch=ch,
                 field_ctx=dict(method="viscous", airfoil=af, re=re, ncrit=ncrit, mach=mach, xtr=xtr,
                                alpha=alphas, states=[r.state for r in res],
                                n_panels=[getattr(r, "n_panels", 160) for r in res]))


def run_multi_viscous(elements, alphas, mach: float, re: float, ncrit: float = 9.0, xtr=(1.0, 1.0),
                      callback=None, name: str = "многоэлементный", chord=None, workers=None) -> Polar:
    """Вязкий расчёт многоэлементного профиля (mviscous.py). Принудительный
    переход xtr (доля хорды по x) задаётся для основного (самого длинного)
    элемента; верх закрылков — турбулентный от носка (mviscous.default_transition)."""
    return run_multi_viscous_batch(elements, alphas, mach, [re], ncrit, xtr, callback, name, chord, workers)[0]


def run_multi_viscous_batch(elements, alphas, mach: float, re_list, ncrit: float = 9.0, xtr=(1.0, 1.0),
                            callback=None, name: str = "многоэлементный", chord=None, workers=None):
    from .mviscous import default_transition
    re_list = [float(r) for r in np.atleast_1d(re_list)]
    for re in re_list:
        if not 1e4 <= re <= 1e9:
            raise ValueError("Число Рейнольдса должно быть в диапазоне 1e4…1e9")
    if mach >= 0.6:
        raise ValueError("Вязкий расчёт механизации — при M < 0,6")
    chords = [float(np.max(e.x) - np.min(e.x)) for e in elements]
    imain = int(np.argmax(chords))
    xt = default_transition(elements, xtr)
    npn = [max(80, int(round(180 * c / max(chords) / 20)) * 20) for c in chords]
    if not chord:
        chord = float(max(np.max(e.x) for e in elements) - min(np.min(e.x) for e in elements))
    # холодный старт механизации дорогой (20–90 с) — куски по α не короче 3 точек
    alphas, results, nproc = _run_parallel(
        "multi", lambda re, al: (elements, re, al, ncrit, mach, xt, chord, npn), alphas, re_list, workers, 3,
        callback)
    out = []
    for re, res in zip(re_list, results):
        ok = np.array([r.converged for r in res])
        notes = [f"Вязкий расчёт механизации: {len(elements)} элемента(ов), у каждого свой пограничный "
                 f"слой и след. Re = {re:.3g}, Ñкрит = {ncrit:g}; принудительный переход xtr — на основном "
                 "элементе, верх закрылков — турбулентный от носка. След переднего элемента идёт над "
                 "закрылком отдельным слоем (слияние со слоем закрылка не моделируется). Сверка с "
                 "продувками GA(W)-1 + закрылок Фаулера 29 % (NASA CR-2443) — в README. Срыв закрылка "
                 "и Cl_max — приближённо.",
                 f"Считалось параллельно: {nproc} процесс(ов)."]
        if not ok.all():
            notes.append(f"Не сошлись: α = {', '.join(f'{a:g}' for a in alphas[~ok])}°.")
        notes.append("Cl по элементам: " + "; ".join(
            f"{r.alpha:g}°: " + ", ".join(f"{e['cl']:.3f}" for e in r.elements) for r in res))
        xs = [np.concatenate([np.append(e["x"], np.nan) for e in r.elements]) for r in res]
        cps = [np.concatenate([np.append(e["cp"], np.nan) for e in r.elements]) for r in res]
        cpi = [np.concatenate([np.append(e["cp_inv"], np.nan) for e in r.elements]) for r in res]
        out.append(Polar(f"{name}, вязкий, Re={re:.3g}, M={mach:g}", "multi_viscous", mach, alphas,
                         np.array([r.cl for r in res]), np.array([r.cd for r in res]),
                         np.array([r.cm for r in res]), ok, xs, cps, notes, re=re,
                         cdf=np.array([r.cdf for r in res]),
                         xtr_u=np.array([r.elements[imain]["xtr_u"] for r in res]),
                         xtr_l=np.array([r.elements[imain]["xtr_l"] for r in res]),
                         bl=[r.elements[imain]["side"] for r in res], cp_inv=cpi,
                         field_ctx=dict(method="multi_viscous", elements=elements, mach=mach, chord=chord,
                                        alpha=alphas, re=re, ncrit=ncrit, xtr=xt, n_panels=npn,
                                        states=[r.state for r in res])))
    return out


def run_rans(af, alphas, mach: float, re: float, callback=None, max_iter: int = 250,
             name: str | None = None, chord=None, mesh=None, mesh_opts=None, conv=None,
             control=None, workers: int = 1, init=None, rref=None, solver_opts=None) -> Polar:
    """RANS (Навье–Стокс + Спаларт–Аллмарас, полностью турбулентный) для профиля
    или механизации (список элементов). Медленно, но описывает отрыв и слияние
    следов в щелях — для Cl_max механизации. callback(text, fraction) -> True для остановки.
    mesh — готовая сетка (иначе строится по mesh_opts: rans_mesh.MeshOptions);
    conv — rans.Convergence; control — rans.RansControl (монитор итераций, «следующий угол»)."""
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError) as e:
        raise ValueError("Для RANS нужен генератор сеток gmsh: pip install gmsh (в офлайн-комплекте "
                         "ставится файлом 2_install_WINDOWS_cpu.bat; в Linux/WSL ещё нужны библиотеки "
                         f"libGLU, libXcursor, libXft, libXinerama). Ошибка: {e}") from None
    from .rans import mesh_for, rans_polar
    elements = af if isinstance(af, list) else [af]
    if not 1e5 <= re <= 1e9:
        raise ValueError("RANS — при Re от 1e5 до 1e9 (модель полностью турбулентная)")
    if mach >= 0.9:
        raise ValueError("RANS — при M < 0,9")
    m_note = None
    if mach < 0.1:              # сжимаемый решатель: при M → 0 сходится медленно
        m_note = f"Задано M = {mach:g}; RANS считался при M = 0,1 (сжимаемость < 0,5 %)."
        mach = 0.1
    alphas = np.asarray(alphas, float)
    if not chord:
        chord = float(max(np.max(e.x) for e in elements) - min(np.min(e.x) for e in elements))
    if mesh is None:
        if callback:
            callback("Построение сетки (gmsh)…", 0.0)
        mesh = mesh_for(elements, float(np.median(alphas)), re, chord, opts=mesh_opts)
    n = len(alphas)

    def cb(k, n_, it, rn, r):
        if callback is None:
            return False
        if r is not None:
            txt = f"α = {r.alpha:g}°: Cl = {r.cl:.3f}, {r.status}"
        else:
            txt = (f"Готово углов {k} из {n}; идёт итерация {it}/{max_iter}, невязка {rn:.2e} "
                   f"(первая итерация каждого процесса включает компиляцию)")
        return callback(txt, min(1.0, (k + (it / max_iter if r is None else 0)) / n))
    res, mesh = rans_polar(elements, alphas, re, mach=mach, chord=chord, callback=cb, max_iter=max_iter,
                           mesh=mesh, conv=conv, control=control, workers=workers, init=init, rref=rref,
                           solver_opts=solver_opts)
    done = np.array([r is not None for r in res])
    nan = float("nan")
    cl = np.array([r.cl if r else nan for r in res])
    cd = np.array([r.cd if r else nan for r in res])
    cdf = np.array([r.cdf if r else nan for r in res])
    cm = np.array([r.cm if r else nan for r in res])
    ok = np.array([bool(r and (r.converged or r.status == "принято вручную")) for r in res])
    # Cp вдоль контура каждого элемента (грани стенки — по порядку точек контура)
    from scipy.spatial import cKDTree
    xs_all, cps_all = [], []
    for r in res:
        if r is None:
            xs_all.append(np.array([nan])); cps_all.append(np.array([nan]))
            continue
        xx, cc = [], []
        for e, el in zip(elements, r.elements):
            _, j = cKDTree(np.column_stack([e.x, e.y])).query(np.column_stack([el["x"], el["y"]]))
            o = np.argsort(j, kind="stable")
            xx.append(np.append(el["x"][o], nan)); cc.append(np.append(el["cp"][o], nan))
        xs_all.append(np.concatenate(xx)); cps_all.append(np.concatenate(cc))
    notes = [f"RANS: уравнения Навье–Стокса, осреднённые по Рейнольдсу, модель турбулентности "
             f"Спаларта–Аллмараса; пограничный слой турбулентный от носка (переход не моделируется). "
             f"Сетка gmsh: {mesh.n_cells} ячеек. Re = {re:.3g}, M = {mach:g}.",
             "Проверка: NACA 0012 (NASA TMR) и GA(W)-1 + закрылок Фаулера (CR-2443) — в README."]
    if m_note:
        notes.append(m_note)
    if not done.all():
        notes.append("Расчёт остановлен: часть углов не посчитана.")
    if workers > 1 and n > 1:
        notes.append(f"Углы считались параллельно: до {min(workers, n - 1)} процессов (первый угол — отдельно, "
                     "остальные куски стартуют от его решения).")
    notes.append("Итог по углам: " + "; ".join(f"{r.alpha:g}° — {r.status}, {r.iterations} итер." for r in res if r))
    bad = done & ~ok
    if bad.any():
        notes.append(f"Не сошлись: α = {', '.join(f'{a:g}' for a in alphas[bad])}° "
                     "(за Cl_max течение может быть нестационарным; ход невязок — вкладка «Невязки»).")
    if len(elements) > 1:
        notes.append("Cl по элементам: " + "; ".join(
            f"{r.alpha:g}°: " + ", ".join(f"{e['cl']:.3f}" for e in r.elements) for r in res if r))
    nm = name or (elements[0].name if len(elements) == 1 else "многоэлементный")
    return Polar(f"{nm}, RANS-SA, Re={re:.3g}, M={mach:g}", "rans", mach, alphas, cl, cd, cm, ok,
                 xs_all, cps_all, notes, re=re, cdf=cdf,
                 field_ctx=dict(method="rans", mesh=mesh, elements=elements, mach=mach, re=re, chord=chord,
                                alpha=alphas, states=[r.state if r else None for r in res],
                                history=[r.history if r else None for r in res]))


def continue_rans(p: Polar, callback=None, max_iter: int = 300, conv=None, control=None, workers: int = 1,
                  only_bad: bool = True, solver_opts=None) -> Polar:
    """Досчитать поляру RANS: несошедшиеся (или все при only_bad=False) углы
    продолжаются каждый со своего решения на той же сетке; не посчитанные —
    от ближайшего посчитанного. Возвращает новую поляру (прежняя не меняется)."""
    import copy
    ctx = p.field_ctx
    if not ctx or ctx.get("method") != "rans":
        raise ValueError("Досчитать можно только поляру RANS этого сеанса")
    alphas = np.asarray(p.alpha, float)
    idx = [k for k in range(len(alphas)) if not (only_bad and p.ok[k])]
    if not idx:
        raise ValueError("Все углы этой поляры уже сошлись")
    have = [k for k in range(len(alphas)) if ctx["states"][k] is not None]
    if not have:
        raise ValueError("В поляре нет ни одного посчитанного угла")
    init = {}
    for j, k in enumerate(idx):
        src = k if ctx["states"][k] is not None else min(have, key=lambda h: abs(alphas[h] - alphas[k]))
        init[j] = ctx["states"][src]
    rref = max([H.get("rref", 0.0) for H in ctx["history"] if H] or [0.0]) or None
    q = run_rans(ctx["elements"], alphas[idx], p.mach, p.re, callback, max_iter, chord=ctx["chord"],
                 mesh=ctx["mesh"], conv=conv, control=control, workers=workers, init=init, rref=rref,
                 solver_opts=solver_opts)
    out = copy.copy(p)
    out.label = p.label if p.label.endswith("(досчитано)") else p.label + " (досчитано)"
    for name in ("cl", "cd", "cm", "cdf", "ok"):
        arr = np.array(getattr(p, name), copy=True)
        arr[idx] = getattr(q, name)
        setattr(out, name, arr)
    out.cp_x, out.cp = list(p.cp_x), list(p.cp)
    states, hist = list(ctx["states"]), list(ctx["history"])
    for j, k in enumerate(idx):
        out.cp_x[k], out.cp[k] = q.cp_x[j], q.cp[j]
        if q.field_ctx["states"][j] is not None:
            states[k] = q.field_ctx["states"][j]
        Hn, Ho = q.field_ctx["history"][j], hist[k]
        if Hn is not None and Ho is not None and len(Ho["it"]):
            H = {key: (np.concatenate([Ho[key], Hn[key] + (Ho["it"][-1] if key == "it" else 0)])
                       if key != "rref" else max(Ho["rref"], Hn["rref"])) for key in Hn}
            hist[k] = H
        elif Hn is not None:
            hist[k] = Hn
    out.field_ctx = dict(ctx, states=states, history=hist)
    out.field_ctx.pop("_tri", None)
    out.notes = list(p.notes) + [f"Досчитано: α = {', '.join(f'{a:g}' for a in alphas[idx])}° — каждый угол "
                                 "продолжен со своего решения на той же сетке."] + \
        [n for n in q.notes if n.startswith("Итог") or n.startswith("Не сошлись")]
    return out


class _Stop(Exception):
    pass


def save_csv(polars, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["расчёт", "метод", "M", "Re", "alpha", "Cl", "Cd", "Cdf", "Cm",
                    "xtr_верх", "xtr_низ", "надёжно"])
        f6 = lambda v: "" if v is None or not np.isfinite(v) else f"{v:.6f}"
        for p in polars:
            for k in range(len(p.alpha)):
                w.writerow([p.label, p.method, f"{p.mach:g}",
                            "" if not np.isfinite(p.re) else f"{p.re:.4g}", f"{p.alpha[k]:g}",
                            f6(p.cl[k]), f6(p.cd[k]), f6(None if p.cdf is None else p.cdf[k]),
                            f6(p.cm[k]), f6(None if p.xtr_u is None else p.xtr_u[k]),
                            f6(None if p.xtr_l is None else p.xtr_l[k]),
                            "да" if p.ok[k] else "нет"])
