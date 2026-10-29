"""Параллельный расчёт поляр на нескольких ядрах процессора.

Число ядер определяется автоматически на том компьютере, где запущена
программа (домашний и рабочий ПК могут отличаться — ничего настраивать не
нужно). Поляра делится на непрерывные куски по углу атаки: внутри куска
работает продолжение по α (каждая следующая точка стартует с соседней), куски
считаются одновременно в отдельных процессах. Несколько чисел Рейнольдса —
тоже одновременно.

Каждому процессу оставляется один поток линейной алгебры (BLAS), иначе
процессы мешают друг другу.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def cpu_count() -> int:
    """Логические процессоры этого компьютера (с учётом ограничения процесса)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def auto_workers() -> int:
    """По умолчанию: все логические процессоры, кроме одного (окну программы
    и системе), но не больше 32."""
    return int(min(32, max(1, cpu_count() - 1)))


def resolve_workers(value) -> int:
    """'авто' / '' / None / 0 → auto_workers(); иначе целое в пределах 1…cpu_count()."""
    if value is None:
        return auto_workers()
    s = str(value).strip().lower()
    if s in ("", "авто", "auto", "0"):
        return auto_workers()
    try:
        n = int(float(s))
    except ValueError:
        raise ValueError("Число ядер — целое число или «авто»") from None
    return int(min(max(1, n), cpu_count()))


def split_alphas(alphas, n_chunks: int, min_len: int, interleave: bool = False):
    """Индексы кусков отсортированных α (не короче min_len): непрерывных или, при
    interleave, через один (каждый кусок — весь диапазон с шагом k·Δα; у каждого
    есть точка у α = 0 — у механизации холодный старт при большом угле может
    искаться десятки минут или не найтись)."""
    order = np.argsort(alphas)
    n = len(order)
    k = int(max(1, min(n_chunks, n // max(1, min_len))))
    if interleave:
        return [order[j::k] for j in range(k) if len(order[j::k])]
    return [c for c in np.array_split(order, k) if len(c)]


_PROGRESS = None          # очередь «точка готова» (в процессах-исполнителях)


def _init_worker(progress=None):
    import warnings
    global _PROGRESS
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    _PROGRESS = progress


def _task(args):
    kind, payload, idx = args

    def cb(done, n, r):
        if _PROGRESS is not None:
            try:
                _PROGRESS.put((idx, float(r.alpha), bool(r.converged)))
            except Exception:                            # noqa: BLE001 — прогресс не обязателен
                pass
    if kind == "single":
        from .viscous import polar_robust
        af, re, alphas, ncrit, mach, xtr = payload
        res = polar_robust(af, re, alphas, ncrit=ncrit, mach=mach, xtr=xtr, callback=cb)
    else:
        from .mviscous import MultiViscousSolver
        elements, re, alphas, ncrit, mach, xtr, chord, npn = payload
        ms = MultiViscousSolver(elements, re, ncrit=ncrit, mach=mach, xtr=xtr, chord=chord, n_panels=npn)
        res = ms.polar(alphas, callback=cb)
        for r in res:
            r.n_panels = npn
    return idx, res


def run_tasks(tasks, workers: int, on_done=None, on_point=None):
    """tasks — список (kind, payload, idx); on_done(idx, results) после каждой
    задачи, on_point(idx, alpha, converged) после каждой точки; оба возвращают
    True для остановки. Возвращает {idx: results}."""
    import queue as _queue
    out = {}
    if workers <= 1 or len(tasks) <= 1:
        stop = [False]

        class _Direct:                                   # в этом же процессе — сразу, без очереди
            @staticmethod
            def put(item):
                if on_point(*item):
                    stop[0] = True
        _init_worker(_Direct if on_point else None)
        try:
            for t in tasks:
                idx, res = _task(t)
                out[idx] = res
                if stop[0] or (on_done and on_done(idx, res)):
                    break
        finally:
            _init_worker(None)
        return out
    keep = {v: os.environ.get(v) for v in _BLAS_VARS}
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    pool = mgr = None
    try:
        ctx = mp.get_context("spawn")       # одинаково в Windows и Linux, без копирования состояния окна
        q = None
        if on_point:
            mgr = ctx.Manager()
            q = mgr.Queue()
        pool = ctx.Pool(min(workers, len(tasks)), initializer=_init_worker, initargs=(q,))
        pend = [pool.apply_async(_task, (t,)) for t in tasks]
        stop = False
        import time as _time
        while pend and not stop:
            while q is not None and not stop:
                try:
                    stop = bool(on_point(*q.get_nowait()))
                except _queue.Empty:
                    break
            for a in [a for a in pend if a.ready()]:
                pend.remove(a)
                idx, res = a.get()
                out[idx] = res
                if on_done and on_done(idx, res):
                    stop = True
                    break
            if pend and not stop:
                _time.sleep(0.2)
        if stop:
            pool.terminate()
        else:
            pool.close()
        pool.join()
    finally:
        if pool is not None:
            pool.terminate()
        if mgr is not None:
            mgr.shutdown()
        for v, val in keep.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    return out
