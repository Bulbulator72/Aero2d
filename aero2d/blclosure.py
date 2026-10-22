"""Замыкающие соотношения интегрального пограничного слоя (Drela & Giles 1987,
Drela 1989; в форме, используемой в XFOIL).

Все функции векторизованы и допускают комплексные аргументы (производные
считаются методом комплексного шага), поэтому:
  * сравнения делаются по вещественной части (_lt, _where);
  * вместо abs/max/min — функции ниже.

Обозначения: hk — кинематический формпараметр, rt — Re_θ, msq — Me².
"""
from __future__ import annotations

import numpy as np

GACON, GBCON, GCCON = 6.70, 0.75, 18.0
CTCON = 0.5 / (GACON**2 * GBCON)
SCCON = 5.6
DLCON = 0.9
GM1 = 0.4


def _re(a):
    return np.real(a)


def _where(c, a, b):
    return np.where(c, a, b)


def _max(a, b):
    return np.where(_re(a) > _re(b), a, b)


def _min(a, b):
    return np.where(_re(a) < _re(b), a, b)


# ---------------------------------------------------------------------------
def hkin(h, msq):
    """Кинематический формпараметр (Whitfield)."""
    return (h - 0.29 * msq) / (1.0 + 0.113 * msq)


def hct(hk, msq):
    """Формпараметр плотности H**."""
    return msq * (0.064 / (hk - 0.8) + 0.251)


# ---------------------------- ламинарные ----------------------------------
def hsl(hk):
    tmp = hk - 4.35
    a = 0.0111 * tmp**2 / (hk + 1.0) - 0.0278 * tmp**3 / (hk + 1.0) + 1.528 - 0.0002 * (tmp * hk) ** 2
    b = 0.015 * tmp**2 / hk + 1.528
    return _where(_re(hk) < 4.35, a, b)


def cfl(hk, rt):
    a = (0.0727 * (5.5 - hk) ** 3 / (hk + 1.0) - 0.07) / rt
    tmp = 1.0 - 1.0 / (hk - 4.5)
    b = (0.015 * tmp**2 - 0.07) / rt
    return _where(_re(hk) < 5.5, a, b)


def dil(hk, rt):
    """2 C_D / H* (ламинарный)."""
    a = (0.00205 * (4.0 - _min(hk, 4.0)) ** 5.5 + 0.207) / rt
    hkb = hk - 4.0
    b = (-0.0016 * hkb**2 / (1.0 + 0.02 * hkb**2) + 0.207) / rt
    return _where(_re(hk) < 4.0, a, b)


def dilw(hk, rt, hs):
    """2 C_D / H* для ламинарного следа."""
    rcd = 1.10 * (1.0 - 1.0 / hk) ** 2 / hk
    return 2.0 * rcd / (hs * rt)


# ---------------------------- турбулентные --------------------------------
def hst(hk, rt, msq):
    hsmin, dhsinf = 1.5, 0.015
    ho = _where(_re(rt) > 400.0, 3.0 + 400.0 / rt, 4.0 + 0 * rt)
    rtz = _where(_re(rt) > 200.0, rt, 200.0 + 0 * rt)
    hr = (ho - hk) / (ho - 1.0)
    a = (2.0 - hsmin - 4.0 / rtz) * hr**2 * 1.5 / (hk + 0.5) + hsmin + 4.0 / rtz
    grt = np.log(rtz)
    hdif = hk - ho
    rtmp = hk - ho + 4.0 / grt
    htmp = 0.007 * grt / rtmp**2 + dhsinf / hk
    b = hdif**2 * htmp + hsmin + 4.0 / rtz
    hs = _where(_re(hk) < _re(ho), a, b)
    return (hs + 0.028 * msq) / (1.0 + 0.014 * msq)


def cft(hk, rt, msq):
    fc = np.sqrt(1.0 + 0.5 * GM1 * msq)
    grt = _max(np.log(rt / fc), 3.0 + 0 * rt)
    gex = -1.74 - 0.31 * hk
    arg = _max(-1.33 * hk, -20.0 + 0 * hk)
    thk = np.tanh(4.0 - hk / 0.875)
    cfo = 0.3 * np.exp(arg) * (grt / 2.3026) ** gex
    return (cfo + 1.1e-4 * (thk - 1.0)) / fc


def us_factor(hs, hk, h, wake):
    us = 0.5 * hs * (1.0 - (hk - 1.0) / (GBCON * h))
    lim = _where(wake, 0.99995, 0.98)
    return _min(us, lim + 0 * us)


def cq_eq(hs, hk, h, us, rt, wake):
    """Равновесное значение sqrt(C_τ)."""
    hkb = hk - 1.0
    gcc = _where(wake, 0.0, GCCON)
    hkc = _max(hk - 1.0 - gcc / rt, 0.01 + 0 * hk)
    return np.sqrt(CTCON * hs * hkb * hkc**2 / ((1.0 - us) * h * hk**2))


def de_thick(hk, th, ds):
    """Толщина слоя δ (для уравнения запаздывания)."""
    de = (3.15 + 1.72 / (hk - 1.0)) * th + ds
    return _min(de, 12.0 * th)


def dit(cf, us, s, rt, hs, wake, hk=None):
    """2 C_D / H* (турбулентный): стенка (с поправкой DFAC при малых Hk, как в
    XFOIL) + внешняя часть + ламинарная добавка."""
    di_w = 0.5 * cf * us * 2.0 / hs
    if hk is not None:
        grt = np.log(rt)
        hmin = 1.0 + 2.1 / grt
        dfac = 0.5 + 0.5 * np.tanh((hk - 1.0) / (hmin - 1.0))
        di_w = di_w * dfac
    di = _where(wake, 0.0 * cf, di_w)
    di = di + s**2 * (0.995 - us) * 2.0 / hs
    di = di + 0.15 * (0.995 - us) ** 2 / rt * 2.0 / hs
    return _where(wake, 2.0 * di, di)


# ---------------------------- переход (eN) --------------------------------
def dampl(hk, th, rt):
    """dñ/dξ по огибающей Drela–Giles (как в XFOIL, DAMPL) с плавным включением."""
    hmi = 1.0 / (hk - 1.0)
    aa = 2.492 * hmi**0.43
    bb = np.tanh(14.0 * hmi - 9.24)
    grcrit = aa + 0.7 * (bb + 1.0)
    gr = np.log10(rt)
    dgr = 0.08
    rnorm = (gr - (grcrit - dgr)) / (2.0 * dgr)
    rnr = _min(_max(rnorm, 0.0 * rnorm), 1.0 + 0 * rnorm)
    rfac = 3.0 * rnr**2 - 2.0 * rnr**3
    arg = 3.87 * hmi - 2.52
    ex = np.exp(-arg**2)
    dadr = 0.028 * (hk - 1.0) - 0.0345 * ex
    af = -0.05 + 2.7 * hmi - 5.5 * hmi**2 + 3.0 * hmi**3
    return af * dadr / th * rfac


# ---------------------------- состояние станции ---------------------------
def station(th, ds, ue, s, re, msq, turb, wake):
    """Все вторичные величины станции. turb, wake — булевы массивы."""
    h = ds / th
    hk = hkin(h, msq)
    hkmin = _where(wake, 1.00005, 1.05)
    hk = _max(hk, hkmin + 0 * hk)
    rt = _max(re * ue * th, 1.0 + 0 * th)
    hs_l = hsl(hk)
    hs_t = hst(hk, rt, msq)
    hs = _where(turb, hs_t, hs_l)
    hc = hct(hk, msq)
    cf_l = cfl(hk, rt)
    cf_t = _max(cft(hk, rt, msq), cf_l)          # как в XFOIL: не меньше ламинарного
    cf = _where(wake, 0.0 * cf_t, _where(turb, cf_t, cf_l))
    us = us_factor(hs, hk, h, wake)
    cq = cq_eq(hs, hk, h, us, rt, wake)
    di_l = _where(wake, dilw(hk, rt, hs), dil(hk, rt))
    di_t = dit(cft(hk, rt, msq), us, s, rt, hs, wake, hk)
    di_t = _max(di_t, _where(wake, dilw(hk, rt, hs), dil(hk, rt)))
    di = _where(turb, di_t, di_l)
    de = de_thick(hk, th, ds)
    ax = dampl(hk, th, rt)
    return dict(h=h, hk=hk, rt=rt, hs=hs, hc=hc, cf=cf, us=us, cq=cq, di=di, de=de, ax=ax,
                th=th, ds=ds, ue=ue, s=s, msq=msq, re=re)


def s_transition(hk, cq):
    """sqrt(C_τ) сразу после перехода (XFOIL: CTR = 1.8 exp(-3.3/(Hk-1)))."""
    return 1.8 * np.exp(-3.3 / (hk - 1.0)) * cq
