"""Массовая сверка вязкого метода с продувками UIUC (набор dataset.json).

Запуск:
  python benchmarks/uiuc_sweep.py --re 1e5 2e5 3e5 --workers 2 --out sweep.jsonl [--xfoil]
            [--ncrit 9] [--only E387,SD7037] [--limit 20]

Для каждого профиля и каждого заданного Re берётся ближайшая поляра
эксперимента (в пределах ±15 %), расчёт — на тех же углах атаки.
Результаты дописываются построчно (JSON lines), повторный запуск пропускает
уже посчитанные случаи.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
warnings.filterwarnings("ignore")


def make_airfoil(coords, n_side=100):
    from aero2d.geometry import Airfoil, normalize, repanel
    c = np.asarray(coords, float)
    return repanel(normalize(Airfoil(c[:, 0], c[:, 1], "uiuc")), n_side)


def run_case(task):
    from aero2d.viscous import ViscousSolver
    np.seterr(all="ignore")
    name, re_exp, rows, coords, ncrit, use_xf, opts = task
    rows = np.asarray(rows)
    t0 = time.time()
    out = dict(name=name, re=re_exp, ncrit=ncrit, alpha=rows[:, 0].tolist(),
               cl_exp=rows[:, 1].tolist(), cd_exp=rows[:, 2].tolist())
    try:
        af = make_airfoil(coords)
        v = ViscousSolver(af, re_exp, ncrit=ncrit)
        for k, val in (opts or {}).items():
            setattr(v, k, val)
        pol = v.polar(rows[:, 0])
        out.update(cl=[p.cl for p in pol], cd=[p.cd for p in pol], cm=[p.cm for p in pol],
                   conv=[bool(p.converged) for p in pol], xtr_u=[p.xtr_u for p in pol],
                   xtr_l=[p.xtr_l for p in pol])
    except Exception as e:                       # noqa: BLE001
        out.update(error=repr(e))
    out["time"] = time.time() - t0
    if use_xf:
        try:
            from benchmarks.xfoil_ref import xfoil_polar
            af = make_airfoil(coords, 80)
            xf = xfoil_polar(name, rows[:, 0], re_exp, coords=(af.x, af.y), ncrit=ncrit)
            out.update(cl_xf=[x["cl"] for x in xf], cd_xf=[x["cd"] for x in xf])
        except Exception as e:                   # noqa: BLE001
            out.update(xf_error=repr(e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, nargs="+", default=[1e5, 2e5, 3e5])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "uiuc_sweep.jsonl"))
    ap.add_argument("--xfoil", action="store_true")
    ap.add_argument("--ncrit", type=float, default=9.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--opts", default="", help="JSON: атрибуты ViscousSolver для эксперимента")
    args = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "data", "uiuc", "dataset.json")))
    only = [s.strip().lower() for s in args.only.split(",") if s.strip()]
    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out):
            try:
                d = json.loads(ln)
                done.add((d["name"], round(d["re"]), d["ncrit"]))
            except Exception:                    # noqa: BLE001
                pass
    tasks, seen = [], set()
    for d in data:
        if only and not any(o in d["name"].lower() for o in only):
            continue
        if d["name"] in seen:                    # одно имя — одна модель
            continue
        seen.add(d["name"])
        for target in args.re:
            best = min(d["polars"], key=lambda r: abs(float(r) / target - 1))
            if abs(float(best) / target - 1) > 0.15 or len(d["polars"][best]) < 4:
                continue
            key = (d["name"], round(float(best)), args.ncrit)
            if key in done:
                continue
            tasks.append((d["name"], float(best), d["polars"][best], d["coords"], args.ncrit, args.xfoil,
                          json.loads(args.opts) if args.opts else None))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} случаев, {sum(len(t[2]) for t in tasks)} точек", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool, open(args.out, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(run_case, tasks), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            nconv = sum(r.get("conv", []))
            print(f"[{i}/{len(tasks)}] {r['name']:<20} Re={r['re']:.0f} сошлось {nconv}/{len(r['alpha'])} "
                  f"{r['time']:.0f}s  (всего {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
