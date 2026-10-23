"""Сборка набора данных UIUC LSAT (тома 1–3: координаты «как испытано» *.dap
и поляры *.DRG) в один JSON: benchmarks/data/uiuc/dataset.json.

Запуск: python benchmarks/uiuc_dataset.py <папка с распакованными volumeNN>
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def sections(path):
    txt = open(path, encoding="latin-1").read()
    parts = re.split(r"^:{10,}\s*$", txt, flags=re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].strip()] = parts[i + 1]
    return out


def parse_coords(body):
    pts = []
    for ln in body.splitlines():
        t = ln.split()
        if len(t) >= 2:
            try:
                pts.append((float(t[0]), float(t[1])))
            except ValueError:
                pass
    return pts


def parse_polar(body):
    lines = body.splitlines()
    head = {}
    for ln in lines[:4]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            head[k.strip().lower()] = v.strip()
    pol = {}
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Average Reynolds"):
            try:
                rey = float(lines[i + 1])
                n = int(lines[i + 3])
                rows = [list(map(float, lines[i + 5 + k].split()[:3])) for k in range(n)]
                pol[f"{rey:.0f}"] = rows
                i += 5 + n
                continue
            except (ValueError, IndexError):
                pass
        i += 1
    return head, pol


def main(root):
    data = []
    for vol in ("01", "02", "03"):
        d = glob.glob(os.path.join(root, f"v{vol}", f"volume{vol}"))[0]
        co = sections(os.path.join(d, f"COORD{vol}.TXT"))
        dr = sections(os.path.join(d, f"DRAG{vol}.TXT"))
        coords = {k.lower(): parse_coords(v) for k, v in co.items()}
        for fname, body in dr.items():
            head, pol = parse_polar(body)
            stem = fname.lower().rsplit(".", 1)[0]
            comment = head.get("comment", "")
            # только «чистые» модели без триммеров/закрылков
            if not re.search(r"clean", comment, re.I) or re.search(r"trip|flap|tab|grit|zz", comment, re.I):
                continue
            c = coords.get(stem + ".dap")
            if c is None or len(c) < 30 or not pol:
                continue
            # координаты по проекту (гладкие): stem.dat, без буквы модели, *sm.dat
            design = None
            for cand in (stem + "sm.dat", stem[:-1] + "sm.dat", stem + ".dat", stem[:-1] + ".dat",
                         stem.split("-")[0] + ".dat"):
                if cand in coords and len(coords[cand]) >= 30:
                    design = coords[cand]
                    break
            data.append(dict(name=head.get("airfoil", stem), file=fname, volume=int(vol),
                             comment=comment, coords=c, coords_design=design, polars=pol))
    out = os.path.join(HERE, "data", "uiuc", "dataset.json")
    with open(out, "w") as fh:
        json.dump(data, fh)
    nre = sum(len(d["polars"]) for d in data)
    npt = sum(len(r) for d in data for r in d["polars"].values())
    nd = sum(d["coords_design"] is not None for d in data)
    print(f"{len(data)} профилей ({nd} с проектными координатами), {nre} поляр, {npt} точек -> {out}")
    for d in data:
        print(f"  {d['name']:<22} vol{d['volume']}  Re: {', '.join(sorted(d['polars'], key=float))}")


if __name__ == "__main__":
    main(sys.argv[1])
