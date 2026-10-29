# -*- mode: python ; coding: utf-8 -*-
# Сборка aero2d в папку dist/aero2d (aero2d.exe + _internal) — PyInstaller.
# Запуск (из корня проекта):  python -m PyInstaller --noconfirm packaging/aero2d.spec
# Режим «одна папка», не «один файл»: JAX и SciPy — сотни МБ, «один файл»
# распаковывался бы во временную папку при каждом запуске и в каждом процессе
# параллельного расчёта (десятки секунд).
import os
import sys
import importlib.util
from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas, binaries, hidden = [], [], []
for pkg in ("jax", "jaxlib", "jax_plugins"):
    if importlib.util.find_spec(pkg) is not None:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
datas += collect_data_files("matplotlib")
datas += [(os.path.join(ROOT, "aero2d", "data"), os.path.join("aero2d", "data"))]

# gmsh: модуль gmsh.py ищет библиотеку рядом с собой и в lib/; pip кладёт её в
# <окружение>/Lib (Windows) или <окружение>/lib (Linux)
spec = importlib.util.find_spec("gmsh")
if spec is not None:
    libname = {"win32": "gmsh-4.15.dll", "darwin": "libgmsh.4.15.dylib"}.get(sys.platform, "libgmsh.so.4.15")
    mod = os.path.dirname(os.path.realpath(spec.origin))
    cands = [os.path.join(d, sub, libname) for d in (mod, os.path.dirname(mod), os.path.dirname(os.path.dirname(mod)))
             for sub in ("", "lib", "Lib", "bin")]
    found = [c for c in cands if os.path.exists(c)]
    if not found:
        raise SystemExit("Не найдена библиотека gmsh (" + libname + ") — проверьте установку пакета gmsh")
    binaries += [(found[0], ".")]
    hidden += ["gmsh"]

hidden += ["aero2d", "aero2d.gui", "aero2d.parallel", "aero2d.mviscous", "aero2d.rans", "aero2d.rans_mesh",
           "aero2d.animate", "aero2d.field", "aero2d.krylov", "aero2d.bilu", "PIL._tkinter_finder"]

a = Analysis([os.path.join(SPECPATH, "aero2d_app.py")], pathex=[ROOT], binaries=binaries, datas=datas,
             hiddenimports=hidden, excludes=["pytest", "IPython", "jupyter", "notebook", "PyQt5", "PySide6"],
             noarchive=False)
pyz = PYZ(a.pure)
if os.environ.get("AERO2D_ONEFILE") == "1":
    # один файл aero2d.exe: при каждом запуске (и в каждом процессе параллельного
    # расчёта) распаковывается во временную папку — старт на 10–30 с дольше
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="aero2d", console=False, icon=None, upx=False,
              runtime_tmpdir=None)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="aero2d", console=False,
              icon=None, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="aero2d", upx=False)
