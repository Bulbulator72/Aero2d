"""Окно программы aero2d: профиль → поляра, графики, таблица, экспорт.

Запуск:  python -m aero2d.gui
"""
from __future__ import annotations

import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402

from . import core  # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


class PlotTab:
    def __init__(self, nb, title):
        self.frame = ttk.Frame(nb)
        nb.add(self.frame, text=title)
        self.fig = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        NavigationToolbar2Tk(self.canvas, self.frame).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self):
        self.canvas.draw_idle()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("aero2d — аэродинамика профилей")
        self.geometry("1280x880")
        self.minsize(1000, 650)
        self.polars: list[core.Polar] = []
        self.airfoil = None
        self.q: queue.Queue = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()
        self._build()
        self.preview_airfoil()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _build(self):
        # строка состояния внизу окна (всегда видна)
        sb = ttk.Frame(self, padding=(8, 3))
        sb.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(sb, mode="determinate", maximum=1.0, length=260)
        self.progress.pack(side="left")
        self.status = tk.StringVar(value="Готово. Выберите профиль и нажмите «Рассчитать».")
        ttk.Label(sb, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True, padx=8)
        ttk.Separator(self, orient="horizontal").pack(side="bottom", fill="x")
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        # --- профиль
        f = ttk.LabelFrame(left, text="Профиль", padding=6)
        f.pack(fill="x", pady=4)
        self.src = tk.StringVar(value="naca")
        ttk.Radiobutton(f, text="NACA (2412, 0012-64, 16-012):", variable=self.src, value="naca",
                        command=self.preview_airfoil).grid(row=0, column=0, sticky="w")
        self.naca = tk.StringVar(value="2412")
        e = ttk.Entry(f, textvariable=self.naca, width=8)
        e.grid(row=0, column=1, sticky="w")
        e.bind("<Return>", lambda _: self.preview_airfoil())
        e.bind("<FocusOut>", lambda _: self.preview_airfoil())
        ttk.Radiobutton(f, text="Файл координат", variable=self.src, value="file",
                        command=self.preview_airfoil).grid(row=1, column=0, sticky="w")
        self.smooth = tk.BooleanVar(value=False)
        ttk.Button(f, text="Открыть…", command=self.open_file).grid(row=1, column=1, sticky="w")
        self.file = tk.StringVar(value="")
        ttk.Radiobutton(f, text="Многоэлементный файл", variable=self.src, value="mfile",
                        command=self.preview_airfoil).grid(row=2, column=0, sticky="w")
        ttk.Button(f, text="Открыть…", command=self.open_mfile).grid(row=2, column=1, sticky="w")
        self.mfile = tk.StringVar(value="")
        self.file_disp = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.file_disp, foreground="#555", wraplength=250).grid(
            row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(f, text="сгладить контур (измеренные координаты)", variable=self.smooth,
                        command=self.preview_airfoil).grid(row=5, column=0, columnspan=2, sticky="w")
        # --- встроенный набор (симметричные профили 6-й серии для оперения и др.)
        lib = core.library()
        self._lib = dict(lib)
        lf = ttk.Frame(f)
        lf.grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(lf, text="Из набора:", variable=self.src, value="lib",
                        command=self.preview_airfoil).pack(side="left")
        self.lib_name = tk.StringVar(value=lib[0][0] if lib else "")
        cl_ = ttk.Combobox(lf, textvariable=self.lib_name, values=[n for n, _ in lib], state="readonly", width=20)
        cl_.pack(side="left")
        cl_.bind("<<ComboboxSelected>>", lambda _: (self.src.set("lib"), self.preview_airfoil()))
        self._control = None
        # --- механизация из одного профиля
        self.hl_cfg = {"flap": None, "slat": None}
        hf = ttk.Frame(f)
        hf.grid(row=6, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Label(hf, text="Механизация:").pack(side="left")
        ttk.Button(hf, text="Настроить…", command=self.open_highlift).pack(side="right")
        self.hl_desc = tk.StringVar(value="нет")
        ttk.Label(f, textvariable=self.hl_desc, foreground="#555", wraplength=250).grid(
            row=7, column=0, columnspan=2, sticky="w")

        # --- режим
        f = ttk.LabelFrame(left, text="Режим", padding=6)
        f.pack(fill="x", pady=4)
        self.mach = tk.StringVar(value="0")
        self.a0, self.a1, self.da = tk.StringVar(value="-4"), tk.StringVar(value="14"), tk.StringVar(value="1")
        ttk.Label(f, text="Число Маха M").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.mach, width=7).grid(row=0, column=1, sticky="w", pady=1)
        ra = ttk.Frame(f)
        ra.grid(row=1, column=0, columnspan=2, sticky="w")
        for lbl, var in (("α, °: от", self.a0), (" до", self.a1), (" шаг", self.da)):
            ttk.Label(ra, text=lbl).pack(side="left")
            ttk.Entry(ra, textvariable=var, width=6).pack(side="left")

        # --- метод
        f = ttk.LabelFrame(left, text="Метод", padding=6)
        f.pack(fill="x", pady=4)
        self.method = tk.StringVar(value="viscous")
        ttk.Radiobutton(f, text="Вязкий: панели + погр. слой (Re, Cd)",
                        variable=self.method, value="viscous").grid(row=0, column=0, columnspan=2, sticky="w")
        self.re = tk.StringVar(value="1e6")
        self.ncrit = tk.StringVar(value="9")
        self.xtru = tk.StringVar(value="1")
        self.xtrl = tk.StringVar(value="1")
        for r, (lbl, var) in enumerate([("   Re (можно несколько: 1e5; 3e5)", self.re), ("   Ñкрит", self.ncrit)],
                                       start=1):
            ttk.Label(f, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(f, textvariable=var, width=8).grid(row=r, column=1, sticky="w")
        rx = ttk.Frame(f)
        rx.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(rx, text="   переход ≤ x/c: верх").pack(side="left")
        ttk.Entry(rx, textvariable=self.xtru, width=5).pack(side="left")
        ttk.Label(rx, text=" низ").pack(side="left")
        ttk.Entry(rx, textvariable=self.xtrl, width=5).pack(side="left")
        ttk.Radiobutton(f, text="Невязкий: панели (мгновенно, без Cd)",
                        variable=self.method, value="panel").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(f, text="Эйлер: сжимаемый, скачки (M ≥ 0,2)",
                        variable=self.method, value="euler").grid(row=6, column=0, columnspan=2, sticky="w")
        re_ = ttk.Frame(f)
        re_.grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Label(re_, text="   сетка").pack(side="left")
        self.grid_name = tk.StringVar(value=list(core.GRIDS)[0])
        ttk.Combobox(re_, textvariable=self.grid_name, values=list(core.GRIDS), state="readonly",
                     width=15).pack(side="left")
        ttk.Label(re_, text=" итераций").pack(side="left")
        self.niter = tk.StringVar(value="100")
        ttk.Entry(re_, textvariable=self.niter, width=5).pack(side="left")
        from .parallel import cpu_count, auto_workers
        rc = ttk.Frame(f)
        rc.grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(rc, text=f"Ядер ЦП (здесь {cpu_count()}, авто = {auto_workers()}):").pack(side="left")
        self.cores = tk.StringVar(value="авто")
        ttk.Combobox(rc, textvariable=self.cores, width=6,
                     values=["авто"] + [str(n) for n in (1, 2, 4, 8, 12, 16, 24, 32) if n <= cpu_count()]
                     ).pack(side="left")
        ttk.Radiobutton(f, text="RANS (Навье–Стокс + модель турбулентности):\n   отрыв, Cl_max механизации (минуты на угол)",
                        variable=self.method, value="rans").grid(row=11, column=0, columnspan=2, sticky="w",
                                                                 pady=(6, 0))
        ttk.Label(f, text="   Итераций на угол, не более:").grid(row=12, column=0, sticky="w")
        self.rans_iter = tk.StringVar(value="400")
        ttk.Entry(f, textvariable=self.rans_iter, width=8).grid(row=12, column=1, sticky="w")
        rm = ttk.Frame(f)
        rm.grid(row=13, column=0, columnspan=2, sticky="w")
        ttk.Label(rm, text="   модель турбулентности").pack(side="left")
        self.turb_model = tk.StringVar(value=list(self.TURB_MODELS)[0])
        ttk.Combobox(rm, textvariable=self.turb_model, values=list(self.TURB_MODELS), state="readonly",
                     width=26).pack(side="left")
        rp = ttk.Frame(f)
        rp.grid(row=14, column=0, columnspan=2, sticky="w")
        ttk.Label(rp, text="   предобусловл. ILU(k), k =").pack(side="left")
        self.prec_fill = tk.StringVar(value="3")
        ttk.Combobox(rp, textvariable=self.prec_fill, values=["0", "1", "2", "3", "4"], state="readonly",
                     width=3).pack(side="left")
        ttk.Label(rp, text=" (больше — быстрее)", foreground="#555").pack(side="left")
        self.coarse_start = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="старт на грубой сетке (если > 20 тыс. ячеек)",
                        variable=self.coarse_start).grid(row=15, column=0, columnspan=2, sticky="w", padx=(14, 0))
        ttk.Label(f, text="   сетка — «Сетка»; сходимость и шаг — «Невязки»", foreground="#555").grid(
            row=16, column=0, columnspan=2, sticky="w")

        # --- кнопки
        f = ttk.Frame(left, padding=(0, 6))
        f.pack(fill="x")
        self.btn_run = ttk.Button(f, text="▶ Рассчитать", command=self.run)
        self.btn_run.pack(fill="x", pady=2)
        self.btn_stop = ttk.Button(f, text="■ Остановить", command=self.stop, state="disabled")
        self.btn_stop.pack(fill="x", pady=2)
        self.btn_skip = ttk.Button(f, text="⏭ Принять угол, следующий (RANS)", command=self.skip_angle,
                                   state="disabled")
        self.btn_skip.pack(fill="x", pady=2)
        mb = tk.Menu(self)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Сохранить таблицу (CSV)…", command=self.save_csv)
        fm.add_command(label="Сохранить графики (PNG)…", command=self.save_png)
        fm.add_command(label="Сохранить ход невязок RANS (CSV)…", command=self.save_res_csv)
        fm.add_separator()
        fm.add_command(label="Очистить графики и таблицу", command=self.clear)
        fm.add_separator()
        fm.add_command(label="Выход", command=self.destroy)
        mb.add_cascade(label="Файл", menu=fm)
        self.config(menu=mb)


        # --- вкладки
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.t_geo = PlotTab(nb, "Профиль")
        self._build_mesh_tab(nb)
        self.t_cl = PlotTab(nb, "Cl(α)")
        self.t_cd = PlotTab(nb, "Cd")
        self.t_cm = PlotTab(nb, "Cm, Ch")
        self.t_cp = PlotTab(nb, "Cp")
        bar = ttk.Frame(self.t_cp.frame)
        bar.pack(side="top", fill="x", before=self.t_cp.canvas.get_tk_widget())
        ttk.Label(bar, text="  α, °: ").pack(side="left")
        self.cp_alpha = tk.StringVar()
        self.cb_alpha = ttk.Combobox(bar, textvariable=self.cp_alpha, width=8, state="readonly")
        self.cb_alpha.pack(side="left")
        self.cb_alpha.bind("<<ComboboxSelected>>", lambda _: self.plot_cp())
        self.t_bl = PlotTab(nb, "Погр. слой")
        bar = ttk.Frame(self.t_bl.frame)
        bar.pack(side="top", fill="x", before=self.t_bl.canvas.get_tk_widget())
        ttk.Label(bar, text="  α, °: ").pack(side="left")
        self.cb_alpha_bl = ttk.Combobox(bar, textvariable=self.cp_alpha, width=8, state="readonly")
        self.cb_alpha_bl.pack(side="left")
        self.cb_alpha_bl.bind("<<ComboboxSelected>>", lambda _: (self.plot_cp(), self.plot_bl()))
        self.cb_alpha.bind("<<ComboboxSelected>>", lambda _: (self.plot_cp(), self.plot_bl()))
        # поле течения
        self.t_fl = PlotTab(nb, "Поле течения")
        bar = ttk.Frame(self.t_fl.frame)
        bar.pack(side="top", fill="x", before=self.t_fl.canvas.get_tk_widget())
        ttk.Label(bar, text="  расчёт: ").pack(side="left")
        self.fl_polar = tk.StringVar()
        self.cb_fl_polar = ttk.Combobox(bar, textvariable=self.fl_polar, width=34, state="readonly")
        self.cb_fl_polar.pack(side="left")
        ttk.Label(bar, text="  α, °: ").pack(side="left")
        self.cb_alpha_fl = ttk.Combobox(bar, textvariable=self.cp_alpha, width=8, state="readonly")
        self.cb_alpha_fl.pack(side="left")
        bar = ttk.Frame(self.t_fl.frame)
        bar.pack(side="top", fill="x", before=self.t_fl.canvas.get_tk_widget())
        ttk.Label(bar, text="  величина: ").pack(side="left")
        self.fl_qty = tk.StringVar(value="Давление Cp")
        cq = ttk.Combobox(bar, textvariable=self.fl_qty,
                          values=["Давление Cp", "Скорость |V|/V∞", "Число Маха (RANS)",
                                  "Турбулентная вязкость ν̃/ν∞ (RANS)", "Невязка по ячейкам (RANS)",
                                  "Нет (только линии тока)"],
                          state="readonly", width=24)
        cq.pack(side="left")
        cq.bind("<<ComboboxSelected>>", lambda _: self._redraw_field())
        ttk.Label(bar, text="   область: ").pack(side="left")
        from .field import REGIONS
        self.fl_reg = tk.StringVar(value=list(REGIONS)[0])
        ttk.Combobox(bar, textvariable=self.fl_reg, values=list(REGIONS), state="readonly",
                     width=20).pack(side="left")
        self.fl_lines = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="линии тока", variable=self.fl_lines,
                        command=self._redraw_field).pack(side="left", padx=6)
        ttk.Button(bar, text="Построить", command=self.plot_field).pack(side="left", padx=6)
        self._field_cache = {}
        self._build_res_tab(nb)
        # таблица
        tf = ttk.Frame(nb)
        nb.add(tf, text="Таблица")
        cols = ("calc", "alpha", "cl", "cd", "cdf", "cm", "ch", "xu", "xl", "ok")
        self.table = ttk.Treeview(tf, columns=cols, show="headings")
        for c, t, w in zip(cols, ["Расчёт", "α, °", "Cl", "Cd", "Cd трения", "Cm", "Ch руля", "Переход верх",
                                  "Переход низ", "Надёжно"], [300, 50, 75, 80, 80, 75, 75, 85, 85, 65]):
            self.table.heading(c, text=t)
            self.table.column(c, width=w, anchor="w" if c == "calc" else "e")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # примечания
        nf = ttk.Frame(nb)
        nb.add(nf, text="Примечания")
        self.notes = tk.Text(nf, wrap="word", height=10)
        self.notes.pack(fill="both", expand=True)
        self.nb = nb

    # ------------------------------------------------------------- профиль
    def open_file(self):
        p = filedialog.askopenfilename(title="Файл координат профиля",
                                       filetypes=[("Координаты профиля", "*.dat *.txt *.csv"), ("Все файлы", "*.*")])
        if p:
            self.file.set(p)
            self.src.set("file")
            self.preview_airfoil()

    def open_mfile(self):
        p = filedialog.askopenfilename(title="Файл многоэлементного профиля (элементы разделены "
                                       "пустой строкой или строкой 999 999)",
                                       filetypes=[("Координаты", "*.dat *.txt *.csv"), ("Все файлы", "*.*")])
        if p:
            self.mfile.set(p)
            self.src.set("mfile")
            self.preview_airfoil()

    def open_highlift(self):
        """Окно настройки механизации: тип закрылка, дефлектор, предкрылок."""
        from .multi import FLAP_TYPES
        if getattr(self, "_hl_win", None) and self._hl_win.winfo_exists():
            self._hl_win.lift()
            return
        w = tk.Toplevel(self)
        self._hl_win = w
        w.title("Механизация профиля")
        w.resizable(False, False)
        presets = {
            "однощелевой": dict(chord=0.25, deflection=30, gap=0.015, overlap=0.02),
            "Фаулера": dict(chord=0.30, deflection=30, gap=0.02, overlap=0.0),
            "двухщелевой": dict(chord=0.35, deflection=45, gap=0.012, overlap=0.0,
                                vane_chord=0.12, vane_deflection=20, vane_gap=0.015, vane_overlap=0.0),
            "руль/элерон (поворотный)": dict(chord=0.30, deflection=10, balance=0.20, gap=0.0),
        }
        fl = self.hl_cfg["flap"] or {}
        sl = self.hl_cfg["slat"] or {}
        inv = {v: k for k, v in FLAP_TYPES.items()}
        ftype = tk.StringVar(value=inv.get(fl.get("type"), "нет"))
        V = {}

        def var(key, default):
            V[key] = tk.StringVar(value=str(default))
            return V[key]

        frm = ttk.LabelFrame(w, text="Закрылок", padding=6)
        frm.grid(row=0, column=0, sticky="nwe", padx=6, pady=4)
        ttk.Label(frm, text="тип").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(frm, textvariable=ftype, values=list(FLAP_TYPES), state="readonly", width=14)
        cb.grid(row=0, column=1, sticky="w")
        rows = [("хорда закрылка/руля (от оси), доли хорды", "chord", 0.30),
                ("угол отклонения δ, °", "deflection", 30),
                ("зазор, доли хорды (руль: щель у носка, 0 — герметична)", "gap", 0.015),
                ("перекрытие, доли хорды", "overlap", 0.0),
                ("осевая компенсация руля, доли хорды руля", "balance", 0.0)]
        for r, (lbl, k, d) in enumerate(rows, start=1):
            ttk.Label(frm, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(frm, textvariable=var("f_" + k, fl.get(k, d)), width=8).grid(row=r, column=1, sticky="w")
        frv = ttk.LabelFrame(w, text="Дефлектор (передний элемент двухщелевого закрылка)", padding=6)
        frv.grid(row=1, column=0, sticky="nwe", padx=6, pady=4)
        rows = [("хорда дефлектора, доли хорды", "vane_chord", 0.12), ("угол δ₁, °", "vane_deflection", 20),
                ("зазор до основной части", "vane_gap", 0.015), ("перекрытие", "vane_overlap", 0.0)]
        for r, (lbl, k, d) in enumerate(rows):
            ttk.Label(frv, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(frv, textvariable=var("f_" + k, fl.get(k, d)), width=8).grid(row=r, column=1, sticky="w")
        frs = ttk.LabelFrame(w, text="Предкрылок", padding=6)
        frs.grid(row=2, column=0, sticky="nwe", padx=6, pady=4)
        slat_on = tk.BooleanVar(value=bool(sl))
        ttk.Checkbutton(frs, text="есть", variable=slat_on).grid(row=0, column=0, sticky="w")
        rows = [("хорда, доли хорды", "chord", 0.15), ("угол (носком вниз), °", "deflection", 25),
                ("зазор до основной части", "gap", 0.02), ("перекрытие", "overlap", 0.01)]
        for r, (lbl, k, d) in enumerate(rows, start=1):
            ttk.Label(frs, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(frs, textvariable=var("s_" + k, sl.get(k, d)), width=8).grid(row=r, column=1, sticky="w")
        ttk.Label(w, wraplength=380, foreground="#555", justify="left", text=(
            "Зазор — наименьшее расстояние от задней кромки впереди стоящего элемента до "
            "следующего элемента. Перекрытие — на сколько носок закрылка (дефлектора) "
            "выдвинут вперёд за эту кромку; у предкрылка — на сколько его кромка позади "
            "носка основной части. Однощелевой: короткая ниша, перекрытие ~0,01–0,03. "
            "Фаулера: закрылок убран в глубокую нишу и выдвинут назад (перекрытие ~0 — "
            "наибольшее увеличение хорды). Коэффициенты отнесены к хорде исходного профиля.\n"
            "Руль/элерон: задняя часть поворачивается вокруг оси (одним контуром), считается "
            "вязким или невязким методом; перекрытие не используется. Зазор — щель у носка "
            "руля: 0 — герметична; 0,001–0,01 — открытая щель, учитывается эмпирической "
            "поправкой по продувкам NACA (меньше наклон cl(α) и эффективность руля). "
            "Осевая компенсация — длина носка руля впереди оси в долях хорды руля "
            "(обычно 0,15–0,30); влияет на шарнирный момент Ch.")
                  ).grid(row=3, column=0, padx=6, pady=4, sticky="w")

        def on_type(_=None):
            t = ftype.get()
            if t in presets:
                for k, v in presets[t].items():
                    V["f_" + k].set(str(v))
        cb.bind("<<ComboboxSelected>>", on_type)

        def apply(close=False):
            num = lambda k: float(V[k].get().replace(",", "."))
            try:
                t = FLAP_TYPES[ftype.get()]
                flap = None
                if t == "control":
                    flap = dict(type=t, chord=num("f_chord"), deflection=num("f_deflection"),
                                balance=num("f_balance"), gap=num("f_gap"))
                    if not 0.0 <= flap["balance"] <= 0.6:
                        raise ValueError("осевая компенсация — от 0 до 0,6 хорды руля")
                    if not 0.0 <= flap["gap"] <= 0.02:
                        raise ValueError("щель у носка руля — от 0 (герметична) до 0,02 хорды")
                elif t:
                    flap = dict(type=t, **{k: num("f_" + k) for k in ("chord", "deflection", "gap", "overlap")})
                    if t == "double":
                        flap.update({k: num("f_" + k) for k in
                                     ("vane_chord", "vane_deflection", "vane_gap", "vane_overlap")})
                        if flap["vane_chord"] >= flap["chord"] - 0.05:
                            raise ValueError("хорда дефлектора должна быть меньше хорды закрылка")
                    if not 0.05 <= flap["chord"] <= 0.5:
                        raise ValueError("хорда закрылка — от 0,05 до 0,5")
                slat = None
                if slat_on.get() and t == "control":
                    raise ValueError("руль считается одним контуром — предкрылок с ним пока не поддерживается")
                if slat_on.get():
                    slat = {k: num("s_" + k) for k in ("chord", "deflection", "gap", "overlap")}
                    if not 0.05 <= slat["chord"] <= 0.3:
                        raise ValueError("хорда предкрылка — от 0,05 до 0,3")
            except (ValueError, KeyError) as e:
                messagebox.showerror("Механизация", str(e), parent=w)
                return
            self.hl_cfg = {"flap": flap, "slat": slat}
            parts = []
            if slat:
                parts.append(f"предкрылок {slat['deflection']:g}°")
            if flap:
                nm = ftype.get()
                d = f"{flap['vane_deflection']:g}/{flap['deflection']:g}" if t == "double" else f"{flap['deflection']:g}"
                if t == "control":
                    parts.append(f"руль {flap['chord']:g}c, δ {d}°, компенсация {100 * flap['balance']:g} %"
                                 + (f", щель {flap['gap']:g}c" if flap.get("gap") else ""))
                else:
                    parts.append(f"закрылок {nm} {d}°")
            self.hl_desc.set(", ".join(parts) if parts else "нет")
            if self.src.get() == "mfile":
                self.src.set("naca")
            self.preview_airfoil()
            self.nb.select(0)
            if close:
                w.destroy()

        bf = ttk.Frame(w, padding=6)
        bf.grid(row=4, column=0, sticky="we")
        ttk.Button(bf, text="Показать", command=apply).pack(side="left")
        ttk.Button(bf, text="Готово", command=lambda: apply(True)).pack(side="left", padx=6)

        def reset():
            ftype.set("нет")
            slat_on.set(False)
            apply()
        ttk.Button(bf, text="Без механизации", command=reset).pack(side="right")

    def get_config(self):
        """Один профиль (Airfoil) или список элементов механизации."""
        num = lambda v: float(v.get().replace(",", "."))
        if self.src.get() == "mfile":
            if not self.mfile.get():
                raise ValueError("Выберите файл многоэлементного профиля")
            from .multi import load_multi
            from .geometry import repanel
            els = load_multi(self.mfile.get())
            return [repanel(e, 80, keep_frame=True) for e in els]
        af = self.get_airfoil()
        self._control = None
        fl = self.hl_cfg["flap"]
        if fl and fl.get("type") == "control":
            from .controls import deflect, hinge_point
            xh = 1.0 - fl["chord"]
            self._control = dict(x_hinge=xh, deflection=fl["deflection"], balance=fl.get("balance", 0.0),
                                 hinge=hinge_point(af, xh), gap=fl.get("gap", 0.0), base=af)
            out = deflect(af, xh, fl["deflection"])
            out.name = af.name + f", руль {fl['deflection']:g}°"
            return out
        if fl or self.hl_cfg["slat"]:
            from .multi import highlift
            return highlift(af, flap=fl, slat=self.hl_cfg["slat"])
        return af

    def get_airfoil(self):
        if self.src.get() == "lib":
            return core.make_airfoil("lib", self._lib[self.lib_name.get()])
        if self.src.get() == "file":
            if not self.file.get():
                raise ValueError("Выберите файл координат")
            return core.make_airfoil("file", self.file.get(), smooth=self.smooth.get())
        return core.make_airfoil("naca", self.naca.get())

    def preview_airfoil(self):
        src = self.src.get()
        self.file_disp.set(os.path.basename(self.file.get()) if src == "file" else
                           os.path.basename(self.mfile.get()) if src == "mfile" else "")
        try:
            af = self.get_config()
        except Exception as e:  # noqa: BLE001
            self.status.set(f"Профиль: {e}")
            return
        self.airfoil = af
        ax = self.t_geo.ax
        ax.clear()
        if isinstance(af, list):
            for i, e in enumerate(af):
                ax.plot(e.x, e.y, "-", color=COLORS[i % len(COLORS)], lw=1.5, label=e.name)
                ax.plot(e.x, e.y, ".", color=COLORS[i % len(COLORS)], ms=2)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_title(f"Многоэлементный профиль: {len(af)} элемента(ов). Расчёт — невязкий")
            ax.set_xlabel("x/c")
            ax.set_ylabel("y/c")
            self.t_geo.draw()
            return
        ax.plot(af.x, af.y, "-", color="#1f77b4", lw=1.5)
        ax.plot(af.x, af.y, ".", color="#1f77b4", ms=2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if self._control:
            h = self._control["hinge"]
            ax.plot([h[0]], [h[1]], "o", color="#d62728", ms=5, label="ось руля")
            b = self._control["balance"] * (1 - self._control["x_hinge"])
            if b > 0:
                ax.plot([h[0] - b, h[0]], [h[1], h[1]], "-", color="#d62728", lw=2.5,
                        label="осевая компенсация (носок руля в нише)")
            ax.legend(fontsize=8)
        tmax = self._thickness(af)
        ax.set_title(f"{af.name}: {len(af.x) - 1} панелей, наибольшая толщина {100 * tmax:.1f} %")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        self.t_geo.draw()

    @staticmethod
    def _thickness(af):
        i = int(np.argmin(af.x))
        xu, yu = af.x[:i + 1][::-1], af.y[:i + 1][::-1]
        xl, yl = af.x[i:], af.y[i:]
        xs = np.linspace(0.001, 0.999, 400)
        return float(np.max(np.interp(xs, xu, yu) - np.interp(xs, xl, yl)))

    # --------------------------------------------------------------- расчёт
    def _params(self):
        num = lambda v: float(v.get().replace(",", ".").replace(" ", ""))
        try:
            mach = num(self.mach)
            alphas = core.alpha_range(num(self.a0), num(self.a1), num(self.da))
            niter = int(self.niter.get())
            txt = self.re.get().strip().replace(", ", ";").replace(" ", ";")
            re_ = [float(t.replace(",", ".")) for t in txt.split(";") if t.strip()]
            if not re_:
                raise ValueError("не задано Re")
            from .parallel import resolve_workers
            self._workers = resolve_workers(self.cores.get())
            ncrit = num(self.ncrit)
            xtr = (num(self.xtru), num(self.xtrl))
        except ValueError as e:
            raise ValueError(f"Проверьте числа в полях: {e}") from None
        if not 0 <= mach < 3.5:
            raise ValueError("M должно быть в диапазоне 0…3")
        return mach, alphas, niter, re_, ncrit, xtr

    def run(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            af = self.get_config()
            mach, alphas, niter, re_, ncrit, xtr = self._params()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Ошибка", str(e))
            return
        self.airfoil = af
        method, grid = self.method.get(), self.grid_name.get()
        if method == "rans":
            try:
                rit = int(self.rans_iter.get())
                conv = self.conv_settings()
                sopts = self.solver_settings()
                els_, re0, a_mid, mopts, _, sig = self._mesh_inputs()
            except ValueError as e:
                messagebox.showerror("Ошибка", str(e) if str(e) else "Итераций на угол — целое число")
                return
            mesh0 = self._mesh[1] if self._mesh and self._mesh[0] == sig else None
            from .rans import RansControl
            self._live = self._new_res()
            self.res_src.set("текущий расчёт")
            self._rans_ctl = RansControl(monitor=lambda d: self.q.put(("iter", d)))
        if isinstance(af, list) and method != "rans":
            if method == "euler":
                messagebox.showinfo("Механизация", "Для многоэлементного профиля доступны вязкий, RANS и "
                                    "невязкий панельный расчёты; выполняется вязкий.")
            method = "multi" if method == "panel" else "multi_viscous"
        self.stop_flag.clear()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal" if method not in ("panel", "multi") else "disabled")
        self.btn_skip.configure(state="normal" if method == "rans" else "disabled")
        self.progress["value"] = 0
        self.status.set("Расчёт…")

        control = self._control

        def job():
            try:
                def cb(text, frac):
                    self.q.put(("progress", text, frac))
                    return self.stop_flag.is_set()
                nw = self._workers
                if method in ("multi", "multi_viscous"):
                    nm = self.hl_desc.get() if self.src.get() != "mfile" else os.path.basename(self.mfile.get())
                    ch = 1.0 if self.src.get() != "mfile" else None
                    if method == "multi":
                        ps = [core.run_multi(af, alphas, mach, nm, chord=ch)]
                    else:
                        ps = core.run_multi_viscous_batch(af, alphas, mach, re_, ncrit, xtr, cb, nm,
                                                          chord=ch, workers=nw)
                elif method == "rans":
                    nm = None
                    if isinstance(af, list):
                        nm = self.hl_desc.get() if self.src.get() != "mfile" else os.path.basename(self.mfile.get())
                    ps = []
                    for r_ in re_:
                        if self.stop_flag.is_set():
                            break
                        ctl = self._rans_ctl
                        ctl.stop = False
                        p_ = core.run_rans(af, alphas, mach, r_, cb, rit, nm,
                                           chord=(1.0 if isinstance(af, list) and self.src.get() != "mfile" else None),
                                           mesh=(mesh0 if r_ == re0 else None), mesh_opts=mopts, conv=conv,
                                           control=ctl, workers=nw, solver_opts=sopts,
                                           coarse_start=("auto" if self.coarse_start.get() else False))
                        ps.append(p_)
                        if p_.field_ctx and r_ == re0 and mesh0 is None:
                            self.q.put(("mesh", sig, p_.field_ctx["mesh"], r_, p_.field_ctx["chord"]))
                elif method == "panel":
                    ps = [core.run_panel(af, alphas, mach, control=control)]
                elif method == "viscous":
                    ps = core.run_viscous_batch(af, alphas, mach, re_, ncrit, xtr, cb, control=control,
                                                workers=nw)
                else:
                    p = core.run_euler(af, alphas, mach, grid, niter, cb)
                    if self.stop_flag.is_set():
                        p.label += " (остановлен)"
                    ps = [p]
                for k, p in enumerate(ps):
                    self.q.put(("done" if k == len(ps) - 1 else "partial", p))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", f"{e}\n\n{traceback.format_exc(limit=3)}"))

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag.set()
        ctl = getattr(self, "_rans_ctl", None)
        if ctl is not None:
            ctl.stop = True
        self.status.set("Остановка после текущей итерации…")

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    self.status.set(msg[1])
                    self.progress["value"] = msg[2]
                elif msg[0] == "partial":
                    self.add_polar(msg[1])
                elif msg[0] == "done":
                    self._finish()
                    self.add_polar(msg[1])
                elif msg[0] == "field":
                    self._draw_field(*msg[1:])
                elif msg[0] == "field_error":
                    self.status.set("Поле течения: " + msg[1])
                elif msg[0] == "iter":
                    self._on_iter(msg[1])
                elif msg[0] == "mesh":
                    self._set_mesh(*msg[1:])
                elif msg[0] == "mesh_error":
                    self.mesh_info.set("Ошибка построения сетки: " + msg[1])
                    messagebox.showerror("Сетка", msg[1])
                elif msg[0] == "error":
                    self._finish()
                    self.status.set("Ошибка расчёта")
                    messagebox.showerror("Ошибка расчёта", msg[1])
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def _finish(self):
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_skip.configure(state="disabled")
        if self._live is not None and self.res_src.get() == "текущий расчёт":
            self.plot_res()
        self.progress["value"] = 1.0

    # ------------------------------------------------------------ графики
    def add_polar(self, p):
        self.polars.append(p)
        n_ok = int(p.ok.sum())
        self.status.set(f"Готово: {p.label}. Надёжных точек {n_ok} из {len(p.alpha)}.")
        self.notes.insert("end", f"■ {p.label}\n" + "".join(f"   • {t}\n" for t in p.notes) + "\n")
        fmt = lambda arr, k, f: "—" if arr is None or not np.isfinite(arr[k]) else format(arr[k], f)
        for k in range(len(p.alpha)):
            self.table.insert("", "end", values=(
                p.label, f"{p.alpha[k]:g}", f"{p.cl[k]:.4f}", fmt(p.cd, k, ".5f"), fmt(p.cdf, k, ".5f"),
                f"{p.cm[k]:.4f}", fmt(p.ch, k, ".4f"), fmt(p.xtr_u, k, ".3f"), fmt(p.xtr_l, k, ".3f"),
                "да" if p.ok[k] else "НЕТ"))
        self.cb_res_src["values"] = ["текущий расчёт"] + [q.label for q in self.polars if q.method == "rans"]
        fl = [q.label for q in self.polars if q.field_ctx is not None]
        self.cb_fl_polar["values"] = fl
        if p.field_ctx is not None:
            self.fl_polar.set(p.label)
        alphas = sorted({f"{a:g}" for q in self.polars for a in q.alpha}, key=float)
        self.cb_alpha["values"] = alphas
        self.cb_alpha_bl["values"] = alphas
        self.cb_alpha_fl["values"] = alphas
        if not self.cp_alpha.get() or self.cp_alpha.get() not in alphas:
            mid = p.alpha[np.argmin(np.abs(p.alpha - 4))]
            self.cp_alpha.set(f"{mid:g}")
        self.replot()
        self.nb.select(self.t_cl.frame)

    def _style(self, ax, xl, yl):
        ax.grid(alpha=0.3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if self.polars:
            ax.legend(fontsize=8)

    def _series(self, ax, x, y, ok, color, label):
        ax.plot(x, y, "-", color=color, lw=1.4, label=label)
        ax.plot(np.asarray(x)[ok], np.asarray(y)[ok], "o", color=color, ms=3.5)
        if (~ok).any():
            ax.plot(np.asarray(x)[~ok], np.asarray(y)[~ok], "x", color=color, ms=7, mew=1.5)

    def replot(self):
        for t in (self.t_cl, self.t_cd, self.t_cm):
            t.fig.clear()
        axl = self.t_cl.fig.add_subplot(111)
        any_ch = any(p.ch is not None for p in self.polars)
        axm = self.t_cm.fig.add_subplot(121 if any_ch else 111)
        axh = self.t_cm.fig.add_subplot(122) if any_ch else None
        axd1 = self.t_cd.fig.add_subplot(121)
        axd2 = self.t_cd.fig.add_subplot(122)
        any_cd = False
        for i, p in enumerate(self.polars):
            c = COLORS[i % len(COLORS)]
            self._series(axl, p.alpha, p.cl, p.ok, c, p.label)
            self._series(axm, p.alpha, p.cm, p.ok, c, p.label)
            if axh is not None and p.ch is not None:
                self._series(axh, p.alpha, p.ch, p.ok, c, p.label)
            if not np.all(np.isnan(p.cd)):
                any_cd = True
                self._series(axd1, p.alpha, p.cd, p.ok, c, p.label)
                self._series(axd2, p.cd, p.cl, p.ok, c, p.label)
        self._style(axl, "α, °", "Cl")
        self._style(axm, "α, °", "Cm (относительно 1/4 хорды)")
        if axh is not None:
            self._style(axh, "α, °", "Ch руля (> 0 — задней кромкой вниз)")
        if any_cd:
            self._style(axd1, "α, °", "Cd")
            self._style(axd2, "Cd", "Cl")
        else:
            axd1.text(0.5, 0.5, "Сопротивление даёт вязкий метод\n(и метод Эйлера — только волновое)",
                      ha="center", va="center", transform=axd1.transAxes, color="#777")
            axd1.set_axis_off(); axd2.set_axis_off()
        axl.set_title("× — расчёт не сошёлся или вне границ применимости метода", fontsize=8, color="#777")
        for t in (self.t_cl, self.t_cd, self.t_cm):
            t.draw()
        self.plot_cp()
        self.plot_bl()

    def plot_cp(self):
        ax = self.t_cp.ax
        ax.clear()
        try:
            a = float(self.cp_alpha.get())
        except ValueError:
            self.t_cp.draw()
            return
        for i, p in enumerate(self.polars):
            k = np.where(np.isclose(p.alpha, a))[0]
            if len(k) == 0:
                continue
            k = k[0]
            ax.plot(p.cp_x[k], p.cp[k], "-", color=COLORS[i % len(COLORS)], lw=1.3,
                    label=p.label + ("" if p.ok[k] else " (НЕ сошёлся)"))
            if p.cp_inv:
                ax.plot(p.cp_x[k], p.cp_inv[k], ":", color=COLORS[i % len(COLORS)], lw=1.0,
                        label="  то же без вязкости")
        if self.polars and self.polars[-1].mach > 0:
            from .panel import critical_cp
            cs = critical_cp(self.polars[-1].mach)
            ax.axhline(cs, color="#999", ls="--", lw=0.8)
            ax.text(1.0, cs, " Cp*", va="center", color="#999", fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"α = {a:g}°")
        self._style(ax, "x/c", "Cp")
        self.t_cp.draw()

    def plot_bl(self):
        fig = self.t_bl.fig
        fig.clear()
        try:
            a = float(self.cp_alpha.get())
        except ValueError:
            self.t_bl.draw()
            return
        axs = fig.subplots(2, 2, sharex=True)
        labels = [("cf_inf", "τ стенки / q∞ (трение)"), ("H", "H = δ*/θ"), ("dstar", "δ*/c"),
                  ("n", "Ñ (рост возмущений)")]
        found = False
        for i, p in enumerate(self.polars):
            k = np.where(np.isclose(p.alpha, a))[0]
            if not p.bl or len(k) == 0:
                continue
            found = True
            side = p.bl[k[0]]
            c = COLORS[i % len(COLORS)]
            for ax, (key, title) in zip(axs.ravel(), labels):
                for nm, ls in (("upper", "-"), ("lower", "--")):
                    d = side[nm]
                    ax.plot(d["x"], np.real(d[key]), ls, color=c, lw=1.2,
                            label=(p.label + (", верх" if nm == "upper" else ", низ")) if key == "cf_inf" else None)
                if key in ("H", "dstar"):
                    d = side["wake"]
                    ax.plot(d["x"], np.real(d[key]), ":", color=c, lw=1.0)
                ax.set_title(title, fontsize=9)
                ax.grid(alpha=0.3)
            if p.xtr_u is not None:
                for ax in axs.ravel():
                    ax.axvline(p.xtr_u[k[0]], color=c, lw=0.6, ls="-", alpha=0.5)
                    ax.axvline(p.xtr_l[k[0]], color=c, lw=0.6, ls="--", alpha=0.5)
        if not found:
            axs[0, 0].text(0.5, 0.5, "Распределения пограничного слоя\nдаёт вязкий метод",
                           ha="center", va="center", transform=axs[0, 0].transAxes, color="#777")
        else:
            axs[0, 0].legend(fontsize=7)
            axs[0, 0].axhline(0, color="#999", lw=0.6)
            axs[0, 1].set_ylim(1, min(8, axs[0, 1].get_ylim()[1]))
            axs[1, 1].axhline(9, color="#999", lw=0.6)
        for ax in axs[1]:
            ax.set_xlabel("x/c (вертикали — точки перехода; пунктир — низ; точки — след)")
        fig.suptitle(f"α = {a:g}°", fontsize=10)
        self.t_bl.draw()

    def plot_field(self):
        """Поле течения для последнего расчёта, в котором есть выбранный α."""
        try:
            a = float(self.cp_alpha.get())
        except ValueError:
            messagebox.showinfo("Поле течения", "Сначала выполните расчёт и выберите α.")
            return
        cand = [p for p in self.polars if p.field_ctx is not None and np.isclose(p.alpha, a).any()]
        sel = [p for p in cand if p.label == self.fl_polar.get()]
        if not cand:
            messagebox.showinfo("Поле течения", "Для выбранного α нет расчёта с полем течения (поле строится "
                                "для вязкого, панельного и RANS-расчёта; для Эйлера — пока нет).")
            return
        if self.fl_polar.get() and not sel:
            messagebox.showinfo("Поле течения", f"В расчёте «{self.fl_polar.get()}» нет α = {a:g}°.")
            return
        p = sel[-1] if sel else cand[-1]
        k = int(np.where(np.isclose(p.alpha, a))[0][0])
        from .field import REGIONS, compute_field
        box = REGIONS[self.fl_reg.get()]
        key = (id(p), k, self.fl_reg.get())
        if key in self._field_cache:
            self._draw_field(p.label, self._field_cache[key], key)
            return
        self.status.set(f"Поле течения: расчёт для α = {a:g}°…")

        def job():
            try:
                f = compute_field(p.field_ctx, k, box)
                self.q.put(("field", p.label, f, key))
            except Exception as e:  # noqa: BLE001
                self.q.put(("field_error", str(e)))
        threading.Thread(target=job, daemon=True).start()

    def _redraw_field(self):
        """Перерисовать уже рассчитанное поле (смена величины, линий тока)."""
        last = getattr(self, "_field_last", None)
        if last:
            self._draw_field(*last)

    def _draw_field(self, label, f, key):
        self._field_cache[key] = f
        self._field_last = (label, f, key)
        fig = self.t_fl.fig
        fig.clear()
        ax = fig.add_subplot(111)
        X, Y = f["X"], f["Y"]
        qty = self.fl_qty.get()
        if qty.startswith("Давление"):
            z, cmap, name = f["cp"], "RdBu_r", "Cp"
            lo = max(np.nanpercentile(z, 1), -6.0)
            levels = np.linspace(lo, 1.0, 41)
        elif qty.startswith("Скорость"):
            z, cmap, name = f["speed"], "viridis", "|V| / V∞"
            levels = np.linspace(np.nanpercentile(z, 0.5), min(np.nanpercentile(z, 99.5), 3.0), 41)
        elif qty.startswith("Невязка"):
            if "res_cell" not in f:
                self.status.set("Невязка по ячейкам есть только в расчёте RANS.")
                z, name = None, "линии тока"
            else:
                z = np.log10(np.maximum(f["res_cell"], 1e-30))
                cmap, name = "inferno", "lg невязки в ячейке (последняя итерация)"
                hi = np.nanmax(z)
                levels = np.linspace(hi - 6, hi + 1e-9, 41)
        elif qty.startswith("Число Маха") or qty.startswith("Турбулентная"):
            key = "mach_local" if qty.startswith("Число") else "nut"
            if key not in f:
                self.status.set("Эта величина есть только в расчёте RANS.")
                z, name = None, "линии тока"
            else:
                z = f[key]
                cmap, name = ("viridis", "M") if key == "mach_local" else ("magma", "ν̃/ν∞ (переменная SA)")
                levels = np.linspace(np.nanmin(z), np.nanpercentile(z, 99.7) + 1e-9, 41)
        else:
            z, name = None, "линии тока"
        if z is not None:
            c = ax.contourf(X, Y, z, levels=levels, cmap=cmap, extend="both")
            fig.colorbar(c, ax=ax, label=name, shrink=0.85)
        if self.fl_lines.get() or z is None:
            ys = np.linspace(Y.min(), Y.max(), 34)
            seeds = np.column_stack([np.full_like(ys, X.min() + 1e-3 * (X.max() - X.min())), ys])
            U = np.nan_to_num(f["U"]); V = np.nan_to_num(f["V"])
            if z is None:      # только линии тока — цвет по скорости
                sp = ax.streamplot(X, Y, U, V, start_points=seeds, density=4, color=f["speed"],
                                   cmap="viridis", linewidth=1.0, arrowsize=0.6, broken_streamlines=False)
                fig.colorbar(sp.lines, ax=ax, label="|V| / V∞", shrink=0.85)
            else:
                ax.streamplot(X, Y, U, V, start_points=seeds, density=4, color="k", linewidth=0.55,
                              arrowsize=0.5, broken_streamlines=False)
        for bx, by in f.get("bodies", [f["body"]]):
            ax.fill(bx, by, color="#444", zorder=3)
        if "worst_xy" in f and len(f["worst_xy"]):
            wx_, wy_ = f["worst_xy"][:, 0], f["worst_xy"][:, 1]
            ax.plot(wx_, wy_, "o", mfc="none", mec="#00e5ff", ms=11, mew=1.6, zorder=5,
                    label="ячейки с наибольшей невязкой")
            ax.legend(loc="upper right", fontsize=7)
        if "disp" in f:
            for dx, dy in f["disp"]:
                ax.plot(dx, dy, "-", color="#ffd400", lw=1.0, zorder=4)
            for wk in f.get("wakes", [f.get("wake")]):
                if wk is not None:
                    ax.plot(*wk, "--", color="#ffd400", lw=0.9, zorder=4)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect("equal")
        extra = "; жёлтым — граница вытеснения (контур + δ*) и линия следа" if "disp" in f else ""
        warn = "" if f.get("converged", True) else " — РАСЧЁТ НЕ СОШЁЛСЯ"
        ax.set_title(f"{label}, α = {f['alpha']:g}°{warn}\n{name}{extra}", fontsize=9)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        self.t_fl.draw()
        self.status.set("Поле течения построено.")

    # ------------------------------------------------------------ сетка RANS
    MESH_FIELDS = [
        ("yplus", "y⁺ первой ячейки", "1", "1 — для SA без пристеночных функций; больше — грубее и быстрее"),
        ("ratio", "рост высоты слоя", "1.2", "отношение высот соседних слоёв у стенки (1.1–1.3)"),
        ("bl_thickness", "толщина слоя четырёхуг., c", "0", "0 — авто: 0,02c, но ≤ 0,3 наименьшей щели"),
        ("n_surface", "точек на сторону (осн. эл.)", "0", "0 — как в контуре; у закрылков — пропорц. хорде"),
        ("near_size", "ячейки у стенок и в щелях, c", "0", "0 — авто: 0,004c (профиль), 0,003c (механизация)"),
        ("growth", "рост треугольников", "0.25", "размер = у стенки + доля × расстояние до стенки"),
        ("wake_size", "ячейки по следу, c", "0.008", "сгущение по линиям следа (механизация)"),
        ("wake_length", "длина сгущения следа, c", "0.6", "за самой дальней задней кромкой"),
        ("far_radius", "радиус внешней границы, c", "50", "20–100 хорд"),
        ("far_size", "ячейки на внешней границе, c", "8", ""),
    ]

    def _build_mesh_tab(self, nb):
        fr = ttk.Frame(nb)
        nb.add(fr, text="Сетка")
        self.t_mesh_frame = fr
        side = ttk.Frame(fr, padding=6)
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Сетка для RANS (gmsh)", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                 sticky="w")
        self.mesh_vars = {}
        for r, (key, lbl, val, tip) in enumerate(self.MESH_FIELDS, start=1):
            ttk.Label(side, text=lbl).grid(row=2 * r - 1, column=0, sticky="w", pady=(3, 0))
            v = tk.StringVar(value=val)
            ttk.Entry(side, textvariable=v, width=8).grid(row=2 * r - 1, column=1, sticky="w", pady=(3, 0))
            if tip:
                ttk.Label(side, text=tip, foreground="#777", font=("", 8), wraplength=230).grid(
                    row=2 * r, column=0, columnspan=2, sticky="w")
            self.mesh_vars[key] = v
        r0 = 2 * len(self.MESH_FIELDS) + 1
        self.mesh_wake = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="сгущать по линиям следа", variable=self.mesh_wake).grid(
            row=r0, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(side, text="Построить и показать сетку", command=self.build_mesh_preview).grid(
            row=r0 + 1, column=0, columnspan=2, sticky="we", pady=(8, 2))
        ttk.Button(side, text="Параметры по умолчанию", command=self.reset_mesh_opts).grid(
            row=r0 + 2, column=0, columnspan=2, sticky="we", pady=2)
        self.mesh_info = tk.StringVar(value="Сетка строится для текущего профиля, первого Re и среднего α "
                                            "(по нему идут линии следа). Построенная здесь сетка используется "
                                            "в расчёте RANS, пока не изменены профиль, Re или параметры.")
        ttk.Label(side, textvariable=self.mesh_info, wraplength=240, justify="left").grid(
            row=r0 + 3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        right = ttk.Frame(fr)
        right.pack(side="left", fill="both", expand=True)
        bar = ttk.Frame(right)
        bar.pack(side="top", fill="x")
        from .field import REGIONS
        ttk.Label(bar, text="  увеличение: ").pack(side="left")
        self.mesh_reg = tk.StringVar(value=list(REGIONS)[1])
        cr = ttk.Combobox(bar, textvariable=self.mesh_reg, values=list(REGIONS), state="readonly", width=26)
        cr.pack(side="left")
        cr.bind("<<ComboboxSelected>>", lambda _: self.draw_mesh())
        self.mesh_fig = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self.mesh_canvas = FigureCanvasTkAgg(self.mesh_fig, master=right)
        NavigationToolbar2Tk(self.mesh_canvas, right).update()
        self.mesh_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._mesh = None            # (подпись, сетка)

    def reset_mesh_opts(self):
        for key, _, val, _ in self.MESH_FIELDS:
            self.mesh_vars[key].set(val)
        self.mesh_wake.set(True)

    def mesh_opts(self):
        from .rans_mesh import MeshOptions
        kw = {}
        for key, lbl, _, _ in self.MESH_FIELDS:
            try:
                v = float(self.mesh_vars[key].get().replace(",", "."))
            except ValueError:
                raise ValueError(f"Сетка: «{lbl}» — не число") from None
            kw[key] = int(v) if key == "n_surface" else v
        if kw["yplus"] <= 0 or not 1.0 < kw["ratio"] < 2.0:
            raise ValueError("Сетка: y⁺ > 0, рост слоя от 1 до 2")
        return MeshOptions(wake_refine=self.mesh_wake.get(), **kw)

    @staticmethod
    def _mesh_signature(af, re, alpha, opts):
        els = af if isinstance(af, list) else [af]
        xy = np.concatenate([np.r_[e.x, e.y] for e in els])
        return (len(xy), round(float(xy.sum()), 9), round(float((xy ** 2).sum()), 9), float(re),
                round(float(alpha), 3), repr(opts))

    def _mesh_inputs(self):
        af = self.get_config()
        mach, alphas, niter, re_, ncrit, xtr = self._params()
        opts = self.mesh_opts()
        chord = 1.0 if isinstance(af, list) and self.src.get() != "mfile" else None
        els = af if isinstance(af, list) else [af]
        if not chord:
            chord = float(max(np.max(e.x) for e in els) - min(np.min(e.x) for e in els))
        a_mid = float(np.median(alphas))
        return els, re_[0], a_mid, opts, chord, self._mesh_signature(af, re_[0], a_mid, opts)

    def build_mesh_preview(self):
        try:
            els, re, a_mid, opts, chord, sig = self._mesh_inputs()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Сетка", str(e))
            return
        self.mesh_info.set("Строю сетку…")

        def job():
            try:
                from .rans import mesh_for
                m = mesh_for(els, a_mid, re, chord, opts=opts)
                self.q.put(("mesh", sig, m, re, chord))
            except Exception as e:  # noqa: BLE001
                self.q.put(("mesh_error", str(e)))
        threading.Thread(target=job, daemon=True).start()

    def _set_mesh(self, sig, m, re, chord):
        from .rans_mesh import quality
        self._mesh = (sig, m)
        q = quality(m)
        wc = m.face_cells[m.face_tag == 1, 0]
        h1 = float(np.median(2 * m.wall_dist[wc]))
        cf = 0.026 * re ** (-1 / 7)
        yp = h1 / chord * re * np.sqrt(cf / 2)
        self.mesh_info.set(
            f"Ячеек: {q['cells']:,} (четырёхугольников {q['quads']:,}); граней на стенках {q['wall_faces']}.\n"
            .replace(",", " ") +
            f"Высота первой ячейки ≈ {h1:.2e} c, оценка y⁺ ≈ {yp:.2f} (по Cf пластины).\n"
            f"Наибольшее отношение объёмов соседних ячеек — {q['vol_ratio_max']:.0f}.\n"
            "Эта сетка будет использована в расчёте RANS." +
            ("\n⚠ y⁺ > 2: для модели SA без пристеночных функций слой у стенки грубоват." if yp > 2 else "") +
            ("\n⚠ Резкий скачок размеров соседних ячеек (> 300): возможен развал счёта — уменьшите "
             "«рост треугольников» или «ячейки у стенок и в щелях»." if q["vol_ratio_max"] > 300 else ""))
        self.draw_mesh()

    def draw_mesh(self):
        if not self._mesh:
            return
        from matplotlib.collections import PolyCollection
        from .field import REGIONS
        m = self._mesh[1]
        fig = self.mesh_fig
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1)
        polys = [m.nodes[c] for c in m.cells]
        for ax, box in ((ax1, None), (ax2, REGIONS[self.mesh_reg.get()])):
            ax.add_collection(PolyCollection(polys, facecolor="none", edgecolor="#333", lw=0.15))
            wf = m.face_tag == 1
            seg = m.nodes[m.face_nodes[wf]]
            for s_ in seg:
                ax.plot(s_[:, 0], s_[:, 1], "-", color="#d62728", lw=0.8)
            if box is None:
                xs, ys = m.nodes[:, 0], m.nodes[:, 1]
                cx = 0.5 * (xs.min() + xs.max())
                ax.set_xlim(-0.6, 1.8)
                ax.set_ylim(-0.9, 0.9)
                ax.set_title("Около профиля (вся область — круг радиусом "
                             f"{np.hypot(xs - cx, ys).max():.0f} c)", fontsize=9)
            else:
                ax.set_xlim(box[0], box[1])
                ax.set_ylim(box[2], box[3])
                ax.set_title(self.mesh_reg.get(), fontsize=9)
            ax.set_aspect("equal")
        self.mesh_canvas.draw_idle()

    # ------------------------------------------------------------ невязки RANS
    def _build_res_tab(self, nb):
        fr = ttk.Frame(nb)
        nb.add(fr, text="Невязки")
        bar = ttk.Frame(fr, padding=(4, 4))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Переход к следующему углу: невязка ниже 10^−").pack(side="left")
        self.cv_orders = tk.StringVar(value="3")
        ttk.Entry(bar, textvariable=self.cv_orders, width=4).pack(side="left")
        ttk.Label(bar, text=" от наибольшей,").pack(side="left")
        self.cv_hold = tk.StringVar(value="20")
        ttk.Entry(bar, textvariable=self.cv_hold, width=4).pack(side="left", padx=(4, 0))
        ttk.Label(bar, text=" итераций подряд;").pack(side="left")
        bar = ttk.Frame(fr, padding=(4, 0))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="за эти итерации Cl меняется меньше чем на").pack(side="left")
        self.cv_cl = tk.StringVar(value="0.02")
        ttk.Entry(bar, textvariable=self.cv_cl, width=5).pack(side="left")
        ttk.Label(bar, text=" %, Cd — меньше чем на").pack(side="left")
        self.cv_cd = tk.StringVar(value="0.00002")
        ttk.Entry(bar, textvariable=self.cv_cd, width=8).pack(side="left")
        bar2 = ttk.Frame(fr, padding=(4, 0))
        bar2.pack(side="top", fill="x")
        self.cv_stall = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar2, text="принимать и при «застрявшей» невязке (упала на 2 порядка, Cl и Cd стоят "
                        "≥ 30 итераций)", variable=self.cv_stall).pack(side="left")
        ttk.Button(bar2, text="⏭ Принять угол, следующий", command=self.skip_angle).pack(side="right")
        # шаг по псевдовремени и защита от развала
        bar4 = ttk.Frame(fr, padding=(4, 2))
        bar4.pack(side="top", fill="x")
        ttk.Label(bar4, text="Шаг: режим").pack(side="left")
        self.cfl_preset = tk.StringVar(value="обычный")
        cp_ = ttk.Combobox(bar4, textvariable=self.cfl_preset, values=["обычный", "осторожный", "свой"],
                           state="readonly", width=10)
        cp_.pack(side="left")
        cp_.bind("<<ComboboxSelected>>", lambda _: self._apply_cfl_preset())
        self.cfl_vars = {}
        bar5 = ttk.Frame(fr, padding=(4, 0))
        bar5.pack(side="top", fill="x")
        for key, lbl, val, w in (("cfl_start", " CFL нач.", "5", 5), ("cfl_limit", " наиб.", "1e6", 6),
                                 ("cfl_growth", " рост ×", "1.25", 5), ("first_order", " 1-й порядок, итер.", "30", 4),
                                 ("rollback_factor", "Откат к лучшему состоянию, если невязка выросла в", "100", 5)):
            bar_ = bar5 if key == "rollback_factor" else bar4
            ttk.Label(bar_, text=lbl).pack(side="left")
            v = tk.StringVar(value=val)
            e_ = ttk.Entry(bar_, textvariable=v, width=w)
            e_.pack(side="left")
            e_.bind("<KeyRelease>", lambda _: self.cfl_preset.set("свой"))
            self.cfl_vars[key] = v
        ttk.Label(bar5, text=" раз").pack(side="left")
        ttk.Button(bar5, text="↶ Откат к лучшему", command=self.rollback_now).pack(side="right")
        ttk.Button(bar5, text="CFL ×0,5", command=self.halve_cfl).pack(side="right", padx=4)
        bar3 = ttk.Frame(fr, padding=(4, 2))
        bar3.pack(side="top", fill="x")
        ttk.Label(bar3, text="Показать: ").pack(side="left")
        self.res_src = tk.StringVar(value="текущий расчёт")
        self.cb_res_src = ttk.Combobox(bar3, textvariable=self.res_src, values=["текущий расчёт"],
                                       state="readonly", width=36)
        self.cb_res_src.pack(side="left")
        self.cb_res_src.bind("<<ComboboxSelected>>", lambda _: (self.res_angle.set("все углы"), self.plot_res()))
        ttk.Label(bar3, text="   угол: ").pack(side="left")
        self.res_angle = tk.StringVar(value="все углы")
        self.cb_res_angle = ttk.Combobox(bar3, textvariable=self.res_angle, values=["все углы"],
                                         state="readonly", width=10)
        self.cb_res_angle.pack(side="left")
        self.cb_res_angle.bind("<<ComboboxSelected>>", lambda _: self.plot_res())
        ttk.Button(bar3, text="Досчитать несошедшиеся углы", command=self.continue_rans).pack(side="right")
        self.res_fig = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=fr)
        NavigationToolbar2Tk(self.res_canvas, fr).update()
        self.res_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._live = None
        self._res_drawn = 0.0

    TURB_MODELS = {"SA (Спаларт–Аллмарас)": "sa",
                   "SA-RC (поправка на кривизну)": "sa-rc",
                   "SA-AFT (переход, Ñкрит из окна)": "sa-aft"}

    CFL_PRESETS = {"обычный": dict(cfl_start="5", cfl_limit="1e6", cfl_growth="1.25", first_order="30",
                                   rollback_factor="100"),
                   "осторожный": dict(cfl_start="2", cfl_limit="1e4", cfl_growth="1.1", first_order="60",
                                      rollback_factor="20")}

    def _apply_cfl_preset(self):
        p = self.CFL_PRESETS.get(self.cfl_preset.get())
        if p:
            for k, v in p.items():
                self.cfl_vars[k].set(v)

    def solver_settings(self):
        """Параметры решателя RANS из окна (для RansSolver через solver_opts)."""
        num = lambda v: float(v.get().replace(",", "."))
        try:
            o = {k: num(v) for k, v in self.cfl_vars.items()}
        except ValueError:
            raise ValueError("Невязки: параметры шага — проверьте числа") from None
        if not (0 < o["cfl_start"] <= o["cfl_limit"]) or o["cfl_growth"] < 1.0 or o["rollback_factor"] <= 1.0:
            raise ValueError("Невязки: CFL нач. > 0 и ≤ наибольшего, рост ≥ 1, откат при росте > 1")
        o["first_order"] = int(o["first_order"])
        o["prec_fill"] = int(self.prec_fill.get())
        o["turb_model"] = self.TURB_MODELS[self.turb_model.get()]
        if o["turb_model"].endswith("aft"):
            try:
                o["ncrit"] = num(self.ncrit)
            except ValueError:
                raise ValueError("Ñкрит — проверьте число") from None
            if not 1.0 <= o["ncrit"] <= 14.0:
                raise ValueError("Ñкрит для модели перехода: от 1 до 14")
        return o

    def halve_cfl(self):
        ctl = getattr(self, "_rans_ctl", None)
        if ctl is not None and self.worker and self.worker.is_alive():
            ctl.cfl_factor = 0.5
            self.status.set("CFL уменьшен вдвое (со следующей итерации)")

    def rollback_now(self):
        ctl = getattr(self, "_rans_ctl", None)
        if ctl is not None and self.worker and self.worker.is_alive():
            ctl.rollback = True
            self.status.set("Откат к состоянию с наименьшей невязкой (со следующей итерации)…")

    def conv_settings(self):
        from .rans import Convergence
        num = lambda v: float(v.get().replace(",", "."))
        try:
            c = Convergence(orders=num(self.cv_orders), hold=int(num(self.cv_hold)),
                            cl_tol=num(self.cv_cl) / 100.0, cd_tol=num(self.cv_cd), stall_ok=self.cv_stall.get())
        except ValueError:
            raise ValueError("Невязки: критерий сходимости — проверьте числа") from None
        if c.orders <= 0 or c.hold < 1:
            raise ValueError("Невязки: порядков > 0, итераций ≥ 1")
        return c

    def skip_angle(self):
        ctl = getattr(self, "_rans_ctl", None)
        if ctl is not None and self.worker and self.worker.is_alive():
            ctl.skip = True
            self.status.set("Текущий угол будет принят после этой итерации…")

    @staticmethod
    def _new_res():
        return dict(angles={}, rref=1.0, last=None, events=[])

    @staticmethod
    def _res_add(D, a, it, res, rn, cl, cd):
        A = D["angles"].setdefault(float(a), dict(it=[], res=[], rn=[], cl=[], cd=[]))
        if A["it"] and it <= A["it"][-1]:          # угол пересчитывается заново
            for v in A.values():
                v.clear()
        A["it"].append(it); A["res"].append(res); A["rn"].append(rn); A["cl"].append(cl); A["cd"].append(cd)
        D["last"] = (float(a), it, cl, cd)

    def _on_iter(self, d):
        L = self._live
        if L is None:
            return
        if "event" in d:                                   # откат и т. п.
            L["events"].append((float(d["alpha"]), int(d["it"]), d["event"]))
            self.status.set(f"α = {d['alpha']:g}°, итерация {d['it']}: {d['event']}")
            return
        self._res_add(L, d["alpha"], d["it"], d["res"], d["rn"], d["cl"], d["cd"])
        L["rref"] = max(L["rref"], d["rref"])
        import time as _t
        if self.res_src.get() == "текущий расчёт" and _t.time() - self._res_drawn > 1.5:
            self.plot_res()

    def _res_data(self):
        """Данные для вкладки «Невязки»: текущий расчёт или история законченной поляры RANS."""
        src = self.res_src.get()
        if src == "текущий расчёт":
            return self._live
        for p in self.polars:
            if p.method == "rans" and p.label == src and p.field_ctx:
                D = self._new_res()
                for a, H in zip(p.field_ctx["alpha"], p.field_ctx["history"]):
                    if H is None or not len(H["it"]):
                        continue
                    for j in range(len(H["it"])):
                        self._res_add(D, a, int(H["it"][j]), H["res"][j], H["rn"][j], H["cl"][j], H["cd"][j])
                    D["rref"] = max(D["rref"], H.get("rref", 1.0))
                    D["events"] += [(float(a), int(e[0]), e[1]) for e in H.get("events", [])]
                return D
        return None

    def plot_res(self):
        import time as _t
        self._res_drawn = _t.time()
        fig = self.res_fig
        fig.clear()
        D = self._res_data()
        if not D or not D["angles"]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Ход невязок появится во время расчёта RANS", ha="center", va="center",
                    transform=ax.transAxes, color="#777")
            ax.set_axis_off()
            self.res_canvas.draw_idle()
            return
        angs = sorted(D["angles"])
        vals = ["все углы"] + [f"{a:g}" for a in angs]
        self.cb_res_angle["values"] = vals
        if self.res_angle.get() not in vals:
            self.res_angle.set("все углы")
        ax1, ax2 = fig.subplots(2, 1, sharex=True, height_ratios=[3, 2])
        try:
            c = self.conv_settings()
            lvl, lbl = 10.0 ** (-c.orders) * D["rref"], f"порог: 10^−{c.orders:g} от наибольшей"
        except ValueError:
            lvl = None
        sel = self.res_angle.get()
        if sel == "все углы":
            import matplotlib as mpl
            cmap = mpl.colormaps["turbo"]
            for k, a in enumerate(angs):
                A = D["angles"][a]
                col = cmap(0.1 + 0.8 * k / max(1, len(angs) - 1))
                ax1.semilogy(A["it"], A["rn"], "-", color=col, lw=1.2, label=f"α = {a:g}°")
                ax2.plot(A["it"], A["cl"], "-", color=col, lw=1.2)
            ax2.set_ylabel("Cl")
            cls = [A["cl"][len(A["cl"]) // 3:] for A in D["angles"].values() if A["cl"]]
            allc = np.concatenate([np.asarray(c_, float) for c_ in cls if len(c_)]) if cls else np.array([])
            if allc.size:
                lo, hi = np.nanmin(allc), np.nanmax(allc)
                ax2.set_ylim(lo - 0.1 * (hi - lo) - 0.01, hi + 0.1 * (hi - lo) + 0.01)
            ax1.set_ylabel("невязка, сумма по уравнениям")
        else:
            A = D["angles"][float(sel)]
            R = np.array(A["res"])
            names = ["неразрывность", "импульс x", "импульс y", "энергия", "турбулентность (SA)"]
            for j in range(5):
                ax1.semilogy(A["it"], R[:, j], lw=1.0, label=names[j])
            ax1.semilogy(A["it"], A["rn"], "k-", lw=1.6, label="сумма")
            ax2.plot(A["it"], A["cl"], "-", color="#1f77b4", lw=1.2)
            ax2.set_ylabel("Cl", color="#1f77b4")
            ax3 = ax2.twinx()
            ax3.plot(A["it"], A["cd"], "-", color="#d62728", lw=1.0)
            ax3.set_ylabel("Cd", color="#d62728")
            m0 = len(A["cl"]) // 3
            for ax_, v_ in ((ax2, A["cl"][m0:]), (ax3, A["cd"][m0:])):
                if len(v_):
                    lo, hi = float(np.nanmin(v_)), float(np.nanmax(v_))
                    pad = 0.1 * (hi - lo) + 1e-4
                    ax_.set_ylim(lo - pad, hi + pad)
            ax1.set_ylabel("невязка (ср. кв. по ячейкам)")
        if lvl:
            ax1.axhline(lvl, color="#d62728", ls="--", lw=1.0, label=lbl)
        evs = [e for e in D.get("events", []) if sel == "все углы" or abs(e[0] - float(sel)) < 1e-9]
        for k_, (a_, it_, txt) in enumerate(evs):
            ax1.axvline(it_, color="#ff7f0e", lw=0.9, ls="-.", label="откат / ручное действие" if k_ == 0 else None)
        if evs:
            ax1.text(0.01, 0.02, f"последнее: α = {evs[-1][0]:g}°, итер. {evs[-1][1]}: {evs[-1][2]}",
                     transform=ax1.transAxes, fontsize=7, color="#b35900")
        ax1.grid(alpha=0.3, which="both")
        ax1.legend(fontsize=7, ncol=3, loc="upper right")
        ax2.grid(alpha=0.3)
        ax2.set_xlabel("итерация Ньютона на угле")
        if D["last"]:
            a, it, cl, cd = D["last"]
            ax1.set_title(f"последнее: α = {a:g}°, итерация {it}: Cl = {cl:.4f}, Cd = {cd:.5f}", fontsize=9)
        self.res_canvas.draw_idle()

    def _rans_polar_for_continue(self):
        src = self.res_src.get()
        cand = [p for p in self.polars if p.method == "rans" and p.field_ctx]
        sel = [p for p in cand if p.label == src]
        return sel[-1] if sel else (cand[-1] if cand else None)

    def continue_rans(self):
        """Досчитать несошедшиеся углы выбранной (или последней) поляры RANS."""
        if self.worker and self.worker.is_alive():
            return
        p = self._rans_polar_for_continue()
        if p is None:
            messagebox.showinfo("Досчитать", "Нет поляры RANS этого сеанса.")
            return
        bad = [f"{a:g}" for a, ok in zip(p.alpha, p.ok) if not ok]
        if not bad:
            messagebox.showinfo("Досчитать", f"В поляре «{p.label}» все углы уже сошлись.")
            return
        try:
            rit = int(self.rans_iter.get())
            conv = self.conv_settings()
            sopts = self.solver_settings()
            from .parallel import resolve_workers
            nw = resolve_workers(self.cores.get())
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        if not messagebox.askyesno("Досчитать", f"Поляра «{p.label}»: продолжить α = {', '.join(bad)}° "
                                   f"(до {rit} итераций на угол, каждый — со своего решения)?"):
            return
        from .rans import RansControl
        self._live = self._new_res()
        self.res_src.set("текущий расчёт")
        self._rans_ctl = RansControl(monitor=lambda d: self.q.put(("iter", d)))
        self.stop_flag.clear()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_skip.configure(state="normal")
        self.progress["value"] = 0
        self.status.set("Досчитываю несошедшиеся углы…")

        def job():
            try:
                def cb(text, frac):
                    self.q.put(("progress", text, frac))
                    return self.stop_flag.is_set()
                q = core.continue_rans(p, cb, rit, conv=conv, control=self._rans_ctl, workers=nw,
                                       solver_opts=sopts)
                self.q.put(("done", q))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", f"{e}\n\n{traceback.format_exc(limit=3)}"))
        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def save_res_csv(self):
        D = self._res_data()
        if not D or not D["angles"]:
            messagebox.showinfo("Нет данных", "Нет хода невязок: выберите расчёт RANS на вкладке «Невязки».")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV (Excel)", "*.csv")],
                                            initialfile="residuals.csv")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["alpha", "итерация", "невязка: неразрывность", "импульс x", "импульс y", "энергия",
                        "турбулентность", "сумма", "Cl", "Cd"])
            for a in sorted(D["angles"]):
                A = D["angles"][a]
                for j in range(len(A["it"])):
                    w.writerow([f"{a:g}", A["it"][j]] + [f"{v:.6e}" for v in A["res"][j]] +
                               [f"{A['rn'][j]:.6e}", f"{A['cl'][j]:.6f}", f"{A['cd'][j]:.6f}"])
        self.status.set(f"Ход невязок сохранён: {path}")

    def clear(self):
        self.polars.clear()
        self.table.delete(*self.table.get_children())
        self.notes.delete("1.0", "end")
        self._field_cache.clear()
        self.cb_alpha["values"] = []
        self.cp_alpha.set("")
        self.replot()

    # ------------------------------------------------------------- экспорт
    def save_csv(self):
        if not self.polars:
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV (Excel)", "*.csv")],
                                         initialfile="polar.csv")
        if p:
            core.save_csv(self.polars, p)
            self.status.set(f"Таблица сохранена: {p}")

    def save_png(self):
        if not self.polars:
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
            return
        d = filedialog.askdirectory(title="Папка для графиков")
        if d:
            for name, t in [("profile", self.t_geo), ("cl_alpha", self.t_cl), ("cd", self.t_cd),
                            ("cm_alpha", self.t_cm), ("cp", self.t_cp), ("boundary_layer", self.t_bl),
                            ("flow_field", self.t_fl)]:
                t.fig.savefig(os.path.join(d, f"{name}.png"), dpi=150)
            self.status.set(f"Графики сохранены в {d}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
