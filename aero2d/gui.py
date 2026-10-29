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
from matplotlib import colors as mpl_colors  # noqa: E402
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


class Tooltip:
    """Подсказка при наведении (через 0,5 с)."""

    def __init__(self, widget, text):
        self.w, self.text, self.tip, self.job = widget, text, None, None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._hide()
        self.job = self.w.after(500, self._show)

    def _show(self):
        if self.tip or not self.text:
            return
        x = self.w.winfo_rootx() + 12
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tk.Toplevel(self.w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", wraplength=360, background="#fffbe6",
                 relief="solid", borderwidth=1, padx=6, pady=4).pack()

    def _hide(self, _=None):
        if self.job:
            self.w.after_cancel(self.job)
            self.job = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App(tk.Tk):
    # методы в списке «Метод»: подпись → значение self.method
    METHODS_UI = [("Вязкий — панели + пограничный слой (как XFOIL)", "viscous"),
                  ("Невязкий — панели (мгновенно, без Cd)", "panel"),
                  ("Эйлер — сжимаемый, скачки (M ≥ 0,2)", "euler"),
                  ("RANS — Навье–Стокс (экспериментально)", "rans")]

    def __init__(self):
        super().__init__()
        self.title("aero2d — аэродинамика профилей")
        self.geometry("1280x880")
        self.minsize(1000, 650)
        self._init_fonts()
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
        left = self._left = ttk.Frame(self, padding=8, width=470)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)                     # ширина панели не «прыгает» при смене метода
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        # --- профиль: источник в левом столбце, его поле или кнопка — в правом
        f = ttk.LabelFrame(left, text="Профиль", padding=6)
        f.pack(fill="x", pady=4)
        f.columnconfigure(1, weight=1)
        self.src = tk.StringVar(value="naca")
        self.naca = tk.StringVar(value="2412")
        self.file = tk.StringVar(value="")
        self.mfile = tk.StringVar(value="")
        self.smooth = tk.BooleanVar(value=False)
        lib = core.library()
        self._lib = dict(lib)
        self.lib_name = tk.StringVar(value=lib[0][0] if lib else "")
        rb = lambda text, val, r: ttk.Radiobutton(f, text=text, variable=self.src, value=val,
                                                 command=self.preview_airfoil).grid(row=r, column=0, sticky="w",
                                                                                    pady=1)
        rb("NACA", "naca", 0)
        e = ttk.Entry(f, textvariable=self.naca, width=12)
        e.grid(row=0, column=1, sticky="w")
        e.bind("<Return>", lambda _: (self.src.set("naca"), self.preview_airfoil()))
        e.bind("<FocusOut>", lambda _: self.preview_airfoil())
        Tooltip(e, "4 цифры (2412), модифицированный 4-значный (0012-64) или серия 16 (16-012); Enter — показать")
        rb("Из набора", "lib", 1)
        cl_ = ttk.Combobox(f, textvariable=self.lib_name, values=[n for n, _ in lib], state="readonly", width=20)
        cl_.grid(row=1, column=1, sticky="w")
        cl_.bind("<<ComboboxSelected>>", lambda _: (self.src.set("lib"), self.preview_airfoil()))
        rb("Файл координат", "file", 2)
        b_ = ttk.Button(f, text="Открыть…", command=self.open_file)
        b_.grid(row=2, column=1, sticky="w")
        Tooltip(b_, "Формат Selig или Lednicer (как на сайте UIUC, airfoiltools)")
        rb("Многоэлементный файл", "mfile", 3)
        b_ = ttk.Button(f, text="Открыть…", command=self.open_mfile)
        b_.grid(row=3, column=1, sticky="w")
        Tooltip(b_, "Элементы механизации в одном файле, блоки разделены пустой строкой или «999 999» (как в MSES)")
        self.file_disp = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.file_disp, foreground="#555", wraplength=300).grid(
            row=4, column=0, columnspan=2, sticky="w")
        c_ = ttk.Checkbutton(f, text="сгладить контур (измеренные координаты)", variable=self.smooth,
                             command=self.preview_airfoil)
        c_.grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))
        Tooltip(c_, "Шум координат у передней кромки (даже 0,1 мм) мешает расчёту пограничного слоя")
        self._control = None
        # --- механизация из одного профиля
        self.hl_cfg = {"flap": None, "slat": None}
        ttk.Separator(f).grid(row=6, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Label(f, text="Механизация, рули").grid(row=7, column=0, sticky="w")
        ttk.Button(f, text="Настроить…", command=self.open_highlift).grid(row=7, column=1, sticky="w")
        self.hl_desc = tk.StringVar(value="нет")
        ttk.Label(f, textvariable=self.hl_desc, foreground="#555", wraplength=300).grid(
            row=8, column=0, columnspan=2, sticky="w")

        # --- метод: выбор списком; параметры — только выбранного метода
        f = ttk.LabelFrame(left, text="Метод", padding=6)
        f.pack(fill="x", pady=4)
        self.method = tk.StringVar(value="viscous")
        self.method_name = tk.StringVar(value=self.METHODS_UI[0][0])
        cm_ = ttk.Combobox(f, textvariable=self.method_name, values=[n for n, _ in self.METHODS_UI],
                           state="readonly", width=44)
        cm_.pack(fill="x")
        cm_.bind("<<ComboboxSelected>>", lambda _: self._on_method())
        Tooltip(cm_, "Вязкий — основной метод: панели + интегральный пограничный слой, переход eᴺ, Cd (как "
                     "XFOIL/XFLR5). Невязкий — мгновенно, без сопротивления. Эйлер — сжимаемый, скачки. "
                     "RANS — Навье–Стокс, экспериментально, долго.")

        # --- условия расчёта: подписи слева, поля в одном столбце
        f = ttk.LabelFrame(left, text="Условия расчёта", padding=6)
        f.pack(fill="x", pady=4)
        f.columnconfigure(1, weight=1)
        self.mach = tk.StringVar(value="0")
        self.re = tk.StringVar(value="1e6")
        self.ncrit = tk.StringVar(value="9")
        self.xtru = tk.StringVar(value="1")
        self.xtrl = tk.StringVar(value="1")
        self.a0, self.a1, self.da = tk.StringVar(value="-4"), tk.StringVar(value="14"), tk.StringVar(value="1")
        self.cref = tk.StringVar(value="авто")
        self._cond_rows = {}

        def row(r, key, label, widgets, tip):
            lb = ttk.Label(f, text=label)
            lb.grid(row=r, column=0, sticky="w", pady=2, padx=(0, 6))
            fr_ = ttk.Frame(f)
            fr_.grid(row=r, column=1, sticky="w", pady=2)
            for w_ in widgets(fr_):
                w_.pack(side="left", padx=(0, 3))
            Tooltip(lb, tip)
            for ch_ in fr_.winfo_children():
                Tooltip(ch_, tip)
            self._cond_rows[key] = (lb, fr_)
        ent = lambda var, w=8: (lambda fr_: [ttk.Entry(fr_, textvariable=var, width=w)])
        row(0, "re", "Число Рейнольдса Re", ent(self.re, 14),
            "По опорной хорде. Можно несколько через «;» или пробел: 1e5; 3e5; 1e6 — все поляры за один запуск.")
        row(1, "mach", "Число Маха M", ent(self.mach),
            "Сжимаемость — поправкой Кармана–Цяня (вязкий и невязкий метод до M ≈ 0,6).")
        row(2, "ncrit", "Ñкрит (переход eᴺ)", ent(self.ncrit),
            "Возмущённость потока: 9 — обычная аэродинамическая труба; 10–11 — малотурбулентная труба "
            "или спокойная атмосфера; 4–7 — турбулентный поток.")
        row(3, "xtr", "Переход не дальше x/c",
            lambda fr_: [ttk.Label(fr_, text="верх"), ttk.Entry(fr_, textvariable=self.xtru, width=5),
                         ttk.Label(fr_, text="низ"), ttk.Entry(fr_, textvariable=self.xtrl, width=5)],
            "Принудительный переход (турбулизатор): 1 — свободный переход; 0,05 — у передней кромки.")
        row(4, "alpha", "Угол атаки α, °",
            lambda fr_: [ttk.Label(fr_, text="от"), ttk.Entry(fr_, textvariable=self.a0, width=5),
                         ttk.Label(fr_, text="до"), ttk.Entry(fr_, textvariable=self.a1, width=5),
                         ttk.Label(fr_, text="шаг"), ttk.Entry(fr_, textvariable=self.da, width=4)],
            "Диапазон углов атаки поляры. Для одной точки: «от» = «до».")
        row(5, "cref", "Опорная хорда c", ent(self.cref),
            "По ней считаются Re и коэффициенты Cl = L′/(q∞·c). «авто»: 1 у профиля и у механизации, "
            "построенной из профиля; габарит по x — у многоэлементного файла.")
        from .parallel import cpu_count, auto_workers
        self.cores = tk.StringVar(value="авто")
        row(6, "cores", f"Ядер ЦП (здесь {cpu_count()})",
            lambda fr_: [ttk.Combobox(fr_, textvariable=self.cores, width=6,
                                      values=["авто"] + [str(n) for n in (1, 2, 4, 8, 12, 16, 24, 32)
                                                         if n <= cpu_count()])],
            f"Поляра делится на куски по α, куски считаются одновременно. «авто» = {auto_workers()} "
            "(все, кроме одного).")

        # --- параметры Эйлера и RANS (показываются только для этих методов)
        self.grid_name = tk.StringVar(value=list(core.GRIDS)[0])
        self.niter = tk.StringVar(value="100")
        fe = ttk.LabelFrame(left, text="Параметры метода Эйлера", padding=6)
        ttk.Label(fe, text="сетка").grid(row=0, column=0, sticky="w")
        ttk.Combobox(fe, textvariable=self.grid_name, values=list(core.GRIDS), state="readonly",
                     width=15).grid(row=0, column=1, sticky="w")
        ttk.Label(fe, text="итераций").grid(row=1, column=0, sticky="w")
        ttk.Entry(fe, textvariable=self.niter, width=6).grid(row=1, column=1, sticky="w")
        fr = ttk.LabelFrame(left, text="Параметры RANS (экспериментально)", padding=6)
        ttk.Label(fr, text="итераций на угол, не более").grid(row=0, column=0, sticky="w")
        self.rans_iter = tk.StringVar(value="400")
        ttk.Entry(fr, textvariable=self.rans_iter, width=7).grid(row=0, column=1, sticky="w")
        ttk.Label(fr, text="модель турбулентности").grid(row=1, column=0, sticky="w")
        self.turb_model = tk.StringVar(value=list(self.TURB_MODELS)[0])
        ttk.Combobox(fr, textvariable=self.turb_model, values=list(self.TURB_MODELS), state="readonly",
                     width=22).grid(row=1, column=1, sticky="w")
        ttk.Label(fr, text="предобусловливатель ILU(k), k").grid(row=2, column=0, sticky="w")
        self.prec_fill = tk.StringVar(value="3")
        ttk.Combobox(fr, textvariable=self.prec_fill, values=["0", "1", "2", "3", "4"], state="readonly",
                     width=3).grid(row=2, column=1, sticky="w")
        self.coarse_start = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr, text="старт на грубой сетке (если > 20 тыс. ячеек)",
                        variable=self.coarse_start).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(fr, text="сетка — вкладка «Сетка»; сходимость и шаг — «Невязки»", foreground="#555").grid(
            row=4, column=0, columnspan=2, sticky="w")
        self._method_frames = {"euler": fe, "rans": fr}

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
        vm = tk.Menu(mb, tearoff=0)
        vm.add_command(label="Крупнее шрифт", accelerator="Ctrl++", command=lambda: self.set_font_scale(+1))
        vm.add_command(label="Мельче шрифт", accelerator="Ctrl+−", command=lambda: self.set_font_scale(-1))
        vm.add_command(label="Обычный размер", accelerator="Ctrl+0", command=lambda: self.set_font_scale(0))
        mb.add_cascade(label="Вид", menu=vm)
        for k in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            self.bind_all(k, lambda _: self.set_font_scale(+1))
        for k in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.bind_all(k, lambda _: self.set_font_scale(-1))
        self.bind_all("<Control-Key-0>", lambda _: self.set_font_scale(0))
        self.config(menu=mb)
        self._buttons_frame = f
        Tooltip(self.btn_run, "Ctrl+Enter")
        Tooltip(self.btn_stop, "Esc")
        self.bind_all("<Control-Return>", lambda _: self.run())
        self.bind_all("<Escape>", lambda _: self.stop() if self.worker and self.worker.is_alive() else None)


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
        b_ = ttk.Button(bar, text="Анимация по времени (GIF, RANS)…", command=self.animate_dialog)
        b_.pack(side="left", padx=6)
        Tooltip(b_, "Нестационарный расчёт RANS от выбранной точки поляры RANS и ролик поля течения. "
                    "Вязкий и панельный методы стационарные — у них поле строится кнопкой «Построить».")
        self._field_cache = {}
        self._build_res_tab(nb)
        # таблица
        tf = ttk.Frame(nb)
        nb.add(tf, text="Таблица")
        tb_ = ttk.Frame(tf)
        tb_.pack(side="top", fill="x")
        ttk.Label(tb_, text="  Cl = L′/(q∞·c), Cd = D′/(q∞·c), Cm = M′/(q∞·c²) — на единицу размаха "
                  "(S = c·1), c — опорная хорда (см. «Примечания»).", foreground="#555").pack(side="left")
        ttk.Button(tb_, text="Размерные силы…", command=self.forces_dialog).pack(side="right", padx=4, pady=2)
        # расчёт — строка-заголовок группы (свернуть/развернуть), под ней его углы атаки
        cols = ("alpha", "cl", "cd", "cdf", "cm", "ch", "xu", "xl", "ok")
        self.table = ttk.Treeview(tf, columns=cols, show="tree headings")
        self.table.heading("#0", text="Расчёт")
        self.table.column("#0", width=260, stretch=True)
        for c, t, w in zip(cols, ["α, °", "Cl", "Cd", "Cd трения", "Cm", "Ch руля", "x/c перехода, верх",
                                  "x/c перехода, низ", "Надёжно"], [55, 75, 80, 80, 75, 75, 145, 145, 70]):
            self.table.heading(c, text=t)
            self.table.column(c, width=w, anchor="e", stretch=False)
        self._table_widths = dict(zip(("#0",) + cols, (260, 55, 75, 80, 80, 75, 75, 145, 145, 70)))
        self.table.tag_configure("bad", foreground="#b03a2e")
        self.table.tag_configure("group", font="BoldFont")
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
        self._on_method()
        self.replot()                                  # пустые графики — с подсказкой, что делать
        self.set_font_scale(None)                      # сохранённый размер шрифта — и для панелей, графиков

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

    SETTINGS = os.path.join(os.path.expanduser("~"), ".aero2d_gui.json")

    def _init_fonts(self):
        """Именованные шрифты окна; их размер меняется меню «Вид» и запоминается между запусками."""
        from tkinter import font as tkfont
        self._fonts = {n: tkfont.nametofont(n) for n in ("TkDefaultFont", "TkTextFont", "TkFixedFont",
                                                         "TkMenuFont", "TkHeadingFont", "TkTooltipFont")}
        self._font_base = {n: abs(int(f.cget("size"))) or 9 for n, f in self._fonts.items()}
        base = self._fonts["TkDefaultFont"].actual()
        self._fonts["BoldFont"] = tkfont.Font(self, name="BoldFont", family=base["family"], weight="bold")
        self._fonts["SmallFont"] = tkfont.Font(self, name="SmallFont", family=base["family"])
        self._font_base["BoldFont"] = self._font_base["TkDefaultFont"] + 1
        self._font_base["SmallFont"] = max(self._font_base["TkDefaultFont"] - 1, 7)
        self._font_step = 0
        try:
            import json
            with open(self.SETTINGS, encoding="utf-8") as fh:
                self._font_step = int(json.load(fh).get("font_step", 0))
        except (OSError, ValueError, AttributeError):
            pass
        self.set_font_scale(None)

    def set_font_scale(self, d):
        """d = +1 / −1 — на пункт крупнее/мельче, 0 — обычный размер, None — применить сохранённый."""
        if d is not None:
            self._font_step = 0 if d == 0 else min(max(self._font_step + d, -3), 6)
        for n, f in self._fonts.items():
            f.configure(size=self._font_base[n] + self._font_step)
        rh = int(self._fonts["TkDefaultFont"].metrics("linespace") * 1.35)
        ttk.Style(self).configure("Treeview", rowheight=rh)
        k = (self._font_base["TkDefaultFont"] + self._font_step) / self._font_base["TkDefaultFont"]
        if hasattr(self, "_left"):
            self._left.configure(width=int(470 * k))
        if hasattr(self, "_table_widths"):
            for c, w in self._table_widths.items():
                self.table.column(c, width=int(w * k))
        # графики: надписи matplotlib крупнее вместе со шрифтом окна (dpi фигуры, размер в пикселях тот же)
        canv = [t.canvas for t in vars(self).values() if isinstance(t, PlotTab)]
        canv += [c for c in (getattr(self, "mesh_canvas", None), getattr(self, "res_canvas", None)) if c]
        for c in canv:
            fig, w = c.figure, c.get_tk_widget()
            fig._original_dpi = 100 * k              # Tk-бэкенд пересчитывает dpi от исходного
            fig.set_dpi(100 * k)
            if w.winfo_width() > 1:
                fig.set_size_inches(w.winfo_width() / fig.dpi, w.winfo_height() / fig.dpi, forward=False)
            c.draw_idle()
        if d is not None:
            try:
                import json
                with open(self.SETTINGS, "w", encoding="utf-8") as fh:
                    json.dump({"font_step": self._font_step}, fh)
            except OSError:
                pass

    def show_error(self, title, text, details=""):
        """Понятное сообщение об ошибке; техническая трасса — по кнопке «Подробнее»."""
        w = tk.Toplevel(self)
        w.title(title)
        w.transient(self)
        w.resizable(False, False)
        fr_ = ttk.Frame(w, padding=12)
        fr_.pack(fill="both", expand=True)
        ttk.Label(fr_, text=text, wraplength=460, justify="left").pack(anchor="w")
        if isinstance(text, str) and not any(ord(c) > 0x400 for c in text):
            ttk.Label(fr_, text="Если сообщение непонятно — пришлите текст из «Подробнее» разработчику.",
                      foreground="#555", wraplength=460).pack(anchor="w", pady=(6, 0))
        box = tk.Text(fr_, width=80, height=14, font="TkFixedFont")
        box.insert("end", details)
        box.configure(state="disabled")
        bb = ttk.Frame(fr_)
        bb.pack(fill="x", pady=(10, 0))

        def toggle():
            if box.winfo_ismapped():
                box.pack_forget()
            else:
                box.pack(fill="both", expand=True, pady=(8, 0), before=bb)

        def copy():
            self.clipboard_clear()
            self.clipboard_append(f"{text}\n\n{details}")
        ttk.Button(bb, text="OK", command=w.destroy).pack(side="right")
        if details:
            ttk.Button(bb, text="Подробнее", command=toggle).pack(side="right", padx=6)
            ttk.Button(bb, text="Копировать", command=copy).pack(side="right")
        w.bind("<Return>", lambda _: w.destroy())
        w.bind("<Escape>", lambda _: w.destroy())
        self._center(w)
        w.grab_set()

    def _center(self, w):
        """Окно-диалог — по центру главного окна."""
        w.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - w.winfo_reqwidth()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - w.winfo_reqheight()) // 3
        w.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _on_method(self):
        """Смена метода: параметры и вкладки — только нужные этому методу."""
        m = dict(self.METHODS_UI).get(self.method_name.get(), "viscous")
        self.method.set(m)
        for key, fr_ in self._method_frames.items():
            if key == m:
                fr_.pack(fill="x", pady=4, before=self._buttons_frame)
            else:
                fr_.pack_forget()
        need = {"re": m in ("viscous", "rans"), "ncrit": m == "viscous" or m == "rans",
                "xtr": m == "viscous", "cref": m in ("viscous", "rans", "panel"),
                "cores": m in ("viscous", "rans")}
        for key, (lb, fr_) in self._cond_rows.items():
            st = "normal" if need.get(key, True) else "disabled"
            lb.configure(foreground="" if st == "normal" else "#999")
            for ch_ in fr_.winfo_children():
                try:
                    ch_.configure(state=st)
                except tk.TclError:
                    pass
        if m == "rans":
            self.btn_skip.pack(fill="x", pady=2)
        else:
            self.btn_skip.pack_forget()
        # вкладки «Сетка» и «Невязки» нужны только для RANS
        for fr_ in (getattr(self, "t_mesh_frame", None), getattr(self, "t_res_frame", None)):
            if fr_ is not None:
                self.nb.tab(fr_, state=("normal" if m == "rans" else "hidden"))

    def _cref(self, default):
        """Опорная хорда из поля окна («авто» — default)."""
        v = self.cref.get().strip().lower().replace(",", ".")
        if v in ("", "авто", "auto"):
            return default
        try:
            c = float(v)
        except ValueError:
            raise ValueError("Опорная хорда — число или «авто»") from None
        if c <= 0:
            raise ValueError("Опорная хорда должна быть > 0")
        return c

    def get_config(self):
        """Один профиль (Airfoil) или список элементов механизации."""
        num = lambda v: float(v.get().replace(",", "."))
        if self.src.get() == "mfile":
            if not self.mfile.get():
                raise ValueError("Выберите файл многоэлементного профиля")
            from .multi import load_multi
            from .geometry import repanel
            els = load_multi(self.mfile.get())
            # не меньше точек, чем в файле (мелкие детали — ниша, кромки — не теряются)
            return [repanel(e, max(80, len(e.x) // 2 + 1), keep_frame=True) for e in els]
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
            ax.set_title(f"Многоэлементный профиль, элементов: {len(af)}")
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
    class InputError(ValueError):
        """Неверное значение в поле окна: какое поле и что в нём не так."""

        def __init__(self, var, field, msg):
            super().__init__(f"«{field}»: {msg}")
            self.var = var

    def _num(self, var, field, lo=None, hi=None, integer=False, example="1,5"):
        """Число из поля var (запятая или точка); при ошибке — InputError с названием поля."""
        t = var.get().strip().replace(" ", "").replace(",", ".")
        try:
            x = int(t) if integer else float(t)
        except ValueError:
            what = "целое число" if integer else f"число, например {example}"
            raise self.InputError(var, field, f"введите {what} (сейчас: «{var.get().strip()}»)") from None
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            g = lambda v: f"{v:g}".replace(".", ",")
            rng = f"от {g(lo)} до {g(hi)}" if lo is not None and hi is not None else \
                f"не меньше {g(lo)}" if lo is not None else f"не больше {g(hi)}"
            raise self.InputError(var, field, f"допустимо {rng} (сейчас: {g(x)})")
        return x

    def _mark_field(self, var):
        """Подсветить поле с ошибкой и поставить в него курсор; подсветка снимается при правке."""
        def walk(w):
            for c in w.winfo_children():
                yield c
                yield from walk(c)
        for w in walk(self):
            try:
                if str(w.cget("textvariable")) != str(var):
                    continue
            except tk.TclError:
                continue
            ttk.Style(self).configure("Bad.TEntry", fieldbackground="#fde2df")
            old = w.cget("style")
            w.configure(style="Bad.TEntry")
            w.focus_set()
            w.select_range(0, "end")
            tid = []
            tid.append(var.trace_add("write", lambda *_: (w.configure(style=old),
                                                          var.trace_remove("write", tid[0]))))
            return

    def _params(self):
        mach = self._num(self.mach, "Число Маха M", 0.0, 3.0, example="0,15")
        alphas = core.alpha_range(self._num(self.a0, "Угол атаки: от", -90, 90, example="-4"),
                                  self._num(self.a1, "Угол атаки: до", -90, 90, example="14"),
                                  self._num(self.da, "Угол атаки: шаг", 1e-3, 90, example="0,5"))
        niter = self._num(self.niter, "Итераций Эйлера", 1, integer=True) if self.method.get() == "euler" else 100
        txt = self.re.get().strip().replace(", ", ";").replace(" ", ";")
        re_ = []
        for t in (t for t in txt.split(";") if t.strip()):
            try:
                re_.append(float(t.replace(",", ".")))
            except ValueError:
                raise self.InputError(self.re, "Число Рейнольдса Re",
                                      f"«{t}» — не число; пример: 1e6 или несколько через «;»: 3e5; 1e6") from None
        if not re_:
            raise self.InputError(self.re, "Число Рейнольдса Re", "введите число, например 1e6")
        for r in re_:
            if not 1e3 <= r <= 1e9:
                raise self.InputError(self.re, "Число Рейнольдса Re", f"допустимо от 1e3 до 1e9 (сейчас: {r:g})")
        from .parallel import resolve_workers
        try:
            self._workers = resolve_workers(self.cores.get())
        except ValueError:
            raise self.InputError(self.cores, "Ядер ЦП", "выберите «авто» или число ядер") from None
        ncrit = self._num(self.ncrit, "Ñкрит", 0.1, 20, example="9")
        xtr = (self._num(self.xtru, "Переход не дальше x/c: верх", 0, 1, example="1"),
               self._num(self.xtrl, "Переход не дальше x/c: низ", 0, 1, example="1"))
        return mach, alphas, niter, re_, ncrit, xtr

    def _input_error(self, e):
        if isinstance(e, self.InputError):
            self._mark_field(e.var)
        messagebox.showerror("Проверьте данные", str(e), parent=self)

    def run(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            af = self.get_config()
            mach, alphas, niter, re_, ncrit, xtr = self._params()
        except Exception as e:  # noqa: BLE001
            self._input_error(e)
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
                self._input_error(e if str(e) else ValueError("Итераций на угол — целое число"))
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
        self.status.set("Расчёт механизации: первая точка (холодный старт) ищется 1–5 мин, дальше — "
                        "десятки секунд на угол…" if method == "multi_viscous" else "Расчёт…")

        control = self._control
        # всё, что нужно из окна, читается здесь: из потока расчёта переменные Tk
        # трогать нельзя (ошибка «main thread is not in main loop» или зависание)
        is_mfile = self.src.get() == "mfile"
        nm_multi = os.path.basename(self.mfile.get()) if is_mfile else self.hl_desc.get()
        try:
            ch_multi = self._cref(None if is_mfile else 1.0)
            ch_rans = self._cref(1.0 if isinstance(af, list) and not is_mfile else None)
        except ValueError as e:
            self._finish()
            self._input_error(self.InputError(self.cref, "Опорная хорда c", str(e)))
            return
        coarse = "auto" if method == "rans" and self.coarse_start.get() else False

        def job():
            try:
                def cb(text, frac):
                    self.q.put(("progress", text, frac))
                    return self.stop_flag.is_set()
                nw = self._workers
                if method in ("multi", "multi_viscous"):
                    nm, ch = nm_multi, ch_multi
                    if method == "multi":
                        ps = [core.run_multi(af, alphas, mach, nm, chord=ch)]
                    else:
                        ps = core.run_multi_viscous_batch(af, alphas, mach, re_, ncrit, xtr, cb, nm,
                                                          chord=ch, workers=nw)
                elif method == "rans":
                    nm = nm_multi if isinstance(af, list) else None
                    ps = []
                    for r_ in re_:
                        if self.stop_flag.is_set():
                            break
                        ctl = self._rans_ctl
                        ctl.stop = False
                        p_ = core.run_rans(af, alphas, mach, r_, cb, rit, nm,
                                           chord=ch_rans,
                                           mesh=(mesh0 if r_ == re0 else None), mesh_opts=mopts, conv=conv,
                                           control=ctl, workers=nw, solver_opts=sopts,
                                           coarse_start=coarse)
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
                self.q.put(("error", str(e) or type(e).__name__, traceback.format_exc()))

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
                elif msg[0] == "anim_done":
                    self._anim_running = False
                    self.progress["value"] = 1.0
                    self.status.set(msg[2])
                    self._show_gif(msg[1])
                elif msg[0] == "anim_error":
                    self._anim_running = False
                    self.status.set("Анимация: ошибка")
                    messagebox.showerror("Анимация", msg[1])
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
                    self.show_error("Ошибка расчёта", msg[1], msg[2] if len(msg) > 2 else "")
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
        self.notes.insert("end", f"■ {p.label}\n" + "".join(f"   • {t}\n" for t in p.notes + [core.coef_note(p)])
                          + "\n")
        fmt = lambda arr, k, f: "—" if arr is None or not np.isfinite(arr[k]) else format(arr[k], f)
        grp = self.table.insert("", "end", text=p.label, open=True, tags=("group",),
                                values=("", "", "", "", "", "", "", "", f"{n_ok} из {len(p.alpha)}"))
        for k in range(len(p.alpha)):
            self.table.insert(grp, "end", text="", tags=() if p.ok[k] else ("bad",), values=(
                f"{p.alpha[k]:g}", f"{p.cl[k]:.4f}", fmt(p.cd, k, ".5f"), fmt(p.cdf, k, ".5f"),
                f"{p.cm[k]:.4f}", fmt(p.ch, k, ".4f"), fmt(p.xtr_u, k, ".3f"), fmt(p.xtr_l, k, ".3f"),
                "да" if p.ok[k] else "нет"))
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
                      ha="center", va="center", transform=axd1.transAxes, color="#555")
            axd1.set_axis_off(); axd2.set_axis_off()
        axl.set_title("× — расчёт не сошёлся или вне границ применимости метода", fontsize=8, color="#555")
        if not self.polars:
            for ax in (axl, axm, axd1, axd2):
                ax.clear()
                ax.set_axis_off()
            for ax in (axl, axm, axd1):
                self._empty(ax)
        for t in (self.t_cl, self.t_cd, self.t_cm):
            t.draw()
        self.plot_cp()
        self.plot_bl()

    EMPTY = "Результатов пока нет.\nЗадайте условия слева и нажмите «Рассчитать» (Ctrl+Enter)."

    def _empty(self, ax, text=None):
        """Пустое состояние графика: что здесь будет и как это получить."""
        ax.set_axis_off()
        ax.text(0.5, 0.5, text or self.EMPTY, ha="center", va="center", transform=ax.transAxes,
                color="#666", fontsize=11)

    def plot_cp(self):
        ax = self.t_cp.ax
        ax.clear()
        ax.set_axis_on()
        try:
            a = float(self.cp_alpha.get())
        except ValueError:
            if not self.polars:
                self._empty(ax)
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
            if not self.polars:
                self._empty(fig.add_subplot(111))
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
                           ha="center", va="center", transform=axs[0, 0].transAxes, color="#555")
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

    # ------------------------------------------------------------ размерные силы
    def forces_dialog(self):
        """Силы в ньютонах по коэффициентам выбранного расчёта: ρ, V, хорда и размах."""
        if not self.polars:
            messagebox.showinfo("Силы", "Сначала выполните расчёт.")
            return
        w = tk.Toplevel(self)
        w.title("Размерные силы")
        w.transient(self)
        fr = ttk.Frame(w, padding=8)
        fr.pack(fill="both", expand=True)
        labels = [p.label for p in self.polars]
        v_p = tk.StringVar(value=labels[-1])
        ttk.Label(fr, text="Расчёт").grid(row=0, column=0, sticky="w")
        ttk.Combobox(fr, textvariable=v_p, values=labels, state="readonly", width=60).grid(row=0, column=1,
                                                                                          columnspan=5, sticky="w")
        p0 = self.polars[-1]
        a_snd = 340.29
        vals = dict(rho=tk.StringVar(value="1.225"), v=tk.StringVar(value=f"{max(p0.mach, 0.05) * a_snd:.1f}"),
                    c=tk.StringVar(value="1.0"), b=tk.StringVar(value="1.0"))
        for j, (k, t) in enumerate((("rho", "ρ, кг/м³"), ("v", "V∞, м/с"), ("c", "хорда c, м"), ("b", "размах b, м"))):
            ttk.Label(fr, text=t).grid(row=1, column=2 * j, sticky="e", padx=(6, 2))
            ttk.Entry(fr, textvariable=vals[k], width=8).grid(row=1, column=2 * j + 1, sticky="w")
        ttk.Label(fr, text="хорда c — физический размер опорной хорды расчёта (её значение в координатах — "
                  "в «Примечаниях»); V∞ по умолчанию — M·340,3 м/с. Для силы на 1 м размаха b = 1.",
                  foreground="#555", wraplength=700).grid(row=2, column=0, columnspan=8, sticky="w", pady=(2, 4))
        txt = tk.Text(fr, width=110, height=22, font="TkFixedFont")
        txt.grid(row=4, column=0, columnspan=8, sticky="nsew")
        out = {}

        def calc(*_):
            p = next(q for q in self.polars if q.label == v_p.get())
            try:
                rho, v, c, b = (float(vals[k].get().replace(",", ".")) for k in ("rho", "v", "c", "b"))
            except ValueError:
                messagebox.showerror("Силы", "Проверьте числа", parent=w)
                return
            f = core.dimensional_forces(p, rho, v, c, b)
            out["p"], out["f"] = p, f
            lines = [f"{p.label}",
                     f"q∞ = ρV²/2 = {f['q']:.1f} Па; S = c·b = {f['S']:.4g} м²; опорная хорда расчёта "
                     f"c = {p.chord_ref:.4g} (в координатах профиля), момент — относительно "
                     f"({p.xref[0]:.4g}; {p.xref[1]:.4g}).",
                     "L — подъёмная (⊥ потоку), D — сопротивление (вдоль потока), X, Y — по осям профиля; "
                     "M > 0 — на кабрирование.", "",
                     f"{'α, °':>7} {'Cl':>8} {'Cd':>9} {'Cm':>8} {'L, Н':>11} {'D, Н':>10} {'M, Н·м':>10} "
                     f"{'X, Н':>10} {'Y, Н':>11}"]
            for i, a in enumerate(p.alpha):
                lines.append(f"{a:7.2f} {p.cl[i]:8.4f} {p.cd[i]:9.5f} {p.cm[i]:8.4f} {f['L'][i]:11.2f} "
                             f"{f['D'][i]:10.3f} {f['M'][i]:10.3f} {f['X'][i]:10.3f} {f['Y'][i]:11.2f}")
            if "elem_L" in f:
                lines += ["", "По элементам (L, Н; D, Н — где рассчитано):"]
                for i, a in enumerate(p.alpha):
                    eL, eD = f["elem_L"][i], f["elem_D"][i]
                    if eL is None:
                        continue
                    parts = [f"эл. {j + 1}: L {l_:.2f}" + (f", D {d_:.3f}" if np.isfinite(d_) else "")
                             for j, (l_, d_) in enumerate(zip(eL, eD))]
                    lines.append(f"{a:7.2f}°  " + ";  ".join(parts))
            txt.delete("1.0", "end")
            txt.insert("end", "\n".join(lines))

        def save():
            if "f" not in out:
                return
            path = filedialog.asksaveasfilename(parent=w, defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            import csv
            p, f = out["p"], out["f"]
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                wr = csv.writer(fh, delimiter=";")
                wr.writerow(["alpha", "Cl", "Cd", "Cm", "L_N", "D_N", "M_Nm", "X_N", "Y_N"])
                for i, a in enumerate(p.alpha):
                    wr.writerow([f"{a:g}"] + [f"{v_:.6g}" for v_ in (p.cl[i], p.cd[i], p.cm[i], f["L"][i], f["D"][i],
                                                                        f["M"][i], f["X"][i], f["Y"][i])])
        bb = ttk.Frame(fr)
        bb.grid(row=3, column=0, columnspan=8, sticky="w")
        ttk.Button(bb, text="Рассчитать", command=calc).pack(side="left")
        ttk.Button(bb, text="Сохранить CSV…", command=save).pack(side="left", padx=6)
        calc()

    # ------------------------------------------------------------ анимация (GIF)
    def animate_dialog(self):
        """Нестационарный RANS для выбранного угла поляры RANS и GIF поля течения."""
        if getattr(self, "_anim_running", False) or (self.worker and self.worker.is_alive()):
            messagebox.showinfo("Анимация", "Дождитесь окончания текущего расчёта.")
            return
        try:
            a = float(self.cp_alpha.get())
        except ValueError:
            a = None
        cand = [p for p in self.polars if p.method == "rans" and p.field_ctx]
        sel = [p for p in cand if p.label == self.fl_polar.get()] or cand[-1:]
        if not sel:
            messagebox.showinfo("Анимация", "Анимация показывает развитие течения во времени, а это есть "
                                "только в нестационарном расчёте RANS.\n\nВязкий метод (панели + пограничный "
                                "слой) стационарный: поле течения для любого α — кнопка «Построить».\n\n"
                                "Для ролика: выберите метод RANS, посчитайте точку или поляру, затем снова "
                                "нажмите эту кнопку.", parent=self)
            return
        p = sel[-1]
        if a is None or not np.isclose(p.alpha, a).any():
            a = float(p.alpha[0])
        k = int(np.where(np.isclose(p.alpha, a))[0][0])
        from .animate import QUANTITIES
        w = tk.Toplevel(self)
        w.title("Анимация поля течения по времени")
        w.transient(self)
        fr = ttk.Frame(w, padding=10)
        fr.pack(fill="both", expand=True)
        ttk.Label(fr, text=f"{p.label}, α = {a:g}°", font="BoldFont").grid(row=0, column=0, columnspan=3,
                                                                                sticky="w", pady=(0, 6))
        v_t = tk.StringVar(value="4")
        v_n = tk.StringVar(value="40")
        v_sub = tk.StringVar(value="12")
        v_fe = tk.StringVar(value="2")
        v_start = tk.StringVar(value="steady")
        v_q = tk.StringVar(value=list(QUANTITIES)[0])
        v_reg = tk.StringVar(value="Весь профиль и след")
        v_sl = tk.BooleanVar(value=False)
        regs = {"Весь профиль и след": (-0.5, 2.5, -0.9, 0.9), "Профиль крупно": (-0.3, 1.5, -0.55, 0.55),
                "Задняя кромка и след": (0.6, 2.0, -0.45, 0.45)}
        rows = [("Время расчёта t·V∞/c (хорд пройдено потоком)", v_t,
                 "срыв вихрей за Cl_max — несколько периодов (~2–6); разгонный вихрь — 1–3"),
                ("Шагов по времени на одну хорду", v_n, "40 — обычно; 80–100 — мелкие вихри точнее"),
                ("Внутренних итераций Ньютона на шаг (не больше)", v_sub, "останавливаются раньше при падении невязки в 100 раз"),
                ("Кадр каждые N шагов", v_fe, "меньше — плавнее и тяжелее GIF")]
        for i, (t_, v_, h_) in enumerate(rows, start=1):
            ttk.Label(fr, text=t_).grid(row=i, column=0, sticky="w")
            ttk.Entry(fr, textvariable=v_, width=8).grid(row=i, column=1, sticky="w", padx=4)
            ttk.Label(fr, text=h_, foreground="#666").grid(row=i, column=2, sticky="w")
        i = len(rows) + 1
        others = [f"{float(x_):g}" for j_, x_ in enumerate(p.alpha)
                  if j_ != k and p.field_ctx["states"][j_] is not None]
        v_from = tk.StringVar(value=(min(others, key=lambda t_: abs(float(t_) - a)) if others else ""))
        ttk.Label(fr, text="Начало").grid(row=i, column=0, sticky="w")
        ttk.Radiobutton(fr, text="с установившегося решения этого угла (устойчиво ли течение)",
                        variable=v_start, value="steady").grid(row=i, column=1, columnspan=2, sticky="w")
        rf = ttk.Frame(fr)
        rf.grid(row=i + 1, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(rf, text="резкий поворот потока: начальное поле — решение при α =", variable=v_start,
                        value="pitch", state=("normal" if others else "disabled")).pack(side="left")
        ttk.Combobox(rf, textvariable=v_from, values=others, width=6, state="readonly").pack(side="left")
        ttk.Label(rf, text="°").pack(side="left")
        ttk.Radiobutton(fr, text="разгон из покоя (поток набирает скорость за 0,5 хорды пути — разгонный вихрь)",
                        variable=v_start, value="start").grid(row=i + 2, column=1, columnspan=2, sticky="w")
        i += 1
        ttk.Label(fr, text="Величина").grid(row=i + 2, column=0, sticky="w")
        ttk.Combobox(fr, textvariable=v_q, values=list(QUANTITIES), state="readonly",
                     width=28).grid(row=i + 2, column=1, columnspan=2, sticky="w")
        ttk.Label(fr, text="Область").grid(row=i + 3, column=0, sticky="w")
        ttk.Combobox(fr, textvariable=v_reg, values=list(regs), state="readonly",
                     width=28).grid(row=i + 3, column=1, columnspan=2, sticky="w")
        scales = {"логарифмическая — видны и слой у стенки, и слабый след, и вихри": "log",
                  "линейная": "linear"}
        v_sc = tk.StringVar(value=next(iter(scales)))
        ttk.Label(fr, text="Шкала цвета").grid(row=i + 4, column=0, sticky="w")
        cb_sc = ttk.Combobox(fr, textvariable=v_sc, values=list(scales), state="readonly", width=56)
        cb_sc.grid(row=i + 4, column=1, columnspan=2, sticky="w")
        Tooltip(cb_sc, "Завихренность у стенки в десятки–сотни раз больше, чем в следе и вихрях: в линейной "
                       "шкале видна только тонкая полоса у профиля. Логарифмическая — для завихренности и ν̃.")
        i += 1
        ttk.Checkbutton(fr, text="линии тока на кадрах (медленнее)", variable=v_sl).grid(
            row=i + 4, column=1, columnspan=2, sticky="w")
        ttk.Label(fr, text="Расчёт по времени — та же сетка и модель, что в поляре; на ЦП это минуты–десятки "
                  "минут (шагов = время × шагов на хорду).", foreground="#666", wraplength=560).grid(
            row=i + 5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        def go():
            try:
                t_end = float(v_t.get().replace(",", "."))
                nsc = int(float(v_n.get()))
                nsub = int(float(v_sub.get()))
                fe = int(float(v_fe.get()))
                if t_end <= 0 or nsc < 2 or nsub < 1 or fe < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Анимация", "Проверьте числа: время > 0, шагов на хорду ≥ 2, итераций ≥ 1.",
                                     parent=w)
                return
            path = filedialog.asksaveasfilename(parent=w, title="Сохранить анимацию", defaultextension=".gif",
                                                initialfile=f"aero2d_a{a:g}.gif", filetypes=[("GIF", "*.gif")])
            if not path:
                return
            k_from = None
            if v_start.get() == "pitch":
                k_from = int(np.where(np.isclose(p.alpha, float(v_from.get())))[0][0])
            w.destroy()
            self._run_animation(p, k, t_end, nsc, nsub, fe, v_start.get(), QUANTITIES[v_q.get()],
                                regs[v_reg.get()], v_sl.get(), path, k_from, scales[v_sc.get()])
        b = ttk.Frame(fr)
        b.grid(row=i + 6, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(b, text="Рассчитать и сохранить GIF", command=go).pack(side="right")
        ttk.Button(b, text="Отмена", command=w.destroy).pack(side="right", padx=6)

    def _run_animation(self, p, k, t_end, nsc, nsub, fe, start, qty, box, streamlines, path, k_from=None,
                       scale="log"):
        from .rans import RansControl
        self._anim_running = True
        self.stop_flag.clear()
        self.btn_stop.configure(state="normal")
        self._rans_ctl = RansControl()
        sopts = {}
        try:
            sopts = self.solver_settings()
        except Exception:                                   # noqa: BLE001
            pass

        def job():
            try:
                def cb(text, frac):
                    self.q.put(("progress", text, 0.9 * frac))
                    return self.stop_flag.is_set()
                run = core.run_rans_unsteady(p.field_ctx, k, t_end, nsc, nsub, start, fe, cb,
                                             control=self._rans_ctl, solver_opts=sopts, k_from=k_from)
                from .animate import make_gif
                self.q.put(("progress", "Сборка GIF…", 0.9))
                nm = p.label.split(",")[0]
                ttl = f"{nm}, α = {run['alpha']:g}°"
                if run.get("alpha_from") is not None:
                    ttl = f"{nm}, α {run['alpha_from']:g}° → {run['alpha']:g}°"
                n = make_gif(run["mesh"], run["elements"], run, path, quantity=qty, box=box,
                             streamlines=streamlines, title=ttl, scale=scale,
                             progress=lambda i, n_: self.q.put(("progress", f"Сборка GIF: кадр {i}/{n_}",
                                                                0.9 + 0.1 * i / max(n_, 1))) and False)
                self._anim_last = run
                self.q.put(("anim_done", path, f"Анимация сохранена: {path} ({n} кадров, "
                                               f"{len(run['t'])} шагов, {run['status']})"))
            except Exception as e:  # noqa: BLE001
                self.q.put(("anim_error", f"{e}\n\n{traceback.format_exc(limit=3)}"))
        threading.Thread(target=job, daemon=True).start()

    def _show_gif(self, path):
        """Просмотр GIF в окне программы (Tk показывает кадры GIF сам)."""
        try:
            frames = []
            i = 0
            while True:
                try:
                    frames.append(tk.PhotoImage(file=path, format=f"gif -index {i}"))
                except tk.TclError:
                    break
                i += 1
        except Exception:                                   # noqa: BLE001
            return
        if not frames:
            return
        w = tk.Toplevel(self)
        w.title(os.path.basename(path))
        lab = ttk.Label(w, image=frames[0])
        lab.pack()
        w._frames = frames

        def step(j=0):
            if not w.winfo_exists():
                return
            lab.configure(image=frames[j % len(frames)])
            w.after(90, step, j + 1)
        step()

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
        ("bl_thickness", "толщина слоя четырёхуг., c", "0", "0 — авто: 0,01c, но ≤ 0,3 наименьшей щели"),
        ("n_surface", "точек на сторону (осн. эл.)", "0", "0 — как в контуре; у закрылков — пропорц. хорде"),
        ("near_size", "ячейки у стенок и в щелях, c", "0", "0 — авто: 0,004c (профиль), 0,003c (механизация)"),
        ("growth", "рост треугольников", "0.25", "размер = на краю слоя + доля × расстояние от края слоя"),
        ("wake_size", "ячейки по следу, c", "0.008", "сгущение по линиям следа за задними кромками"),
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
        ttk.Label(side, text="Сетка для RANS (gmsh)", font="BoldFont").grid(row=0, column=0, columnspan=2,
                                                                                 sticky="w")
        self.mesh_vars = {}
        for r, (key, lbl, val, tip) in enumerate(self.MESH_FIELDS, start=1):
            ttk.Label(side, text=lbl).grid(row=2 * r - 1, column=0, sticky="w", pady=(3, 0))
            v = tk.StringVar(value=val)
            ttk.Entry(side, textvariable=v, width=8).grid(row=2 * r - 1, column=1, sticky="w", pady=(3, 0))
            if tip:
                ttk.Label(side, text=tip, foreground="#555", font="SmallFont", wraplength=230).grid(
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
        from .rans_mesh import QUALITY_METRICS
        self._qm_names = {"нет (только сетка)": None}
        self._qm_names.update({v[0]: k for k, v in QUALITY_METRICS.items()})
        ttk.Label(bar, text="   качество (как в ICEM): ").pack(side="left")
        self.mesh_qm = tk.StringVar(value="нет (только сетка)")
        cq_ = ttk.Combobox(bar, textvariable=self.mesh_qm, values=list(self._qm_names), state="readonly", width=20)
        cq_.pack(side="left")
        cq_.bind("<<ComboboxSelected>>", lambda _: self.draw_mesh())
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
        # угол атаки в подпись не входит: сетка от него не зависит (MeshOptions.wake_alphas)
        return (len(xy), round(float(xy.sum()), 9), round(float((xy ** 2).sum()), 9), float(re), repr(opts))

    def _mesh_inputs(self):
        af = self.get_config()
        mach, alphas, niter, re_, ncrit, xtr = self._params()
        opts = self.mesh_opts()
        chord = self._cref(1.0 if isinstance(af, list) and self.src.get() != "mfile" else None)
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
        from .rans_mesh import quality, cell_quality, quality_report
        self._mesh = (sig, m)
        q = quality(m)
        self._mesh_cq = cell_quality(m)
        wc = m.face_cells[m.face_tag == 1, 0]
        h1 = float(np.median(2 * m.wall_dist[wc]))
        cf = 0.026 * re ** (-1 / 7)
        yp = h1 / chord * re * np.sqrt(cf / 2)
        self.mesh_info.set(
            f"Ячеек: {q['cells']:,} (четырёхугольников {q['quads']:,}); граней на стенках {q['wall_faces']}.\n"
            .replace(",", " ") +
            f"Высота первой ячейки ≈ {h1:.2e} c, оценка y⁺ ≈ {yp:.2f} (по Cf пластины).\n"
            f"Наибольшее отношение объёмов соседних ячеек — {q['vol_ratio_max']:.0f}.\n"
            "Эта сетка будет использована в расчёте RANS.\n\nКачество:\n" + quality_report(m, self._mesh_cq) +
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
        key = self._qm_names.get(self.mesh_qm.get())
        cq = getattr(self, "_mesh_cq", None)
        for ax, box in ((ax1, None), (ax2, REGIONS[self.mesh_reg.get()])):
            if key and cq is not None:
                from .rans_mesh import QUALITY_METRICS
                v = cq[key]
                # «хорошо» — светлое: для det, quality, angle малое плохо; для vol_ratio, nonortho — большое
                if key in ("det", "quality"):
                    norm, cmap, cv = mpl_colors.Normalize(0.0, 1.0), "RdYlGn", v
                elif key == "angle":
                    norm, cmap, cv = mpl_colors.Normalize(0.0, 90.0), "RdYlGn", v
                elif key == "vol_ratio":
                    norm, cmap, cv = mpl_colors.LogNorm(1.0, max(10.0, float(v.max()))), "RdYlGn_r", v
                else:
                    norm, cmap, cv = mpl_colors.Normalize(0.0, 90.0), "RdYlGn_r", v
                pc = PolyCollection(polys, array=cv, cmap=cmap, norm=norm, edgecolor="#333", lw=0.1)
                ax.add_collection(pc)
                if box is None:
                    fig.colorbar(pc, ax=ax, shrink=0.8, label=QUALITY_METRICS[key][0])
                worst = np.argsort(v if key in ("det", "quality", "angle") else -v)[:10]
                ax.plot(m.centroid[worst, 0], m.centroid[worst, 1], "o", mfc="none", mec="#0050ff", ms=9,
                        mew=1.4, zorder=5, label="10 худших ячеек")
                if box is not None:
                    ax.legend(fontsize=7, loc="upper right")
            else:
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
        self.t_res_frame = fr
        bar = ttk.Frame(fr, padding=(4, 4))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Переход к следующему углу: невязка ниже 10^−").pack(side="left")
        self.cv_orders = tk.StringVar(value="4.5")
        ttk.Entry(bar, textvariable=self.cv_orders, width=4).pack(side="left")
        ttk.Label(bar, text=" от невязки однородного потока,").pack(side="left")
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
        ttk.Checkbutton(bar2, text="принимать и при «застрявшей» невязке (на порядок выше заданной, Cl и Cd стоят "
                        "≥ 30 итераций)", variable=self.cv_stall).pack(side="left")
        ttk.Button(bar2, text="⏭ Принять угол, следующий", command=self.skip_angle).pack(side="right")
        # шаг по псевдовремени и защита от развала
        bar4 = ttk.Frame(fr, padding=(4, 2))
        bar4.pack(side="top", fill="x")
        ttk.Label(bar4, text="Шаг: режим").pack(side="left")
        self.cfl_preset = tk.StringVar(value="обычный")
        self.cfl_mode_v = "grow"
        cp_ = ttk.Combobox(bar4, textvariable=self.cfl_preset, values=list(self.CFL_PRESETS) + ["свой"],
                           state="readonly", width=22)
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
        ttk.Label(bar5, text=" раз;  ограничитель:").pack(side="left")
        self.lim_zone = tk.StringVar(value="у задних кромок")
        ttk.Combobox(bar5, textvariable=self.lim_zone, values=list(self.LIMITER_ZONES), state="readonly",
                     width=16).pack(side="left")
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

    # режимы шага: "обычный" — CFL растёт, пока шаги Ньютона удаются (адаптивный, как в SU2);
    # «как Fluent» — CFL растёт до 200 и дальше постоянный (число Куранта coupled-решателя
    # Fluent: по умолчанию 200, при плохой сходимости советуют 50–100); «осторожный» — медленно
    CFL_PRESETS = {"обычный": dict(cfl_start="5", cfl_limit="1e6", cfl_growth="1.25", first_order="30",
                                   rollback_factor="100", mode="grow"),
                   "как Fluent (Courant 200)": dict(cfl_start="5", cfl_limit="200", cfl_growth="1.25",
                                                    first_order="30", rollback_factor="100", mode="fixed"),
                   "осторожный": dict(cfl_start="2", cfl_limit="1e4", cfl_growth="1.1", first_order="60",
                                      rollback_factor="20", mode="grow")}

    def _apply_cfl_preset(self):
        p = self.CFL_PRESETS.get(self.cfl_preset.get())
        if p:
            for k, v in p.items():
                if k == "mode":
                    self.cfl_mode_v = v
                else:
                    self.cfl_vars[k].set(v)

    # зона ограничителя Венкатакришнана (RansSolver.limiter_zone)
    LIMITER_ZONES = {"у задних кромок": "te", "везде (как раньше)": "all", "нет": "off"}

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
        o["cfl_mode"] = getattr(self, "cfl_mode_v", "grow")
        o["prec_fill"] = int(self.prec_fill.get())
        o["limiter_zone"] = self.LIMITER_ZONES.get(self.lim_zone.get(), "te")
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
        if d.get("r_fs"):
            L.setdefault("r_fs", {})[float(d["alpha"])] = float(d["r_fs"])
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
                    if H.get("r_fs"):
                        D.setdefault("r_fs", {})[float(a)] = float(H["r_fs"])
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
                    transform=ax.transAxes, color="#555")
            ax.set_axis_off()
            self.res_canvas.draw_idle()
            return
        angs = sorted(D["angles"])
        vals = ["все углы"] + [f"{a:g}" for a in angs]
        self.cb_res_angle["values"] = vals
        if self.res_angle.get() not in vals:
            self.res_angle.set("все углы")
        sel = self.res_angle.get()
        try:
            orders = self.conv_settings().orders
        except ValueError:
            orders = None
        rfs = D.get("r_fs", {})
        names = ["неразрывность", "импульс x", "импульс y", "энергия", "турбулентность ν̃", "переход ñ", "перемежаемость γ"]
        import matplotlib as mpl
        cmap = mpl.colormaps["turbo"]
        show = angs if sel == "все углы" else [float(sel)]
        neq = max(np.array(D["angles"][a]["res"]).shape[1] for a in show)
        # невязки по уравнениям — каждая на своём графике (у каждого уравнения свой масштаб)
        nc_ = 3 if neq <= 5 else 4
        gs = fig.add_gridspec(3, nc_, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.3)
        axs = [fig.add_subplot(gs[r_, c_]) for r_ in range(2) for c_ in range(nc_)]
        ax2 = fig.add_subplot(gs[2, :])
        for k, a in enumerate(show):
            A = D["angles"][a]
            R = np.array(A["res"])
            col = cmap(0.1 + 0.8 * k / max(1, len(show) - 1)) if len(show) > 1 else "#1f77b4"
            for j in range(R.shape[1]):
                axs[j].semilogy(A["it"], R[:, j], "-", color=col, lw=1.0, label=f"α = {a:g}°" if j == 0 else None)
            axs[neq].semilogy(A["it"], A["rn"], "-", color=col, lw=1.0)
            ax2.plot(A["it"], A["cl"], "-", color=col, lw=1.2)
            if orders is not None and a in rfs:
                axs[neq].axhline(10.0 ** (-orders) * rfs[a], color=col, ls="--", lw=0.9)
        for j in range(neq):
            axs[j].set_title(names[j], fontsize=8)
        axs[neq].set_title("сумма" + (f"; пунктир — порог 10^−{orders:g}" if orders is not None else ""), fontsize=8)
        for ax_ in axs[neq + 1:]:
            ax_.set_axis_off()
        for ax_ in axs[:neq + 1]:
            ax_.grid(alpha=0.3, which="both")
            ax_.tick_params(labelsize=7)
        if len(show) > 1:
            axs[0].legend(fontsize=6, ncol=2, loc="upper right")
        cls = [A["cl"][len(A["cl"]) // 3:] for a, A in D["angles"].items() if a in show and A["cl"]]
        allc = np.concatenate([np.asarray(c_, float) for c_ in cls if len(c_)]) if cls else np.array([])
        if allc.size:
            lo, hi = np.nanmin(allc), np.nanmax(allc)
            ax2.set_ylim(lo - 0.1 * (hi - lo) - 0.01, hi + 0.1 * (hi - lo) + 0.01)
        ax2.set_ylabel("Cl")
        ax1 = axs[neq]
        evs = [e for e in D.get("events", []) if sel == "все углы" or abs(e[0] - float(sel)) < 1e-9]
        for k_, (a_, it_, txt) in enumerate(evs):
            ax1.axvline(it_, color="#ff7f0e", lw=0.9, ls="-.", label="откат / ручное действие" if k_ == 0 else None)
        if evs:
            ax1.text(0.01, 0.02, f"последнее: α = {evs[-1][0]:g}°, итер. {evs[-1][1]}: {evs[-1][2]}",
                     transform=ax1.transAxes, fontsize=7, color="#b35900")
        if evs:
            ax1.legend(fontsize=6, loc="upper right")
        if sel != "все углы":                      # один угол — ещё Cd на второй оси
            A = D["angles"][float(sel)]
            ax3 = ax2.twinx()
            ax3.plot(A["it"], A["cd"], "-", color="#d62728", lw=1.0)
            ax3.set_ylabel("Cd", color="#d62728")
            v_ = A["cd"][len(A["cd"]) // 3:]
            if len(v_):
                lo, hi = float(np.nanmin(v_)), float(np.nanmax(v_))
                ax3.set_ylim(lo - 0.1 * (hi - lo) - 1e-4, hi + 0.1 * (hi - lo) + 1e-4)
        ax2.grid(alpha=0.3)
        ax2.set_xlabel("итерация Ньютона на угле")
        if D["last"]:
            a, it, cl, cd = D["last"]
            fig.suptitle(f"Невязки по уравнениям (ср. кв. по ячейкам). Последнее: α = {a:g}°, итерация {it}: "
                         f"Cl = {cl:.4f}, Cd = {cd:.5f}", fontsize=9)
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
                self.q.put(("error", str(e) or type(e).__name__, traceback.format_exc()))
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
