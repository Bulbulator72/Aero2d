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
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        # --- профиль
        f = ttk.LabelFrame(left, text="Профиль", padding=6)
        f.pack(fill="x", pady=4)
        self.src = tk.StringVar(value="naca")
        ttk.Radiobutton(f, text="NACA, 4 цифры:", variable=self.src, value="naca",
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
                        command=self.preview_airfoil).grid(row=4, column=0, columnspan=2, sticky="w")
        # --- механизация из одного профиля
        self.flap_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Щелевой закрылок (из профиля NACA/файла)", variable=self.flap_on,
                        command=self.preview_airfoil).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.flap_x, self.flap_d = tk.StringVar(value="0.75"), tk.StringVar(value="20")
        self.flap_gap, self.flap_ov = tk.StringVar(value="0.015"), tk.StringVar(value="0")
        ff = ttk.Frame(f)
        ff.grid(row=6, column=0, columnspan=2, sticky="w")
        for k, (lbl, var) in enumerate([("срез x/c", self.flap_x), ("δ, °", self.flap_d),
                                        ("зазор", self.flap_gap), ("перекр.", self.flap_ov)]):
            ttk.Label(ff, text=("   " if k % 2 == 0 else " ") + lbl).grid(row=k // 2, column=2 * (k % 2), sticky="w")
            e = ttk.Entry(ff, textvariable=var, width=6)
            e.grid(row=k // 2, column=2 * (k % 2) + 1, sticky="w")
            e.bind("<Return>", lambda _: self.preview_airfoil())

        # --- режим
        f = ttk.LabelFrame(left, text="Режим", padding=6)
        f.pack(fill="x", pady=4)
        self.mach = tk.StringVar(value="0")
        self.a0, self.a1, self.da = tk.StringVar(value="-4"), tk.StringVar(value="14"), tk.StringVar(value="1")
        for r, (lbl, var) in enumerate([("Число Маха M", self.mach), ("α от, °", self.a0),
                                        ("α до, °", self.a1), ("шаг α, °", self.da)]):
            ttk.Label(f, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(f, textvariable=var, width=8).grid(row=r, column=1, sticky="w", pady=1)

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
        for r, (lbl, var) in enumerate([("   Re", self.re), ("   Ñкрит", self.ncrit),
                                        ("   переход верх ≤ x/c", self.xtru),
                                        ("   переход низ ≤ x/c", self.xtrl)], start=1):
            ttk.Label(f, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(f, textvariable=var, width=8).grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(f, text="Невязкий: панели (мгновенно, без Cd)",
                        variable=self.method, value="panel").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(f, text="Эйлер: сжимаемый, скачки (M ≥ 0,2)",
                        variable=self.method, value="euler").grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(f, text="   Сетка:").grid(row=7, column=0, sticky="w")
        self.grid_name = tk.StringVar(value=list(core.GRIDS)[0])
        ttk.Combobox(f, textvariable=self.grid_name, values=list(core.GRIDS), state="readonly",
                     width=16).grid(row=7, column=1, sticky="w")
        ttk.Label(f, text="   Итераций:").grid(row=8, column=0, sticky="w")
        self.niter = tk.StringVar(value="100")
        ttk.Entry(f, textvariable=self.niter, width=8).grid(row=8, column=1, sticky="w")

        # --- кнопки
        f = ttk.Frame(left, padding=(0, 6))
        f.pack(fill="x")
        self.btn_run = ttk.Button(f, text="▶ Рассчитать", command=self.run)
        self.btn_run.pack(fill="x", pady=2)
        self.btn_stop = ttk.Button(f, text="■ Остановить", command=self.stop, state="disabled")
        self.btn_stop.pack(fill="x", pady=2)
        ttk.Button(f, text="Очистить графики", command=self.clear).pack(fill="x", pady=2)
        ttk.Button(f, text="Сохранить таблицу (CSV)…", command=self.save_csv).pack(fill="x", pady=2)
        ttk.Button(f, text="Сохранить графики (PNG)…", command=self.save_png).pack(fill="x", pady=2)

        self.progress = ttk.Progressbar(left, mode="determinate", maximum=1.0)
        self.progress.pack(fill="x", pady=(8, 2))
        self.status = tk.StringVar(value="Готово. Выберите профиль и нажмите «Рассчитать».")
        ttk.Label(left, textvariable=self.status, wraplength=250, justify="left").pack(fill="x")

        # --- вкладки
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.t_geo = PlotTab(nb, "Профиль")
        self.t_cl = PlotTab(nb, "Cl(α)")
        self.t_cd = PlotTab(nb, "Cd(α), Cl(Cd)")
        self.t_cm = PlotTab(nb, "Cm(α)")
        self.t_cp = PlotTab(nb, "Распределение Cp")
        bar = ttk.Frame(self.t_cp.frame)
        bar.pack(side="top", fill="x", before=self.t_cp.canvas.get_tk_widget())
        ttk.Label(bar, text="  α, °: ").pack(side="left")
        self.cp_alpha = tk.StringVar()
        self.cb_alpha = ttk.Combobox(bar, textvariable=self.cp_alpha, width=8, state="readonly")
        self.cb_alpha.pack(side="left")
        self.cb_alpha.bind("<<ComboboxSelected>>", lambda _: self.plot_cp())
        self.t_bl = PlotTab(nb, "Пограничный слой")
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
        ttk.Label(bar, text="  α, °: ").pack(side="left")
        self.cb_alpha_fl = ttk.Combobox(bar, textvariable=self.cp_alpha, width=8, state="readonly")
        self.cb_alpha_fl.pack(side="left")
        ttk.Label(bar, text="   величина: ").pack(side="left")
        self.fl_qty = tk.StringVar(value="Давление Cp")
        ttk.Combobox(bar, textvariable=self.fl_qty, values=["Давление Cp", "Скорость |V|/V∞"],
                     state="readonly", width=16).pack(side="left")
        ttk.Label(bar, text="   область: ").pack(side="left")
        from .field import REGIONS
        self.fl_reg = tk.StringVar(value=list(REGIONS)[0])
        ttk.Combobox(bar, textvariable=self.fl_reg, values=list(REGIONS), state="readonly",
                     width=20).pack(side="left")
        self.fl_lines = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="линии тока", variable=self.fl_lines).pack(side="left", padx=6)
        ttk.Button(bar, text="Построить", command=self.plot_field).pack(side="left", padx=6)
        self._field_cache = {}
        # таблица
        tf = ttk.Frame(nb)
        nb.add(tf, text="Таблица")
        cols = ("calc", "alpha", "cl", "cd", "cdf", "cm", "xu", "xl", "ok")
        self.table = ttk.Treeview(tf, columns=cols, show="headings")
        for c, t, w in zip(cols, ["Расчёт", "α, °", "Cl", "Cd", "Cd трения", "Cm", "Переход верх",
                                  "Переход низ", "Надёжно"], [330, 50, 75, 80, 80, 75, 85, 85, 65]):
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
        if self.flap_on.get():
            from .multi import split_flap
            return split_flap(af, num(self.flap_x), num(self.flap_d), num(self.flap_gap), num(self.flap_ov))
        return af

    def get_airfoil(self):
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
            re_ = num(self.re)
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
        if isinstance(af, list):
            if method != "panel":
                messagebox.showinfo("Механизация", "Для многоэлементного профиля пока доступен только "
                                    "невязкий панельный расчёт (распределение давления и нагрузки "
                                    "по элементам). Вязкий расчёт механизации — следующий этап.")
            method = "multi"
        self.stop_flag.clear()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal" if method not in ("panel", "multi") else "disabled")
        self.progress["value"] = 0
        self.status.set("Расчёт…")

        def job():
            try:
                def cb(text, frac):
                    self.q.put(("progress", text, frac))
                    return self.stop_flag.is_set()
                if method == "multi":
                    nm = "закрылок δ=" + self.flap_d.get() + "°" if self.src.get() != "mfile" else \
                        os.path.basename(self.mfile.get())
                    p = core.run_multi(af, alphas, mach, nm, chord=1.0 if self.src.get() != "mfile" else None)
                elif method == "panel":
                    p = core.run_panel(af, alphas, mach)
                elif method == "viscous":
                    p = core.run_viscous(af, alphas, mach, re_, ncrit, xtr, cb)
                else:
                    p = core.run_euler(af, alphas, mach, grid, niter, cb)
                    if self.stop_flag.is_set():
                        p.label += " (остановлен)"
                self.q.put(("done", p))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", f"{e}\n\n{traceback.format_exc(limit=3)}"))

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag.set()
        self.status.set("Остановка после текущей итерации…")

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    self.status.set(msg[1])
                    self.progress["value"] = msg[2]
                elif msg[0] == "done":
                    self._finish()
                    self.add_polar(msg[1])
                elif msg[0] == "field":
                    self._draw_field(*msg[1:])
                elif msg[0] == "field_error":
                    self.status.set("Поле течения: " + msg[1])
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
                f"{p.cm[k]:.4f}", fmt(p.xtr_u, k, ".3f"), fmt(p.xtr_l, k, ".3f"),
                "да" if p.ok[k] else "НЕТ"))
        alphas = sorted({f"{a:g}" for q in self.polars for a in q.alpha}, key=float)
        self.cb_alpha["values"] = alphas
        self.cb_alpha_bl["values"] = alphas
        self.cb_alpha_fl["values"] = alphas
        if not self.cp_alpha.get() or self.cp_alpha.get() not in alphas:
            mid = p.alpha[np.argmin(np.abs(p.alpha - 4))]
            self.cp_alpha.set(f"{mid:g}")
        self.replot()
        self.nb.select(1)

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
        axm = self.t_cm.fig.add_subplot(111)
        axd1 = self.t_cd.fig.add_subplot(121)
        axd2 = self.t_cd.fig.add_subplot(122)
        any_cd = False
        for i, p in enumerate(self.polars):
            c = COLORS[i % len(COLORS)]
            self._series(axl, p.alpha, p.cl, p.ok, c, p.label)
            self._series(axm, p.alpha, p.cm, p.ok, c, p.label)
            if not np.all(np.isnan(p.cd)):
                any_cd = True
                self._series(axd1, p.alpha, p.cd, p.ok, c, p.label)
                self._series(axd2, p.cd, p.cl, p.ok, c, p.label)
        self._style(axl, "α, °", "Cl")
        self._style(axm, "α, °", "Cm (относительно 1/4 хорды)")
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
        if not cand:
            messagebox.showinfo("Поле течения", "Поле строится для вязкого и невязкого (панельного) "
                                "расчёта. Для метода Эйлера — в следующей версии.")
            return
        p = cand[-1]
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

    def _draw_field(self, label, f, key):
        self._field_cache[key] = f
        fig = self.t_fl.fig
        fig.clear()
        ax = fig.add_subplot(111)
        X, Y = f["X"], f["Y"]
        if self.fl_qty.get().startswith("Давление"):
            z, cmap, name = f["cp"], "RdBu_r", "Cp"
            lo = max(np.nanpercentile(z, 1), -6.0)
            levels = np.linspace(lo, 1.0, 41)
        else:
            z, cmap, name = f["speed"], "viridis", "|V| / V∞"
            levels = np.linspace(np.nanpercentile(z, 0.5), min(np.nanpercentile(z, 99.5), 3.0), 41)
        c = ax.contourf(X, Y, z, levels=levels, cmap=cmap, extend="both")
        fig.colorbar(c, ax=ax, label=name, shrink=0.85)
        if self.fl_lines.get():
            ys = np.linspace(Y.min(), Y.max(), 34)
            seeds = np.column_stack([np.full_like(ys, X.min() + 1e-3 * (X.max() - X.min())), ys])
            U = np.nan_to_num(f["U"]); V = np.nan_to_num(f["V"])
            ax.streamplot(X, Y, U, V, start_points=seeds, density=4, color="k", linewidth=0.55,
                          arrowsize=0.5, broken_streamlines=False)
        for bx, by in f.get("bodies", [f["body"]]):
            ax.fill(bx, by, color="#444", zorder=3)
        if "disp" in f:
            for dx, dy in f["disp"]:
                ax.plot(dx, dy, "-", color="#ffd400", lw=1.0, zorder=4)
            ax.plot(*f["wake"], "--", color="#ffd400", lw=0.9, zorder=4)
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
