"""
Aplicación de escritorio para el análisis de escenarios de XRP.
Ábrela con doble clic en "Abrir_XRP.bat" (Windows) o con:  python analysis/app_xrp.py
"""
import json, os, subprocess, sys, threading, queue
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
RES = os.path.join(OUT, "resultados_v2.json")
BT = os.path.join(OUT, "backtest_v1_vs_v2.json")
IMG_ESC = os.path.join(OUT, "xrp_escenarios_30d_v2.png")
IMG_CAL = os.path.join(OUT, "xrp_calibracion_v1_vs_v2.png")

BG, CARD, INK, MUTED, LINE = "#fcfcfb", "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"Bajista": "#e34948", "Base": "#2a78d6", "Alcista": "#1baf7a"}
FONT = "Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans"


def python_cmd():
    """Intérprete para lanzar los scripts (python.exe aunque la app corra con pythonw.exe)."""
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(cand):
            exe = cand
    return exe


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XRP · Escenarios a 30 días")
        self.configure(bg=BG)
        self.geometry("1400x860")
        self.minsize(1100, 700)
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self.q = queue.Queue()
        self.running = False
        self._style()
        self._build()
        self.load_results()
        self.after(150, self._poll)

    # ------------------------------------------------------------------ estilo
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=INK, font=(FONT, 10))
        st.configure("Card.TFrame", background=CARD, relief="flat")
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(14, 6), font=(FONT, 10))
        st.configure("Accent.TButton", font=(FONT, 10, "bold"), padding=(14, 8),
                     background="#2a78d6", foreground="#ffffff")
        st.map("Accent.TButton", background=[("active", "#1f62b3"), ("disabled", "#9bb9e3")])
        st.configure("TButton", padding=(10, 8))
        st.configure("Treeview", rowheight=24, font=(FONT, 10), background=CARD, fieldbackground=CARD)
        st.configure("Treeview.Heading", font=(FONT, 10, "bold"))
        st.configure("Horizontal.TProgressbar", background="#2a78d6")

    # ------------------------------------------------------------------ interfaz
    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(top, text="XRP / USD", font=(FONT, 20, "bold"), bg=BG, fg=INK).pack(side="left")
        self.lbl_price = tk.Label(top, text="—", font=(FONT, 20), bg=BG, fg=INK)
        self.lbl_price.pack(side="left", padx=(14, 0))
        self.lbl_date = tk.Label(top, text="", font=(FONT, 10), bg=BG, fg=MUTED)
        self.lbl_date.pack(side="left", padx=(12, 0), pady=(8, 0))

        ttk.Button(top, text="Abrir carpeta de resultados", command=self.open_folder).pack(side="right", padx=(8, 0))
        self.var_bt = tk.BooleanVar(value=not os.path.exists(BT))
        ttk.Checkbutton(top, text="Recalcular backtest (≈20 s)", variable=self.var_bt).pack(side="right", padx=8)
        self.btn = ttk.Button(top, text="⟳  Actualizar análisis", style="Accent.TButton", command=self.run_update)
        self.btn.pack(side="right")

        body = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=6, bd=0)
        body.pack(fill="both", expand=True, padx=16, pady=6)

        # Panel izquierdo: resumen
        left = tk.Frame(body, bg=BG, width=380)
        body.add(left, minsize=340)
        self._section(left, "Escenarios (probabilidad del modelo)")
        self.sc_frame = tk.Frame(left, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        self.sc_frame.pack(fill="x", pady=(0, 12))

        self._section(left, "Cono de precios")
        self.tv_cone = self._tree(left, ("p", "d7", "d30"), ("Percentil", "Día 7", "Día 30"), (90, 110, 110), 5)

        self._section(left, "Probabilidad por nivel (30 días)")
        self.tv_lv = self._tree(left, ("lv", "toca", "cierre"), ("Nivel $", "Lo toca", "Cierra por encima"),
                                (80, 110, 140), 9)

        self._section(left, "Fiabilidad del cono (backtest)")
        self.lbl_cal = tk.Label(left, text="", justify="left", anchor="w", wraplength=350,
                                bg=BG, fg=INK, font=(FONT, 9))
        self.lbl_cal.pack(fill="x")

        # Panel derecho: gráficos
        right = tk.Frame(body, bg=BG)
        body.add(right, minsize=600)
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.fig_esc, self.cv_esc = self._figure_tab(nb, "Gráfico de escenarios")
        self.fig_cal, self.cv_cal = self._figure_tab(nb, "Calibración v1 vs v2")
        logf = tk.Frame(nb, bg=BG)
        nb.add(logf, text="Registro")
        self.log = tk.Text(logf, bg=CARD, fg=INK, font=("Consolas" if sys.platform.startswith("win") else "DejaVu Sans Mono", 9),
                           relief="flat", wrap="word")
        self.log.pack(fill="both", expand=True)
        self.nb = nb

        # Barra inferior
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=16, pady=(0, 10))
        self.pb = ttk.Progressbar(bottom, mode="indeterminate", length=180)
        self.pb.pack(side="right")
        self.lbl_status = tk.Label(bottom, text="Listo.", bg=BG, fg=MUTED, font=(FONT, 9))
        self.lbl_status.pack(side="right", padx=10)
        tk.Label(bottom, text="Estimación probabilística, no una predicción ni una recomendación de inversión. "
                              "Puedes perder todo el capital.", bg=BG, fg=MUTED, font=(FONT, 9)).pack(side="left")

    def _section(self, parent, text):
        tk.Label(parent, text=text.upper(), bg=BG, fg=MUTED, font=(FONT, 9, "bold"), anchor="w").pack(fill="x", pady=(4, 4))

    def _tree(self, parent, cols, heads, widths, height):
        tv = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        for c, h, w in zip(cols, heads, widths):
            tv.heading(c, text=h)
            tv.column(c, width=w, anchor="e" if c != cols[0] else "w")
        tv.pack(fill="x", pady=(0, 12))
        return tv

    def _figure_tab(self, nb, title):
        f = tk.Frame(nb, bg=BG)
        nb.add(f, text=title)
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=BG)
        cv = FigureCanvasTkAgg(fig, master=f)
        tb = NavigationToolbar2Tk(cv, f, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        cv.get_tk_widget().pack(fill="both", expand=True)
        return fig, cv

    # ------------------------------------------------------------------ datos
    def show_image(self, fig, cv, path, empty_msg):
        fig.clear()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path))
        else:
            ax.text(0.5, 0.5, empty_msg, ha="center", va="center", fontsize=12, color=MUTED, transform=ax.transAxes)
        cv.draw_idle()

    def load_results(self):
        msg = "Aún no hay resultados.\nPulsa «Actualizar análisis»."
        self.show_image(self.fig_esc, self.cv_esc, IMG_ESC, msg)
        self.show_image(self.fig_cal, self.cv_cal, IMG_CAL, "Marca «Recalcular backtest» y pulsa Actualizar.")
        if not os.path.exists(RES):
            return
        r = json.load(open(RES, encoding="utf-8"))
        self.lbl_price.config(text=f"{r['precio']:.4f} $")
        self.lbl_date.config(text=f"cierre {r['ultima_vela']} (Bitstamp) · σ diaria prevista {r['sigma_diaria_prevista_pct']:.2f} %")

        for w in self.sc_frame.winfo_children():
            w.destroy()
        for name, s in r["escenarios"].items():
            row = tk.Frame(self.sc_frame, bg=CARD)
            row.pack(fill="x", padx=12, pady=6)
            lo, hi = s["rango"]
            rng = f"< {hi:.2f} $" if lo is None else (f"> {lo:.2f} $" if hi is None else f"{lo:.2f} – {hi:.2f} $")
            tk.Label(row, text=name, width=8, anchor="w", bg=CARD, fg=INK, font=(FONT, 11, "bold")).pack(side="left")
            bar = tk.Canvas(row, width=130, height=14, bg=CARD, highlightthickness=0)
            bar.pack(side="left", padx=6)
            bar.create_rectangle(0, 2, 130, 12, fill="#efeee9", width=0)
            bar.create_rectangle(0, 2, 130 * s["prob_pct"] / 100, 12, fill=COL.get(name, "#2a78d6"), width=0)
            tk.Label(row, text=f"{s['prob_pct']:.0f} %", width=5, anchor="e", bg=CARD, fg=INK, font=(FONT, 11, "bold")).pack(side="left")
            tk.Label(row, text=rng, anchor="e", bg=CARD, fg=MUTED, font=(FONT, 10)).pack(side="right")
        sims = r.get("simulaciones", {})
        if sims:
            txt = " · ".join(f"{k}: {v['dia30']:.2f} $" for k, v in sims.items())
            tk.Label(self.sc_frame, text=f"Ejemplos típicos día 30 → {txt}", bg=CARD, fg=MUTED,
                     font=(FONT, 9), anchor="w").pack(fill="x", padx=12, pady=(0, 8))

        self.tv_cone.delete(*self.tv_cone.get_children())
        for k in r["cono_dia30"]:
            self.tv_cone.insert("", "end", values=(k, f"{r['cono_dia7'][k]:.3f} $", f"{r['cono_dia30'][k]:.3f} $"))
        self.tv_lv.delete(*self.tv_lv.get_children())
        for lv, v in sorted(r["prob_niveles"].items(), key=lambda kv: -float(kv[0])):
            self.tv_lv.insert("", "end", values=(lv, f"{v['toca_en_30d_pct']:.0f} %", f"{v['cierre_dia30_por_encima_pct']:.0f} %"))

        if os.path.exists(BT):
            b = json.load(open(BT, encoding="utf-8"))
            per = list(k for k in b if k != "direccion_30d")
            lines = []
            for p in per:
                v1, v2 = b[p]["v1"], b[p]["v2_todo"]
                lines.append(f"{p}: dentro del rango del 90 % → v2 {v2['cobertura_90']:.0f} % (v1 {v1['cobertura_90']:.0f} %); "
                             f"por encima del P95 → v2 {v2['bandas_pct']['>P95']:.0f} % (ideal 5 %).")
            d = b.get("direccion_30d")
            if d:
                lines.append(f"Dirección a 30 días: ningún modelo probado acierta de forma fiable "
                             f"(logística {d['logistica_acierto_pct']:.0f} %, peor que la referencia en Brier).")
            self.lbl_cal.config(text="\n".join(lines))

    # ------------------------------------------------------------------ actualización
    def run_update(self):
        if self.running:
            return
        self.running = True
        self.btn.state(["disabled"])
        self.pb.start(12)
        self.log.delete("1.0", "end")
        self.nb.select(2)
        steps = [("Descargando datos…", "fetch_data.py")]
        if self.var_bt.get() or not os.path.exists(BT):
            steps.append(("Recalculando backtest…", "backtest_v1_vs_v2.py"))
        steps.append(("Calculando escenarios…", "xrp_scenarios_v2.py"))
        threading.Thread(target=self._worker, args=(steps,), daemon=True).start()

    def _worker(self, steps):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        ok, fallos = True, 0
        for label, script in steps:
            self.q.put(("status", label))
            self.q.put(("log", f"\n=== {label} ({script}) ===\n"))
            try:
                p = subprocess.Popen([python_cmd(), os.path.join(HERE, script)], cwd=os.path.dirname(HERE),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     creationflags=flags, text=True, encoding="utf-8", errors="replace")
                for line in p.stdout:
                    if line.startswith("FAIL"):
                        fallos += 1
                    self.q.put(("log", line))
                p.wait()
                if p.returncode != 0:
                    ok = False
                    self.q.put(("log", f"\n✗ {script} terminó con error (código {p.returncode}).\n"))
                    break
            except Exception as e:
                ok = False
                self.q.put(("log", f"\n✗ No se pudo ejecutar {script}: {e}\n"))
                break
        if ok and fallos:
            self.q.put(("log", f"\n⚠ {fallos} descargas fallaron (¿sin conexión?). Se han usado los últimos datos guardados.\n"))
        self.q.put(("done", (ok, fallos)))

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", val)
                    self.log.see("end")
                elif kind == "status":
                    self.lbl_status.config(text=val)
                elif kind == "done":
                    self.running = False
                    self.btn.state(["!disabled"])
                    self.pb.stop()
                    ok, fallos = val
                    if ok:
                        self.lbl_status.config(text="Actualizado." if not fallos else
                                               f"Actualizado con datos guardados ({fallos} descargas fallaron).")
                        self.var_bt.set(False)
                        self.load_results()
                        self.nb.select(0)
                    else:
                        self.lbl_status.config(text="Error: revisa la pestaña Registro.")
                        messagebox.showerror("Error", "La actualización falló. Revisa la pestaña «Registro».")
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def open_folder(self):
        os.makedirs(OUT, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(OUT)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", OUT])
        else:
            subprocess.Popen(["xdg-open", OUT])


if __name__ == "__main__":
    App().mainloop()
