"""
Aplicación de escritorio para el análisis de escenarios de XRP.
Ábrela con doble clic en "Abrir_XRP.bat" (Windows) o con:  python analysis/app_xrp.py
"""
import json, os, subprocess, sys, threading, queue, time, traceback, urllib.request
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
RES = os.path.join(OUT, "resultados_v2.json")
BT = os.path.join(OUT, "backtest_v1_vs_v2.json")
IMG_ESC = os.path.join(OUT, "xrp_escenarios_v2.png")
DATOS_GRAF = os.path.join(OUT, "grafico_datos.json")
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
TICKER = "https://www.bitstamp.net/api/v2/ticker/xrpusd/"
SEG_DIRECTO = 10                                  # cada cuánto se consulta el precio en directo
MIN_RECALCULO = {"24 h": 15}                      # minutos entre recálculos automáticos (60 en los demás modos)


def leer_json(path):
    """Lee un JSON en UTF-8; si se guardó con la codificación de Windows (versiones antiguas), también."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, encoding="cp1252") as f:
            return json.load(f)
IMG_CAL = os.path.join(OUT, "xrp_calibracion_v1_vs_v2.png")
IMG_VER = os.path.join(OUT, "verificacion.png")

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
        self.title("XRP · Escenarios")
        self.configure(bg=BG)
        self.geometry("1400x860")
        self.minsize(1100, 700)
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self.q = queue.Queue()
        self.running = False
        self.auto = False                          # actualización lanzada por el modo en directo
        self.pendiente = False
        self.live = []                             # [(fecha UTC, precio)] recibidos en directo
        self.live_fetching = False
        self.hovering = False
        self.ultima_act = time.monotonic()
        self.goto_tab = 0
        self._style()
        self._build()
        self.load_results()
        self.after(150, self._poll)
        self.after(1500, self._tick_live)
        if self._resultados_viejos():              # al abrir, si lo guardado no está al día, se recalcula
            self.after(500, lambda: self.run_update(auto=True, rapido=False))

    def report_callback_exception(self, exc, val, tb):
        """Cualquier error de la interfaz se muestra (con pythonw no hay consola donde verlo)."""
        txt = "".join(traceback.format_exception(exc, val, tb))
        try:
            self.log.insert("end", "\n✗ Error en la app:\n" + txt)
            self.lbl_status.config(text="Error: revisa la pestaña Registro.")
        except Exception:
            pass
        messagebox.showerror("Error en la app", f"{val}\n\nEl detalle está en la pestaña «Registro».")

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

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(0, 6))
        top = bar                                   # los controles van en una segunda fila
        ttk.Button(top, text="Abrir carpeta de resultados", command=self.open_folder).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="Verificar a ciegas", command=self.run_verify).pack(side="left")
        self.var_bt = tk.BooleanVar(value=not os.path.exists(BT))
        ttk.Checkbutton(top, text="Recalcular backtest (≈20 s)", variable=self.var_bt).pack(side="right", padx=8)
        self.btn = ttk.Button(top, text="⟳  Actualizar análisis", style="Accent.TButton", command=self.run_update)
        self.btn.pack(side="right")
        self.var_h = tk.StringVar(value=str(self._horizonte_guardado()))
        self.cb_h = ttk.Combobox(top, textvariable=self.var_h, values=["24 h", "7 días", "14 días", "30 días"], width=8,
                                 state="readonly")
        self.cb_h.pack(side="right", padx=(4, 10))
        self.cb_h.bind("<<ComboboxSelected>>", lambda e: self.run_update())   # cambiar de modo recalcula
        self.var_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="En directo",
                        variable=self.var_live).pack(side="left", padx=(12, 0))
        tk.Label(top, text="Horizonte:", bg=BG, fg=INK, font=(FONT, 10)).pack(side="right")

        body = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=6, bd=0)
        body.pack(fill="both", expand=True, padx=16, pady=6)

        # Panel izquierdo: resumen
        left = tk.Frame(body, bg=BG, width=380)
        body.add(left, minsize=340)
        self._section(left, "Escenarios (probabilidad del modelo)")
        self.sc_frame = tk.Frame(left, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        self.sc_frame.pack(fill="x", pady=(0, 12))

        self._section(left, "Cono de precios")
        self.tv_cone = self._tree(left, ("p", "a", "b", "c"), ("Percentil", "", "", ""), (80, 85, 85, 85), 5)

        self.lbl_lv = self._section(left, "Probabilidad por nivel")
        self.tv_lv = self._tree(left, ("lv", "toca", "cierre"), ("Nivel $", "Lo toca", "Acaba por encima"),
                                (80, 110, 140), 7)

        self._section(left, "Fiabilidad del cono (backtest)")
        self.lbl_cal = tk.Label(left, text="", justify="left", anchor="w", wraplength=350,
                                bg=BG, fg=INK, font=(FONT, 9))
        self.lbl_cal.pack(fill="x")

        # Panel derecho: gráficos
        right = tk.Frame(body, bg=BG)
        body.add(right, minsize=600)
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.fig_esc, self.cv_esc = self._figure_tab(nb, "Gráfico de escenarios", cabecera=True)
        self.cv_esc.mpl_connect("motion_notify_event", self._on_hover)
        self.cv_esc.mpl_connect("axes_leave_event", lambda e: (setattr(self, "hovering", False), self._hover_reset()))
        self.graf = None
        self.fig_cal, self.cv_cal = self._figure_tab(nb, "Calibración v1 vs v2")
        self.fig_ver, self.cv_ver = self._figure_tab(nb, "Verificación a ciegas")
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
        lbl = tk.Label(parent, text=text.upper(), bg=BG, fg=MUTED, font=(FONT, 9, "bold"), anchor="w")
        lbl.pack(fill="x", pady=(4, 4))
        return lbl

    def _tree(self, parent, cols, heads, widths, height):
        tv = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        for c, h, w in zip(cols, heads, widths):
            tv.heading(c, text=h)
            tv.column(c, width=w, anchor="e" if c != cols[0] else "w")
        tv.pack(fill="x", pady=(0, 12))
        return tv

    def _figure_tab(self, nb, title, cabecera=False):
        f = tk.Frame(nb, bg=BG)
        nb.add(f, text=title)
        if cabecera:                                 # fecha y precios bajo el cursor
            self.lbl_hover = tk.Label(f, text="", bg=CARD, fg=INK, font=(FONT, 12, "bold"), anchor="w", justify="left",
                                      padx=12, pady=6, highlightbackground=LINE, highlightthickness=1)
            self.lbl_hover.pack(side="top", fill="x", pady=(4, 0))
            self.lbl_hover.bind("<Configure>", lambda e: self.lbl_hover.config(wraplength=max(200, e.width - 30)))
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


    # ------------------------------------------------------------------ gráfico interactivo
    def draw_interactive(self):
        """Dibuja el gráfico de escenarios con datos reales: al pasar el ratón muestra fecha y precios."""
        d = leer_json(DATOS_GRAF)
        ht = np.array([np.datetime64(t) for t in d["historico"]["t"]]).astype("datetime64[s]").astype(object)
        ft = np.array([np.datetime64(t) for t in d["futuro"]["t"]]).astype("datetime64[s]").astype(object)
        hp = np.array(d["historico"]["p"]); fu = {k: np.array(v) for k, v in d["futuro"].items() if k != "t"}
        sims = {k: np.array(v) for k, v in d["simulaciones"].items()}
        horas = d.get("paso") == "hora"
        dec = 4 if horas else 3
        fig = self.fig_esc
        fig.clear()
        ax = fig.add_axes([0.10, 0.09, 0.78, 0.86])
        ax.set_facecolor(BG)
        ax.fill_between(ft, fu["P5"], fu["P95"], color="#2a78d6", alpha=0.13, lw=0, label="Cono 90 %")
        ax.fill_between(ft, fu["P25"], fu["P75"], color="#2a78d6", alpha=0.28, lw=0, label="Cono 50 %")
        ax.plot(ft, fu["P50"], color="#2a78d6", lw=1.5, ls="--", label="Mediana")
        cols = {"Simulación 1": "#4a3aa7", "Simulación 2": "#eb6834"}
        for k, v in sims.items():
            ax.plot(ft, v, color=cols.get(k, "#4a3aa7"), lw=1.5, label=k)
        if "sma50" in d:
            st = np.array([np.datetime64(t) for t in d["sma50"]["t"]]).astype("datetime64[s]").astype(object)
            ax.plot(st, d["sma50"]["p"], color="#8a8984", lw=1, label="SMA 50")
        ax.plot(ht, hp, color=INK, lw=1.5, label="Precio XRP/USD")
        self.live_line, = ax.plot([], [], color="#d1342f", lw=2.2, label="Precio en directo", zorder=6)
        self.live_dot, = ax.plot([], [], "o", ms=8, color="#d1342f", mec=BG, mew=2, zorder=7)
        for j, v in enumerate(d["soportes"]):
            ax.axhline(v, color="#008300", lw=0.8, ls=(0, (4, 3)), alpha=0.7)
            ax.text(1.002, v, f"S{j+1} {v:.{dec}f}", transform=ax.get_yaxis_transform(), fontsize=8, color="#008300", va="center")
        for j, v in enumerate(d["resistencias"]):
            ax.axhline(v, color="#e34948", lw=0.8, ls=(0, (4, 3)), alpha=0.7)
            ax.text(1.002, v, f"R{j+1} {v:.{dec}f}", transform=ax.get_yaxis_transform(), fontsize=8, color="#b8302f", va="center")
        ax.axvline(ft[0], color=MUTED, lw=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh" if horas else "%d-%b"))
        ax.grid(axis="y", color=LINE, lw=0.6); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_ylabel("USD por XRP")
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
        # elementos del cursor
        self.h_vline = ax.axvline(ft[0], color=MUTED, lw=0.8, ls=":", visible=False)
        self.h_pts = {k: ax.plot([], [], "o", ms=6, color=c, mec=BG, mew=1.5, zorder=5)[0]
                      for k, c in [("precio", INK), ("Simulación 1", "#4a3aa7"), ("Simulación 2", "#eb6834"), ("mediana", "#2a78d6")]}
        self.h_box = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", fontsize=9, visible=False,
                                 bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=LINE, alpha=0.95), zorder=10)
        self.graf = dict(ax=ax, ht=ht, hp=hp, ft=ft, fu=fu, sims=sims, dec=dec, horas=horas,
                         hn=mdates.date2num(ht), fn=mdates.date2num(ft))
        self._pintar_directo()
        self._hover_reset()
        self.cv_esc.draw_idle()

    # ------------------------------------------------------------------ precio en directo
    def _tick_live(self):
        """Cada pocos segundos pide el último precio; cada 15/60 min recalcula el análisis."""
        if self.var_live.get():
            if not self.live_fetching:
                self.live_fetching = True
                threading.Thread(target=self._fetch_live, daemon=True).start()
            minutos = MIN_RECALCULO.get(self.var_h.get(), 60)
            if not self.running and time.monotonic() - self.ultima_act > minutos * 60:
                self.run_update(auto=True)
        self.after(SEG_DIRECTO * 1000, self._tick_live)

    def _obtener_precio(self):
        req = urllib.request.Request(TICKER, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        return datetime.fromtimestamp(int(d["timestamp"]), timezone.utc).replace(tzinfo=None), float(d["last"])

    def _fetch_live(self):
        try:
            self.q.put(("live", self._obtener_precio()))
        except Exception as e:
            self.q.put(("live_err", str(e)))

    def _nuevo_precio(self, t, p):
        prev = self.live[-1][1] if self.live else None
        if not self.live or t > self.live[-1][0]:
            self.live.append((t, p))
            self.live = self.live[-5000:]
        flecha, color = ("▲", "#0a7d32") if prev is not None and p > prev else (("▼", "#d1342f") if prev is not None and p < prev else ("", INK))
        self.lbl_price.config(text=f"{p:.4f} $ {flecha}", fg=color)
        self._pintar_directo()
        if not self.hovering:
            self._hover_reset()

    def _pintar_directo(self):
        g = self.graf
        if not g or not hasattr(self, "live_line"):
            return
        pts = [(t, p) for t, p in self.live if t >= g["ft"][0]]
        if pts:
            ts, ps = zip(*pts)
            self.live_line.set_data([g["ft"][0], *ts], [g["hp"][-1], *ps])
            self.live_dot.set_data([ts[-1]], [ps[-1]])
        else:
            self.live_line.set_data([], []); self.live_dot.set_data([], [])
        self.cv_esc.draw_idle()

    def _percentil_directo(self, t, p):
        """Dónde cae el precio en directo dentro del cono previsto para ese momento."""
        g = self.graf
        x = mdates.date2num(t)
        qs = [float(np.interp(x, g["fn"], g["fu"][k])) for k in ("P5", "P25", "P50", "P75", "P95")]
        if p < qs[0]:
            return "por debajo del P5 (fuera del rango del 90 %)"
        if p > qs[-1]:
            return "por encima del P95 (fuera del rango del 90 %)"
        pc = float(np.interp(p, qs, [5, 25, 50, 75, 95]))
        return f"percentil {pc:.0f} del cono" + (" (dentro del rango del 50 %)" if 25 <= pc <= 75 else "")

    def _fecha(self, t):
        """Fecha UTC → texto en la hora local del ordenador."""
        loc = t.replace(tzinfo=timezone.utc).astimezone()
        txt = f"{DIAS[loc.weekday()]} {loc:%d-%m-%Y}"
        return txt + (f" {loc:%H:%M} (hora local)" if self.graf and self.graf["horas"] else "")

    def _hover_reset(self):
        g = self.graf
        if not g:
            return
        for a in [self.h_vline, self.h_box, *self.h_pts.values()]:
            a.set_visible(False)
        dec = g["dec"]
        pts = [(t, p) for t, p in self.live if t >= g["ft"][0]]
        if pts and self.var_live.get():
            t, p = pts[-1]
            loc = t.replace(tzinfo=timezone.utc).astimezone()
            var = (p / g["hp"][-1] - 1) * 100
            texto = (f"EN DIRECTO {DIAS[loc.weekday()]} {loc:%d-%m-%Y %H:%M:%S}  ·  Precio {p:.{dec}f} $  ·  "
                     f"{var:+.2f} % desde la previsión  ·  {self._percentil_directo(t, p)}")
            if t > g["ft"][-1]:
                texto += "  ·  la previsión ya ha caducado: pulsa Actualizar"
        else:
            texto = (f"Previsión hecha el {self._fecha(g['ft'][0])}  ·  Precio {g['hp'][-1]:.{dec}f} $  "
                     f"(pasa el ratón por el gráfico para ver fecha y precios)")
        self.lbl_hover.config(text=texto)
        self.cv_esc.draw_idle()

    def _on_hover(self, ev):
        g = self.graf
        if not g or ev.inaxes is not g["ax"] or ev.xdata is None:
            return
        self.hovering = True
        dec = g["dec"]
        for p in self.h_pts.values():
            p.set_data([], [])
        if ev.xdata <= g["fn"][0]:                   # zona histórica: precio real
            i = int(np.abs(g["hn"] - ev.xdata).argmin())
            t, y = g["ht"][i], g["hp"][i]
            self.h_pts["precio"].set_data([t], [y])
            linea = f"Precio {y:.{dec}f} $"
            caja = f"{self._fecha(t)}\nPrecio: {y:.{dec}f} $"
        else:                                        # zona futura: estimaciones
            i = int(np.abs(g["fn"] - ev.xdata).argmin())
            t, y = g["ft"][i], g["fu"]["P50"][i]
            s1, s2 = (v[i] for v in g["sims"].values())
            self.h_pts["Simulación 1"].set_data([t], [s1])
            self.h_pts["Simulación 2"].set_data([t], [s2])
            self.h_pts["mediana"].set_data([t], [y])
            lo, hi = g["fu"]["P5"][i], g["fu"]["P95"][i]
            paso = (f"+{i} h" if g["horas"] else f"día +{i}") if i else "ahora"
            linea = f"{paso}  ·  Sim. 1: {s1:.{dec}f} $  ·  Sim. 2: {s2:.{dec}f} $  ·  Mediana: {y:.{dec}f} $"
            caja = (f"{self._fecha(t)} ({paso})\nSimulación 1: {s1:.{dec}f} $\nSimulación 2: {s2:.{dec}f} $\n"
                    f"Mediana: {y:.{dec}f} $\nRango 90 %: {lo:.{dec}f} – {hi:.{dec}f} $")
            # precio real en directo cerca de ese momento (si ya ha llegado)
            paso_td = (g["fn"][1] - g["fn"][0]) / 2 if len(g["fn"]) > 1 else 0
            cerca = [(tt, pp) for tt, pp in self.live if abs(mdates.date2num(tt) - mdates.date2num(t)) <= paso_td]
            if cerca:
                pr = cerca[-1][1]
                linea += f"  ·  Real: {pr:.{dec}f} $"
                caja += f"\nPrecio real (directo): {pr:.{dec}f} $"
        self.h_vline.set_xdata([t, t]); self.h_vline.set_visible(True)
        self.h_box.xy = (mdates.date2num(t), y)
        self.h_box.set_text(caja)
        # la caja se coloca a la izquierda del cursor en la mitad derecha del gráfico
        x0, x1 = g["ax"].get_xlim()
        self.h_box.set_position((-150, 12) if ev.xdata > (x0 + x1) / 2 else (12, 12))
        self.h_box.set_visible(True)
        self.lbl_hover.config(text=f"{self._fecha(t)}  ·  {linea}")
        self.cv_esc.draw_idle()

    def load_results(self):
        msg = "Aún no hay resultados.\nPulsa «Actualizar análisis»."
        if os.path.exists(DATOS_GRAF):
            self.draw_interactive()
        else:
            self.graf = None
            self.lbl_hover.config(text="")
            self.show_image(self.fig_esc, self.cv_esc, IMG_ESC, msg)
        self.show_image(self.fig_cal, self.cv_cal, IMG_CAL, "Marca «Recalcular backtest» y pulsa Actualizar.")
        self.show_image(self.fig_ver, self.cv_ver, IMG_VER, "Pulsa «Verificar a ciegas» para comprobar el modelo\ncontra el precio real que siguió.")
        if not os.path.exists(RES):
            return
        r = leer_json(RES)
        self.lbl_price.config(text=f"{r['precio']:.4f} $")
        HT = r.get("horizonte_texto", f"{r.get('horizonte_dias', 7)} días")
        dec = 4 if r.get("paso") == "hora" else 3
        self.title(f"XRP · Escenarios a {HT}")
        self.lbl_date.config(text=f"a las {r['precio_hora_utc']} UTC (Bitstamp) · horizonte {HT}, hasta el {r['hasta']} · "
                                  f"σ diaria prevista {r['sigma_diaria_prevista_pct']:.2f} %")
        self.lbl_lv.config(text=f"PROBABILIDAD POR NIVEL ({HT.upper()})")

        for w in self.sc_frame.winfo_children():
            w.destroy()
        for name, s in r["escenarios"].items():
            row = tk.Frame(self.sc_frame, bg=CARD)
            row.pack(fill="x", padx=12, pady=6)
            lo, hi = s["rango"]
            rng = f"< {hi:.{dec}f} $" if lo is None else (f"> {lo:.{dec}f} $" if hi is None else f"{lo:.{dec}f} – {hi:.{dec}f} $")
            tk.Label(row, text=name, width=8, anchor="w", bg=CARD, fg=INK, font=(FONT, 11, "bold")).pack(side="left")
            bar = tk.Canvas(row, width=130, height=14, bg=CARD, highlightthickness=0)
            bar.pack(side="left", padx=6)
            bar.create_rectangle(0, 2, 130, 12, fill="#efeee9", width=0)
            bar.create_rectangle(0, 2, 130 * s["prob_pct"] / 100, 12, fill=COL.get(name, "#2a78d6"), width=0)
            tk.Label(row, text=f"{s['prob_pct']:.0f} %", width=5, anchor="e", bg=CARD, fg=INK, font=(FONT, 11, "bold")).pack(side="left")
            tk.Label(row, text=rng, anchor="e", bg=CARD, fg=MUTED, font=(FONT, 10)).pack(side="right")
        sims = r.get("simulaciones", {})
        if sims:
            txt = " · ".join(f"{k}: {v['final']:.{dec}f} $" for k, v in sims.items())
            tk.Label(self.sc_frame, text=f"Ejemplos típicos al final ({HT}) → {txt}", bg=CARD, fg=MUTED,
                     font=(FONT, 9), anchor="w").pack(fill="x", padx=12, pady=(0, 8))

        self.tv_cone.delete(*self.tv_cone.get_children())
        cono = r["cono"]
        etq = cono["etiquetas"][-3:]
        cols = ("a", "b", "c")[-len(etq):]
        for col in ("a", "b", "c"):
            self.tv_cone.heading(col, text="")
        for col, e in zip(cols, etq):
            self.tv_cone.heading(col, text=e)
        for k in ("P5", "P25", "P50", "P75", "P95"):
            vals = [""] * (3 - len(etq)) + [f"{v:.{dec}f} $" for v in cono[k][-3:]]
            self.tv_cone.insert("", "end", values=(k, *vals))
        self.tv_lv.delete(*self.tv_lv.get_children())
        for lv, v in sorted(r["prob_niveles"].items(), key=lambda kv: -float(kv[0])):
            self.tv_lv.insert("", "end", values=(lv, f"{v['toca_pct']:.0f} %", f"{v['cierre_final_por_encima_pct']:.0f} %"))

        b = leer_json(BT) if os.path.exists(BT) else None
        if b and b.get("horizonte_texto", f"{b.get('horizonte_dias')} días") == HT:
            per = [k for k in b if k.startswith("desde")]
            lines = []
            for p in per:
                v1, v2 = b[p]["v1"], b[p]["v2_todo"]
                lines.append(f"{p}: dentro del rango del 90 % → v2 {v2['cobertura_90']:.0f} % (v1 {v1['cobertura_90']:.0f} %); "
                             f"por encima del P95 → v2 {v2['bandas_pct']['>P95']:.0f} % (ideal 5 %).")
            d = b.get("direccion")
            if d:
                lines.append(f"Dirección a {HT}: ningún modelo probado acierta de forma fiable "
                             f"(acierto {d['logistica_acierto_pct']:.0f} %, como tirar una moneda).")
            self.lbl_cal.config(text="\n".join(lines))
        else:
            self.lbl_cal.config(text="Sin backtest para este horizonte: marca «Recalcular backtest» y pulsa Actualizar.")

    # ------------------------------------------------------------------ actualización
    def _resultados_viejos(self):
        """True si no hay resultados o se calcularon hace más de lo que espera el modo en directo (15/60 min)."""
        try:
            t = datetime.strptime(leer_json(RES)["precio_hora_utc"], "%Y-%m-%d %H:%M")
        except Exception:
            return True
        minutos = MIN_RECALCULO.get(self.var_h.get(), 60)
        return (datetime.now(timezone.utc).replace(tzinfo=None) - t).total_seconds() > minutos * 60

    def run_update(self, auto=False, rapido=None):
        """auto: lanzado por la app (sin cambiar de pestaña ni ventanas de error); rapido: descarga solo XRP."""
        rapido = auto if rapido is None else rapido
        if self.running:
            if not auto:
                self.pendiente = True              # se recalcula al terminar el cálculo en curso
            return
        self.running = True
        self.auto = auto
        self.btn.state(["disabled"])
        self.pb.start(12)
        self.log.delete("1.0", "end")
        if auto:                                   # recálculo del modo en directo: no cambia de pestaña
            self.goto_tab = self.nb.index(self.nb.select())
        else:
            self.nb.select(3)
            self.goto_tab = 0
        steps = [("Descargando datos…", "fetch_data.py --rapido" if rapido else "fetch_data.py")]
        Hs = self.var_h.get()
        H = 1 if Hs == "24 h" else int(Hs.split()[0])
        horas = Hs == "24 h"
        if self.var_bt.get() or self._horizonte_guardado(BT) != Hs:
            steps.append(("Recalculando backtest…", "xrp_24h.py --backtest" if horas else "backtest_v1_vs_v2.py"))
        steps.append(("Calculando escenarios…", "xrp_24h.py" if horas else "xrp_scenarios_v2.py"))
        threading.Thread(target=self._worker, args=(steps, H), daemon=True).start()

    def run_verify(self):
        """Predice desde una fecha aleatoria de hace ~15 días (solo con datos de entonces) y lo compara con el precio real."""
        if self.running:
            return
        self.running = True
        self.btn.state(["disabled"])
        self.pb.start(12)
        self.log.delete("1.0", "end")
        self.nb.select(3)
        self.goto_tab = 2
        script = "xrp_24h.py --verificar" if self.var_h.get() == "24 h" else "verificar.py"
        threading.Thread(target=self._worker, args=([("Verificando a ciegas…", script)], 7, True), daemon=True).start()

    def _horizonte_guardado(self, path=RES):
        """Horizonte de unos resultados guardados, con el texto del desplegable ("24 h", "7 días"…)."""
        try:
            d = leer_json(path)
            if d.get("horizonte_texto") == "24 horas":
                return "24 h"
            return f"{int(d.get('horizonte_dias', 7))} días"
        except Exception:
            return "24 h" if path == RES else None

    def _worker(self, steps, H, verificar=False):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg", HORIZONTE=str(H))
        if verificar:
            env.pop("HORIZONTE")                     # la verificación llega hasta el último día conocido
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        ok, fallos = True, 0
        for label, script in steps:
            self.q.put(("status", label))
            self.q.put(("log", f"\n=== {label} ({script}) ===\n"))
            try:
                nombre, *args = script.split()
                p = subprocess.Popen([python_cmd(), os.path.join(HERE, nombre), *args], cwd=os.path.dirname(HERE),
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
                    self.ultima_act = time.monotonic()
                    ok, fallos = val
                    if self.pendiente:
                        self.pendiente = False
                        self.after(100, self.run_update)
                    if ok:
                        hora = datetime.now().strftime("%H:%M")
                        self.lbl_status.config(text=f"Actualizado a las {hora}." if not fallos else
                                               f"Actualizado a las {hora} con datos guardados ({fallos} descargas fallaron).")
                        self.var_bt.set(False)
                        self.load_results()
                        self.nb.select(self.goto_tab)
                    else:
                        self.lbl_status.config(text="Error: revisa la pestaña Registro.")
                        if not self.auto:
                            messagebox.showerror("Error", "La actualización falló. Revisa la pestaña «Registro».")
                elif kind == "live":
                    self.live_fetching = False
                    self._nuevo_precio(*val)
                elif kind == "live_err":
                    self.live_fetching = False
                    self.lbl_status.config(text=f"Sin precio en directo ({val[:60]}).")
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
