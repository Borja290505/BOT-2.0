"""
XRP — cono a 30 días, modelo v2 (HAR Garman-Klass + cuantiles conformales + trayectorias análogas).
Uso:  python analysis/fetch_data.py   (datos frescos)
      python analysis/xrp_scenarios_v2.py
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelo_v2 as m

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
H = m.H
PCT = [5, 25, 50, 75, 95]

x = m.load_xrp()
c = x.close; lc = np.log(c.values); n = len(lc)
P0, last_date = c.iloc[-1], c.index[-1]
sig = m.har_sigma(x)
Z = m.z_paths(lc, sig)
t = n - 1
qlog, zz = m.conformal_quantiles(Z, sig, t, qs=np.array(PCT) / 100)
q = {p: P0 * np.exp(qlog[i]) for i, p in enumerate(PCT)}

# Trayectorias análogas: cada ventana histórica de 30 días, estandarizada por la sigma prevista
# en su momento y reescalada a la sigma prevista hoy. Son formas de camino que ya ocurrieron.
k = np.sqrt(np.arange(1, H + 1))
paths = P0 * np.exp(zz * sig[t] * k)
fin = paths[:, -1]

# Escenarios: rangos fijos, probabilidad calculada con el modelo (no subjetiva)
SC = {"Bajista": dict(lo=None, hi=1.38, inv=1.66, col="#e34948"),
      "Base": dict(lo=1.38, hi=1.70, inv=None, col="#2a78d6"),
      "Alcista": dict(lo=1.70, hi=None, inv=1.30, col="#1baf7a")}
p_esc = os.path.join(OUT, "escenarios_v2.json")
if os.path.exists(p_esc):
    SC.update(json.load(open(p_esc)))
for s in SC.values():
    lo = -np.inf if s["lo"] is None else s["lo"]; hi = np.inf if s["hi"] is None else s["hi"]
    s["p"] = float(((fin >= lo) & (fin < hi)).mean() * 100)
    sub = fin[(fin >= lo) & (fin < hi)]
    s["rango_80"] = [float(np.percentile(sub, 10)), float(np.percentile(sub, 90))] if len(sub) else [np.nan, np.nan]

# Dos simulaciones representativas del escenario base (las más típicas, una por encima y otra por debajo de la mediana)
lp = np.log(paths)
l25, l75 = np.percentile(lp, 25, axis=0), np.percentile(lp, 75, axis=0)
inside = ((lp >= l25) & (lp <= l75)).mean(1)
base = SC["Base"]
sims = {}
for name, (a, b, tgt) in {"Simulación 1": (50, 75, 62.5), "Simulación 2": (25, 50, 37.5)}.items():
    va, vb = np.percentile(fin, [a, b])
    msk = (fin >= va) & (fin <= vb) & (fin >= base["lo"]) & (fin <= base["hi"])
    score = inside - np.abs(np.log(fin) - np.log(np.percentile(fin, tgt))) * 2
    score[~msk] = -np.inf
    i = int(np.argmax(score))
    sims[name] = dict(camino=paths[i], dia30=float(fin[i]), max=float(paths[i].max()), min=float(paths[i].min()))

# Calibración (generada por backtest_v1_vs_v2.py)
bt_path = os.path.join(OUT, "backtest_v1_vs_v2.json")
bt = json.load(open(bt_path)) if os.path.exists(bt_path) else None

levels = [1.00, 1.20, 1.30, 1.40, 1.45, 1.60, 1.70, 1.80, 2.00]
res = {"modelo": "v2: HAR Garman-Klass + cuantiles conformales + trayectorias análogas",
       "ultima_vela": str(last_date.date()), "precio": float(P0),
       "sigma_diaria_prevista_pct": float(sig[t] * 100), "n_trayectorias_analogas": int(len(zz)),
       "cono_dia30": {f"P{p}": float(q[p][-1]) for p in PCT}, "cono_dia7": {f"P{p}": float(q[p][6]) for p in PCT},
       "escenarios": {k: {"prob_pct": round(v["p"], 1), "rango": [v["lo"], v["hi"]], "rango_80_dentro": v["rango_80"],
                          "invalidacion": v["inv"]} for k, v in SC.items()},
       "simulaciones": {k: {kk: vv for kk, vv in v.items() if kk != "camino"} for k, v in sims.items()},
       "prob_niveles": {f"{L:.2f}": {"cierre_dia30_por_encima_pct": float((fin > L).mean() * 100),
                                     "toca_en_30d_pct": float(((paths.max(1) >= L) if L > P0 else (paths.min(1) <= L)).mean() * 100)}
                        for L in levels}}
json.dump(res, open(os.path.join(OUT, "resultados_v2.json"), "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- gráfico
SUP = [(1.45, "S1"), (1.39, "S2 (SMA50)"), (1.25, "S3"), (1.00, "S4")]
RES = [(1.57, "R1"), (1.70, "R2"), (2.00, "R3")]
if os.path.exists(os.path.join(OUT, "niveles.json")):
    nv = json.load(open(os.path.join(OUT, "niveles.json"))); SUP, RES = nv["sup"], nv["res"]
hx = c[-180:]
fut = pd.date_range(last_date, periods=H + 1, freq="D")
band = lambda p: np.concatenate([[P0], q[p]])
fig, ax = plt.subplots(figsize=(13, 7.2), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.fill_between(fut, band(5), band(95), color="#2a78d6", alpha=0.13, lw=0, label="Cono P5–P95 (90 %)")
ax.fill_between(fut, band(25), band(75), color="#2a78d6", alpha=0.28, lw=0, label="Cono P25–P75 (50 %)")
ax.plot(fut, band(50), color="#2a78d6", lw=1.6, ls="--", label="Mediana (P50)")
for (name, s), col in zip(sims.items(), ["#4a3aa7", "#eb6834"]):
    ax.plot(fut, np.concatenate([[P0], s["camino"]]), color=col, lw=1.5, label=f"{name} (ejemplo típico): {s['dia30']:.2f} $")
ax.plot(hx.index, hx.values, color="#0b0b0b", lw=1.6, label="XRP/USD cierre diario (Bitstamp)")
ax.plot(hx.index, c.rolling(50).mean()[-180:], color="#8a8984", lw=1, label="SMA 50")
ax.plot(hx.index, c.rolling(200).mean()[-180:], color="#8a8984", lw=1, ls=":", label="SMA 200")
for p in PCT:
    ax.annotate(f"P{p} {q[p][-1]:.2f}", (fut[-1], q[p][-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")
for v, lab in SUP:
    ax.axhline(v, color="#008300", lw=0.8, ls=(0, (4, 3)), alpha=0.7); ax.text(hx.index[0], v, f" {lab} {v:.2f}", va="bottom", fontsize=8, color="#008300")
for v, lab in RES:
    ax.axhline(v, color="#e34948", lw=0.8, ls=(0, (4, 3)), alpha=0.7); ax.text(hx.index[0], v, f" {lab} {v:.2f}", va="bottom", fontsize=8, color="#b8302f")
xe = fut[-1] + pd.Timedelta(days=15)
ylo, yhi = q[5][-1] * 0.97, q[95][-1] * 1.03
for j, (name, s) in enumerate(SC.items()):
    xx = xe + pd.Timedelta(days=9 * j)
    lo = s["lo"] if s["lo"] is not None else max(s["rango_80"][0], ylo)
    hi = s["hi"] if s["hi"] is not None else min(s["rango_80"][1], yhi)
    ax.plot([xx, xx], [lo, hi], color=s["col"], lw=6, solid_capstyle="round")
    txt = f"< {s['hi']:.2f}" if s["lo"] is None else (f"> {s['lo']:.2f}" if s["hi"] is None else f"{s['lo']:.2f}–{s['hi']:.2f}")
    ax.text(xx + pd.Timedelta(days=1.2), (lo + hi) / 2, f"{name}\n{s['p']:.0f} %\n{txt}", fontsize=8, va="center")
    if s.get("inv"):
        ax.plot([xx - pd.Timedelta(days=1.2), xx + pd.Timedelta(days=1.2)], [s["inv"]] * 2, color=s["col"], lw=1.5)
        ax.text(xx, s["inv"], f"invalida {s['inv']:.2f}", fontsize=7, ha="center", va="bottom", color=s["col"])
ax.axvline(last_date, color="#52514e", lw=0.8)
ax.text(last_date, ax.get_ylim()[1], f" Hoy {last_date:%d-%m-%Y}\n cierre {P0:.4f} $", va="top", fontsize=8, color="#52514e")
ax.set_xlim(hx.index[0], xe + pd.Timedelta(days=30))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.set_ylabel("USD por XRP")
ax.set_title(f"XRP/USD — escenarios a {H} días (hasta {fut[-1]:%d-%m-%Y}) · modelo v2\nEstimación probabilística, no una predicción",
             fontsize=13, loc="left")
cal = ""
if bt:
    a = bt["2020-06→2026-09"]["v2_todo"]; b = bt["2024-01→2026-09"]["v2_todo"]
    cal = (f"Backtest walk-forward 2020-06→2026-09: dentro de P5–P95 {a['cobertura_90']:.0f} % (ideal 90 %), por encima de P95 {a['bandas_pct']['>P95']:.0f} % (ideal 5 %); "
           f"2024→2026: {b['cobertura_90']:.0f} %.")
fig.text(0.01, 0.01, f"Datos: Bitstamp XRP/USD diario, {x.index[0]:%Y-%m-%d}→{last_date:%Y-%m-%d}. Modelo: volatilidad HAR (Garman-Klass) σ diaria {sig[t]*100:.2f} % + cuantiles conformales "
         f"sobre {len(zz)} ventanas históricas de 30 d.\n{cal} Probabilidad de los escenarios calculada por el modelo. Generado {pd.Timestamp.now('UTC'):%Y-%m-%d}.",
         fontsize=7.2, color="#52514e")
ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.08, 1.0))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, "xrp_escenarios_30d_v2.png"))

if bt:
    fig2, ax2 = plt.subplots(figsize=(9, 4.2), dpi=150)
    labels = list(bt["2020-06→2026-09"]["v1"]["bandas_pct"].keys()); xi = np.arange(6); w = 0.27
    ax2.bar(xi - w, [5, 20, 25, 25, 20, 5], w, color="#c3c2b7", label="Ideal")
    ax2.bar(xi, list(bt["2020-06→2026-09"]["v1"]["bandas_pct"].values()), w, color="#eb6834", label="Modelo v1")
    ax2.bar(xi + w, list(bt["2020-06→2026-09"]["v2_todo"]["bandas_pct"].values()), w, color="#2a78d6", label="Modelo v2")
    ax2.set_xticks(xi, labels); ax2.set_ylabel("% de ventanas de 30 d")
    ax2.set_title("Calibración walk-forward 2020-06→2026-09: ¿en qué banda acabó el precio real?", loc="left", fontsize=10.5)
    for s_ in ["top", "right"]: ax2.spines[s_].set_visible(False)
    ax2.legend(frameon=False); fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "xrp_calibracion_v1_vs_v2.png"))
print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
