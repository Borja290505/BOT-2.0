"""
XRP — cono de escenarios, modelo v2 (HAR Garman-Klass + cuantiles conformales + trayectorias análogas).
Horizonte: 7 días por defecto. Para otro horizonte, variable de entorno HORIZONTE (p. ej. HORIZONTE=30).
Uso:  python analysis/fetch_data.py        (datos frescos)
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
IMG = "xrp_escenarios_v2.png"

# ---------------------------------------------------------------- datos y precio actual
x = m.load_xrp()                                   # velas cerradas: alimentan el modelo
xa = m.load_xrp(incluir_hoy=True)                  # incluye la vela de hoy: da el precio actual
c = x.close; lc = np.log(c.values); n = len(lc)
P0 = float(xa.close.iloc[-1])                      # último precio negociado al descargar los datos
meta_p = os.path.join(m.D, "meta.json")
descargado = json.load(open(meta_p))["descargado_utc"] if os.path.exists(meta_p) else None
ahora = pd.Timestamp(descargado) if descargado else xa.index[-1]
ahora = ahora.tz_localize(None) if ahora.tzinfo else ahora

sig = m.har_sigma(x)
Z = m.z_paths(lc, sig)
t = n - 1
qlog, zz = m.conformal_quantiles(Z, sig, t, qs=np.array(PCT) / 100)
q = {p: P0 * np.exp(qlog[i]) for i, p in enumerate(PCT)}

# Trayectorias análogas: cada ventana histórica de H días, estandarizada por la sigma prevista
# en su momento y reescalada a la sigma prevista hoy. Son formas de camino que ya ocurrieron.
k = np.sqrt(np.arange(1, H + 1))
paths = P0 * np.exp(zz * sig[t] * k)
fin = paths[:, -1]

# ---------------------------------------------------------------- soportes y resistencias automáticos
def pivots(df, w=5, dias=180):
    d = df[-dias:]
    hi = [v for i, v in enumerate(d.high) if w <= i < len(d) - w and v == d.high.iloc[i - w:i + w + 1].max()]
    lo = [v for i, v in enumerate(d.low) if w <= i < len(d) - w and v == d.low.iloc[i - w:i + w + 1].min()]
    return hi, lo

def agrupa(vals, tol=0.015):
    """Une niveles a menos de un 1,5 % entre sí (media del grupo)."""
    out = []
    for v in sorted(vals):
        if out and abs(v / out[-1][0] - 1) < tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(g)) for g in out]

ph, pl = pivots(x)
sma50 = float(c.rolling(50).mean().iloc[-1]); sma200 = float(c.rolling(200).mean().iloc[-1])
extra = [sma50, sma200]
if H <= 10:                                        # horizonte corto: rango de la última semana
    extra += [float(xa.low[-8:].min()), float(xa.high[-8:].max())]
cand = agrupa(ph + pl + extra)
gap = max(0.005, 0.5 * sig[t] * np.sqrt(H))      # distancia mínima: media sigma del horizonte
SUP = sorted([v for v in cand if v < P0 * (1 - gap)], reverse=True)[:3]
RES = sorted([v for v in cand if v > P0 * (1 + gap)])[:3]
if not SUP:
    SUP = [float(q[25][-1])]
if not RES:
    RES = [float(q[75][-1])]
S1, R1 = SUP[0], RES[0]

# ---------------------------------------------------------------- escenarios (probabilidad calculada por el modelo)
SC = {"Bajista": dict(lo=None, hi=S1, inv=R1, col="#e34948"),
      "Base": dict(lo=S1, hi=R1, inv=None, col="#2a78d6"),
      "Alcista": dict(lo=R1, hi=None, inv=S1, col="#1baf7a")}
for s in SC.values():
    lo = -np.inf if s["lo"] is None else s["lo"]; hi = np.inf if s["hi"] is None else s["hi"]
    s["p"] = float(((fin >= lo) & (fin < hi)).mean() * 100)
    sub = fin[(fin >= lo) & (fin < hi)]
    s["rango_80"] = [float(np.percentile(sub, 10)), float(np.percentile(sub, 90))] if len(sub) else [np.nan, np.nan]
mas_prob = max(SC, key=lambda k_: SC[k_]["p"])

# Dos simulaciones representativas del escenario más probable (las más típicas, una por encima
# y otra por debajo de la mediana)
lp = np.log(paths)
l25, l75 = np.percentile(lp, 25, axis=0), np.percentile(lp, 75, axis=0)
inside = ((lp >= l25) & (lp <= l75)).mean(1)
ref = SC[mas_prob]
rlo = -np.inf if ref["lo"] is None else ref["lo"]; rhi = np.inf if ref["hi"] is None else ref["hi"]
sims = {}
for name, (a, b, tgt) in {"Simulación 1": (50, 75, 62.5), "Simulación 2": (25, 50, 37.5)}.items():
    va, vb = np.percentile(fin, [a, b])
    msk = (fin >= va) & (fin <= vb) & (fin >= rlo) & (fin <= rhi)
    if not msk.any():
        msk = (fin >= va) & (fin <= vb)
    score = inside - np.abs(np.log(fin) - np.log(np.percentile(fin, tgt))) * 2
    score[~msk] = -np.inf
    i = int(np.argmax(score))
    sims[name] = dict(camino=paths[i], final=float(fin[i]), max=float(paths[i].max()), min=float(paths[i].min()))

# ---------------------------------------------------------------- niveles y resultados
dias_tabla = sorted({max(1, round(H / 4)), max(1, round(H / 2)), H})
grid = np.round(np.linspace(q[5][-1], q[95][-1], 7) / 0.05) * 0.05
levels = sorted({round(float(v), 2) for v in list(grid) + SUP + RES if v > 0}, reverse=True)
bt_path = os.path.join(OUT, "backtest_v1_vs_v2.json")
bt = json.load(open(bt_path)) if os.path.exists(bt_path) else None
if bt and bt.get("horizonte_dias") != H:
    bt = None                                          # backtest de otro horizonte: no se muestra

res = {"modelo": "v2: HAR Garman-Klass + cuantiles conformales + trayectorias análogas",
       "horizonte_dias": H, "precio": P0, "precio_hora_utc": f"{ahora:%Y-%m-%d %H:%M}",
       "hasta": f"{ahora + pd.Timedelta(days=H):%Y-%m-%d}",
       "ultima_vela_cerrada": str(c.index[-1].date()), "ultimo_cierre": float(c.iloc[-1]),
       "sigma_diaria_prevista_pct": float(sig[t] * 100), "n_trayectorias_analogas": int(len(zz)),
       "cono_por_dia": {str(d): {f"P{p}": float(q[p][d - 1]) for p in PCT} for d in dias_tabla},
       "horizonte_texto": f"{H} días",
       "cono": {"etiquetas": [f"Día {d}" for d in dias_tabla], **{f"P{p}": [float(q[p][d - 1]) for d in dias_tabla] for p in PCT}},
       "niveles": {"soportes": SUP, "resistencias": RES, "sma50": sma50, "sma200": sma200},
       "escenarios": {k_: {"prob_pct": round(v["p"], 1), "rango": [v["lo"], v["hi"]], "rango_80_dentro": v["rango_80"],
                           "invalidacion": v["inv"]} for k_, v in SC.items()},
       "escenario_mas_probable": mas_prob,
       "simulaciones": {k_: {kk: vv for kk, vv in v.items() if kk != "camino"} for k_, v in sims.items()},
       "prob_niveles": {f"{L:.2f}": {"cierre_final_por_encima_pct": float((fin > L).mean() * 100),
                                     "toca_pct": float(((paths.max(1) >= L) if L > P0 else (paths.min(1) <= L)).mean() * 100)}
                        for L in levels},
       "imagen": IMG}
json.dump(res, open(os.path.join(OUT, "resultados_v2.json"), "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- gráfico
hist_n = 90 if H <= 10 else 180
hx = c[-hist_n:]
fut = pd.date_range(ahora, periods=H + 1, freq="D")
band = lambda p: np.concatenate([[P0], q[p]])
m.guardar_datos_grafico(os.path.join(OUT, "grafico_datos.json"), pd.concat([hx, pd.Series([P0], index=[ahora])]),
                        fut, q, P0, sims, SUP, RES, SC, f"XRP/USD — próximos {H} días", "dia",
                        {"sma50": {"t": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in hx.index],
                                   "p": [float(v) for v in c.rolling(50).mean()[-hist_n:]]}})
span = hist_n + H
dd = lambda f: pd.Timedelta(days=span * f)
fig, ax = plt.subplots(figsize=(13, 7.2), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.fill_between(fut, band(5), band(95), color="#2a78d6", alpha=0.13, lw=0, label="Cono P5–P95 (90 %)")
ax.fill_between(fut, band(25), band(75), color="#2a78d6", alpha=0.28, lw=0, label="Cono P25–P75 (50 %)")
ax.plot(fut, band(50), color="#2a78d6", lw=1.6, ls="--", label="Mediana (P50)")
for (name, s), col in zip(sims.items(), ["#4a3aa7", "#eb6834"]):
    ax.plot(fut, np.concatenate([[P0], s["camino"]]), color=col, lw=1.5, label=f"{name} (ejemplo típico): {s['final']:.3f} $")
ax.plot(list(hx.index) + [ahora], list(hx.values) + [P0], color="#0b0b0b", lw=1.6, label="XRP/USD diario (Bitstamp) + precio actual")
ax.plot(hx.index, c.rolling(50).mean()[-hist_n:], color="#8a8984", lw=1, label="SMA 50")
ax.plot(hx.index, c.rolling(200).mean()[-hist_n:], color="#8a8984", lw=1, ls=":", label="SMA 200")
for p in PCT:
    ax.annotate(f"P{p} {q[p][-1]:.3f}", (fut[-1], q[p][-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")
for j, v in enumerate(SUP):
    ax.axhline(v, color="#008300", lw=0.8, ls=(0, (4, 3)), alpha=0.7); ax.text(hx.index[0], v, f" S{j+1} {v:.3f}", va="bottom", fontsize=8, color="#008300")
for j, v in enumerate(RES):
    ax.axhline(v, color="#e34948", lw=0.8, ls=(0, (4, 3)), alpha=0.7); ax.text(hx.index[0], v, f" R{j+1} {v:.3f}", va="bottom", fontsize=8, color="#b8302f")
xe = fut[-1] + dd(0.07)
ylo, yhi = q[5][-1] * 0.97, q[95][-1] * 1.03
for j, (name, s) in enumerate(SC.items()):
    xx = xe + dd(0.045 * j)
    lo = s["lo"] if s["lo"] is not None else max(s["rango_80"][0], ylo)
    hi = s["hi"] if s["hi"] is not None else min(s["rango_80"][1], yhi)
    ax.plot([xx, xx], [lo, hi], color=s["col"], lw=6, solid_capstyle="round")
    txt = f"< {s['hi']:.3f}" if s["lo"] is None else (f"> {s['lo']:.3f}" if s["hi"] is None else f"{s['lo']:.3f}–{s['hi']:.3f}")
    ax.text(xx + dd(0.006), (lo + hi) / 2, f"{name}\n{s['p']:.0f} %\n{txt}", fontsize=8, va="center")
ax.axvline(ahora, color="#52514e", lw=0.8)
ax.text(ahora, ax.get_ylim()[1], f" Ahora {ahora:%d-%m-%Y %H:%M} UTC\n precio {P0:.4f} $", va="top", fontsize=8, color="#52514e")
ax.set_xlim(hx.index[0], xe + dd(0.15))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b" if H <= 10 else "%b-%y"))
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.set_ylabel("USD por XRP")
ax.set_title(f"XRP/USD — escenarios a {H} días (hasta {fut[-1]:%d-%m-%Y}) · modelo v2\nEstimación probabilística, no una predicción",
             fontsize=13, loc="left")
cal = ""
if bt:
    per = [p_ for p_ in bt if p_.startswith("desde")]
    a = bt[per[0]]["v2_todo"]; b = bt[per[-1]]["v2_todo"]
    cal = (f"Backtest walk-forward a {H} d {per[0]}: dentro de P5–P95 {a['cobertura_90']:.0f} % (ideal 90 %), por encima de P95 {a['bandas_pct']['>P95']:.0f} % (ideal 5 %); "
           f"{per[-1]}: {b['cobertura_90']:.0f} %.")
fig.text(0.01, 0.01, f"Datos: Bitstamp XRP/USD, {x.index[0]:%Y-%m-%d}→{ahora:%Y-%m-%d %H:%M} UTC. Modelo: volatilidad HAR (Garman-Klass) σ diaria {sig[t]*100:.2f} % + cuantiles conformales "
         f"sobre {len(zz)} ventanas históricas de {H} d. Soportes/resistencias: pivotes de 180 d y medias móviles.\n{cal} "
         f"Probabilidad de los escenarios calculada por el modelo. Generado {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC.",
         fontsize=7.2, color="#52514e")
ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.04, 1.0))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, IMG))
plt.close(fig)

if bt:
    per = [p_ for p_ in bt if p_.startswith("desde")]
    fig2, ax2 = plt.subplots(figsize=(9, 4.2), dpi=150)
    labels = list(bt[per[0]]["v1"]["bandas_pct"].keys()); xi = np.arange(6); w = 0.27
    ax2.bar(xi - w, [5, 20, 25, 25, 20, 5], w, color="#c3c2b7", label="Ideal")
    ax2.bar(xi, list(bt[per[0]]["v1"]["bandas_pct"].values()), w, color="#eb6834", label="Modelo v1")
    ax2.bar(xi + w, list(bt[per[0]]["v2_todo"]["bandas_pct"].values()), w, color="#2a78d6", label="Modelo v2")
    ax2.set_xticks(xi, labels); ax2.set_ylabel(f"% de ventanas de {H} d")
    ax2.set_title(f"Calibración walk-forward a {H} días ({per[0]}): ¿en qué banda acabó el precio real?", loc="left", fontsize=10.5)
    for s_ in ["top", "right"]: ax2.spines[s_].set_visible(False)
    ax2.legend(frameon=False); fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "xrp_calibracion_v1_vs_v2.png"))
print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
