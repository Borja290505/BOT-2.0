"""
XRP: cono de probabilidad a 30 días (Monte Carlo t-Student + bootstrap por bloques),
backtest de calibración, indicadores técnicos y gráfico de escenarios.
Uso: python analysis/xrp_scenarios.py   (requiere numpy, pandas, matplotlib y los CSV de analysis/data)
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

D = os.path.join(os.path.dirname(__file__), "data")
OUTDIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)
H = 30            # horizonte (días)
N = 20000         # trayectorias
PCT = [5, 25, 50, 75, 95]
rng = np.random.default_rng(42)

def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[df.close > 0]

xrp = load("bitstamp_xrpusd_1d.csv")
src = "Bitstamp XRP/USD diario"
if xrp is None or len(xrp) < 800:
    xrp, src = load("kraken_xrpusd_1d.csv"), "Kraken XRP/USD diario"
# La vela del día en curso está incompleta: se descarta
xrp = xrp.iloc[:-1]
c = xrp.close
r = np.log(c).diff().dropna()
last_date, P0 = c.index[-1], c.iloc[-1]
res = {"fuente": src, "ultima_vela_cerrada": str(last_date.date()), "precio_cierre": P0,
       "n_dias": int(len(c)), "inicio": str(c.index[0].date())}

# ---------- Indicadores técnicos ----------
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)
def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
wk = xrp.resample("W-SUN").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
tech = {
    "SMA20": c.rolling(20).mean().iloc[-1], "SMA50": c.rolling(50).mean().iloc[-1],
    "SMA200": c.rolling(200).mean().iloc[-1], "EMA21_semanal": ema(wk.close, 21).iloc[-1],
    "SMA50_semanal": wk.close.rolling(50).mean().iloc[-1],
    "RSI14_diario": rsi(c).iloc[-1], "RSI14_semanal": rsi(wk.close).iloc[-1],
    "MACD": macd.iloc[-1], "MACD_senal": sig.iloc[-1], "MACD_hist": (macd - sig).iloc[-1],
    "MACD_hist_hace5d": (macd - sig).iloc[-6],
    "ATR14": atr(xrp).iloc[-1], "ATR14_pct": atr(xrp).iloc[-1] / P0 * 100,
    "vol_real_30d_anual_pct": r[-30:].std() * np.sqrt(365) * 100,
    "vol_real_90d_anual_pct": r[-90:].std() * np.sqrt(365) * 100,
    "vol_real_365d_anual_pct": r[-365:].std() * np.sqrt(365) * 100,
    "max_365d": c[-365:].max(), "fecha_max_365d": str(c[-365:].idxmax().date()),
    "min_365d": c[-365:].min(), "fecha_min_365d": str(c[-365:].idxmin().date()),
    "max_historico_cierre": c.max(), "fecha_max_historico": str(c.idxmax().date()),
    "max_historico_intradia": xrp.high.max(), "fecha_max_intradia": str(xrp.high.idxmax().date()),
    "ret_7d_pct": (P0 / c.iloc[-8] - 1) * 100, "ret_30d_pct": (P0 / c.iloc[-31] - 1) * 100,
    "ret_90d_pct": (P0 / c.iloc[-91] - 1) * 100, "ret_365d_pct": (P0 / c.iloc[-366] - 1) * 100,
    "max_90d": c[-90:].max(), "min_90d": c[-90:].min(),
    "max_30d": xrp.high[-30:].max(), "min_30d": xrp.low[-30:].min(),
}
tech["dist_ATH_pct"] = (P0 / tech["max_historico_intradia"] - 1) * 100
# Tendencia por marco temporal
tech["tendencia_diaria"] = "alcista" if P0 > tech["SMA50"] > tech["SMA200"] else ("bajista" if P0 < tech["SMA50"] < tech["SMA200"] else "mixta")
tech["tendencia_semanal"] = "alcista" if wk.close.iloc[-1] > tech["EMA21_semanal"] and ema(wk.close, 21).diff().iloc[-1] > 0 else ("bajista" if wk.close.iloc[-1] < tech["EMA21_semanal"] else "mixta")

# Pivotes (máximos/mínimos locales de 10 días a cada lado) en los últimos 365 días
w = 10; cc = xrp[-365:]
piv_hi = [(d, v) for i, (d, v) in enumerate(cc.high.items()) if w <= i < len(cc) - w and v == cc.high.iloc[i-w:i+w+1].max()]
piv_lo = [(d, v) for i, (d, v) in enumerate(cc.low.items()) if w <= i < len(cc) - w and v == cc.low.iloc[i-w:i+w+1].min()]
tech["pivotes_altos"] = [(str(d.date()), round(v, 4)) for d, v in piv_hi[-8:]]
tech["pivotes_bajos"] = [(str(d.date()), round(v, 4)) for d, v in piv_lo[-8:]]

# Perfil de volumen aproximado (velas diarias, 180 días, volumen asignado al precio típico)
vp = xrp[-180:].copy(); vp["tp"] = (vp.high + vp.low + vp.close) / 3
bins = np.linspace(vp.low.min(), vp.high.max(), 25)
hist, edges = np.histogram(vp.tp, bins=bins, weights=vp.volume * vp.tp)
poc = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
order = np.argsort(hist)[::-1]; cum, va = 0, []
for i in order:
    cum += hist[i]; va.append(i)
    if cum >= 0.7 * hist.sum(): break
tech["perfil_vol_POC"] = poc
tech["perfil_vol_VAL"] = edges[min(va)]; tech["perfil_vol_VAH"] = edges[max(va) + 1]
res["tecnico"] = tech

# ---------- Correlaciones ----------
corr = {}
for t in ["btc", "eth"]:  # SOL/ADA/XLM en Bitstamp vienen incompletos
    o = load(f"bitstamp_{t}usd_1d.csv")
    if o is None: continue
    ro = np.log(o.close).diff()
    j = pd.concat([r, ro], axis=1, join="inner").dropna()
    corr[t.upper()] = {"90d": j[-90:].corr().iloc[0, 1], "365d": j[-365:].corr().iloc[0, 1],
                       "beta_90d": np.cov(j.iloc[-90:, 0], j.iloc[-90:, 1])[0, 1] / j.iloc[-90:, 1].var(),
                       "ret_30d_pct": (o.close.iloc[-2] / o.close.iloc[-32] - 1) * 100,
                       "ret_365d_pct": (o.close.iloc[-2] / o.close.iloc[-367] - 1) * 100,
                       "vol_30d_anual_pct": ro[-30:].std() * np.sqrt(365) * 100,
                       "precio": o.close.iloc[-2]}
res["correlaciones"] = corr

# ---------- Modelos del cono ----------
def sigma_now(rr):
    """Volatilidad diaria: mezcla 50/50 de EWMA (lambda 0.94) y realizada 90 d."""
    ew = np.sqrt((rr[-250:] ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1])
    return 0.5 * ew + 0.5 * rr[-90:].std()

def cone_t(rr, h=H, n=N, nu=3.5):
    s = sigma_now(rr)
    z = rng.standard_t(nu, size=(n, h)) / np.sqrt(nu / (nu - 2))
    return np.cumsum(s * z - 0.5 * s ** 2, axis=1)   # deriva cero en precio esperado, mediana ligeramente < P0

def cone_boot(rr, h=H, n=N, block=5, lookback=1095):
    """Bootstrap por bloques de retornos históricos (3 años), sin deriva, reescalados a la vol actual."""
    hist_r = rr[-lookback:].values
    hist_r = hist_r - hist_r.mean()
    hist_r = hist_r * (sigma_now(rr) / hist_r.std())
    nb = int(np.ceil(h / block))
    starts = rng.integers(0, len(hist_r) - block, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :h]
    return np.cumsum(hist_r[idx], axis=1)

paths_t = cone_t(r); paths_b = cone_boot(r)
paths = np.vstack([paths_t, paths_b])      # cono final: mezcla de ambos modelos
q = {p: P0 * np.exp(np.percentile(paths, p, axis=0)) for p in PCT}
res["sigma_diaria_modelo_pct"] = sigma_now(r) * 100
res["cono_dia30"] = {f"P{p}": q[p][-1] for p in PCT}
res["cono_dia30_t"] = {f"P{p}": P0 * np.exp(np.percentile(paths_t[:, -1], p)) for p in PCT}
res["cono_dia30_boot"] = {f"P{p}": P0 * np.exp(np.percentile(paths_b[:, -1], p)) for p in PCT}
res["cono_dia7"] = {f"P{p}": q[p][6] for p in PCT}
fin = P0 * np.exp(paths[:, -1]); mn = P0 * np.exp(paths.min(axis=1)); mx = P0 * np.exp(paths.max(axis=1))

# ---------- Backtest de calibración ----------
def backtest(model, step=1, start_min=730):
    out = []
    lr = np.log(c)
    for i in range(start_min, len(c) - H, step):
        rr = r.iloc[:i]                       # solo información disponible en t
        global N
        n0, N = N, 2000
        pth = model(rr, n=2000)
        N = n0
        qs = np.percentile(pth[:, -1], PCT)
        real = lr.iloc[i + H] - lr.iloc[i]
        out.append((c.index[i], np.searchsorted(qs, real)))  # 0:<P5 1:P5-25 2:P25-50 3:P50-75 4:P75-95 5:>P95
    return pd.DataFrame(out, columns=["fecha", "banda"]).set_index("fecha")

labels = ["<P5", "P5-P25", "P25-P50", "P50-P75", "P75-P95", ">P95"]
ideal = [5, 20, 25, 25, 20, 5]
bt = {}
for name, model in [("t-Student", cone_t), ("Bootstrap", cone_boot)]:
    b = backtest(model, step=3)
    freq = b.banda.value_counts(normalize=True).reindex(range(6), fill_value=0) * 100
    by_year = b.groupby(b.index.year).banda.apply(lambda x: {
        "dentro_P5_P95": ((x >= 1) & (x <= 4)).mean() * 100, "dentro_P25_P75": ((x >= 2) & (x <= 3)).mean() * 100,
        "n": len(x)})
    # muestras no solapadas (cada 30 días) para medir la dispersión real
    nb = b.iloc[::10]
    bt[name] = {"n_ventanas": int(len(b)), "desde": str(b.index[0].date()), "hasta": str(b.index[-1].date()),
                "frecuencia_por_banda_pct": dict(zip(labels, freq.round(1).tolist())),
                "ideal_pct": dict(zip(labels, ideal)),
                "dentro_P5_P95_pct": float(freq[1:5].sum()), "dentro_P25_P75_pct": float(freq[2:4].sum()),
                "dentro_P5_P95_no_solapado_pct": float(((nb.banda >= 1) & (nb.banda <= 4)).mean() * 100),
                "n_no_solapado": int(len(nb)),
                "por_anio": {str(k[0]) + "_" + k[1]: round(v, 1) for k, v in by_year.items()}}
res["backtest"] = bt

# Probabilidades del cono para niveles clave (día 30 y toque intramensual)
levels = [1.00, 1.20, 1.30, 1.40, 1.45, 1.60, 1.70, 1.80, 2.00]
res["prob_niveles"] = {f"{L:.2f}": {"cierre_dia30_por_encima_pct": float((fin > L).mean() * 100),
                                    "toca_en_30d_pct": float(((mx >= L) if L > P0 else (mn <= L)).mean() * 100)}
                       for L in levels}
json.dump(res, open(os.path.join(OUTDIR, "resultados.json"), "w"), indent=1, default=float, ensure_ascii=False)

# ---------- Gráfico ----------
SC = {  # probabilidad subjetiva, rango objetivo día 30, invalidación  (ver informe)
    "Bajista": dict(p=25, lo=1.15, hi=1.38, inv=1.62, col="#e34948"),
    "Base":    dict(p=50, lo=1.38, hi=1.70, inv=None, col="#2a78d6"),
    "Alcista": dict(p=25, lo=1.70, hi=2.05, inv=1.38, col="#1baf7a"),
}
json_sc = os.path.join(OUTDIR, "escenarios.json")
if os.path.exists(json_sc):
    SC.update(json.load(open(json_sc)))
SUP = [(1.45, "S1"), (1.30, "S2"), (1.20, "S3")]
RES = [(1.55, "R1"), (1.70, "R2"), (2.00, "R3")]
if os.path.exists(os.path.join(OUTDIR, "niveles.json")):
    nv = json.load(open(os.path.join(OUTDIR, "niveles.json"))); SUP, RES = nv["sup"], nv["res"]

hist_n = 180
hx = c[-hist_n:]
fut = pd.date_range(last_date, periods=H + 1, freq="D")
band = lambda p: np.concatenate([[P0], q[p]])
fig, ax = plt.subplots(figsize=(13, 7.2), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.fill_between(fut, band(5), band(95), color="#2a78d6", alpha=0.13, lw=0, label="Cono P5–P95 (90 %)")
ax.fill_between(fut, band(25), band(75), color="#2a78d6", alpha=0.28, lw=0, label="Cono P25–P75 (50 %)")
ax.plot(fut, band(50), color="#2a78d6", lw=1.6, ls="--", label="Mediana del cono (P50)")
ax.plot(hx.index, hx.values, color="#0b0b0b", lw=1.6, label=f"XRP/USD cierre diario ({src.split()[0]})")
ax.plot(hx.index, c.rolling(50).mean()[-hist_n:], color="#8a8984", lw=1, label="SMA 50")
ax.plot(hx.index, c.rolling(200).mean()[-hist_n:], color="#8a8984", lw=1, ls=":", label="SMA 200")
for p in PCT:
    ax.annotate(f"P{p} {q[p][-1]:.2f}", (fut[-1], q[p][-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color="#52514e")
x0 = hx.index[0]
for v, lab in SUP:
    ax.axhline(v, color="#008300", lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    ax.text(x0, v, f" {lab} {v:.2f}", va="bottom", fontsize=8, color="#008300")
for v, lab in RES:
    ax.axhline(v, color="#e34948", lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    ax.text(x0, v, f" {lab} {v:.2f}", va="bottom", fontsize=8, color="#b8302f")
# Escenarios: barras de rango al final del horizonte
xe = fut[-1] + pd.Timedelta(days=9)
for k, (name, s) in enumerate(SC.items()):
    xx = xe + pd.Timedelta(days=4 * k)
    ax.plot([xx, xx], [s["lo"], s["hi"]], color=s["col"], lw=6, solid_capstyle="round")
    ax.text(xx + pd.Timedelta(days=1.2), (s["lo"] + s["hi"]) / 2, f"{name}\n{s['p']} %\n{s['lo']:.2f}–{s['hi']:.2f}",
            fontsize=8, va="center", color="#0b0b0b")
    if s.get("inv"):
        ax.plot([xx - pd.Timedelta(days=1.2), xx + pd.Timedelta(days=1.2)], [s["inv"]] * 2, color=s["col"], lw=1.5)
        ax.text(xx, s["inv"], "✕ inval.", fontsize=7, ha="center", va="bottom", color=s["col"])
ax.axvline(last_date, color="#52514e", lw=0.8)
ax.text(last_date, ax.get_ylim()[1], f" Hoy {last_date:%d-%m-%Y}\n cierre {P0:.4f} $", va="top", fontsize=8, color="#52514e")
ax.set_xlim(hx.index[0], xe + pd.Timedelta(days=22))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.set_ylabel("USD por XRP")
bt_t = bt["t-Student"]; bt_b = bt["Bootstrap"]
ax.set_title(f"XRP/USD — escenarios a {H} días (hasta {fut[-1]:%d-%m-%Y})\n"
             f"Estimación probabilística, no una predicción", fontsize=13, loc="left", color="#0b0b0b")
fig.text(0.01, 0.01,
         f"Datos: {src} (API pública), {res['inicio']}→{res['ultima_vela_cerrada']}. Cono: 40.000 trayectorias (t-Student ν=3,5 + bootstrap por bloques de 5 d), "
         f"σ diaria {res['sigma_diaria_modelo_pct']:.2f} %, sin deriva.\nBacktest {bt_t['desde']}→{bt_t['hasta']}: dentro de P5–P95 "
         f"{bt_t['dentro_P5_P95_pct']:.0f} % (t) / {bt_b['dentro_P5_P95_pct']:.0f} % (boot) vs 90 % ideal; P25–P75 {bt_t['dentro_P25_P75_pct']:.0f} % / "
         f"{bt_b['dentro_P25_P75_pct']:.0f} % vs 50 %. Niveles: pivotes propios + análisis citados. Escenarios: probabilidad subjetiva. Generado {pd.Timestamp.utcnow():%Y-%m-%d}.",
         fontsize=7.2, color="#52514e")
ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0, 0.93))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUTDIR, "xrp_escenarios_30d.png"))

# Gráfico de calibración
fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=150)
xi = np.arange(6); wdt = 0.27
ax2.bar(xi - wdt, ideal, wdt, color="#c3c2b7", label="Ideal")
ax2.bar(xi, list(bt_t["frecuencia_por_banda_pct"].values()), wdt, color="#2a78d6", label="t-Student")
ax2.bar(xi + wdt, list(bt_b["frecuencia_por_banda_pct"].values()), wdt, color="#eb6834", label="Bootstrap")
ax2.set_xticks(xi, labels); ax2.set_ylabel("% de ventanas de 30 d")
ax2.set_title("Calibración: ¿en qué banda acabó el precio real a 30 días?", loc="left", fontsize=11)
for s_ in ["top", "right"]: ax2.spines[s_].set_visible(False)
ax2.legend(frameon=False); fig2.tight_layout()
fig2.savefig(os.path.join(OUTDIR, "xrp_calibracion.png"))
print(json.dumps(res, indent=1, default=float, ensure_ascii=False))
