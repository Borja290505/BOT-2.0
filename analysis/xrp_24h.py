"""
XRP — escenarios para las próximas 24 horas con velas de 1 hora.
Mismo método que el modelo v2 (volatilidad HAR + cuantiles conformales + trayectorias análogas),
con un ajuste horario: solo se usan como análogas las ventanas históricas que empezaron a la
misma hora UTC que ahora (la volatilidad de XRP cambia mucho según la hora del día).

Uso:  python analysis/xrp_24h.py               → escenarios de las próximas 24 h
      python analysis/xrp_24h.py --backtest    → calibración walk-forward del último año
      python analysis/xrp_24h.py --verificar   → predicción a ciegas desde un momento aleatorio
                                                 de los últimos 15 días frente al precio real
      (con --verificar, FECHA="2026-09-28 14:00" fija el momento)
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
H = 24                                   # pasos de 1 hora
WINS = (1, 24, 168, 720)                 # 1 h, 1 día, 1 semana, 1 mes
MIN_TRAIN = 24 * 120                     # 120 días de historia antes de predecir
REFIT = 24 * 7                           # reajuste semanal del HAR
PCT = [5, 25, 50, 75, 95]
QS = np.array(PCT) / 100


def load_1h():
    df = pd.read_csv(os.path.join(m.D, "bitstamp_xrpusd_1h.csv"))
    df.index = pd.to_datetime(df.timestamp, unit="s")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[df.close > 0]


def preparar(x):
    lc = np.log(x.close.values)
    sig = m.har_sigma(x, h=H, wins=WINS, min_train=MIN_TRAIN, refit=REFIT)
    Z = m.z_paths(lc, sig, h=H)
    return lc, sig, Z


def cuantiles(Z, sig, t, horas, misma_hora=True, qs=QS):
    """Cuantiles del retorno log 1..24 h desde t, solo con ventanas cuyo resultado ya se conoce en t."""
    hi = t - H + 1
    idx = np.arange(0, max(hi, 0))
    if misma_hora:
        idx = idx[horas[idx] == horas[t]]
    zz = Z[idx]; zz = zz[np.isfinite(zz).all(1)]
    if len(zz) < 200:
        return None, None
    k = np.sqrt(np.arange(1, H + 1))
    return np.quantile(zz, qs, axis=0) * sig[t] * k, zz


# ======================================================================= backtest
def backtest():
    x = load_1h().iloc[:-1]
    lc, sig, Z = preparar(x)
    horas = x.index.hour.values
    n = len(lc)
    start = max(int(np.searchsorted(x.index, x.index[-1] - pd.Timedelta(days=365))), MIN_TRAIN + H)
    rows = []
    for t in range(start, n - H, 6):                       # cada 6 horas durante el último año
        if not np.isfinite(sig[t]):
            continue
        real = lc[t + 1:t + H + 1] - lc[t]
        r = {"fecha": x.index[t], "real": real}
        for name, mh in (("v1", False), ("v2_todo", True)):
            q, _ = cuantiles(Z, sig, t, horas, misma_hora=mh)
            r[name] = q
        if r["v1"] is not None and r["v2_todo"] is not None:
            rows.append(r)

    def evalua(rr, key):
        band = np.array([np.searchsorted(w[key][:, -1], w["real"][-1]) for w in rr])
        freq = np.bincount(band, minlength=6) / len(band) * 100
        por_hora = [float(np.mean([(w[key][0, h] <= w["real"][h] <= w[key][-1, h]) for w in rr]) * 100) for h in range(H)]
        pin = np.mean([np.mean([max(q * (w["real"][-1] - v), (q - 1) * (w["real"][-1] - v)) for q, v in zip(QS, w[key][:, -1])]) for w in rr])
        return {"bandas_pct": dict(zip(["<P5", "P5-25", "P25-50", "P50-75", "P75-95", ">P95"], freq.round(1).tolist())),
                "cobertura_90": round(float(freq[1:5].sum()), 1), "cobertura_50": round(float(freq[2:4].sum()), 1),
                "error_calibracion": round(float(np.abs(freq - [5, 20, 25, 25, 20, 5]).sum()), 1),
                "pinball": round(float(pin * 1000), 3), "cobertura_90_por_hora": [round(v, 1) for v in por_hora]}

    ultimos90 = [w for w in rows if w["fecha"] >= x.index[-1] - pd.Timedelta(days=90)]
    med = np.array([w["v2_todo"][2, -1] for w in rows]); real = np.array([w["real"][-1] for w in rows])
    err_med = np.abs(np.exp(med - real) - 1) * 100; err_rw = np.abs(np.exp(-real) - 1) * 100
    out = {"desde el último año": {k: evalua(rows, k) for k in ("v1", "v2_todo")},
           "desde últimos 90 días": {k: evalua(ultimos90, k) for k in ("v1", "v2_todo")},
           "direccion": {"logistica_acierto_pct": round(float((np.sign(med) == np.sign(real))[med != 0].mean() * 100), 1),
                         "nota": "acierto de la mediana del cono en la dirección a 24 h"},
           "precio_exacto": {"error_medio_mediana_pct": round(float(err_med.mean()), 2),
                             "error_medio_sin_cambio_pct": round(float(err_rw.mean()), 2),
                             "acierto_±1%_pct": round(float((err_med <= 1).mean() * 100), 1)},
           "nombres": {"v1": "sin ajuste horario", "v2_todo": "con ajuste horario"},
           "horizonte_dias": 1, "horizonte_texto": "24 horas", "n_ventanas": len(rows),
           "hasta": str(rows[-1]["fecha"])}
    # Las claves "desde…" mantienen el formato de backtest_v1_vs_v2.json para la app y la web
    out = {("desde " + k.split("desde ")[1] if k.startswith("desde") else k): v for k, v in out.items()}
    json.dump(out, open(os.path.join(OUT, "backtest_v1_vs_v2.json"), "w"), indent=1, ensure_ascii=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.1, 1]})
    per = "desde el último año"
    xi = np.arange(6); w = 0.27
    a1.bar(xi - w, [5, 20, 25, 25, 20, 5], w, color="#c3c2b7", label="Ideal")
    a1.bar(xi, list(out[per]["v1"]["bandas_pct"].values()), w, color="#eb6834", label="Sin ajuste horario")
    a1.bar(xi + w, list(out[per]["v2_todo"]["bandas_pct"].values()), w, color="#2a78d6", label="Con ajuste horario")
    a1.set_xticks(xi, list(out[per]["v1"]["bandas_pct"].keys())); a1.set_ylabel("% de ventanas de 24 h")
    a1.set_title("¿En qué banda acabó el precio real a 24 h?", loc="left", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    hh = np.arange(1, H + 1)
    a2.axhline(90, color="#c3c2b7", lw=1.5, label="Ideal 90 %")
    a2.plot(hh, out[per]["v1"]["cobertura_90_por_hora"], color="#eb6834", lw=1.8, label="Sin ajuste horario")
    a2.plot(hh, out[per]["v2_todo"]["cobertura_90_por_hora"], color="#2a78d6", lw=1.8, label="Con ajuste horario")
    a2.set_ylim(70, 100); a2.set_xlabel("Horas desde la previsión"); a2.set_ylabel("% dentro del rango del 90 %")
    a2.set_title("Cobertura del rango del 90 % hora a hora", loc="left", fontsize=10)
    a2.legend(frameon=False, fontsize=8)
    for a in (a1, a2):
        for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
    fig.suptitle(f"Calibración walk-forward del modelo de 24 h (último año, {len(rows)} previsiones)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "xrp_calibracion_v1_vs_v2.png")); plt.close(fig)
    print(json.dumps(out, indent=1, ensure_ascii=False))


# ======================================================================= verificación a ciegas
def verificar():
    full = load_1h().iloc[:-1]
    fecha = os.environ.get("FECHA")
    if fecha:
        corte = pd.Timestamp(fecha)
    else:
        rng = np.random.default_rng()
        corte = full.index[-1] - pd.Timedelta(hours=int(rng.integers(H, 15 * 24)))
    i0 = int(np.searchsorted(full.index, corte, side="right")) - 1
    i0 = min(i0, len(full) - 1 - H)
    corte = full.index[i0]
    x = full.iloc[: i0 + 1]
    lc, sig, Z = preparar(x)
    t = len(lc) - 1
    qlog, zz = cuantiles(Z, sig, t, x.index.hour.values)
    P0 = float(x.close.iloc[-1])
    q = {p: P0 * np.exp(qlog[j]) for j, p in enumerate(PCT)}
    paths = P0 * np.exp(zz * sig[t] * np.sqrt(np.arange(1, H + 1)))
    real = full.close.iloc[i0 + 1: i0 + 1 + H]
    filas = [{"hora": h + 1, "fecha": str(d), "real": float(pr), "mediana": float(q[50][h]),
              "P5": float(q[5][h]), "P95": float(q[95][h]), "percentil_real": round(float((paths[:, h] < pr).mean() * 100), 1),
              "dentro_50": bool(q[25][h] <= pr <= q[75][h]), "dentro_90": bool(q[5][h] <= pr <= q[95][h])}
             for h, (d, pr) in enumerate(real.items())]
    df = pd.DataFrame(filas)
    res = {"fecha_corte": str(corte), "precio_corte": P0, "horizonte_texto": "24 horas",
           "horas_dentro_50_pct": float(df.dentro_50.mean() * 100), "horas_dentro_90_pct": float(df.dentro_90.mean() * 100),
           "error_final_mediana_pct": float((q[50][-1] / real.iloc[-1] - 1) * 100), "final": filas[-1], "horas": filas}
    json.dump(res, open(os.path.join(OUT, "verificacion.json"), "w"), indent=1, ensure_ascii=False)

    hx = full.close.iloc[max(0, i0 - 72): i0 + 1]
    fut = [corte] + list(real.index)
    band = lambda p: np.concatenate([[P0], q[p]])
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=140)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.fill_between(fut, band(5), band(95), color="#2a78d6", alpha=0.13, lw=0, label="Cono P5–P95 previsto (90 %)")
    ax.fill_between(fut, band(25), band(75), color="#2a78d6", alpha=0.28, lw=0, label="Cono P25–P75 previsto (50 %)")
    ax.plot(fut, band(50), color="#2a78d6", lw=1.6, ls="--", label="Mediana prevista")
    ax.plot(hx.index, hx.values, color="#0b0b0b", lw=1.4, label="Precio conocido en el momento de corte")
    ax.plot(fut, [P0] + list(real.values), color="#eb6834", lw=2.2, marker="o", ms=3, label="Precio REAL que siguió")
    ax.axvline(corte, color="#52514e", lw=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh"))
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    ax.set_ylabel("USD por XRP")
    f = filas[-1]
    ax.set_title(f"Verificación a ciegas: previsión a 24 horas desde el {corte:%d-%m-%Y %H:%M} UTC vs precio real\n"
                 f"A las 24 h: real {f['real']:.4f} USD · mediana prevista {f['mediana']:.4f} USD · percentil {f['percentil_real']:.0f} del cono",
                 loc="left", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.text(0.01, 0.01, f"El modelo solo usó datos hasta el momento de corte. Horas dentro del rango del 50 %: {res['horas_dentro_50_pct']:.0f} % · "
             f"dentro del 90 %: {res['horas_dentro_90_pct']:.0f} %.", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, "verificacion.png")); plt.close(fig)
    print(df[["hora", "fecha", "real", "mediana", "P5", "P95", "percentil_real", "dentro_90"]].to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("horas", "final")}, indent=1, ensure_ascii=False))


# ======================================================================= escenarios de las próximas 24 h
def escenarios():
    xa = load_1h()                                  # incluye la hora en curso: da el precio actual
    x = xa.iloc[:-1]                                # horas cerradas: alimentan el modelo
    lc, sig, Z = preparar(x)
    t = len(lc) - 1
    P0 = float(xa.close.iloc[-1])
    meta_p = os.path.join(m.D, "meta.json")
    ahora = pd.Timestamp(json.load(open(meta_p))["descargado_utc"]) if os.path.exists(meta_p) else xa.index[-1]
    ahora = ahora.tz_localize(None) if ahora.tzinfo else ahora
    horas = x.index.hour.values.copy()
    horas[t] = ahora.hour                           # analogía con la hora actual
    qlog, zz = cuantiles(Z, sig, t, horas)
    q = {p: P0 * np.exp(qlog[j]) for j, p in enumerate(PCT)}
    paths = P0 * np.exp(zz * sig[t] * np.sqrt(np.arange(1, H + 1)))
    fin = paths[:, -1]
    sig24 = float(np.std(np.log(fin / P0)))

    # Soportes y resistencias: pivotes de velas de 1 h (últimos 10 días) y máximos/mínimos de 24 h y 72 h
    d = xa[-240:]
    w = 6
    ph = [v for i, v in enumerate(d.high) if w <= i < len(d) - w and v == d.high.iloc[i - w:i + w + 1].max()]
    pl = [v for i, v in enumerate(d.low) if w <= i < len(d) - w and v == d.low.iloc[i - w:i + w + 1].min()]
    extra = [float(xa.high[-24:].max()), float(xa.low[-24:].min()), float(xa.high[-72:].max()), float(xa.low[-72:].min())]
    cand = sorted(ph + pl + extra); grupos = []
    for v in cand:
        if grupos and abs(v / grupos[-1][0] - 1) < 0.006:
            grupos[-1].append(v)
        else:
            grupos.append([v])
    cand = [float(np.mean(g)) for g in grupos]
    gap = max(0.003, 0.5 * sig24)
    SUP = sorted([v for v in cand if v < P0 * (1 - gap)], reverse=True)[:3] or [float(q[25][-1])]
    RES = sorted([v for v in cand if v > P0 * (1 + gap)])[:3] or [float(q[75][-1])]
    S1, R1 = SUP[0], RES[0]
    SC = {"Bajista": dict(lo=None, hi=S1, inv=R1, col="#e34948"),
          "Base": dict(lo=S1, hi=R1, inv=None, col="#2a78d6"),
          "Alcista": dict(lo=R1, hi=None, inv=S1, col="#1baf7a")}
    for s in SC.values():
        lo = -np.inf if s["lo"] is None else s["lo"]; hi = np.inf if s["hi"] is None else s["hi"]
        s["p"] = float(((fin >= lo) & (fin < hi)).mean() * 100)
        sub = fin[(fin >= lo) & (fin < hi)]
        s["rango_80"] = [float(np.percentile(sub, 10)), float(np.percentile(sub, 90))] if len(sub) else [np.nan, np.nan]
    mas_prob = max(SC, key=lambda k_: SC[k_]["p"])

    lp = np.log(paths)
    inside = ((lp >= np.percentile(lp, 25, axis=0)) & (lp <= np.percentile(lp, 75, axis=0))).mean(1)
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

    pasos = [6, 12, 24]
    grid = np.round(np.linspace(q[5][-1], q[95][-1], 7) / 0.005) * 0.005
    levels = sorted({round(float(v), 3) for v in list(grid) + SUP + RES}, reverse=True)
    res = {"modelo": "24 h: velas de 1 h, HAR Garman-Klass + cuantiles conformales con ajuste horario",
           "horizonte_dias": 1, "horizonte_texto": "24 horas", "paso": "hora",
           "precio": P0, "precio_hora_utc": f"{ahora:%Y-%m-%d %H:%M}", "hasta": f"{ahora + pd.Timedelta(hours=H):%Y-%m-%d %H:%M} UTC",
           "ultima_vela_cerrada": str(x.index[-1]), "sigma_diaria_prevista_pct": sig24 * 100,
           "sigma_horaria_prevista_pct": float(sig[t] * 100), "n_trayectorias_analogas": int(len(zz)),
           "cono": {"etiquetas": [f"{p_} h" for p_ in pasos], **{f"P{p}": [float(q[p][p_ - 1]) for p_ in pasos] for p in PCT}},
           "niveles": {"soportes": SUP, "resistencias": RES},
           "escenarios": {k_: {"prob_pct": round(v["p"], 1), "rango": [v["lo"], v["hi"]], "rango_80_dentro": v["rango_80"],
                               "invalidacion": v["inv"]} for k_, v in SC.items()},
           "escenario_mas_probable": mas_prob,
           "simulaciones": {k_: {kk: vv for kk, vv in v.items() if kk != "camino"} for k_, v in sims.items()},
           "prob_niveles": {f"{L:.3f}": {"cierre_final_por_encima_pct": float((fin > L).mean() * 100),
                                         "toca_pct": float(((paths.max(1) >= L) if L > P0 else (paths.min(1) <= L)).mean() * 100)}
                            for L in levels},
           "imagen": "xrp_escenarios_v2.png"}
    json.dump(res, open(os.path.join(OUT, "resultados_v2.json"), "w"), indent=1, ensure_ascii=False)

    # gráfico
    hx = xa.close[-96:]
    fut = pd.date_range(ahora, periods=H + 1, freq="h")
    band = lambda p: np.concatenate([[P0], q[p]])
    span = pd.Timedelta(hours=96 + H)
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.fill_between(fut, band(5), band(95), color="#2a78d6", alpha=0.13, lw=0, label="Cono P5–P95 (90 %)")
    ax.fill_between(fut, band(25), band(75), color="#2a78d6", alpha=0.28, lw=0, label="Cono P25–P75 (50 %)")
    ax.plot(fut, band(50), color="#2a78d6", lw=1.6, ls="--", label="Mediana (P50)")
    for (name, s), col in zip(sims.items(), ["#4a3aa7", "#eb6834"]):
        ax.plot(fut, np.concatenate([[P0], s["camino"]]), color=col, lw=1.5, label=f"{name} (ejemplo típico): {s['final']:.4f} $")
    ax.plot(hx.index, hx.values, color="#0b0b0b", lw=1.4, label="XRP/USD velas de 1 h (Bitstamp)")
    for p in PCT:
        ax.annotate(f"P{p} {q[p][-1]:.4f}", (fut[-1], q[p][-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")
    for j, v in enumerate(SUP):
        ax.axhline(v, color="#008300", lw=0.8, ls=(0, (4, 3)), alpha=0.7); ax.text(hx.index[0], v, f" S{j+1} {v:.4f}", va="bottom", fontsize=8, color="#008300")
    for j, v in enumerate(RES):
        ax.axhline(v, color="#e34948", lw=0.8, ls=(0, (4, 3)), alpha=0.7); ax.text(hx.index[0], v, f" R{j+1} {v:.4f}", va="bottom", fontsize=8, color="#b8302f")
    xe = fut[-1] + span * 0.07
    ylo, yhi = q[5][-1] * 0.99, q[95][-1] * 1.01
    for j, (name, s) in enumerate(SC.items()):
        xx = xe + span * (0.045 * j)
        lo = s["lo"] if s["lo"] is not None else max(s["rango_80"][0], ylo)
        hi = s["hi"] if s["hi"] is not None else min(s["rango_80"][1], yhi)
        ax.plot([xx, xx], [lo, hi], color=s["col"], lw=6, solid_capstyle="round")
        txt = f"< {s['hi']:.4f}" if s["lo"] is None else (f"> {s['lo']:.4f}" if s["hi"] is None else f"{s['lo']:.4f}–{s['hi']:.4f}")
        ax.text(xx + span * 0.006, (lo + hi) / 2, f"{name}\n{s['p']:.0f} %\n{txt}", fontsize=8, va="center")
    ax.axvline(ahora, color="#52514e", lw=0.8)
    ax.text(ahora, ax.get_ylim()[1], f" Ahora {ahora:%d-%m %H:%M} UTC\n precio {P0:.4f} $", va="top", fontsize=8, color="#52514e")
    ax.set_xlim(hx.index[0], xe + span * 0.15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh"))
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    ax.set_ylabel("USD por XRP")
    ax.set_title(f"XRP/USD — escenarios para las próximas 24 horas (hasta {fut[-1]:%d-%m %H:%M} UTC)\nEstimación probabilística, no una predicción",
                 fontsize=13, loc="left")
    bt_p = os.path.join(OUT, "backtest_v1_vs_v2.json")
    cal = ""
    if os.path.exists(bt_p):
        bt = json.load(open(bt_p))
        if bt.get("horizonte_texto") == "24 horas":
            a = bt["desde el último año"]["v2_todo"]
            cal = (f"Backtest walk-forward del último año: dentro de P5–P95 {a['cobertura_90']:.0f} % (ideal 90 %), "
                   f"por encima de P95 {a['bandas_pct']['>P95']:.0f} % (ideal 5 %).")
    fig.text(0.01, 0.01, f"Datos: Bitstamp XRP/USD velas de 1 h. Modelo: volatilidad HAR (1 h, 1 d, 1 sem, 1 mes) σ 24 h {sig24*100:.2f} % + cuantiles conformales "
             f"sobre {len(zz)} ventanas de 24 h que empezaron a las {ahora.hour:02d}h UTC. Niveles: pivotes horarios y rangos de 24/72 h.\n{cal} "
             f"Generado {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC.", fontsize=7.2, color="#52514e")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.04, 1.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "xrp_escenarios_v2.png")); plt.close(fig)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        backtest()
    elif "--verificar" in sys.argv:
        verificar()
    else:
        escenarios()
