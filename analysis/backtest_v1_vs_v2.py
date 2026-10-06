"""Compara walk-forward el modelo v1 con las variantes del v2 y la señal direccional."""
import json, os
import numpy as np, pandas as pd
import modelo_v2 as m

rng = np.random.default_rng(7)
x = m.load_xrp(); btc = m.load("bitstamp_btcusd_1d.csv")
lc = np.log(x.close.values); r = np.log(x.close).diff().dropna()
n = len(lc); H = m.H
sig = m.har_sigma(x)
Z = m.z_paths(lc, sig)
dates = x.index
start = int(np.where(dates >= "2020-06-01")[0][0])
variants = {"v2_todo": dict(window=None, demean=False), "v2_3a": dict(window=1095, demean=False),
            "v2_todo_sin_deriva": dict(window=None, demean=True), "v2_3a_sin_deriva": dict(window=1095, demean=True)}
rows = []
for t in range(start, n - H, 2):
    real = lc[t + H] - lc[t]
    row = {"fecha": dates[t], "real": real}
    q1 = m.v1_quantiles(r.iloc[: t], rng)
    row["v1"] = q1
    for k, kw in variants.items():
        q, _ = m.conformal_quantiles(Z, sig, t, **kw)
        row[k] = q[:, -1]
    rows.append(row)

def evaluate(rows, key):
    band = np.array([np.searchsorted(rw[key], rw["real"]) for rw in rows])
    freq = np.bincount(band, minlength=6) / len(band) * 100
    pin = np.mean([m.pinball(rw[key], rw["real"]) for rw in rows])
    isc = np.mean([m.interval_score(rw[key][0], rw[key][-1], rw["real"]) for rw in rows])
    width = np.mean([np.exp(rw[key][-1]) - np.exp(rw[key][0]) for rw in rows])
    return {"bandas_pct": dict(zip(["<P5", "P5-25", "P25-50", "P50-75", "P75-95", ">P95"], freq.round(1).tolist())),
            "cobertura_90": round(freq[1:5].sum(), 1), "cobertura_50": round(freq[2:4].sum(), 1),
            "error_calibracion": round(float(np.abs(freq - [5, 20, 25, 25, 20, 5]).sum()), 1),
            "pinball": round(pin * 1000, 3), "interval_score_90": round(isc, 4), "ancho_90_relativo": round(width, 3)}

out = {}
periods = {"desde 2020-06": rows, "desde 2024-01": [rw for rw in rows if rw["fecha"] >= pd.Timestamp("2024-01-01")]}
for pname, rr in periods.items():
    out[pname] = {k: evaluate(rr, k) for k in ["v1"] + list(variants)}

# Señal direccional
X = m.direction_features(x, btc)
p, y, _ = m.walk_forward_direction(X, lc)
ev = np.arange(start, n - H)
pv, yv = p[ev], y[ev]
base = np.array([np.nanmean(y[: max(t - H + 1, 1)]) for t in ev])          # frecuencia histórica de subidas
z_up = np.array([(m.conformal_quantiles(Z, sig, t)[1][:, -1] > 0).mean() for t in ev[::5]])
dir_out = {
    "n": int(len(ev)),
    "logistica_acierto_pct": round(float(((pv > 0.5) == yv).mean() * 100), 1),
    "logistica_brier": round(float(np.mean((pv - yv) ** 2)), 4),
    "climatologia_brier": round(float(np.mean((base - yv) ** 2)), 4),
    "moneda_brier": 0.25,
    "siempre_sube_acierto_pct": round(float(yv.mean() * 100), 1),
    "logistica_acierto_confianza_alta_pct": round(float(((pv[np.abs(pv - .5) > .1] > .5) == yv[np.abs(pv - .5) > .1]).mean() * 100), 1),
    "n_confianza_alta": int((np.abs(pv - .5) > .1).sum()),
    "por_anio": {str(yr): round(float(((pv[s] > .5) == yv[s]).mean() * 100), 1)
                 for yr in sorted(set(dates[ev].year)) for s in [dates[ev].year == yr]},
}
out["direccion"] = dir_out
out["horizonte_dias"] = H
out["hasta"] = str(rows[-1]["fecha"].date())
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "output", "backtest_v1_vs_v2.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
