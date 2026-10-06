"""
¿Se puede predecir el precio EXACTO a h días mejor que suponer "se queda igual"?
Compara walk-forward (solo datos pasados en cada fecha) varios predictores del retorno a h días:
  sin_cambio (paseo aleatorio), mediana v2, momentum, reversión, Ridge y Gradient Boosting
con 20+ variables (precio, volumen, volatilidad, BTC, ETH, día de la semana).
Uso: HORIZONTE=7 python analysis/mejora_puntual.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelo_v2 as m
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = m.H
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
x = m.load_xrp(); btc = m.load("bitstamp_btcusd_1d.csv"); eth = m.load("bitstamp_ethusd_1d.csv")
c = x.close; lc = np.log(c); r = lc.diff()
b = np.log(btc.close.reindex(x.index).ffill()); e = np.log(eth.close.reindex(x.index).ffill())
d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
vol = np.log(x.volume * c)
F = pd.DataFrame({
    **{f"ret{k}": lc.diff(k) for k in (1, 3, 7, 14, 30, 90)},
    **{f"btc{k}": b.diff(k) for k in (1, 7, 30)}, **{f"eth{k}": e.diff(k) for k in (1, 7)},
    "rel7": lc.diff(7) - b.diff(7), "rel30": lc.diff(30) - b.diff(30),
    "dist20": lc - np.log(c.rolling(20).mean()), "dist50": lc - np.log(c.rolling(50).mean()),
    "dist200": lc - np.log(c.rolling(200).mean()), "rsi": 100 - 100 / (1 + up / dn),
    "vol7": r.rolling(7).std(), "vol30": r.rolling(30).std(), "volratio": np.log(r.rolling(7).std() / r.rolling(90).std()),
    "volumen_z": (vol - vol.rolling(90).mean()) / vol.rolling(90).std(),
    "rango": np.log(x.high / x.low), "dow": x.index.dayofweek,
    "max30": lc - np.log(x.high.rolling(30).max()), "min30": lc - np.log(x.low.rolling(30).min()),
})
n = len(c); y = np.full(n, np.nan); y[: n - H] = lc.values[H:] - lc.values[:-H]
Xv = F.values; ok = np.isfinite(Xv).all(1)
sig = m.har_sigma(x); Z = m.z_paths(lc.values, sig)
start = int(np.where(x.index >= "2020-06-01")[0][0])

pred = {k: np.full(n, np.nan) for k in ["sin_cambio", "mediana_v2", "momentum_7", "reversion_7", "ridge", "gboost"]}
models = {}
for t in range(start, n - H):
    pred["sin_cambio"][t] = 0.0
    qv, _ = m.conformal_quantiles(Z, sig, t, qs=np.array([0.5]))
    pred["mediana_v2"][t] = qv[0, -1] if qv is not None else 0.0
    pred["momentum_7"][t] = F["ret7"].iloc[t] * H / 7
    pred["reversion_7"][t] = -F["ret7"].iloc[t] * H / 7 * 0.5
    if t % 30 == 0 or not models:
        idx = np.where(ok[: t - H + 1] & np.isfinite(y[: t - H + 1]))[0]
        models["ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=50)).fit(Xv[idx], y[idx])
        models["gboost"] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_depth=3,
                                                         min_samples_leaf=40, l2_regularization=1.0,
                                                         random_state=0).fit(Xv[idx], y[idx])
    if ok[t]:
        pred["ridge"][t] = models["ridge"].predict(Xv[t:t + 1])[0]
        pred["gboost"][t] = models["gboost"].predict(Xv[t:t + 1])[0]

ev = np.arange(start, n - H); ev = ev[np.isfinite(pred["gboost"][ev])]
real = y[ev]
out = {"horizonte_dias": H, "n_fechas": int(len(ev)), "desde": str(x.index[ev[0]].date()), "hasta": str(x.index[ev[-1]].date())}
for k, p in pred.items():
    p = p[ev]; err = np.abs(np.exp(p) / np.exp(real) - 1) * 100          # error del precio previsto, %
    dirok = np.sign(p) == np.sign(real)
    out[k] = {"error_medio_pct": round(float(err.mean()), 2), "error_mediano_pct": round(float(np.median(err)), 2),
              "acierto_±1%_pct": round(float((err <= 1).mean() * 100), 1), "acierto_±3%_pct": round(float((err <= 3).mean() * 100), 1),
              "direccion_acierto_pct": round(float(dirok[p != 0].mean() * 100), 1) if (p != 0).any() else None,
              "error_medio_2024+_pct": round(float(err[x.index[ev] >= "2024-01-01"].mean()), 2)}
json.dump(out, open(os.path.join(OUT, f"mejora_puntual_{H}d.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
