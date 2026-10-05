"""
Modelo v2 del cono de XRP: volatilidad HAR con Garman-Klass + cuantiles conformales
(empíricos, sin suponer una distribución) + trayectorias análogas históricas.
Todo se evalúa walk-forward: en cada fecha solo se usa información disponible ese día.
"""
import os
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(__file__), "data")
H = 30
QS = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
MIN_TRAIN = 365          # muestras mínimas para ajustar el HAR y los cuantiles
REFIT = 20               # reajuste del HAR cada N días

def load(name):
    df = pd.read_csv(os.path.join(D, name), parse_dates=["date"]).set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[df.close > 0]

def load_xrp():
    x = load("bitstamp_xrpusd_1d.csv").iloc[:-1]      # se descarta la vela del día en curso
    return x

# ---------------------------------------------------------------- volatilidad HAR
def har_features(df):
    o, h, l, c = (np.log(df[k]) for k in ("open", "high", "low", "close"))
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2     # varianza Garman-Klass
    r = c.diff()
    gk = np.maximum(gk, 0.25 * r ** 2).clip(lower=1e-6)            # incluye saltos entre velas
    F = pd.DataFrame({"g1": gk, "g7": gk.rolling(7).mean(), "g30": gk.rolling(30).mean(),
                      "g90": gk.rolling(90).mean(), "r30": (r ** 2).rolling(30).mean()})
    return np.log(F), r

def har_sigma(df, h=H):
    """sigma diaria prevista para los próximos h días (walk-forward)."""
    F, r = har_features(df)
    n = len(df)
    fut = (r ** 2).rolling(h).mean().shift(-h).values            # varianza realizada futura
    y = np.log(np.maximum(fut, 1e-8))
    X = np.column_stack([np.ones(n), F.values])
    ok = np.isfinite(X).all(1)
    pred = np.full(n, np.nan); beta = None
    for t in range(n):
        if not ok[t]:
            continue
        if beta is None or t % REFIT == 0:
            idx = np.where(ok[: max(t - h + 1, 0)] & np.isfinite(y[: max(t - h + 1, 0)]))[0]
            if len(idx) >= MIN_TRAIN:
                beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        if beta is not None:
            pred[t] = X[t] @ beta
    return np.sqrt(np.exp(pred))

# ---------------------------------------------------------------- cuantiles conformales
def z_paths(lc, sig, h=H):
    """retorno acumulado estandarizado por la sigma prevista, para cada horizonte 1..h."""
    n = len(lc); Z = np.full((n, h), np.nan)
    for k in range(1, h + 1):
        Z[: n - k, k - 1] = (lc[k:] - lc[:-k]) / (sig[: n - k] * np.sqrt(k))
    return Z

def conformal_quantiles(Z, sig, t, qs=QS, window=None, demean=False, h=H):
    """cuantiles del retorno log a cada horizonte para la fecha t, solo con datos conocidos en t."""
    hi = t - h + 1                         # ventanas cuyo resultado a 30 d ya se conoce en t
    lo = 0 if window is None else max(0, hi - window)
    zz = Z[lo:hi]; zz = zz[np.isfinite(zz).all(1)]
    if len(zz) < MIN_TRAIN:
        return None, None
    if demean:
        zz = zz - zz.mean(0)
    k = np.sqrt(np.arange(1, h + 1))
    return np.quantile(zz, qs, axis=0) * sig[t] * k, zz

# ---------------------------------------------------------------- modelo v1 (referencia)
def v1_quantiles(r_hist, rng, n=4000, h=H, qs=QS):
    ew = np.sqrt((r_hist[-250:] ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1])
    s = 0.5 * ew + 0.5 * r_hist[-90:].std()
    nu = 3.5
    zt = rng.standard_t(nu, size=(n // 2, h)) / np.sqrt(nu / (nu - 2))
    a = np.cumsum(s * zt - 0.5 * s ** 2, axis=1)[:, -1]
    hr = r_hist[-1095:].values; hr = (hr - hr.mean()) * (s / hr.std())
    st = rng.integers(0, len(hr) - 5, size=(n // 2, 6))
    b = hr[(st[:, :, None] + np.arange(5)).reshape(n // 2, -1)[:, :h]].sum(1)
    return np.quantile(np.concatenate([a, b]), qs)

# ---------------------------------------------------------------- señal direccional
def direction_features(df, btc):
    c = df.close; lc = np.log(c)
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    b = np.log(btc.close.reindex(df.index).ffill())
    r = lc.diff()
    X = pd.DataFrame({
        "ret7": lc.diff(7), "ret30": lc.diff(30), "ret90": lc.diff(90),
        "dist50": lc - np.log(c.rolling(50).mean()), "dist200": lc - np.log(c.rolling(200).mean()),
        "rsi": 100 - 100 / (1 + up / dn), "btc30": b.diff(30), "rel30": lc.diff(30) - b.diff(30),
        "volratio": np.log(r.rolling(7).std() / r.rolling(90).std()),
    })
    return X

def walk_forward_direction(X, lc, h=H, refit=30):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    n = len(lc); y = np.full(n, np.nan); y[: n - h] = (lc[h:] > lc[:-h]).astype(float)
    Xv = X.values; ok = np.isfinite(Xv).all(1)
    p = np.full(n, np.nan); m = None
    for t in range(n):
        if not ok[t]:
            continue
        if m is None or t % refit == 0:
            idx = np.where(ok[: max(t - h + 1, 0)] & np.isfinite(y[: max(t - h + 1, 0)]))[0]
            if len(idx) >= MIN_TRAIN:
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(Xv[idx], y[idx])
        if m is not None:
            p[t] = m.predict_proba(Xv[t:t + 1])[0, 1]
    return p, y, m

# ---------------------------------------------------------------- métricas
def pinball(qpred, real, qs=QS):
    return np.mean([max(q * (real - v), (q - 1) * (real - v)) for q, v in zip(qs, qpred)])

def interval_score(lo, hi, real, alpha=0.10):
    return (hi - lo) + 2 / alpha * max(lo - real, 0) + 2 / alpha * max(real - hi, 0)
