"""
Verificación a ciegas: el modelo v2 "predice" desde una fecha pasada usando SOLO los datos
disponibles ese día y se compara con el precio que siguió de verdad.

Uso:  python analysis/verificar.py                  (fecha aleatoria de hace ~15 días)
      FECHA=2026-09-20 python analysis/verificar.py  (fecha concreta)
      HORIZONTE=7 / 15 ...                            (días a comprobar; por defecto, hasta hoy)
Salida: analysis/output/verificacion.png y verificacion.json
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
PCT = [5, 25, 50, 75, 95]

full = m.load_xrp()                                   # velas cerradas hasta ayer
fecha = os.environ.get("FECHA")
if fecha:
    corte = pd.Timestamp(fecha)
else:                                                 # aleatoria entre hace 13 y 17 días
    rng = np.random.default_rng()
    corte = full.index[-1] - pd.Timedelta(days=int(rng.integers(13, 18)))
i0 = int(np.searchsorted(full.index, corte, side="right")) - 1
corte = full.index[i0]
H = int(os.environ.get("HORIZONTE", len(full) - 1 - i0))
H = max(1, min(H, len(full) - 1 - i0))

# --- el modelo solo ve datos hasta el cierre del día de corte
x = full.iloc[: i0 + 1]
lc = np.log(x.close.values); t = len(lc) - 1
sig = m.har_sigma(x, h=H)
Z = m.z_paths(lc, sig, h=H)
qlog, zz = m.conformal_quantiles(Z, sig, t, qs=np.array(PCT) / 100, h=H)
P0 = float(x.close.iloc[-1])
q = {p: P0 * np.exp(qlog[j]) for j, p in enumerate(PCT)}
paths = P0 * np.exp(zz * sig[t] * np.sqrt(np.arange(1, H + 1)))

# --- lo que pasó de verdad
real = full.close.iloc[i0 + 1: i0 + 1 + H]
filas = []
for d, (fecha_d, pr) in enumerate(real.items()):
    pct = float((paths[:, d] < pr).mean() * 100)          # percentil del precio real dentro del cono
    filas.append({"dia": d + 1, "fecha": str(fecha_d.date()), "real": float(pr),
                  "mediana": float(q[50][d]), "P5": float(q[5][d]), "P25": float(q[25][d]),
                  "P75": float(q[75][d]), "P95": float(q[95][d]), "percentil_real": round(pct, 1),
                  "error_mediana_pct": round(float((q[50][d] / pr - 1) * 100), 2),
                  "dentro_50": bool(q[25][d] <= pr <= q[75][d]), "dentro_90": bool(q[5][d] <= pr <= q[95][d])})
df = pd.DataFrame(filas)
res = {"fecha_corte": str(corte.date()), "precio_corte": P0, "horizonte_dias": H,
       "sigma_diaria_pct": float(sig[t] * 100),
       "dias_dentro_50_pct": float(df.dentro_50.mean() * 100), "dias_dentro_90_pct": float(df.dentro_90.mean() * 100),
       "error_medio_mediana_pct": float(df.error_mediana_pct.abs().mean()),
       "error_medio_sin_cambio_pct": float(((P0 / real - 1).abs() * 100).mean()),
       "final": filas[-1], "dias": filas}
json.dump(res, open(os.path.join(OUT, "verificacion.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# --- gráfico
hx = full.close.iloc[max(0, i0 - 60): i0 + 1]
fut = [corte] + list(real.index)
band = lambda p: np.concatenate([[P0], q[p]])
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=140)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.fill_between(fut, band(5), band(95), color="#2a78d6", alpha=0.13, lw=0, label="Cono P5–P95 previsto (90 %)")
ax.fill_between(fut, band(25), band(75), color="#2a78d6", alpha=0.28, lw=0, label="Cono P25–P75 previsto (50 %)")
ax.plot(fut, band(50), color="#2a78d6", lw=1.6, ls="--", label="Mediana prevista")
ax.plot(hx.index, hx.values, color="#0b0b0b", lw=1.5, label="Precio conocido el día de corte")
ax.plot(fut, [P0] + list(real.values), color="#eb6834", lw=2.2, marker="o", ms=3.5, label="Precio REAL que siguió")
ax.axvline(corte, color="#52514e", lw=0.8)
ax.text(corte, ax.get_ylim()[1], f" Corte {corte:%d-%m-%Y}\n {P0:.4f} $", va="top", fontsize=8, color="#52514e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.set_ylabel("USD por XRP")
f = filas[-1]
ax.set_title(f"Verificación a ciegas: previsión desde el {corte:%d-%m-%Y} a {H} días vs precio real\n"
             f"Día {H}: real {f['real']:.4f} USD · mediana prevista {f['mediana']:.4f} USD · el real cayó en el percentil {f['percentil_real']:.0f} del cono",
             loc="left", fontsize=11.5)
ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.text(0.01, 0.01, f"El modelo solo usó datos hasta el {corte:%d-%m-%Y}. Días dentro del rango del 50 %: {res['dias_dentro_50_pct']:.0f} % · "
         f"dentro del 90 %: {res['dias_dentro_90_pct']:.0f} %. Error medio de la mediana {res['error_medio_mediana_pct']:.1f} % "
         f"(suponer «sin cambio»: {res['error_medio_sin_cambio_pct']:.1f} %).", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "verificacion.png"))
print(df[["dia", "fecha", "real", "mediana", "P5", "P95", "percentil_real", "dentro_50", "dentro_90"]].to_string(index=False))
print(json.dumps({k: v for k, v in res.items() if k not in ("dias", "final")}, indent=1, ensure_ascii=False))
