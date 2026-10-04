"""Descarga velas diarias (OHLCV) de exchanges públicos y las guarda en analysis/data/."""
import json, os, time, urllib.request, csv

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def bitstamp(pair, start=1514764800):
    rows, t = {}, start
    while True:
        d = get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step=86400&limit=1000&start={t}")
        data = d["data"]["ohlc"]
        if not data:
            break
        for c in data:
            rows[int(c["timestamp"])] = [c["open"], c["high"], c["low"], c["close"], c["volume"]]
        last = int(data[-1]["timestamp"])
        if last <= t or len(data) < 1000:
            break
        t = last + 86400
        time.sleep(1)
    return rows

def kraken(pair, interval=1440):
    d = get(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}")
    key = [k for k in d["result"] if k != "last"][0]
    return {int(c[0]): [c[1], c[2], c[3], c[4], c[6]] for c in d["result"][key]}

def save(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "date", "open", "high", "low", "close", "volume"])
        for ts in sorted(rows):
            w.writerow([ts, time.strftime("%Y-%m-%d", time.gmtime(ts))] + rows[ts])
    print(name, len(rows))

jobs = [
    ("bitstamp_xrpusd_1d.csv", lambda: bitstamp("xrpusd")),
    ("bitstamp_btcusd_1d.csv", lambda: bitstamp("btcusd")),
    ("bitstamp_ethusd_1d.csv", lambda: bitstamp("ethusd")),
    ("bitstamp_solusd_1d.csv", lambda: bitstamp("solusd", 1609459200)),
    ("bitstamp_xlmusd_1d.csv", lambda: bitstamp("xlmusd")),
    ("bitstamp_adausd_1d.csv", lambda: bitstamp("adausd", 1609459200)),
    ("kraken_xrpusd_1d.csv", lambda: kraken("XRPUSD")),
    ("kraken_xrpusd_4h.csv", lambda: kraken("XRPUSD", 240)),
    ("kraken_solusd_1d.csv", lambda: kraken("SOLUSD")),
    ("kraken_xlmusd_1d.csv", lambda: kraken("XLMUSD")),
    ("kraken_adausd_1d.csv", lambda: kraken("ADAUSD")),
    ("kraken_hbarusd_1d.csv", lambda: kraken("HBARUSD")),
]
for name, fn in jobs:
    try:
        save(name, fn())
    except Exception as e:
        print("FAIL", name, e)

# Instantánea de mercado (CoinGecko) para contraste
try:
    ids = "ripple,bitcoin,ethereum,solana,stellar,cardano,hedera-hashgraph"
    snap = get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=" + ids +
               "&price_change_percentage=7d,30d,1y")
    snap = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data": snap}
    json.dump(snap, open(os.path.join(OUT, "coingecko_markets.json"), "w"), indent=1)
    print("coingecko ok")
except Exception as e:
    print("FAIL coingecko", e)
# Funding / OI de Bitstamp no existe; intentamos OKX (no bloqueado en EE. UU. para datos públicos)
for name, url in [
    ("okx_xrp_funding.json", "https://www.okx.com/api/v5/public/funding-rate-history?instId=XRP-USDT-SWAP&limit=100"),
    ("okx_xrp_oi.json", "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=XRP-USDT-SWAP"),
    ("okx_xrp_ls.json", "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy=XRP&period=1D"),
    ("okx_xrp_book.json", "https://www.okx.com/api/v5/market/books?instId=XRP-USDT&sz=400"),
    ("kraken_xrp_book.json", "https://api.kraken.com/0/public/Depth?pair=XRPUSD&count=500"),
]:
    try:
        json.dump(get(url), open(os.path.join(OUT, name), "w"))
        print("ok", name)
    except Exception as e:
        print("FAIL", name, e)
