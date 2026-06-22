"""
Colector de liquidaciones via REST polling de OKX (publico, sin API key).
Endpoint: GET /api/v5/public/liquidation-orders?instType=SWAP&instFamily=BTC-USDT&state=filled

Cada liquidacion:
  side     = buy  → short liquidado
  side     = sell → long  liquidado
  bkPx     = precio de liquidacion
  sz       = cantidad BTC
  ts       = timestamp ms

Hace polling cada 30s, deduplica por ts+bkPx+sz, guarda buffer rolling 24h.

Uso (en VPS, como servicio):
    python src/liq_collector.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")
OUTPUT_FILE = os.path.join(REPORTS_DIR, "liq_rolling.json")
ROLLING_HOURS = 24
POLL_INTERVAL = 30  # segundos entre consultas

OKX_URL = (
    "https://www.okx.com/api/v5/public/liquidation-orders"
    "?instType=SWAP&instFamily=BTC-USDT&state=filled&limit=100"
)

os.makedirs(REPORTS_DIR, exist_ok=True)


def load_buffer() -> list[dict]:
    if os.path.exists(OUTPUT_FILE):
        try:
            return json.load(open(OUTPUT_FILE))
        except Exception:
            return []
    return []


def save_buffer(buf: list[dict]) -> None:
    cutoff = (time.time() - ROLLING_HOURS * 3600) * 1000
    trimmed = [e for e in buf if e["T"] >= cutoff]
    json.dump(trimmed, open(OUTPUT_FILE, "w"))
    return trimmed


def fetch_liquidations() -> list[dict]:
    try:
        req = urllib.request.Request(OKX_URL, headers={"User-Agent": "btc-bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        if data.get("code") != "0":
            print(f"[liq] OKX error: {data.get('msg')}")
            return []
        results = []
        for item in data.get("data", []):
            for d in item.get("details", []):
                side = "BUY" if d["side"] == "buy" else "SELL"
                try:
                    results.append({
                        "T": int(d["ts"]),
                        "p": float(d["bkPx"]),
                        "q": float(d["sz"]),
                        "S": side,
                        "usd": float(d["bkPx"]) * float(d["sz"]),
                    })
                except Exception:
                    continue
        return results
    except Exception as e:
        print(f"[liq] fetch error: {e}")
        return []


def run():
    buffer = load_buffer()
    seen = {(e["T"], e["p"], e["q"]) for e in buffer}
    print(f"[liq] iniciando. Buffer: {len(buffer)} eventos")

    while True:
        new_items = fetch_liquidations()
        added = 0
        for e in new_items:
            key = (e["T"], e["p"], e["q"])
            if key not in seen:
                seen.add(key)
                buffer.append(e)
                added += 1

        if added > 0:
            print(f"[liq] +{added} nuevas liquidaciones (total: {len(buffer)})")

        buffer = save_buffer(buffer)
        # Limpiar seen de entradas antiguas para no crecer indefinidamente
        cutoff = (time.time() - ROLLING_HOURS * 3600) * 1000
        seen = {(e["T"], e["p"], e["q"]) for e in buffer if e["T"] >= cutoff}

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
