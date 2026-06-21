"""
Colector de liquidaciones en tiempo real via WebSocket de Binance.
Stream: wss://fstream.binance.com/ws/btcusdt@forceOrder

Cada liquidacion forzada de BTCUSDT llega con:
  S  = BUY  → short liquidado (alguien apostaba a la baja y fue liquidado)
  S  = SELL → long  liquidado (alguien apostaba al alza y fue liquidado)
  p  = precio de ejecucion
  q  = cantidad BTC
  T  = timestamp ms

El proceso escribe en /root/btc-bot/reports/liq_rolling.json un buffer
rolling de las ultimas 24h. El señal diaria lo lee para mostrar clusters.

Uso (en VPS, como servicio):
    python src/liq_collector.py
"""
from __future__ import annotations

import json
import os
import time
import threading
import websocket  # pip install websocket-client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")
OUTPUT_FILE = os.path.join(REPORTS_DIR, "liq_rolling.json")
ROLLING_HOURS = 24

os.makedirs(REPORTS_DIR, exist_ok=True)

buffer: list[dict] = []
lock = threading.Lock()


def load_buffer() -> list[dict]:
    if os.path.exists(OUTPUT_FILE):
        try:
            return json.load(open(OUTPUT_FILE))
        except Exception:
            return []
    return []


def save_buffer() -> None:
    cutoff = (time.time() - ROLLING_HOURS * 3600) * 1000
    with lock:
        trimmed = [e for e in buffer if e["T"] >= cutoff]
        buffer.clear()
        buffer.extend(trimmed)
        json.dump(buffer, open(OUTPUT_FILE, "w"))


def on_message(ws, message: str) -> None:
    try:
        data = json.loads(message)
        o = data.get("o", data)
        entry = {
            "T": int(o["T"]),
            "p": float(o["p"]),
            "q": float(o["q"]),
            "S": o["S"],
            "usd": float(o["p"]) * float(o["q"]),
        }
        with lock:
            buffer.append(entry)
        # Guardar cada 500 eventos
        if len(buffer) % 500 == 0:
            save_buffer()
    except Exception as e:
        print(f"[liq] parse error: {e}")


def on_error(ws, error):
    print(f"[liq] error: {error}")


def on_close(ws, *args):
    print("[liq] conexion cerrada — reconectando en 5s...")
    time.sleep(5)
    start()


def on_open(ws):
    print("[liq] conectado a Binance forceOrder stream")


def start():
    global buffer
    buffer = load_buffer()
    print(f"[liq] buffer cargado: {len(buffer)} liquidaciones")
    ws = websocket.WebSocketApp(
        "wss://fstream.binance.com/ws/btcusdt@forceOrder",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    # Guardar periodicamente cada 60s
    def periodic_save():
        while True:
            time.sleep(60)
            save_buffer()
            print(f"[liq] guardado: {len(buffer)} liquidaciones en rolling")
    t = threading.Thread(target=periodic_save, daemon=True)
    t.start()
    ws.run_forever(ping_interval=30, ping_timeout=10)


if __name__ == "__main__":
    start()
