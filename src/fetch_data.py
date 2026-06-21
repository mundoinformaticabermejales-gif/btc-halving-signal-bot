"""
Fase 0 - Descargador de histórico OHLCV de Bitcoin.

- Para TRADING real usaremos Kraken (BTC/EUR). Su API pública OHLC, sin embargo,
  limita el histórico que devuelve, así que para el BACKTEST de ciclos (que
  necesita varios años) usamos Binance como fuente de histórico largo.
- Solo datos públicos: NO hace falta ninguna API key para esto.

Uso:
    python src/fetch_data.py            # descarga por defecto (Binance BTC/USDT diario + Kraken BTC/EUR reciente)
    python src/fetch_data.py --probe    # solo diagnostica cuánto histórico da cada exchange
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import ccxt
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_ohlcv_paginated(exchange, symbol: str, timeframe: str, since_ms: int,
                          max_pages: int = 1000) -> pd.DataFrame:
    """Descarga OHLCV paginando con `since` hasta alcanzar el presente.

    Se detiene cuando el exchange deja de devolver velas nuevas (para los que
    limitan el histórico, como Kraken) o cuando llega a la última vela.
    """
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    all_rows: list[list] = []
    since = since_ms
    last_ts = None

    for _ in range(max_pages):
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except ccxt.NetworkError as e:
            print(f"  red lenta, reintento: {e}", file=sys.stderr)
            time.sleep(2)
            continue
        if not batch:
            break
        all_rows += batch
        new_last = batch[-1][0]
        if new_last == last_ts:  # el exchange no avanza -> tope de histórico alcanzado
            break
        last_ts = new_last
        since = new_last + tf_ms
        time.sleep((exchange.rateLimit or 200) / 1000)
        if new_last >= exchange.milliseconds() - tf_ms:
            break

    if not all_rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def save(df: pd.DataFrame, name: str) -> None:
    pq = os.path.join(DATA_DIR, f"{name}.parquet")
    csv = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    span = f"{df['date'].min().date()} -> {df['date'].max().date()}" if len(df) else "vacío"
    print(f"  guardado {name}: {len(df)} velas [{span}]")


def probe() -> None:
    """Diagnóstico: cuánto histórico diario devuelve cada exchange."""
    targets = [("binance", "BTC/USDT"), ("kraken", "BTC/USD"), ("kraken", "BTC/EUR")]
    since = ccxt.binance().parse8601("2013-01-01T00:00:00Z")
    for ex_id, symbol in targets:
        ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
        try:
            df = fetch_ohlcv_paginated(ex, symbol, "1d", since)
            span = f"{df['date'].min().date()} -> {df['date'].max().date()}" if len(df) else "sin datos"
            print(f"{ex_id:8} {symbol:9} velas diarias: {len(df):5}  [{span}]")
        except Exception as e:  # noqa: BLE001
            print(f"{ex_id:8} {symbol:9} ERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="solo diagnosticar cobertura de histórico")
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    since = ccxt.binance().parse8601("2013-01-01T00:00:00Z")

    # 1) Histórico LARGO para backtest de ciclos: Binance BTC/USDT diario
    print("Descargando Binance BTC/USDT diario (histórico para backtest)...")
    binance = ccxt.binance({"enableRateLimit": True})
    btc_usdt_1d = fetch_ohlcv_paginated(binance, "BTC/USDT", "1d", since)
    save(btc_usdt_1d, "btc_usdt_1d_binance")

    # 2) Semanal (para análisis de ciclo: media de 200 semanas, etc.)
    print("Descargando Binance BTC/USDT semanal...")
    btc_usdt_1w = fetch_ohlcv_paginated(binance, "BTC/USDT", "1w", since)
    save(btc_usdt_1w, "btc_usdt_1w_binance")

    # 3) Temporalidades TACTICAS para busqueda de patrones (Binance, histórico largo).
    #    4h = punto dulce para semi-automatico; 1h = afinar la entrada.
    print("Descargando Binance BTC/USDT 4h (capa táctica)...")
    btc_usdt_4h = fetch_ohlcv_paginated(binance, "BTC/USDT", "4h", since)
    save(btc_usdt_4h, "btc_usdt_4h_binance")

    print("Descargando Binance BTC/USDT 1h (afinar entrada; tarda más)...")
    btc_usdt_1h = fetch_ohlcv_paginated(binance, "BTC/USDT", "1h", since)
    save(btc_usdt_1h, "btc_usdt_1h_binance")

    # 4) Par real de trading en Kraken: BTC/EUR diario (lo que cotizaremos en vivo)
    print("Descargando Kraken BTC/EUR diario (par de trading real)...")
    kraken = ccxt.kraken({"enableRateLimit": True})
    btc_eur_1d = fetch_ohlcv_paginated(kraken, "BTC/EUR", "1d", since)
    save(btc_eur_1d, "btc_eur_1d_kraken")

    print("\nListo. Datos en", DATA_DIR)


if __name__ == "__main__":
    main()
