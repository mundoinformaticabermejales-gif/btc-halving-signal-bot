"""
Fase 0 - Motor de analisis de CICLO de Bitcoin.

Convierte "estudiar las curvas y los ciclos" en metricas objetivas:
  - Dias desde el ultimo halving y % de avance del ciclo (~4 anios).
  - Media movil de 200 dias y Mayer Multiple (precio / MA200d) -> valoracion.
  - Media movil de 200 SEMANAS (suelo historico de los mercados bajistas).
  - Drawdown desde el maximo historico (ATH).

Estas metricas alimentan el backtester (backtest.py). NO predicen el futuro;
miden DONDE estamos respecto al patron historico, en porcentajes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Fechas reales de los halvings (UTC). El de 2028 es estimacion.
HALVINGS = pd.to_datetime(
    ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20", "2028-04-01"], utc=True
)
CYCLE_DAYS = 4 * 365.25  # duracion nominal de un ciclo


def load_daily() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_usdt_1d_binance.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def load_weekly() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_usdt_1w_binance.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def days_since_halving(dt: pd.Timestamp) -> float:
    past = HALVINGS[HALVINGS <= dt]
    if len(past) == 0:
        return np.nan
    return (dt - past[-1]).days


def cycle_phase_pct(dt: pd.Timestamp) -> float:
    """% de avance dentro del ciclo de 4 anios (0 = halving, 100 = siguiente halving)."""
    d = days_since_halving(dt)
    return np.nan if np.isnan(d) else 100.0 * d / CYCLE_DAYS


def enrich(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["ma200d"] = df["close"].rolling(200).mean()
    df["mayer"] = df["close"] / df["ma200d"]            # valoracion vs MA200d
    df["ath"] = df["close"].cummax()
    df["drawdown"] = df["close"] / df["ath"] - 1.0      # caida desde maximo (negativo)
    df["days_since_halving"] = df["date"].apply(days_since_halving)
    df["cycle_phase_pct"] = df["date"].apply(cycle_phase_pct)

    # Media de 200 semanas desde la serie semanal, mapeada al diario por fecha.
    weekly = weekly.copy()
    weekly["ma200w"] = weekly["close"].rolling(200).mean()
    w = weekly[["date", "ma200w"]].dropna()
    df = pd.merge_asof(df, w, on="date", direction="backward")
    df["mult_200w"] = df["close"] / df["ma200w"]        # precio / MA200w
    return df


def snapshot(df: pd.DataFrame) -> None:
    """Imprime el estado ACTUAL del ciclo."""
    r = df.iloc[-1]
    print("=" * 60)
    print("ESTADO ACTUAL DEL CICLO DE BITCOIN")
    print("=" * 60)
    print(f"Fecha:                 {r['date'].date()}")
    print(f"Precio (USDT):         {r['close']:,.0f}")
    print(f"MA 200 dias:           {r['ma200d']:,.0f}")
    print(f"Mayer Multiple:        {r['mayer']:.2f}   (precio / MA200d)")
    if not np.isnan(r.get("ma200w", np.nan)):
        print(f"MA 200 semanas:        {r['ma200w']:,.0f}")
        print(f"Precio / MA200w:       {r['mult_200w']:.2f}")
    print(f"Drawdown desde ATH:    {r['drawdown']*100:,.1f}%")
    print(f"Dias desde halving:    {int(r['days_since_halving'])}  (halving 2024-04-20)")
    print(f"Avance del ciclo:      {r['cycle_phase_pct']:.0f}%  (0=halving, 100=siguiente)")

    # Lectura del Mayer en lenguaje de valoracion.
    m = r["mayer"]
    if m < 0.8:
        zona = "MUY INFRAVALORADO (zona historica de suelo)"
    elif m < 1.0:
        zona = "infravalorado (por debajo de la MA200d)"
    elif m < 1.4:
        zona = "neutral / valor justo"
    elif m < 2.0:
        zona = "caro"
    elif m < 2.4:
        zona = "muy caro"
    else:
        zona = "EUFORIA (zona historica de techo)"
    print(f"Lectura valoracion:    {zona}")

    # Referencia historica: pico tipico ~500-550 dias tras el halving.
    print("-" * 60)
    print("Referencia historica: el pico suele darse ~500-550 dias")
    print("tras el halving; el suelo, en la 2a mitad del ciclo.")
    print("=" * 60)


def main() -> None:
    daily = load_daily()
    weekly = load_weekly()
    df = enrich(daily, weekly)
    out = os.path.join(DATA_DIR, "btc_enriched_1d.parquet")
    df.to_parquet(out, index=False)
    snapshot(df)
    print(f"\nDataset enriquecido guardado en {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
