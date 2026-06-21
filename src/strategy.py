"""
Fase 1 - Estrategias condicionadas al REGIMEN vs comprar-y-aguantar.

La leccion de la Fase 0: ni el ciclo en crudo ni los patrones en crudo baten a
HODL. Aqui probamos la hipotesis correcta: ¿aporta valor CONDICIONAR la exposicion
al regimen de mercado? Comparamos, todas con comision 0.26%/op:

  - HODL ................... comprar y aguantar (benchmark).
  - Filtro de tendencia .... 100% BTC si precio > MA200d, fuera si no (con histeresis).
  - Ciclo (Mayer) .......... escalera de valoracion de la Fase 0.
  - COMBINADA .............. peso por valoracion (Mayer) PERO solo si la tendencia
                             mayor acompaña (precio sobre MA200d). El patron/valor
                             es el gatillo; la tendencia es el permiso.

Se reporta el periodo COMPLETO y un tramo FUERA DE MUESTRA (desde 2022) que
incluye un ciclo bajista+alcista entero. Sin optimizar umbrales = sin sobreajuste.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from backtest import FEE, INITIAL, metrics  # reutilizamos del modulo de Fase 0

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def mayer_weight(m: float) -> float:
    if np.isnan(m):
        return 0.0
    for hi, w in [(0.8, 1.0), (1.0, 0.85), (1.4, 0.60), (1.8, 0.40), (2.2, 0.20), (2.4, 0.10)]:
        if m <= hi:
            return w
    return 0.0


def trend_state(close: np.ndarray, ma: np.ndarray, up=1.02, dn=0.98) -> np.ndarray:
    """1 = tendencia alcista (con histeresis para evitar bandazos), 0 = bajista."""
    state = np.zeros(len(close))
    cur = 0
    for i in range(len(close)):
        if np.isnan(ma[i]):
            state[i] = 0
            continue
        if cur == 0 and close[i] > ma[i] * up:
            cur = 1
        elif cur == 1 and close[i] < ma[i] * dn:
            cur = 0
        state[i] = cur
    return state


def simulate(df: pd.DataFrame, target: np.ndarray, band: float = 0.04) -> tuple[pd.Series, int]:
    """Simula con banda muerta: solo opera si el peso se desvia del objetivo > band."""
    px = df["close"].values
    cash, units, trades = INITIAL, 0.0, 0
    equity = np.empty(len(df))
    for i in range(len(df)):
        eq = cash + units * px[i]
        cur_w = (units * px[i]) / eq if eq > 0 else 0.0
        tgt = target[i]
        if abs(cur_w - tgt) > band:
            desired_val = tgt * eq
            delta = desired_val - units * px[i]
            fee = abs(delta) * FEE
            units += delta / px[i]
            cash -= delta + fee
            trades += 1
        equity[i] = cash + units * px[i]
    return pd.Series(equity, index=df.index), trades


def fmt(d: dict) -> str:
    return (f"CAGR={d['cagr']*100:>6,.1f}%  maxDD={d['max_drawdown']*100:>6,.1f}%  "
            f"vol={d['volatility']*100:>4,.0f}%  Sharpe={d['sharpe']:>5,.2f}")


def report(df: pd.DataFrame, curves: dict[str, pd.Series], trades: dict[str, int], titulo: str) -> None:
    print("=" * 86)
    print(f"{titulo}   {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    print("=" * 86)
    base = None
    for name, eq in curves.items():
        eqn = eq / eq.iloc[0] * INITIAL  # renormalizar al inicio del tramo
        m = metrics(eqn, df["date"])
        base = base or name
        print(f"  {name:24} {fmt(m)}  ops={trades.get(name, 0):>3}")


def main() -> None:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_enriched_1d.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["ma200d"].notna()].reset_index(drop=True)

    close = df["close"].values
    ma = df["ma200d"].values
    mayer = df["mayer"].values
    trend = trend_state(close, ma)

    targets = {
        "HODL":               np.ones(len(df)),
        "Filtro tendencia":   trend.astype(float),
        "Ciclo (Mayer)":      np.array([mayer_weight(m) for m in mayer]),
        "COMBINADA":          np.array([mayer_weight(m) for m in mayer]) * trend,
    }
    curves, trades = {}, {}
    for name, tgt in targets.items():
        eq, ntr = simulate(df, tgt)
        curves[name] = eq
        trades[name] = ntr

    # Periodo completo
    report(df, curves, trades, "PERIODO COMPLETO")

    # Fuera de muestra: desde 2022 (incluye bajista 2022 + alcista 2023-25 + 2026)
    split = df["date"].searchsorted(pd.Timestamp("2022-01-01", tz="UTC"))
    df_oos = df.iloc[split:].reset_index(drop=True)
    curves_oos = {k: v.iloc[split:].reset_index(drop=True) for k, v in curves.items()}
    report(df_oos, curves_oos, {k: "" for k in trades}, "FUERA DE MUESTRA (desde 2022)")

    # Veredicto automatico sobre el tramo OOS.
    m = {k: metrics(v / v.iloc[0] * INITIAL, df_oos["date"]) for k, v in curves_oos.items()}
    print("=" * 86)
    print("VEREDICTO (fuera de muestra, ajustado por riesgo):")
    best_sharpe = max(m, key=lambda k: m[k]["sharpe"])
    best_dd = max(m, key=lambda k: m[k]["max_drawdown"])
    print(f"  Mejor Sharpe:        {best_sharpe}  ({m[best_sharpe]['sharpe']:.2f} vs HODL {m['HODL']['sharpe']:.2f})")
    print(f"  Menor drawdown:      {best_dd}  ({m[best_dd]['max_drawdown']*100:.0f}% vs HODL {m['HODL']['max_drawdown']*100:.0f}%)")
    if m["COMBINADA"]["sharpe"] > m["HODL"]["sharpe"]:
        print("  -> La COMBINADA mejora el retorno ajustado por riesgo frente a HODL.")
        print("     Candidata para Fase 1 (paper). Falta walk-forward rodante y robustez.")
    else:
        print("  -> La COMBINADA NO bate a HODL en Sharpe aqui. Seguir iterando antes de paper.")
    print("=" * 86)

    out = pd.DataFrame({"date": df["date"].dt.date, "price": close,
                        **{k: v.values for k, v in curves.items()}})
    csv = os.path.join(REPORTS_DIR, "strategy_equity.csv")
    out.to_csv(csv, index=False)
    print(f"Curvas guardadas en {csv}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
