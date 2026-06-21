"""
Fase 0 - Backtester honesto: estrategia de CICLO/VALORACION vs comprar y aguantar.

Estrategia ("escalera de Mayer"): mantiene un % objetivo en BTC segun lo caro o
barato que esta respecto a su media de 200 dias (Mayer Multiple). Compra barato,
reduce caro. Reajusta UNA vez al mes para no disparar comisiones.

  Mayer <= 0.8 ......... 100% BTC   (zona de suelo: maxima exposicion)
  0.8 - 1.0 ............  85%
  1.0 - 1.4 ............  60%
  1.4 - 1.8 ............  40%
  1.8 - 2.2 ............  20%
  2.2 - 2.4 ............  10%
  > 2.4 ...............    0% BTC   (euforia: fuera)

IMPORTANTE: estos umbrales son ILUSTRATIVOS. Ajustarlos para maximizar el
retorno pasado seria sobreajuste. En la Fase 1 los validaremos con
walk-forward (optimizar en una ventana, comprobar en otra distinta).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

FEE = 0.0026          # 0.26% comision taker (Kraken)
INITIAL = 10_000.0    # capital inicial


def mayer_target_weight(mayer: float) -> float:
    if np.isnan(mayer):
        return 0.0
    if mayer <= 0.8:
        return 1.00
    if mayer <= 1.0:
        return 0.85
    if mayer <= 1.4:
        return 0.60
    if mayer <= 1.8:
        return 0.40
    if mayer <= 2.2:
        return 0.20
    if mayer <= 2.4:
        return 0.10
    return 0.0


def metrics(equity: pd.Series, dates: pd.Series) -> dict:
    eq = equity.values
    rets = np.diff(eq) / eq[:-1]
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else np.nan
    running_max = np.maximum.accumulate(eq)
    max_dd = (eq / running_max - 1).min()
    vol = np.std(rets) * np.sqrt(365)
    sharpe = (np.mean(rets) * 365) / vol if vol > 0 else np.nan
    return {
        "final": eq[-1],
        "total_return": eq[-1] / eq[0] - 1,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "volatility": vol,
        "sharpe": sharpe,
    }


def run_buy_hold(df: pd.DataFrame) -> pd.Series:
    px = df["close"].values
    units = INITIAL * (1 - FEE) / px[0]
    return pd.Series(units * px, index=df.index)


def run_strategy(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """Reajuste mensual a peso objetivo segun Mayer. Devuelve (equity, n_trades)."""
    px = df["close"].values
    mayer = df["mayer"].values
    month = df["date"].dt.tz_localize(None).dt.to_period("M").values

    cash = INITIAL
    units = 0.0
    equity = np.empty(len(df))
    trades = 0
    last_month = None

    for i in range(len(df)):
        price = px[i]
        eq = cash + units * price
        if month[i] != last_month:  # primer dia de cada mes -> reajuste
            target_value = mayer_target_weight(mayer[i]) * eq
            current_value = units * price
            delta = target_value - current_value  # >0 compra, <0 vende
            if abs(delta) > eq * 0.01:            # banda muerta: no operar por calderilla
                fee = abs(delta) * FEE
                units += delta / price
                cash -= delta + fee
                trades += 1
            last_month = month[i]
        equity[i] = cash + units * price
    return pd.Series(equity, index=df.index), trades


def run_dca(df: pd.DataFrame, monthly: float = 200.0) -> pd.Series:
    """Referencia: aportar una cantidad fija cada mes (sin vender nunca)."""
    px = df["close"].values
    month = df["date"].dt.tz_localize(None).dt.to_period("M").values
    units = 0.0
    invested = 0.0
    equity = np.empty(len(df))
    last_month = None
    for i in range(len(df)):
        if month[i] != last_month:
            units += monthly * (1 - FEE) / px[i]
            invested += monthly
            last_month = month[i]
        equity[i] = units * px[i]
    # Normalizamos a retorno sobre lo invertido para poder comparar.
    return pd.Series(equity, index=df.index), invested


def fmt(d: dict) -> str:
    return (f"final={d['final']:>12,.0f}  ret={d['total_return']*100:>8,.0f}%  "
            f"CAGR={d['cagr']*100:>6,.1f}%  maxDD={d['max_drawdown']*100:>6,.1f}%  "
            f"vol={d['volatility']*100:>5,.0f}%  Sharpe={d['sharpe']:>5,.2f}")


def main() -> None:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_enriched_1d.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["ma200d"].notna()].reset_index(drop=True)  # empezar cuando hay MA200d

    start, end = df["date"].iloc[0].date(), df["date"].iloc[-1].date()
    print("=" * 78)
    print(f"BACKTEST  {start} -> {end}   capital inicial {INITIAL:,.0f}   comision {FEE*100:.2f}%/op")
    print("=" * 78)

    bh = run_buy_hold(df)
    strat, ntr = run_strategy(df)
    dca_eq, invested = run_dca(df)

    m_bh = metrics(bh, df["date"])
    m_st = metrics(strat, df["date"])

    print(f"Comprar y aguantar : {fmt(m_bh)}")
    print(f"Estrategia ciclo   : {fmt(m_st)}   (operaciones: {ntr})")
    print("-" * 78)
    print(f"DCA (200/mes, ref) : invertido {invested:,.0f} -> valor {dca_eq.iloc[-1]:,.0f} "
          f"({(dca_eq.iloc[-1]/invested-1)*100:,.0f}% sobre lo aportado)")
    print("=" * 78)

    # Lectura honesta del resultado (compara retorno, drawdown Y Sharpe).
    print("\nLECTURA:")
    mejor_cagr = m_st["cagr"] >= m_bh["cagr"]
    mejor_dd = m_st["max_drawdown"] > m_bh["max_drawdown"]   # menos negativo = mejor
    mejor_sharpe = m_st["sharpe"] >= m_bh["sharpe"]
    if mejor_cagr and mejor_dd:
        print("- La estrategia bate a comprar-y-aguantar en retorno Y en drawdown. Prometedor,")
        print("  pero OJO: sigue siendo DENTRO de la muestra. Falta walk-forward (Fase 1).")
    elif mejor_sharpe or (mejor_dd and not mejor_cagr):
        print(f"- Comprar-y-aguantar gana en retorno bruto ({m_bh['cagr']*100:.1f}% vs "
              f"{m_st['cagr']*100:.1f}% CAGR).")
        print(f"- La estrategia recorta el drawdown ({m_st['max_drawdown']*100:.0f}% vs "
              f"{m_bh['max_drawdown']*100:.0f}%) y la volatilidad, pero su Sharpe "
              f"({m_st['sharpe']:.2f}) {'supera' if mejor_sharpe else 'NO supera'} "
              f"al de HODL ({m_bh['sharpe']:.2f}).")
        print("- Conclusion: esta escalera de Mayer 'naive' NO bate a HODL aqui. Justo por")
        print("  esto existe la Fase 0: descartar lo que no funciona antes de arriesgar dinero.")
    else:
        print("- Comprar-y-aguantar bate a la estrategia en este periodo. El 'market timing'")
        print("  por ciclos NO siempre supera al HODL: hay que ganarse el edge, no asumirlo.")
    print("- Recuerda: solo 2 halvings de muestra. Nada aqui prueba que funcione a futuro.")

    # Guardar curvas de capital para inspeccion/grafico posterior.
    out = pd.DataFrame({
        "date": df["date"].dt.date,
        "price": df["close"].values,
        "mayer": df["mayer"].values,
        "buy_hold": bh.values,
        "strategy": strat.values,
    })
    csv = os.path.join(REPORTS_DIR, "backtest_equity.csv")
    out.to_csv(csv, index=False)
    print(f"\nCurvas de capital guardadas en {csv}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
