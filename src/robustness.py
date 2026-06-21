"""
Fase 1 - Pruebas de ROBUSTEZ de la estrategia combinada (valor + tendencia).

Un buen resultado en un solo backtest no prueba nada (puede ser suerte/sobreajuste).
Dos pruebas que lo distinguen:

  A) SENSIBILIDAD DE PARAMETROS
     ¿El edge sobrevive en un RANGO de medias (150..250) y de histeresis, o solo
     en el "numero magico" 200? Si solo funciona en un punto exacto -> sobreajuste.
     Robusto = la mayoria de las combinaciones baten a HODL en Sharpe.

  B) WALK-FORWARD RODANTE
     Simula la realidad: en cada ventana se elige el parametro con datos PASADOS
     (entreno) y se opera la siguiente ventana hacia delante (test), encadenando.
     Si el resultado encadenado bate a HODL, el metodo aguanta la seleccion real
     de parametros, no solo un ajuste a posteriori.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from backtest import INITIAL, metrics
from strategy import mayer_weight, simulate, trend_state

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MA_LENS = [100, 125, 150, 175, 200, 225, 250]
HYST = [(1.00, 1.00), (1.02, 0.98), (1.05, 0.95)]  # sin banda / media / amplia
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")


def load() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_enriched_1d.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[df["ma200d"].notna()].reset_index(drop=True)


def combined_target(close, ma, mayer, up, dn) -> np.ndarray:
    trend = trend_state(close, ma, up, dn)
    base = np.array([mayer_weight(m) for m in mayer])
    return base * trend


def oos_metrics(df: pd.DataFrame, target: np.ndarray) -> dict:
    eq, _ = simulate(df, target)
    split = df["date"].searchsorted(OOS_START)
    eq_oos = eq.iloc[split:].reset_index(drop=True)
    eq_oos = eq_oos / eq_oos.iloc[0] * INITIAL
    return metrics(eq_oos, df["date"].iloc[split:].reset_index(drop=True))


def sensitivity(df: pd.DataFrame) -> None:
    close, mayer = df["close"].values, df["mayer"].values
    # MAs precomputadas
    mas = {L: df["close"].rolling(L).mean().values for L in MA_LENS}
    hodl = oos_metrics(df, np.ones(len(df)))

    print("=" * 78)
    print("A) SENSIBILIDAD DE PARAMETROS  (Sharpe fuera de muestra 2022+, combinada)")
    print(f"   Referencia HODL: Sharpe {hodl['sharpe']:.2f},  maxDD {hodl['max_drawdown']*100:.0f}%")
    print("=" * 78)
    header = "  MA \\ histeresis " + "".join(f"{f'{u:.2f}/{d:.2f}':>14}" for u, d in HYST)
    print(header)
    beats = total = 0
    sharpes = []
    for L in MA_LENS:
        row = f"  MA{L:<4}          "
        for (u, d) in HYST:
            m = oos_metrics(df, combined_target(close, mas[L], mayer, u, d))
            sharpes.append(m["sharpe"])
            mark = "*" if m["sharpe"] > hodl["sharpe"] else " "
            row += f"{m['sharpe']:>12.2f}{mark}"
            beats += m["sharpe"] > hodl["sharpe"]
            total += 1
        print(row)
    print("-" * 78)
    print(f"  Combinaciones que baten a HODL en Sharpe: {beats}/{total} "
          f"({100*beats/total:.0f}%)   (* = bate a HODL)")
    print(f"  Sharpe combinada: min {min(sharpes):.2f}  mediana {np.median(sharpes):.2f}  max {max(sharpes):.2f}")
    if beats / total >= 0.7 and min(sharpes) > hodl["sharpe"] * 0.8:
        print("  -> ROBUSTO: el edge no depende de un parametro magico; aguanta todo el rango.")
    elif beats / total >= 0.5:
        print("  -> PARCIALMENTE robusto: funciona en buena parte del rango, con zonas flojas.")
    else:
        print("  -> FRAGIL: depende demasiado del parametro elegido. Sospecha de sobreajuste.")


def walk_forward(df: pd.DataFrame, train_years=3, test_months=6) -> None:
    close, mayer = df["close"].values, df["mayer"].values
    mas = {L: df["close"].rolling(L).mean().values for L in MA_LENS}
    dates = df["date"]

    print("\n" + "=" * 78)
    print(f"B) WALK-FORWARD RODANTE  (entreno {train_years}a -> opera {test_months}m, encadenado)")
    print("=" * 78)

    start = dates.iloc[0] + pd.DateOffset(years=train_years)
    wf_rets = []          # retornos diarios encadenados del test
    wf_dates = []
    elecciones = []
    t0 = start
    while True:
        tr_lo = t0 - pd.DateOffset(years=train_years)
        te_hi = t0 + pd.DateOffset(months=test_months)
        if te_hi > dates.iloc[-1]:
            break
        i_trlo = dates.searchsorted(tr_lo)
        i_t0 = dates.searchsorted(t0)
        i_tehi = dates.searchsorted(te_hi)

        # elegir MA que maximiza Sharpe en el tramo de ENTRENO
        best_L, best_s = None, -1e9
        tr = df.iloc[i_trlo:i_t0].reset_index(drop=True)
        for L in MA_LENS:
            tgt = combined_target(close, mas[L], mayer, 1.02, 0.98)[i_trlo:i_t0]
            eq, _ = simulate(tr, tgt)
            m = metrics(eq, tr["date"])
            if m["sharpe"] > best_s:
                best_s, best_L = m["sharpe"], L

        # operar el tramo de TEST con esa MA
        te = df.iloc[i_t0:i_tehi].reset_index(drop=True)
        tgt_te = combined_target(close, mas[best_L], mayer, 1.02, 0.98)[i_t0:i_tehi]
        eq, _ = simulate(te, tgt_te)
        rets = eq.pct_change().fillna(0).values
        wf_rets.extend(rets.tolist())
        wf_dates.extend(te["date"].tolist())
        elecciones.append((t0.date(), best_L))
        t0 = te_hi

    if not wf_rets:
        print("  (histórico insuficiente para walk-forward)")
        return

    wf_dates = pd.Series(wf_dates)
    wf_eq = pd.Series(INITIAL * np.cumprod(1 + np.array(wf_rets)))
    m_wf = metrics(wf_eq, wf_dates)

    # HODL en el mismo tramo encadenado
    i0 = df["date"].searchsorted(pd.Timestamp(wf_dates.iloc[0]))
    hodl_eq = df["close"].iloc[i0:].reset_index(drop=True)
    hodl_eq = hodl_eq / hodl_eq.iloc[0] * INITIAL
    m_hodl = metrics(hodl_eq, df["date"].iloc[i0:].reset_index(drop=True))

    print(f"  Periodo walk-forward: {wf_dates.iloc[0].date()} -> {wf_dates.iloc[-1].date()}")
    print(f"  MAs elegidas por ventana: {', '.join(f'{d}:MA{L}' for d, L in elecciones)}")
    print("-" * 78)
    print(f"  {'WALK-FORWARD':16} CAGR={m_wf['cagr']*100:>6.1f}%  maxDD={m_wf['max_drawdown']*100:>6.1f}%  "
          f"Sharpe={m_wf['sharpe']:.2f}")
    print(f"  {'HODL (mismo)':16} CAGR={m_hodl['cagr']*100:>6.1f}%  maxDD={m_hodl['max_drawdown']*100:>6.1f}%  "
          f"Sharpe={m_hodl['sharpe']:.2f}")
    print("-" * 78)
    if m_wf["sharpe"] > m_hodl["sharpe"] and m_wf["max_drawdown"] > m_hodl["max_drawdown"]:
        print("  -> APRUEBA: con seleccion REAL de parametros bate a HODL en Sharpe y drawdown.")
        print("     Candidata solida para paper trading (Fase 1 operativa).")
    elif m_wf["sharpe"] > m_hodl["sharpe"]:
        print("  -> Mejora el Sharpe pero no el drawdown. Aceptable; vigilar en paper.")
    else:
        print("  -> NO bate a HODL bajo seleccion real. No llevar a real todavia.")
    print("=" * 78)


def main() -> None:
    df = load()
    sensitivity(df)
    walk_forward(df)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
