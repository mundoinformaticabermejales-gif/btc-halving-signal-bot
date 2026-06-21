"""
Patrones INTRADIA (velas de 1h) CONDICIONADOS a la fase del ciclo.

Idea del usuario: en velas de 1h, buscar ventanas de ~15 velas con porcentajes
parecidos (con un margen de error) que ocurran DENTRO de la misma fase del ciclo
(p.ej. el mercado BAJISTA) y ver que paso despues -> base para trading intradia.

Metodo honesto:
  - Etiqueta cada vela de 1h como 'alcista'/'bajista' segun los techos/suelos macro
    (de cycle_phases.py): BAJISTA = entre techo y suelo; ALCISTA = entre suelo y techo.
  - Dentro de la fase elegida, representa cada ventana de L=15 velas por su secuencia
    de % de variacion. Busca las mas PARECIDAS (distancia pequena = tu margen de error).
  - Mide la direccion del siguiente tramo de H velas, FUERA DE MUESTRA y con comision.
  - Como el bot es SPOT (no se puede vender en corto), mide tambien la version
    LONG-ONLY: solo se entra cuando el patron predice SUBIDA.

Uso:
    python src/intraday_patterns.py --phase bajista --window 15 --horizon 24
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from cycle_phases import analyze as cycle_analyze
from cycle_phases import load as load_daily

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FEE_ROUNDTRIP = 0.0052


def load_1h() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_usdt_1h_binance.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def phase_intervals():
    """Devuelve listas de intervalos (inicio, fin) de BAJISTA y ALCISTA macro."""
    cyc = cycle_analyze(load_daily())
    now = pd.Timestamp.now(tz="UTC")
    bears, bulls = [], []
    for i, c in enumerate(cyc):
        if c["is_current"]:
            bears.append((c["peak_date"], now))          # bajista en curso
        else:
            bears.append((c["peak_date"], c["bottom_date"]))
        if not c["is_current"]:
            nxt = cyc[i + 1]["peak_date"] if i + 1 < len(cyc) else now
            bulls.append((c["bottom_date"], nxt))         # alcista: suelo -> sig. techo
    return bears, bulls


def label_phase(dates: pd.Series, bears, bulls) -> np.ndarray:
    lab = np.array(["neutral"] * len(dates), dtype=object)
    d = dates.dt.tz_localize(None).values                # naive UTC para comparar sin warning
    def naive(x):
        return pd.Timestamp(x).tz_convert("UTC").tz_localize(None).to_datetime64()
    for a, b in bulls:
        lab[(d >= naive(a)) & (d < naive(b))] = "alcista"
    for a, b in bears:
        lab[(d >= naive(a)) & (d < naive(b))] = "bajista"
    return lab


def build(df: pd.DataFrame, L: int, H: int):
    close = df["close"].values
    rets = np.diff(np.log(close))                         # % por vela (log)
    wins = sliding_window_view(rets, L)                   # (M x L) secuencia de % de cada ventana
    end_idx = np.arange(L, len(close))                    # indice de cierre de cada ventana (en close)
    fwd = np.full(len(close), np.nan)
    fwd[:-H] = close[H:] / close[:-H] - 1.0               # retorno H velas despues
    return wins, end_idx, fwd[end_idx]


def snapshot(df, lab, wins, end_idx, fwd, phase, L, H, topn):
    mask_phase = lab[end_idx] == phase
    cur_phase = lab[-1]
    q = wins[-1]
    # candidatos: misma fase, desenlace conocido, sin solapar con la ventana actual
    valid = mask_phase & (np.arange(len(end_idx)) < len(end_idx) - L) & ~np.isnan(fwd)
    if valid.sum() < topn:
        print(f"  (pocas ventanas en fase '{phase}' para el snapshot)")
        return
    dist = np.sqrt(((wins[valid] - q) ** 2).sum(axis=1))  # distancia = margen de error
    order = np.argsort(dist)[:topn]
    best = fwd[valid][order]
    print("-" * 72)
    print(f"SNAPSHOT  (fase actual del mercado: {cur_phase.upper()})")
    print(f"De las {topn} ventanas de {L} velas mas PARECIDAS (en fase {phase}) a la de ahora,")
    print(f"lo que paso {H}h despues:")
    print(f"  subio:            {100*np.mean(best>0):.0f}% de las veces")
    print(f"  mov. medio:       {np.mean(best)*100:+.2f}%   mediana {np.median(best)*100:+.2f}%")
    print(f"  margen de error:  distancia media {dist[order].mean():.3f} (0 = identico)")
    if abs(np.mean(best)) < FEE_ROUNDTRIP:
        print(f"  -> el mov. medio no cubre comisiones ({FEE_ROUNDTRIP*100:.2f}%).")


def walkforward(df, lab, wins, end_idx, fwd, phase, L, H, topn, test_frac=0.3):
    in_phase = np.where((lab[end_idx] == phase) & ~np.isnan(fwd))[0]
    if len(in_phase) < 500:
        print(f"  (muestra insuficiente en fase '{phase}')")
        return
    split = int(len(in_phase) * (1 - test_frac))
    test_pts = in_phase[split:]
    stride = max(1, H)                                    # no solapar trades
    hits = preds = 0
    rets_dir, rets_long = [], []
    for t in test_pts[::stride]:
        cand = in_phase[(end_idx[in_phase] <= end_idx[t] - L) & (end_idx[in_phase] + H < end_idx[t])]
        if len(cand) < topn:
            continue
        dist = np.sqrt(((wins[cand] - wins[t]) ** 2).sum(axis=1))
        idx = cand[np.argsort(dist)[:topn]]
        pred = np.mean(fwd[idx])
        actual = fwd[t]
        if np.isnan(actual) or abs(pred) < 1e-9:
            continue
        preds += 1
        if np.sign(pred) == np.sign(actual):
            hits += 1
        rets_dir.append(np.sign(pred) * actual - FEE_ROUNDTRIP)
        if pred > 0:                                      # SPOT: solo largos
            rets_long.append(actual - FEE_ROUNDTRIP)
    if preds == 0:
        print("  (sin señales evaluables)")
        return
    rd = np.array(rets_dir)
    print("-" * 72)
    print(f"WALK-FORWARD en fase '{phase}'  (ultimo {int(test_frac*100)}% de esa fase, sin lookahead)")
    print(f"  señales:              {preds}")
    print(f"  acierto direccion:    {100*hits/preds:.1f}%   (50% = azar; tras comision)")
    print(f"  retorno medio/op:     {rd.mean()*100:+.2f}%   (direccional, tras comision)")
    if rets_long:
        rl = np.array(rets_long)
        print(f"  LONG-ONLY (spot):     {len(rl)} entradas | medio {rl.mean()*100:+.2f}%/op | "
              f"acumulado {(np.prod(1+rl)-1)*100:+.0f}%")
    veredicto = (hits / preds > 0.53 and rd.mean() > 0)
    print(f"  -> {'INDICIOS DE EDGE: merece optimizar.' if veredicto else 'SIN edge fiable tras comisiones (esperable en 1h).'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="bajista", choices=["bajista", "alcista"])
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--topn", type=int, default=50)
    args = ap.parse_args()

    df = load_1h()
    bears, bulls = phase_intervals()
    lab = label_phase(df["date"], bears, bulls)
    wins, end_idx, fwd = build(df, args.window, args.horizon)

    n_phase = int((lab == args.phase).sum())
    print("=" * 72)
    print(f"PATRONES 1h CONDICIONADOS  |  fase={args.phase}  ventana={args.window}  horizonte={args.horizon}h")
    print(f"velas 1h totales: {len(df)}  |  en fase '{args.phase}': {n_phase} "
          f"({100*n_phase/len(df):.0f}%)")
    print("=" * 72)
    snapshot(df, lab, wins, end_idx, fwd, args.phase, args.window, args.horizon, args.topn)
    walkforward(df, lab, wins, end_idx, fwd, args.phase, args.window, args.horizon, args.topn)
    print("=" * 72)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
