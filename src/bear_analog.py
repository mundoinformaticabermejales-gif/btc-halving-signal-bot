"""
Analogos en los DOS ULTIMOS BAJISTAS (2018 y 2022) - idea del usuario.

"Segun el patron de las ultimas 40/60/80 horas, ¿que probabilidad hay de que
Bitcoin baje en las proximas X horas, mirando solo los 2 ultimos ciclos bajistas?"

Clave de honestidad: en un bajista, BTC ya baja MAS veces que sube por pura TASA
BASE (la tendencia arrastra). Por eso comparamos SIEMPRE:
   prob_baja(patron)  vs  prob_baja(base del bajista)
Si el patron no supera a la tasa base, el patron no aporta nada: lo que manda es
"estamos en bajista", que es justo lo que ya captura el filtro macro del bot.

Ademas validamos CRUZANDO los dos bajistas (lo aprendido en 2018 -> probar en 2022
y al reves): asi vemos si "el patron se cumple en los dos" de verdad o es ilusion.

Uso:
    python src/bear_analog.py --tf 1h --window 60 --horizon 72
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from intraday_patterns import phase_intervals

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TF_HOURS = {"1h": 1, "4h": 4, "1d": 24}


def load(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, f"btc_usdt_{tf}_binance.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def in_intervals(dates, intervals):
    d = dates.dt.tz_localize(None).values
    m = np.zeros(len(dates), dtype=bool)
    for a, b in intervals:
        aa = pd.Timestamp(a).tz_convert("UTC").tz_localize(None).to_datetime64()
        bb = pd.Timestamp(b).tz_convert("UTC").tz_localize(None).to_datetime64()
        m |= (d >= aa) & (d < bb)
    return m


def windows_and_fwd(close, L, H):
    rets = np.diff(np.log(close))
    wins = sliding_window_view(rets, L)          # (M x L)
    end = np.arange(L, len(close))               # cierre de cada ventana
    fwd = np.full(len(close), np.nan)
    fwd[:-H] = close[H:] / close[:-H] - 1.0
    return wins, end, fwd[end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h", choices=["1h", "4h", "1d"])
    ap.add_argument("--window", type=int, default=60, help="ventana del patron en HORAS")
    ap.add_argument("--horizon", type=int, default=72, help="horizonte a futuro en HORAS")
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()

    tfh = TF_HOURS[args.tf]
    L = max(3, round(args.window / tfh))         # ventana en velas
    H = max(1, round(args.horizon / tfh))        # horizonte en velas
    df = load(args.tf)

    bears, _ = phase_intervals()
    # bears = [2013, 2018, 2022, ACTUAL]. Los 2 ultimos COMPLETOS = 2018 y 2022.
    # (El de 2013 no existe en los datos de 1h, que empiezan en 2017.)
    prev_bears = bears[:-1][-2:]                  # 2018 y 2022
    cur_bear = [bears[-1]]                        # bajista actual (donde esta "ahora")

    wins, end, fwd = windows_and_fwd(df["close"].values, L, H)
    end_dates = df["date"].iloc[end].reset_index(drop=True)

    in_prev = in_intervals(end_dates, prev_bears) & ~np.isnan(fwd)
    in_cur = in_intervals(end_dates, cur_bear)

    print("=" * 74)
    print(f"ANALOGOS EN LOS 2 ULTIMOS BAJISTAS (2018 + 2022)")
    print(f"tf={args.tf}  ventana={args.window}h ({L} velas)  horizonte={args.horizon}h ({H} velas)")
    print(f"ventanas en esos 2 bajistas: {int(in_prev.sum())}")
    print("=" * 74)

    # ---- TASA BASE del bajista (sin patron) ----
    base_down = 100 * np.mean(fwd[in_prev] < 0)
    base_move = np.mean(fwd[in_prev]) * 100
    print(f"TASA BASE (cualquier momento del bajista): baja el {base_down:.0f}% de las veces, "
          f"mov. medio {base_move:+.2f}% en {args.horizon}h")

    # ---- SNAPSHOT: patron de AHORA vs analogos en los 2 bajistas ----
    print("-" * 74)
    if not in_cur.any():
        print("Ahora mismo NO estamos en fase bajista por la definicion macro; "
              "uso la ultima ventana disponible como 'patron actual'.")
        q = wins[-1]
    else:
        q = wins[np.where(in_cur)[0][-1]]
    cand = np.where(in_prev)[0]
    dist = np.sqrt(((wins[cand] - q) ** 2).sum(axis=1))
    sel = cand[np.argsort(dist)[:args.topn]]
    pat_down = 100 * np.mean(fwd[sel] < 0)
    pat_move = np.mean(fwd[sel]) * 100
    print(f"PATRON ACTUAL (ultimas {args.window}h) vs sus {args.topn} analogos en 2018/2022:")
    print(f"  baja despues:   {pat_down:.0f}% de las veces   (base {base_down:.0f}%)")
    print(f"  mov. medio:     {pat_move:+.2f}%   (base {base_move:+.2f}%)")
    print(f"  parecido:       distancia media {dist[np.argsort(dist)[:args.topn]].mean():.3f}")
    edge = pat_down - base_down
    print(f"  EDGE del patron sobre la tasa base: {edge:+.0f} puntos")

    # ---- VALIDACION CRUZADA: aprender en un bajista, probar en el otro ----
    print("-" * 74)
    print("VALIDACION CRUZADA (analogos de UN bajista para predecir el OTRO, sin lookahead):")
    b18 = in_intervals(end_dates, [prev_bears[0]]) & ~np.isnan(fwd)
    b22 = in_intervals(end_dates, [prev_bears[1]]) & ~np.isnan(fwd)
    hit = tot = 0
    pred_down_when_signal = 0
    for src, dst in [(b18, b22), (b22, b18)]:
        src_i, dst_i = np.where(src)[0], np.where(dst)[0]
        for t in dst_i[::H]:                       # no solapar
            dd = np.sqrt(((wins[src_i] - wins[t]) ** 2).sum(axis=1))
            nn = src_i[np.argsort(dd)[:args.topn]]
            pred = np.mean(fwd[nn])
            actual = fwd[t]
            if np.isnan(actual) or abs(pred) < 1e-12:
                continue
            tot += 1
            if np.sign(pred) == np.sign(actual):
                hit += 1
    base_acc = max(base_down, 100 - base_down)     # acierto de "siempre la clase mayoritaria"
    if tot:
        acc = 100 * hit / tot
        print(f"  acierto del PATRON: {acc:.0f}%   ({tot} señales)")
        print(f"  acierto de la REGLA SIMPLE 'en bajista, baja': {base_acc:.0f}%")
        if acc > base_acc + 3:
            print("  -> El patron MEJORA a la tasa base. Merece estudiarse mas.")
        else:
            print("  -> El patron NO mejora a 'estamos en bajista'. La info util es la FASE,")
            print("     no la forma de las ultimas horas. (Eso ya lo da el filtro macro del bot.)")
    print("=" * 74)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
