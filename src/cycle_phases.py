"""
Analisis de la ESTRUCTURA de los ciclos de Bitcoin (lo que SI se repite).

A diferencia de los patrones de velas cortas (que probamos y son ruido), el ciclo
macro de ~4 anios SI tiene una estructura recurrente, anclada al halving:
    suelo -> [halving] -> mercado ALCISTA -> techo -> mercado BAJISTA -> suelo ...

Este modulo trocea los 15 anios de historia (Bitstamp BTC/USD desde 2011) en sus
ciclos, mide lo que se repite (tiempos, multiplicadores, caidas) y situa el
momento ACTUAL respecto a los ciclos anteriores en la misma posicion.

Salida tambien en reports/cycle_overlay.json para el grafico de superposicion.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")

HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"], utc=True)


def load() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_usd_1d_bitstamp.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def price_at(df: pd.DataFrame, dt: pd.Timestamp) -> float:
    i = df["date"].searchsorted(dt)
    i = min(i, len(df) - 1)
    return float(df["close"].iloc[i])


def analyze(df: pd.DataFrame) -> list[dict]:
    last = df["date"].iloc[-1]
    cycles = []
    for i, hv in enumerate(HALVINGS):
        # El techo del ciclo ocurre ~12-18 meses tras el halving. Acotamos la
        # busqueda a 750 dias para NO confundirlo con el rally PRE-halving del
        # ciclo siguiente (que en 2024 superaba el techo real de 2021).
        peak_end = min(hv + pd.Timedelta(days=750), last)
        seg = df[(df["date"] >= hv) & (df["date"] <= peak_end)]
        if len(seg) < 30:
            continue
        # TECHO = maximo del tramo post-halving (ventana acotada)
        pk = seg.loc[seg["close"].idxmax()]
        peak_date, peak_price = pk["date"], float(pk["close"])
        # SUELO bajista = minimo desde el techo hasta ~15 meses despues (o fin de datos)
        bear_end = min(peak_date + pd.Timedelta(days=450), last)
        bseg = df[(df["date"] >= peak_date) & (df["date"] <= bear_end)]
        bt = bseg.loc[bseg["close"].idxmin()]
        bottom_date, bottom_price = bt["date"], float(bt["close"])

        hv_price = price_at(df, hv)
        cycles.append({
            "n": i + 1,
            "halving": hv, "halving_price": hv_price,
            "peak_date": peak_date, "peak_price": peak_price,
            "bottom_date": bottom_date, "bottom_price": bottom_price,
            "halving_to_peak_days": (peak_date - hv).days,
            "peak_mult_from_halving": peak_price / hv_price,
            "bear_drawdown": bottom_price / peak_price - 1.0,
            "bear_days": (bottom_date - peak_date).days,
            "is_current": (i == len(HALVINGS) - 1),
        })
    # multiplicador del ALCISTA (suelo previo -> techo)
    for j in range(1, len(cycles)):
        prev_bottom = cycles[j - 1]["bottom_price"]
        cycles[j]["bull_mult_from_prev_bottom"] = cycles[j]["peak_price"] / prev_bottom
    return cycles


def print_structure(cycles: list[dict]) -> None:
    print("=" * 90)
    print("ESTRUCTURA DE LOS CICLOS DE BITCOIN  (lo que se repite)")
    print("=" * 90)
    print(f"{'Ciclo':6}{'Halving':12}{'Techo':12}{'dias->techo':12}{'x desde halving':16}"
          f"{'Caida bajista':14}{'dias bajista':12}")
    for c in cycles:
        dd = f"{c['bear_drawdown']*100:.0f}%"
        days_bear = c["bear_days"] if not c["is_current"] else 0
        cur = "  <-- ACTUAL" if c["is_current"] else ""
        bear_txt = (f"{dd:>10}  " if not c["is_current"] else "  (en curso)")
        print(f"{c['n']:<6}{str(c['halving'].date()):12}{str(c['peak_date'].date()):12}"
              f"{c['halving_to_peak_days']:<12}{c['peak_mult_from_halving']:>10.1f}x     "
              f"{bear_txt:<14}{days_bear if days_bear else '':<12}{cur}")
    print("-" * 90)


def print_repeats(cycles: list[dict]) -> None:
    done = [c for c in cycles if not c["is_current"]]
    h2p = [c["halving_to_peak_days"] for c in done]
    dd = [c["bear_drawdown"] for c in done]
    bd = [c["bear_days"] for c in done]
    mult = [c["peak_mult_from_halving"] for c in done]
    print("LO QUE SE REPITE (ciclos completos):")
    print(f"  - Techo tras el halving:   {min(h2p)}-{max(h2p)} dias  (media {int(np.mean(h2p))}, ~{int(np.mean(h2p)/30)} meses)")
    print(f"  - Mercado BAJISTA dura:    {min(bd)}-{max(bd)} dias  (media {int(np.mean(bd))}, ~{int(np.mean(bd)/30)} meses)")
    print(f"  - Caida del bajista:       {min(dd)*100:.0f}% a {max(dd)*100:.0f}%  (media {np.mean(dd)*100:.0f}%)")
    print(f"  - Subida halving->techo:   {min(mult):.1f}x a {max(mult):.1f}x  -> DECRECIENTE: " +
          " > ".join(f"{m:.0f}x" for m in mult))
    print("  Conclusion: la FORMA se repite (alcista 12-18m post-halving, luego bajista")
    print("  ~12-14m con caida 75-85%). La MAGNITUD, no: cada ciclo sube mucho menos.")
    print("-" * 90)


def locate_now(df: pd.DataFrame, cycles: list[dict]) -> dict:
    cur = cycles[-1]
    last = df["date"].iloc[-1]
    days_since = (last - cur["halving"]).days
    price_now = float(df["close"].iloc[-1])
    dd_from_peak = price_now / cur["peak_price"] - 1.0

    print("DONDE ESTAMOS AHORA (ciclo del halving 2024-04-20):")
    print(f"  - Dia {days_since} post-halving. Techo del ciclo hasta hoy: "
          f"{cur['peak_price']:,.0f}$ ({cur['peak_date'].date()}).")
    print(f"  - Precio hoy {price_now:,.0f}$  =>  {dd_from_peak*100:.0f}% desde ese techo.")

    # ¿Donde estaban los ciclos ANTERIORES a este mismo dia post-halving?
    print(f"  - En el dia {days_since} post-halving, los ciclos anteriores estaban:")
    analog = []
    for c in cycles[:-1]:
        target = c["halving"] + pd.Timedelta(days=days_since)
        if target > df["date"].iloc[-1]:
            continue
        p = price_at(df, target)
        dd = p / c["peak_price"] - 1.0
        # ¿ya habian hecho techo? ¿cuanto faltaba para el suelo?
        d_to_bottom = (c["bottom_date"] - target).days
        fase = "BAJISTA" if target > c["peak_date"] else "alcista"
        print(f"      Ciclo {c['n']}: {dd*100:>4.0f}% desde su techo, en fase {fase}, "
              f"{'suelo en ' + str(d_to_bottom) + ' dias' if d_to_bottom>0 else 'suelo ya pasado'}")
        analog.append(d_to_bottom)
    if analog:
        print(f"  -> ANALOGO: en esta posicion, los ciclos previos estaban en BAJISTA tardio, "
              f"a {min(analog)}-{max(analog)} dias del suelo.")
    print("=" * 90)
    return {"days_since_halving": days_since, "dd_from_peak": dd_from_peak}


def dump_overlay(df: pd.DataFrame, cycles: list[dict]) -> None:
    """Curvas normalizadas (precio/precio_halving) alineadas por dias-desde-halving."""
    out = {"cycles": []}
    for c in cycles:
        start = c["halving"] - pd.Timedelta(days=120)
        end = (c["halving"] + pd.Timedelta(days=1400))
        seg = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        seg["d"] = (seg["date"] - c["halving"]).dt.days
        norm = seg["close"].values / c["halving_price"]
        # downsample cada 5 dias
        idx = np.arange(0, len(seg), 5)
        out["cycles"].append({
            "n": c["n"], "halving": str(c["halving"].date()),
            "is_current": c["is_current"],
            "days": seg["d"].values[idx].tolist(),
            "norm": np.round(norm[idx], 3).tolist(),
        })
    os.makedirs(REPORTS_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(REPORTS_DIR, "cycle_overlay.json"), "w"))
    print(f"Curvas de superposicion -> reports/cycle_overlay.json")


def main() -> None:
    df = load()
    cycles = analyze(df)
    print_structure(cycles)
    print_repeats(cycles)
    locate_now(df, cycles)
    dump_overlay(df, cycles)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
