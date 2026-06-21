"""
Lee el buffer de liquidaciones acumulado por liq_collector.py
y construye el mapa de niveles de precio con clusters de liquidacion.
Uso standalone o importado desde daily_signal.py.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")
LIQ_FILE = os.path.join(REPORTS_DIR, "liq_rolling.json")


def fetch_liq_data(hours: int = 24) -> pd.DataFrame:
    """Lee el archivo rolling y devuelve un DataFrame de liquidaciones."""
    if not os.path.exists(LIQ_FILE):
        return pd.DataFrame()
    try:
        data = json.load(open(LIQ_FILE))
    except Exception:
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    cutoff = (time.time() - hours * 3600) * 1000
    df = df[df["T"] >= cutoff].copy()
    df["ts"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    return df


def build_liq_map(current_price: float, hours: int = 24,
                  bin_size: float = 500) -> dict:
    """
    Construye el mapa de liquidaciones por nivel de precio.
    Devuelve dict con:
      - total_usd_longs / total_usd_shorts  : volumen total liquidado
      - top_clusters : lista de {price, side, usd_m} con los mayores clusters
      - dominant_side : LONGS o SHORTS
      - summary_text  : texto listo para Telegram
    """
    df = fetch_liq_data(hours)
    if df.empty:
        return {"available": False, "summary_text": "sin datos de liquidaciones (colector no activo)"}

    longs_liq  = df[df["S"] == "SELL"]["usd"].sum()   # SELL = long liquidado
    shorts_liq = df[df["S"] == "BUY"]["usd"].sum()    # BUY  = short liquidado
    total = longs_liq + shorts_liq

    # Clusters por nivel de precio
    df["bin"] = (df["p"] // bin_size * bin_size).astype(int)
    clusters = (df.groupby(["bin", "S"])["usd"]
                  .sum()
                  .reset_index()
                  .rename(columns={"usd": "usd_total"}))
    clusters["usd_m"] = clusters["usd_total"] / 1e6
    clusters["lado"] = clusters["S"].map({"SELL": "LONGS liq", "BUY": "SHORTS liq"})

    # Top 10 clusters por USD
    top = clusters.sort_values("usd_m", ascending=False).head(10)

    # Zona de mayor concentracion relativa al precio actual
    above = clusters[clusters["bin"] > current_price]["usd_m"].sum()
    below = clusters[clusters["bin"] <= current_price]["usd_m"].sum()

    dominant = "LONGS" if longs_liq > shorts_liq else "SHORTS"
    longs_pct = longs_liq / total * 100 if total > 0 else 0
    shorts_pct = shorts_liq / total * 100 if total > 0 else 0

    # Nivel de precio con mayor cluster (resistencia/soporte de liquidacion)
    if not top.empty:
        biggest = top.iloc[0]
        biggest_str = (f"${int(biggest['bin']):,} ({biggest['lado']}, "
                       f"${biggest['usd_m']:.1f}M)")
    else:
        biggest_str = "N/A"

    # Construir texto del mensaje
    lines = [
        f"💥 LIQUIDACIONES {hours}h: ${total/1e6:.1f}M total",
        f"   Largos liquidados: ${longs_liq/1e6:.1f}M ({longs_pct:.0f}%)",
        f"   Cortos liquidados: ${shorts_liq/1e6:.1f}M ({shorts_pct:.0f}%)",
        f"   Mayor cluster: {biggest_str}",
        f"   Por encima precio actual: ${above:.1f}M | Por debajo: ${below:.1f}M",
    ]

    # Señal interpretada
    if longs_pct > 70:
        lines.append(
            "   → Dominan liquidaciones de LARGOS. Muchos stops barridos. "
            "Posible rebote si se agota la presión vendedora.")
    elif shorts_pct > 70:
        lines.append(
            "   → Dominan liquidaciones de CORTOS. Short squeeze activo. "
            "Posible continuation alcista si hay volumen.")
    else:
        lines.append("   → Liquidaciones equilibradas entre largos y cortos.")

    # Top 5 niveles mas importantes
    lines.append("   TOP niveles:")
    for _, row in top.head(5).iterrows():
        marker = "▲" if row["bin"] > current_price else "▼"
        lines.append(f"     {marker} ${row['bin']:,} — {row['lado']} ${row['usd_m']:.1f}M")

    return {
        "available": True,
        "total_usd": total,
        "longs_usd": longs_liq,
        "shorts_usd": shorts_liq,
        "dominant_side": dominant,
        "top_clusters": top.to_dict("records"),
        "above_price_usd": above * 1e6,
        "below_price_usd": below * 1e6,
        "summary_text": "\n".join(lines),
        "n_events": len(df),
    }


if __name__ == "__main__":
    # Test rapido
    result = build_liq_map(current_price=64000)
    print(result["summary_text"])
    print(f"\n(n={result.get('n_events', 0)} liquidaciones en el buffer)")
