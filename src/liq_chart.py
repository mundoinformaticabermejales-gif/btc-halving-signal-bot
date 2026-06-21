"""
Genera un gráfico visual del mapa de liquidaciones y lo envía por Telegram.
Usa los datos acumulados por liq_collector.py.
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")
LIQ_FILE = os.path.join(REPORTS_DIR, "liq_rolling.json")


def build_chart(current_price: float, hours: int = 24,
                bin_size: float = 500) -> io.BytesIO | None:
    """
    Genera el gráfico PNG del mapa de liquidaciones.
    Devuelve un BytesIO con la imagen o None si no hay datos.
    """
    if not os.path.exists(LIQ_FILE):
        return None
    try:
        data = json.load(open(LIQ_FILE))
    except Exception:
        return None
    if not data:
        return None

    df = pd.DataFrame(data)
    cutoff = (time.time() - hours * 3600) * 1000
    df = df[df["T"] >= cutoff].copy()
    if len(df) < 3:
        return None

    df["bin"] = (df["p"] // bin_size * bin_size).astype(int)
    clusters = df.groupby(["bin", "S"])["usd"].sum().reset_index()
    clusters["usd_m"] = clusters["usd"] / 1e6

    # Pivotar: una columna por lado
    pivot = clusters.pivot(index="bin", columns="S", values="usd_m").fillna(0)
    if "SELL" not in pivot.columns:
        pivot["SELL"] = 0.0
    if "BUY" not in pivot.columns:
        pivot["BUY"] = 0.0
    pivot = pivot.rename(columns={"SELL": "longs_liq", "BUY": "shorts_liq"})
    pivot = pivot.sort_index()

    # Filtrar niveles con actividad relevante (top 20 filas por total)
    pivot["total"] = pivot["longs_liq"] + pivot["shorts_liq"]
    pivot = pivot.nlargest(20, "total").sort_index()

    prices = pivot.index.values
    longs_vals = pivot["longs_liq"].values
    shorts_vals = pivot["shorts_liq"].values
    y = np.arange(len(prices))

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(5, len(prices) * 0.42)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bar_h = 0.38
    bars_long  = ax.barh(y + bar_h/2, longs_vals,  height=bar_h,
                          color="#ef4444", alpha=0.85, label="Largos liquidados")
    bars_short = ax.barh(y - bar_h/2, shorts_vals, height=bar_h,
                          color="#22c55e", alpha=0.85, label="Cortos liquidados")

    # Precio actual — línea vertical
    ax.axvline(0, color="#555", linewidth=0.5)

    # Resaltar precio actual como línea horizontal
    price_bin = int(current_price // bin_size * bin_size)
    if price_bin in pivot.index:
        idx = list(pivot.index).index(price_bin)
        ax.axhspan(idx - 0.5, idx + 0.5, color="#facc15", alpha=0.12)

    # Línea horizontal del precio actual
    if len(prices) > 0:
        price_y = np.interp(current_price, prices, y)
        ax.axhline(price_y, color="#facc15", linewidth=1.8,
                   linestyle="--", alpha=0.9, label=f"Precio ${current_price:,.0f}")

    # Etiquetas eje Y (precios)
    ax.set_yticks(y)
    ax.set_yticklabels([f"${p:,}" for p in prices],
                        color="#e5e7eb", fontsize=9)

    # Valores en las barras
    for bar in bars_long:
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                    f"${w:.1f}M", va="center", ha="left",
                    color="#ef4444", fontsize=7.5, fontweight="bold")
    for bar in bars_short:
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                    f"${w:.1f}M", va="center", ha="left",
                    color="#22c55e", fontsize=7.5, fontweight="bold")

    # Totales
    total_longs  = longs_vals.sum()
    total_shorts = shorts_vals.sum()
    total = total_longs + total_shorts

    ax.set_xlabel("USD Millones liquidados", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    # Título
    ax.set_title(
        f"Mapa de Liquidaciones BTC — últimas {hours}h\n"
        f"Largos: ${total_longs:.1f}M  |  Cortos: ${total_shorts:.1f}M  "
        f"|  Total: ${total:.1f}M  |  n={len(df)} eventos",
        color="#f9fafb", fontsize=10, pad=10
    )

    legend = ax.legend(
        handles=[
            mpatches.Patch(color="#ef4444", label=f"Largos liq: ${total_longs:.1f}M"),
            mpatches.Patch(color="#22c55e", label=f"Cortos liq: ${total_shorts:.1f}M"),
            plt.Line2D([0], [0], color="#facc15", linestyle="--",
                       label=f"Precio actual ${current_price:,.0f}"),
        ],
        facecolor="#1f2937", edgecolor="#374151",
        labelcolor="#e5e7eb", fontsize=8.5,
        loc="lower right"
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def send_chart_telegram(token: str, chat_id: str, buf: io.BytesIO,
                        caption: str = "") -> bool:
    """Envía una imagen PNG a Telegram via sendPhoto."""
    boundary = "----BotBoundary"
    img_data = buf.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f"{chat_id}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="photo"; filename="liq_map.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + img_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="caption"\r\n\r\n'
        f"{caption}\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendPhoto",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.load(r)
            return resp.get("ok", False)
    except Exception as e:
        print(f"[liq_chart] error Telegram: {e}")
        return False


if __name__ == "__main__":
    # Test: genera el gráfico y lo guarda localmente
    buf = build_chart(current_price=64000)
    if buf:
        out = os.path.join(REPORTS_DIR, "liq_map_test.png")
        with open(out, "wb") as f:
            f.write(buf.read())
        print(f"Grafico guardado en {out}")
    else:
        print("Sin datos de liquidaciones (colector no activo o mercado tranquilo)")
