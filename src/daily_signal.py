"""
Fase 1 (operativo) - Generador de SEÑAL DIARIA para el bot semi-automatico.

(Antes signal.py; renombrado porque 'signal' colisiona con el modulo estandar de
Python y rompe a pyarrow/pandas al importar.)

Calcula, sobre el par REAL de trading (Kraken BTC/EUR), el regimen de mercado de
HOY y la exposicion objetivo segun la estrategia combinada VALIDADA (valoracion
Mayer condicionada al filtro de tendencia MA200 con histeresis). Emite una
recomendacion para que el HUMANO la apruebe; NO ejecuta ordenes.

Uso:
    python src/daily_signal.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

import ccxt
import numpy as np
import pandas as pd
import urllib.error

try:
    from liq_map import build_liq_map
    from liq_chart import build_chart, send_chart_telegram
    _LIQ_AVAILABLE = True
except Exception:
    _LIQ_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")
STATE_FILE = os.path.join(REPORTS_DIR, "position_state.json")

# NOTA: motif_search.py demostro que NINGUN patron de velas (8h-200h) predice la
# direccion (pares 100% identicos -> direccion opuesta 49% de las veces). El edge
# esta SOLO en el regimen macro de abajo. No anadir señales intradia de velas.
TREND_MA = 200          # robusto en 150-225 segun robustness.py
VAL_MA = 200            # base del Mayer Multiple
HYST_UP, HYST_DN = 1.02, 0.98
HALVING_2024 = pd.Timestamp("2024-04-20", tz="UTC")

# Constantes del ciclo derivadas de cycle_phases.py (15 anios, 3 ciclos):
# techo a 525-546 dias post-halving; suelo a 777/889/924 dias.
PEAK_DAY = 535
BOTTOM_LO, BOTTOM_HI = 860, 925


def cycle_expectation(days: int) -> tuple[str, str]:
    """Fase macro del ciclo y expectativa segun el patron historico."""
    if days < PEAK_DAY:
        return "ALCISTA", f"techo historico ~dia {PEAK_DAY} (en ~{PEAK_DAY - days} dias)"
    if days < BOTTOM_HI:
        lo, hi = max(0, BOTTOM_LO - days), BOTTOM_HI - days
        return "BAJISTA", f"suelo historico ~dia {BOTTOM_LO}-{BOTTOM_HI} (en ~{lo}-{hi} dias)"
    return "SUELO / transicion", "mas alla del suelo tipico; vigilar reversion al alza"


def mayer_weight(m: float) -> float:
    if np.isnan(m):
        return 0.0
    for hi, w in [(0.8, 1.0), (1.0, 0.85), (1.4, 0.60), (1.8, 0.40), (2.2, 0.20), (2.4, 0.10)]:
        if m <= hi:
            return w
    return 0.0


def compute_rsi(close: np.ndarray, period: int = 14) -> float:
    """RSI clásico de Wilder sobre los últimos `period+1` cierres."""
    delta = np.diff(close[-(period + 1):]).astype(float)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def fetch_long_short_ratio(symbol="BTCUSDT", period="4h", limit=1) -> dict:
    """Long/Short ratio global de Binance futuros. Gratis, sin API key."""
    try:
        url = (f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
               f"?symbol={symbol}&period={period}&limit={limit}")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.load(r)
        ls = float(data[-1]["longShortRatio"])
        longs = float(data[-1]["longAccount"]) * 100
        shorts = float(data[-1]["shortAccount"]) * 100
        if ls > 1.8:
            interp = "dominan LARGOS — mercado sobrecalentado (contrario bajista)"
        elif ls < 0.7:
            interp = "dominan CORTOS — capitulacion/miedo (contrario alcista)"
        else:
            interp = "equilibrado"
        return {"ratio": round(ls, 2), "longs_pct": round(longs, 1),
                "shorts_pct": round(shorts, 1), "interp": interp}
    except Exception:  # noqa: BLE001
        return {"ratio": None, "longs_pct": None, "shorts_pct": None, "interp": "no disponible"}


def fetch_open_interest(symbol="BTCUSDT") -> dict:
    """Open Interest de Binance futuros + cambio 24h. Proxy del mapa de liquidaciones."""
    try:
        # OI actual
        url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        with urllib.request.urlopen(url_oi, timeout=10) as r:
            oi_now = float(json.load(r)["openInterest"])
        # OI histórico (últimas 2 velas de 1d para calcular cambio)
        url_hist = (f"https://fapi.binance.com/futures/data/openInterestHist"
                    f"?symbol={symbol}&period=1d&limit=2")
        with urllib.request.urlopen(url_hist, timeout=10) as r:
            hist = json.load(r)
        oi_prev = float(hist[0]["sumOpenInterest"])
        oi_curr = float(hist[1]["sumOpenInterest"])
        chg_pct = (oi_curr / oi_prev - 1) * 100
        if chg_pct > 5:
            interp = "OI creciendo fuerte — apalancamiento en aumento (riesgo de barrida)"
        elif chg_pct < -5:
            interp = "OI cayendo — desapalancamiento/liquidaciones recientes"
        else:
            interp = "OI estable"
        return {"oi_btc": round(oi_now, 0), "chg_24h_pct": round(chg_pct, 1), "interp": interp}
    except Exception:  # noqa: BLE001
        return {"oi_btc": None, "chg_24h_pct": None, "interp": "no disponible"}


def fetch_fear_greed() -> dict:
    """Fear & Greed Index (0-100). Fuente: alternative.me (gratuito, sin key)."""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.load(r)["data"][0]
            val = int(data["value"])
            label = data["value_classification"]
            if val <= 24:
                interp = "miedo extremo (suelos historicos)"
            elif val <= 49:
                interp = "miedo"
            elif val <= 74:
                interp = "codicia"
            else:
                interp = "codicia extrema (techos historicos)"
            return {"value": val, "label": label, "interp": interp}
    except Exception:  # noqa: BLE001
        return {"value": None, "label": "N/A", "interp": "no disponible"}


def fetch_funding_rate(symbol="BTCUSDT") -> dict:
    """Funding rate del perpetuo BTC/USDT en Binance (cada 8h). Gratuito, sin key."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=3"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.load(r)
        rates = [float(d["fundingRate"]) for d in data]
        last = rates[-1] * 100          # en %
        avg = sum(rates) / len(rates) * 100
        if last > 0.05:
            interp = "mercado sobrecalentado (señal contraria bajista)"
        elif last < -0.01:
            interp = "capitulacion/miedo en futuros (señal contraria alcista)"
        else:
            interp = "neutral"
        return {"last_pct": round(last, 4), "avg3_pct": round(avg, 4), "interp": interp}
    except Exception:  # noqa: BLE001
        return {"last_pct": None, "avg3_pct": None, "interp": "no disponible"}


def fetch_live_daily(symbol="BTC/USD", limit=600) -> pd.DataFrame:
    ex = ccxt.kraken({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("date").reset_index(drop=True)


def trend_state(close: np.ndarray, ma: np.ndarray) -> int:
    cur = 0
    for i in range(len(close)):
        if np.isnan(ma[i]):
            continue
        if cur == 0 and close[i] > ma[i] * HYST_UP:
            cur = 1
        elif cur == 1 and close[i] < ma[i] * HYST_DN:
            cur = 0
    return cur


def load_position() -> float:
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE)).get("weight", 0.0)
    return 0.0


def compute_choppiness(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Choppiness Index. <38.2 = tendencia fuerte. >61.8 = mercado en rango."""
    n = period
    if len(close) < n + 1:
        return float("nan")
    h = high[-n:]
    l = low[-n:]
    c = close[-(n + 1):]
    tr = np.maximum(h - l,
         np.maximum(np.abs(h - c[:-1]), np.abs(l - c[:-1])))
    atr_sum = tr.sum()
    hh = h.max()
    ll = l.min()
    if hh == ll:
        return float("nan")
    return round(100 * np.log10(atr_sum / (hh - ll)) / np.log10(n), 1)


def compute_bb_width(close: np.ndarray, period: int = 20) -> tuple[float, float]:
    """Bollinger Bands Width (bb_width) y %B actual.
    Devuelve (bb_width_pct, bb_pct_b) donde:
      bb_width_pct = (upper-lower)/mid  →  alta expansion = momentum activo
      bb_pct_b     = (price-lower)/(upper-lower)  →  0=lower, 1=upper
    """
    s = pd.Series(close[-period - 20:])   # histórico suficiente para cuantiles
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = ((upper - lower) / mid).dropna()
    if len(width) < 20:
        return float("nan"), float("nan")
    bb_width_pct = round(float(width.iloc[-1]) * 100, 2)
    # %B actual
    m = float(mid.iloc[-1])
    u = float(upper.iloc[-1])
    lo = float(lower.iloc[-1])
    bb_pct_b = round((close[-1] - lo) / (u - lo + 1e-9), 3)
    # Percentil histórico del width (últimas 200 velas disponibles)
    s_full = pd.Series(close)
    mid_f = s_full.rolling(period).mean()
    std_f = s_full.rolling(period).std()
    w_full = ((mid_f + 2*std_f - (mid_f - 2*std_f)) / mid_f).dropna()
    pct = round(float((w_full < width.iloc[-1]).mean()) * 100, 0)
    return bb_width_pct, bb_pct_b, int(pct)


def build_signal() -> dict:
    df = fetch_live_daily()
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    ma_trend = df["close"].rolling(TREND_MA).mean().values
    ma_val = df["close"].rolling(VAL_MA).mean().values

    price = close[-1]
    mayer = price / ma_val[-1]
    trend = trend_state(close, ma_trend)        # 1 alcista / 0 bajista
    target = mayer_weight(mayer) * trend
    current = load_position()
    delta = target - current
    days_hv = (df["date"].iloc[-1] - HALVING_2024).days

    if delta > 0.05:
        accion = f"COMPRAR hasta {target*100:.0f}% (ahora {current*100:.0f}%)"
    elif delta < -0.05:
        accion = f"VENDER hasta {target*100:.0f}% (ahora {current*100:.0f}%)"
    else:
        accion = f"MANTENER en {current*100:.0f}% (objetivo {target*100:.0f}%)"

    regimen = "ALCISTA (precio sobre MA%d)" % TREND_MA if trend else "BAJISTA (precio bajo MA%d)" % TREND_MA
    fase_ciclo, expectativa = cycle_expectation(int(days_hv))

    rsi = compute_rsi(close)
    chop = compute_choppiness(high, low, close)
    bb_width_pct, bb_pct_b, bb_width_pctile = compute_bb_width(close)
    fg = fetch_fear_greed()
    fr = fetch_funding_rate()
    ls = fetch_long_short_ratio()
    oi = fetch_open_interest()
    liq = build_liq_map(float(price)) if _LIQ_AVAILABLE else {"available": False}

    return {
        "fecha": str(df["date"].iloc[-1].date()),
        "precio_usd": round(float(price), 2),
        "ma200": round(float(ma_val[-1]), 2),
        "mayer": round(float(mayer), 3),
        "rsi": rsi,
        "choppiness": chop,
        "bb_width_pct": bb_width_pct,
        "bb_pct_b": bb_pct_b,
        "bb_width_pctile": bb_width_pctile,
        "regimen": regimen,
        "exposicion_objetivo_pct": round(float(target) * 100, 1),
        "exposicion_actual_pct": round(float(current) * 100, 1),
        "accion_sugerida": accion,
        "dias_desde_halving": int(days_hv),
        "fase_ciclo": fase_ciclo,
        "expectativa_ciclo": expectativa,
        "fear_greed": fg,
        "funding_rate": fr,
        "long_short": ls,
        "open_interest": oi,
        "liq_map": liq,
    }


def build_conclusion(s: dict) -> str:
    fg = s.get("fear_greed", {})
    fr = s.get("funding_rate", {})
    ls = s.get("long_short", {})
    oi = s.get("open_interest", {})
    trend = s["regimen"].startswith("ALCISTA")
    mayer = s["mayer"]
    rsi = s.get("rsi")
    chop = s.get("choppiness")
    bb_width_pct = s.get("bb_width_pct")
    bb_pct_b = s.get("bb_pct_b")
    bb_width_pctile = s.get("bb_width_pctile")
    fg_val = fg.get("value")
    fr_val = fr.get("last_pct")
    ls_ratio = ls.get("ratio")
    oi_chg = oi.get("chg_24h_pct")
    days = s["dias_desde_halving"]

    signals = []
    bullish = 0
    bearish = 0

    # Régimen MA200
    if trend:
        signals.append(
            "✅ MA200: ALCISTA\n"
            f"   Precio sobre su media de 200 días. La tendencia mayor es tu aliada.")
        bullish += 1
    else:
        pct_needed = (s["ma200"] / s["precio_usd"] - 1) * 100
        signals.append(
            f"🔴 MA200: BAJISTA (faltan +{pct_needed:.0f}%)\n"
            f"   BTC está por debajo de su media de 200 días. Históricamente el 70% de las caídas fuertes ocurren en este régimen. No comprar hasta que lo cruce.")
        bearish += 1

    # RSI — umbrales optimizados en backtest 50 fechas:
    # sobrevendido=20 (63% precision), sobrecomprado=80 (76% precision)
    if rsi is not None:
        if rsi <= 20:
            signals.append(
                f"✅ RSI: {rsi} — SOBREVENDIDO EXTREMO\n"
                f"   Capitulacion historica. Backtest: 63% sube en 30d desde estos niveles.")
            bullish += 1
        elif rsi <= 35:
            signals.append(
                f"🟡 RSI: {rsi} — sobrevendido moderado\n"
                f"   Zona de apoyo pero no señal de compra urgente.")
        elif rsi >= 80:
            signals.append(
                f"🔴 RSI: {rsi} — SOBRECOMPRADO EXTREMO\n"
                f"   Backtest: 76% corrige en 30d desde estos niveles.")
            bearish += 1
        elif rsi >= 65:
            signals.append(
                f"🟡 RSI: {rsi} — sobrecomprado moderado\n"
                f"   Movimiento extendido. Precaucion sin señal de venta urgente.")
        else:
            signals.append(
                f"⚪ RSI: {rsi} — neutral\n"
                f"   Sin señal extrema. Recorrido en ambas direcciones.")

    # Mayer — umbral optimo barato=0.8 (61% precision), caro=2.2 (79% precision)
    if mayer < 0.80:
        signals.append(
            f"✅ Mayer: {mayer:.2f} — MUY BARATO\n"
            f"   BTC cotiza {(1-mayer)*100:.0f}% bajo su media. Backtest: 61% sube en 30d con Mayer <0.80.")
        bullish += 1
    elif mayer < 0.95:
        signals.append(
            f"🟡 Mayer: {mayer:.2f} — barato moderado\n"
            f"   Por debajo de la media de 200d. Zona de acumulacion historica pero sin señal extrema.")
    elif mayer > 2.2:
        signals.append(
            f"🔴 Mayer: {mayer:.2f} — EUFORIA\n"
            f"   Backtest: 79% corrige en 30d con Mayer >2.2. Zona de distribucion.")
        bearish += 1
    elif mayer > 1.6:
        signals.append(
            f"🟡 Mayer: {mayer:.2f} — caro moderado\n"
            f"   Por encima de la media. Reducir exposicion progresivamente.")
    else:
        signals.append(
            f"⚪ Mayer: {mayer:.2f} — valoracion neutra\n"
            f"   Precio en rango normal. Sin señal extrema.")

    # Fear & Greed — umbral optimo miedo=15 (69% precision), codicia=80 (52% precision)
    if fg_val is not None:
        if fg_val <= 15:
            signals.append(
                f"✅ Fear&Greed: {fg_val}/100 — PANICO EXTREMO\n"
                f"   Backtest: 69% sube en 30d cuando F&G ≤15. Los suelos de ciclo BTC ocurrieron siempre en esta zona.")
            bullish += 1
        elif fg_val <= 30:
            signals.append(
                f"🟡 Fear&Greed: {fg_val}/100 — miedo\n"
                f"   Sentimiento negativo pero no panico. Zona de acumulacion de largo plazo.")
        elif fg_val >= 80:
            signals.append(
                f"🔴 Fear&Greed: {fg_val}/100 — EUFORIA\n"
                f"   Backtest: 52% corrige en 30d con F&G ≥80. Los techos de ciclo coinciden con esta zona.")
            bearish += 1
        elif fg_val >= 65:
            signals.append(
                f"🟡 Fear&Greed: {fg_val}/100 — codicia\n"
                f"   Optimismo elevado. Reducir exposicion progresivamente.")
        else:
            signals.append(
                f"⚪ Fear&Greed: {fg_val}/100 — sentimiento neutro\n"
                f"   Ni panico ni euforia. Mercado equilibrado.")

    # Funding rate
    if fr_val is not None:
        if fr_val > 0.05:
            signals.append(
                f"🔴 Funding: {fr_val:+.4f}%/8h — SOBRECALENTADO\n"
                f"   Los largos están pagando mucho para mantener posición. Señal de exceso de optimismo en futuros. Suele preceder correcciones.")
            bearish += 1
        elif fr_val < -0.01:
            signals.append(
                f"✅ Funding: {fr_val:+.4f}%/8h — CAPITULACIÓN\n"
                f"   Los cortos pagan por mantener posición. El mercado de futuros está apostando fuerte a la baja, lo que puede provocar un short squeeze.")
            bullish += 1
        else:
            signals.append(
                f"⚪ Funding: {fr_val:+.4f}%/8h — neutral\n"
                f"   Coste de financiación equilibrado. Sin presión extrema en futuros.")

    # Long/Short ratio
    if ls_ratio is not None:
        lp = ls.get("longs_pct", 0)
        sp = ls.get("shorts_pct", 0)
        if ls_ratio > 1.8:
            signals.append(
                f"🔴 L/S ratio: {ls_ratio} ({lp:.0f}% largos / {sp:.0f}% cortos)\n"
                f"   Demasiada gente apostando al alza. Si el precio cae un poco más, sus stops se liquidan en cascada empujando aún más abajo (barrida de largos).")
            bearish += 1
        elif ls_ratio < 0.7:
            signals.append(
                f"✅ L/S ratio: {ls_ratio} ({lp:.0f}% largos / {sp:.0f}% cortos)\n"
                f"   Mayoría apostando a la baja. Si el precio rebota, los cortos se liquidan en cadena (short squeeze) acelerando la subida.")
            bullish += 1
        else:
            signals.append(
                f"⚪ L/S ratio: {ls_ratio} ({lp:.0f}% largos / {sp:.0f}% cortos)\n"
                f"   Posicionamiento equilibrado. Sin riesgo de barrida masiva en ninguna dirección.")

    # Open Interest
    if oi_chg is not None:
        oi_btc = oi.get("oi_btc", 0)
        if oi_chg > 5:
            signals.append(
                f"🔴 Open Interest: {oi_btc:,.0f} BTC ({oi_chg:+.1f}% 24h)\n"
                f"   El apalancamiento total está creciendo. Más dinero en futuros = más combustible para movimientos bruscos en ambas direcciones. Riesgo elevado.")
            bearish += 1
        elif oi_chg < -5:
            signals.append(
                f"✅ Open Interest: {oi_btc:,.0f} BTC ({oi_chg:+.1f}% 24h)\n"
                f"   El mercado se está desapalancando. Las posiciones forzadas ya se liquidaron. Suele preceder movimientos más limpios y sostenibles.")
            bullish += 1
        else:
            signals.append(
                f"⚪ Open Interest: {oi_btc:,.0f} BTC ({oi_chg:+.1f}% 24h)\n"
                f"   Apalancamiento estable. Sin señal de acumulación ni liquidación masiva reciente.")

    # Choppiness Index — backtest: 65.4% precision, edge +11.4%
    if chop is not None and not (chop != chop):  # not NaN
        if chop < 38.2:
            signals.append(
                f"✅ Choppiness: {chop} — TENDENCIA FUERTE\n"
                f"   Backtest: 65.4% de precision (+11.4% vs azar). BTC en tendencia clara; el regimen actual probablemente se mantiene.")
            bullish += 1
        elif chop > 61.8:
            signals.append(
                f"🔴 Choppiness: {chop} — MERCADO EN RANGO\n"
                f"   Mercado lateral sin direccion. Las señales de tendencia son menos fiables en este entorno.")
            bearish += 1
        else:
            signals.append(
                f"⚪ Choppiness: {chop} — mercado mixto\n"
                f"   Ni tendencia clara ni rango puro. Zona intermedia (38.2-61.8).")

    # Bollinger Bands Width — backtest: 58.0% precision, edge +3.9%
    if bb_width_pct is not None and not (bb_width_pct != bb_width_pct):
        pctile_str = f"percentil {bb_width_pctile}%" if bb_width_pctile is not None else ""
        if bb_width_pctile is not None and bb_width_pctile >= 80:
            signals.append(
                f"✅ BB Width: {bb_width_pct}% ({pctile_str}) — EXPANSION ALTA\n"
                f"   Las bandas se estan abriendo. Backtest: 58% precision. Los arranques de ciclo y los crashes comienzan con alta expansion. Vigilar la direccion del movimiento.")
            bullish += 1
        elif bb_width_pctile is not None and bb_width_pctile <= 20:
            signals.append(
                f"🟡 BB Width: {bb_width_pct}% ({pctile_str}) — COMPRESION\n"
                f"   Bandas comprimidas. Movimiento fuerte inminente en cualquier direccion. Puede ser el inicio de un nuevo tramo.")
        else:
            signals.append(
                f"⚪ BB Width: {bb_width_pct}% ({pctile_str}) — volatilidad normal\n"
                f"   Sin señal de expansion ni compresion extrema.")

    # Ciclo
    lo = max(0, BOTTOM_LO - days)
    hi = BOTTOM_HI - days
    if 0 <= lo <= 60:
        signals.append(
            f"✅ Ciclo: día {days} post-halving — SUELO MUY PRÓXIMO\n"
            f"   En los 3 ciclos anteriores el suelo llegó entre los días {BOTTOM_LO}-{BOTTOM_HI}. Quedan ~{lo}-{hi} días. La fase más dura está casi terminada.")
        bullish += 1
    elif lo > 0:
        signals.append(
            f"⚪ Ciclo: día {days} post-halving\n"
            f"   Suelo histórico esperado en ~{lo}-{hi} días (días {BOTTOM_LO}-{BOTTOM_HI} del ciclo). Paciencia.")
    else:
        signals.append(
            f"⚪ Ciclo: día {days} post-halving\n"
            f"   Más allá del suelo típico del ciclo. Vigilar señales de giro.")

    # Mapa de liquidaciones (si el colector tiene datos)
    liq = s.get("liq_map", {})
    if liq.get("available"):
        signals.append(liq["summary_text"])
        # Contar señal: si dominan liquidaciones de largos (posible suelo) → bullish
        if liq.get("dominant_side") == "LONGS":
            longs_pct_liq = liq["longs_usd"] / max(liq["total_usd"], 1) * 100
            if longs_pct_liq > 65:
                bullish += 1
        elif liq.get("dominant_side") == "SHORTS":
            shorts_pct_liq = liq["shorts_usd"] / max(liq["total_usd"], 1) * 100
            if shorts_pct_liq > 65:
                bearish += 1

    # Veredicto
    if not trend:
        veredicto = "⏳ ESPERAR. Sin tendencia alcista confirmada, el riesgo supera al beneficio de entrar antes."
    elif bullish >= 4 and bearish == 0:
        veredicto = "🚀 TODAS LAS SEÑALES ALINEADAS. Estrategia recomienda comprar."
    elif bullish > bearish:
        veredicto = f"📈 Señales mayoritariamente alcistas ({bullish} vs {bearish}). Estrategia recomienda exposición parcial."
    elif bearish > bullish:
        veredicto = f"📉 Señales mayoritariamente bajistas ({bearish} vs {bullish}). Estrategia recomienda reducir exposición."
    else:
        veredicto = "⚖️ Señales mixtas. Mantener posición actual."

    return "\n".join(signals) + f"\n\n{veredicto}"


def render(s: dict) -> str:
    conclusion = build_conclusion(s)
    return (
        f"📊 SEÑAL BITCOIN — {s['fecha']}\n"
        f"Precio: ${s['precio_usd']:,.0f}   |   MA200: ${s['ma200']:,.0f}\n"
        f"Mayer: {s['mayer']:.2f}   |   Régimen: {s['regimen']}\n"
        f"🔄 Ciclo: día {s['dias_desde_halving']} post-halving · fase {s['fase_ciclo']}\n"
        f"   {s['expectativa_ciclo']}\n"
        f"———\n"
        f"{conclusion}\n"
        f"———\n"
        f"Exposición objetivo: {s['exposicion_objetivo_pct']:.0f}%  "
        f"(actual {s['exposicion_actual_pct']:.0f}%)\n"
        f"👉 {s['accion_sugerida']}\n"
        f"(Aprueba tú la orden; el bot no ejecuta nada solo.)"
    )


def notify(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[Telegram no configurado: define TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env]")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=15) as r:
            print("[Telegram] enviado" if r.status == 200 else f"[Telegram] HTTP {r.status}")
    except Exception as e:  # noqa: BLE001
        print(f"[Telegram] error: {e}")


def main() -> None:
    s = build_signal()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    json.dump(s, open(os.path.join(REPORTS_DIR, "latest_signal.json"), "w"), indent=2, ensure_ascii=False)
    text = render(s)
    print(text)
    notify(text)

    # Gráfico de liquidaciones (si hay datos)
    if _LIQ_AVAILABLE:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat  = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat:
            buf = build_chart(current_price=s["precio_usd"])
            if buf:
                liq = s.get("liq_map", {})
                total = liq.get("total_usd", 0)
                longs_usd = liq.get("longs_usd", 0)
                shorts_usd = liq.get("shorts_usd", 0)
                caption = (
                    f"Mapa de liquidaciones BTC 24h\n"
                    f"Total: ${total/1e6:.1f}M  "
                    f"Largos: ${longs_usd/1e6:.1f}M  "
                    f"Cortos: ${shorts_usd/1e6:.1f}M\n"
                    f"Linea amarilla = precio actual ${s['precio_usd']:,.0f}"
                )
                ok = send_chart_telegram(token, chat, buf, caption)
                print(f"[liq_chart] {'enviado' if ok else 'sin datos o error'}")
            else:
                print("[liq_chart] sin datos de liquidaciones aun")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
