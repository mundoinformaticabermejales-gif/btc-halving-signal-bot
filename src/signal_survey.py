"""
Backtest masivo de señales encontradas en bots de trading gratuitos.
Testa ~20 indicadores nuevos contra datos BTC 2018-2026 a horizonte 30d.
Salida: ranking por precision + envio a Telegram.
"""
from __future__ import annotations
import os, sys, json, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

HORIZON = 30
N = 500
SEED = 42

# ── Carga de precios ─────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "btc_usdt_1d_binance.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ── Calculo de todos los indicadores ─────────────────────────────────────────

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].values
    v = df["volume"].values if "volume" in df.columns else np.ones(len(c))
    h = df["high"].values if "high" in df.columns else c
    l = df["low"].values if "low" in df.columns else c

    # --- Medias moviles ---
    def sma(n): return pd.Series(c).rolling(n).mean().values
    def ema(n):
        s = pd.Series(c)
        return s.ewm(span=n, adjust=False).mean().values

    df["ma50"]  = sma(50)
    df["ma200"] = sma(200)
    df["ema20"] = ema(20)
    df["ema50"] = ema(50)
    df["ema200"]= ema(200)
    df["mayer"] = df["close"] / df["ma200"]

    # Golden/Death Cross: ma50 vs ma200
    df["golden_cross"] = (df["ma50"] > df["ma200"]).astype(int)  # 1=alcista, 0=bajista

    # --- RSI ---
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # RSI semanal (periodo 14 sobre velas de 7d proxy = rolling 98d con step 7)
    delta7 = pd.Series(c).diff(7)
    g7 = delta7.clip(lower=0).rolling(14).mean()
    l7 = (-delta7.clip(upper=0)).rolling(14).mean()
    df["rsi_weekly"] = 100 - 100 / (1 + g7 / l7.replace(0, np.nan))

    # --- MACD ---
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd_line - signal_line).values
    df["macd_cross"] = (macd_line > signal_line).astype(int).values  # 1=bull, 0=bear

    # --- Bollinger Bands ---
    bb_mid = pd.Series(c).rolling(20).mean()
    bb_std = pd.Series(c).rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_pct"] = ((pd.Series(c) - bb_lower) / (bb_upper - bb_lower)).values  # 0=lower, 1=upper
    df["bb_width"] = ((bb_upper - bb_lower) / bb_mid).values  # volatilidad relativa

    # --- Bollinger Bands %B extremos ---
    # <0.05 = muy abajo (alcista), >0.95 = muy arriba (bajista)

    # --- ATR (Average True Range) ---
    tr = pd.DataFrame({
        "hl": pd.Series(h) - pd.Series(l),
        "hc": (pd.Series(h) - pd.Series(c).shift()).abs(),
        "lc": (pd.Series(l) - pd.Series(c).shift()).abs(),
    }).max(axis=1)
    df["atr14"] = tr.rolling(14).mean().values
    df["atr_pct"] = (df["atr14"] / df["close"]).values  # ATR como % del precio

    # --- SuperTrend (ATR x3 sobre HL midpoint) ---
    hl_mid = (pd.Series(h) + pd.Series(l)) / 2
    st_mult = 3.0
    upper_band = hl_mid + st_mult * tr.rolling(14).mean()
    lower_band = hl_mid - st_mult * tr.rolling(14).mean()
    supertrend = pd.Series(np.nan, index=range(len(c)))
    direction = pd.Series(0, index=range(len(c)))  # 1=bull, -1=bear
    for i in range(1, len(c)):
        if np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]):
            continue
        prev_upper = upper_band.iloc[i-1] if not np.isnan(upper_band.iloc[i-1]) else upper_band.iloc[i]
        prev_lower = lower_band.iloc[i-1] if not np.isnan(lower_band.iloc[i-1]) else lower_band.iloc[i]
        upper_band.iloc[i] = min(upper_band.iloc[i], prev_upper) if c[i-1] <= prev_upper else upper_band.iloc[i]
        lower_band.iloc[i] = max(lower_band.iloc[i], prev_lower) if c[i-1] >= prev_lower else lower_band.iloc[i]
        if direction.iloc[i-1] <= 0 and c[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif direction.iloc[i-1] >= 0 and c[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    df["supertrend"] = direction.values  # 1=alcista, -1=bajista

    # --- Stochastic RSI ---
    rsi_s = df["rsi"]
    rsi_min = rsi_s.rolling(14).min()
    rsi_max = rsi_s.rolling(14).max()
    df["stoch_rsi"] = ((rsi_s - rsi_min) / (rsi_max - rsi_min + 1e-9)).values

    # --- CCI ---
    tp = (pd.Series(h) + pd.Series(l) + pd.Series(c)) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df["cci"] = ((tp - tp_ma) / (0.015 * tp_md)).values

    # --- Williams %R ---
    highest_h = pd.Series(h).rolling(14).max()
    lowest_l  = pd.Series(l).rolling(14).min()
    df["willr"] = (-(highest_h - pd.Series(c)) / (highest_h - lowest_l + 1e-9) * 100).values

    # --- MFI (Money Flow Index) ---
    tp2 = (pd.Series(h) + pd.Series(l) + pd.Series(c)) / 3
    mf = tp2 * pd.Series(v)
    pos_mf = mf.where(tp2 > tp2.shift(), 0.0)
    neg_mf = mf.where(tp2 < tp2.shift(), 0.0)
    mfr = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-9)
    df["mfi"] = (100 - 100 / (1 + mfr)).values

    # --- OBV momentum (cambio % OBV en 20 dias) ---
    obv = pd.Series(np.where(pd.Series(c).diff() >= 0, pd.Series(v), -pd.Series(v))).cumsum()
    df["obv_mom"] = (obv / obv.shift(20) - 1).values  # >0 = flujo positivo

    # --- ADX ---
    plus_dm  = pd.Series(np.where((pd.Series(h).diff() > 0) & (pd.Series(h).diff() > -pd.Series(l).diff()), pd.Series(h).diff(), 0))
    minus_dm = pd.Series(np.where((-pd.Series(l).diff() > 0) & (-pd.Series(l).diff() > pd.Series(h).diff()), -pd.Series(l).diff(), 0))
    tr_sm   = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / (tr_sm + 1e-9)
    minus_di= 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / (tr_sm + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean().values

    # --- Momentum (ROC 10) ---
    df["roc10"] = (pd.Series(c) / pd.Series(c).shift(10) - 1).values * 100

    # --- TSI (True Strength Index) ---
    pc = pd.Series(c).diff()
    double_smooth = lambda s, r, s2: s.ewm(span=r, adjust=False).mean().ewm(span=s2, adjust=False).mean()
    df["tsi"] = (100 * double_smooth(pc, 25, 13) / (double_smooth(pc.abs(), 25, 13) + 1e-9)).values

    # --- Ichimoku: precio vs Kumo (nube) ---
    tenkan = (pd.Series(h).rolling(9).max() + pd.Series(l).rolling(9).min()) / 2
    kijun  = (pd.Series(h).rolling(26).max() + pd.Series(l).rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((pd.Series(h).rolling(52).max() + pd.Series(l).rolling(52).min()) / 2).shift(26)
    df["ichimoku_bull"] = ((pd.Series(c) > span_a) & (pd.Series(c) > span_b)).astype(int).values  # 1=sobre nube

    # --- Choppiness Index (mercado en tendencia o rango) ---
    atr_sum = tr.rolling(14).sum()
    hh14 = pd.Series(h).rolling(14).max()
    ll14 = pd.Series(l).rolling(14).min()
    df["chop"] = (100 * np.log10(atr_sum / (hh14 - ll14 + 1e-9)) / np.log10(14)).values
    # <38.2 = tendencia fuerte, >61.8 = rango/chop

    # --- Retorno forward ---
    df["fwd_30d"] = pd.Series(c).shift(-HORIZON) / pd.Series(c) - 1

    return df


# ── Backtest de un indicador ──────────────────────────────────────────────────

def backtest_indicator(df: pd.DataFrame, col: str, bull_condition, bear_condition,
                       label: str, bull_label="SUBE", bear_label="BAJA") -> dict:
    """
    bull_condition / bear_condition: funciones lambda(series) -> bool series
    """
    sub = df.dropna(subset=[col, "fwd_30d"]).reset_index(drop=True)
    if len(sub) < 100:
        return {"label": label, "n": len(sub), "error": "datos insuficientes"}

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(sub), size=min(N, len(sub)), replace=False)
    sample = sub.iloc[idx].reset_index(drop=True)

    bull_mask = bull_condition(sample[col])
    bear_mask = bear_condition(sample[col])
    neutral_mask = ~bull_mask & ~bear_mask

    results = []
    for i in range(len(sample)):
        fwd = sample["fwd_30d"].iloc[i]
        if bull_mask.iloc[i]:
            signal = "SUBE"
            correct = fwd > 0
        elif bear_mask.iloc[i]:
            signal = "BAJA"
            correct = fwd < 0
        else:
            signal = "NEUTRO"
            correct = None
        results.append({"signal": signal, "correct": correct, "fwd": fwd})

    res_df = pd.DataFrame(results)
    actionable = res_df[res_df["signal"] != "NEUTRO"]
    n_act = len(actionable)
    if n_act == 0:
        return {"label": label, "n": n_act, "error": "sin señales accionables"}

    accuracy = float(actionable["correct"].apply(lambda x: 1.0 if x else 0.0).mean())
    n_bull = int((actionable["signal"] == "SUBE").sum())
    n_bear = int((actionable["signal"] == "BAJA").sum())
    base_rate = float((sub["fwd_30d"] > 0).mean())

    bull_rows = res_df[res_df["signal"] == "SUBE"]
    bear_rows = res_df[res_df["signal"] == "BAJA"]
    bull_acc = float(bull_rows["correct"].apply(lambda x: 1.0 if x else 0.0).mean()) if n_bull > 0 else None
    bear_acc = float(bear_rows["correct"].apply(lambda x: 1.0 if x else 0.0).mean()) if n_bear > 0 else None

    return {
        "label": label,
        "col": col,
        "n_total": len(sample),
        "n_actionable": n_act,
        "n_bull": n_bull,
        "n_bear": n_bear,
        "accuracy": round(accuracy * 100, 1),
        "bull_accuracy": round(bull_acc * 100, 1) if bull_acc else None,
        "bear_accuracy": round(bear_acc * 100, 1) if bear_acc else None,
        "base_rate": round(base_rate * 100, 1),
        "edge": round((accuracy - base_rate) * 100, 1),
        "neutral_pct": round(float(neutral_mask.mean()) * 100, 1),
    }


# ── Definicion de todos los indicadores a testar ────────────────────────────

def run_all(df: pd.DataFrame) -> list[dict]:
    results = []

    # MA200 (referencia — ya en el bot)
    results.append(backtest_indicator(df, "golden_cross",
        lambda s: s == 1, lambda s: s == 0,
        "MA200 regimen (precio>MA200)"))

    # Golden/Death Cross 50/200
    results.append(backtest_indicator(df, "golden_cross",
        lambda s: s == 1, lambda s: s == 0,
        "Golden/Death Cross 50/200"))

    # RSI extremos (ya en el bot)
    results.append(backtest_indicator(df, "rsi",
        lambda s: s <= 20, lambda s: s >= 80,
        "RSI extremos (<=20 / >=80)"))

    # RSI semanal
    results.append(backtest_indicator(df, "rsi_weekly",
        lambda s: s <= 25, lambda s: s >= 75,
        "RSI semanal proxy (<=25 / >=75)"))

    # Mayer Multiple (ya en el bot)
    results.append(backtest_indicator(df, "mayer",
        lambda s: s < 0.80, lambda s: s > 2.20,
        "Mayer Multiple (<0.80 / >2.20)"))

    # MACD cruce
    results.append(backtest_indicator(df, "macd_cross",
        lambda s: s == 1, lambda s: s == 0,
        "MACD cruce linea/señal"))

    # MACD histograma positivo
    results.append(backtest_indicator(df, "macd_hist",
        lambda s: s > 0, lambda s: s < 0,
        "MACD histograma (>0 / <0)"))

    # Bollinger Bands %B
    results.append(backtest_indicator(df, "bb_pct",
        lambda s: s < 0.05, lambda s: s > 0.95,
        "Bollinger Bands %B extremos (<0.05 / >0.95)"))

    # Bollinger Bands width (expansión = momentum)
    results.append(backtest_indicator(df, "bb_width",
        lambda s: s > s.quantile(0.80),  # expansion = tendencia activa
        lambda s: s < s.quantile(0.20),  # compresion = suelo/techo proximo
        "Bollinger Bands Width (expansion/compresion)"))

    # SuperTrend
    results.append(backtest_indicator(df, "supertrend",
        lambda s: s == 1, lambda s: s == -1,
        "SuperTrend (ATR x3)"))

    # Stochastic RSI
    results.append(backtest_indicator(df, "stoch_rsi",
        lambda s: s < 0.10, lambda s: s > 0.90,
        "StochRSI extremos (<0.10 / >0.90)"))

    # CCI
    results.append(backtest_indicator(df, "cci",
        lambda s: s < -150, lambda s: s > 150,
        "CCI extremos (<-150 / >150)"))

    # Williams %R
    results.append(backtest_indicator(df, "willr",
        lambda s: s < -90, lambda s: s > -10,
        "Williams %R (<-90 / >-10)"))

    # MFI
    results.append(backtest_indicator(df, "mfi",
        lambda s: s < 20, lambda s: s > 80,
        "MFI (Money Flow Index) (<20 / >80)"))

    # OBV momentum
    results.append(backtest_indicator(df, "obv_mom",
        lambda s: s > 0.30,   # volumen positivo fuerte
        lambda s: s < -0.20,  # volumen negativo
        "OBV momentum 20d (>30% / <-20%)"))

    # ADX (tendencia fuerte)
    results.append(backtest_indicator(df, "adx",
        lambda s: s > 30,   # tendencia fuerte (cualquier direccion — combinar con MA)
        lambda s: s < 15,   # mercado sin tendencia
        "ADX tendencia (>30 fuerte / <15 rango)"))

    # ATR % del precio (volatilidad)
    results.append(backtest_indicator(df, "atr_pct",
        lambda s: s > s.quantile(0.85),  # alta volatilidad = posible suelo/techo
        lambda s: s < s.quantile(0.20),
        "ATR% volatilidad (alta/baja)"))

    # ROC 10
    results.append(backtest_indicator(df, "roc10",
        lambda s: s < -20,  # caida fuerte = posible rebote
        lambda s: s > 25,   # subida fuerte = posible techo
        "ROC10 momentum (<-20% / >+25%)"))

    # TSI
    results.append(backtest_indicator(df, "tsi",
        lambda s: s < -30, lambda s: s > 30,
        "TSI (True Strength Index) (<-30 / >30)"))

    # Ichimoku vs nube
    results.append(backtest_indicator(df, "ichimoku_bull",
        lambda s: s == 1, lambda s: s == 0,
        "Ichimoku (precio sobre/bajo la nube)"))

    # Choppiness Index
    results.append(backtest_indicator(df, "chop",
        lambda s: s < 38.2,  # tendencia fuerte
        lambda s: s > 61.8,  # mercado en rango
        "Choppiness Index (<38.2 tendencia / >61.8 rango)"))

    return results


# ── Presentacion de resultados ────────────────────────────────────────────────

def render_results(results: list[dict]) -> str:
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x["edge"], reverse=True)

    lines = []
    lines.append("RANKING DE INDICADORES — BTC 2018-2026")
    lines.append(f"Horizonte: {HORIZON} dias | n={N} fechas aleatorias")
    lines.append("Tasa base: ~51% (BTC sube en 30d)")
    lines.append("-" * 52)
    lines.append(f"{'INDICADOR':<35} {'PREC':>5} {'EDGE':>6} {'N':>4}")
    lines.append("-" * 52)

    for r in valid:
        label = r["label"][:34]
        acc   = f"{r['accuracy']}%"
        edge  = f"{r['edge']:+.1f}%"
        n     = str(r["n_actionable"])
        flag  = " ✓" if r["edge"] > 3 else ("  " if r["edge"] > 0 else " ✗")
        lines.append(f"{label:<35} {acc:>5} {edge:>6} {n:>4}{flag}")

    lines.append("-" * 52)
    lines.append("")
    lines.append("NUEVOS CANDIDATOS A INTEGRAR (edge >+3%):")
    candidates = [r for r in valid if r["edge"] > 3.0 and r["col"] not in ("golden_cross", "rsi", "mayer")]
    if candidates:
        for r in candidates:
            lines.append(f"  + {r['label']} (prec {r['accuracy']}%, edge {r['edge']:+.1f}%)")
    else:
        lines.append("  Ninguno supera el umbral de +3% de ventaja.")

    lines.append("")
    lines.append("YA EN EL BOT (referencia):")
    existing = [r for r in valid if r["col"] in ("golden_cross", "rsi", "mayer")]
    for r in existing:
        lines.append(f"  * {r['label']}: {r['accuracy']}% (edge {r['edge']:+.1f}%)")

    return "\n".join(lines)


def send_telegram(text: str) -> None:
    env_path = os.path.join(ROOT, ".env")
    token = chat_id = ""
    if os.path.exists(env_path):
        for line in open(env_path):
            k, _, v = line.strip().partition("=")
            if k == "TELEGRAM_BOT_TOKEN": token = v
            if k == "TELEGRAM_CHAT_ID":   chat_id = v
    if not token:
        print("Sin credenciales Telegram")
        return
    data = json.dumps({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req)
    ok = json.loads(r.read())["ok"]
    print(f"[Telegram] {'enviado' if ok else 'ERROR'}")


if __name__ == "__main__":
    print("Cargando precios BTC...")
    df_raw = load_prices()
    print(f"  {len(df_raw)} dias ({df_raw['date'].iloc[0].date()} -> {df_raw['date'].iloc[-1].date()})")

    print("Calculando indicadores...")
    df = build_indicators(df_raw)

    print(f"Ejecutando backtest {N} fechas, horizonte {HORIZON}d...")
    results = run_all(df)

    output = render_results(results)
    print("\n" + output)

    path = os.path.join(REPORTS_DIR, "signal_survey.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados en {path}")

    send_telegram(output)
