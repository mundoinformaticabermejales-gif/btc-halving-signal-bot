"""
Backtest historico de TODOS los indicadores del bot.

Para cada uno de 50 puntos de tiempo historicos:
  - Recoge el valor de cada indicador en ese momento
  - Mide lo que hizo BTC en los siguientes 7, 14 y 30 dias
  - Evalua si el indicador habria dado la señal correcta
  - Calcula el umbral optimo para maximizar precision

Indicadores testados:
  MA200 (regimen), RSI, Mayer Multiple, Fear & Greed,
  Funding Rate, Long/Short Ratio, Open Interest cambio 24h

Uso:
    python src/indicator_backtest.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

HORIZONS = [7, 14, 15, 30, 60, 90]   # dias hacia adelante que medimos


# ── Descarga de datos historicos ────────────────────────────────────────────

def load_daily_price() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "btc_usdt_1d_binance.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["ma200"] = df["close"].rolling(200).mean()
    df["mayer"] = df["close"] / df["ma200"]
    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return df


def fetch_fear_greed_history(limit=700) -> pd.DataFrame:
    """Historico F&G desde alternative.me (hasta ~2 años)."""
    print("  Descargando Fear & Greed historico...")
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = json.load(r)["data"]
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.normalize()
    df["fg"] = df["value"].astype(int)
    return df[["date", "fg"]].sort_values("date").reset_index(drop=True)


def fetch_funding_history(symbol="BTCUSDT", limit=1000) -> pd.DataFrame:
    """Funding rate cada 8h de Binance (hasta ~333 dias)."""
    print("  Descargando Funding Rate historico...")
    url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
           f"?symbol={symbol}&limit={limit}")
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = json.load(r)
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True).dt.normalize()
    df["funding"] = df["fundingRate"].astype(float) * 100
    # media diaria (3 pagos/dia)
    return df.groupby("date")["funding"].mean().reset_index()


def fetch_ls_history(symbol="BTCUSDT", period="1d", limit=30) -> pd.DataFrame:
    """Long/Short ratio diario de Binance (solo ultimos 30 dias)."""
    print("  Descargando Long/Short Ratio historico (30d max)...")
    url = (f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
           f"?symbol={symbol}&period={period}&limit={limit}")
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = json.load(r)
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True).dt.normalize()
    df["ls_ratio"] = df["longShortRatio"].astype(float)
    df["longs_pct"] = df["longAccount"].astype(float) * 100
    return df[["date", "ls_ratio", "longs_pct"]].sort_values("date").reset_index(drop=True)


def fetch_oi_history(symbol="BTCUSDT", period="1d", limit=60) -> pd.DataFrame:
    """Open Interest diario de Binance (solo ultimos ~60 dias)."""
    print("  Descargando Open Interest historico...")
    url = (f"https://fapi.binance.com/futures/data/openInterestHist"
           f"?symbol={symbol}&period={period}&limit={limit}")
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = json.load(r)
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True).dt.normalize()
    df["oi"] = df["sumOpenInterest"].astype(float)
    df["oi_chg"] = df["oi"].pct_change() * 100
    return df[["date", "oi", "oi_chg"]].sort_values("date").reset_index(drop=True)


# ── Construccion del dataset unificado ──────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    print("Descargando datos historicos...")
    price = load_daily_price()
    fg = fetch_fear_greed_history()
    funding = fetch_funding_history()
    ls = fetch_ls_history()
    oi = fetch_oi_history()

    df = price.merge(fg, on="date", how="left")
    df = df.merge(funding, on="date", how="left")
    df = df.merge(ls, on="date", how="left")
    df = df.merge(oi, on="date", how="left")

    # retornos forward
    for h in HORIZONS:
        df[f"fwd_{h}d"] = df["close"].shift(-h) / df["close"] - 1

    df = df.dropna(subset=["ma200", "mayer", "rsi"]).reset_index(drop=True)
    print(f"  Dataset: {len(df)} dias ({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    return df


# ── Analisis de indicadores ──────────────────────────────────────────────────

def accuracy_at_threshold(series: pd.Series, fwd: pd.Series,
                           threshold: float, direction: str) -> tuple[float, int]:
    """Precision cuando indicador cruza el umbral en la direccion dada."""
    if direction == "below":
        mask = series <= threshold
    else:
        mask = series >= threshold
    sub = fwd[mask].dropna()
    if len(sub) < 5:
        return 0.0, 0
    acc = (sub > 0).mean()
    return float(acc), len(sub)


def find_best_threshold(series: pd.Series, fwd: pd.Series,
                        candidates: list, direction: str) -> dict:
    best = {"threshold": None, "accuracy": 0.0, "n": 0}
    for thr in candidates:
        acc, n = accuracy_at_threshold(series, fwd, thr, direction)
        if n >= 5 and acc > best["accuracy"]:
            best = {"threshold": thr, "accuracy": acc, "n": n}
    return best


def analyze_indicator(df: pd.DataFrame, col: str, label: str,
                      bullish_direction: str, candidates: list,
                      horizon: int = 30) -> dict:
    """Analiza un indicador contra retornos forward."""
    sub = df[[col, f"fwd_{horizon}d"]].dropna()
    if len(sub) < 20:
        return {"label": label, "error": "datos insuficientes"}

    fwd = sub[f"fwd_{horizon}d"]
    series = sub[col]

    best = find_best_threshold(series, fwd, candidates, bullish_direction)

    # correlacion simple
    corr = series.corr(fwd)

    # base rate
    base_up = float((fwd > 0).mean())

    return {
        "label": label,
        "col": col,
        "horizon_days": horizon,
        "n_total": len(sub),
        "base_rate_up": round(base_up * 100, 1),
        "correlation": round(corr, 3),
        "best_threshold": best["threshold"],
        "best_accuracy": round(best["accuracy"] * 100, 1),
        "best_n": best["n"],
    }


# ── Comprobaciones puntuales ──────────────────────────────────────────────────

def run_checks(df: pd.DataFrame, n: int = 500, horizon: int = 14) -> list[dict]:
    """
    Selecciona N fechas ALEATORIAS del historico y comprueba si los
    indicadores habrian dado la señal correcta en el horizonte indicado.
    """
    fwd_col = f"fwd_{horizon}d"
    # solo indicadores con historia larga; funding/ls/oi son opcionales (historico corto)
    needed_core = ["ma200", "rsi", "mayer", "fg"]
    sub = df.dropna(subset=needed_core + [fwd_col]).reset_index(drop=True)

    rng = np.random.default_rng(42)
    idxs = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
    idxs.sort()
    checks = []
    for i in idxs:
        row = sub.iloc[i]
        fwd_val = row[fwd_col]
        actual = "SUBIO" if fwd_val > 0 else "BAJO"
        actual_pct = round(fwd_val * 100, 1)

        # señales con umbrales ACTUALES
        signals_bullish = 0
        signals_bearish = 0
        detail = []

        # MA200
        trend = row["close"] > row["ma200"]
        if trend:
            signals_bullish += 1
            detail.append("MA200: ALCISTA ✅")
        else:
            signals_bearish += 1
            detail.append("MA200: BAJISTA 🔴")

        # RSI — umbrales optimizados
        rsi = row["rsi"]
        if rsi <= 20:
            signals_bullish += 1
            detail.append(f"RSI {rsi:.0f}: sobrevendido extremo ✅")
        elif rsi >= 80:
            signals_bearish += 1
            detail.append(f"RSI {rsi:.0f}: sobrecomprado extremo 🔴")
        else:
            detail.append(f"RSI {rsi:.0f}: neutral ⚪")

        # Mayer — umbrales optimizados
        mayer = row["mayer"]
        if mayer < 0.80:
            signals_bullish += 1
            detail.append(f"Mayer {mayer:.2f}: muy barato ✅")
        elif mayer > 2.2:
            signals_bearish += 1
            detail.append(f"Mayer {mayer:.2f}: euforia 🔴")
        else:
            detail.append(f"Mayer {mayer:.2f}: neutro ⚪")

        # F&G — umbrales optimizados
        fg = row["fg"]
        if fg <= 15:
            signals_bullish += 1
            detail.append(f"F&G {fg:.0f}: panico extremo ✅")
        elif fg >= 80:
            signals_bearish += 1
            detail.append(f"F&G {fg:.0f}: euforia ✅")
        else:
            detail.append(f"F&G {fg:.0f}: neutro ⚪")

        # Funding
        fund = row["funding"] if not np.isnan(row["funding"]) else 0
        if fund < -0.01:
            signals_bullish += 1
            detail.append(f"Funding {fund:+.4f}%: capitulacion ✅")
        elif fund > 0.05:
            signals_bearish += 1
            detail.append(f"Funding {fund:+.4f}%: sobrecalentado 🔴")
        else:
            detail.append(f"Funding {fund:+.4f}%: neutro ⚪")

        # prediccion del bot
        if signals_bullish > signals_bearish:
            pred = "SUBE"
        elif signals_bearish > signals_bullish:
            pred = "BAJA"
        else:
            pred = "NEUTRO"

        acierto = (pred == "SUBE" and actual == "SUBIO") or (pred == "BAJA" and actual == "BAJO")

        checks.append({
            "fecha": str(row["date"].date()),
            "precio": round(row["close"], 0),
            "pred": pred,
            "actual": actual,
            "actual_pct": actual_pct,
            "acierto": acierto,
            "bullish": signals_bullish,
            "bearish": signals_bearish,
            "detalle": detail,
        })

    return checks


# ── Optimizacion de umbrales ─────────────────────────────────────────────────

def optimize_thresholds(df: pd.DataFrame) -> dict:
    print("\nOptimizando umbrales de cada indicador...")
    results = {}

    horizon = 30

    # MA200 — regimen
    df["above_ma200"] = (df["close"] > df["ma200"]).astype(int)
    r = analyze_indicator(df, "above_ma200", "MA200 (alcista=1)", "above",
                          [0.5], horizon)
    sub = df[["above_ma200", f"fwd_{horizon}d"]].dropna()
    acc_bull = float((sub[sub["above_ma200"] == 1][f"fwd_{horizon}d"] > 0).mean())
    acc_bear = float((sub[sub["above_ma200"] == 0][f"fwd_{horizon}d"] > 0).mean())
    results["ma200"] = {
        "label": "MA200",
        "precision_alcista": round(acc_bull * 100, 1),
        "precision_bajista": round((1 - acc_bear) * 100, 1),
        "umbral_actual": "precio > MA200",
        "umbral_optimo": "precio > MA200",
        "cambio": "sin cambio",
    }

    # RSI
    r = analyze_indicator(df, "rsi", "RSI", "below",
                          list(range(20, 45, 5)), horizon)
    r2 = analyze_indicator(df, "rsi", "RSI overbought", "above",
                           list(range(60, 85, 5)), horizon)
    results["rsi"] = {
        "label": "RSI",
        "umbral_sobrevendido_actual": 30,
        "umbral_sobrevendido_optimo": r["best_threshold"],
        "precision_sobrevendido": r["best_accuracy"],
        "n_sobrevendido": r["best_n"],
        "umbral_sobrecomprado_actual": 70,
        "umbral_sobrecomprado_optimo": r2["best_threshold"],
        "precision_sobrecomprado": r2["best_accuracy"],
        "n_sobrecomprado": r2["best_n"],
    }

    # Mayer
    r_low = analyze_indicator(df, "mayer", "Mayer barato", "below",
                              [0.7, 0.75, 0.8, 0.85, 0.9, 0.95], horizon)
    r_high = analyze_indicator(df, "mayer", "Mayer caro", "above",
                               [1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.4], horizon)
    results["mayer"] = {
        "label": "Mayer Multiple",
        "umbral_barato_actual": 0.85,
        "umbral_barato_optimo": r_low["best_threshold"],
        "precision_barato": r_low["best_accuracy"],
        "n_barato": r_low["best_n"],
        "umbral_caro_actual": 1.8,
        "umbral_caro_optimo": r_high["best_threshold"],
        "precision_caro": r_high["best_accuracy"],
        "n_caro": r_high["best_n"],
    }

    # Fear & Greed
    df_fg = df.dropna(subset=["fg"])
    if len(df_fg) > 30:
        r_fear = analyze_indicator(df_fg, "fg", "F&G miedo", "below",
                                   [15, 20, 25, 30, 35], horizon)
        r_greed = analyze_indicator(df_fg, "fg", "F&G codicia", "above",
                                    [65, 70, 75, 80], horizon)
        results["fear_greed"] = {
            "label": "Fear & Greed",
            "umbral_miedo_actual": 25,
            "umbral_miedo_optimo": r_fear["best_threshold"],
            "precision_miedo": r_fear["best_accuracy"],
            "n_miedo": r_fear["best_n"],
            "umbral_codicia_actual": 75,
            "umbral_codicia_optimo": r_greed["best_threshold"],
            "precision_codicia": r_greed["best_accuracy"],
            "n_codicia": r_greed["best_n"],
        }

    # Funding rate
    df_fr = df.dropna(subset=["funding"])
    if len(df_fr) > 30:
        r_neg = analyze_indicator(df_fr, "funding", "Funding negativo", "below",
                                  [-0.03, -0.02, -0.01, 0.0], horizon)
        r_pos = analyze_indicator(df_fr, "funding", "Funding positivo", "above",
                                  [0.01, 0.02, 0.03, 0.05, 0.08], horizon)
        results["funding"] = {
            "label": "Funding Rate",
            "umbral_capituacion_actual": -0.01,
            "umbral_capitulacion_optimo": r_neg["best_threshold"],
            "precision_capitulacion": r_neg["best_accuracy"],
            "n_capitulacion": r_neg["best_n"],
            "umbral_caliente_actual": 0.05,
            "umbral_caliente_optimo": r_pos["best_threshold"],
            "precision_caliente": r_pos["best_accuracy"],
            "n_caliente": r_pos["best_n"],
        }

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_checks(checks: list[dict], horizon: int = 14) -> float:
    n = len(checks)
    print("\n" + "=" * 72)
    print(f"{n} COMPROBACIONES ALEATORIAS — señal del bot vs lo que pasó en {horizon}d")
    print("=" * 72)

    aciertos = 0
    # imprimir todas pero agrupar para no saturar — mostramos resumen por año
    by_year: dict = {}
    for c in checks:
        yr = c["fecha"][:4]
        by_year.setdefault(yr, {"total": 0, "aciertos": 0, "sube": 0, "baja": 0, "neutro": 0})
        by_year[yr]["total"] += 1
        if c["acierto"]:
            by_year[yr]["aciertos"] += 1
            aciertos += 1
        by_year[yr][c["pred"].lower() if c["pred"] != "NEUTRO" else "neutro"] += 1

    # tabla por año
    print(f"\n{'AÑO':<6} {'TOTAL':>6} {'ACIERTOS':>9} {'PRECISION':>10}  SEÑALES")
    print("-" * 60)
    for yr in sorted(by_year):
        d = by_year[yr]
        pct_yr = d["aciertos"] / d["total"] * 100
        print(f"{yr:<6} {d['total']:>6} {d['aciertos']:>9} {pct_yr:>9.0f}%"
              f"  🟢{d.get('sube',0)} 🔴{d.get('baja',0)} ⚪{d.get('neutro',0)}")

    pct = aciertos / n * 100
    neutros = sum(1 for c in checks if c["pred"] == "NEUTRO")
    accionables = n - neutros
    aciertos_acc = sum(1 for c in checks if c["acierto"] and c["pred"] != "NEUTRO")

    print("\n" + "=" * 72)
    print(f"TOTAL:                {aciertos}/{n} = {pct:.1f}%")
    print(f"ACCIONABLE (sin ⚪): {aciertos_acc}/{accionables} = "
          f"{aciertos_acc/max(accionables,1)*100:.1f}%")
    print(f"Tasa base (BTC sube en {horizon}d): "
          f"{sum(1 for c in checks if c['actual']=='SUBIO')/n*100:.1f}%")

    # los 10 peores fallos para diagnostico
    fallos = [c for c in checks if not c["acierto"] and c["pred"] != "NEUTRO"]
    fallos.sort(key=lambda x: abs(x["actual_pct"]), reverse=True)
    print(f"\nPEORES FALLOS ({min(10,len(fallos))} de {len(fallos)} errores accionables):")
    for c in fallos[:10]:
        print(f"  ❌ {c['fecha']}  ${c['precio']:,.0f}  "
              f"Bot:{c['pred']:5s}  Real:{c['actual']:5s}({c['actual_pct']:+.1f}%)  "
              f"[🟢{c['bullish']} 🔴{c['bearish']}]  {' | '.join(c['detalle'])}")
    return pct


def print_optimization(opt: dict) -> None:
    print("\n" + "=" * 72)
    print("OPTIMIZACIÓN DE UMBRALES — ¿hay que cambiar algo?")
    print("=" * 72)
    changes = []
    for key, r in opt.items():
        print(f"\n📊 {r['label']}")
        if key == "ma200":
            print(f"   Régimen alcista → {r['precision_alcista']}% de las veces sube en 30d")
            print(f"   Régimen bajista → {r['precision_bajista']}% de las veces baja en 30d")
            print(f"   → Umbral sin cambio (precio vs MA200 ya es óptimo)")
        elif key == "rsi":
            sv_old, sv_new = r["umbral_sobrevendido_actual"], r["umbral_sobrevendido_optimo"]
            sc_old, sc_new = r["umbral_sobrecomprado_actual"], r["umbral_sobrecomprado_optimo"]
            print(f"   Sobrevendido:  actual={sv_old} → optimo={sv_new}  "
                  f"(precisión {r['precision_sobrevendido']}%, n={r['n_sobrevendido']})")
            print(f"   Sobrecomprado: actual={sc_old} → optimo={sc_new}  "
                  f"(precisión {r['precision_sobrecomprado']}%, n={r['n_sobrecomprado']})")
            if sv_new and sv_new != sv_old:
                changes.append(f"RSI sobrevendido: {sv_old} → {sv_new}")
            if sc_new and sc_new != sc_old:
                changes.append(f"RSI sobrecomprado: {sc_old} → {sc_new}")
        elif key == "mayer":
            b_old, b_new = r["umbral_barato_actual"], r["umbral_barato_optimo"]
            c_old, c_new = r["umbral_caro_actual"], r["umbral_caro_optimo"]
            print(f"   Barato:  actual={b_old} → optimo={b_new}  "
                  f"(precisión {r['precision_barato']}%, n={r['n_barato']})")
            print(f"   Caro:    actual={c_old} → optimo={c_new}  "
                  f"(precisión {r['precision_caro']}%, n={r['n_caro']})")
            if b_new and b_new != b_old:
                changes.append(f"Mayer barato: {b_old} → {b_new}")
            if c_new and c_new != c_old:
                changes.append(f"Mayer caro: {c_old} → {c_new}")
        elif key == "fear_greed":
            m_old, m_new = r["umbral_miedo_actual"], r["umbral_miedo_optimo"]
            g_old, g_new = r["umbral_codicia_actual"], r["umbral_codicia_optimo"]
            print(f"   Miedo:   actual={m_old} → optimo={m_new}  "
                  f"(precisión {r['precision_miedo']}%, n={r['n_miedo']})")
            print(f"   Codicia: actual={g_old} → optimo={g_new}  "
                  f"(precisión {r['precision_codicia']}%, n={r['n_codicia']})")
            if m_new and m_new != m_old:
                changes.append(f"Fear&Greed miedo: {m_old} → {m_new}")
            if g_new and g_new != g_old:
                changes.append(f"Fear&Greed codicia: {g_old} → {g_new}")
        elif key == "funding":
            print(f"   Capitulación: actual={r['umbral_capituacion_actual']} → optimo={r['umbral_capitulacion_optimo']}  "
                  f"(precisión {r['precision_capitulacion']}%, n={r['n_capitulacion']})")
            print(f"   Caliente:     actual={r['umbral_caliente_actual']} → optimo={r['umbral_caliente_optimo']}  "
                  f"(precisión {r['precision_caliente']}%, n={r['n_caliente']})")

    print("\n" + "-" * 72)
    if changes:
        print("CAMBIOS RECOMENDADOS:")
        for c in changes:
            print(f"  • {c}")
    else:
        print("Los umbrales actuales ya son óptimos o cercanos al óptimo.")
    return changes


def main():
    print("=" * 72)
    print("BACKTEST DE INDICADORES — BTC bot")
    print("=" * 72)

    df = build_dataset()

    HORIZON = 14
    N = 500
    checks = run_checks(df, n=N, horizon=HORIZON)
    accuracy = print_checks(checks, horizon=HORIZON)

    opt = optimize_thresholds(df)
    changes = print_optimization(opt)

    # guardar resultados
    out = {
        "accuracy_50_checks": round(accuracy, 1),
        "checks": checks,
        "optimization": opt,
        "recommended_changes": changes,
    }
    path = os.path.join(REPORTS_DIR, "indicator_backtest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultados guardados en {path}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
