"""
Fase 0 (táctica) - Busqueda HONESTA de patrones en 4h / 1h.

Dos enfoques complementarios, ambos con la misma disciplina anti-autoengaño
(division dentro/fuera de muestra, comisiones, SIN mirar el futuro):

  1) ANALOGOS HISTORICOS  (la idea literal del usuario)
     "De las N situaciones pasadas mas parecidas a la forma de la curva de AHORA,
      ¿cuantas veces subio BTC despues y cuanto de media?" -> decision por %.
     - snapshot():  foto del momento actual.
     - walkforward(): mide si ese predictor ACIERTA fuera de muestra (hit rate),
       usando solo el pasado en cada punto (sin lookahead) y restando comisiones.

  2) PATRONES DE VELAS clasicos (engulfing, martillo, estrella fugaz, doji...)
     Mide el retorno medio H velas despues de cada patron, separando la primera
     mitad (in-sample) de la segunda (out-of-sample). Si el edge desaparece fuera
     de muestra, el patron no sirve.

Uso:
    python src/patterns.py --tf 4h
    python src/patterns.py --tf 1h --horizon 24
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FEE_ROUNDTRIP = 0.0052  # 0.26% entrada + 0.26% salida


def load(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, f"btc_usdt_{tf}_binance.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  1) ANALOGOS HISTORICOS
# --------------------------------------------------------------------------- #
def build_shapes(close: np.ndarray, L: int) -> np.ndarray:
    """Matriz (N x L) de ventanas normalizadas en FORMA (z-score por ventana)."""
    wins = sliding_window_view(close, L)            # (N x L), ventana i acaba en i+L-1
    mean = wins.mean(axis=1, keepdims=True)
    std = wins.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (wins - mean) / std                       # solo importa la forma, no el precio


def analog_forward(close: np.ndarray, L: int, H: int):
    """Devuelve (shapes, end_idx, fwd_ret) alineados."""
    shapes = build_shapes(close, L)
    end_idx = np.arange(L - 1, len(close))           # indice de cierre de cada ventana
    fwd = np.full(len(close), np.nan)
    fwd[:-H] = close[H:] / close[:-H] - 1.0          # retorno H velas despues
    return shapes, end_idx, fwd[end_idx]


def snapshot(df: pd.DataFrame, L: int, H: int, topn: int) -> None:
    close = df["close"].values
    shapes, end_idx, fwd = analog_forward(close, L, H)
    q = shapes[-1]                                   # forma de la ventana ACTUAL
    # candidatos: ventanas cuyo desenlace ya es conocido y NO solapan con la actual
    valid = (end_idx + H < len(close) - 1) & (end_idx < end_idx[-1] - L)
    sims = (shapes[valid] @ q) / L                   # correlacion de forma (~Pearson)
    cand_fwd = fwd[valid]
    order = np.argsort(sims)[::-1][:topn]
    best_fwd = cand_fwd[order]
    pct_up = 100.0 * np.mean(best_fwd > 0)
    print("-" * 70)
    print(f"ANALOGOS HISTORICOS  (forma de {L} velas, mirando {H} velas adelante)")
    print(f"De las {topn} situaciones pasadas mas parecidas a la de AHORA:")
    print(f"  subio despues:     {pct_up:.0f}% de las veces")
    print(f"  movimiento medio:  {np.mean(best_fwd)*100:+.2f}%   mediana: {np.median(best_fwd)*100:+.2f}%")
    print(f"  mejor / peor:      {best_fwd.max()*100:+.1f}% / {best_fwd.min()*100:+.1f}%")
    print(f"  similitud media:   {sims[order].mean():.2f}  (1.0 = forma idéntica)")
    edge = abs(np.mean(best_fwd))
    if edge < FEE_ROUNDTRIP:
        print(f"  -> El movimiento medio ({np.mean(best_fwd)*100:+.2f}%) NO cubre comisiones "
              f"({FEE_ROUNDTRIP*100:.2f}%): señal demasiado débil para operar.")


def walkforward(df: pd.DataFrame, L: int, H: int, topn: int, test_frac: float = 0.3) -> None:
    """Mide el acierto del predictor de analogos FUERA de muestra, sin lookahead."""
    close = df["close"].values
    shapes, end_idx, fwd = analog_forward(close, L, H)
    n = len(end_idx)
    test_start = int(n * (1 - test_frac))
    stride = max(1, H // 2)                           # no solapar trades consecutivos

    hits, rets, preds = 0, [], 0
    for t in range(test_start, n - H - 1, stride):
        # candidatos: solo ventanas del PASADO con desenlace ya conocido en el pasado
        mask = (end_idx <= end_idx[t] - L) & (end_idx + H < end_idx[t])
        if mask.sum() < topn:
            continue
        sims = (shapes[mask] @ shapes[t]) / L
        idx = np.argsort(sims)[::-1][:topn]
        pred_mean = np.mean(fwd[mask][idx])
        if abs(pred_mean) < 1e-9:
            continue
        actual = fwd[t]
        if np.isnan(actual):
            continue
        preds += 1
        direction = np.sign(pred_mean)
        if direction == np.sign(actual):
            hits += 1
        # retorno si operamos en la direccion predicha, restando comisiones
        rets.append(direction * actual - FEE_ROUNDTRIP)

    if preds == 0:
        print("  (no hubo suficientes analogos para evaluar)")
        return
    rets = np.array(rets)
    hit_rate = 100.0 * hits / preds
    print("-" * 70)
    print(f"VALIDACION FUERA DE MUESTRA (ultimo {int(test_frac*100)}% de los datos, sin lookahead)")
    print(f"  señales evaluadas:   {preds}")
    print(f"  acierto direccion:   {hit_rate:.1f}%   (50% = azar; descontando comisiones)")
    print(f"  retorno medio/op:    {rets.mean()*100:+.2f}%   tras comisiones")
    print(f"  expectativa total:   {(np.prod(1+rets)-1)*100:+.0f}%   acumulado en el test")
    if hit_rate <= 53 or rets.mean() <= 0:
        print("  -> Veredicto: SIN edge fiable. El emparejamiento de forma por si solo no")
        print("     predice mejor que el azar una vez restas comisiones. (Esperable.)")
    else:
        print("  -> Veredicto: hay indicios de edge. Merece optimizacion walk-forward seria.")


# --------------------------------------------------------------------------- #
#  2) PATRONES DE VELAS CLASICOS
# --------------------------------------------------------------------------- #
def detect_patterns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    body = np.abs(c - o)
    rng = (h - l)
    rng[rng == 0] = 1e-9
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    up = c > o
    prev_up = np.r_[False, up[:-1]]
    prev_o = np.r_[o[0], o[:-1]]
    prev_c = np.r_[c[0], c[:-1]]

    pats = {
        "alcista_envolvente": up & ~prev_up & (c >= prev_o) & (o <= prev_c),
        "bajista_envolvente": ~up & prev_up & (o >= prev_c) & (c <= prev_o),
        "martillo": (lower > 2 * body) & (upper < body) & (body / rng < 0.35),
        "estrella_fugaz": (upper > 2 * body) & (lower < body) & (body / rng < 0.35),
        "doji": (body / rng < 0.1),
        "vela_alcista_fuerte": up & (body / rng > 0.7),
        "vela_bajista_fuerte": ~up & (body / rng > 0.7),
    }
    return pats


def candlestick_edge(df: pd.DataFrame, H: int) -> None:
    close = df["close"].values
    fwd = np.full(len(close), np.nan)
    fwd[:-H] = close[H:] / close[:-H] - 1.0
    pats = detect_patterns(df)
    split = len(df) // 2
    print("-" * 70)
    print(f"PATRONES DE VELAS  (retorno medio {H} velas despues; in-sample vs out-of-sample)")
    print(f"{'patron':22} {'n':>6} {'in-sample':>11} {'out-sample':>11}  veredicto")
    for name, sig in pats.items():
        idx = np.where(sig & ~np.isnan(fwd))[0]
        if len(idx) < 30:
            continue
        ins = fwd[idx[idx < split]]
        out = fwd[idx[idx >= split]]
        if len(ins) < 10 or len(out) < 10:
            continue
        mi, mo = ins.mean() * 100, out.mean() * 100
        # un patron sirve si mantiene el signo y bate comisiones fuera de muestra
        ok = (np.sign(mi) == np.sign(mo)) and (abs(mo) > FEE_ROUNDTRIP * 100)
        verd = "consistente" if ok else "no generaliza"
        print(f"{name:22} {len(idx):>6} {mi:>+10.2f}% {mo:>+10.2f}%  {verd}")
    print("(Nota: 'consistente' != rentable. Es un filtro minimo, no una garantia.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h", choices=["1h", "4h", "1d"])
    ap.add_argument("--window", type=int, default=24, help="velas que forman la 'forma'")
    ap.add_argument("--horizon", type=int, default=12, help="velas a futuro a evaluar")
    ap.add_argument("--topn", type=int, default=50, help="nº de analogos mas parecidos")
    args = ap.parse_args()

    df = load(args.tf)
    print("=" * 70)
    print(f"BUSQUEDA DE PATRONES  tf={args.tf}  ventana={args.window}  horizonte={args.horizon}")
    print(f"datos: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}  ({len(df)} velas)")
    print("=" * 70)
    snapshot(df, args.window, args.horizon, args.topn)
    walkforward(df, args.window, args.horizon, args.topn)
    candlestick_edge(df, args.horizon)
    print("=" * 70)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
