"""
Busqueda de PATRONES REPETIDOS (motif search) en velas de 1h.

Lo que pide el usuario: barrer un rango amplio de longitudes de ventana (8h, 20h,
50h, 100h, 200h), encontrar los patrones del historico que mas se parecen al
patron ACTUAL (margen de error pequeno = similitud alta) y dar ejemplos concretos
con su prediccion (arriba/abajo). Sin sesgo "bajista": prediccion de direccion.

Dos partes:
  A) ANALOGOS DE AHORA: por cada longitud, los patrones historicos mas parecidos
     al de ahora, con fecha, similitud y lo que paso despues. -> ejemplos.
  B) PRUEBA DECISIVA: cuando dos patrones son CASI IDENTICOS (similitud >97%),
     ¿van en la misma direccion despues? Si es ~50%, el parecido NO predice.

similitud = correlacion de la FORMA (secuencia de % normalizada). 100% = identica.

Uso:
    python src/motif_search.py --horizon 48
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WINDOWS_H = [8, 20, 50, 100, 200]   # longitudes de patron en horas


def load_1h() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA_DIR, "btc_usdt_1h_binance.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def shapes_fwd(close: np.ndarray, L: int, H: int):
    """Formas normalizadas (z-score de los % de cada ventana) + retorno H velas despues."""
    rets = np.diff(np.log(close))
    W = sliding_window_view(rets, L).astype(float)         # (M x L)
    mu = W.mean(1, keepdims=True)
    sd = W.std(1, keepdims=True); sd[sd == 0] = 1
    Z = (W - mu) / sd                                      # forma (shape)
    end = np.arange(L, len(close))                        # cierre de cada ventana
    fwd = np.full(len(close), np.nan)
    fwd[:-H] = close[H:] / close[:-H] - 1.0
    return Z, end, fwd[end]


def part_a(df, H, topk=20):
    close = df["close"].values
    dates = df["date"]
    print("=" * 78)
    print(f"A) PATRONES MAS PARECIDOS AL DE AHORA  (horizonte futuro: {H}h)")
    print("=" * 78)
    agg_up = agg_dn = 0
    for L in WINDOWS_H:
        Z, end, fwd = shapes_fwd(close, L, H)
        zq = Z[-1]                                         # forma del patron ACTUAL
        # candidatos: con desenlace conocido y sin solapar con el patron actual
        valid = (~np.isnan(fwd)) & (np.arange(len(Z)) < len(Z) - L)
        corr = (Z[valid] @ zq) / L                         # similitud de forma [-1,1]
        vi = np.where(valid)[0]
        order = vi[np.argsort(corr)[::-1]]                 # mas parecidos primero
        top = order[:topk]
        sims = (Z[top] @ zq) / L
        f = fwd[top]
        up = int(np.sum(f > 0)); dn = int(np.sum(f < 0))
        agg_up += up; agg_dn += dn
        print(f"\nVentana {L}h  |  similitud media top-{topk}: {sims.mean()*100:.0f}%")
        print(f"  de los {topk} mas parecidos:  SUBIO {up}  /  BAJO {dn}  "
              f"-> {100*up/topk:.0f}% arriba   (mov. medio {f.mean()*100:+.2f}%)")
        # 3 ejemplos concretos (los mas parecidos)
        for j in top[:3]:
            d = dates.iloc[int(end[j])].date()
            s = (Z[j] @ zq) / L
            print(f"     ej: {d}  similitud {s*100:.0f}%  ->  {fwd[j]*100:+.1f}% en {H}h")
    tot = agg_up + agg_dn
    print("\n" + "-" * 78)
    print(f"PREDICCION AGREGADA (todas las longitudes):  "
          f"{100*agg_up/tot:.0f}% ARRIBA  /  {100*agg_dn/tot:.0f}% ABAJO")
    return 100 * agg_up / tot


def part_b(df, H, L=8, sim_thr=0.97, n_anchor=800, gap_days=30):
    """¿Patrones casi identicos van en la misma direccion despues?"""
    close = df["close"].values
    Z, end, fwd = shapes_fwd(close, L, H)
    M = len(Z)
    valid = ~np.isnan(fwd)
    gap = gap_days * 24
    rng = np.linspace(L, M - H - 1, n_anchor).astype(int)   # anclas repartidas (deterministico)
    pairs = agree = 0
    examples = []
    for a in rng:
        if not valid[a]:
            continue
        corr = (Z[valid] @ Z[a]) / L
        vi = np.where(valid)[0]
        # excluir el propio entorno temporal
        far = np.abs(vi - a) > gap
        c = corr.copy(); c[~far] = -2
        b = vi[int(np.argmax(c))]
        if (Z[b] @ Z[a]) / L < sim_thr:
            continue
        pairs += 1
        same = np.sign(fwd[a]) == np.sign(fwd[b])
        agree += int(same)
        if len(examples) < 4:
            examples.append((a, b, (Z[b] @ Z[a]) / L, fwd[a], fwd[b], same))
    print("\n" + "=" * 78)
    print(f"B) PRUEBA DECISIVA: patrones CASI IDENTICOS (similitud >{sim_thr*100:.0f}%, ventana {L}h)")
    print(f"   ¿van en la MISMA direccion {H}h despues?")
    print("=" * 78)
    if pairs == 0:
        print("  (no se hallaron pares suficientemente identicos)")
        return
    dts = df["date"]
    for a, b, s, fa, fb, same in examples:
        da = dts.iloc[int(end[a])].date(); dbb = dts.iloc[int(end[b])].date()
        mark = "IGUAL" if same else "OPUESTO"
        print(f"  {da} ({fa*100:+.1f}%)  vs  {dbb} ({fb*100:+.1f}%)  "
              f"sim {s*100:.0f}%  -> {mark}")
    print("-" * 78)
    print(f"  pares casi identicos: {pairs}   |   coinciden en direccion: "
          f"{100*agree/pairs:.0f}%")
    if abs(100 * agree / pairs - 50) <= 5:
        print("  -> ~50% = AZAR. Que dos patrones sean identicos NO dice nada de la direccion.")
    else:
        print("  -> distinto de 50%: habria algo. Validar con cuidado (puede ser sesgo de muestra).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48, help="horizonte futuro en horas")
    args = ap.parse_args()
    df = load_1h()
    print(f"Datos 1h: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} ({len(df)} velas)")
    part_a(df, args.horizon)
    part_b(df, args.horizon)
    print("=" * 78)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
