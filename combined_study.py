"""
LONQ — Combined Signal Study: Trigger + Régimen + VIX

Ejecuta el sistema de señales combinado en versión Base y +VIX
y muestra comparativa lado a lado.

Uso:
    python combined_study.py --csv ko_data.csv
    python combined_study.py --csv tsla_data.csv --vix vix_data.csv
    python combined_study.py --csv spy_data.csv  --vix vix_data.csv --threshold 1.5
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="LONQ Combined Signal Study")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv",    help="CSV local con columnas Date,Open,High,Low,Close,Volume")
    g.add_argument("--ticker", help="Ticker real (requiere internet + yfinance)")
    p.add_argument("--vix",        default=None, help="CSV local del VIX (columnas Date,Close)")
    p.add_argument("--start",      default="2018-01-01")
    p.add_argument("--threshold",  type=float, default=2.0)
    p.add_argument("--horizon",    type=int,   default=5,
                   help="Días adelante para precio objetivo (default 5)")
    p.add_argument("--min-score",  type=int,   default=2,
                   help="Puntuación mínima para emitir señal (default 2)")
    p.add_argument("--no-ensemble", action="store_true",
                   help="Omitir el ensemble (más rápido, sin contexto de régimen)")
    return p.parse_args()


def load_data(args):
    if args.csv:
        raw    = pd.read_csv(args.csv, index_col="Date", parse_dates=True)
        ticker = os.path.splitext(os.path.basename(args.csv))[0].upper()
    else:
        from src.data import fetch_ohlcv
        raw    = fetch_ohlcv(args.ticker, start_date=args.start)
        ticker = args.ticker.upper()

    prices = raw["Close"]

    vix = None
    if args.vix:
        from src.vix import load_vix
        vix = load_vix(args.vix)

    return raw, prices, ticker, vix


def run_ensemble(raw, ticker):
    from src.ensemble import fit_ensemble
    print(f"  Entrenando ensemble (KMeans + GMM + HMM)...")
    result = fit_ensemble(ticker, raw, show_progress=False)
    dist = result.consensus.value_counts(normalize=True).mul(100).round(1)
    print(f"  Distribución: {dict(dist)}")
    return result


def print_comparison(base_result, vix_result):
    """Tabla comparativa Base vs +VIX."""
    print("\n" + "═"*72)
    print("  COMPARATIVA BASE vs +VIX")
    print("═"*72)

    rows = []
    for r in [base_result, vix_result]:
        if r is None:
            continue
        s = r.summary
        rows.append({
            "Versión":             s.get("Versión", "—"),
            "Total anomalías":     s.get("Total anomalías", 0),
            "TAKE_PROFIT":         s.get("Señales TAKE_PROFIT", 0),
            "BUY_DIP":             s.get("Señales BUY_DIP", 0),
            "Señales activas":     s.get("Señales activas", 0),
            "% con señal":         s.get("% con señal", "—"),
            "Win rate +5d":        s.get("Win rate +5d", "—"),
            "Win rate +10d":       s.get("Win rate +10d", "—"),
        })

    if rows:
        df = pd.DataFrame(rows).set_index("Versión")
        print(df.to_string())

    # Señales activas recientes (últimas 10)
    for r in [base_result, vix_result]:
        if r is None or r.active_signals.empty:
            continue
        recent = r.active_signals.tail(10)
        print(f"\n  Últimas señales [{r.version}]:")
        cols = ["Fecha","Señal","Puntos","Confianza","Retorno hoy","Z-score",
                "P(reversión)","Precio","Target","Stop Loss","Potencial","Régimen","VIX"]
        print(recent[[c for c in cols if c in recent.columns]].to_string(index=False))

    # Señales de alta confianza en ambas versiones
    print("\n" + "─"*72)
    print("  SEÑALES DE ALTA CONFIANZA (score ≥ 4/5)")
    print("─"*72)
    found_any = False
    for r in [base_result, vix_result]:
        if r is None or r.active_signals.empty:
            continue
        hc = r.active_signals[r.active_signals["Confianza"] == "Alta"]
        if hc.empty:
            print(f"\n  [{r.version}] Sin señales de alta confianza.")
            continue
        found_any = True
        print(f"\n  [{r.version}] {len(hc)} señal(es) de alta confianza:")
        for _, row in hc.iterrows():
            print(f"\n    [{row['Fecha']}] {row['Señal']}  score {row['Puntos']}/5")
            print(f"      Precio: ${float(row['Precio']):.2f}  →  Target: ${float(row['Target']):.2f}  |  Stop: ${float(row['Stop Loss']):.2f}")
            print(f"      Movimiento: {row['Retorno hoy']}  ({row['Z-score']})  |  P(rev): {row['P(reversión)']}")
            print(f"      Razones: {row['Razones']}")


def main():
    args = parse_args()

    print("=" * 72)
    print("  LONQ — COMBINED SIGNAL STUDY")
    print("=" * 72)

    # Cargar datos
    print("\nCargando datos...")
    raw, prices, ticker, vix = load_data(args)
    print(f"  Activo:  {ticker}")
    print(f"  Período: {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} días)")
    if vix is not None:
        common = prices.index.intersection(vix.index)
        print(f"  VIX:     {len(common)} días en común")

    # Ensemble (opcional)
    ensemble = None
    if not args.no_ensemble:
        print("\nEntrenando ensemble...")
        try:
            ensemble = run_ensemble(raw, ticker)
        except Exception as e:
            print(f"  Ensemble falló ({e}), continuando sin contexto de régimen.")

    # ── Versión BASE ──────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  VERSIÓN BASE (sin VIX)")
    print(f"{'─'*72}")
    from src.combined_signal import run_combined_signal, print_combined_report

    base = run_combined_signal(
        prices, ticker,
        ensemble_result=ensemble,
        vix_series=None,
        threshold=args.threshold,
        target_horizon=args.horizon,
        min_score=args.min_score,
    )
    print_combined_report(base)

    # ── Versión +VIX ──────────────────────────────────────────────────────────
    vix_result = None
    if vix is not None:
        print(f"\n{'─'*72}")
        print("  VERSIÓN +VIX")
        print(f"{'─'*72}")
        vix_result = run_combined_signal(
            prices, ticker,
            ensemble_result=ensemble,
            vix_series=vix,
            threshold=args.threshold,
            target_horizon=args.horizon,
            min_score=args.min_score,
        )
        print_combined_report(vix_result)

    # ── Comparativa ───────────────────────────────────────────────────────────
    if vix_result is not None:
        print_comparison(base, vix_result)
    else:
        print("\n  (Pasa --vix vix_data.csv para comparar con la versión +VIX)")


if __name__ == "__main__":
    main()
