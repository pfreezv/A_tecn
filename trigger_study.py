"""
LONQ — Trigger Study: análisis de anomalías y reversión a la media.

Uso:
    python trigger_study.py --csv tsla_data.csv
    python trigger_study.py --csv spy_data.csv  --threshold 1.5
    python trigger_study.py --ticker AAPL       --start 2018-01-01
    python trigger_study.py --csv ko_data.csv   --tf daily weekly
    python trigger_study.py --csv tsla_data.csv --summary-only

Marcos temporales disponibles (--tf):
    daily      → retorno 1 día
    weekly     → retorno 5 días
    biweekly   → retorno 10 días
    monthly    → retorno 21 días
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

TF_MAP = {
    "daily":    ("Diario (1d)",    1),
    "weekly":   ("Semanal (5d)",   5),
    "biweekly": ("Quincenal (10d)", 10),
    "monthly":  ("Mensual (21d)",  21),
}


def parse_args():
    p = argparse.ArgumentParser(description="LONQ Trigger Study — Anomaly & Mean Reversion")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv",    help="CSV local con columnas Date,Open,High,Low,Close,Volume")
    g.add_argument("--ticker", help="Ticker real (requiere internet + yfinance)")
    p.add_argument("--start",     default="2018-01-01", help="Fecha inicio (solo con --ticker)")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="Z-score umbral para anomalía (default 2.0 ≈ 95%% de días normales)")
    p.add_argument("--window",    type=int, default=252,
                   help="Ventana rodante en días (default 252 = 1 año)")
    p.add_argument("--tf", nargs="+",
                   choices=list(TF_MAP.keys()),
                   default=list(TF_MAP.keys()),
                   help="Marcos temporales a analizar (default: todos)")
    p.add_argument("--summary-only", action="store_true",
                   help="Solo tabla resumen, sin detalle por timeframe")
    p.add_argument("--no-verbose", action="store_true",
                   help="Sin estadísticas anuales detalladas")
    return p.parse_args()


def load_prices(args) -> tuple[pd.Series, str]:
    if args.csv:
        raw = pd.read_csv(args.csv, index_col="Date", parse_dates=True)
        prices = raw["Close"]
        ticker = os.path.splitext(os.path.basename(args.csv))[0].upper()
    else:
        from src.data import fetch_ohlcv
        raw    = fetch_ohlcv(args.ticker, start_date=args.start)
        prices = raw["Close"]
        ticker = args.ticker.upper()

    print(f"\nActivo:  {ticker}")
    print(f"Período: {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} días)")
    print(f"Precio:  min=${prices.min():.2f}  max=${prices.max():.2f}  último=${prices.iloc[-1]:.2f}")
    vol_ann = prices.pct_change().std() * np.sqrt(252)
    print(f"Vol anual estimada: {vol_ann:.1%}")
    return prices, ticker


def print_quick_profile(result, args):
    """Imprime un resumen rápido del perfil del ticker."""
    print("\n" + "═"*72)
    print(f"  PERFIL DE NORMALIDAD — {result.ticker}")
    print(f"  (umbral anomalía: ±{result.threshold}σ  |  ventana: {result.window} días)")
    print("═"*72)

    for tf_name, tfr in result.timeframes.items():
        df  = tfr.df
        mu  = df["ret"].mean()
        sig = df["ret"].std()
        n_h = (df["event_type"] == "ANOMALY_HIGH").sum()
        n_l = (df["event_type"] == "ANOMALY_LOW").sum()
        pct = (n_h + n_l) / len(df)

        bar_lo = "▓" * max(1, int(abs(tfr.normal_range[0]) * 200))
        bar_hi = "▓" * max(1, int(tfr.normal_range[1] * 200))

        print(f"\n  {tf_name}")
        print(f"    Rango normal:  [{tfr.normal_range[0]:+.3%}  ──────────── {tfr.normal_range[1]:+.3%}]")
        print(f"    Media:         {mu:+.4%}  |  Std: {sig:.4%}")
        print(f"    Anomalías ↑:   {n_h} eventos ({n_h/len(df):.1%} de los días)")
        print(f"    Anomalías ↓:   {n_l} eventos ({n_l/len(df):.1%} de los días)")
        print(f"    Total anómalos:{pct:.1%} de los días  ({'esperado ~5%' if abs(pct-0.05)<0.03 else 'inusual'})")

        # Evento más extremo
        if not tfr.anomalies.empty:
            idx_max = tfr.anomalies["Z-score"].abs().idxmax()
            row = tfr.anomalies.loc[idx_max]
            print(f"    Evento más extremo: {idx_max.date()}  {row['Retorno']:+.3%}  ({row['Z-score']:+.2f}σ  {row['Tipo']})")


def print_reversion_summary(result):
    """Tabla resumen de reversión cross-timeframe."""
    print("\n" + "═"*72)
    print("  RESUMEN DE REVERSIÓN — ¿Se confirma la hipótesis?")
    print("═"*72)
    print(f"\n  Hipótesis: tras un movimiento anómalo, el precio revierte hacia la media")
    print(f"  Criterio:  retorno post-evento significativamente distinto a días normales\n")

    header = f"  {'Timeframe':<18} {'Subida anómala → baja?':<28} {'Bajada anómala → sube?':<28} {'Score'}"
    print(header)
    print("  " + "─"*70)

    for tf_name, tfr in result.timeframes.items():
        if tfr.ttest.empty:
            print(f"  {tf_name:<18} {'N/A':<28} {'N/A':<28} —")
            continue

        high_tests = tfr.ttest[tfr.ttest["Tipo"] == "ANOMALY_HIGH"]
        low_tests  = tfr.ttest[tfr.ttest["Tipo"] == "ANOMALY_LOW"]

        def _summarize(tests, direction):
            if tests.empty:
                return "sin datos"
            sig = tests[tests["Sig. (p<0.05)"] == "✓"]
            rev = sig[sig["Diferencia"] < 0] if direction == "high" else sig[sig["Diferencia"] > 0]
            if len(rev) == 0:
                return f"✗ no revierte ({len(sig)}/{len(tests)} sig.)"
            best = rev.loc[rev["Diferencia"].abs().idxmax()]
            return f"✓ revierte ({len(rev)}/{len(tests)} sig.) mejor:{best['Horizonte']}"

        high_str = _summarize(high_tests, "high")
        low_str  = _summarize(low_tests,  "low")

        # Score
        h_rev = len(high_tests[(high_tests["Sig. (p<0.05)"]=="✓") & (high_tests["Diferencia"]<0)])
        l_rev = len(low_tests[ (low_tests[ "Sig. (p<0.05)"]=="✓") & (low_tests[ "Diferencia"]>0)])
        score = h_rev + l_rev

        print(f"  {tf_name:<18} {high_str:<28} {low_str:<28} {score}/8")

    print(f"\n  {result.verdict}")


def main():
    args = parse_args()

    print("=" * 72)
    print("  LONQ — TRIGGER STUDY: Anomalías y Reversión a la Media")
    print("=" * 72)

    # Cargar precios
    prices, ticker = load_prices(args)

    # Construir dict de timeframes seleccionados
    selected_tfs = {TF_MAP[k][0]: TF_MAP[k][1] for k in args.tf}

    # Ejecutar análisis multi-timeframe
    from src.trigger import run_multitf_trigger, print_multitf_report

    print(f"\nAnalizando {len(selected_tfs)} marcos temporales con umbral ±{args.threshold}σ...")
    result = run_multitf_trigger(
        prices, ticker,
        timeframes=selected_tfs,
        window=args.window,
        threshold=args.threshold,
    )

    if args.summary_only:
        print_quick_profile(result, args)
        print_reversion_summary(result)
    else:
        print_quick_profile(result, args)
        print_reversion_summary(result)
        print_multitf_report(result, verbose=not args.no_verbose)


if __name__ == "__main__":
    main()
