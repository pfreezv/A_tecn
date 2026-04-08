"""
LONQ - Demo interactivo de detección de regímenes de mercado.

Cómo correr:
    python demo.py                  # Demo completo con datos sintéticos
    python demo.py --ticker SPY     # Con datos reales (requiere internet)
    python demo.py --no-plots       # Sin gráficos

Flujo:
    1. Genera / descarga datos
    2. Ensemble (K-Means + GMM + HMM) → régimen por día
    3. Backtest régimen vs Buy & Hold
    4. Walk-forward opcional (más lento)
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import matplotlib
matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LONQ regime detection demo")
    p.add_argument("--ticker", default=None,
                   help="Ticker real (ej. SPY, AAPL). Sin --ticker usa datos sintéticos.")
    p.add_argument("--start", default="2020-01-01", help="Fecha inicio (solo con --ticker)")
    p.add_argument("--no-plots", action="store_true", help="No mostrar gráficos")
    p.add_argument("--walkforward", action="store_true", help="Ejecutar walk-forward (lento)")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=5)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Datos
# ──────────────────────────────────────────────────────────────────────────────

def load_data(ticker, start):
    """Carga datos reales vía yfinance."""
    from src.data import fetch_ohlcv
    print(f"Descargando {ticker} desde {start}...")
    raw = fetch_ohlcv(ticker, start_date=start)
    print(f"  → {len(raw)} días cargados ({raw.index[0].date()} – {raw.index[-1].date()})")
    return raw, ticker


def load_synthetic():
    """Genera datos sintéticos con regímenes conocidos."""
    from src.synthetic import generate_regime_data
    print("Generando datos sintéticos (600 días, 3 regímenes bull/sideways/bear)...")
    raw, true_regimes = generate_regime_data(n_days=600, seed=42)
    print(f"  → {len(raw)} días | regímenes reales: {true_regimes.value_counts().to_dict()}")
    return raw, true_regimes


# ──────────────────────────────────────────────────────────────────────────────
# Sección 1: Ensemble
# ──────────────────────────────────────────────────────────────────────────────

def run_ensemble(raw, label, k_min, k_max):
    from src.ensemble import fit_ensemble
    print(f"\n{'='*60}")
    print("SECCIÓN 1: ENSEMBLE (K-Means + GMM + HMM)")
    print('='*60)
    result = fit_ensemble(label, raw, k_min=k_min, k_max=k_max, show_progress=True)

    print("\n--- Distribución de regímenes (consenso) ---")
    dist = result.consensus.value_counts(normalize=True).mul(100).round(1)
    for reg, pct in dist.items():
        bar = "█" * int(pct / 2)
        print(f"  {reg:10s} {pct:5.1f}%  {bar}")

    print("\n--- Fuerza de señal ---")
    ss = result.signal_strength.value_counts(normalize=True).mul(100).round(1)
    for s, pct in ss.items():
        print(f"  {s:10s} {pct:.1f}%")

    print("\n--- Retornos futuros por régimen ---")
    print(result.forward_returns.round(4).to_string())

    print("\n--- Matriz de transición HMM ---")
    print(result.transition_matrix.round(3).to_string())

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Sección 2: Backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest_section(ensemble_result):
    from src.backtest import run_backtest, run_buyhold, compare_strategies

    print(f"\n{'='*60}")
    print("SECCIÓN 2: BACKTEST régimen vs Buy & Hold")
    print('='*60)

    prices = ensemble_result.df["Close"]
    regimes = ensemble_result.consensus

    strat = run_backtest(prices, regimes, strategy_name="Regime Ensemble")
    bh    = run_buyhold(prices)

    table = compare_strategies([strat, bh])
    print(table.to_string())

    return strat, bh


# ──────────────────────────────────────────────────────────────────────────────
# Sección 3: Walk-forward (opcional)
# ──────────────────────────────────────────────────────────────────────────────

def run_walkforward_section(raw, k):
    from src.features import build_features, get_feature_matrix, FEATURE_COLS
    from src.walkforward import walk_forward

    print(f"\n{'='*60}")
    print("SECCIÓN 3: WALK-FORWARD (re-entrenamiento periódico)")
    print('='*60)

    df = build_features(raw)
    df_model = get_feature_matrix(df)

    print(f"Corriendo walk-forward sobre {len(df_model)} muestras...")
    wf = walk_forward(df_model, FEATURE_COLS, k_kmeans=k, k_gmm=k, n_hmm=k)

    print(f"  → {len(wf)} días etiquetados")
    print("\nDistribución consenso walk-forward:")
    dist = wf["Consensus"].value_counts(normalize=True).mul(100).round(1)
    for reg, pct in dist.items():
        print(f"  {reg:10s} {pct:.1f}%")

    print("\nFuerza de señal walk-forward:")
    ss = wf["SignalStrength"].value_counts(normalize=True).mul(100).round(1)
    for s, pct in ss.items():
        print(f"  {s:10s} {pct:.1f}%")

    return wf


# ──────────────────────────────────────────────────────────────────────────────
# Gráficos
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(ensemble_result, strat, bh):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"LONQ – {ensemble_result.ticker}", fontsize=14, fontweight="bold")

    df = ensemble_result.df
    colors = {"bull": "#2ecc71", "sideways": "#f39c12", "bear": "#e74c3c", "unknown": "#95a5a6"}

    # 1. Precio con regímenes
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#2c3e50", linewidth=1, label="Close")
    prev_date = df.index[0]
    prev_regime = ensemble_result.consensus.iloc[0]
    for date, regime in zip(df.index[1:], ensemble_result.consensus.iloc[1:]):
        if regime != prev_regime or date == df.index[-1]:
            ax1.axvspan(prev_date, date, alpha=0.2, color=colors.get(prev_regime, "gray"), label=f"_{prev_regime}")
            prev_date = date
            prev_regime = regime
    # Legend manual
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.4, label=r) for r, c in colors.items() if r != "unknown"]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax1.set_title("Precio con regímenes detectados")
    ax1.set_ylabel("Precio")

    # 2. Equity curves
    ax2 = axes[1]
    ax2.plot(strat.equity_curve.index, strat.equity_curve, label="Regime Strategy", color="#3498db", linewidth=1.5)
    ax2.plot(bh.equity_curve.index, bh.equity_curve, label="Buy & Hold", color="#e74c3c", linewidth=1.5, linestyle="--")
    ax2.set_title("Curva de capital")
    ax2.set_ylabel("Valor (base 1.0)")
    ax2.legend()
    ax2.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")

    # 3. Confianza del ensemble
    ax3 = axes[2]
    conf = ensemble_result.confidence
    ax3.fill_between(conf.index, conf, alpha=0.5, color="#9b59b6")
    ax3.set_title("Confianza del ensemble (fracción de modelos en acuerdo)")
    ax3.set_ylabel("Confianza")
    ax3.set_ylim(0, 1)
    ax3.axhline(0.67, color="orange", linewidth=1, linestyle="--", label="Moderado")
    ax3.axhline(1.0, color="green", linewidth=1, linestyle="--", label="Fuerte")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("lonq_demo_output.png", dpi=120, bbox_inches="tight")
    print("\nGráfico guardado en: lonq_demo_output.png")
    try:
        plt.show()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  LONQ — Sistema de detección de regímenes de mercado")
    print("=" * 60)

    # Datos
    if args.ticker:
        raw, label = load_data(args.ticker, args.start)
        true_regimes = None
    else:
        raw, true_regimes = load_synthetic()
        label = "SYNTHETIC"

    # Ensemble
    ensemble_result = run_ensemble(raw, label, args.k_min, args.k_max)

    # Validación si tenemos verdad de tierra (datos sintéticos)
    if true_regimes is not None:
        regime_map = {0: "bull", 1: "sideways", 2: "bear"}
        true_sem = true_regimes.map(regime_map)
        common = ensemble_result.consensus.index.intersection(true_sem.index)
        if len(common) > 0:
            accuracy = (ensemble_result.consensus.loc[common] == true_sem.loc[common]).mean()
            print(f"\n[Validación] Precisión vs regímenes reales: {accuracy:.1%}")

    # Backtest
    strat, bh = run_backtest_section(ensemble_result)

    # Walk-forward (opcional)
    if args.walkforward:
        run_walkforward_section(raw, k=3)

    # Gráficos
    if not args.no_plots:
        plot_results(ensemble_result, strat, bh)

    print("\nDemo completado.")


if __name__ == "__main__":
    main()
