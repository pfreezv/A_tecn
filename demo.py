"""
LONQ - Demo interactivo de detección de regímenes de mercado.

Cómo correr:
    python demo.py                        # Demo con datos sintéticos
    python demo.py --ticker SPY           # Con datos reales (requiere internet)
    python demo.py --csv ko_data.csv      # Con CSV local (sin internet)
    python demo.py --no-plots             # Sin gráficos
    python demo.py --walkforward          # + walk-forward (lento)

Flujo:
    1. Genera / descarga / carga datos
    2. Ensemble (K-Means + GMM + HMM) → régimen por día
    3. Señales de compra y venta (BUY / SELL / HOLD)
    4. Backtest régimen vs Buy & Hold
    5. Walk-forward opcional
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
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ticker", default=None,
                   help="Ticker real (ej. SPY, AAPL, KO). Requiere internet + yfinance.")
    g.add_argument("--csv", default=None,
                   help="Ruta a CSV local con columnas Date,Open,High,Low,Close,Volume")
    p.add_argument("--start", default="2020-01-01", help="Fecha inicio (solo con --ticker)")
    p.add_argument("--no-plots", action="store_true", help="No mostrar gráficos")
    p.add_argument("--walkforward", action="store_true", help="Ejecutar walk-forward (lento)")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=5)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Datos
# ──────────────────────────────────────────────────────────────────────────────

def load_from_ticker(ticker, start):
    """Descarga datos reales vía yfinance."""
    from src.data import fetch_ohlcv
    print(f"Descargando {ticker} desde {start} (vía yfinance)...")
    raw = fetch_ohlcv(ticker, start_date=start)
    print(f"  → {len(raw)} días  ({raw.index[0].date()} – {raw.index[-1].date()})")
    return raw, ticker.upper()


def load_from_csv(path):
    """Carga OHLCV desde un CSV local."""
    print(f"Cargando datos desde {path}...")
    raw = pd.read_csv(path, index_col="Date", parse_dates=True)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en CSV: {missing}")
    label = os.path.splitext(os.path.basename(path))[0].upper()
    print(f"  → {len(raw)} días  ({raw.index[0].date()} – {raw.index[-1].date()})")
    return raw, label


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
    for s, pct in result.signal_strength.value_counts(normalize=True).mul(100).round(1).items():
        print(f"  {s:10s} {pct:.1f}%")

    print("\n--- Retornos futuros por régimen ---")
    print(result.forward_returns.round(4).to_string())

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Sección 2: Señales BUY / SELL / HOLD
# ──────────────────────────────────────────────────────────────────────────────

def run_signals_section(ensemble_result):
    from src.signals import generate_signals, print_signals

    print(f"\n{'='*60}")
    print("SECCIÓN 2: SEÑALES DE COMPRA Y VENTA")
    print('='*60)

    sig_result = generate_signals(
        consensus=ensemble_result.consensus,
        confidence=ensemble_result.confidence,
        prices=ensemble_result.df["Close"],
    )
    print_signals(sig_result)
    return sig_result


# ──────────────────────────────────────────────────────────────────────────────
# Sección 3: Backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest_section(ensemble_result):
    from src.backtest import run_backtest, run_buyhold, compare_strategies

    print(f"\n{'='*60}")
    print("SECCIÓN 3: BACKTEST régimen vs Buy & Hold")
    print('='*60)

    prices = ensemble_result.df["Close"]
    strat = run_backtest(prices, ensemble_result.consensus, strategy_name="Regime Ensemble")
    bh    = run_buyhold(prices)
    print(compare_strategies([strat, bh]).to_string())
    return strat, bh


# ──────────────────────────────────────────────────────────────────────────────
# Sección 4: Walk-forward (opcional)
# ──────────────────────────────────────────────────────────────────────────────

def run_walkforward_section(raw, k):
    from src.features import build_features, get_feature_matrix, FEATURE_COLS
    from src.walkforward import walk_forward

    print(f"\n{'='*60}")
    print("SECCIÓN 4: WALK-FORWARD (re-entrenamiento periódico)")
    print('='*60)

    df = build_features(raw)
    df_model = get_feature_matrix(df)
    print(f"Corriendo walk-forward sobre {len(df_model)} muestras...")
    wf = walk_forward(df_model, FEATURE_COLS, k_kmeans=k, k_gmm=k, n_hmm=k)

    print(f"  → {len(wf)} días etiquetados")
    for reg, pct in wf["Consensus"].value_counts(normalize=True).mul(100).round(1).items():
        print(f"  {reg:10s} {pct:.1f}%")
    return wf


# ──────────────────────────────────────────────────────────────────────────────
# Gráficos
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(ensemble_result, sig_result, strat, bh):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f"LONQ — {ensemble_result.ticker}", fontsize=14, fontweight="bold")

    df = ensemble_result.df
    colors = {"bull": "#2ecc71", "sideways": "#f39c12", "bear": "#e74c3c", "unknown": "#95a5a6"}

    # 1. Precio + regímenes + señales
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#2c3e50", linewidth=1, zorder=3)
    prev_date, prev_regime = df.index[0], ensemble_result.consensus.iloc[0]
    for date, regime in zip(df.index[1:], ensemble_result.consensus.iloc[1:]):
        if regime != prev_regime or date == df.index[-1]:
            ax1.axvspan(prev_date, date, alpha=0.18,
                        color=colors.get(prev_regime, "gray"))
            prev_date, prev_regime = date, regime

    # Señales BUY/SELL
    sigs = sig_result.signals
    buys  = sigs[sigs["Signal"] == "BUY"]
    sells = sigs[sigs["Signal"] == "SELL"]
    ax1.scatter(buys.index,  buys["Close"],  marker="^", color="#27ae60", s=80, zorder=5, label="BUY")
    ax1.scatter(sells.index, sells["Close"], marker="v", color="#c0392b", s=80, zorder=5, label="SELL")

    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor=c, alpha=0.4, label=r) for r, c in colors.items() if r != "unknown"]
    ax1.legend(handles=legend_el + [
        plt.Line2D([0],[0], marker="^", color="w", markerfacecolor="#27ae60", markersize=9, label="BUY"),
        plt.Line2D([0],[0], marker="v", color="w", markerfacecolor="#c0392b", markersize=9, label="SELL"),
    ], loc="upper left", fontsize=8, ncol=3)
    ax1.set_title("Precio con regímenes y señales")
    ax1.set_ylabel("Precio")

    # 2. Posición (0/1)
    ax2 = axes[1]
    ax2.fill_between(sigs.index, sigs["Position"], alpha=0.5, color="#3498db", step="post")
    ax2.set_title("Posición (1 = en mercado, 0 = cash)")
    ax2.set_ylabel("Posición")
    ax2.set_ylim(-0.1, 1.4)

    # 3. Equity curves
    ax3 = axes[2]
    ax3.plot(strat.equity_curve.index, strat.equity_curve,
             label="Regime Strategy", color="#3498db", linewidth=1.5)
    ax3.plot(bh.equity_curve.index, bh.equity_curve,
             label="Buy & Hold", color="#e74c3c", linewidth=1.5, linestyle="--")
    ax3.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax3.set_title("Curva de capital")
    ax3.set_ylabel("Valor (base 1.0)")
    ax3.legend()

    # 4. Confianza del ensemble
    ax4 = axes[3]
    conf = ensemble_result.confidence
    ax4.fill_between(conf.index, conf, alpha=0.5, color="#9b59b6")
    ax4.axhline(0.67, color="orange", linewidth=1, linestyle="--", label="Moderado (0.67)")
    ax4.axhline(1.0,  color="green",  linewidth=1, linestyle="--", label="Fuerte (1.0)")
    ax4.set_title("Confianza del ensemble")
    ax4.set_ylabel("Confianza")
    ax4.set_ylim(0, 1.1)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    out = "lonq_demo_output.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nGráfico guardado en: {out}")
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
    true_regimes = None
    if args.ticker:
        raw, label = load_from_ticker(args.ticker, args.start)
    elif args.csv:
        raw, label = load_from_csv(args.csv)
    else:
        raw, true_regimes = load_synthetic()
        label = "SYNTHETIC"

    # Ensemble
    ensemble_result = run_ensemble(raw, label, args.k_min, args.k_max)

    # Validación (solo datos sintéticos)
    if true_regimes is not None:
        regime_map = {0: "bull", 1: "sideways", 2: "bear"}
        true_sem = true_regimes.map(regime_map)
        common = ensemble_result.consensus.index.intersection(true_sem.index)
        if len(common) > 0:
            acc = (ensemble_result.consensus.loc[common] == true_sem.loc[common]).mean()
            print(f"\n[Validación] Precisión vs regímenes reales: {acc:.1%}")

    # Señales
    sig_result = run_signals_section(ensemble_result)

    # Backtest
    strat, bh = run_backtest_section(ensemble_result)

    # Walk-forward (opcional)
    if args.walkforward:
        run_walkforward_section(raw, k=3)

    # Gráficos
    if not args.no_plots:
        plot_results(ensemble_result, sig_result, strat, bh)

    print("\nDemo completado.")


if __name__ == "__main__":
    main()
