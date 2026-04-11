"""LONQ — Screener masivo de activos.

Uso:
    python screen_tickers.py                          # todos los tickers de tickers.yaml
    python screen_tickers.py --sectors consumer_staples utilities
    python screen_tickers.py --tickers KO PG JNJ NEE  # lista manual
    python screen_tickers.py --deep --top 20          # análisis profundo en top 20
    python screen_tickers.py --fast-only              # solo screening rápido (sin ensemble)
    python screen_tickers.py --workers 8              # 8 hilos paralelos
    python screen_tickers.py --save results/          # guardar resultados
"""

import argparse
import os
import sys
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yaml

from src.screener import (
    fast_screen, deep_screen, print_screen_result,
    screen_summary_df, ScreenResult,
)


# ── Carga de tickers ─────────────────────────────────────────────────────────

def load_tickers(args) -> dict[str, list[str]]:
    """Devuelve dict {sector: [tickers]}."""
    if args.tickers:
        return {"manual": [t.upper() for t in args.tickers]}

    yaml_path = os.path.join(os.path.dirname(__file__), "tickers.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No se encontró {yaml_path}")

    with open(yaml_path) as f:
        all_tickers = yaml.safe_load(f)

    if args.sectors:
        return {s: all_tickers[s] for s in args.sectors if s in all_tickers}
    return all_tickers


def fetch_prices(ticker: str, start_date: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Descarga o carga desde CSV local. Devuelve (raw, prices) o None si falla."""
    CSV_MAP = {
        "KO":      "ko_data.csv",
        "TSLA":    "tsla_data.csv",
        "SPY":     "spy_data.csv",
        "BTC-USD": "btc_data.csv",
        "BTC":     "btc_data.csv",
    }
    t = ticker.upper()

    # CSV local
    if t in CSV_MAP and os.path.exists(CSV_MAP[t]):
        raw = pd.read_csv(CSV_MAP[t], index_col="Date", parse_dates=True)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw = raw[raw.index >= start_date].dropna(subset=["Close"])
        return raw, raw["Close"]

    # yfinance
    try:
        import yfinance as yf
        raw = yf.download(t, start=start_date, progress=False, auto_adjust=True)
        if raw.empty:
            return None
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw = raw[raw.index >= start_date].dropna(subset=["Close"])
        return raw, raw["Close"]
    except Exception:
        return None


# ── Worker ────────────────────────────────────────────────────────────────────

def _screen_one(ticker: str, start_date: str, deep: bool, vix_series) -> ScreenResult:
    """Función ejecutada en cada hilo."""
    result = fetch_prices(ticker, start_date)
    if result is None:
        return ScreenResult(
            ticker=ticker, score=0, grade="✗ ",
            recommendation="Sin datos — no se pudo descargar",
            vol_annual=0, autocorr_5d=0, trend_strength=0, n_days=0,
            error="Sin datos",
        )
    raw, prices = result

    if len(prices) < 100:
        return ScreenResult(
            ticker=ticker, score=0, grade="✗ ",
            recommendation="Histórico insuficiente",
            vol_annual=0, autocorr_5d=0, trend_strength=0, n_days=len(prices),
            error="Histórico insuficiente",
        )

    fast = fast_screen(prices, ticker)

    if deep and fast.score >= 25:   # solo hace deep si pasa el mínimo rápido
        return deep_screen(prices, raw, ticker, vix_series=vix_series, fast_result=fast)
    return fast


# ── CLI principal ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LONQ Asset Screener")
    p.add_argument("--tickers",    nargs="+", help="Lista manual de tickers")
    p.add_argument("--sectors",    nargs="+", help="Sectores de tickers.yaml")
    p.add_argument("--start",      default="2018-01-01", help="Fecha inicio")
    p.add_argument("--deep",       action="store_true",
                   help="Análisis profundo con ensemble (lento pero preciso)")
    p.add_argument("--fast-only",  action="store_true",
                   help="Solo screening rápido, ignora --deep")
    p.add_argument("--top",        type=int, default=None,
                   help="Mostrar solo el top N (útil con --deep)")
    p.add_argument("--workers",    type=int, default=4,
                   help="Hilos paralelos (default 4)")
    p.add_argument("--min-score",  type=int, default=0,
                   help="Filtrar resultados por score mínimo")
    p.add_argument("--vix",        default="vix_data.csv",
                   help="CSV del VIX (solo para --deep)")
    p.add_argument("--save",       default=None,
                   help="Carpeta donde guardar resultados CSV")
    return p.parse_args()


def main():
    args = parse_args()
    deep = args.deep and not args.fast_only

    print("=" * 65)
    print("  LONQ — ASSET SCREENER")
    print(f"  Modo: {'PROFUNDO (ensemble + señales)' if deep else 'RÁPIDO (estadístico)'}")
    print("=" * 65)

    # Cargar tickers
    sectors = load_tickers(args)
    all_tickers = []
    for sec, tks in sectors.items():
        all_tickers.extend(tks)
    all_tickers = list(dict.fromkeys(all_tickers))   # deduplicar
    print(f"\n  Universo: {len(all_tickers)} tickers  |  {len(sectors)} sectores")
    print(f"  Período:  desde {args.start}")
    print(f"  Hilos:    {args.workers}")

    # VIX
    vix_series = None
    if deep and os.path.exists(args.vix):
        from src.vix import get_vix
        vix_series = get_vix(path=args.vix, auto_update=False)
        print(f"  VIX:      {len(vix_series)} días cargados")

    print(f"\n  Iniciando screening...\n")

    # ── Ejecución paralela ─────────────────────────────────────────────────
    results = []
    failed  = []
    t0      = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_screen_one, tk, args.start, deep, vix_series): tk
            for tk in all_tickers
        }
        done = 0
        for future in as_completed(futures):
            tk = futures[future]
            done += 1
            try:
                r = future.result()
                results.append(r)
                status = f"{r.grade} {r.score:>3}/100"
            except Exception as e:
                failed.append(tk)
                status = f"✗  ERROR"
            elapsed = time.time() - t0
            print(f"  [{done:>3}/{len(all_tickers)}]  {tk:<10} {status}  "
                  f"({elapsed:.0f}s)")

    elapsed_total = time.time() - t0

    # ── Ordenar y filtrar ──────────────────────────────────────────────────
    results.sort(key=lambda r: r.score, reverse=True)
    if args.min_score > 0:
        results = [r for r in results if r.score >= args.min_score]

    # ── Resumen por sectores ───────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  RESULTADOS — {len(results)} tickers analizados en {elapsed_total:.0f}s")
    print(f"{'═'*65}")

    # Distribución por grado
    grades = {"✓✓": 0, "✓ ": 0, "~ ": 0, "✗ ": 0}
    for r in results:
        grades[r.grade] = grades.get(r.grade, 0) + 1

    print(f"\n  Distribución:")
    print(f"  ✓✓ Excelente: {grades.get('✓✓',0):>4} tickers")
    print(f"  ✓  Bueno:     {grades.get('✓ ',0):>4} tickers")
    print(f"  ~  Marginal:  {grades.get('~ ',0):>4} tickers")
    print(f"  ✗  No apto:   {grades.get('✗ ',0):>4} tickers")

    # Top resultados
    show = results[:args.top] if args.top else results
    print(f"\n{'─'*65}")
    print(f"  RANKING{'  (top ' + str(args.top) + ')' if args.top else ''}")
    print(f"{'─'*65}")
    for r in show:
        print_screen_result(r)

    # ── Tabla resumen ──────────────────────────────────────────────────────
    df_summary = screen_summary_df(results)
    print(f"\n{'─'*65}")
    print("  TABLA RESUMEN")
    print(f"{'─'*65}")
    cols = ["Ticker", "Score", "Grado", "Vol. anual", "Autocorr 5d", "Tendencia R²"]
    if deep:
        cols += ["Sil. avg", "Rev. score", "WR +10d", "Sharpe est.", "Sharpe B&H"]
    print(df_summary[cols].to_string(index=False))

    # ── Top 10 recomendados ────────────────────────────────────────────────
    top10 = [r for r in results if r.score >= 50][:10]
    if top10:
        print(f"\n  ★ TOP CANDIDATOS PARA ANÁLISIS COMPLETO:")
        for i, r in enumerate(top10, 1):
            print(f"  {i:>2}. {r.ticker:<8}  score {r.score}/100  {r.grade}")

    # ── Guardar resultados ─────────────────────────────────────────────────
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        today = date.today().isoformat()
        path  = os.path.join(args.save, f"screen_{today}.csv")
        df_summary.to_csv(path, index=False)
        print(f"\n  💾 Resultados guardados: {path}")

    if failed:
        print(f"\n  ⚠  Tickers con error: {', '.join(failed)}")

    return results


if __name__ == "__main__":
    main()
