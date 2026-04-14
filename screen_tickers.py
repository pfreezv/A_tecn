"""LONQ — Screener masivo de activos.

Uso:
    python screen_tickers.py                                  # universo custom (tickers.yaml)
    python screen_tickers.py --universe custom                # ídem explícito
    python screen_tickers.py --universe ibex35,stoxx50        # índices EU
    python screen_tickers.py --universe sp500,ibex35,stoxx50,crypto  # todo
    python screen_tickers.py --sectors consumer_staples utilities     # sectores del custom
    python screen_tickers.py --tickers KO PG JNJ NEE          # lista manual
    python screen_tickers.py --deep --top 20                  # análisis profundo en top 20
    python screen_tickers.py --workers 8                      # 8 hilos paralelos
    python screen_tickers.py --save results/                  # guardar CSV
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
from src.data import fetch_ohlcv, is_blacklisted


# ── Carga de tickers ──────────────────────────────────────────────────────────

def load_tickers(args) -> list[tuple[str, str, str]]:
    """Devuelve lista de (ticker, market, mode).

    Fuentes (en orden de prioridad):
      1. --tickers  → lista manual (market=US, mode=full)
      2. --universe → src/universe.py
      3. --sectors  → secciones de tickers.yaml (market=US, mode=full)
      4. por defecto: tickers.yaml completo (market=US, mode=full)
    """
    # 1. Lista manual
    if args.tickers:
        return [(t.upper(), "US", "full") for t in args.tickers]

    # 2. Universos dinámicos
    if getattr(args, "universe", None):
        from src.universe import get_universe
        names = [u.strip() for u in args.universe.split(",") if u.strip()]
        tickers = get_universe(names, use_cache=True)
        return [(ut.ticker, ut.market, ut.mode) for ut in tickers]

    # 3. tickers.yaml (custom)
    yaml_path = os.path.join(os.path.dirname(__file__), "tickers.yaml")
    with open(yaml_path) as f:
        all_data = yaml.safe_load(f)

    # Solo leer secciones que sean listas (ignorar 'universes:' dict)
    all_sectors = {
        k: v for k, v in all_data.items()
        if isinstance(v, list)
    }

    if args.sectors:
        selected = {s: all_sectors[s] for s in args.sectors if s in all_sectors}
    else:
        selected = all_sectors

    result = []
    for tickers_list in selected.values():
        for t in tickers_list:
            result.append((str(t).strip().upper(), "US", "full"))

    # Deduplicar manteniendo orden
    seen = set()
    deduped = []
    for item in result:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    return deduped


# ── Descarga de precios ───────────────────────────────────────────────────────

def fetch_prices(ticker: str, start_date: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Descarga precios usando src/data.py (con retry+backoff+blacklist).
    Fallback a CSV locales para demos.
    """
    CSV_MAP = {
        "KO":      "ko_data.csv",
        "TSLA":    "tsla_data.csv",
        "SPY":     "spy_data.csv",
        "BTC-USD": "btc_data.csv",
        "BTC":     "btc_data.csv",
    }
    t = ticker.upper()

    # CSV local (demo / CI sin red)
    if t in CSV_MAP and os.path.exists(CSV_MAP[t]):
        raw = pd.read_csv(CSV_MAP[t], index_col="Date", parse_dates=True)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw = raw[raw.index >= start_date].dropna(subset=["Close"])
        if not raw.empty:
            return raw, raw["Close"]

    # Blacklist check rápido
    if is_blacklisted(t):
        return None

    try:
        raw = fetch_ohlcv(t, start_date=start_date, use_cache=True)
        if raw.empty:
            return None
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw.index   = pd.to_datetime(raw.index).tz_localize(None)
        raw = raw[raw.index >= start_date].dropna(subset=["Close"])
        if raw.empty:
            return None
        return raw, raw["Close"]
    except Exception:
        return None


# ── Worker ─────────────────────────────────────────────────────────────────────

def _screen_one(
    ticker: str,
    start_date: str,
    deep: bool,
    vix_series,
    mode: str = "full",
) -> ScreenResult:
    """Analiza un ticker. mode='trigger_only' omite el gate de score para deep."""
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

    # trigger_only: crypto y tendenciales — fast_screen da contexto estadístico
    # pero no bloqueamos el pipeline por score bajo (son tendenciales por diseño)
    if mode == "trigger_only":
        fast = ScreenResult(
            **{**fast.__dict__,
               "recommendation": f"[trigger_only] {fast.recommendation}"}
        )
        return fast   # sin deep ensemble para activos tendenciales

    # full: solo deep si pasa el mínimo rápido
    if deep and fast.score >= 25:
        return deep_screen(prices, raw, ticker, vix_series=vix_series, fast_result=fast)
    return fast


# ── CLI principal ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LONQ Asset Screener")
    p.add_argument("--universe",   default=None,
                   help="Universos separados por coma: custom,sp500,ibex35,stoxx50,crypto")
    p.add_argument("--tickers",    nargs="+", help="Lista manual de tickers")
    p.add_argument("--sectors",    nargs="+", help="Sectores de tickers.yaml (solo con --universe custom)")
    p.add_argument("--start",      default="2018-01-01", help="Fecha inicio")
    p.add_argument("--deep",       action="store_true",
                   help="Análisis profundo con ensemble (lento pero preciso)")
    p.add_argument("--fast-only",  action="store_true",
                   help="Solo screening rápido, ignora --deep")
    p.add_argument("--top",        type=int, default=None,
                   help="Deep solo al top N por fast_score (acelera modo deep)")
    p.add_argument("--workers",    type=int, default=4,
                   help="Hilos paralelos (default 4)")
    p.add_argument("--min-score",  type=int, default=0,
                   help="Filtrar resultados mostrados por score mínimo")
    p.add_argument("--vix",        default="vix_data.csv",
                   help="CSV del VIX (solo para --deep)")
    p.add_argument("--save",       default=None,
                   help="Carpeta donde guardar resultados CSV")
    p.add_argument("--no-blacklist", action="store_true",
                   help="Ignorar blacklist (fuerza reintento de tickers fallidos)")
    return p.parse_args()


def main():
    args = parse_args()
    deep = args.deep and not args.fast_only

    print("=" * 70)
    print("  LONQ — ASSET SCREENER")
    print(f"  Modo: {'PROFUNDO (ensemble + señales)' if deep else 'RÁPIDO (estadístico)'}")
    print("=" * 70)

    # ── Cargar universo ────────────────────────────────────────────────────
    raw_tickers = load_tickers(args)

    # Filtrar blacklist salvo --no-blacklist
    if not args.no_blacklist:
        from src.data import get_blacklist
        bl = set(get_blacklist())
        if bl:
            skipped = [t for t, _, _ in raw_tickers if t in bl]
            raw_tickers = [(t, m, md) for t, m, md in raw_tickers if t not in bl]
            if skipped:
                print(f"  ⚠  Blacklist: {len(skipped)} tickers omitidos — {', '.join(skipped[:10])}"
                      + (" …" if len(skipped) > 10 else ""))

    # Separar por modo
    full_tickers    = [(t, m) for t, m, md in raw_tickers if md == "full"]
    trigger_tickers = [(t, m) for t, m, md in raw_tickers if md == "trigger_only"]

    # Contar por mercado
    markets = {}
    for t, m, _ in raw_tickers:
        markets[m] = markets.get(m, 0) + 1

    print(f"\n  Universo: {len(raw_tickers)} tickers únicos")
    for mkt, cnt in sorted(markets.items()):
        print(f"    {mkt:<8}: {cnt} tickers")
    if trigger_tickers:
        print(f"  Modo trigger_only (crypto): {len(trigger_tickers)} tickers")
    print(f"  Período:  desde {args.start}")
    print(f"  Hilos:    {args.workers}")

    # VIX
    vix_series = None
    if deep and os.path.exists(args.vix):
        from src.vix import get_vix
        vix_series = get_vix(path=args.vix, auto_update=False)
        print(f"  VIX:      {len(vix_series)} días cargados")

    print(f"\n  Iniciando screening...\n")

    # ── FASE A: fast_screen sobre todo el universo ─────────────────────────
    all_items = [(t, m, "full") for t, m in full_tickers] + \
                [(t, m, "trigger_only") for t, m in trigger_tickers]

    results_fast: list[ScreenResult] = []
    results_map:  dict[str, tuple[str, str]] = {t: (m, md) for t, m, md in all_items}

    t0   = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _screen_one, t, args.start,
                False,          # fast pass siempre — deep se hace después
                None,
                md,
            ): t
            for t, _, md in all_items
        }
        for future in as_completed(futures):
            tk = futures[future]
            done += 1
            try:
                r = future.result()
                results_fast.append(r)
                market, _ = results_map[tk]
                mkt_tag = f"[{market}]" if market != "US" else ""
                status = f"{r.grade} {r.score:>3}/100 {mkt_tag}"
            except Exception as exc:
                results_fast.append(ScreenResult(
                    ticker=tk, score=0, grade="✗ ",
                    recommendation="Error interno",
                    vol_annual=0, autocorr_5d=0, trend_strength=0, n_days=0,
                    error=str(exc)[:60],
                ))
                status = "✗  ERROR"
            elapsed = time.time() - t0
            print(f"  [{done:>4}/{len(all_items)}]  {tk:<12} {status}  ({elapsed:.0f}s)")

    # ── FASE B: deep_screen sobre top N (solo mode=full, score >= 25) ──────
    results_final = list(results_fast)   # copia que se irá actualizando

    if deep:
        candidates = sorted(
            [r for r in results_fast
             if r.error is None and r.score >= 25
             and results_map.get(r.ticker, ("US", "full"))[1] == "full"],
            key=lambda r: r.score, reverse=True,
        )
        if args.top:
            candidates = candidates[:args.top]

        print(f"\n  ── DEEP SCREEN sobre {len(candidates)} candidatos ──")
        deep_results: dict[str, ScreenResult] = {}

        with ThreadPoolExecutor(max_workers=max(1, args.workers // 2)) as executor:
            futures = {
                executor.submit(
                    _screen_one, r.ticker, args.start, True, vix_series, "full"
                ): r.ticker
                for r in candidates
            }
            d = 0
            for future in as_completed(futures):
                tk = futures[future]
                d += 1
                try:
                    deep_r = future.result()
                    deep_results[tk] = deep_r
                    status = f"{deep_r.grade} {deep_r.score:>3}/100"
                except Exception as exc:
                    status = f"✗  ERROR: {exc}"
                elapsed = time.time() - t0
                print(f"  [deep {d:>3}/{len(candidates)}]  {tk:<12} {status}  ({elapsed:.0f}s)")

        # Sustituir resultados fast por deep donde aplique
        results_final = [
            deep_results.get(r.ticker, r) for r in results_fast
        ]

    elapsed_total = time.time() - t0

    # ── Ordenar y filtrar ──────────────────────────────────────────────────
    results_final.sort(key=lambda r: r.score, reverse=True)
    if args.min_score > 0:
        display = [r for r in results_final if r.score >= args.min_score]
    else:
        display = results_final

    # ── Resumen global ─────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  RESULTADOS — {len(results_final)} tickers en {elapsed_total:.0f}s")
    print(f"{'═'*70}")

    grades = {"✓✓": 0, "✓ ": 0, "~ ": 0, "✗ ": 0}
    for r in results_final:
        grades[r.grade] = grades.get(r.grade, 0) + 1

    print(f"\n  Distribución:")
    print(f"  ✓✓ Excelente: {grades.get('✓✓',0):>4} tickers")
    print(f"  ✓  Bueno:     {grades.get('✓ ',0):>4} tickers")
    print(f"  ~  Marginal:  {grades.get('~ ',0):>4} tickers")
    print(f"  ✗  No apto:   {grades.get('✗ ',0):>4} tickers")

    # Top resultados
    show = display[:args.top] if (args.top and not deep) else display
    print(f"\n{'─'*70}")
    print(f"  RANKING{'  (top ' + str(args.top) + ')' if (args.top and not deep) else ''}")
    print(f"{'─'*70}")
    for r in show:
        print_screen_result(r)

    # Tabla resumen
    df_summary = screen_summary_df(display)
    print(f"\n{'─'*70}")
    print("  TABLA RESUMEN")
    print(f"{'─'*70}")
    cols = ["Ticker", "Score", "Grado", "Vol. anual", "Autocorr 5d", "Tendencia R²"]
    if deep:
        cols += ["Sil. avg", "Rev. score", "WR +10d", "Sharpe est.", "Sharpe B&H"]
    print(df_summary[cols].to_string(index=False))

    # Top candidatos
    top_candidates = [r for r in display if r.score >= 50 and not r.error][:10]
    if top_candidates:
        print(f"\n  ★ TOP CANDIDATOS (score ≥ 50):")
        for i, r in enumerate(top_candidates, 1):
            mkt = results_map.get(r.ticker, ("US", "full"))[0]
            mkt_tag = f" [{mkt}]" if mkt != "US" else ""
            print(f"  {i:>2}. {r.ticker:<10}  score {r.score}/100  {r.grade}{mkt_tag}")

    # ── Guardar resultados ─────────────────────────────────────────────────
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        today = date.today().isoformat()

        # CSV completo
        df_all = screen_summary_df(results_final)
        # Añadir columna de mercado
        market_col = [results_map.get(r.ticker, ("US",))[0] for r in results_final]
        df_all.insert(1, "Mercado", market_col)
        path_all = os.path.join(args.save, f"screen_{today}.csv")
        df_all.to_csv(path_all, index=False)
        print(f"\n  Resultados completos: {path_all}  ({len(df_all)} tickers)")

        # CSV top (score >= 50, sin errores)
        top_df = df_all[df_all["Score"] >= 50]
        if not top_df.empty:
            path_top = os.path.join(args.save, f"top_{today}.csv")
            top_df.to_csv(path_top, index=False)
            print(f"  Top candidatos:       {path_top}  ({len(top_df)} tickers)")

    failed = [r.ticker for r in results_final if r.error and "datos" in (r.error or "")]
    if failed:
        print(f"\n  ⚠  Sin datos: {', '.join(failed[:15])}"
              + (" …" if len(failed) > 15 else ""))

    return results_final


if __name__ == "__main__":
    main()
