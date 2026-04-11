"""LONQ — Generador de reportes diarios de screening.

Genera un informe Markdown con:
  - Resumen ejecutivo (fecha, universo analizado, tiempo)
  - Distribución por grados
  - Ranking de mejores candidatos
  - Tabla completa de resultados
  - Alertas de señales activas (si se pasa combined_results)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional
import os

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 100, width: int = 20) -> str:
    """Barra de progreso ASCII para Markdown."""
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def _pct_bar(value: float, width: int = 20) -> str:
    return _bar(value * 100, 100, width)


# ── Reporte de screener ───────────────────────────────────────────────────────

def generate_screen_report(
    results: list,
    run_date: Optional[date] = None,
    mode: str = "RÁPIDO",
    elapsed: float = 0.0,
    sectors_count: int = 0,
    failed: Optional[list] = None,
    output_path: Optional[str] = None,
) -> str:
    """Genera reporte Markdown del screening diario.

    Args:
        results:      Lista de ScreenResult ordenada por score desc.
        run_date:     Fecha del análisis (default: hoy).
        mode:         "RÁPIDO" o "PROFUNDO".
        elapsed:      Segundos que tardó el análisis.
        sectors_count: Número de sectores analizados.
        failed:       Tickers que fallaron.
        output_path:  Si se indica, guarda el archivo .md en esa ruta.

    Returns:
        String con el reporte en Markdown.
    """
    if run_date is None:
        run_date = date.today()

    failed = failed or []
    now    = datetime.now().strftime("%H:%M UTC")

    lines = []
    a = lines.append   # shorthand

    # ── Cabecera ──────────────────────────────────────────────────────────────
    a(f"# LONQ — Reporte Diario de Screening")
    a(f"")
    a(f"> **Fecha:** {run_date.isoformat()}  |  **Hora:** {now}  |  **Modo:** {mode}")
    a(f"")
    a(f"---")
    a(f"")

    # ── Resumen ejecutivo ─────────────────────────────────────────────────────
    total   = len(results)
    elapsed_min = elapsed / 60

    grades = {"✓✓": [], "✓ ": [], "~ ": [], "✗ ": []}
    for r in results:
        grades.setdefault(r.grade, []).append(r.ticker)

    n_excellent = len(grades.get("✓✓", []))
    n_good      = len(grades.get("✓ ", []))
    n_marginal  = len(grades.get("~ ", []))
    n_bad       = len(grades.get("✗ ", []))

    a(f"## 📊 Resumen Ejecutivo")
    a(f"")
    a(f"| Métrica | Valor |")
    a(f"|---------|-------|")
    a(f"| Tickers analizados | **{total}** |")
    a(f"| Sectores | {sectors_count} |")
    a(f"| Tiempo total | {elapsed_min:.1f} min |")
    a(f"| Fallos | {len(failed)} |")
    a(f"")

    # ── Distribución por grado ────────────────────────────────────────────────
    a(f"## 🎯 Distribución por Calidad")
    a(f"")
    a(f"| Grado | N | % | Barra |")
    a(f"|-------|---|---|-------|")

    for label, sym, count in [
        ("✓✓ Excelente", "✓✓", n_excellent),
        ("✓  Bueno",     "✓ ", n_good),
        ("~  Marginal",  "~ ", n_marginal),
        ("✗  No apto",   "✗ ", n_bad),
    ]:
        pct = count / total * 100 if total else 0
        bar = _bar(pct, 100, 15)
        a(f"| {label} | {count} | {pct:.1f}% | `{bar}` |")

    a(f"")

    # ── Top candidatos ────────────────────────────────────────────────────────
    top = [r for r in results if r.score >= 50][:15]
    if top:
        a(f"## ⭐ Top Candidatos (score ≥ 50)")
        a(f"")
        a(f"| # | Ticker | Score | Grado | Vol. anual | Autocorr 5d | Tendencia R² | Días |")
        a(f"|---|--------|-------|-------|-----------|-------------|-------------|------|")
        for i, r in enumerate(top, 1):
            a(f"| {i} | **{r.ticker}** | {r.score}/100 | {r.grade} | "
              f"{r.vol_annual:.1%} | {r.autocorr_5d:+.3f} | {r.trend_strength:.2f} | {r.n_days} |")
        a(f"")

    # ── Detalle profundo (si aplica) ──────────────────────────────────────────
    deep_results = [r for r in results if r.sil_avg is not None]
    if deep_results:
        a(f"## 🔬 Análisis Profundo")
        a(f"")
        a(f"| Ticker | Score | Sil. avg | Rev. score | WR +10d | Sharpe est. | Sharpe B&H |")
        a(f"|--------|-------|----------|-----------|---------|------------|-----------|")
        for r in deep_results[:20]:
            wr  = f"{r.win_rate_10d:.0%}" if r.win_rate_10d is not None else "—"
            ss  = f"{r.sharpe_strategy:.2f}" if r.sharpe_strategy is not None else "—"
            sbh = f"{r.sharpe_bh:.2f}" if r.sharpe_bh is not None else "—"
            sil = f"{r.sil_avg:.3f}" if r.sil_avg is not None else "—"
            rev = f"{r.reversion_score:.2f}" if r.reversion_score is not None else "—"
            a(f"| {r.ticker} | {r.score} | {sil} | {rev} | {wr} | {ss} | {sbh} |")
        a(f"")

    # ── Tabla completa ────────────────────────────────────────────────────────
    a(f"## 📋 Ranking Completo")
    a(f"")
    a(f"<details>")
    a(f"<summary>Ver todos los {total} tickers</summary>")
    a(f"")
    a(f"| Ticker | Score | Grado | Vol. anual | Autocorr 5d | Tendencia R² |")
    a(f"|--------|-------|-------|-----------|-------------|-------------|")
    for r in results:
        a(f"| {r.ticker} | {r.score} | {r.grade} | "
          f"{r.vol_annual:.1%} | {r.autocorr_5d:+.3f} | {r.trend_strength:.2f} |")
    a(f"")
    a(f"</details>")
    a(f"")

    # ── Tickers excelentes para seguimiento ───────────────────────────────────
    excellent = [r for r in results if r.score >= 70]
    if excellent:
        a(f"## 🏆 Lista de Seguimiento — Score ≥ 70")
        a(f"")
        tickers_str = ", ".join([f"`{r.ticker}`" for r in excellent])
        a(f"{tickers_str}")
        a(f"")
        a(f"```")
        a(f"# Para análisis completo en Colab:")
        a(f"TICKERS = {[r.ticker for r in excellent]}")
        a(f"```")
        a(f"")

    # ── Fallos ────────────────────────────────────────────────────────────────
    if failed:
        a(f"## ⚠️ Tickers con Error")
        a(f"")
        a(f"Los siguientes tickers no pudieron ser analizados:")
        a(f"")
        a(", ".join([f"`{t}`" for t in failed]))
        a(f"")

    # ── Pie de página ──────────────────────────────────────────────────────────
    a(f"---")
    a(f"")
    a(f"*Generado automáticamente por LONQ — {run_date.isoformat()} {now}*  ")
    a(f"*Sistema de detección de regímenes de mercado y reversión a la media*")

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def generate_daily_summary(
    results_path: str,
    top_n: int = 10,
) -> str:
    """Lee un CSV de resultados y genera un resumen de una línea para Slack/email.

    Args:
        results_path: Ruta al CSV con columnas Ticker, Score, Grado, ...
        top_n:        Número de tickers a mencionar en el resumen.

    Returns:
        Texto de resumen de una línea.
    """
    df = pd.read_csv(results_path)
    df = df.sort_values("Score", ascending=False)

    n_excellent = (df["Grado"] == "✓✓").sum()
    n_good      = (df["Grado"] == "✓ ").sum()
    top_tickers = df.head(top_n)["Ticker"].tolist()

    return (
        f"LONQ Daily Scan — {n_excellent} excellent, {n_good} good. "
        f"Top {top_n}: {', '.join(top_tickers)}"
    )
