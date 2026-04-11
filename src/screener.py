"""LONQ Asset Screener — filtra activos idóneos para la estrategia de reversión.

Dos etapas:
  RÁPIDA  (~2s/ticker)  — métricas estadísticas sin ensemble
  PROFUNDA (~60s/ticker) — análisis completo con las 4 capas

Puntuación 0–100:
  ≥ 70  ✓✓ Excelente candidato
  50–69 ✓  Buen candidato
  30–49 ~  Candidato marginal
  < 30  ✗  No recomendado
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ── Umbrales de scoring ───────────────────────────────────────────────────────

VOL_IDEAL_LO  = 0.10   # 10% anual — mínimo para tener movimiento
VOL_IDEAL_HI  = 0.28   # 28% anual — máximo antes de volverse errático
VOL_MAX       = 0.55   # >55% descalifica directamente (crypto extremo, meme)

MIN_DAYS      = 500    # mínimo de datos históricos para análisis fiable


# ── Resultado del screener ────────────────────────────────────────────────────

@dataclass
class ScreenResult:
    ticker:        str
    score:         int              # 0–100
    grade:         str              # ✓✓ / ✓ / ~ / ✗
    recommendation: str

    # Métricas rápidas
    vol_annual:    float
    autocorr_5d:   float            # autocorrelación lag-5 (negativa = revierte)
    trend_strength: float           # R² de la tendencia lineal (alto = tendencia)
    n_days:        int

    # Métricas profundas (None si solo se hizo screening rápido)
    sil_avg:       Optional[float]  = None
    reversion_score: Optional[float] = None
    win_rate_10d:  Optional[float]  = None
    sharpe_strategy: Optional[float] = None
    sharpe_bh:     Optional[float]  = None
    pct_in_market: Optional[float]  = None

    # Detalle de puntos
    score_breakdown: dict = field(default_factory=dict)

    # Errores si los hubo
    error:         Optional[str]    = None


# ── Screening rápido ──────────────────────────────────────────────────────────

def _trend_r2(prices: pd.Series) -> float:
    """R² de regresión lineal sobre log-precios. Alto = tendencia fuerte."""
    y = np.log(prices.values)
    x = np.arange(len(y))
    if len(x) < 10:
        return 0.0
    p  = np.polyfit(x, y, 1)
    yh = np.polyval(p, x)
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _autocorr(returns: pd.Series, lag: int = 5) -> float:
    """Autocorrelación de retornos a lag días. Negativa sugiere reversión."""
    if len(returns) < lag + 10:
        return 0.0
    return float(returns.autocorr(lag=lag))


def fast_screen(prices: pd.Series, ticker: str = "") -> ScreenResult:
    """Screening rápido (~2s): métricas estadísticas sin ML.

    Criterios:
      - Volatilidad anual en rango ideal (10–28%)
      - Autocorrelación negativa (señal de reversión)
      - Tendencia lineal no dominante (R² < 0.85)
      - Suficiente histórico (≥ 500 días)
    """
    ret = np.log(prices / prices.shift(1)).dropna()
    n   = len(prices)

    # ── Métricas base ──────────────────────────────────────────────────────
    vol_ann    = float(ret.std() * np.sqrt(252))
    autocorr5  = _autocorr(ret, lag=5)
    trend_r2   = _trend_r2(prices)

    score_bd   = {}
    score      = 0

    # ── 1. Suficiente histórico (0–10 pts) ────────────────────────────────
    if n >= 1000:
        score_bd["histórico"] = 10
    elif n >= 500:
        score_bd["histórico"] = 5
    else:
        score_bd["histórico"] = 0
    score += score_bd["histórico"]

    # ── 2. Volatilidad en rango ideal (0–25 pts) ──────────────────────────
    if vol_ann > VOL_MAX:
        score_bd["volatilidad"] = 0   # demasiado errático
    elif VOL_IDEAL_LO <= vol_ann <= VOL_IDEAL_HI:
        score_bd["volatilidad"] = 25  # rango ideal
    elif vol_ann < VOL_IDEAL_LO:
        score_bd["volatilidad"] = 8   # muy baja vol (poco movimiento)
    else:
        score_bd["volatilidad"] = 12  # alta pero no extrema
    score += score_bd["volatilidad"]

    # ── 3. Señal de reversión: autocorrelación negativa (0–30 pts) ────────
    if autocorr5 < -0.08:
        score_bd["reversión_autocorr"] = 30
    elif autocorr5 < -0.03:
        score_bd["reversión_autocorr"] = 18
    elif autocorr5 < 0.03:
        score_bd["reversión_autocorr"] = 8   # neutral
    else:
        score_bd["reversión_autocorr"] = 0   # momentum, no revierte
    score += score_bd["reversión_autocorr"]

    # ── 4. Tendencia no dominante (0–20 pts) ─────────────────────────────
    # Penaliza activos con tendencia muy fuerte (TSLA-like)
    if trend_r2 < 0.50:
        score_bd["anti_tendencia"] = 20
    elif trend_r2 < 0.70:
        score_bd["anti_tendencia"] = 12
    elif trend_r2 < 0.85:
        score_bd["anti_tendencia"] = 5
    else:
        score_bd["anti_tendencia"] = 0   # tendencia dominante
    score += score_bd["anti_tendencia"]

    # ── 5. Consistencia (varianza de volatilidad rolling) (0–15 pts) ─────
    roll_vol = ret.rolling(60).std() * np.sqrt(252)
    vol_consistency = float(roll_vol.std() / (roll_vol.mean() + 1e-9))
    if vol_consistency < 0.25:
        score_bd["consistencia"] = 15
    elif vol_consistency < 0.50:
        score_bd["consistencia"] = 8
    else:
        score_bd["consistencia"] = 0
    score += score_bd["consistencia"]

    score = min(score, 100)

    grade, rec = _grade(score, deep=False)

    return ScreenResult(
        ticker=ticker.upper(),
        score=score,
        grade=grade,
        recommendation=rec,
        vol_annual=vol_ann,
        autocorr_5d=autocorr5,
        trend_strength=trend_r2,
        n_days=n,
        score_breakdown=score_bd,
    )


# ── Screening profundo ────────────────────────────────────────────────────────

def deep_screen(
    prices:     pd.Series,
    raw:        pd.DataFrame,
    ticker:     str = "",
    vix_series: Optional[pd.Series] = None,
    fast_result: Optional[ScreenResult] = None,
    threshold:  float = 2.0,
    horizon:    int   = 10,
) -> ScreenResult:
    """Análisis completo con las 4 capas LONQ (~60s/ticker).

    Añade a los puntos del fast_screen:
      - Silhouette del ensemble (+20 pts)
      - Reversión estadística del trigger (+20 pts)
      - Win rate histórico de señales (+20 pts)
      - Ajuste por Sharpe relativo (−10 a +5 pts)
    """
    if fast_result is None:
        fast_result = fast_screen(prices, ticker)

    score_bd = dict(fast_result.score_breakdown)
    base     = fast_result.score
    bonus    = 0

    sil_avg       = None
    rev_score     = None
    win_rate_10d  = None
    sharpe_strat  = None
    sharpe_bh     = None
    pct_in        = None
    error         = None

    try:
        # ── Ensemble ──────────────────────────────────────────────────────
        from .ensemble import fit_ensemble
        ensemble = fit_ensemble(ticker, raw, k_min=2, k_max=5, show_progress=False)
        m   = ensemble.metrics
        sils = [float(m.loc[k, "Sil_Train"])
                for k in ["K-Means", "GMM", "HMM"]]
        sil_avg = float(np.nanmean(sils))

        if sil_avg > 0.35:
            b = 20; label = "✓✓ Excelente"
        elif sil_avg > 0.25:
            b = 12; label = "✓  Bueno"
        elif sil_avg > 0.15:
            b = 5;  label = "~  Aceptable"
        else:
            b = 0;  label = "⚠  Pobre"
        score_bd["ensemble_sil"] = b
        bonus += b

        # ── Trigger Multi-TF ──────────────────────────────────────────────
        from .trigger import run_multitf_trigger, TIMEFRAMES
        trigger = run_multitf_trigger(
            prices, ticker,
            timeframes=TIMEFRAMES,
            window=252,
            threshold=threshold,
        )
        rev_total = 0
        for tfr in trigger.timeframes.values():
            if not tfr.ttest.empty:
                ht = tfr.ttest[tfr.ttest["Tipo"] == "ANOMALY_HIGH"]
                lt = tfr.ttest[tfr.ttest["Tipo"] == "ANOMALY_LOW"]
                h  = len(ht[(ht["Sig. (p<0.05)"] == "✓") & (ht["Diferencia"] < 0)])
                l  = len(lt[(lt["Sig. (p<0.05)"] == "✓") & (lt["Diferencia"] > 0)])
                rev_total += (h / max(len(ht), 1)) + (l / max(len(lt), 1))
        rev_score = rev_total / len(trigger.timeframes)

        if rev_score >= 0.75:
            b = 20; label = "✓✓ Fuerte"
        elif rev_score >= 0.40:
            b = 12; label = "✓  Moderada"
        elif rev_score >= 0.20:
            b = 5;  label = "~  Débil"
        else:
            b = 0;  label = "⚠  Sin evidencia"
        score_bd["reversión_trigger"] = b
        bonus += b

        # ── Señales + P&L ─────────────────────────────────────────────────
        from .combined_signal import run_combined_signal, run_signal_pnl
        sig = run_combined_signal(
            prices, ticker,
            ensemble_result=ensemble,
            vix_series=vix_series,
            threshold=threshold,
            target_horizon=horizon,
            min_score=2,
        )
        pnl = run_signal_pnl(prices, sig, horizon=horizon)

        # Win rate
        bt = sig.backtest
        if not bt.empty and "hit_10d" in bt.columns:
            win_rate_10d = float(bt["hit_10d"].mean())
            if win_rate_10d >= 0.65:
                b = 20
            elif win_rate_10d >= 0.58:
                b = 12
            elif win_rate_10d >= 0.52:
                b = 5
            else:
                b = 0
            score_bd["win_rate_10d"] = b
            bonus += b

        # Sharpe relativo: penaliza si B&H es claramente mejor
        sharpe_strat = float(pnl["metrics"][0]["Sharpe"])
        sharpe_bh    = float(pnl["metrics"][1]["Sharpe"])
        pct_in       = float(pnl["pct_in"])
        ratio        = sharpe_strat / (sharpe_bh + 1e-6)

        if ratio >= 1.5:
            b = 10
        elif ratio >= 0.8:
            b = 5
        elif ratio >= 0.4:
            b = 0
        else:
            b = -10   # B&H abrumadoramente mejor
        score_bd["sharpe_relativo"] = b
        bonus += b

    except Exception as e:
        error = str(e)

    total = min(max(base + bonus, 0), 100)
    grade, rec = _grade(total, deep=True)

    return ScreenResult(
        ticker=ticker.upper(),
        score=total,
        grade=grade,
        recommendation=rec,
        vol_annual=fast_result.vol_annual,
        autocorr_5d=fast_result.autocorr_5d,
        trend_strength=fast_result.trend_strength,
        n_days=fast_result.n_days,
        sil_avg=sil_avg,
        reversion_score=rev_score,
        win_rate_10d=win_rate_10d,
        sharpe_strategy=sharpe_strat,
        sharpe_bh=sharpe_bh,
        pct_in_market=pct_in,
        score_breakdown=score_bd,
        error=error,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _grade(score: int, deep: bool = False) -> tuple[str, str]:
    suffix = "" if deep else " (rápido)"
    if score >= 70:
        return "✓✓", f"Excelente candidato{suffix} — aplicar análisis completo"
    elif score >= 50:
        return "✓ ", f"Buen candidato{suffix} — vale la pena analizar"
    elif score >= 30:
        return "~ ", f"Candidato marginal{suffix} — usar con precaución"
    return "✗ ", f"No recomendado{suffix} — activo poco apto para reversión"


def print_screen_result(r: ScreenResult) -> None:
    """Imprime resultado de screening de un ticker."""
    print(f"\n  {r.grade} {r.ticker:<8}  score {r.score:>3}/100  — {r.recommendation}")
    print(f"     Vol={r.vol_annual:.1%}  Autocorr5d={r.autocorr_5d:+.3f}  Tendencia R²={r.trend_strength:.2f}  Días={r.n_days}")
    if r.sil_avg is not None:
        print(f"     Sil.avg={r.sil_avg:.3f}  Rev.score={r.reversion_score:.2f}  "
              f"WR10d={r.win_rate_10d:.0%}" if r.win_rate_10d else
              f"     Sil.avg={r.sil_avg:.3f}  Rev.score={r.reversion_score:.2f}")
    if r.sharpe_strategy is not None:
        print(f"     Sharpe estrategia={r.sharpe_strategy:.2f}  B&H={r.sharpe_bh:.2f}  "
              f"Tiempo mercado={r.pct_in_market:.0%}")
    if r.error:
        print(f"     ⚠ Error parcial: {r.error[:80]}")


def screen_summary_df(results: list[ScreenResult]) -> pd.DataFrame:
    """Convierte lista de resultados en DataFrame ordenado por score."""
    rows = []
    for r in results:
        rows.append({
            "Ticker":      r.ticker,
            "Score":       r.score,
            "Grado":       r.grade,
            "Vol. anual":  f"{r.vol_annual:.1%}",
            "Autocorr 5d": f"{r.autocorr_5d:+.3f}",
            "Tendencia R²": f"{r.trend_strength:.2f}",
            "Sil. avg":    f"{r.sil_avg:.3f}" if r.sil_avg else "—",
            "Rev. score":  f"{r.reversion_score:.2f}" if r.reversion_score else "—",
            "WR +10d":     f"{r.win_rate_10d:.0%}" if r.win_rate_10d else "—",
            "Sharpe est.": f"{r.sharpe_strategy:.2f}" if r.sharpe_strategy else "—",
            "Sharpe B&H":  f"{r.sharpe_bh:.2f}" if r.sharpe_bh else "—",
            "Días":        r.n_days,
            "Error":       r.error or "",
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
