"""Combined Signal System: Trigger + Regime + VIX → Señal de disparo accionable.

Integra tres capas:
  1. TRIGGER     — detecta movimientos anómalos (z-score > umbral)
  2. RÉGIMEN     — contexto del ensemble (bull/sideways/bear + confianza)
  3. VIX         — contexto macro de miedo (opcional)

Genera señales:
  TAKE_PROFIT  — subida anómala + reversión probable → salir / ponerse corto
  BUY_DIP      — bajada anómala + rebote probable    → entrar / añadir
  HOLD         — condiciones no cumplidas

Cada señal incluye:
  - Precio objetivo (target)
  - Stop loss
  - Risk/Reward ratio
  - Puntuación de confianza (0–5 puntos)
  - Razonamiento detallado
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .trigger import run_multitf_trigger, TIMEFRAMES


# ── Configuración por defecto ────────────────────────────────────────────────

# Probabilidad mínima de reversión para emitir señal
MIN_REVERSAL_PROB   = 0.55   # 55% de los casos históricos revirtieron

# Horizonte de referencia para el precio objetivo (días)
TARGET_HORIZON      = 5      # +5 días por defecto

# Umbral de régimen para confirmar señal
BEARISH_REGIMES     = {"bear", "sideways"}
BULLISH_REGIMES     = {"bull"}

# VIX umbrales
VIX_CONFIRMS_TP     = 25.0   # VIX elevado → confirma Take Profit
VIX_CONFIRMS_DIP    = 35.0   # VIX extremo → confirma Buy Dip (pánico = oportunidad)

# Puntuación mínima para emitir señal
MIN_SCORE           = 2      # de 5 posibles


# ── Estructuras de datos ─────────────────────────────────────────────────────

@dataclass
class SignalEvent:
    """Una señal de disparo para un día concreto."""

    date:          pd.Timestamp
    signal:        str            # TAKE_PROFIT / BUY_DIP / HOLD
    score:         int            # 0–5 puntos
    confidence:    str            # "Alta" / "Media" / "Baja"

    # Trigger
    ret_today:     float          # retorno del día
    z_score:       float          # z-score del retorno
    anomaly_type:  str            # ANOMALY_HIGH / ANOMALY_LOW

    # Probabilidades históricas de reversión
    p_reversal:    float          # P(reversión en best_horizon días)
    best_horizon:  int            # horizonte con mayor P(reversión)
    expected_ret:  float          # retorno medio esperado en best_horizon

    # Precio y niveles
    price:         float
    target:        float          # precio objetivo (reversión)
    stop_loss:     float          # precio de stop
    risk_reward:   float          # |target-price| / |stop-price|
    potential_pct: float          # % ganancia potencial

    # Contexto régimen (puede ser None si no se usó ensemble)
    regime:        Optional[str]
    regime_conf:   Optional[float]

    # Contexto VIX (puede ser None si no se usó VIX)
    vix_level:     Optional[float]
    vix_regime:    Optional[str]

    # Razonamiento
    reasons:       list[str] = field(default_factory=list)


@dataclass
class CombinedSignalResult:
    """Resultado completo del sistema de señales combinado."""

    ticker:         str
    version:        str            # "Base" o "+VIX"
    signals:        pd.DataFrame   # todos los eventos de señal
    active_signals: pd.DataFrame   # solo TAKE_PROFIT y BUY_DIP
    backtest:       pd.DataFrame   # rendimiento de las señales históricas
    summary:        dict


# ── Cálculo de probabilidades de reversión ────────────────────────────────────

def _compute_reversal_probs(
    prices: pd.Series,
    threshold: float = 2.0,
    horizon: int = TARGET_HORIZON,
) -> dict:
    """Pre-computa probabilidades de reversión para ANOMALY_HIGH y ANOMALY_LOW.

    Retorna dict con claves:
      p_down_after_high: P(ret < 0 en +horizon días | ANOMALY_HIGH)
      p_up_after_low:    P(ret > 0 en +horizon días | ANOMALY_LOW)
      mean_ret_high:     retorno medio post-ANOMALY_HIGH
      mean_ret_low:      retorno medio post-ANOMALY_LOW
      sigma:             std global de retornos diarios
    """
    ret = np.log(prices / prices.shift(1)).dropna()
    mu  = ret.rolling(252, min_periods=30).mean()
    sig = ret.rolling(252, min_periods=30).std()
    z   = (ret - mu) / sig

    fwd  = np.log(prices / prices.shift(-horizon)).dropna()  # retorno inverso (hacia adelante)
    # Realmente necesitamos forward-looking: ret de date a date+horizon
    fwd_pos = {}
    for i in range(len(prices) - horizon):
        date = prices.index[i]
        fwd_ret = np.log(prices.iloc[i + horizon] / prices.iloc[i])
        fwd_pos[date] = fwd_ret
    fwd_series = pd.Series(fwd_pos)

    common = z.index.intersection(fwd_series.index)
    z_c    = z.loc[common].dropna()
    fwd_c  = fwd_series.loc[z_c.index]

    high_mask = z_c >  threshold
    low_mask  = z_c < -threshold

    high_fwd = fwd_c[high_mask]
    low_fwd  = fwd_c[low_mask]

    p_down_high = float((high_fwd < -0.002).sum() / len(high_fwd)) if len(high_fwd) > 0 else 0.5
    p_up_low    = float((low_fwd  >  0.002).sum() / len(low_fwd))  if len(low_fwd)  > 0 else 0.5

    return {
        "p_down_after_high": p_down_high,
        "p_up_after_low":    p_up_low,
        "mean_ret_high":     float(high_fwd.mean()) if len(high_fwd) > 0 else 0.0,
        "mean_ret_low":      float(low_fwd.mean())  if len(low_fwd)  > 0 else 0.0,
        "sigma_daily":       float(ret.std()),
        "n_high":            int(high_mask.sum()),
        "n_low":             int(low_mask.sum()),
    }


# ── Generador de señales ──────────────────────────────────────────────────────

def _score_signal(
    anomaly_type: str,
    p_reversal:   float,
    regime:       Optional[str],
    regime_conf:  Optional[float],
    vix_level:    Optional[float],
    vix_regime_cat: Optional[str],
) -> tuple[int, list[str]]:
    """Calcula puntuación 0–5 y lista de razones."""

    score   = 0
    reasons = []

    # Punto 1: anomalía detectada (condición base, siempre presente)
    score += 1
    reasons.append(f"Movimiento anómalo detectado (z > ±2σ)")

    # Punto 2: probabilidad histórica de reversión
    if p_reversal >= 0.70:
        score += 2
        reasons.append(f"Reversión histórica muy probable ({p_reversal:.0%} de los casos)")
    elif p_reversal >= MIN_REVERSAL_PROB:
        score += 1
        reasons.append(f"Reversión histórica probable ({p_reversal:.0%} de los casos)")

    # Punto 3: régimen confirma
    if regime is not None and regime_conf is not None:
        if anomaly_type == "ANOMALY_HIGH" and regime in BEARISH_REGIMES:
            score += 1
            reasons.append(f"Régimen {regime} apoya corrección (conf {regime_conf:.0%})")
        elif anomaly_type == "ANOMALY_HIGH" and regime == "bull" and regime_conf < 0.67:
            score += 1
            reasons.append(f"Bull débil (conf {regime_conf:.0%}) → reversión factible")
        elif anomaly_type == "ANOMALY_LOW" and regime in BULLISH_REGIMES:
            score += 1
            reasons.append(f"Régimen bull apoya rebote (conf {regime_conf:.0%})")

    # Punto 4: VIX confirma
    if vix_level is not None:
        if anomaly_type == "ANOMALY_HIGH" and vix_level > VIX_CONFIRMS_TP:
            score += 1
            reasons.append(f"VIX elevado ({vix_level:.1f} > {VIX_CONFIRMS_TP}) → miedo macro confirma TP")
        elif anomaly_type == "ANOMALY_LOW" and vix_level > VIX_CONFIRMS_DIP:
            score += 1
            reasons.append(f"VIX extremo ({vix_level:.1f} > {VIX_CONFIRMS_DIP}) → pánico = oportunidad")
        elif vix_level < 15 and anomaly_type == "ANOMALY_HIGH":
            # VIX bajo en subida anómala → complacencia → riesgo real
            score += 1
            reasons.append(f"VIX bajo ({vix_level:.1f}) → complacencia extrema, corrección probable")

    return min(score, 5), reasons


def _confidence_label(score: int) -> str:
    if score >= 4:
        return "Alta"
    elif score >= 2:
        return "Media"
    return "Baja"


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_combined_signal(
    prices:          pd.Series,
    ticker:          str,
    ensemble_result  = None,    # EnsembleResult opcional
    vix_series:      Optional[pd.Series] = None,
    threshold:       float = 2.0,
    target_horizon:  int = TARGET_HORIZON,
    min_score:       int = MIN_SCORE,
) -> CombinedSignalResult:
    """Genera señales de disparo combinando trigger + régimen + VIX.

    Args:
        prices:          Serie de precios de cierre.
        ticker:          Nombre del activo.
        ensemble_result: Resultado del ensemble (opcional).
        vix_series:      Serie VIX diaria (opcional, activa la versión +VIX).
        threshold:       Z-score umbral para anomalía.
        target_horizon:  Días hacia adelante para precio objetivo.
        min_score:       Puntuación mínima para emitir señal.
    """
    version = "+VIX" if vix_series is not None else "Base"

    # Pre-computar probabilidades históricas de reversión
    rev_probs = _compute_reversal_probs(prices, threshold, target_horizon)

    # Estadísticas rodantes del trigger (diario)
    from .trigger import _compute_tf_stats
    tf_df = _compute_tf_stats(prices, tf_days=1, window=252, threshold=threshold)

    # Preparar contexto de régimen
    regime_series  = None
    regime_conf_s  = None
    if ensemble_result is not None:
        regime_series  = ensemble_result.consensus
        regime_conf_s  = ensemble_result.confidence

    # Preparar contexto VIX
    vix_regime_map = None
    if vix_series is not None:
        from .vix import build_vix_features
        vix_feat = build_vix_features(vix_series)

    # ── Generar señales día a día ─────────────────────────────────────────────
    events = []

    for date, row in tf_df.iterrows():
        if row["event_type"] == "NORMAL":
            continue

        anomaly = row["event_type"]
        z       = float(row["z_score"])
        ret_t   = float(row["ret"])
        price   = float(prices.loc[date]) if date in prices.index else None
        if price is None or np.isnan(price):
            continue

        # Probabilidad de reversión para este tipo de anomalía
        if anomaly == "ANOMALY_HIGH":
            p_rev    = rev_probs["p_down_after_high"]
            exp_ret  = rev_probs["mean_ret_high"]
            p_label  = "P(baja)"
        else:
            p_rev    = rev_probs["p_up_after_low"]
            exp_ret  = rev_probs["mean_ret_low"]
            p_label  = "P(sube)"

        # Régimen del día
        regime = regime_conf = None
        if regime_series is not None and date in regime_series.index:
            regime      = regime_series.loc[date]
            regime_conf = float(regime_conf_s.loc[date]) if date in regime_conf_s.index else None

        # VIX del día
        vix_level = vix_cat = None
        if vix_series is not None and date in vix_series.index:
            vix_level = float(vix_series.loc[date])
            if vix_series is not None and "vix_feat" in dir():
                if date in vix_feat.index:
                    vix_cat = str(vix_feat.loc[date, "vix_regime"])

        # Puntuación
        score, reasons = _score_signal(
            anomaly, p_rev, regime, regime_conf, vix_level, vix_cat
        )

        # Niveles de precio
        sigma_d = rev_probs["sigma_daily"]
        if anomaly == "ANOMALY_HIGH":
            # Esperamos caída → Target es precio más bajo, Stop si sigue subiendo
            target    = price * (1 + exp_ret)          # exp_ret < 0 → target < price
            stop_loss = price * (1 + sigma_d)          # 1σ más arriba
            signal    = "TAKE_PROFIT" if score >= min_score else "HOLD"
        else:
            # Esperamos subida → Target es precio más alto, Stop si sigue bajando
            target    = price * (1 + exp_ret)          # exp_ret > 0 → target > price
            stop_loss = price * (1 - sigma_d)          # 1σ más abajo
            signal    = "BUY_DIP" if score >= min_score else "HOLD"

        dist_target = abs(target - price)
        dist_stop   = abs(stop_loss - price)
        rr          = dist_target / dist_stop if dist_stop > 0 else 0
        pot_pct     = (target / price - 1)

        events.append(SignalEvent(
            date=date, signal=signal, score=score,
            confidence=_confidence_label(score),
            ret_today=ret_t, z_score=z, anomaly_type=anomaly,
            p_reversal=p_rev, best_horizon=target_horizon, expected_ret=exp_ret,
            price=price, target=round(target, 2),
            stop_loss=round(stop_loss, 2), risk_reward=round(rr, 2),
            potential_pct=pot_pct,
            regime=regime, regime_conf=regime_conf,
            vix_level=vix_level, vix_regime=vix_cat,
            reasons=reasons,
        ))

    # ── Construir DataFrames ──────────────────────────────────────────────────
    if not events:
        empty = pd.DataFrame()
        return CombinedSignalResult(ticker, version, empty, empty, empty, {})

    rows = []
    for e in events:
        rows.append({
            "Fecha":       e.date.date(),
            "Señal":       e.signal,
            "Puntos":      e.score,
            "Confianza":   e.confidence,
            "Retorno hoy": f"{e.ret_today:+.3%}",
            "Z-score":     f"{e.z_score:+.2f}σ",
            "Tipo":        e.anomaly_type,
            "P(reversión)": f"{e.p_reversal:.0%}",
            "Precio":      e.price,
            "Target":      e.target,
            "Stop Loss":   e.stop_loss,
            "R/R":         e.risk_reward,
            "Potencial":   f"{e.potential_pct:+.2%}",
            "Régimen":     e.regime or "—",
            "Conf. Régimen": f"{e.regime_conf:.0%}" if e.regime_conf else "—",
            "VIX":         f"{e.vix_level:.1f}" if e.vix_level else "—",
            "VIX Régimen": e.vix_regime or "—",
            "Razones":     " | ".join(e.reasons),
        })

    df_all    = pd.DataFrame(rows)
    df_active = df_all[df_all["Señal"] != "HOLD"].copy()

    # ── Backtest de señales ───────────────────────────────────────────────────
    backtest_rows = []
    for e in events:
        if e.signal == "HOLD":
            continue
        loc = prices.index.get_loc(e.date)
        results_at = {}
        for h in [1, 5, 10, 20]:
            if loc + h < len(prices):
                fwd = np.log(prices.iloc[loc + h] / prices.iloc[loc])
                if e.signal == "TAKE_PROFIT":
                    hit = fwd < -0.002   # precio bajó → TP exitoso
                else:
                    hit = fwd > 0.002    # precio subió → DIP exitoso
                results_at[f"ret_{h}d"]  = fwd
                results_at[f"hit_{h}d"]  = hit

        row = {
            "Fecha":    e.date.date(),
            "Señal":    e.signal,
            "Puntos":   e.score,
            "Confianza": e.confidence,
            "Precio":   e.price,
            "Target":   e.target,
            "Stop":     e.stop_loss,
            "P(rev) hist": e.p_reversal,
        }
        row.update(results_at)
        backtest_rows.append(row)

    bt_df = pd.DataFrame(backtest_rows) if backtest_rows else pd.DataFrame()

    # ── Resumen ───────────────────────────────────────────────────────────────
    n_tp   = (df_active["Señal"] == "TAKE_PROFIT").sum()
    n_dip  = (df_active["Señal"] == "BUY_DIP").sum()
    summary = {
        "Versión":            version,
        "Total anomalías":    len(df_all),
        "Señales TAKE_PROFIT": int(n_tp),
        "Señales BUY_DIP":    int(n_dip),
        "Señales activas":    int(n_tp + n_dip),
        "% con señal":        f"{(n_tp+n_dip)/max(len(df_all),1):.0%}",
    }

    if not bt_df.empty and "hit_5d" in bt_df.columns:
        hit5 = bt_df["hit_5d"].mean()
        summary["Win rate +5d"] = f"{hit5:.0%}"
    if not bt_df.empty and "hit_10d" in bt_df.columns:
        hit10 = bt_df["hit_10d"].mean()
        summary["Win rate +10d"] = f"{hit10:.0%}"

    return CombinedSignalResult(
        ticker=ticker, version=version,
        signals=df_all, active_signals=df_active,
        backtest=bt_df, summary=summary,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_combined_report(result: CombinedSignalResult) -> None:
    """Imprime el informe completo de señales combinadas."""

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  SEÑALES COMBINADAS — {result.ticker}  [{result.version}]")
    print(sep)

    if result.signals.empty:
        print("  No se generaron señales.")
        return

    # Resumen
    print("\n--- Resumen ---")
    for k, v in result.summary.items():
        print(f"  {k:<28} {v}")

    # Señales activas
    if not result.active_signals.empty:
        print(f"\n--- Señales de disparo ({len(result.active_signals)} eventos) ---")
        cols = ["Fecha","Señal","Puntos","Confianza","Retorno hoy","Z-score",
                "P(reversión)","Precio","Target","Stop Loss","R/R","Potencial",
                "Régimen","VIX"]
        print(result.active_signals[[c for c in cols if c in result.active_signals.columns]].to_string(index=False))

        # Detalles de las señales de alta confianza
        high_conf = result.active_signals[result.active_signals["Confianza"] == "Alta"]
        if not high_conf.empty:
            print(f"\n--- Señales de ALTA confianza ({len(high_conf)}) ---")
            for _, row in high_conf.iterrows():
                print(f"\n  [{row['Fecha']}] {row['Señal']}  (score {row['Puntos']}/5)")
                print(f"    Precio: ${row['Precio']:.2f}  →  Target: ${row['Target']:.2f}  |  Stop: ${row['Stop Loss']:.2f}")
                print(f"    Movimiento: {row['Retorno hoy']} ({row['Z-score']})  |  P(reversión): {row['P(reversión)']}")
                print(f"    Razones: {row['Razones']}")

    # Backtest
    if not result.backtest.empty and "hit_5d" in result.backtest.columns:
        print(f"\n--- Validación histórica (¿acertó la señal?) ---")
        bt = result.backtest.copy()
        for h in [1, 5, 10, 20]:
            col = f"hit_{h}d"
            if col in bt.columns:
                wr = bt[col].mean()
                n  = bt[col].notna().sum()
                ret_col = f"ret_{h}d"
                avg_ret = bt[ret_col].mean() if ret_col in bt.columns else 0
                print(f"  +{h:2d}d  Win rate: {wr:.0%}  |  Ret medio: {avg_ret:+.3%}  |  n={n}")

        # Por confianza
        print(f"\n  Win rate +5d por nivel de confianza:")
        for conf in ["Alta","Media","Baja"]:
            sub = bt[bt["Confianza"] == conf]
            if not sub.empty and "hit_5d" in sub.columns:
                wr = sub["hit_5d"].mean()
                print(f"    {conf:<8}  {wr:.0%}  (n={len(sub)})")
