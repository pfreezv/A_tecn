"""Trigger module: mean-reversion anomaly detector — Multi-timeframe edition.

Hipótesis: los precios tienden a volver a su nivel basal después de movimientos
anómalos (retornos fuera del rango estadísticamente "normal").

Soporta 4 marcos temporales para el cálculo del retorno:
  - Diario    (1d):  ln(P_t / P_{t-1})
  - Semanal   (5d):  ln(P_t / P_{t-5})
  - Quincenal (10d): ln(P_t / P_{t-10})
  - Mensual   (21d): ln(P_t / P_{t-21})

Para cada marco temporal calcula:
  1. Media y std rodante (ventana 1 año)
  2. Rango "normal" del ticker: [mu - k*sigma, mu + k*sigma]
  3. Eventos anómalos: ANOMALY_HIGH / ANOMALY_LOW
  4. Comportamiento post-evento: P(sube/plano/baja) a N períodos vista
  5. Validación estadística (t-test)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── Timeframes soportados ────────────────────────────────────────────────────
TIMEFRAMES = {
    "Diario (1d)":      1,
    "Semanal (5d)":     5,
    "Quincenal (10d)": 10,
    "Mensual (21d)":   21,
}

# Horizons de análisis relativos a cada timeframe (en número de períodos)
FORWARD_PERIODS = [1, 2, 4, 8]   # x1 TF, x2 TF, x4 TF, x8 TF

WINDOW     = 252   # días de ventana rodante (1 año de trading)
THRESHOLD  = 2.0   # z-score para anomalía


# ── Estructuras de datos ─────────────────────────────────────────────────────

@dataclass
class TimeframeResult:
    """Resultado del análisis para un solo timeframe."""
    name:          str             # "Diario (1d)"
    tf_days:       int             # 1, 5, 10, 21
    df:            pd.DataFrame    # retornos + estadísticas rodantes
    stats_summary: pd.DataFrame    # por año
    anomalies:     pd.DataFrame    # log de eventos
    post_event:    dict            # {horizonte_días: DataFrame}
    ttest:         pd.DataFrame    # validación estadística
    normal_range:  tuple[float, float]  # rango típico del ticker


@dataclass
class MultiTFResult:
    """Resultado completo multi-timeframe para un ticker."""
    ticker:     str
    threshold:  float
    window:     int
    timeframes: dict[str, TimeframeResult]   # keyed by tf name
    profile:    pd.DataFrame   # tabla resumen cross-timeframe
    verdict:    str            # veredicto final sobre la hipótesis


# ── Cálculo de retornos y estadísticas rodantes ──────────────────────────────

def _compute_tf_stats(
    prices: pd.Series,
    tf_days: int,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Calcula retornos a tf_days períodos y estadísticas rodantes.

    Ventana rodante en días calendario independientemente del TF,
    pero min_periods escala proporcionalmente.
    """
    df = pd.DataFrame(index=prices.index)
    df["ret"] = np.log(prices / prices.shift(tf_days))

    min_p = max(30, window // 4)
    df["mu"]    = df["ret"].rolling(window, min_periods=min_p).mean()
    df["sigma"] = df["ret"].rolling(window, min_periods=min_p).std()
    df["z_score"]   = (df["ret"] - df["mu"]) / df["sigma"]
    df["normal_lo"] = df["mu"] - threshold * df["sigma"]
    df["normal_hi"] = df["mu"] + threshold * df["sigma"]

    conditions = [df["z_score"] > threshold, df["z_score"] < -threshold]
    df["event_type"] = np.select(conditions,
                                  ["ANOMALY_HIGH", "ANOMALY_LOW"],
                                  default="NORMAL")
    return df.dropna()


def _yearly_stats(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby(df.index.year):
        mu    = g["ret"].mean()
        sigma = g["ret"].std()
        n_h   = (g["event_type"] == "ANOMALY_HIGH").sum()
        n_l   = (g["event_type"] == "ANOMALY_LOW").sum()
        rows.append({
            "Año":          year,
            "Media":        round(mu, 5),
            "Std":          round(sigma, 5),
            "Rango normal": f"[{mu-threshold*sigma:+.3%}, {mu+threshold*sigma:+.3%}]",
            "↑ anómalos":   int(n_h),
            "↓ anómalos":   int(n_l),
            "% anómalos":   f"{(n_h+n_l)/len(g):.1%}",
        })
    return pd.DataFrame(rows).set_index("Año")


# ── Análisis post-evento ─────────────────────────────────────────────────────

def _post_event_analysis(
    df: pd.DataFrame,
    prices: pd.Series,
    tf_days: int,
    fwd_periods: list[int] = FORWARD_PERIODS,
) -> tuple[dict, pd.DataFrame]:
    """Calcula qué ocurre N períodos (tf_days * N) después de cada evento."""

    event_types  = ["ANOMALY_HIGH", "ANOMALY_LOW", "NORMAL"]
    post_event   = {}
    ttest_rows   = []

    horizons_days = [tf_days * p for p in fwd_periods]

    for h_days in horizons_days:
        rows = []
        fwd_by_type = {et: [] for et in event_types}

        for et in event_types:
            events = df[df["event_type"] == et].index
            fwd_rets = []
            for date in events:
                loc = prices.index.get_loc(date)
                if loc + h_days < len(prices):
                    fwd_ret = np.log(prices.iloc[loc + h_days] / prices.iloc[loc])
                    fwd_rets.append(fwd_ret)
                    fwd_by_type[et].append(fwd_ret)

            if not fwd_rets:
                continue
            arr = np.array(fwd_rets)
            n_up   = (arr >  0.002).sum()
            n_flat = (np.abs(arr) <= 0.002).sum()
            n_down = (arr < -0.002).sum()
            rows.append({
                "Tipo evento":   et,
                "N eventos":     len(arr),
                "Ret medio":     arr.mean(),
                "Ret std":       arr.std(),
                "% Sube":        n_up   / len(arr),
                "% Plano":       n_flat / len(arr),
                "% Baja":        n_down / len(arr),
                "Mejor ret":     arr.max(),
                "Peor ret":      arr.min(),
            })

        if rows:
            post_event[h_days] = pd.DataFrame(rows).set_index("Tipo evento")

        # T-test vs NORMAL
        normal_fwd = fwd_by_type["NORMAL"]
        for et in ["ANOMALY_HIGH", "ANOMALY_LOW"]:
            anom_fwd = fwd_by_type[et]
            if len(anom_fwd) >= 5 and len(normal_fwd) >= 5:
                t_stat, p_val = scipy_stats.ttest_ind(anom_fwd, normal_fwd)
                ttest_rows.append({
                    "Horizonte":           f"+{h_days}d",
                    "Tipo":                et,
                    "N anómalos":          len(anom_fwd),
                    "Ret anómalo":         np.mean(anom_fwd),
                    "Ret normal":          np.mean(normal_fwd),
                    "Diferencia":          np.mean(anom_fwd) - np.mean(normal_fwd),
                    "p-value":             p_val,
                    "Sig. (p<0.05)":       "✓" if p_val < 0.05 else "✗",
                })

    ttest_df = pd.DataFrame(ttest_rows) if ttest_rows else pd.DataFrame()
    return post_event, ttest_df


# ── Pipeline por timeframe ────────────────────────────────────────────────────

def _run_single_tf(
    prices: pd.Series,
    tf_name: str,
    tf_days: int,
    window: int,
    threshold: float,
) -> TimeframeResult:
    df = _compute_tf_stats(prices, tf_days, window, threshold)

    stats = _yearly_stats(df, threshold)

    anom = df[df["event_type"] != "NORMAL"][["ret","z_score","mu","sigma","event_type"]].copy()
    anom.columns = ["Retorno","Z-score","Media_rod","Std_rod","Tipo"]

    post_ev, ttest = _post_event_analysis(df, prices, tf_days)

    # Rango normal representativo (mediana del período)
    norm_lo = float(df["normal_lo"].median())
    norm_hi = float(df["normal_hi"].median())

    return TimeframeResult(
        name=tf_name,
        tf_days=tf_days,
        df=df,
        stats_summary=stats,
        anomalies=anom,
        post_event=post_ev,
        ttest=ttest,
        normal_range=(norm_lo, norm_hi),
    )


# ── Pipeline completo multi-timeframe ────────────────────────────────────────

def run_multitf_trigger(
    prices: pd.Series,
    ticker: str,
    timeframes: Optional[dict] = None,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
) -> MultiTFResult:
    """Ejecuta el análisis de anomalías en múltiples marcos temporales.

    Args:
        prices:     Serie de precios de cierre.
        ticker:     Nombre del activo.
        timeframes: Dict {nombre: días}. Default: diario/semanal/quincenal/mensual.
        window:     Ventana rodante en días (1 año = 252).
        threshold:  Z-score umbral (default 2.0 = ~95% de los días son normales).
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    tf_results = {}
    for tf_name, tf_days in timeframes.items():
        tf_results[tf_name] = _run_single_tf(prices, tf_name, tf_days, window, threshold)

    # Tabla resumen cross-timeframe
    profile_rows = []
    for tf_name, tfr in tf_results.items():
        df   = tfr.df
        n_h  = (df["event_type"] == "ANOMALY_HIGH").sum()
        n_l  = (df["event_type"] == "ANOMALY_LOW").sum()
        total = len(df)

        # ¿La hipótesis se cumple? Contar tests significativos con dirección correcta
        rev_score = 0
        if not tfr.ttest.empty:
            for _, row in tfr.ttest.iterrows():
                if row["Sig. (p<0.05)"] == "✓":
                    if "HIGH" in row["Tipo"] and row["Diferencia"] < 0:
                        rev_score += 1
                    if "LOW" in row["Tipo"] and row["Diferencia"] > 0:
                        rev_score += 1

        profile_rows.append({
            "Timeframe":      tf_name,
            "Rango normal":   f"[{tfr.normal_range[0]:+.3%}, {tfr.normal_range[1]:+.3%}]",
            "Media período":  f"{df['ret'].mean():+.4%}",
            "Std período":    f"{df['ret'].std():.4%}",
            "↑ anómalos":     f"{n_h} ({n_h/total:.1%})",
            "↓ anómalos":     f"{n_l} ({n_l/total:.1%})",
            "Reversión ✓":    rev_score,
        })
    profile = pd.DataFrame(profile_rows).set_index("Timeframe")

    # Veredicto global
    total_rev = profile["Reversión ✓"].sum()
    max_rev   = len(timeframes) * 4  # 4 tests de reversión posibles por TF
    if total_rev >= max_rev * 0.5:
        verdict = f"✓ HIPÓTESIS CONFIRMADA ({total_rev}/{max_rev} tests con reversión significativa)"
    elif total_rev >= max_rev * 0.25:
        verdict = f"~ HIPÓTESIS PARCIAL ({total_rev}/{max_rev} tests con reversión significativa)"
    else:
        verdict = f"✗ HIPÓTESIS DÉBIL ({total_rev}/{max_rev} tests con reversión significativa)"

    return MultiTFResult(
        ticker=ticker,
        threshold=threshold,
        window=window,
        timeframes=tf_results,
        profile=profile,
        verdict=verdict,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_multitf_report(result: MultiTFResult, verbose: bool = True) -> None:
    """Imprime el informe completo multi-timeframe."""

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  TRIGGER ANALYSIS — {result.ticker}")
    print(f"  Ventana: {result.window}d | Umbral anomalía: ±{result.threshold}σ")
    print(sep)

    # Tabla resumen cross-timeframe
    print(f"\n{'─'*72}")
    print("  PERFIL DEL TICKER — Rangos normales por marco temporal")
    print(f"{'─'*72}")
    print(result.profile.to_string())

    # Por cada timeframe
    for tf_name, tfr in result.timeframes.items():
        print(f"\n{'─'*72}")
        print(f"  TIMEFRAME: {tf_name}")
        print(f"{'─'*72}")

        df   = tfr.df
        n_h  = (df["event_type"] == "ANOMALY_HIGH").sum()
        n_l  = (df["event_type"] == "ANOMALY_LOW").sum()

        print(f"  Rango normal del ticker:  [{tfr.normal_range[0]:+.3%} , {tfr.normal_range[1]:+.3%}]")
        print(f"  Anomalías ↑ (sobre rango): {n_h}  ({n_h/len(df):.1%})")
        print(f"  Anomalías ↓ (bajo rango):  {n_l}  ({n_l/len(df):.1%})")

        if verbose:
            print(f"\n  Estadísticas anuales:")
            print(tfr.stats_summary.to_string())

        # Top eventos más extremos
        if not tfr.anomalies.empty:
            print(f"\n  Top 5 eventos más extremos:")
            top5 = tfr.anomalies.reindex(
                tfr.anomalies["Z-score"].abs().nlargest(5).index
            )[["Retorno","Z-score","Tipo"]]
            top5["Retorno"] = top5["Retorno"].map(lambda x: f"{x:+.3%}")
            top5["Z-score"] = top5["Z-score"].map(lambda x: f"{x:+.2f}σ")
            print(top5.to_string())

        # Post-event probabilities
        print(f"\n  ¿Qué ocurre después? (reversión = ↑ HIGH→baja / LOW→sube)")
        for h_days, table in tfr.post_event.items():
            n_periods = h_days // tfr.tf_days
            label = f"+{n_periods} período{'s' if n_periods>1 else ''} (+{h_days}d)"
            print(f"\n    {label}:")
            t = table.copy()
            for col in ["Ret medio","Ret std","Mejor ret","Peor ret"]:
                if col in t.columns:
                    t[col] = t[col].map(lambda x: f"{x:+.3%}")
            for col in ["% Sube","% Plano","% Baja"]:
                if col in t.columns:
                    t[col] = t[col].map(lambda x: f"{x:.0%}")
            print(t.to_string())

        # T-test
        if not tfr.ttest.empty:
            print(f"\n  Validación estadística:")
            t = tfr.ttest.copy()
            for col in ["Ret anómalo","Ret normal","Diferencia"]:
                t[col] = t[col].map(lambda x: f"{x:+.4%}")
            t["p-value"] = t["p-value"].map(lambda x: f"{x:.3f}")
            print(t[["Horizonte","Tipo","N anómalos","Ret anómalo","Ret normal",
                      "Diferencia","p-value","Sig. (p<0.05)"]].to_string(index=False))

    # Veredicto
    print(f"\n{'='*72}")
    print(f"  VEREDICTO FINAL: {result.verdict}")
    print(f"{'='*72}\n")


# ── Función de conveniencia para un solo timeframe (backward compat) ─────────

def run_trigger(
    prices: pd.Series,
    ticker: str,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
) -> "TriggerResult":
    """Backward-compatible wrapper: analiza solo el timeframe diario."""
    from dataclasses import dataclass as _dc

    tf = _run_single_tf(prices, "Diario (1d)", 1, window, threshold)

    # Wrap in old TriggerResult shape (compatible with old code)
    @_dc
    class TriggerResult:
        ticker: str
        df: pd.DataFrame
        stats_summary: pd.DataFrame
        anomalies: pd.DataFrame
        post_event: dict
        ttest: pd.DataFrame
        threshold: float
        window: int
        normal_range: tuple

    return TriggerResult(
        ticker=ticker, df=tf.df, stats_summary=tf.stats_summary,
        anomalies=tf.anomalies, post_event=tf.post_event,
        ttest=tf.ttest, threshold=threshold, window=window,
        normal_range=tf.normal_range,
    )


def print_trigger_report(result) -> None:
    """Backward-compatible: imprime informe del timeframe diario."""
    # Build a fake MultiTFResult with just the daily TF
    fake = MultiTFResult(
        ticker=result.ticker,
        threshold=result.threshold,
        window=result.window,
        timeframes={"Diario (1d)": TimeframeResult(
            name="Diario (1d)", tf_days=1,
            df=result.df, stats_summary=result.stats_summary,
            anomalies=result.anomalies, post_event=result.post_event,
            ttest=result.ttest, normal_range=result.normal_range,
        )},
        profile=pd.DataFrame(),
        verdict="",
    )
    print_multitf_report(fake, verbose=True)
