"""Trigger module: mean-reversion anomaly detector.

Hipótesis: los precios tienden a volver a su nivel basal después de movimientos
anómalos (retornos fuera del rango estadísticamente "normal").

Metodología:
  1. Calcular retorno diario: ret_t = ln(P_t / P_{t-1})
  2. Para cada día, usar una ventana rodante de 1 año (252 días) para obtener:
       - media (mu) y desviación estándar (sigma) de los retornos
  3. Definir rangos normales: [mu - k*sigma, mu + k*sigma]  (default k=2)
  4. Clasificar cada día:
       - NORMAL:         |z| <= threshold
       - ANOMALY_HIGH:   z  >  threshold  (subida anómala)
       - ANOMALY_LOW:    z  < -threshold  (bajada anómala)
  5. Para cada evento anómalo, mirar hacia adelante N días y calcular:
       - ¿El precio subió, bajó, o quedó plano?
       - Retorno medio post-evento vs días normales
       → Esto valida o refuta la hipótesis de reversión a la media
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── Configuración por defecto ────────────────────────────────────────────────
WINDOW       = 252          # días en ventana rodante (1 año de trading)
THRESHOLD    = 2.0          # z-score para considerar anómalo
HORIZONS     = [1, 5, 10, 20]  # días hacia adelante para analizar


@dataclass
class TriggerResult:
    """Resultado completo del análisis de anomalías."""

    ticker: str

    # Serie de retornos con estadísticas rodantes
    df: pd.DataFrame           # columnas: ret, mu, sigma, z_score, event_type

    # Tabla de estadísticas base
    stats_summary: pd.DataFrame   # por año: media, std, rango normal

    # Eventos anómalos
    anomalies: pd.DataFrame    # log de cada evento (fecha, retorno, z-score, tipo)

    # Análisis post-evento → el núcleo de la hipótesis
    post_event: dict           # {horizont: DataFrame con probabilidades y retornos medios}

    # Validación estadística
    ttest: pd.DataFrame        # p-values: ¿retornos post-anomalía difieren significativamente?

    threshold: float
    window: int


# ── Cálculo de estadísticas rodantes ────────────────────────────────────────

def compute_return_stats(
    prices: pd.Series,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Calcula retornos diarios y estadísticas rodantes.

    Returns DataFrame con:
    - ret:        retorno log diario
    - mu:         media rodante (window días)
    - sigma:      desviación estándar rodante
    - z_score:    (ret - mu) / sigma
    - normal_lo:  límite inferior del rango normal (mu - k*sigma)
    - normal_hi:  límite superior del rango normal (mu + k*sigma)
    - event_type: NORMAL / ANOMALY_HIGH / ANOMALY_LOW
    """
    df = pd.DataFrame(index=prices.index)
    df["ret"] = np.log(prices / prices.shift(1))

    df["mu"]    = df["ret"].rolling(window, min_periods=30).mean()
    df["sigma"] = df["ret"].rolling(window, min_periods=30).std()

    df["z_score"]   = (df["ret"] - df["mu"]) / df["sigma"]
    df["normal_lo"] = df["mu"] - threshold * df["sigma"]
    df["normal_hi"] = df["mu"] + threshold * df["sigma"]

    # Clasificación
    conditions = [
        df["z_score"] >  threshold,
        df["z_score"] < -threshold,
    ]
    choices = ["ANOMALY_HIGH", "ANOMALY_LOW"]
    df["event_type"] = np.select(conditions, choices, default="NORMAL")

    return df.dropna()


# ── Estadísticas por año ─────────────────────────────────────────────────────

def yearly_stats(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Resumen por año: media, std, rango normal, número de anomalías."""
    rows = []
    for year, g in df.groupby(df.index.year):
        mu    = g["ret"].mean()
        sigma = g["ret"].std()
        n_high = (g["event_type"] == "ANOMALY_HIGH").sum()
        n_low  = (g["event_type"] == "ANOMALY_LOW").sum()
        rows.append({
            "Año":          year,
            "Media diaria": f"{mu:.4f}",
            "Std diaria":   f"{sigma:.4f}",
            "Rango normal": f"[{mu - threshold*sigma:.3f}, {mu + threshold*sigma:.3f}]",
            "Anomalías ↑":  int(n_high),
            "Anomalías ↓":  int(n_low),
            "Total eventos": int(n_high + n_low),
            "% anómalos":   f"{(n_high + n_low) / len(g):.1%}",
        })
    return pd.DataFrame(rows).set_index("Año")


# ── Análisis post-evento ─────────────────────────────────────────────────────

def analyze_post_event(
    df: pd.DataFrame,
    prices: pd.Series,
    horizons: list[int] = HORIZONS,
) -> tuple[dict, pd.DataFrame]:
    """Para cada tipo de evento, calcula qué sucede en los N días siguientes.

    Returns:
        post_event: dict {horizon: DataFrame} con probabilidades y retornos medios
        ttest:      DataFrame con p-values comparando anomalías vs días normales
    """
    event_types = ["ANOMALY_HIGH", "ANOMALY_LOW", "NORMAL"]
    post_event  = {}

    for h in horizons:
        rows = []
        for etype in event_types:
            mask   = df["event_type"] == etype
            events = df[mask].index

            fwd_rets = []
            for date in events:
                loc = prices.index.get_loc(date)
                if loc + h < len(prices):
                    fwd_ret = np.log(prices.iloc[loc + h] / prices.iloc[loc])
                    fwd_rets.append(fwd_ret)

            if not fwd_rets:
                continue

            arr = np.array(fwd_rets)
            n_up   = (arr > 0.002).sum()   # sube > 0.2%
            n_flat = (np.abs(arr) <= 0.002).sum()
            n_down = (arr < -0.002).sum()

            rows.append({
                "Tipo evento":   etype,
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
            post_event[h] = pd.DataFrame(rows).set_index("Tipo evento")

    # T-test: ¿retornos post-anomalía son estadísticamente distintos a los normales?
    ttest_rows = []
    for h in horizons:
        for etype in ["ANOMALY_HIGH", "ANOMALY_LOW"]:
            normal_fwd, anom_fwd = [], []
            for date, row in df.iterrows():
                loc = prices.index.get_loc(date)
                if loc + h >= len(prices):
                    continue
                fwd = np.log(prices.iloc[loc + h] / prices.iloc[loc])
                if row["event_type"] == etype:
                    anom_fwd.append(fwd)
                elif row["event_type"] == "NORMAL":
                    normal_fwd.append(fwd)

            if len(anom_fwd) >= 5 and len(normal_fwd) >= 5:
                t_stat, p_val = scipy_stats.ttest_ind(anom_fwd, normal_fwd)
                ttest_rows.append({
                    "Horizonte": f"{h}d",
                    "Tipo":      etype,
                    "N anómalos": len(anom_fwd),
                    "Ret medio anómalo": np.mean(anom_fwd),
                    "Ret medio normal":  np.mean(normal_fwd),
                    "t-stat": t_stat,
                    "p-value": p_val,
                    "Significativo": "✓" if p_val < 0.05 else "✗",
                })

    ttest_df = pd.DataFrame(ttest_rows) if ttest_rows else pd.DataFrame()
    return post_event, ttest_df


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_trigger(
    prices: pd.Series,
    ticker: str,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
    horizons: list[int] = HORIZONS,
) -> TriggerResult:
    """Ejecuta el análisis completo de anomalías y reversión a la media.

    Args:
        prices:    Serie de precios de cierre (DatetimeIndex).
        ticker:    Nombre del activo para informes.
        window:    Ventana rodante en días (default 252 = 1 año).
        threshold: Z-score para considerar anómalo (default 2.0).
        horizons:  Días hacia adelante para analizar (default [1,5,10,20]).
    """
    # 1. Estadísticas rodantes
    df = compute_return_stats(prices, window, threshold)

    # 2. Estadísticas anuales
    stats = yearly_stats(df, threshold)

    # 3. Log de anomalías
    anom_mask = df["event_type"] != "NORMAL"
    anomalies = df[anom_mask][["ret", "z_score", "mu", "sigma", "event_type"]].copy()
    anomalies.columns = ["Retorno", "Z-score", "Media_rodante", "Std_rodante", "Tipo"]

    # 4. Análisis post-evento
    post_event, ttest = analyze_post_event(df, prices, horizons)

    return TriggerResult(
        ticker=ticker,
        df=df,
        stats_summary=stats,
        anomalies=anomalies,
        post_event=post_event,
        ttest=ttest,
        threshold=threshold,
        window=window,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_trigger_report(result: TriggerResult) -> None:
    """Imprime el informe completo del análisis de triggers."""

    print("\n" + "=" * 70)
    print(f"  TRIGGER ANALYSIS — {result.ticker}")
    print(f"  Hipótesis: reversión a la media tras movimientos anómalos")
    print(f"  Ventana: {result.window} días | Umbral: ±{result.threshold}σ")
    print("=" * 70)

    # Stats globales
    df = result.df
    print(f"\n--- Estadísticas globales del período ---")
    print(f"  Período:          {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Total días:       {len(df)}")
    print(f"  Media diaria:     {df['ret'].mean():.5f}  ({df['ret'].mean()*252:.2%} anualizada)")
    print(f"  Std diaria:       {df['ret'].std():.5f}  ({df['ret'].std()*np.sqrt(252):.2%} anualizada)")
    print(f"  Rango normal:     [{df['normal_lo'].mean():.4f}, {df['normal_hi'].mean():.4f}]  (promedio)")

    total = len(df)
    n_high = (df["event_type"] == "ANOMALY_HIGH").sum()
    n_low  = (df["event_type"] == "ANOMALY_LOW").sum()
    print(f"\n  Anomalías ↑:      {n_high}  ({n_high/total:.1%} de los días)")
    print(f"  Anomalías ↓:      {n_low}   ({n_low/total:.1%} de los días)")
    print(f"  Días normales:    {total - n_high - n_low}  ({(total-n_high-n_low)/total:.1%})")

    # Tabla anual
    print(f"\n--- Estadísticas por año ---")
    print(result.stats_summary.to_string())

    # Top 10 anomalías más extremas
    print(f"\n--- Top 10 movimientos más extremos ---")
    top = result.anomalies.reindex(
        result.anomalies["Z-score"].abs().nlargest(10).index
    )[["Retorno", "Z-score", "Tipo"]]
    top["Retorno"] = top["Retorno"].map(lambda x: f"{x:+.3%}")
    top["Z-score"] = top["Z-score"].map(lambda x: f"{x:+.2f}σ")
    print(top.to_string())

    # Análisis post-evento — el núcleo
    print(f"\n--- ¿Qué ocurre DESPUÉS de un movimiento anómalo? ---")
    print(f"    (Hipótesis de reversión: ANOMALY_HIGH debería tener % Baja alto)")
    print(f"    (y ANOMALY_LOW debería tener % Sube alto)\n")

    for h, table in result.post_event.items():
        print(f"  Horizonte +{h} días:")
        t = table.copy()
        for col in ["Ret medio", "Ret std", "Mejor ret", "Peor ret"]:
            if col in t.columns:
                t[col] = t[col].map(lambda x: f"{x:+.3%}")
        for col in ["% Sube", "% Plano", "% Baja"]:
            if col in t.columns:
                t[col] = t[col].map(lambda x: f"{x:.0%}")
        print(t.to_string())
        print()

    # Validación estadística
    if not result.ttest.empty:
        print(f"--- Validación estadística (t-test vs días normales) ---")
        print(f"    ✓ = diferencia estadísticamente significativa (p < 0.05)\n")
        t = result.ttest.copy()
        for col in ["Ret medio anómalo", "Ret medio normal"]:
            t[col] = t[col].map(lambda x: f"{x:+.4%}")
        t["p-value"] = t["p-value"].map(lambda x: f"{x:.3f}")
        t["t-stat"]  = t["t-stat"].map(lambda x: f"{x:.2f}")
        print(t.to_string(index=False))

    # Veredicto final
    print(f"\n--- VEREDICTO sobre la hipótesis ---")
    sig_count = 0
    reversion_count = 0
    if not result.ttest.empty:
        sig = result.ttest[result.ttest["Significativo"] == "✓"]
        sig_count = len(sig)
        for _, row in sig.iterrows():
            if "HIGH" in row["Tipo"] and row["Ret medio anómalo"] < row["Ret medio normal"]:
                reversion_count += 1
            if "LOW" in row["Tipo"] and row["Ret medio anómalo"] > row["Ret medio normal"]:
                reversion_count += 1

    total_tests = len(result.ttest) if not result.ttest.empty else 1
    print(f"  Tests significativos: {sig_count} / {total_tests}")
    print(f"  Con dirección de reversión: {reversion_count} / {sig_count if sig_count else 1}")
    if reversion_count > sig_count / 2:
        print(f"\n  ✓ HIPÓTESIS APOYADA: los movimientos anómalos tienden a revertir")
    else:
        print(f"\n  ✗ HIPÓTESIS DÉBIL: no hay evidencia clara de reversión consistente")
