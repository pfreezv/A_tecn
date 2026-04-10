"""VIX overlay extension for LONQ regime detection.

Two-layer architecture:
  Layer 1 — Base ensemble (KMeans + GMM + HMM) → raw regime
  Layer 2 — VIX overlay → adjusts regime + confidence using macro fear signal

VIX regimes:
  < 15        Low fear    → confirms bull, boosts confidence
  15 – 25     Normal      → neutral, no change
  25 – 35     Elevated    → caution, downgrades bull → sideways
  > 35        Extreme     → override, forces bear regardless of ensemble

VIX spike detection:
  Single-day VIX jump > 25%  → immediate bear override (flash crash signal)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── VIX thresholds ──────────────────────────────────────────────────────────
VIX_LOW      = 15.0   # complacency → bull confirmed
VIX_NORMAL   = 25.0   # neutral zone
VIX_HIGH     = 35.0   # fear → bull downgraded
VIX_SPIKE_PCT = 0.25  # 1-day VIX jump % that triggers bear override


@dataclass
class VixOverlayResult:
    """Output of the VIX overlay layer."""

    # Adjusted signals
    regime:       pd.Series   # Final regime after VIX overlay
    confidence:   pd.Series   # Adjusted confidence
    signal_strength: pd.Series

    # VIX features
    vix_level:    pd.Series   # Raw VIX
    vix_regime:   pd.Series   # low / normal / high / extreme
    vix_pct_252:  pd.Series   # Percentile rank in 252-day window (0–1)
    vix_change_5d: pd.Series  # 5-day VIX change
    vix_spike:    pd.Series   # Boolean: VIX jumped >25% in 1 day

    # What the overlay changed
    overrides:    pd.DataFrame  # Rows where VIX changed the base regime

    # Full DataFrame
    df: pd.DataFrame


# ── Data loading ─────────────────────────────────────────────────────────────

def load_vix(path: str) -> pd.Series:
    """Load VIX CSV with columns Date, Close.

    Returns a Series indexed by date named 'VIX'.
    """
    vix = pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]
    vix.name = "VIX"
    vix = vix.sort_index()
    vix = vix[vix > 0].dropna()
    return vix


# ── VIX feature engineering ──────────────────────────────────────────────────

def build_vix_features(vix: pd.Series) -> pd.DataFrame:
    """Compute VIX-derived features used by the overlay.

    Returns DataFrame with columns:
    - vix_level, vix_regime, vix_pct_252, vix_change_1d, vix_change_5d, vix_spike
    """
    df = pd.DataFrame({"vix_level": vix})

    # Categorical regime
    df["vix_regime"] = pd.cut(
        df["vix_level"],
        bins=[0, VIX_LOW, VIX_NORMAL, VIX_HIGH, np.inf],
        labels=["low", "normal", "high", "extreme"],
    )

    # Rolling percentile rank (252-day window)
    df["vix_pct_252"] = (
        df["vix_level"]
        .rolling(252, min_periods=30)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # Rate of change
    df["vix_change_1d"] = df["vix_level"].pct_change(1)
    df["vix_change_5d"] = df["vix_level"].pct_change(5)

    # Spike flag: single-day jump above threshold
    df["vix_spike"] = df["vix_change_1d"] > VIX_SPIKE_PCT

    return df


# ── Overlay logic ─────────────────────────────────────────────────────────────

def _adjust_regime(
    base_regime: str,
    base_conf: float,
    vix_level: float,
    vix_spike: bool,
) -> tuple[str, float, str]:
    """Apply VIX rules to a single day. Returns (regime, confidence, reason)."""

    # VIX spike → immediate bear override
    if vix_spike:
        return "bear", 1.0, f"VIX spike (>{VIX_SPIKE_PCT:.0%} jump)"

    # Extreme fear → force bear
    if vix_level > VIX_HIGH:
        return "bear", 1.0, f"VIX extreme ({vix_level:.1f}>{VIX_HIGH})"

    # High fear → downgrade bull to sideways
    if vix_level > VIX_NORMAL and base_regime == "bull":
        new_conf = max(0.33, base_conf - 0.34)
        return "sideways", new_conf, f"VIX high ({vix_level:.1f}) downgraded bull"

    # Low fear → boost bull confidence
    if vix_level < VIX_LOW and base_regime == "bull":
        new_conf = min(1.0, base_conf + 0.33)
        return "bull", new_conf, f"VIX low ({vix_level:.1f}) confirmed bull"

    # Neutral — no change
    return base_regime, base_conf, ""


def apply_vix_overlay(
    base_regime: pd.Series,
    base_confidence: pd.Series,
    vix: pd.Series,
) -> VixOverlayResult:
    """Apply VIX overlay to base ensemble output.

    Args:
        base_regime:     Consensus regime from ensemble ('bull','bear','sideways').
        base_confidence: Confidence from ensemble (0.33–1.0).
        vix:             VIX closing price series.

    Returns VixOverlayResult with adjusted signals and full diagnostics.
    """
    # Align all series
    common = base_regime.index.intersection(base_confidence.index).intersection(vix.index)
    br  = base_regime.loc[common]
    bc  = base_confidence.loc[common]
    vix = vix.loc[common]

    vix_feat = build_vix_features(vix)

    new_regimes    = []
    new_confs      = []
    override_rows  = []

    for date in common:
        orig_r = br[date]
        orig_c = bc[date]
        vl     = vix[date]
        spike  = bool(vix_feat.loc[date, "vix_spike"]) if date in vix_feat.index else False

        adj_r, adj_c, reason = _adjust_regime(orig_r, orig_c, vl, spike)

        new_regimes.append(adj_r)
        new_confs.append(adj_c)

        if reason:  # something changed
            override_rows.append({
                "date":        date,
                "base_regime": orig_r,
                "vix_regime":  adj_r,
                "vix_level":   round(vl, 2),
                "reason":      reason,
            })

    regime_s = pd.Series(new_regimes, index=common, name="VIX_Regime")
    conf_s   = pd.Series(new_confs,   index=common, name="VIX_Confidence")
    strength_s = conf_s.map(
        lambda c: "strong" if c == 1.0 else ("moderate" if c >= 0.67 else "weak")
    ).rename("VIX_SignalStrength")

    overrides_df = pd.DataFrame(override_rows).set_index("date") if override_rows else pd.DataFrame()

    # Full merged DataFrame
    df_out = pd.DataFrame({
        "Base_Regime":    br,
        "Base_Conf":      bc,
        "VIX":            vix,
        "VIX_Regime_Cat": vix_feat["vix_regime"].reindex(common),
        "VIX_Pct":        vix_feat["vix_pct_252"].reindex(common),
        "VIX_Change_5d":  vix_feat["vix_change_5d"].reindex(common),
        "VIX_Spike":      vix_feat["vix_spike"].reindex(common),
        "Final_Regime":   regime_s,
        "Final_Conf":     conf_s,
        "Signal_Strength": strength_s,
    })

    return VixOverlayResult(
        regime=regime_s,
        confidence=conf_s,
        signal_strength=strength_s,
        vix_level=vix,
        vix_regime=vix_feat["vix_regime"].reindex(common),
        vix_pct_252=vix_feat["vix_pct_252"].reindex(common),
        vix_change_5d=vix_feat["vix_change_5d"].reindex(common),
        vix_spike=vix_feat["vix_spike"].reindex(common),
        overrides=overrides_df,
        df=df_out,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_vix_summary(result: VixOverlayResult) -> None:
    """Print VIX overlay summary to console."""
    print("\n" + "=" * 60)
    print("CAPA VIX — Overlay de miedo de mercado")
    print("=" * 60)

    vix = result.vix_level
    print(f"\n--- Estadísticas VIX ---")
    print(f"  Nivel actual:    {vix.iloc[-1]:.1f}")
    print(f"  Promedio:        {vix.mean():.1f}")
    print(f"  Máximo:          {vix.max():.1f}  ({vix.idxmax().date()})")
    print(f"  Mínimo:          {vix.min():.1f}  ({vix.idxmin().date()})")

    print(f"\n--- Distribución régimen VIX ---")
    dist = result.vix_regime.value_counts(normalize=True).mul(100).round(1)
    labels = {"low": f"<{VIX_LOW}", "normal": f"{VIX_LOW}–{VIX_NORMAL}",
              "high": f"{VIX_NORMAL}–{VIX_HIGH}", "extreme": f">{VIX_HIGH}"}
    for reg in ["low", "normal", "high", "extreme"]:
        if reg in dist:
            pct = dist[reg]
            bar = "█" * int(pct / 3)
            print(f"  {reg:8s} (VIX {labels[reg]:>8}) {pct:5.1f}%  {bar}")

    print(f"\n--- Overrides aplicados: {len(result.overrides)} días ---")
    if not result.overrides.empty:
        changes = result.overrides["reason"].value_counts()
        for reason, count in changes.items():
            print(f"  {count:4d} días — {reason}")

    print(f"\n--- Comparativa base vs +VIX ---")
    changed = (result.df["Base_Regime"] != result.df["Final_Regime"]).sum()
    total   = len(result.df)
    print(f"  Días con régimen cambiado: {changed} / {total}  ({changed/total:.1%})")

    print(f"\n--- Distribución final (post-VIX) ---")
    dist2 = result.regime.value_counts(normalize=True).mul(100).round(1)
    for reg, pct in dist2.items():
        bar = "█" * int(pct / 2)
        print(f"  {reg:10s} {pct:5.1f}%  {bar}")
