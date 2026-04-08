"""Trading signal generation from regime consensus.

Signal rules:
- BUY  : consensus turns to 'bull' AND confidence >= buy_threshold (2/3 modelos de acuerdo)
- SELL : consensus turns to 'bear' with confidence >= sell_threshold,
         OR leaves 'bull' regime with confidence >= sell_threshold
- HOLD : todo lo demás

Each signal is delayed 1 day to avoid look-ahead bias (we act on next open).
"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class SignalResult:
    """Container for all signal outputs."""
    signals: pd.DataFrame    # Date-indexed: Signal, Reason, Position, Close, Consensus
    trades: pd.DataFrame     # One row per completed (or open) trade
    summary: dict = field(default_factory=dict)


def generate_signals(
    consensus: pd.Series,
    confidence: pd.Series,
    prices: pd.Series,
    buy_threshold: float = 0.67,
    sell_threshold: float = 0.33,
) -> SignalResult:
    """Generate BUY/SELL/HOLD signals from ensemble regime output.

    Args:
        consensus:       Regime label per day ('bull', 'bear', 'sideways').
        confidence:      Fraction of models in agreement (0.33–1.0).
        prices:          Close price series aligned with consensus.
        buy_threshold:   Min confidence to trigger a BUY (default: 0.67 = 2/3 models).
        sell_threshold:  Min confidence to trigger a SELL (default: 0.33 = any model).
    """
    df = pd.DataFrame({
        "Close": prices,
        "Consensus": consensus,
        "Confidence": confidence,
    }).dropna()

    prev = df["Consensus"].shift(1)

    # --- Signal conditions ---
    bull_entry = (
        (df["Consensus"] == "bull") &
        (prev != "bull") &
        (df["Confidence"] >= buy_threshold)
    )
    bear_signal = (
        (df["Consensus"] == "bear") &
        (df["Confidence"] >= sell_threshold)
    )
    bull_exit = (
        (prev == "bull") &
        (df["Consensus"] != "bull") &
        (df["Confidence"] >= sell_threshold)
    )

    df["Signal"] = "HOLD"
    df.loc[bull_entry, "Signal"] = "BUY"
    df.loc[bear_signal | bull_exit, "Signal"] = "SELL"

    # --- Human-readable reason ---
    reasons = []
    for idx in df.index:
        sig = df.at[idx, "Signal"]
        cons = df.at[idx, "Consensus"]
        conf = df.at[idx, "Confidence"]
        prev_c = prev.get(idx, "?")
        if sig == "BUY":
            reasons.append(f"Régimen → bull  (conf {conf:.0%}, venía de {prev_c})")
        elif sig == "SELL":
            if cons == "bear":
                reasons.append(f"Régimen bear  (conf {conf:.0%})")
            else:
                reasons.append(f"Salida bull → {cons}  (conf {conf:.0%})")
        else:
            reasons.append("")
    df["Reason"] = reasons

    # --- Position tracking (1 = in market, 0 = cash) ---
    position = 0
    positions = []
    for sig in df["Signal"]:
        if sig == "BUY":
            position = 1
        elif sig == "SELL":
            position = 0
        positions.append(position)
    df["Position"] = positions

    # --- Trade log ---
    trades = []
    entry_date = entry_price = None

    for date, row in df.iterrows():
        if row["Signal"] == "BUY" and entry_date is None:
            entry_date = date
            entry_price = row["Close"]
        elif row["Signal"] == "SELL" and entry_date is not None:
            ret = (row["Close"] / entry_price) - 1
            trades.append({
                "Entrada":      entry_date.date(),
                "Salida":       date.date(),
                "P. Entrada":   round(entry_price, 2),
                "P. Salida":    round(row["Close"], 2),
                "Retorno":      f"{ret:+.2%}",
                "Días":         (date - entry_date).days,
                "Resultado":    "✓ WIN" if ret > 0 else "✗ LOSS",
            })
            entry_date = entry_price = None

    # Posición aún abierta
    if entry_date is not None:
        last = df.iloc[-1]
        ret = (last["Close"] / entry_price) - 1
        trades.append({
            "Entrada":      entry_date.date(),
            "Salida":       "ABIERTA",
            "P. Entrada":   round(entry_price, 2),
            "P. Salida":    round(last["Close"], 2),
            "Retorno":      f"{ret:+.2%}",
            "Días":         (df.index[-1] - entry_date).days,
            "Resultado":    "→ ABIERTA",
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # --- Summary stats ---
    buy_count = (df["Signal"] == "BUY").sum()
    completed = [t for t in trades if t["Resultado"] != "→ ABIERTA"]
    wins = sum(1 for t in completed if "WIN" in t["Resultado"])
    summary = {
        "Total señales BUY":   int(buy_count),
        "Operaciones cerradas": len(completed),
        "Win rate":            f"{wins/len(completed):.0%}" if completed else "N/A",
        "Días en mercado":     int(df["Position"].sum()),
        "% tiempo en mercado": f"{df['Position'].mean():.1%}",
    }

    return SignalResult(signals=df, trades=trades_df, summary=summary)


def print_signals(result: SignalResult, last_n: int = 15) -> None:
    """Pretty-print signal results to console."""
    print("\n" + "=" * 60)
    print("SEÑALES DE COMPRA / VENTA")
    print("=" * 60)

    # Resumen
    for k, v in result.summary.items():
        print(f"  {k:<28} {v}")

    # Últimas señales BUY/SELL
    active = result.signals[result.signals["Signal"] != "HOLD"].tail(last_n)
    if not active.empty:
        print("\n--- Últimas señales activas ---")
        print(active[["Close", "Consensus", "Confidence", "Signal", "Reason"]].to_string())

    # Log de operaciones
    if not result.trades.empty:
        print("\n--- Log de operaciones ---")
        print(result.trades.to_string(index=False))

    # Señal más reciente
    last_sig = result.signals.iloc[-1]
    print(f"\n--- SEÑAL HOY ({result.signals.index[-1].date()}) ---")
    print(f"  Régimen:    {last_sig['Consensus']}")
    print(f"  Confianza:  {last_sig['Confidence']:.0%}")
    print(f"  Señal:      {last_sig['Signal']}")
    print(f"  Posición:   {'EN MERCADO' if last_sig['Position'] == 1 else 'EN CASH'}")
    if last_sig["Reason"]:
        print(f"  Motivo:     {last_sig['Reason']}")
