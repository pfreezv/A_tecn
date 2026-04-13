"""LONQ — Dashboard visual para Discord.

Genera embeds detallados por cada ticker seleccionado por el modelo
y los envía al webhook de Discord en lotes.

Cada dashboard incluye:
  - Score y desglose de puntuación con barras visuales
  - Precio actual y rango de precios (soporte/resistencia)
  - Señal de acción: COMPRAR / VENDER / MANTENER / VIGILAR / PRECAUCIÓN
  - Métricas profundas si están disponibles

Flujo:
  1. Lee el CSV de resultados del screener
  2. Descarga precios en vivo para los tickers seleccionados
  3. Computa señal (z-score) y rango de precios por ticker
  4. Genera embed resumen global + embeds individuales
  5. Envía todo al webhook de Discord en lotes
"""

from __future__ import annotations

import json
import subprocess
import datetime
import time
from typing import Optional

import numpy as np
import pandas as pd


# ── Colores por grado ────────────────────────────────────────────────────────

COLORS = {
    "✓✓": 0x2ecc71,  # verde — excelente
    "✓ ": 0xf1c40f,  # dorado — bueno
    "~ ": 0xe67e22,  # naranja — marginal
    "✗ ": 0xe74c3c,  # rojo — no apto
}

GRADE_EMOJI = {
    "✓✓": "🟢",
    "✓ ": "🟡",
    "~ ": "🟠",
    "✗ ": "🔴",
}

# ── Barras visuales ──────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 10) -> str:
    """Barra visual con bloques Unicode."""
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def _score_bar(score: int, width: int = 15) -> str:
    """Barra del score global con bloques llenos/vacíos."""
    filled = int(round(score / 100 * width))
    return "▰" * filled + "▱" * (width - filled)


# ── Reconstruir desglose de puntuación desde métricas CSV ────────────────────

def _parse_pct(val) -> Optional[float]:
    """Parsea '22.1%' → 0.221 o devuelve None."""
    s = str(val).strip()
    if s in ("—", "", "nan", "None"):
        return None
    s = s.replace("%", "")
    try:
        v = float(s)
        return v / 100 if "%" in str(val) else v
    except ValueError:
        return None


def _parse_float(val) -> Optional[float]:
    """Parsea un float o devuelve None."""
    s = str(val).strip()
    if s in ("—", "", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _reconstruct_breakdown(row: pd.Series) -> dict:
    """Reconstruye el score_breakdown a partir de las métricas del CSV.

    Usa la misma lógica que fast_screen() en src/screener.py.
    """
    vol = _parse_pct(row.get("Vol. anual", 0)) or 0.0
    ac5 = _parse_float(row.get("Autocorr 5d", 0)) or 0.0
    tr2 = _parse_float(row.get("Tendencia R²", 0)) or 0.0
    n   = int(row.get("Días", 0))

    bd = {}

    # 1. Histórico (0-10)
    if n >= 1000:
        bd["Histórico"] = (10, 10)
    elif n >= 500:
        bd["Histórico"] = (5, 10)
    else:
        bd["Histórico"] = (0, 10)

    # 2. Volatilidad (0-25)
    if vol > 0.55:
        bd["Volatilidad"] = (0, 25)
    elif 0.10 <= vol <= 0.28:
        bd["Volatilidad"] = (25, 25)
    elif vol < 0.10:
        bd["Volatilidad"] = (8, 25)
    else:
        bd["Volatilidad"] = (12, 25)

    # 3. Reversión autocorr (0-30)
    if ac5 < -0.08:
        bd["Reversión"] = (30, 30)
    elif ac5 < -0.03:
        bd["Reversión"] = (18, 30)
    elif ac5 < 0.03:
        bd["Reversión"] = (8, 30)
    else:
        bd["Reversión"] = (0, 30)

    # 4. Anti-tendencia (0-20)
    if tr2 < 0.50:
        bd["Anti-tend."] = (20, 20)
    elif tr2 < 0.70:
        bd["Anti-tend."] = (12, 20)
    elif tr2 < 0.85:
        bd["Anti-tend."] = (5, 20)
    else:
        bd["Anti-tend."] = (0, 20)

    # 5. Consistencia (0-15) — deducida del total
    known = sum(v for v, _ in bd.values())
    score = int(row.get("Score", 0))
    # En modo deep, el score puede tener bonuses, así que cap a 15
    consistency = min(max(score - known, 0), 15)
    bd["Consistencia"] = (consistency, 15)

    return bd


# ── Señal en vivo y rango de precios ─────────────────────────────────────────

ACTION_MAP = {
    "COMPRAR":    {"emoji": "🟢", "label": "COMPRAR",    "color": 0x2ecc71},
    "VENDER":     {"emoji": "🔴", "label": "VENDER",     "color": 0xe74c3c},
    "MANTENER":   {"emoji": "⏸️", "label": "MANTENER",   "color": 0x3498db},
    "VIGILAR":    {"emoji": "👀", "label": "VIGILAR",    "color": 0x9b59b6},
    "PRECAUCIÓN": {"emoji": "⚠️", "label": "PRECAUCIÓN", "color": 0xf39c12},
}


def compute_signal_snapshot(ticker: str, vol_annual: float = 0.0) -> Optional[dict]:
    """Calcula señal de acción y rango de precios a partir de datos en vivo.

    Usa la misma lógica del trigger (z-score sobre retornos rodantes) para
    determinar si el precio está en zona de anomalía.

    Returns:
        dict con: current_price, support, resistance, z_score,
                  action, action_emoji, action_detail, target, stop_loss
        None si no se pudieron obtener datos.
    """
    try:
        import yfinance as yf
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty:
            return None

        # Manejar MultiIndex de yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        prices = data["Close"].dropna()
        if len(prices) < 60:
            return None

        price = float(prices.iloc[-1])

        # Retornos diarios logarítmicos
        ret = np.log(prices / prices.shift(1)).dropna()

        # Estadísticas rodantes (252d = 1 año, como en trigger.py)
        window = min(252, len(ret) - 1)
        mu    = float(ret.rolling(window, min_periods=30).mean().iloc[-1])
        sigma = float(ret.rolling(window, min_periods=30).std().iloc[-1])

        if sigma < 1e-9:
            return None

        # Z-score del último retorno
        z = float((ret.iloc[-1] - mu) / sigma)

        # Rango de precios mensual (±2σ sobre 21 días)
        monthly_vol = sigma * np.sqrt(21)
        support    = round(price * np.exp(-2 * monthly_vol), 2)
        resistance = round(price * np.exp(+2 * monthly_vol), 2)

        # Probabilidades históricas de reversión (simplificado)
        z_series = (ret - ret.rolling(window, min_periods=30).mean()) / ret.rolling(window, min_periods=30).std()
        z_series = z_series.dropna()

        # Retornos forward +5 días después de anomalías
        fwd_5d = np.log(prices / prices.shift(-5)).dropna() * -1  # forward return
        common = z_series.index.intersection(fwd_5d.index)

        p_rev_high = p_rev_low = 0.5
        mean_ret_high = mean_ret_low = 0.0

        if len(common) > 50:
            z_c   = z_series.loc[common]
            fwd_c = fwd_5d.loc[common]

            high_fwd = fwd_c[z_c > 2.0]
            low_fwd  = fwd_c[z_c < -2.0]

            if len(high_fwd) > 3:
                p_rev_high    = float((high_fwd < -0.002).mean())
                mean_ret_high = float(high_fwd.mean())
            if len(low_fwd) > 3:
                p_rev_low    = float((low_fwd > 0.002).mean())
                mean_ret_low = float(low_fwd.mean())

        # Determinar acción según z-score
        if z > 2.0:
            action = "VENDER"
            detail = (f"Subida anómala (z={z:+.1f}σ) — "
                      f"P(baja) histórica: {p_rev_high:.0%}")
            target    = round(price * np.exp(mean_ret_high), 2) if mean_ret_high else None
            stop_loss = round(price * (1 + sigma), 2)
        elif z < -2.0:
            action = "COMPRAR"
            detail = (f"Caída anómala (z={z:+.1f}σ) — "
                      f"P(sube) histórica: {p_rev_low:.0%}")
            target    = round(price * np.exp(mean_ret_low), 2) if mean_ret_low else None
            stop_loss = round(price * (1 - sigma), 2)
        elif z > 1.5:
            action = "PRECAUCIÓN"
            detail = f"Precio cerca de resistencia (z={z:+.1f}σ)"
            target = stop_loss = None
        elif z < -1.5:
            action = "VIGILAR"
            detail = f"Precio acercándose a soporte (z={z:+.1f}σ) — preparar entrada"
            target = stop_loss = None
        else:
            action = "MANTENER"
            detail = f"Precio en rango normal (z={z:+.1f}σ)"
            target = stop_loss = None

        info = ACTION_MAP[action]

        return {
            "current_price":  price,
            "support":        support,
            "resistance":     resistance,
            "support_pct":    (support / price - 1),
            "resistance_pct": (resistance / price - 1),
            "z_score":        z,
            "action":         action,
            "action_emoji":   info["emoji"],
            "action_detail":  detail,
            "target":         target,
            "stop_loss":      stop_loss,
        }

    except Exception as e:
        print(f"  ⚠ Signal snapshot {ticker}: {e}")
        return None


def fetch_live_signals(tickers: list[str]) -> dict[str, dict]:
    """Descarga precios y calcula señales para una lista de tickers.

    Returns:
        dict {ticker: signal_snapshot} para los que se pudieron computar.
    """
    signals = {}
    for tk in tickers:
        snap = compute_signal_snapshot(tk)
        if snap is not None:
            signals[tk] = snap
    return signals


# ── Embed individual por ticker ──────────────────────────────────────────────

def build_ticker_embed(row: pd.Series, signal: Optional[dict] = None) -> dict:
    """Genera un Discord embed detallado para un ticker.

    Args:
        row:    Fila del CSV de resultados con métricas del screener.
        signal: Dict de compute_signal_snapshot() con precio, rango y acción.
    """
    ticker = row["Ticker"]
    score  = int(row["Score"])
    grade  = str(row.get("Grado", "✗ ")).strip()
    # Normalizar grado a 2 caracteres
    if grade == "✓✓":
        pass
    elif grade.startswith("✓"):
        grade = "✓ "
    elif grade.startswith("~"):
        grade = "~ "
    else:
        grade = "✗ "

    color = COLORS.get(grade, 0x95a5a6)
    emoji = GRADE_EMOJI.get(grade, "⚪")

    # Barra principal
    main_bar = _score_bar(score)

    # Recomendación basada en grado
    rec_map = {
        "✓✓": "Excelente candidato",
        "✓ ": "Buen candidato",
        "~ ": "Candidato marginal",
        "✗ ": "No recomendado",
    }
    rec = rec_map.get(grade, "—")

    # ── Descripción con barra principal ──
    description = f"```\n{main_bar}  {score}/100\n```\n{emoji} **{rec}**"

    # ── Desglose de puntuación ──
    bd = _reconstruct_breakdown(row)
    breakdown_lines = []
    for name, (pts, max_pts) in bd.items():
        bar = _bar(pts, max_pts, 8)
        breakdown_lines.append(f"{name:<13} {bar} {pts:>2}/{max_pts}")
    breakdown_text = "```\n" + "\n".join(breakdown_lines) + "\n```"

    fields = [
        {"name": "📊 Desglose", "value": breakdown_text, "inline": False},
    ]

    # ── Precio actual + Rango de precios + Acción ──
    if signal:
        p  = signal["current_price"]
        sp = signal["support"]
        rs = signal["resistance"]
        sp_pct = signal["support_pct"]
        rs_pct = signal["resistance_pct"]
        z  = signal["z_score"]

        # Visualizar posición del precio dentro del rango
        # Mapear z de [-3, +3] → posición en barra [0, 10]
        z_clamped = max(-3, min(3, z))
        pos = int(round((z_clamped + 3) / 6 * 10))
        range_bar = "░" * pos + "●" + "░" * (10 - pos)

        price_text = (
            f"```\n"
            f"💰 Precio:      ${p:,.2f}\n"
            f"📉 Soporte:     ${sp:,.2f}  ({sp_pct:+.1%})\n"
            f"📈 Resistencia: ${rs:,.2f}  ({rs_pct:+.1%})\n"
            f"\n"
            f"Soporte {range_bar} Resistencia\n"
            f"         z-score: {z:+.2f}σ\n"
            f"```"
        )
        fields.append({"name": "💰 Precio y Rango (mensual ±2σ)", "value": price_text, "inline": False})

        # Acción
        act  = signal["action"]
        aem  = signal["action_emoji"]
        adet = signal["action_detail"]

        action_text = f"{aem} **{act}**\n{adet}"

        # Target y Stop si hay señal activa
        if signal.get("target") and signal.get("stop_loss"):
            tgt = signal["target"]
            stp = signal["stop_loss"]
            tgt_pct = (tgt / p - 1)
            stp_pct = (stp / p - 1)
            action_text += (
                f"\n```\n"
                f"🎯 Target:    ${tgt:,.2f}  ({tgt_pct:+.1%})\n"
                f"🛑 Stop Loss: ${stp:,.2f}  ({stp_pct:+.1%})\n"
                f"```"
            )

        fields.append({"name": "🎯 Acción", "value": action_text, "inline": False})
    else:
        fields.append({
            "name": "🎯 Acción",
            "value": "⏸️ **MANTENER** — Sin datos de precio en vivo",
            "inline": False,
        })

    # ── Métricas rápidas ──
    vol = row.get("Vol. anual", "—")
    ac5 = row.get("Autocorr 5d", "—")
    tr2 = row.get("Tendencia R²", "—")
    dias = row.get("Días", "—")

    fields.extend([
        {"name": "Vol. anual",    "value": f"`{vol}`",  "inline": True},
        {"name": "Autocorr 5d",   "value": f"`{ac5}`",  "inline": True},
        {"name": "Tendencia R²",  "value": f"`{tr2}`",  "inline": True},
        {"name": "Histórico",     "value": f"`{dias} días`", "inline": True},
    ])

    # ── Métricas profundas (si existen) ──
    sil = _parse_float(row.get("Sil. avg"))
    rev = _parse_float(row.get("Rev. score"))
    wr  = _parse_float(row.get("WR +10d"))
    ss  = _parse_float(row.get("Sharpe est."))
    sbh = _parse_float(row.get("Sharpe B&H"))

    if sil is not None:
        deep_lines = []
        deep_lines.append(f"Silhouette   {_bar(sil, 1.0, 8)}  {sil:.3f}")
        if rev is not None:
            deep_lines.append(f"Rev. score   {_bar(rev, 1.0, 8)}  {rev:.2f}")
        if wr is not None:
            deep_lines.append(f"Win Rate 10d {_bar(wr, 1.0, 8)}  {wr:.0%}")

        deep_text = "```\n" + "\n".join(deep_lines) + "\n```"
        fields.append({"name": "🔬 Análisis Profundo", "value": deep_text, "inline": False})

        if ss is not None:
            fields.append({"name": "Sharpe est.", "value": f"`{ss:.2f}`", "inline": True})
        if sbh is not None:
            fields.append({"name": "Sharpe B&H", "value": f"`{sbh:.2f}`", "inline": True})

    # ── Error parcial ──
    error = row.get("Error")
    if error and str(error).strip() and str(error) != "nan":
        fields.append({
            "name": "⚠️ Error",
            "value": f"`{str(error)[:200]}`",
            "inline": False,
        })

    return {
        "title": f"{emoji}  {ticker} — {score}/100  {grade}",
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": "LONQ · Sistema de reversión a la media"},
    }


# ── Embed resumen global ────────────────────────────────────────────────────

def build_summary_embed(
    df: pd.DataFrame,
    run_date: str,
    job_url: str = "",
    mode: str = "FAST",
) -> dict:
    """Genera el embed resumen con distribución de grados."""
    total = len(df)
    excellent = df[df["Score"] >= 70]["Ticker"].tolist()
    good      = df[(df["Score"] >= 50) & (df["Score"] < 70)]["Ticker"].tolist()
    marginal  = df[(df["Score"] >= 30) & (df["Score"] < 50)]["Ticker"].tolist()

    color = 0x2ecc71 if excellent else (0xf39c12 if good else 0xe74c3c)

    # Distribución visual
    n_exc = len(excellent)
    n_goo = len(good)
    n_mar = len(marginal)
    n_bad = total - n_exc - n_goo - n_mar

    dist_lines = [
        f"🟢 Excelentes ≥70  {_bar(n_exc, total, 12)}  {n_exc}",
        f"🟡 Buenos    50-69  {_bar(n_goo, total, 12)}  {n_goo}",
        f"🟠 Marginales 30-49 {_bar(n_mar, total, 12)}  {n_mar}",
        f"🔴 No aptos   <30   {_bar(n_bad, total, 12)}  {n_bad}",
    ]
    dist_text = "```\n" + "\n".join(dist_lines) + "\n```"

    # Listas de tickers
    exc_str = ", ".join([f"`{t}`" for t in excellent]) or "—"
    goo_str = ", ".join([f"`{t}`" for t in good[:10]]) or "—"
    if len(good) > 10:
        goo_str += f" (+{len(good)-10} más)"

    link = f"[{run_date}]({job_url})" if job_url else run_date
    description = f"Screening completado · {link}\n\n{dist_text}"

    fields = [
        {"name": "🏆 Excelentes ≥70", "value": exc_str, "inline": False},
        {"name": "✓ Buenos 50–69",    "value": goo_str, "inline": False},
        {"name": "Total analizados",  "value": str(total), "inline": True},
        {"name": "Modo",              "value": mode,       "inline": True},
        {"name": "Seleccionados",     "value": str(n_exc + n_goo), "inline": True},
    ]

    return {
        "title": "📊 LONQ — Daily Asset Scan",
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": "LONQ · Sistema de reversión a la media · github-actions[bot]"},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ── Envío a Discord ─────────────────────────────────────────────────────────

def _send_webhook(webhook_url: str, payload: dict) -> int:
    """Envía un payload JSON al webhook de Discord. Devuelve HTTP status code."""
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "-X", "POST", webhook_url,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    code = int(result.stdout.strip() or "0")
    return code


def send_dashboards(
    webhook_url: str,
    csv_path: str,
    run_date: str = "",
    job_url: str = "",
    mode: str = "FAST",
    min_score: int = 50,
    max_tickers: int = 10,
    batch_size: int = 5,
) -> dict:
    """Lee el CSV, genera dashboards y los envía a Discord.

    Args:
        webhook_url:  URL del webhook de Discord.
        csv_path:     Ruta al CSV de resultados del screener.
        run_date:     Fecha del análisis (YYYY-MM-DD).
        job_url:      URL del job de GitHub Actions.
        mode:         FAST o DEEP.
        min_score:    Score mínimo para generar dashboard individual.
        max_tickers:  Máximo de dashboards individuales a enviar.
        batch_size:   Embeds por mensaje (Discord max = 10).

    Returns:
        dict con estadísticas del envío.
    """
    if not run_date:
        run_date = datetime.date.today().isoformat()

    df = pd.read_csv(csv_path).sort_values("Score", ascending=False)

    stats = {"total": len(df), "dashboards_sent": 0, "messages_sent": 0, "errors": []}

    # ── 1. Enviar resumen global ──────────────────────────────────────────
    summary = build_summary_embed(df, run_date, job_url, mode)
    code = _send_webhook(webhook_url, {"embeds": [summary]})
    stats["messages_sent"] += 1

    if code not in (200, 204):
        stats["errors"].append(f"Summary embed HTTP {code}")
        print(f"  ⚠ Error enviando resumen: HTTP {code}")
    else:
        print(f"  ✓ Resumen global enviado")

    # Rate limit: esperar entre mensajes
    time.sleep(1)

    # ── 2. Generar dashboards individuales ────────────────────────────────
    selected = df[df["Score"] >= min_score].head(max_tickers)

    if selected.empty:
        print("  — No hay tickers con score suficiente para dashboard individual")
        return stats

    # Descargar precios en vivo y computar señales
    tickers_list = selected["Ticker"].tolist()
    print(f"  📡 Descargando precios en vivo para {len(tickers_list)} tickers...")
    signals = fetch_live_signals(tickers_list)
    print(f"  ✓ Señales calculadas: {len(signals)}/{len(tickers_list)}")

    embeds = []
    for _, row in selected.iterrows():
        tk = row["Ticker"]
        sig = signals.get(tk)
        embeds.append(build_ticker_embed(row, signal=sig))

    print(f"  📊 {len(embeds)} dashboards generados para tickers con score ≥ {min_score}")

    # ── 3. Enviar en lotes ────────────────────────────────────────────────
    for i in range(0, len(embeds), batch_size):
        batch = embeds[i : i + batch_size]
        code = _send_webhook(webhook_url, {"embeds": batch})
        stats["messages_sent"] += 1

        batch_tickers = [e["title"].split("—")[0].strip().split()[-1] for e in batch]
        batch_str = ", ".join(batch_tickers)

        if code not in (200, 204):
            stats["errors"].append(f"Batch {i//batch_size+1} HTTP {code}")
            print(f"  ⚠ Error lote {i//batch_size+1} ({batch_str}): HTTP {code}")
        else:
            stats["dashboards_sent"] += len(batch)
            print(f"  ✓ Lote {i//batch_size+1} enviado: {batch_str}")

        # Rate limit entre lotes
        time.sleep(1.5)

    print(f"\n  ✅ Resumen: {stats['dashboards_sent']}/{len(embeds)} dashboards enviados "
          f"en {stats['messages_sent']} mensajes")

    return stats


# ── CLI standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob
    import os

    p = argparse.ArgumentParser(description="LONQ — Enviar dashboards a Discord")
    p.add_argument("--webhook",     required=True, help="URL del webhook de Discord")
    p.add_argument("--csv",         default=None,  help="CSV de resultados (auto-detecta el más reciente)")
    p.add_argument("--date",        default="",    help="Fecha del análisis")
    p.add_argument("--job-url",     default="",    help="URL del job de GitHub Actions")
    p.add_argument("--mode",        default="FAST", help="Modo de análisis")
    p.add_argument("--min-score",   type=int, default=50, help="Score mínimo para dashboard")
    p.add_argument("--max-tickers", type=int, default=10, help="Máx. dashboards individuales")
    p.add_argument("--batch-size",  type=int, default=5,  help="Embeds por mensaje")
    args = p.parse_args()

    # Auto-detectar CSV
    csv_path = args.csv
    if not csv_path:
        csvs = sorted(glob.glob("results/screen_*.csv"), reverse=True)
        if not csvs:
            print("No se encontró CSV de resultados.")
            exit(1)
        csv_path = csvs[0]
        print(f"  CSV detectado: {csv_path}")

    stats = send_dashboards(
        webhook_url=args.webhook,
        csv_path=csv_path,
        run_date=args.date or datetime.date.today().isoformat(),
        job_url=args.job_url,
        mode=args.mode,
        min_score=args.min_score,
        max_tickers=args.max_tickers,
        batch_size=args.batch_size,
    )

    if stats["errors"]:
        print(f"\n  ⚠ Errores: {stats['errors']}")
        exit(1)
