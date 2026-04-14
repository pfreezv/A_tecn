"""Fase 0 — PoC validación de datos: IBEX 35, Euro Stoxx 50, Crypto.

Comprueba:
  - Histórico disponible en yfinance (días, primera/última fecha)
  - Huecos en la serie (días sin datos > 5 días hábiles consecutivos)
  - Calidad mínima para el modelo (>= 500 días sin huecos graves)
  - fast_screen score para los que pasan el corte mínimo (>= 100 días)

Salida:
  - Tabla en consola por grupo (IBEX / Stoxx / Crypto)
  - CSV: poc_results_YYYY-MM-DD.csv
"""

import sys
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Permitir importar src/ desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import yfinance as yf

from src.screener import fast_screen, MIN_DAYS

# ── Universos de prueba ───────────────────────────────────────────────────────

IBEX_35 = [
    ("SAN.MC",  "Santander"),
    ("BBVA.MC", "BBVA"),
    ("ITX.MC",  "Inditex"),
    ("TEF.MC",  "Telefónica"),
    ("IBE.MC",  "Iberdrola"),
    ("REP.MC",  "Repsol"),
    ("AMS.MC",  "Amadeus IT"),
    ("CLNX.MC", "Cellnex"),
    ("FER.MC",  "Ferrovial"),
    ("ACS.MC",  "ACS"),
    ("ELE.MC",  "Endesa"),
    ("NTGY.MC", "Naturgy"),
    ("ENG.MC",  "Enagás"),
    ("MAP.MC",  "Mapfre"),
    ("CIE.MC",  "CIE Automotive"),
    ("VIS.MC",  "Viscofan"),
    ("ACX.MC",  "Acerinox"),
    ("MRL.MC",  "Merlin Properties"),
    ("LOG.MC",  "Logista"),
    ("CABK.MC", "CaixaBank"),
]

EURO_STOXX_50 = [
    ("ASML.AS",  "ASML"),
    ("MC.PA",    "LVMH"),
    ("SAP.DE",   "SAP"),
    ("SIE.DE",   "Siemens"),
    ("TTE.PA",   "TotalEnergies"),
    ("AIR.PA",   "Airbus"),
    ("OR.PA",    "L'Oreal"),
    ("BNP.PA",   "BNP Paribas"),
    ("DTE.DE",   "Deutsche Telekom"),
    ("ALV.DE",   "Allianz"),
    ("MUV2.DE",  "Munich Re"),
    ("BAS.DE",   "BASF"),
    ("IFX.DE",   "Infineon"),
    ("ENEL.MI",  "Enel"),
    ("UCG.MI",   "UniCredit"),
    ("PHIA.AS",  "Philips"),
    ("AD.AS",    "Ahold Delhaize"),
    ("ABI.BR",   "AB InBev"),
    ("SU.PA",    "Schneider Electric"),
    ("CS.PA",    "AXA"),
]

CRYPTO = [
    ("BTC-USD",  "Bitcoin"),
    ("ETH-USD",  "Ethereum"),
    ("SOL-USD",  "Solana"),
    ("BNB-USD",  "BNB"),
    ("XRP-USD",  "XRP"),
    ("AVAX-USD", "Avalanche"),
    ("DOGE-USD", "Dogecoin"),
    ("ADA-USD",  "Cardano"),
]

START_DATE = "2020-01-01"
MIN_DAYS_SCREEN = 100  # mínimo para correr fast_screen (relajado para PoC)
MAX_GAP_DAYS = 7       # hueco > 7 días hábiles = problema de datos


# ── Lógica de validación ──────────────────────────────────────────────────────

def detect_gaps(prices: pd.Series) -> int:
    """Cuenta huecos > MAX_GAP_DAYS días naturales entre datos consecutivos."""
    idx = prices.index
    if len(idx) < 2:
        return 0
    diffs = (idx[1:] - idx[:-1]).days
    return int((diffs > MAX_GAP_DAYS).sum())


def validate_ticker(ticker: str, name: str, mode: str) -> dict:
    """Descarga datos y genera métricas de calidad + fast_screen si hay datos."""
    row = {
        "ticker":    ticker,
        "name":      name,
        "mode":      mode,
        "n_days":    0,
        "first_date": None,
        "last_date":  None,
        "gaps":      0,
        "apto_modelo": False,
        "fast_score": None,
        "grade":     None,
        "vol_annual": None,
        "autocorr5": None,
        "trend_r2":  None,
        "error":     None,
    }

    try:
        data = yf.download(
            ticker,
            start=START_DATE,
            progress=False,
            auto_adjust=True,
            repair=True,
        )

        if data.empty:
            row["error"] = "sin datos"
            return row

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        prices = data["Close"].dropna()

        if prices.empty:
            row["error"] = "Close vacío"
            return row

        row["n_days"]     = len(prices)
        row["first_date"] = prices.index[0].strftime("%Y-%m-%d")
        row["last_date"]  = prices.index[-1].strftime("%Y-%m-%d")
        row["gaps"]       = detect_gaps(prices)
        row["apto_modelo"] = (
            len(prices) >= MIN_DAYS and row["gaps"] == 0
        )

        if len(prices) >= MIN_DAYS_SCREEN:
            result = fast_screen(prices, ticker=ticker)
            row["fast_score"] = result.score
            row["grade"]      = result.grade.strip()
            row["vol_annual"] = result.vol_annual
            row["autocorr5"]  = result.autocorr_5d
            row["trend_r2"]   = result.trend_strength

    except Exception as exc:
        row["error"] = str(exc)[:80]

    return row


# ── Presentación ──────────────────────────────────────────────────────────────

def print_group(title: str, rows: list[dict]) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"{'Ticker':<12} {'Nombre':<22} {'Días':>5}  {'Último':>10}  "
          f"{'Gaps':>4}  {'Apto':>4}  {'Score':>5}  {'Grade':>4}  "
          f"{'Vol%':>5}  {'AC5':>6}  {'R²':>5}  Error")
    print("-" * 110)
    for r in rows:
        apto = "SI" if r["apto_modelo"] else "no"
        score = f"{r['fast_score']:3d}" if r['fast_score'] is not None else "  —"
        grade = r["grade"] if r["grade"] else " —"
        vol   = f"{r['vol_annual']:.0%}" if r["vol_annual"] is not None else "  —"
        ac5   = f"{r['autocorr5']:+.3f}" if r["autocorr5"] is not None else "    —"
        r2    = f"{r['trend_r2']:.2f}" if r["trend_r2"] is not None else "   —"
        err   = r["error"] or ""
        last  = r["last_date"] or "—"
        print(f"{r['ticker']:<12} {r['name']:<22} {r['n_days']:>5}  {last:>10}  "
              f"{r['gaps']:>4}  {apto:>4}  {score:>5}  {grade:>4}  "
              f"{vol:>5}  {ac5:>6}  {r2:>5}  {err}")


def print_summary(all_rows: list[dict]) -> None:
    total = len(all_rows)
    ok    = sum(1 for r in all_rows if r["apto_modelo"])
    errores = sum(1 for r in all_rows if r["error"])
    sin_hist = sum(1 for r in all_rows if not r["error"] and r["n_days"] < MIN_DAYS)

    by_mode: dict[str, list] = {}
    for r in all_rows:
        by_mode.setdefault(r["mode"], []).append(r)

    print(f"\n{'='*72}")
    print(f"  RESUMEN GLOBAL ({date.today()})")
    print(f"{'='*72}")
    print(f"  Total tickers probados : {total}")
    print(f"  Aptos para el modelo   : {ok}  ({ok/total:.0%})")
    print(f"  Sin suficiente histórico: {sin_hist}")
    print(f"  Errores / sin datos    : {errores}")

    for mode, rows in by_mode.items():
        ok_m = sum(1 for r in rows if r["apto_modelo"])
        scored = [r for r in rows if r["fast_score"] is not None]
        avg_score = sum(r["fast_score"] for r in scored) / len(scored) if scored else 0
        print(f"\n  [{mode}]  {len(rows)} tickers — "
              f"{ok_m} aptos — score medio {avg_score:.0f}")
        top = sorted(scored, key=lambda x: x["fast_score"], reverse=True)[:5]
        for r in top:
            print(f"    {r['ticker']:<12} score={r['fast_score']:3d}  "
                  f"grade={r['grade']}  vol={r['vol_annual']:.0%}  "
                  f"ac5={r['autocorr5']:+.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nLONQ — PoC validación de datos EU + Crypto")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start date: {START_DATE}  |  MIN_DAYS modelo: {MIN_DAYS}  "
          f"|  MAX_GAP: {MAX_GAP_DAYS}d")

    groups = [
        ("IBEX 35",      IBEX_35,       "ibex35"),
        ("Euro Stoxx 50", EURO_STOXX_50, "stoxx50"),
        ("Crypto",       CRYPTO,        "crypto"),
    ]

    all_rows: list[dict] = []

    for title, tickers, mode in groups:
        print(f"\n→ Descargando {title} ({len(tickers)} tickers)…", flush=True)
        rows = []
        for ticker, name in tickers:
            print(f"  {ticker:<12}", end="", flush=True)
            row = validate_ticker(ticker, name, mode)
            rows.append(row)
            status = f"OK {row['n_days']}d" if not row["error"] else f"ERR {row['error'][:20]}"
            print(f" {status}")
        print_group(title, rows)
        all_rows.extend(rows)

    print_summary(all_rows)

    # Guardar CSV
    out_path = Path(__file__).parent / f"poc_results_{date.today()}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
