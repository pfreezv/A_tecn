"""LONQ Universe — fuentes dinámicas de componentes de índices bursátiles.

Soporta:
  - S&P 500        : scraping Wikipedia (tabla estática, muy estable)
  - IBEX 35        : scraping Wikipedia ES
  - Euro Stoxx 50  : scraping Wikipedia EN
  - Crypto         : lista curada de 8 principales (sufijos -USD para yfinance)
  - Custom         : lista manual de tickers.yaml (universo original LONQ)

Caché local: .cache/universes/<nombre>_YYYY-MM-DD.json (TTL 7 días)
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "universes"
CACHE_TTL_DAYS = 7


# ── Tipos ────────────────────────────────────────────────────────────────────

class UniverseTicker(NamedTuple):
    ticker: str     # símbolo yfinance  (ej. "SAN.MC", "BTC-USD")
    name:   str     # nombre legible    (ej. "Santander")
    mode:   str     # "full" | "trigger_only"
    market: str     # "US" | "EU" | "CRYPTO"
    index:  str     # "sp500" | "ibex35" | "stoxx50" | "crypto" | "custom"


# ── Caché ─────────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_{date.today()}.json"


def _load_cache(name: str) -> list[dict] | None:
    """Devuelve datos cacheados si existen y no han expirado (TTL 7 días)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = date.today() - timedelta(days=CACHE_TTL_DAYS)
    for path in sorted(CACHE_DIR.glob(f"{name}_*.json"), reverse=True):
        try:
            file_date = date.fromisoformat(path.stem.split("_")[-1])
        except ValueError:
            continue
        if file_date >= cutoff:
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    return None


def _save_cache(name: str, data: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(name).write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ── Scrapers ─────────────────────────────────────────────────────────────────

def _scrape_wikipedia_table(url: str, symbol_col: str, name_col: str,
                             ticker_transform=None) -> list[tuple[str, str]]:
    """Descarga la primera tabla de Wikipedia que contenga symbol_col.

    Returns:
        Lista de (ticker, nombre) — puede ser vacía si falla la descarga.
    """
    try:
        import pandas as pd
        tables = pd.read_html(url, flavor="lxml")
    except Exception as exc:
        log.warning("No se pudo leer tabla Wikipedia (%s): %s", url, exc)
        # Intentar sin lxml
        try:
            import pandas as pd
            tables = pd.read_html(url)
        except Exception as exc2:
            log.error("Fallo definitivo scraping %s: %s", url, exc2)
            return []

    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        # Buscar coincidencia flexible en el nombre de la columna
        sym_match  = next((c for c in cols if symbol_col.lower() in c.lower()), None)
        name_match = next((c for c in cols if name_col.lower()   in c.lower()), None)
        if sym_match and name_match:
            pairs = []
            for _, row in df.iterrows():
                sym  = str(row[sym_match]).strip()
                name = str(row[name_match]).strip()
                if not sym or sym.lower() in ("nan", "—", ""):
                    continue
                if ticker_transform:
                    sym = ticker_transform(sym)
                if sym:
                    pairs.append((sym, name))
            if pairs:
                return pairs

    log.warning("Ninguna tabla de %s tiene columnas '%s' y '%s'",
                url, symbol_col, name_col)
    return []


def get_sp500(use_cache: bool = True) -> list[UniverseTicker]:
    """Componentes del S&P 500 desde Wikipedia."""
    cached = _load_cache("sp500") if use_cache else None
    if cached is not None:
        return [UniverseTicker(**d) for d in cached]

    log.info("Descargando componentes S&P 500 de Wikipedia…")
    pairs = _scrape_wikipedia_table(
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        symbol_col="Symbol",
        name_col="Security",
        # Wikipedia usa puntos; yfinance prefiere guiones para BRK.B → BRK-B
        ticker_transform=lambda s: s.replace(".", "-"),
    )

    if not pairs:
        log.error("No se obtuvieron tickers del S&P 500 — devolviendo lista vacía")
        return []

    result = [
        UniverseTicker(ticker=t, name=n, mode="full", market="US", index="sp500")
        for t, n in pairs
    ]
    _save_cache("sp500", [r._asdict() for r in result])
    log.info("S&P 500: %d tickers obtenidos", len(result))
    return result


def get_ibex35(use_cache: bool = True) -> list[UniverseTicker]:
    """Componentes del IBEX 35 desde Wikipedia (sufijo .MC para yfinance)."""
    cached = _load_cache("ibex35") if use_cache else None
    if cached is not None:
        return [UniverseTicker(**d) for d in cached]

    log.info("Descargando componentes IBEX 35 de Wikipedia…")
    # Fallback estático si Wikipedia falla (componentes a 2026-04)
    static_pairs = [
        ("SAN.MC",  "Santander"),     ("BBVA.MC",  "BBVA"),
        ("ITX.MC",  "Inditex"),       ("TEF.MC",   "Telefónica"),
        ("IBE.MC",  "Iberdrola"),     ("REP.MC",   "Repsol"),
        ("AMS.MC",  "Amadeus IT"),    ("CLNX.MC",  "Cellnex"),
        ("FER.MC",  "Ferrovial"),     ("ACS.MC",   "ACS"),
        ("ELE.MC",  "Endesa"),        ("NTGY.MC",  "Naturgy"),
        ("ENG.MC",  "Enagás"),        ("MAP.MC",   "Mapfre"),
        ("CIE.MC",  "CIE Automotive"),("VIS.MC",   "Viscofan"),
        ("ACX.MC",  "Acerinox"),      ("MRL.MC",   "Merlin Properties"),
        ("LOG.MC",  "Logista"),       ("CABK.MC",  "CaixaBank"),
        ("BKIA.MC", "Bankinter"),     ("GRF.MC",   "Grifols"),
        ("COL.MC",  "Colonial"),      ("MTS.MC",   "ArcelorMittal ES"),
        ("MEL.MC",  "Meliá Hotels"),  ("IAG.MC",   "IAG"),
        ("AENA.MC", "AENA"),          ("SLR.MC",   "Solaria"),
        ("PUIG.MC", "Puig Brands"),   ("ROVI.MC",  "Laboratorios ROVI"),
        ("PHM.MC",  "Pharma Mar"),    ("TRE.MC",   "Técnicas Reunidas"),
        ("BKT.MC",  "Bankinter"),     ("SAB.MC",   "Banco Sabadell"),
        ("UNI.MC",  "Unicaja"),
    ]

    # Intentar Wikipedia primero
    pairs = _scrape_wikipedia_table(
        url="https://es.wikipedia.org/wiki/IBEX_35",
        symbol_col="Ticker",
        name_col="Empresa",
        ticker_transform=lambda s: s if s.endswith(".MC") else f"{s}.MC",
    )

    if not pairs:
        log.warning("Scraping IBEX 35 fallido — usando lista estática (%d tickers)",
                    len(static_pairs))
        pairs = static_pairs

    result = [
        UniverseTicker(ticker=t, name=n, mode="full", market="EU", index="ibex35")
        for t, n in pairs
    ]
    _save_cache("ibex35", [r._asdict() for r in result])
    log.info("IBEX 35: %d tickers obtenidos", len(result))
    return result


def get_eurostoxx50(use_cache: bool = True) -> list[UniverseTicker]:
    """Componentes del Euro Stoxx 50 desde Wikipedia (sufijos EU para yfinance)."""
    cached = _load_cache("stoxx50") if use_cache else None
    if cached is not None:
        return [UniverseTicker(**d) for d in cached]

    log.info("Descargando componentes Euro Stoxx 50 de Wikipedia…")
    # Fallback estático (componentes a 2026-04)
    static_pairs = [
        # Países Bajos (.AS)
        ("ASML.AS",  "ASML"),           ("PHIA.AS",  "Philips"),
        ("AD.AS",    "Ahold Delhaize"), ("ING.AS",   "ING Group"),
        ("HEIA.AS",  "Heineken"),       ("NN.AS",    "NN Group"),
        # Francia (.PA)
        ("MC.PA",    "LVMH"),           ("OR.PA",    "L'Oreal"),
        ("TTE.PA",   "TotalEnergies"),  ("AIR.PA",   "Airbus"),
        ("BNP.PA",   "BNP Paribas"),    ("CS.PA",    "AXA"),
        ("SAN.PA",   "Sanofi"),         ("SU.PA",    "Schneider Electric"),
        ("SGO.PA",   "Saint-Gobain"),   ("GLE.PA",   "Société Générale"),
        ("CAP.PA",   "Capgemini"),      ("DG.PA",    "Vinci"),
        # Alemania (.DE)
        ("SAP.DE",   "SAP"),            ("SIE.DE",   "Siemens"),
        ("ALV.DE",   "Allianz"),        ("MUV2.DE",  "Munich Re"),
        ("BAS.DE",   "BASF"),           ("DTE.DE",   "Deutsche Telekom"),
        ("IFX.DE",   "Infineon"),       ("BAYN.DE",  "Bayer"),
        ("VOW3.DE",  "Volkswagen"),     ("BMW.DE",   "BMW"),
        ("MBG.DE",   "Mercedes-Benz"),  ("DB1.DE",   "Deutsche Börse"),
        ("RWE.DE",   "RWE"),            ("HNR1.DE",  "Hannover Re"),
        # Italia (.MI)
        ("ENEL.MI",  "Enel"),           ("UCG.MI",   "UniCredit"),
        ("ISP.MI",   "Intesa Sanpaolo"),("ENI.MI",   "ENI"),
        ("STLAM.MI", "Stellantis"),
        # España (.MC)
        ("SAN.MC",   "Santander ES"),   ("BBVA.MC",  "BBVA"),
        ("ITX.MC",   "Inditex"),        ("IBE.MC",   "Iberdrola"),
        # Bélgica (.BR)
        ("ABI.BR",   "AB InBev"),
        # Finlandia (.HE)
        ("NOKIA.HE", "Nokia"),
        # Irlanda (.IR)
        ("CRH.IR",   "CRH"),
        # Luxemburgo (sin sufijo estándar — omitido)
    ]

    # Intentar Wikipedia
    pairs = _scrape_wikipedia_table(
        url="https://en.wikipedia.org/wiki/Euro_Stoxx_50",
        symbol_col="Ticker",
        name_col="Company",
    )

    if not pairs:
        log.warning("Scraping Euro Stoxx 50 fallido — usando lista estática (%d tickers)",
                    len(static_pairs))
        pairs = static_pairs

    result = [
        UniverseTicker(ticker=t, name=n, mode="full", market="EU", index="stoxx50")
        for t, n in pairs
    ]
    _save_cache("stoxx50", [r._asdict() for r in result])
    log.info("Euro Stoxx 50: %d tickers obtenidos", len(result))
    return result


def get_crypto(use_cache: bool = True) -> list[UniverseTicker]:
    """8 principales cryptomonedas (modo trigger_only — sin ensemble).

    Nota: el modelo LONQ de reversión NO aplica directamente a crypto
    (son activos tendenciales). Solo se usa la capa de trigger/z-score
    para detectar pullbacks anómalos. El ensemble y walk-forward se omiten.
    """
    _ = use_cache  # crypto: lista fija, sin caché necesaria
    return [
        UniverseTicker("BTC-USD",  "Bitcoin",    "trigger_only", "CRYPTO", "crypto"),
        UniverseTicker("ETH-USD",  "Ethereum",   "trigger_only", "CRYPTO", "crypto"),
        UniverseTicker("SOL-USD",  "Solana",     "trigger_only", "CRYPTO", "crypto"),
        UniverseTicker("BNB-USD",  "BNB",        "trigger_only", "CRYPTO", "crypto"),
        UniverseTicker("XRP-USD",  "XRP",        "trigger_only", "CRYPTO", "crypto"),
        UniverseTicker("AVAX-USD", "Avalanche",  "trigger_only", "CRYPTO", "crypto"),
        UniverseTicker("DOGE-USD", "Dogecoin",   "trigger_only", "CRYPTO", "crypto"),
        UniverseTicker("ADA-USD",  "Cardano",    "trigger_only", "CRYPTO", "crypto"),
    ]


def get_custom(tickers_yaml_path: str | Path | None = None, use_cache: bool = True) -> list[UniverseTicker]:
    """Lista original de LONQ desde tickers.yaml (113 tickers curados).

    Todos en modo "full" (análisis completo con ensemble).
    """
    import yaml

    if tickers_yaml_path is None:
        tickers_yaml_path = Path(__file__).resolve().parent.parent / "tickers.yaml"

    with open(tickers_yaml_path) as f:
        data = yaml.safe_load(f)

    result = []
    for section, tickers in data.items():
        if not isinstance(tickers, list):
            continue
        for t in tickers:
            ticker = str(t).strip()
            result.append(UniverseTicker(
                ticker=ticker,
                name=ticker,
                mode="full",
                market="US",
                index="custom",
            ))
    return result


# ── API pública ───────────────────────────────────────────────────────────────

AVAILABLE = {
    "sp500":   get_sp500,
    "ibex35":  get_ibex35,
    "stoxx50": get_eurostoxx50,
    "crypto":  get_crypto,
    "custom":  get_custom,
}


def get_universe(names: list[str], use_cache: bool = True) -> list[UniverseTicker]:
    """Devuelve la unión de universos indicados, sin duplicados de ticker.

    Args:
        names:     Lista de nombres de universo, ej. ["sp500", "ibex35", "crypto"]
        use_cache: Usar caché local si existe y no expiró (TTL 7 días)

    Returns:
        Lista de UniverseTicker deduplicados (el primero visto gana si hay colisión).

    Example:
        tickers = get_universe(["ibex35", "stoxx50", "crypto"])
    """
    seen: set[str] = set()
    result: list[UniverseTicker] = []

    for name in names:
        if name not in AVAILABLE:
            log.warning("Universo desconocido: '%s'. Disponibles: %s",
                        name, list(AVAILABLE))
            continue
        for ut in AVAILABLE[name](use_cache=use_cache):
            if ut.ticker not in seen:
                seen.add(ut.ticker)
                result.append(ut)

    log.info("Universo total: %d tickers únicos de %s", len(result), names)
    return result


def split_by_mode(
    tickers: list[UniverseTicker],
) -> tuple[list[UniverseTicker], list[UniverseTicker]]:
    """Separa en (full_analysis, trigger_only)."""
    full    = [t for t in tickers if t.mode == "full"]
    trigger = [t for t in tickers if t.mode == "trigger_only"]
    return full, trigger
