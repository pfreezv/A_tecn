"""Data acquisition and caching for market data.

Incluye:
  - Descarga vía yfinance con retry+backoff exponencial
  - Caché local en parquet (.cache/)
  - Blacklist persistente (.cache/blacklist.json) para tickers con fallos repetidos
"""

import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR      = Path(__file__).resolve().parent.parent / ".cache"
BLACKLIST_PATH = CACHE_DIR / "blacklist.json"

MAX_RETRIES        = 3      # intentos por descarga
BACKOFF_BASE_S     = 2.0    # espera inicial en segundos (2, 4, 8)
BLACKLIST_THRESHOLD = 3     # fallos consecutivos para entrar en blacklist


# ── Blacklist ─────────────────────────────────────────────────────────────────

def _load_blacklist() -> dict:
    """Carga blacklist desde disco. Devuelve {ticker: {fails, last_fail}}."""
    if BLACKLIST_PATH.exists():
        try:
            return json.loads(BLACKLIST_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_blacklist(bl: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BLACKLIST_PATH.write_text(json.dumps(bl, indent=2))


def is_blacklisted(ticker: str) -> bool:
    """True si el ticker tiene >= BLACKLIST_THRESHOLD fallos consecutivos."""
    bl = _load_blacklist()
    entry = bl.get(ticker.upper(), {})
    return entry.get("fails", 0) >= BLACKLIST_THRESHOLD


def mark_failure(ticker: str) -> int:
    """Registra un fallo para el ticker. Devuelve el nº de fallos acumulados."""
    bl = _load_blacklist()
    t  = ticker.upper()
    entry = bl.get(t, {"fails": 0})
    entry["fails"]     = entry.get("fails", 0) + 1
    entry["last_fail"] = date.today().isoformat()
    bl[t] = entry
    _save_blacklist(bl)
    if entry["fails"] >= BLACKLIST_THRESHOLD:
        log.warning("Ticker %s añadido a blacklist (%d fallos consecutivos)",
                    t, entry["fails"])
    return entry["fails"]


def clear_failure(ticker: str) -> None:
    """Resetea el contador de fallos (llamar tras descarga exitosa)."""
    bl = _load_blacklist()
    t  = ticker.upper()
    if t in bl:
        del bl[t]
        _save_blacklist(bl)


def get_blacklist() -> list[str]:
    """Devuelve lista de tickers en blacklist activa."""
    bl = _load_blacklist()
    return [t for t, e in bl.items() if e.get("fails", 0) >= BLACKLIST_THRESHOLD]


# ── Caché parquet ─────────────────────────────────────────────────────────────

def _cache_path(ticker: str, start: str, end: str) -> Path:
    key = hashlib.md5(f"{ticker}_{start}_{end}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.parquet"


# ── Descarga con retry ────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
    skip_blacklist: bool = False,
) -> pd.DataFrame:
    """Descarga OHLCV con caché parquet y retry+backoff exponencial.

    Args:
        ticker:         Símbolo yfinance (ej. "KO", "SAN.MC", "BTC-USD")
        start_date:     Inicio del histórico (YYYY-MM-DD)
        end_date:       Fin del histórico (None = hoy+1)
        use_cache:      Usar caché parquet local si existe
        skip_blacklist: Si True, ignora la blacklist (útil para forzar reintento)

    Returns:
        DataFrame OHLCV con índice DatetimeIndex.

    Raises:
        ValueError: si no se obtienen datos tras los reintentos.
    """
    if end_date is None:
        end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    t = ticker.upper()

    # Blacklist check
    if not skip_blacklist and is_blacklisted(t):
        raise ValueError(f"{t} está en blacklist ({BLACKLIST_THRESHOLD}+ fallos consecutivos)")

    # Caché
    cache_file = _cache_path(t, start_date, end_date)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    # Descarga con retry+backoff
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                t,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                repair=True,
            )

            if data.empty:
                raise ValueError(f"Sin datos para {t}")

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Descarga exitosa — limpiar fallos previos y cachear
            clear_failure(t)
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                data.to_parquet(cache_file)

            return data

        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_S * (2 ** (attempt - 1))   # 2s, 4s, 8s
                log.warning("%s intento %d/%d fallido (%s) — reintentando en %.0fs",
                            t, attempt, MAX_RETRIES, exc, wait)
                time.sleep(wait)

    # Todos los intentos fallaron
    fails = mark_failure(t)
    raise ValueError(
        f"No se pudo descargar {t} tras {MAX_RETRIES} intentos "
        f"(fallos acumulados: {fails}). Último error: {last_exc}"
    )
