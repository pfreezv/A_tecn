"""Analizador individual de tickers — análisis pormenorizado para un ticker específico.

Uso:
    from src.ticker_analyzer import TickerAnalyzer

    analyzer = TickerAnalyzer(ticker="AAPL")
    analyzer.analyze(deep=True)
    analyzer.print_report()
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from pathlib import Path

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screener import fast_screen, deep_screen, ScreenResult


class TickerAnalyzer:
    """Analizador individual de tickers con reportes detallados."""

    def __init__(self, ticker: str):
        """Inicializa analizador para un ticker específico.

        Args:
            ticker: Símbolo del activo (ej: AAPL, BTC, SPY)
        """
        self.ticker = ticker.upper()
        self.prices = None
        self.raw_data = None
        self.fast_result = None
        self.deep_result = None
        self.vix_data = None

    def load_data(self, data_path: Optional[str] = None, auto_download: bool = False) -> bool:
        """Carga datos históricos del ticker.

        Args:
            data_path: Ruta al archivo CSV. Si no se proporciona, busca automáticamente.
            auto_download: Si True, descarga de yfinance si no encuentra datos locales.

        Returns:
            True si se cargó exitosamente.
        """
        if data_path:
            return self._load_from_path(data_path)

        # Buscar automáticamente en el directorio actual
        csv_pattern = f"{self.ticker.lower()}_data.csv"
        if Path(csv_pattern).exists():
            return self._load_from_path(csv_pattern)

        # Buscar en parent directory
        parent_csv = Path("..") / csv_pattern
        if parent_csv.exists():
            return self._load_from_path(str(parent_csv))

        # Intentar descargar de yfinance
        if auto_download:
            print(f"📥 Descargando datos para {self.ticker} desde yfinance...")
            return self._download_yfinance()

        print(f"⚠ No se encontraron datos para {self.ticker}")
        print(f"  Buscó: {csv_pattern}, {parent_csv}")
        return False

    def _download_yfinance(self, period: str = "5y") -> bool:
        """Descarga datos de yfinance."""
        try:
            import yfinance as yf
            df = yf.download(self.ticker, period=period, progress=False)

            if df.empty:
                print(f"✗ {self.ticker} no es un ticker válido")
                return False

            # Guardar localmente
            csv_file = f"{self.ticker.lower()}_data.csv"
            df.to_csv(csv_file)
            print(f"✓ {len(df)} filas descargadas y guardadas en {csv_file}")

            # Cargar desde el archivo guardado
            return self._load_from_path(csv_file)

        except ImportError:
            print("✗ yfinance no está instalado")
            print("  Instalar con: pip install yfinance")
            return False
        except Exception as e:
            print(f"✗ Error descargando datos: {e}")
            return False

    def _load_from_path(self, path: str) -> bool:
        """Carga datos desde un archivo CSV."""
        try:
            df = pd.read_csv(path, index_col='Date', parse_dates=True)
            df = df.sort_index()
            self.raw_data = df
            self.prices = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
            print(f"✓ Datos cargados: {len(df)} filas ({df.index[0].date()} a {df.index[-1].date()})")
            return True
        except Exception as e:
            print(f"✗ Error cargando {path}: {e}")
            return False

    def load_vix(self, vix_path: Optional[str] = None) -> bool:
        """Carga datos de VIX (opcional para análisis profundo)."""
        try:
            if vix_path:
                vix_df = pd.read_csv(vix_path, index_col='Date', parse_dates=True)
            else:
                if Path("vix_data.csv").exists():
                    vix_df = pd.read_csv("vix_data.csv", index_col='Date', parse_dates=True)
                elif Path("../vix_data.csv").exists():
                    vix_df = pd.read_csv("../vix_data.csv", index_col='Date', parse_dates=True)
                else:
                    return False

            vix_df = vix_df.sort_index()
            self.vix_data = vix_df['Close'] if 'Close' in vix_df.columns else vix_df.iloc[:, 0]
            print(f"✓ VIX cargado: {len(vix_df)} filas")
            return True
        except Exception:
            return False

    def analyze(self, deep: bool = False, threshold: float = 2.0, horizon: int = 10) -> bool:
        """Ejecuta análisis del ticker.

        Args:
            deep: Si True, ejecuta análisis profundo (~60s). Si False, solo rápido (~2s).
            threshold: Umbral para detección de anomalías.
            horizon: Horizonte de predicción en días.

        Returns:
            True si el análisis fue exitoso.
        """
        if self.prices is None:
            print("✗ Carga datos primero con load_data()")
            return False

        try:
            # Screening rápido
            print(f"\n📊 Analizando {self.ticker}...")
            self.fast_result = fast_screen(self.prices, self.ticker)

            if deep and self.raw_data is not None:
                print("  Ejecutando análisis profundo (esto puede tomar ~60s)...")
                self.deep_result = deep_screen(
                    self.prices,
                    self.raw_data,
                    ticker=self.ticker,
                    vix_series=self.vix_data,
                    fast_result=self.fast_result,
                    threshold=threshold,
                    horizon=horizon,
                )

            return True
        except Exception as e:
            print(f"✗ Error en análisis: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_report(self):
        """Imprime reporte detallado de análisis."""
        if self.fast_result is None:
            print("✗ Ejecuta analyze() primero")
            return

        result = self.deep_result if self.deep_result else self.fast_result

        print("\n" + "="*70)
        print(f"  ANÁLISIS DE {result.ticker}".center(70))
        print("="*70)

        # Puntuación general
        print(f"\n🎯 PUNTUACIÓN: {result.score}/100  [{result.grade}]")
        print(f"   Recomendación: {result.recommendation}")

        # Métricas rápidas
        print(f"\n📈 MÉTRICAS RÁPIDAS")
        print(f"   Datos históricos:     {result.n_days:,} días")
        print(f"   Volatilidad anual:    {result.vol_annual:.2%}")
        print(f"   Autocorrelación (lag-5): {result.autocorr_5d:.4f}")
        print(f"   Tendencia (R²):       {result.trend_strength:.4f}")

        # Desglose de puntos
        print(f"\n📊 DESGLOSE DE PUNTOS")
        for key, pts in sorted(result.score_breakdown.items()):
            print(f"   {key:25s}: {pts:>3d} pts")

        # Métricas profundas si existen
        if self.deep_result:
            print(f"\n🔬 ANÁLISIS PROFUNDO")
            if self.deep_result.sil_avg is not None:
                print(f"   Silhouette ensemble:  {self.deep_result.sil_avg:.4f}")
            if self.deep_result.reversion_score is not None:
                print(f"   Score reversión:      {self.deep_result.reversion_score:.4f}")
            if self.deep_result.win_rate_10d is not None:
                print(f"   Win rate 10d:         {self.deep_result.win_rate_10d:.2%}")
            if self.deep_result.sharpe_strategy is not None:
                print(f"   Sharpe estrategia:    {self.deep_result.sharpe_strategy:.4f}")
            if self.deep_result.sharpe_bh is not None:
                print(f"   Sharpe Buy & Hold:    {self.deep_result.sharpe_bh:.4f}")
            if self.deep_result.pct_in_market is not None:
                print(f"   % en mercado:         {self.deep_result.pct_in_market:.2%}")

        # Errores si los hubo
        if result.error:
            print(f"\n⚠ ERROR: {result.error}")

        print("\n" + "="*70 + "\n")

    def to_dict(self) -> Dict:
        """Retorna resultados como diccionario (útil para Colab/JSON)."""
        result = self.deep_result if self.deep_result else self.fast_result

        if result is None:
            return {}

        return {
            "ticker": result.ticker,
            "score": result.score,
            "grade": result.grade,
            "recommendation": result.recommendation,
            "métricas": {
                "volatilidad_anual": result.vol_annual,
                "autocorrelación": result.autocorr_5d,
                "tendencia_r2": result.trend_strength,
                "días_histórico": result.n_days,
            },
            "puntos": result.score_breakdown,
            "profundo": {
                "silhouette": self.deep_result.sil_avg if self.deep_result else None,
                "reversion_score": self.deep_result.reversion_score if self.deep_result else None,
                "win_rate_10d": self.deep_result.win_rate_10d if self.deep_result else None,
                "sharpe_strategy": self.deep_result.sharpe_strategy if self.deep_result else None,
                "sharpe_bh": self.deep_result.sharpe_bh if self.deep_result else None,
            }
        }

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Retorna métricas como DataFrame (útil para comparar múltiples tickers)."""
        result = self.deep_result if self.deep_result else self.fast_result

        if result is None:
            return pd.DataFrame()

        data = {
            "Ticker": [result.ticker],
            "Score": [result.score],
            "Grade": [result.grade],
            "Vol Anual": [result.vol_annual],
            "Autocorr": [result.autocorr_5d],
            "Tendencia R²": [result.trend_strength],
            "Días": [result.n_days],
        }

        if self.deep_result:
            data.update({
                "Silhouette": [self.deep_result.sil_avg],
                "Rev Score": [self.deep_result.reversion_score],
                "Win Rate": [self.deep_result.win_rate_10d],
                "Sharpe Strat": [self.deep_result.sharpe_strategy],
                "Sharpe BH": [self.deep_result.sharpe_bh],
            })

        return pd.DataFrame(data)
