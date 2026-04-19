#!/usr/bin/env python3
"""Ejemplos de uso del TickerAnalyzer.

Ejecutar con: python example_usage.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.ticker_analyzer import TickerAnalyzer
import pandas as pd


def ejemplo_1_basico():
    """Ejemplo 1: Análisis básico de un ticker."""
    print("\n" + "="*70)
    print("EJEMPLO 1: Análisis Rápido de un Ticker")
    print("="*70)

    analyzer = TickerAnalyzer(ticker="AAPL")

    if analyzer.load_data():
        if analyzer.analyze(deep=False):
            analyzer.print_report()
    else:
        print("⚠ Saltando ejemplo 1 (sin datos)")


def ejemplo_2_profundo():
    """Ejemplo 2: Análisis profundo con ensemble."""
    print("\n" + "="*70)
    print("EJEMPLO 2: Análisis Profundo")
    print("="*70)

    analyzer = TickerAnalyzer(ticker="TSLA")

    if analyzer.load_data():
        analyzer.load_vix()
        print("⏳ Análisis profundo en progreso... (puede tomar ~60s)")
        if analyzer.analyze(deep=True):
            analyzer.print_report()
    else:
        print("⚠ Saltando ejemplo 2 (sin datos)")


def ejemplo_3_json():
    """Ejemplo 3: Exportar resultados como JSON."""
    print("\n" + "="*70)
    print("EJEMPLO 3: Exportar a JSON")
    print("="*70)

    analyzer = TickerAnalyzer(ticker="SPY")

    if analyzer.load_data():
        if analyzer.analyze(deep=False):
            import json
            resultados = analyzer.to_dict()
            print(json.dumps(resultados, indent=2))
    else:
        print("⚠ Saltando ejemplo 3 (sin datos)")


def ejemplo_4_comparacion():
    """Ejemplo 4: Comparar múltiples tickers."""
    print("\n" + "="*70)
    print("EJEMPLO 4: Comparar Múltiples Tickers")
    print("="*70)

    tickers = ["AAPL", "TSLA", "SPY", "KO"]
    resultados = []

    for ticker in tickers:
        print(f"\n  Analizando {ticker}...", end=" ")
        analyzer = TickerAnalyzer(ticker=ticker)

        if analyzer.load_data():
            if analyzer.analyze(deep=False):
                resultados.append(analyzer.get_metrics_dataframe())
                print("✓")
            else:
                print("✗ (análisis falló)")
        else:
            print("✗ (sin datos)")

    if resultados:
        df = pd.concat(resultados, ignore_index=True)
        print("\n📊 COMPARACIÓN DE TICKERS")
        print(df.to_string(index=False))

        # Ranking
        print("\n🏆 RANKING POR PUNTUACIÓN")
        df_sorted = df.sort_values("Score", ascending=False)
        for i, row in df_sorted.iterrows():
            print(f"  {i+1}. {row['Ticker']:6s} - Score: {row['Score']:3.0f} [{row['Grade']}]")


def ejemplo_5_uso_colab():
    """Ejemplo 5: Template para Colab (code only)."""
    print("\n" + "="*70)
    print("EJEMPLO 5: Template para Colab")
    print("="*70)

    codigo = '''
# En Google Colab, copiar esto en una celda:

!git clone https://github.com/pfreezv/a_tecn.git
%cd a_tecn
!pip install -q -r requirements.txt

import sys
sys.path.insert(0, '/content/a_tecn')

from src.ticker_analyzer import TickerAnalyzer

# Cambiar "AAPL" por el ticker que quieras
ticker = input("Ingresa ticker (AAPL, TSLA, etc): ").upper()

analyzer = TickerAnalyzer(ticker=ticker)

# Intentar cargar datos
if analyzer.load_data():
    print("✓ Datos cargados")

    # Opción: análisis rápido
    analyzer.analyze(deep=False)
    analyzer.print_report()

    # Opción: análisis profundo (descomenta para activar)
    # analyzer.load_vix()
    # analyzer.analyze(deep=True)
    # analyzer.print_report()
else:
    print("Datos no disponibles. Descargar con yfinance:")
    print("\\nimport yfinance as yf")
    print(f"df = yf.download(\\"{ticker}\\", period=\\"5y\\", progress=False)")
    print('df.to_csv(f\\"{ticker.lower()}_data.csv\\")')
'''

    print(codigo)


if __name__ == "__main__":
    print("\n🚀 EJEMPLOS DE TICKER ANALYZER")
    print("Ejecuta cada función individualmente o comenta las que no necesites.")

    # Ejecutar ejemplos
    ejemplo_1_basico()
    # ejemplo_2_profundo()  # Descomenta para análisis profundo
    ejemplo_3_json()
    ejemplo_4_comparacion()
    ejemplo_5_uso_colab()

    print("\n✓ Ejemplos completados")
