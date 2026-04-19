#!/usr/bin/env python3
"""Script CLI para analizar un ticker específico.

Uso:
    python analyze_ticker.py AAPL                    # análisis rápido
    python analyze_ticker.py AAPL --deep             # análisis profundo
    python analyze_ticker.py AAPL --data datos.csv   # con datos personalizados
    python analyze_ticker.py AAPL --deep --threshold 2.5 --horizon 15
"""

import argparse
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.dirname(__file__))

from src.ticker_analyzer import TickerAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analiza un ticker específico con métricas detalladas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python analyze_ticker.py AAPL                    # análisis rápido (~2s)
  python analyze_ticker.py TSLA --deep             # análisis profundo (~60s)
  python analyze_ticker.py BTC --data btc_data.csv # con archivo personalizado
  python analyze_ticker.py SPY --deep --json       # salida JSON
        """
    )

    parser.add_argument("ticker", help="Símbolo del activo (AAPL, TSLA, BTC, etc.)")
    parser.add_argument(
        "--deep", action="store_true",
        help="Ejecutar análisis profundo con ensemble, trigger y señales"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Ruta al archivo CSV con datos históricos"
    )
    parser.add_argument(
        "--vix", type=str, default=None,
        help="Ruta al archivo CSV con datos de VIX"
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0,
        help="Umbral para detección de anomalías (default: 2.0)"
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Horizonte de predicción en días (default: 10)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Salida en formato JSON (para integración)"
    )

    args = parser.parse_args()

    # Crear analizador
    analyzer = TickerAnalyzer(args.ticker)

    # Cargar datos
    if not analyzer.load_data(args.data):
        print(f"✗ No se pudieron cargar datos para {args.ticker}")
        return 1

    # Cargar VIX si está disponible
    if args.deep:
        analyzer.load_vix(args.vix)

    # Ejecutar análisis
    if not analyzer.analyze(deep=args.deep, threshold=args.threshold, horizon=args.horizon):
        print(f"✗ Error durante el análisis")
        return 1

    # Salida
    if args.json:
        import json
        print(json.dumps(analyzer.to_dict(), indent=2))
    else:
        analyzer.print_report()
        # Mostrar también como tabla
        df = analyzer.get_metrics_dataframe()
        print("\n📋 TABLA DE RESUMEN")
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
