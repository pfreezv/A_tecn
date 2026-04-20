#!/usr/bin/env python3
"""Script para generar notebook Colab automáticamente.

Crea un archivo .ipynb listo para usar en Google Colab.
"""

import json
from pathlib import Path


def crear_notebook_colab():
    """Genera notebook Colab completo para análisis de tickers."""

    notebook = {
        "cells": [
            # Celda 1: Setup y instalación
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📊 LONQ Ticker Analyzer - Colab\n",
                    "\n",
                    "Análisis completo de cualquier ticker con métricas detalladas.\n",
                    "\n",
                    "**Solo necesitas ingresar el ticker que quieras analizar.**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1️⃣ SETUP - Ejecuta esta celda primero\n",
                    "import os\n",
                    "import sys\n",
                    "\n",
                    "# Clonar repositorio\n",
                    "if not os.path.exists('/content/a_tecn'):\n",
                    "    print('📥 Clonando repositorio...')\n",
                    "    !git clone https://github.com/pfreezv/a_tecn.git 2>&1 | grep -E \"(Cloning|done)\"\n",
                    "else:\n",
                    "    print('✓ Repositorio ya existe')\n",
                    "\n",
                    "# Cambiar al directorio\n",
                    "os.chdir('/content/a_tecn')\n",
                    "sys.path.insert(0, '/content/a_tecn')\n",
                    "\n",
                    "# Instalar dependencias\n",
                    "print('📦 Instalando dependencias...')\n",
                    "!pip install -q -r requirements.txt\n",
                    "print('✓ Dependencias instaladas')"
                ]
            },

            # Celda 2: Importar módulos
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2️⃣ IMPORTAR MÓDULOS\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from src.ticker_analyzer import TickerAnalyzer\n",
                    "\n",
                    "# Configurar visualización\n",
                    "plt.style.use('seaborn-v0_8-darkgrid')\n",
                    "sns.set_palette('husl')\n",
                    "\n",
                    "print('✓ Módulos importados')"
                ]
            },

            # Celda 3: Ingresar ticker
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3️⃣ INGRESA EL TICKER QUE QUIERES ANALIZAR\n",
                    "ticker = input('Ingresa ticker (ej: AAPL, TSLA, SPY, BTC): ').upper().strip()\n",
                    "\n",
                    "if not ticker:\n",
                    "    ticker = 'AAPL'  # Default\n",
                    "    print(f'⚠ Usando default: {ticker}')\n",
                    "\n",
                    "print(f'\\n📊 Analizando: {ticker}')\n",
                    "print('='*60)"
                ]
            },

            # Celda 4: Cargar datos
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4️⃣ CARGAR DATOS\n",
                    "analyzer = TickerAnalyzer(ticker=ticker)\n",
                    "\n",
                    "# Intentar cargar datos\n",
                    "if not analyzer.load_data():\n",
                    "    print(f'\\n⚠ No hay datos locales para {ticker}')\n",
                    "    print('\\nOpciones:')\n",
                    "    print('  1. Descargar con yfinance (ejecuta la celda siguiente)')\n",
                    "    print('  2. Usar otro ticker del repositorio: AAPL, TSLA, SPY, KO, BTC')\n",
                    "    DATA_AVAILABLE = False\n",
                    "else:\n",
                    "    print(f'✓ Datos cargados')\n",
                    "    DATA_AVAILABLE = True\n",
                    "    \n",
                    "    # Cargar VIX\n",
                    "    vix_loaded = analyzer.load_vix()\n",
                    "    if vix_loaded:\n",
                    "        print('✓ VIX cargado')\n",
                    "    else:\n",
                    "        print('⚠ VIX no disponible (análisis profundo sin VIX)')"
                ]
            },

            # Celda 5: Descargar datos con yfinance (opcional)
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5️⃣ DESCARGAR DATOS CON YFINANCE (si no están disponibles)\n",
                    "if not DATA_AVAILABLE:\n",
                    "    print(f'📥 Descargando datos para {ticker}...')\n",
                    "    \n",
                    "    import yfinance as yf\n",
                    "    \n",
                    "    try:\n",
                    "        # Descargar 5 años de datos\n",
                    "        df = yf.download(ticker, period='5y', progress=False)\n",
                    "        \n",
                    "        if df.empty:\n",
                    "            print(f'✗ {ticker} no es un ticker válido')\n",
                    "        else:\n",
                    "            # Guardar localmente\n",
                    "            df.to_csv(f'{ticker.lower()}_data.csv')\n",
                    "            print(f'✓ {len(df)} filas descargadas y guardadas')\n",
                    "            \n",
                    "            # Reintentar cargar\n",
                    "            analyzer = TickerAnalyzer(ticker=ticker)\n",
                    "            if analyzer.load_data():\n",
                    "                print('✓ Datos cargados exitosamente')\n",
                    "                analyzer.load_vix()\n",
                    "                DATA_AVAILABLE = True\n",
                    "    except Exception as e:\n",
                    "        print(f'✗ Error descargando: {e}')"
                ]
            },

            # Celda 6: Análisis rápido
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 6️⃣ ANÁLISIS RÁPIDO\n",
                    "if DATA_AVAILABLE:\n",
                    "    print('\\n⚡ ANÁLISIS RÁPIDO (~2 segundos)\\n')\n",
                    "    \n",
                    "    if analyzer.analyze(deep=False):\n",
                    "        analyzer.print_report()\n",
                    "    else:\n",
                    "        print('✗ Error en análisis rápido')\n",
                    "else:\n",
                    "    print('⚠ Carga datos primero con la celda anterior')"
                ]
            },

            # Celda 7: Análisis profundo
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 7️⃣ ANÁLISIS PROFUNDO (60 segundos)\n",
                    "if DATA_AVAILABLE:\n",
                    "    print('\\n🔬 ANÁLISIS PROFUNDO (~60 segundos)\\n')\n",
                    "    print('⏳ En progreso...')\n",
                    "    \n",
                    "    if analyzer.analyze(deep=True, threshold=2.0, horizon=10):\n",
                    "        print('\\n' + '='*60)\n",
                    "        analyzer.print_report()\n",
                    "        print('='*60)\n",
                    "    else:\n",
                    "        print('✗ Error en análisis profundo')\n",
                    "        print('  Posibles causas:')\n",
                    "        print('  - Datos insuficientes (<500 días)')\n",
                    "        print('  - Error en ensemble o trigger')\n",
                    "else:\n",
                    "    print('⚠ Carga datos primero')"
                ]
            },

            # Celda 8: Visualizaciones
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 8️⃣ VISUALIZACIONES\n",
                    "if DATA_AVAILABLE and analyzer.fast_result:\n",
                    "    fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "    fig.suptitle(f'{ticker} - Análisis Detallado', fontsize=16, fontweight='bold')\n",
                    "    \n",
                    "    # 1. Desglose de puntos\n",
                    "    ax = axes[0, 0]\n",
                    "    score_breakdown = analyzer.fast_result.score_breakdown\n",
                    "    colors = plt.cm.Set3(np.linspace(0, 1, len(score_breakdown)))\n",
                    "    ax.barh(list(score_breakdown.keys()), list(score_breakdown.values()), color=colors)\n",
                    "    ax.set_xlabel('Puntos')\n",
                    "    ax.set_title('Desglose de Puntuación')\n",
                    "    ax.grid(axis='x', alpha=0.3)\n",
                    "    \n",
                    "    # 2. Métricas clave\n",
                    "    ax = axes[0, 1]\n",
                    "    metrics = {\n",
                    "        'Volatilidad': f\"{analyzer.fast_result.vol_annual:.1%}\",\n",
                    "        'Autocorr': f\"{analyzer.fast_result.autocorr_5d:.3f}\",\n",
                    "        'Tendencia R²': f\"{analyzer.fast_result.trend_strength:.3f}\",\n",
                    "        'Score': f\"{analyzer.fast_result.score}/100\"\n",
                    "    }\n",
                    "    ax.axis('off')\n",
                    "    y_pos = 0.9\n",
                    "    for key, val in metrics.items():\n",
                    "        ax.text(0.1, y_pos, f'{key}:', fontweight='bold', fontsize=11)\n",
                    "        ax.text(0.6, y_pos, str(val), fontsize=11)\n",
                    "        y_pos -= 0.2\n",
                    "    ax.set_title('Métricas Rápidas', loc='left', fontweight='bold')\n",
                    "    \n",
                    "    # 3. Distribución de precios\n",
                    "    ax = axes[1, 0]\n",
                    "    prices = analyzer.prices\n",
                    "    ax.plot(prices.index, prices.values, linewidth=1.5, color='#2E86AB')\n",
                    "    ax.fill_between(prices.index, prices.values, alpha=0.3, color='#2E86AB')\n",
                    "    ax.set_xlabel('Fecha')\n",
                    "    ax.set_ylabel('Precio')\n",
                    "    ax.set_title('Histórico de Precios')\n",
                    "    ax.grid(alpha=0.3)\n",
                    "    \n",
                    "    # 4. Retornos\n",
                    "    ax = axes[1, 1]\n",
                    "    returns = np.log(prices / prices.shift(1)).dropna()\n",
                    "    ax.hist(returns, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')\n",
                    "    ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Media: {returns.mean():.4f}')\n",
                    "    ax.set_xlabel('Retorno Diario')\n",
                    "    ax.set_ylabel('Frecuencia')\n",
                    "    ax.set_title('Distribución de Retornos')\n",
                    "    ax.legend()\n",
                    "    ax.grid(alpha=0.3)\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('⚠ Sin datos para visualizar')"
                ]
            },

            # Celda 9: Exportar resultados
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 9️⃣ EXPORTAR RESULTADOS\n",
                    "if DATA_AVAILABLE and analyzer.fast_result:\n",
                    "    import json\n",
                    "    from google.colab import files\n",
                    "    \n",
                    "    # Exportar como JSON\n",
                    "    resultados = analyzer.to_dict()\n",
                    "    \n",
                    "    with open(f'{ticker}_analysis.json', 'w') as f:\n",
                    "        json.dump(resultados, f, indent=2)\n",
                    "    \n",
                    "    # Exportar como CSV\n",
                    "    df = analyzer.get_metrics_dataframe()\n",
                    "    df.to_csv(f'{ticker}_metrics.csv', index=False)\n",
                    "    \n",
                    "    print(f'✓ Archivos guardados:')\n",
                    "    print(f'  - {ticker}_analysis.json')\n",
                    "    print(f'  - {ticker}_metrics.csv')\n",
                    "    print(f'\\n📥 Descargando...')\n",
                    "    \n",
                    "    files.download(f'{ticker}_analysis.json')\n",
                    "    files.download(f'{ticker}_metrics.csv')\n",
                    "else:\n",
                    "    print('⚠ Sin datos para exportar')"
                ]
            },

            # Celda 10: Comparar múltiples tickers
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 🔟 COMPARAR MÚLTIPLES TICKERS (OPCIONAL)\n",
                    "tickers_a_comparar = ['AAPL', 'TSLA', 'SPY', 'KO', 'BTC']  # Modificar lista\n",
                    "\n",
                    "print('📊 Comparando tickers...')\n",
                    "resultados = []\n",
                    "\n",
                    "for t in tickers_a_comparar:\n",
                    "    analyzer_tmp = TickerAnalyzer(ticker=t)\n",
                    "    if analyzer_tmp.load_data():\n",
                    "        if analyzer_tmp.analyze(deep=False):\n",
                    "            resultados.append(analyzer_tmp.get_metrics_dataframe())\n",
                    "            print(f'  ✓ {t}')\n",
                    "        else:\n",
                    "            print(f'  ✗ {t} (análisis falló)')\n",
                    "    else:\n",
                    "        print(f'  ✗ {t} (sin datos)')\n",
                    "\n",
                    "if resultados:\n",
                    "    df_comp = pd.concat(resultados, ignore_index=True)\n",
                    "    print(f'\\n🏆 RANKING')\n",
                    "    df_sorted = df_comp.sort_values('Score', ascending=False)\n",
                    "    print(df_sorted.to_string(index=False))\n",
                    "else:\n",
                    "    print('⚠ No hay resultados para comparar')"
                ]
            },

            # Celda 11: Resumen
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📋 Resumen\n",
                    "\n",
                    "Has completado un análisis completo del ticker usando LONQ Ticker Analyzer.\n",
                    "\n",
                    "### ¿Qué significa la puntuación?\n",
                    "\n",
                    "- **≥70**: Excelente candidato para la estrategia\n",
                    "- **50-69**: Buen candidato\n",
                    "- **30-49**: Candidato marginal\n",
                    "- **<30**: No recomendado\n",
                    "\n",
                    "### Próximos pasos\n",
                    "\n",
                    "1. Cambiar el ticker en la celda 3 para analizar otros\n",
                    "2. Ejecutar análisis profundo en la celda 7 (recomendado)\n",
                    "3. Comparar múltiples tickers en la celda 10\n",
                    "4. Descargar resultados en la celda 9\n",
                    "\n",
                    "---\n",
                    "\n",
                    "**Más información**: Consulta `COLAB_GUIDE.md` y `README_ANALYZER.md` en el repositorio."
                ]
            }
        ],
        "metadata": {
            "colab": {
                "provenance": [],
                "collapsed_sections": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


if __name__ == "__main__":
    notebook = crear_notebook_colab()

    # Guardar como .ipynb
    output_path = "lonq_analyzer_colab.ipynb"
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Notebook creado: {output_path}")
    print(f"  Tamaño: {len(notebook['cells'])} celdas")
    print("\nListo para usar en Google Colab:")
    print("  1. Abre Google Colab")
    print("  2. Carga el archivo lonq_analyzer_colab.ipynb")
    print("  3. Ejecuta las celdas en orden")
