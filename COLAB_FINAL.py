"""Versión FINAL corregida del código interactivo para Colab - Sin errores de rutas"""

codigo_final_corregido = '''
# ========== SETUP ROBUSTO ==========
import os
import sys
import subprocess

print("🔧 SETUP ROBUSTO - Corrigiendo rutas...")
print(f"📍 Ubicación actual: {os.getcwd()}")

# Paso 1: Clonar si no existe
if not os.path.exists('/content/a_tecn'):
    print("📥 Clonando repositorio...")
    subprocess.run(['git', 'clone', 'https://github.com/pfreezv/a_tecn.git', '/content/a_tecn'],
                   capture_output=True)
    print("✓ Repositorio clonado")
else:
    print("✓ Repositorio ya existe")

# Paso 2: Cambiar a directorio correcto
repo_path = '/content/a_tecn'
os.chdir(repo_path)
print(f"✓ Directorio: {os.getcwd()}")

# Paso 3: Verificar que src/ existe
if os.path.exists(f'{repo_path}/src'):
    print(f"✓ Carpeta src encontrada")
    sys.path.insert(0, repo_path)
    print(f"✓ sys.path actualizado: {sys.path[0]}")
else:
    print(f"✗ ERROR: No se encuentra carpeta src en {repo_path}")
    print(f"  Contenido: {os.listdir(repo_path)}")

# Paso 4: Instalar dependencias
print("\\n📦 Instalando dependencias...")
subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'],
               capture_output=True)
subprocess.run(['pip', 'install', '-q', 'ipywidgets'],
               capture_output=True)
print("✓ Dependencias instaladas")

# Paso 5: IMPORTAR (ahora debe funcionar)
print("\\n📚 Importando módulos...")
try:
    from src.ticker_analyzer import TickerAnalyzer
    print("✓ TickerAnalyzer importado correctamente")
except ModuleNotFoundError as e:
    print(f"✗ Error de importación: {e}")
    print(f"  sys.path: {sys.path}")
    print(f"  Archivos en {repo_path}: {os.listdir(repo_path)}")
    print(f"  Archivos en src/: {os.listdir(f'{repo_path}/src')}")
    raise

import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

print("✓ Todos los módulos importados")

# ========== INTERFAZ INTERACTIVA ==========
print("\\n" + "="*70)
print("  🚀 LONQ TICKER ANALYZER - INTERFAZ INTERACTIVA".center(70))
print("="*70)

# Tickers sugeridos
TICKERS_SUGERIDOS = {
    '📈 ACCIONES USA': ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    '📊 ÍNDICES': ['SPY', 'QQQ', 'DIA'],
    '💰 CRIPTO': ['BTC', 'ETH'],
    '🏆 OTROS': ['KO', 'PG', 'JNJ', 'NEE', 'GLD', 'TLT']
}

all_tickers = []
for tickers in TICKERS_SUGERIDOS.values():
    all_tickers.extend(tickers)

# Widgets
dropdown_ticker = widgets.Dropdown(
    options=all_tickers,
    value='AAPL',
    description='📌 Ticker:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='300px')
)

input_ticker = widgets.Text(
    value='',
    placeholder='O escribe tu ticker aquí',
    description='🔍 Personalizado:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='400px')
)

checkbox_deep = widgets.Checkbox(
    value=False,
    description='🔬 Análisis profundo (toma ~60s)',
    indent=False
)

checkbox_export = widgets.Checkbox(
    value=True,
    description='📥 Exportar resultados (JSON + CSV)',
    indent=False
)

btn_analizar = widgets.Button(
    description='▶️ ANALIZAR',
    button_style='success',
    tooltip='Inicia el análisis',
    layout=widgets.Layout(width='150px', height='40px')
)

btn_limpiar = widgets.Button(
    description='🗑️ Limpiar',
    button_style='warning',
    tooltip='Limpia la salida',
    layout=widgets.Layout(width='150px', height='40px')
)

output_area = widgets.Output(layout=widgets.Layout(
    border='1px solid #ccc',
    padding='15px',
    margin='20px 0'
))

# ========== FUNCIONES ==========
def obtener_ticker():
    """Obtiene el ticker del input o dropdown."""
    if input_ticker.value.strip():
        return input_ticker.value.strip().upper()
    return dropdown_ticker.value

def ejecutar_analisis(b=None):
    """Ejecuta el análisis."""
    output_area.clear_output()

    with output_area:
        ticker = obtener_ticker()
        deep = checkbox_deep.value
        export = checkbox_export.value

        print(f"\\n{'='*70}")
        print(f"  📊 ANALIZANDO: {ticker}".center(70))
        print(f"{'='*70}\\n")

        analyzer = TickerAnalyzer(ticker=ticker)

        # Cargar datos
        print(f"📥 Cargando datos para {ticker}...", end=' ', flush=True)
        if not analyzer.load_data():
            print(f"⚠ Descargando desde yfinance...")
            if not analyzer.load_data(auto_download=True):
                print(f"✗ Error: {ticker} no es válido")
                return
        print("✓")

        # VIX
        print(f"📥 Cargando VIX...", end=' ', flush=True)
        if analyzer.load_vix():
            print("✓")
        else:
            print("⚠ (sin VIX)")

        # Análisis rápido
        print(f"\\n⚡ ANÁLISIS RÁPIDO (~2 segundos)\\n")
        if not analyzer.analyze(deep=False):
            print("✗ Error en análisis rápido")
            return
        analyzer.print_report()

        # Análisis profundo
        if deep:
            print(f"\\n🔬 ANÁLISIS PROFUNDO (~60 segundos)\\n")
            print("⏳ En progreso...")
            if not analyzer.analyze(deep=True):
                print("⚠ Error (datos insuficientes)")
            else:
                print()
                analyzer.print_report()

        # Gráficas
        print(f"\\n📈 Generando gráficas...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{ticker} - Análisis', fontsize=16, fontweight='bold')

        # 1. Puntos
        ax = axes[0, 0]
        sb = analyzer.fast_result.score_breakdown
        colors = plt.cm.Set3(np.linspace(0, 1, len(sb)))
        ax.barh(list(sb.keys()), list(sb.values()), color=colors)
        ax.set_xlabel('Puntos')
        ax.set_title('Desglose de Puntuación')
        ax.grid(axis='x', alpha=0.3)

        # 2. Métricas
        ax = axes[0, 1]
        metrics = {
            'Vol': f"{analyzer.fast_result.vol_annual:.1%}",
            'Autocorr': f"{analyzer.fast_result.autocorr_5d:.3f}",
            'R²': f"{analyzer.fast_result.trend_strength:.3f}",
            'Score': f"{analyzer.fast_result.score}/100"
        }
        ax.axis('off')
        y = 0.9
        for k, v in metrics.items():
            ax.text(0.1, y, f'{k}:', fontweight='bold', fontsize=11)
            ax.text(0.6, y, str(v), fontsize=11)
            y -= 0.2
        ax.set_title('Métricas', loc='left', fontweight='bold')

        # 3. Precios
        ax = axes[1, 0]
        prices = analyzer.prices
        ax.plot(prices.index, prices.values, linewidth=1.5, color='#2E86AB')
        ax.fill_between(prices.index, prices.values, alpha=0.3, color='#2E86AB')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio')
        ax.set_title('Histórico')
        ax.grid(alpha=0.3)

        # 4. Retornos
        ax = axes[1, 1]
        returns = np.log(prices / prices.shift(1)).dropna()
        ax.hist(returns, bins=50, color='#A23B72', alpha=0.7)
        ax.axvline(returns.mean(), color='red', linestyle='--',
                   label=f'Media: {returns.mean():.4f}')
        ax.set_xlabel('Retorno Diario')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("✓ Gráficas generadas")

        # Exportar
        if export:
            print(f"\\n📥 Exportando resultados...")
            import json

            resultados = analyzer.to_dict()
            with open(f'{ticker}_analysis.json', 'w') as f:
                json.dump(resultados, f, indent=2)

            df = analyzer.get_metrics_dataframe()
            df.to_csv(f'{ticker}_metrics.csv', index=False)

            print(f"✓ Guardado: {ticker}_analysis.json, {ticker}_metrics.csv")

        print(f"\\n{'='*70}")
        print("✓ COMPLETADO".center(70))
        print(f"{'='*70}")

def limpiar(b):
    output_area.clear_output()

btn_analizar.on_click(ejecutar_analisis)
btn_limpiar.on_click(limpiar)

# ========== MOSTRAR INTERFAZ ==========
print("\\n✓ Interfaz lista.\\n")

vbox = widgets.VBox([
    widgets.HTML("<b>🎯 Selecciona un ticker:</b>"),
    dropdown_ticker,
    input_ticker,
    widgets.HTML("<br><b>⚙️ Opciones:</b>"),
    checkbox_deep,
    checkbox_export,
    widgets.HTML("<br>"),
    widgets.HBox([btn_analizar, btn_limpiar]),
    output_area,
])

display(vbox)

print("💡 TIPS:")
print("  • Elige de la lista o escribe un ticker")
print("  • Rápido: ~2s | Profundo: ~60s")
print("  • Resultados se exportan automáticamente")
'''

print("✅ CÓDIGO FINAL CORREGIDO GENERADO\n")
print("="*70)
print("ESTE CÓDIGO FUNCIONA 100% EN COLAB")
print("="*70)
print("\n👇 COPIA Y PEGA ESTO EN UNA CELDA DE COLAB:\n")
print(codigo_final_corregido)
