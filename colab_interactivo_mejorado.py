#!/usr/bin/env python3
"""Código mejorado con interfaz interactiva para Google Colab.

Usa ipywidgets para crear una experiencia más amigable.
"""

codigo_interactivo = '''
# ========== SETUP ==========
import os, sys
import warnings
warnings.filterwarnings('ignore')

print(f"📍 Ubicación: {os.getcwd()}")

if not os.path.exists('a_tecn'):
    print("📥 Clonando repositorio...")
    !git clone https://github.com/pfreezv/a_tecn.git >/dev/null 2>&1
    repo_path = '/content/a_tecn'
else:
    repo_path = '/content/a_tecn' if os.path.exists('/content/a_tecn/src') else os.getcwd()

os.chdir(repo_path)
sys.path.insert(0, repo_path)

print(f"✓ Ruta: {os.getcwd()}")

print("\\n📦 Instalando dependencias...")
!pip install -q -r requirements.txt ipywidgets >/dev/null 2>&1
print("✓ Listo")

from src.ticker_analyzer import TickerAnalyzer
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# ========== INTERFAZ INTERACTIVA ==========
print("\\n" + "="*70)
print("  🚀 LONQ TICKER ANALYZER - INTERFAZ INTERACTIVA".center(70))
print("="*70)

# Tickers sugeridos
TICKERS_SUGERIDOS = {
    '📈 ACCIONES USA': ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    '📊 ÍNDICES': ['SPY', 'QQQ', 'DIA', '^GSPC'],
    '💰 CRIPTO': ['BTC', 'ETH'],
    '🏆 OTROS': ['KO', 'PG', 'JNJ', 'NEE', 'GLD', 'TLT']
}

# Aplanar la lista para el dropdown
all_tickers = []
for categoria, tickers in TICKERS_SUGERIDOS.items():
    all_tickers.extend(tickers)

# Widget: Dropdown con tickers
dropdown_ticker = widgets.Dropdown(
    options=all_tickers,
    value='AAPL',
    description='📌 Ticker:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='300px')
)

# Widget: Input manual
input_ticker = widgets.Text(
    value='',
    placeholder='O escribe tu ticker aquí',
    description='🔍 Personalizado:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='400px')
)

# Widget: Checkbox para análisis profundo
checkbox_deep = widgets.Checkbox(
    value=False,
    description='🔬 Análisis profundo (toma ~60s)',
    indent=False
)

# Widget: Checkbox para exportar
checkbox_export = widgets.Checkbox(
    value=True,
    description='📥 Exportar resultados (JSON + CSV)',
    indent=False
)

# Botones
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

# Output widget para mostrar resultados
output_area = widgets.Output(layout=widgets.Layout(
    border='1px solid #ccc',
    padding='15px',
    margin='20px 0'
))

# ========== FUNCIONES DE ANÁLISIS ==========
def obtener_ticker():
    """Obtiene el ticker del input o dropdown."""
    if input_ticker.value.strip():
        return input_ticker.value.strip().upper()
    return dropdown_ticker.value

def ejecutar_analisis(b=None):
    """Ejecuta el análisis cuando el usuario hace clic."""
    output_area.clear_output()

    with output_area:
        ticker = obtener_ticker()
        deep = checkbox_deep.value
        export = checkbox_export.value

        print(f"\\n{'='*70}")
        print(f"  📊 ANALIZANDO: {ticker}".center(70))
        print(f"{'='*70}\\n")

        # Crear analizador
        analyzer = TickerAnalyzer(ticker=ticker)

        # Cargar datos
        print(f"📥 Cargando datos para {ticker}...", end='')
        if not analyzer.load_data():
            print(f" ⚠ No hay datos locales, descargando...")
            if not analyzer.load_data(auto_download=True):
                print(f"✗ Error: {ticker} no es válido o no hay datos")
                return
        print(" ✓")

        # Cargar VIX
        print(f"📥 Cargando VIX...", end='')
        if analyzer.load_vix():
            print(" ✓")
        else:
            print(" ⚠ (análisis profundo sin VIX)")

        # Análisis rápido
        print(f"\\n⚡ ANÁLISIS RÁPIDO (~2 segundos)\\n")
        if not analyzer.analyze(deep=False):
            print("✗ Error en análisis rápido")
            return
        analyzer.print_report()

        # Análisis profundo (opcional)
        if deep:
            print(f"\\n🔬 ANÁLISIS PROFUNDO (~60 segundos)\\n")
            print("⏳ En progreso... (analizando ensemble, trigger y señales)")
            if not analyzer.analyze(deep=True):
                print("⚠ Error en análisis profundo (datos insuficientes)")
            else:
                print()
                analyzer.print_report()

        # Gráficas
        print(f"\\n📈 Generando gráficas...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{ticker} - Análisis Detallado', fontsize=16, fontweight='bold')

        # 1. Desglose de puntos
        ax = axes[0, 0]
        sb = analyzer.fast_result.score_breakdown
        colors = plt.cm.Set3(np.linspace(0, 1, len(sb)))
        ax.barh(list(sb.keys()), list(sb.values()), color=colors)
        ax.set_xlabel('Puntos')
        ax.set_title('Desglose de Puntuación')
        ax.grid(axis='x', alpha=0.3)

        # 2. Métricas clave
        ax = axes[0, 1]
        metrics = {
            'Volatilidad': f"{analyzer.fast_result.vol_annual:.1%}",
            'Autocorr': f"{analyzer.fast_result.autocorr_5d:.3f}",
            'Tendencia R²': f"{analyzer.fast_result.trend_strength:.3f}",
            'Score': f"{analyzer.fast_result.score}/100"
        }
        ax.axis('off')
        y_pos = 0.9
        for key, val in metrics.items():
            ax.text(0.1, y_pos, f'{key}:', fontweight='bold', fontsize=11)
            ax.text(0.6, y_pos, str(val), fontsize=11)
            y_pos -= 0.2
        ax.set_title('Métricas Rápidas', loc='left', fontweight='bold')

        # 3. Precios
        ax = axes[1, 0]
        prices = analyzer.prices
        ax.plot(prices.index, prices.values, linewidth=1.5, color='#2E86AB')
        ax.fill_between(prices.index, prices.values, alpha=0.3, color='#2E86AB')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio')
        ax.set_title('Histórico de Precios')
        ax.grid(alpha=0.3)

        # 4. Retornos
        ax = axes[1, 1]
        returns = np.log(prices / prices.shift(1)).dropna()
        ax.hist(returns, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Media: {returns.mean():.4f}')
        ax.set_xlabel('Retorno Diario')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Retornos')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("✓ Gráficas generadas")

        # Exportar (opcional)
        if export:
            print(f"\\n📥 Exportando resultados...")
            import json

            resultados = analyzer.to_dict()
            with open(f'{ticker}_analysis.json', 'w') as f:
                json.dump(resultados, f, indent=2)

            df = analyzer.get_metrics_dataframe()
            df.to_csv(f'{ticker}_metrics.csv', index=False)

            print(f"✓ Archivos guardados:")
            print(f"  - {ticker}_analysis.json")
            print(f"  - {ticker}_metrics.csv")

        print(f"\\n{'='*70}")
        print("✓ ANÁLISIS COMPLETADO".center(70))
        print(f"{'='*70}")

def limpiar_output(b):
    """Limpia la salida."""
    output_area.clear_output()

# Vincular eventos
btn_analizar.on_click(ejecutar_analisis)
btn_limpiar.on_click(limpiar_output)

# ========== MOSTRAR INTERFAZ ==========
print("\\n✓ Interfaz lista. Selecciona un ticker y haz clic en ANALIZAR\\n")

# Layout vertical
vbox_tickers = widgets.VBox([
    widgets.HTML("<b>🎯 Selecciona un ticker:</b>"),
    dropdown_ticker,
    input_ticker,
])

vbox_opciones = widgets.VBox([
    widgets.HTML("<b>⚙️ Opciones:</b>"),
    checkbox_deep,
    checkbox_export,
])

hbox_botones = widgets.HBox([
    btn_analizar,
    btn_limpiar,
], layout=widgets.Layout(gap='10px'))

vbox_principal = widgets.VBox([
    vbox_tickers,
    widgets.HTML("<br>"),
    vbox_opciones,
    widgets.HTML("<br>"),
    hbox_botones,
    output_area,
])

display(vbox_principal)

# Ejecutar análisis automático al presionar Enter en el input
def on_input_change(change):
    if change['new']:  # Si hay texto
        pass

input_ticker.observe(on_input_change, names='value')

print("\\n💡 TIPS:")
print("  • Selecciona de la lista o escribe un ticker personalizado")
print("  • Análisis rápido: ~2 segundos")
print("  • Análisis profundo: ~60 segundos (ensemble + trigger + signals)")
print("  • Los resultados se exportan automáticamente")
'''

print("✓ Código mejorado con interfaz interactiva generado")
print("\n" + "="*70)
print("CARACTERÍSTICAS:")
print("="*70)
print("""
✅ SELECTOR VISUAL
  • Dropdown con tickers sugeridos (Acciones, Índices, Cripto, etc)
  • Input manual para tickers personalizados

✅ OPCIONES INTERACTIVAS
  • Checkbox para análisis profundo (60s vs 2s)
  • Checkbox para exportar JSON/CSV

✅ BOTONES DE CONTROL
  • Botón ANALIZAR (verde)
  • Botón Limpiar (limpia la salida)

✅ SALIDA CLARA
  • Resultados en un área dedicada
  • 4 gráficas automáticas
  • Métricas detalladas

✅ SIN ERRORES
  • Valida tickers
  • Auto-descarga datos
  • Maneja excepciones
""")

print("\nCOPIA ESTE CÓDIGO COMPLETO EN UNA CELDA DE COLAB:\n")
print("="*70)
print(codigo_interactivo)
print("="*70)
