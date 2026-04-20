# 🎨 LONQ Analyzer - Versión Interactiva con Interfaz Gráfica

**Versión MEJORADA con widgets visuales para Google Colab**

El usuario puede elegir el ticker de varias formas, con una interfaz amigable y profesional.

---

## 🚀 Cómo Usar

### 1️⃣ En Google Colab

Abre [colab.google.com](https://colab.research.google.com) y ejecuta este código en **UNA SOLA CELDA**:

```python
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

print("\n📦 Instalando dependencias...")
!pip install -q -r requirements.txt ipywidgets >/dev/null 2>&1
print("✓ Listo")

from src.ticker_analyzer import TickerAnalyzer
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# ========== INTERFAZ INTERACTIVA ==========
print("\n" + "="*70)
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

        print(f"\n{'='*70}")
        print(f"  📊 ANALIZANDO: {ticker}".center(70))
        print(f"{'='*70}\n")

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
        print(f"\n⚡ ANÁLISIS RÁPIDO (~2 segundos)\n")
        if not analyzer.analyze(deep=False):
            print("✗ Error en análisis rápido")
            return
        analyzer.print_report()

        # Análisis profundo (opcional)
        if deep:
            print(f"\n🔬 ANÁLISIS PROFUNDO (~60 segundos)\n")
            print("⏳ En progreso... (analizando ensemble, trigger y señales)")
            if not analyzer.analyze(deep=True):
                print("⚠ Error en análisis profundo (datos insuficientes)")
            else:
                print()
                analyzer.print_report()

        # Gráficas
        print(f"\n📈 Generando gráficas...")
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
            print(f"\n📥 Exportando resultados...")
            import json

            resultados = analyzer.to_dict()
            with open(f'{ticker}_analysis.json', 'w') as f:
                json.dump(resultados, f, indent=2)

            df = analyzer.get_metrics_dataframe()
            df.to_csv(f'{ticker}_metrics.csv', index=False)

            print(f"✓ Archivos guardados:")
            print(f"  - {ticker}_analysis.json")
            print(f"  - {ticker}_metrics.csv")

        print(f"\n{'='*70}")
        print("✓ ANÁLISIS COMPLETADO".center(70))
        print(f"{'='*70}")

def limpiar_output(b):
    """Limpia la salida."""
    output_area.clear_output()

# Vincular eventos
btn_analizar.on_click(ejecutar_analisis)
btn_limpiar.on_click(limpiar_output)

# ========== MOSTRAR INTERFAZ ==========
print("\n✓ Interfaz lista. Selecciona un ticker y haz clic en ANALIZAR\n")

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

print("\n💡 TIPS:")
print("  • Selecciona de la lista o escribe un ticker personalizado")
print("  • Análisis rápido: ~2 segundos")
print("  • Análisis profundo: ~60 segundos (ensemble + trigger + signals)")
print("  • Los resultados se exportan automáticamente")
```

---

## ✨ Características

### 🎯 Interfaz Gráfica Completa

```
┌─────────────────────────────────────────────────────────┐
│  🚀 LONQ TICKER ANALYZER - INTERFAZ INTERACTIVA        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  🎯 Selecciona un ticker:                             │
│  ┌─ Dropdown: [AAPL ▼]  (Sugeridos)                   │
│  └─ Input:   [MSFT_______________]  (Personalizado)   │
│                                                         │
│  ⚙️ Opciones:                                          │
│  ☐ 🔬 Análisis profundo (toma ~60s)                   │
│  ☑ 📥 Exportar resultados (JSON + CSV)                │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐                   │
│  │ ▶️ ANALIZAR  │  │ 🗑️ Limpiar   │                   │
│  └──────────────┘  └──────────────┘                   │
│                                                         │
│  ┌─ RESULTADOS ─────────────────────────────────────┐ │
│  │ (Se muestran aquí)                                │ │
│  │ - Análisis rápido                                 │ │
│  │ - Análisis profundo (si está activado)            │ │
│  │ - 4 Gráficas automáticas                          │ │
│  │ - Archivos exportados                             │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 🎮 Opciones del Usuario

### 1️⃣ **Seleccionar Ticker**

**Opción A**: Dropdown con sugerencias
- 📈 ACCIONES USA: AAPL, MSFT, TSLA, GOOGL, AMZN, META, NVDA
- 📊 ÍNDICES: SPY, QQQ, DIA
- 💰 CRIPTO: BTC, ETH
- 🏆 OTROS: KO, PG, JNJ, NEE, GLD, TLT

**Opción B**: Input manual
- Escribe cualquier ticker válido (ej: PARA, UNILEVER, etc)

### 2️⃣ **Activar Análisis Profundo**

```
☐ 🔬 Análisis profundo (toma ~60s)
```

- **Desactivado** (default): Solo análisis rápido (~2s)
- **Activado**: Incluye ensemble, trigger, signals (~60s total)

### 3️⃣ **Exportar Resultados**

```
☑ 📥 Exportar resultados (JSON + CSV)
```

- **Activado** (default): Guarda automáticamente
  - `{ticker}_analysis.json` - Todos los resultados
  - `{ticker}_metrics.csv` - Tabla de métricas

### 4️⃣ **Botones de Control**

- **▶️ ANALIZAR** - Inicia el análisis (verde)
- **🗑️ Limpiar** - Limpia la salida anterior (naranja)

---

## 📊 Salida del Análisis

Una vez que presionas "ANALIZAR", ves:

```
======================================================================
  📊 ANALIZANDO: AAPL
======================================================================

📥 Cargando datos para AAPL... ✓
📥 Cargando VIX... ✓

⚡ ANÁLISIS RÁPIDO (~2 segundos)

======================================================================
  ANÁLISIS DE AAPL
======================================================================

🎯 PUNTUACIÓN: 72/100  [✓]
   Recomendación: Buen candidato...

📈 MÉTRICAS RÁPIDAS
   Datos históricos:     2,009 días
   Volatilidad anual:    28.45%
   Autocorrelación:      -0.0621
   Tendencia (R²):       0.6234

📊 DESGLOSE DE PUNTOS
   histórico:                   10 pts
   volatilidad:                 25 pts
   reversión_autocorr:          18 pts
   anti_tendencia:              12 pts
   consistencia:                 7 pts

(Si activaste análisis profundo, también ves):

🔬 ANÁLISIS PROFUNDO
   Silhouette ensemble:  0.2834
   Score reversión:      0.5421
   Win rate 10d:         58.92%
   Sharpe estrategia:    0.7823
   Sharpe Buy & Hold:    0.6234

📈 Generando gráficas...
(4 gráficas: puntos, métricas, precios, retornos)

✓ Gráficas generadas

📥 Exportando resultados...
✓ Archivos guardados:
  - AAPL_analysis.json
  - AAPL_metrics.csv

======================================================================
✓ ANÁLISIS COMPLETADO
======================================================================
```

---

## 💡 Tips

1. **Análisis rápido primero** → Desactiva profundo para pruebas rápidas
2. **Tickers sugeridos** → Usa el dropdown para tickers populares
3. **Personalizado** → Escribe cualquier ticker válido en el input
4. **Exporta siempre** → Dejar activado para guardar resultados
5. **Compara tickers** → Ejecuta múltiples análisis y compara JSON

---

## 🔄 Flujo Típico

```
1. Abre la celda en Colab
   ↓
2. Ejecuta (Ctrl+Enter)
   ↓
3. Ve la interfaz interactiva
   ↓
4. Elige ticker (dropdown o escribe)
   ↓
5. Activa análisis profundo (opcional)
   ↓
6. Haz clic en "▶️ ANALIZAR"
   ↓
7. Espera resultados (2-60s)
   ↓
8. Ve:
   - Reportes detallados
   - 4 Gráficas
   - Archivos descargables
   ↓
9. Presiona "▶️ ANALIZAR" de nuevo para otro ticker
```

---

## 🎯 Casos de Uso

### Usar el Dropdown
```
Haz clic en la caja dropdown → Elige TSLA → Haz clic en ANALIZAR
```

### Usar Input Personal
```
Escribe "UNILEVER" en el input → Haz clic en ANALIZAR
```

### Comparar Tickers
```
1. Analiza AAPL → Guarda resultados
2. Limpia con botón 🗑️
3. Analiza MSFT → Compara JSON
```

### Análisis Profundo
```
1. Activa ☑ 🔬 Análisis profundo
2. Haz clic en ANALIZAR
3. Espera ~60 segundos
4. Ve ensemble + trigger + signals
```

---

## 🐛 Troubleshooting

**P: No puedo encontrar mi ticker**
R: Usa el input personalizado para escribirlo. Se descargará automáticamente.

**P: El análisis profundo es muy lento**
R: Normal, toma ~60s. Desactívalo si solo quieres métricas rápidas.

**P: Los archivos no se descargan**
R: Están en `/content/a_tecn/{ticker}*.csv` - Colab te pide permiso.

---

**¿Listo para usar?** Copia el código arriba en Colab y ¡disfruta! 🚀
