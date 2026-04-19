# 📊 LONQ Ticker Analyzer

Analiza tickers individuales con métricas detalladas para evaluar su idoneidad en la estrategia LONQ de reversión.

## ⚡ Inicio Rápido

### Línea de Comandos

```bash
# Análisis rápido (2 segundos)
python analyze_ticker.py AAPL

# Análisis profundo (60 segundos)
python analyze_ticker.py AAPL --deep

# Con parámetros personalizados
python analyze_ticker.py TSLA --deep --threshold 2.5 --horizon 15

# Exportar a JSON
python analyze_ticker.py AAPL --json > resultados.json
```

### Python

```python
from src.ticker_analyzer import TickerAnalyzer

# Crear analizador
analyzer = TickerAnalyzer(ticker="AAPL")

# Cargar datos
analyzer.load_data()

# Análisis rápido
analyzer.analyze(deep=False)
analyzer.print_report()

# O análisis profundo
analyzer.load_vix()
analyzer.analyze(deep=True)
analyzer.print_report()
```

### Google Colab

```python
!git clone https://github.com/pfreezv/a_tecn.git
%cd a_tecn
!pip install -q -r requirements.txt

from src.ticker_analyzer import TickerAnalyzer

analyzer = TickerAnalyzer(ticker="AAPL")
analyzer.load_data()
analyzer.analyze(deep=True)
analyzer.print_report()
```

## 📋 Características

### TickerAnalyzer

Módulo Python para análisis individual de tickers:

```python
analyzer = TickerAnalyzer(ticker="AAPL")

# Cargar datos
analyzer.load_data()                    # Busca automáticamente
analyzer.load_data("path/to/data.csv")  # O especificar ruta
analyzer.load_vix()                     # Datos de VIX para análisis profundo

# Ejecutar análisis
analyzer.analyze(
    deep=True,          # Análisis profundo (default: False)
    threshold=2.0,      # Umbral de anomalías
    horizon=10          # Horizonte de predicción
)

# Salida
analyzer.print_report()              # Imprime reporte formateado
analyzer.to_dict()                   # Retorna diccionario
analyzer.get_metrics_dataframe()     # Retorna DataFrame
```

### Script CLI

```bash
analyze_ticker.py TICKER [opciones]

Opciones:
  --deep              Análisis profundo
  --data PATH         Ruta al archivo CSV
  --vix PATH          Ruta a datos de VIX
  --threshold FLOAT   Umbral de anomalías (default: 2.0)
  --horizon INT       Horizonte en días (default: 10)
  --json              Salida en JSON
```

## 📊 Métricas Explicadas

### Puntuación (0-100)

| Score | Grado | Interpretación |
|-------|-------|---|
| ≥ 70 | ✓✓ | Excelente candidato |
| 50-69 | ✓ | Buen candidato |
| 30-49 | ~ | Candidato marginal |
| < 30 | ✗ | No recomendado |

### Componentes de Puntuación

1. **Histórico (0-10 pts)**
   - 500-999 días: 5 pts
   - ≥1000 días: 10 pts
   - <500 días: 0 pts

2. **Volatilidad (0-25 pts)**
   - Ideal 10-28%: 25 pts
   - 28-55%: 12 pts
   - <10%: 8 pts
   - >55%: 0 pts (descalifica)

3. **Reversión - Autocorrelación (0-30 pts)**
   - Autocorr < -0.08: 30 pts (fuerte reversión)
   - Autocorr < -0.03: 18 pts
   - Autocorr ≈ 0: 8 pts (neutral)
   - Autocorr > 0: 0 pts (momentum)

4. **Tendencia - R² (0-20 pts)**
   - R² < 0.50: 20 pts (sin tendencia dominante)
   - R² < 0.70: 12 pts
   - R² < 0.85: 5 pts
   - R² ≥ 0.85: 0 pts (tendencia fuerte)

5. **Consistencia (0-15 pts)**
   - Baja volatilidad de volatilidad: 15 pts
   - Media: 8 pts
   - Alta: 0 pts

### Análisis Profundo (adicionales)

- **Silhouette (0-20 pts)**: Calidad de clustering en ensemble
- **Reversión Trigger (0-20 pts)**: Señal de anomalías multi-timeframe
- **Win Rate (0-20 pts)**: % de señales acertadas en 10 días
- **Sharpe Relativo (-10 a +10 pts)**: Comparación vs Buy & Hold

## 📁 Estructura

```
A_tecn/
├── analyze_ticker.py          # Script CLI principal
├── example_usage.py            # Ejemplos de uso
├── COLAB_GUIDE.md             # Guía para Google Colab
├── README_ANALYZER.md         # Este archivo
├── src/
│   ├── screener.py            # Módulo base de screening
│   ├── ticker_analyzer.py     # ✨ NUEVO: Analizador individual
│   ├── ensemble.py
│   ├── trigger.py
│   ├── combined_signal.py
│   └── ...
├── tickers.yaml               # Lista de tickers disponibles
├── *_data.csv                 # Datos históricos (AAPL, TSLA, SPY, etc)
└── tests/
```

## 🔄 Workflow de Análisis

```
Ticker Input (AAPL)
    ↓
Load Data (CSV automático o personalizado)
    ↓
Fast Screen (2s)
    ├─ Volatilidad
    ├─ Autocorrelación
    ├─ Tendencia (R²)
    ├─ Consistencia
    └─ Score 0-100
    ↓
[Opcional] Deep Analysis (60s)
    ├─ Ensemble (K-Means, GMM, HMM)
    ├─ Multi-TF Trigger
    ├─ Combined Signals
    ├─ P&L Backtest
    └─ Score mejorado
    ↓
Report (texto, JSON, DataFrame)
```

## 💡 Casos de Uso

### 1. Evaluar un Ticker Rápido

```bash
python analyze_ticker.py AAPL
```

### 2. Análisis Profundo para Due Diligence

```bash
python analyze_ticker.py AAPL --deep --vix vix_data.csv
```

### 3. Comparar Candidatos

```python
from src.ticker_analyzer import TickerAnalyzer
import pandas as pd

resultados = []
for ticker in ["AAPL", "MSFT", "GOOGL"]:
    analyzer = TickerAnalyzer(ticker)
    analyzer.load_data()
    analyzer.analyze()
    resultados.append(analyzer.get_metrics_dataframe())

df = pd.concat(resultados)
df.sort_values("Score", ascending=False)
```

### 4. Automatizar en Colab

Ver `COLAB_GUIDE.md` para template listo para copiar/pegar.

## 🛠️ Requisitos

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
scipy>=1.7.0
```

Instalar con:
```bash
pip install -r requirements.txt
```

## 📝 Ejemplos

Ver `example_usage.py` para código ejecutable con múltiples casos de uso.

Ejecutar con:
```bash
python example_usage.py
```

## 🔗 Integración con Colab

Copia este código en una celda de Colab:

```python
!git clone https://github.com/pfreezv/a_tecn.git
%cd a_tecn
!pip install -q -r requirements.txt

from src.ticker_analyzer import TickerAnalyzer

# Tu análisis aquí
analyzer = TickerAnalyzer(ticker="AAPL")
analyzer.load_data()
analyzer.analyze(deep=False)
analyzer.print_report()
```

## 📈 Datos Disponibles

El repositorio incluye datos CSV para:
- `aapl_data.csv` - Apple
- `tsla_data.csv` - Tesla
- `spy_data.csv` - S&P 500 ETF
- `ko_data.csv` - Coca-Cola
- `btc_data.csv` - Bitcoin
- `vix_data.csv` - VIX Index

Para otros tickers, descargar con yfinance o proporcionar CSV personalizado.

## 🚨 Troubleshooting

**Error: "No se encontraron datos"**
- Asegúrate que `ticker_data.csv` exista
- O usa `analyzer.load_data("ruta/archivo.csv")`

**Error: "ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**Análisis profundo muy lento**
- Usa `analyze(deep=False)` para rápido (~2s)
- Deep toma ~60s por las 4 capas de análisis

## 📧 Feedback

¿Sugerencias? Abre una issue en el repositorio.

---

**Última actualización**: 2024-04-19
**Versión**: 1.0 - TickerAnalyzer Release
