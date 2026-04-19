# 🚀 Guía de Uso en Google Colab

Analiza cualquier ticker con métricas detalladas desde Google Colab.

## Instalación Rápida

Copia y ejecuta en la primera celda de Colab:

```python
# Clonar repositorio
!git clone https://github.com/pfreezv/a_tecn.git
%cd a_tecn

# Instalar dependencias
!pip install -q -r requirements.txt
```

## Uso Básico

### 1. Análisis Rápido (2 segundos)

```python
import sys
sys.path.insert(0, '/content/a_tecn')

from src.ticker_analyzer import TickerAnalyzer

# Crear analizador para un ticker
analyzer = TickerAnalyzer(ticker="AAPL")

# Cargar datos (busca automáticamente)
analyzer.load_data()

# Ejecutar análisis rápido
analyzer.analyze(deep=False)

# Ver reporte
analyzer.print_report()
```

### 2. Análisis Profundo (60 segundos)

```python
# Cargar VIX si está disponible
analyzer.load_vix()

# Ejecutar análisis profundo
analyzer.analyze(deep=True, threshold=2.0, horizon=10)

# Ver reporte detallado
analyzer.print_report()
```

### 3. Exportar Resultados como JSON

```python
import json

# Obtener diccionario
resultados = analyzer.to_dict()

# Guardar como JSON
with open("resultados_aapl.json", "w") as f:
    json.dump(resultados, f, indent=2)

print(json.dumps(resultados, indent=2))
```

### 4. Comparar Múltiples Tickers

```python
tickers = ["AAPL", "TSLA", "SPY"]
resultados = []

for ticker in tickers:
    analyzer = TickerAnalyzer(ticker=ticker)
    if analyzer.load_data():
        analyzer.analyze(deep=False)
        resultados.append(analyzer.get_metrics_dataframe())

# Combinar en una tabla
import pandas as pd
df_comparacion = pd.concat(resultados, ignore_index=True)
df_comparacion
```

## Datos Disponibles

El repositorio incluye datos CSV para estos tickers:

- `aapl_data.csv` - Apple
- `tsla_data.csv` - Tesla
- `spy_data.csv` - S&P 500 ETF
- `ko_data.csv` - Coca-Cola
- `btc_data.csv` - Bitcoin
- `vix_data.csv` - Volatility Index

## Usar Datos Personalizados

Si quieres analizar un ticker que no está en el repositorio:

```python
# Descargar datos (ejemplo con yfinance)
!pip install -q yfinance

import yfinance as yf

# Descargar datos
df = yf.download("MSFT", start="2020-01-01", progress=False)
df.to_csv("msft_data.csv")

# Analizar
analyzer = TickerAnalyzer(ticker="MSFT")
analyzer.load_data("msft_data.csv")
analyzer.analyze(deep=True)
analyzer.print_report()
```

## Interpretación de Resultados

### Puntuación (0-100)

| Rango | Interpretación | Símbolo |
|-------|---|---|
| ≥ 70 | Excelente candidato | ✓✓ |
| 50-69 | Buen candidato | ✓ |
| 30-49 | Candidato marginal | ~ |
| < 30 | No recomendado | ✗ |

### Métricas Clave

- **Volatilidad anual**: Ideal 10-28%. Muy alta (>55%) descalifica.
- **Autocorrelación (lag-5)**: Negativa indica reversión (bueno para nuestra estrategia)
- **Tendencia (R²)**: Baja es mejor (evita activos con tendencia dominante)
- **Silhouette**: Calidad de clustering (0.35+ = excelente)
- **Win Rate 10d**: % de señales correctas en 10 días
- **Sharpe**: Retorno ajustado por riesgo

## Reportar Resultados

Guarda los análisis para referencia:

```python
# Guardar reporte como texto
with open("reporte_aapl.txt", "w") as f:
    # Redirect print a archivo
    import io
    from contextlib import redirect_stdout
    
    with redirect_stdout(f):
        analyzer.print_report()

# Descargar desde Colab
from google.colab import files
files.download("reporte_aapl.txt")
```

## Troubleshooting

### Error: "No se encontraron datos para TICKER"

Asegúrate de:
1. Que el archivo `ticker_data.csv` exista en el repositorio
2. O que lo cargues manualmente: `analyzer.load_data("ruta/al/archivo.csv")`

### Error: "ModuleNotFoundError"

Ejecuta en una celda:
```python
!pip install -r requirements.txt
```

### El análisis profundo es muy lento

El análisis profundo (~60s) incluye:
- Ensemble (K-Means, GMM, HMM)
- Multi-timeframe trigger
- Señales combinadas

Usa `deep=False` para análisis rápido (~2s).

## Ejemplos Completos

Ver `demo.py` para más ejemplos.

---

¿Preguntas? Abre una issue en el repositorio.
