# ⚡ Inicio Rápido en Google Colab

**Solo necesitas un ticker para hacer un análisis completo.**

## 🚀 3 Pasos para Analizar Cualquier Ticker

### 1️⃣ Abre Google Colab

Ve a [colab.google.com](https://colab.research.google.com)

### 2️⃣ Carga el Notebook

- **Opción A**: Sube el archivo `lonq_analyzer_colab.ipynb`
- **Opción B**: Crea un notebook nuevo y copia las celdas de abajo

### 3️⃣ Ejecuta y Ingresa tu Ticker

Ejecuta las celdas en orden:

1. **Celda 1**: Setup (instala todo)
2. **Celda 2**: Importa módulos
3. **Celda 3**: Ingresa el ticker (ej: AAPL, TSLA, SPY, BTC)
4. **Celda 4-5**: Carga datos (descarga automáticamente si no existen)
5. **Celda 6**: Análisis rápido (~2s)
6. **Celda 7**: Análisis profundo (~60s) ← **RECOMENDADO**
7. **Celda 8**: Ve las gráficas
8. **Celda 9**: Descarga resultados

---

## 📋 Código Mínimo para Colab

Si solo quieres el código sin notebook:

```python
# 1. Setup
!git clone https://github.com/pfreezv/a_tecn.git
%cd a_tecn
!pip install -q -r requirements.txt

# 2. Importar
import sys
sys.path.insert(0, '/content/a_tecn')
from src.ticker_analyzer import TickerAnalyzer

# 3. Elegir ticker (cambiar AAPL por el que quieras)
ticker = "AAPL"

# 4. Analizar
analyzer = TickerAnalyzer(ticker=ticker)
analyzer.load_data(auto_download=True)  # Descarga automáticamente si no existe
analyzer.load_vix()

# 5. Análisis rápido
analyzer.analyze(deep=False)
analyzer.print_report()

# 6. Análisis profundo (opcional, ~60s)
analyzer.analyze(deep=True)
analyzer.print_report()

# 7. Gráficas
import matplotlib.pyplot as plt
# ... (ver notebook para código de gráficas)
```

---

## 🎯 Ejemplos de Tickers

**Acciones USA**:
- AAPL (Apple)
- TSLA (Tesla)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)

**Índices**:
- SPY (S&P 500)
- QQQ (Nasdaq)
- DIA (Dow Jones)

**Otros**:
- BTC (Bitcoin)
- GLD (Gold)
- TLT (Bonos USA)

---

## ❓ FAQ

**P: ¿Qué ticker debo analizar?**
A: Cualquiera. El script busca datos locales primero y descarga automáticamente de yfinance si no los encuentra.

**P: ¿Cuánto tarda?**
A: 
- Análisis rápido: ~2 segundos
- Análisis profundo: ~60 segundos (ensemble, trigger, señales)

**P: ¿Qué es el análisis profundo?**
A: Análisis completo con:
- Clustering (K-Means, GMM, HMM)
- Detección de anomalías multi-timeframe
- Señales combinadas y backtesting
- Sharpe ratio vs Buy & Hold

**P: ¿Qué significa la puntuación?**
A: 
- ✓✓ ≥70: Excelente candidato
- ✓ 50-69: Buen candidato
- ~ 30-49: Marginal
- ✗ <30: No recomendado

---

## 🔄 Flujo Típico

1. Ingresa ticker (ej: MSFT)
2. Se descarga automáticamente si no existe (5-10s)
3. Análisis rápido (~2s) → ves métricas básicas
4. Análisis profundo (~60s) → ves análisis completo
5. Gráficas → visualizas los datos
6. Exporta → descarga JSON y CSV

---

## 📂 Archivos Necesarios

El repo tiene todo que necesitas:
- ✓ Módulo analizador (`src/ticker_analyzer.py`)
- ✓ Notebook Colab (`lonq_analyzer_colab.ipynb`)
- ✓ Datos de ejemplo (AAPL, TSLA, SPY, etc)
- ✓ VIX para análisis profundo

---

## 🎓 Próximos Pasos

- Lee `README_ANALYZER.md` para más detalles
- Ver `COLAB_GUIDE.md` para guía completa
- Ejecutar `example_usage.py` para ver todos los casos

---

**¿Listo? Ve a [colab.research.google.com](https://colab.research.google.com) y comienza!** 🚀
