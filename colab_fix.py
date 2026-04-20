#!/usr/bin/env python3
"""Código corregido para Colab - Maneja correctamente los directorios."""

codigo_corregido = '''
# ========== SETUP (Ejecuta ESTO primero) ==========
import os
import sys

# Verificar dónde estamos
print(f"📍 Ubicación: {os.getcwd()}")

# Si ya está clonado, no clonar de nuevo
if not os.path.exists('a_tecn'):
    print("📥 Clonando repositorio...")
    !git clone https://github.com/pfreezv/a_tecn.git
    repo_path = '/content/a_tecn'
else:
    # Ya está clonado
    if os.path.exists('/content/a_tecn/src'):
        repo_path = '/content/a_tecn'
    else:
        repo_path = os.getcwd()

print(f"✓ Ruta del repo: {repo_path}")

# Cambiar a la ruta correcta
os.chdir(repo_path)
sys.path.insert(0, repo_path)

print(f"✓ Directorio actual: {os.getcwd()}")
print(f"✓ sys.path actualizado")

# Instalar dependencias
print("\\n📦 Instalando dependencias...")
!pip install -q -r requirements.txt
print("✓ Listo")

# ========== IMPORTAR ==========
print("\\n📚 Importando módulos...")
import warnings
warnings.filterwarnings('ignore')

from src.ticker_analyzer import TickerAnalyzer
print("✓ TickerAnalyzer importado")

# ========== ELEGIR TICKER ==========
print("\\n" + "="*60)
ticker = input("Ingresa el ticker (ej: AAPL, TSLA, SPY, KO, BTC): ").upper().strip()

if not ticker:
    ticker = "AAPL"
    print(f"⚠ Usando default: {ticker}")

print(f"\\n📊 Analizando: {ticker}")
print("="*60)

# ========== CARGAR DATOS ==========
print(f"\\n📥 Cargando datos para {ticker}...")
analyzer = TickerAnalyzer(ticker=ticker)

# Auto-descargar si no existen
if not analyzer.load_data():
    print(f"⚠ No hay datos locales, intentando descargar...")
    if analyzer.load_data(auto_download=True):
        print("✓ Datos descargados exitosamente")
    else:
        print("✗ Error al descargar datos")
        print("  Verifica que el ticker sea válido")
else:
    print("✓ Datos cargados localmente")

# Cargar VIX
print("\\nCargando VIX...")
if analyzer.load_vix():
    print("✓ VIX cargado")
else:
    print("⚠ VIX no disponible (análisis profundo sin VIX)")

# ========== ANÁLISIS RÁPIDO ==========
print("\\n⚡ ANÁLISIS RÁPIDO (~2 segundos)\\n")
if analyzer.analyze(deep=False):
    analyzer.print_report()
else:
    print("✗ Error en análisis rápido")

# ========== ANÁLISIS PROFUNDO ==========
print("\\n🔬 ANÁLISIS PROFUNDO (~60 segundos)\\n")
print("⏳ En progreso... (paciencia, está analizando ensemble, trigger y señales)")

if analyzer.analyze(deep=True):
    analyzer.print_report()
    print("\\n✓ Análisis profundo completado")
else:
    print("✗ Error en análisis profundo")
    print("  (Algunos datos pueden ser insuficientes)")

# ========== GRÁFICAS ==========
print("\\n📈 Generando gráficas...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'{ticker} - Análisis Detallado', fontsize=16, fontweight='bold')

# 1. Desglose de puntos
ax = axes[0, 0]
score_breakdown = analyzer.fast_result.score_breakdown
colors = plt.cm.Set3(np.linspace(0, 1, len(score_breakdown)))
ax.barh(list(score_breakdown.keys()), list(score_breakdown.values()), color=colors)
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

# 3. Distribución de precios
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

# ========== EXPORTAR ==========
print("\\n📥 Exportando resultados...")
import json

resultados = analyzer.to_dict()
with open(f'{ticker}_analysis.json', 'w') as f:
    json.dump(resultados, f, indent=2)

df = analyzer.get_metrics_dataframe()
df.to_csv(f'{ticker}_metrics.csv', index=False)

print(f"✓ Archivos guardados:")
print(f"  - {ticker}_analysis.json")
print(f"  - {ticker}_metrics.csv")

print("\\n" + "="*60)
print("✓ ANÁLISIS COMPLETADO")
print("="*60)
'''

print("✓ Código corregido generado")
print("\nCopia este código completo en una celda de Colab:")
print("\n" + "="*70)
print(codigo_corregido)
print("="*70)
