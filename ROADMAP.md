# LONQ — Roadmap de desarrollo

## Pendiente para próximas sesiones

### 1. Análisis de coste de oportunidad
El sistema está en mercado solo ~7% del tiempo. El 93% restante el capital
está en cash. Modelar ese rendimiento (~5% anual en T-bills/money market)
y añadirlo a las métricas de P&L para tener una comparación real vs B&H.

### 2. Colab v2 — prueba con ticker real
Ejecutar `lonq_colab_v2.ipynb` de principio a fin con un ticker real
(KO, NEE o PG) y verificar que las 4 capas + P&L + dashboard funcionan
correctamente sin errores.

### 3. Backtesting más robusto — Walk-forward validation
El scoring actual puede tener overfitting porque entrena y valida sobre
el mismo período histórico. Implementar walk-forward:
- Dividir histórico en ventanas (ej. 2 años train / 6 meses test)
- Mover la ventana hacia adelante y promediar métricas
- Añadir el resultado al `ScreenResult` como `wf_score`

### 4. Ampliar histórico por ticker — más de 100 días mínimo
El umbral actual en `screen_tickers.py` descarta tickers con < 100 días.
Subir el mínimo a 500 días (ya definido en `screener.py` como `MIN_DAYS`)
y asegurar consistencia entre ambos archivos.

### 5. Ejecución completa vía Discord + GitHub Actions
Idea: el usuario envía un mensaje a Discord con el ticker deseado
(ej. `!analyze KO`) y GitHub Actions ejecuta el análisis completo
y responde con los resultados en el mismo canal.

Arquitectura posible:
```
Usuario → Discord (comando !analyze TICKER)
        → Discord Bot (escucha mensajes)
        → GitHub API (dispara workflow_dispatch con input TICKER)
        → GitHub Actions (ejecuta análisis completo ~5 min)
        → resultados → Discord embed con las 4 capas + señal actual
```

Piezas necesarias:
- Bot de Discord con permisos de lectura de mensajes (discord.py o
  un bot serverless en Railway/Fly.io)
- GitHub Personal Access Token con scope `repo` para disparar workflows
- Modificar `daily_scan.yml` para aceptar input `ticker` individual
- Nuevo script `analyze_one.py` que corre las 4 capas y genera embed

---

## En curso — Multi-Index Scanner (rama: `feature/multi-index-scanner`)

### FASE 0 — PoC validación de datos (activa)
Validar antes de construir infraestructura:
- Bajar ~30 tickers IBEX 35 + Euro Stoxx con sufijos yfinance (`.MC`, `.PA`, `.DE`, `.AS`, `.MI`)
  y confirmar histórico limpio, sin huecos excesivos y con ≥ 500 días
- Bajar 8 cryptos principales (`BTC-USD`, `ETH-USD`, `SOL-USD`, `BNB-USD`, `XRP-USD`,
  `AVAX-USD`, `DOGE-USD`, `ADA-USD`) y confirmar que la capa trigger (z-score)
  funciona sin ensemble ni walk-forward

### FASE 1 — Infraestructura de universos
- Nuevo `src/universe.py`: scrapers Wikipedia para SP500, IBEX 35, Euro Stoxx 50/600
  + lista estática crypto con flag `mode: trigger_only`
- Reestructurar `tickers.yaml` en secciones `universes:` con TTL de caché de 7 días
- CLI flag `--universe sp500,ibex35,stoxx50,crypto` en `screen_tickers.py`

### FASE 2 — Pipeline escalable (~900 tickers)
- Fast screen sobre todo el universo (~15 min paralelizado, 4–8 hilos)
- Deep screen solo sobre top 40–60 por fast_score (~50 min)
- Retry + backoff en `src/data.py` + blacklist persistente de tickers fallidos
- Outputs: `results/screen_YYYY-MM-DD.csv` completo + `results/top_YYYY-MM-DD.csv` (deep)

### FASE 3 — Informe Discord estratificado
- Sección "🔥 ALERTAS DE ENTRADA": tickers con `action == COMPRAR` + `grade ✓✓/✓`
  → embed especial con campo "POR QUÉ AHORA" (z-score, historial reversión, confirmación régimen)
- Top 10 global por calidad del modelo (sin señal activa o en VIGILAR)
- Top 5 por índice (SP500 / IBEX / Stoxx) — formato compacto
- Sección "🪙 Crypto — Pullbacks": solo activos con z < −2.5, sin mezclar con ranking general
- "Nuevas entradas / Salidas" vs el día anterior

### FASE 4 — Resiliencia operativa
- Logging estructurado: tickers procesados / skipped / errores por índice
- Blacklist persistente `.cache/blacklist.json` (3 fallos consecutivos → excluir)
- Validación de calidad de datos EU antes de pasar al screener

### FASE 5 — Validación del modelo en mercados EU
- Backtest histórico 2 años sobre IBEX + Stoxx
- Medir hit rate y Sharpe por región
- Si EU rinde < 70% del hit rate USA → documentar y etiquetar como "experimental"

---

## Pendiente futuro — Tickers momentum/tendenciales

El modelo LONQ está calibrado para **reversión a la media** (autocorrelación negativa,
bajo R² de tendencia). Los tickers tendenciales (crypto, growth tech, momentum puro)
obtienen scores bajos en `fast_screen` y son rechazados correctamente por el modelo.

### Problema
Actualmente el sistema descarta activos como NVDA, TSLA, BTC en bull market porque
son tendenciales — pero esos activos generan las mayores ganancias absolutas.

### Propuesta de solución: módulo LONQ-TREND (paralelo al modelo actual)

Implementar un segundo pipeline que comparte la infraestructura de datos pero
usa una lógica de scoring inversa para activos tendenciales:

**Criterios de selección (inverso al modelo actual):**
- Autocorrelación positiva lag-5 (> +0.05) — momentum persistente
- Tendencia clara (R² > 0.6 sobre 252 días)
- Volatilidad media-alta (15–60% anual) — suficiente movimiento
- Volumen creciente (confirma tendencia real, no ruido)

**Señales del módulo TREND:**
- `TREND_ENTRY`: precio supera media móvil 50d + ADX > 25 (tendencia fuerte)
- `TREND_PULLBACK`: retroceso al 38.2% / 50% Fibonacci sobre swing reciente
- `TREND_EXIT`: precio rompe por debajo de MA50 o ADX < 20

**Universo objetivo:**
- Crypto (BTC, ETH, SOL...) en fases alcistas
- Growth tech de alta beta (NVDA, TSLA, SHOP...)
- Commodities en superciclo (GLD, SLV, cobre)

**Integración con informe Discord:**
- Sección separada "📈 TREND — Momentum activo"
- No se mezcla con el ranking de reversión para no confundir señales
- Flag `--module trend` en CLI para ejecutarlo independientemente

**Prioridad:** Media — implementar después de completar las 5 fases del
Multi-Index Scanner y validar el modelo en EU.

---
*Última actualización: 2026-04-14*
