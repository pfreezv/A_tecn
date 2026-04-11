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
*Última actualización: 2026-04-11*
