# BTC Halving Signal Bot

Bot de señales para Bitcoin basado en ciclos de halving, con indicadores backtestados y mapa de liquidaciones en tiempo real. Envía una señal diaria por Telegram.

---

## Qué hace el bot

Cada mañana a las 08:00 (hora Madrid), el bot analiza el mercado de Bitcoin y envía un mensaje completo a Telegram con:

### 1. Régimen de mercado (MA200)
El bot determina si Bitcoin está en tendencia **alcista** o **bajista** mirando su posición respecto a la media móvil de 200 días:
- Precio > MA200 × 1.02 → **Alcista**
- Precio < MA200 × 0.98 → **Bajista**
- Entre ambos → **Neutro** (zona de transición)

### 2. Mayer Multiple
Divide el precio actual entre la MA200. Históricamente:
- < 0.8 → BTC muy infravalorado (fondo de ciclo)
- 1.0 → precio justo según tendencia
- > 2.4 → zona de euforia, máximos de ciclo probables

El bot ajusta el porcentaje de capital recomendado según el múltiplo (100% en zona de fondo, 0% en zona de burbuja).

### 3. Choppiness Index (nuevo, backtestado)
Mide si el mercado está en tendencia o en rango lateral:
- < 38.2 → **Tendencia fuerte** (señal de mayor confianza)
- > 61.8 → **Mercado lateral** (reducir exposición)
- Precisión validada: **65.4%** a 30 días (edge +11.4% sobre aleatoriedad)

### 4. Bollinger Bands Width (nuevo, backtestado)
Mide si la volatilidad está en máximos históricos (posible ruptura inminente):
- Percentil ≥ 80 → compresión extrema, ruptura probable
- Precisión validada: **58.0%** a 30 días (edge +3.9%)

### 5. Indicadores de sentimiento
- **Fear & Greed Index** (alternative.me): termómetro del pánico/euforia del mercado
- **Funding Rate** (Binance futuros): si es muy positivo, el mercado está largo en exceso
- **Long/Short Ratio**: proporción de posiciones largas vs cortas en futuros
- **Open Interest**: volumen total de contratos abiertos

### 6. Fase del ciclo de halving
El bot sitúa a Bitcoin dentro de su ciclo de 4 años:
- Calcula los días desde el último halving (abril 2024)
- Compara la trayectoria actual con los ciclos anteriores (2012, 2016, 2020)
- Muestra si estamos en acumulación, expansión o distribución

### 7. Mapa de liquidaciones (tiempo real)
Un servicio corriendo 24/7 en el VPS captura cada liquidación forzada de Binance futuros via WebSocket. Con estos datos el bot:
- Identifica en qué niveles de precio hay más "deudas" concentradas
- Diferencia entre largos liquidados (rojo) y cortos liquidados (verde)
- Envía un **gráfico PNG** con las zonas de mayor concentración alrededor del precio actual

> Las zonas con muchas liquidaciones actúan como imanes: el precio tiende a moverse hacia ellas para "barrer" los stops.

### 8. Conclusión accionable
El bot sintetiza todos los indicadores y da una recomendación clara:
- Porcentaje de cartera recomendado (0-100%)
- Razón principal de la recomendación
- Nivel de confianza basado en cuántos indicadores coinciden

---

## Resultados del backtest

Backtestado sobre 500 fechas aleatorias entre 2018 y 2026, horizonte de 30 días:

| Indicador | Precisión | Edge vs aleatoriedad |
|-----------|-----------|----------------------|
| Choppiness Index | 65.4% | +11.4% ★ |
| BB Width percentil | 58.0% | +3.9% ★ |
| Mayer Multiple | 56.2% | +2.4% |
| MA200 régimen | referencia | — |

**Estrategia combinada (MA200 + Mayer Multiple):**
- Sharpe ratio OOS: **0.97**
- Drawdown máximo: **-21%** vs -67% de HODL puro
- Horizonte óptimo: **30 días**

---

## Infraestructura

```
VPS Linux
├── Cron 08:00 Madrid → daily_signal.py    ← señal diaria
├── btc-liq-collector.service              ← WebSocket Binance 24/7
│     └── wss://fstream.binance.com/ws/btcusdt@forceOrder
│     └── guarda rolling buffer 24h en reports/liq_rolling.json
└── .env (chmod 600)
      ├── TELEGRAM_BOT_TOKEN
      └── TELEGRAM_CHAT_ID
```

**Fuentes de datos (todas gratuitas):**
- Kraken REST API → precio y OHLCV de BTC/USD
- Binance Futures REST → funding rate, long/short ratio, open interest
- Binance Futures WebSocket → liquidaciones forzadas en tiempo real
- alternative.me API → Fear & Greed Index

---

## Instalación (Claude Code skill)

```bash
npx skills add <tu-usuario>/btc-halving-signal-bot
```

O clona el repo directamente:

```bash
git clone https://github.com/<tu-usuario>/btc-halving-signal-bot.git
cd btc-halving-signal-bot
python3 -m venv venv
venv/bin/pip install ccxt pandas numpy matplotlib websocket-client python-dotenv pyarrow
```

Crea el archivo `.env`:
```
TELEGRAM_BOT_TOKEN=tu_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui
```

Configura el cron y el servicio de liquidaciones:
```bash
# Cron diario
echo 'CRON_TZ=Europe/Madrid
0 8 * * * root cd /ruta/btc-bot && venv/bin/python src/daily_signal.py' > /etc/cron.d/btc-signal

# Colector de liquidaciones como servicio
systemctl enable btc-liq-collector
systemctl start btc-liq-collector
```

---

## Archivos principales

| Archivo | Descripción |
|---------|-------------|
| `src/daily_signal.py` | Señal principal — ejecuta todos los indicadores y envía a Telegram |
| `src/liq_collector.py` | Servicio WebSocket — captura liquidaciones Binance en tiempo real |
| `src/liq_map.py` | Analiza el buffer de liquidaciones y encuentra clusters por nivel de precio |
| `src/liq_chart.py` | Genera el gráfico PNG del mapa de liquidaciones (matplotlib dark) |
| `src/indicator_backtest.py` | Backtesta cualquier indicador contra histórico 2018-2026 |
| `src/signal_survey.py` | Survey de 21 indicadores de bots open source con ranking de edge |
| `src/cycle.py` | Lógica del ciclo de halving y comparación con ciclos anteriores |
| `src/fetch_data.py` | Descarga y cachea datos OHLCV de Kraken |

---

## Seguridad

- API keys Kraken: permisos solo **Query + Trade**, NUNCA Withdraw
- IP whitelist con la IP del VPS en Kraken
- `.env` con `chmod 600`, nunca en el código ni en el repo
- El repo no contiene ninguna credencial

---

## Aviso

Este software es para uso educativo y de investigación personal. El trading de criptomonedas conlleva riesgo de pérdida. El bot no ejecuta órdenes automáticamente — solo genera señales informativas.
