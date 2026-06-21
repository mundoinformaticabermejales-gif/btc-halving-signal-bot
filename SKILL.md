---
name: btc-halving-signal-bot
description: Bitcoin halving-cycle signal bot with backtested indicators (MA200, Mayer Multiple, Choppiness Index, BB Width, Fear & Greed, Funding Rate, Liquidation Map). Sends daily Telegram signals. Validated OOS Sharpe 0.97.
triggers:
  - send btc signal
  - bitcoin signal telegram
  - run btc backtest
  - check bitcoin liquidations
  - btc signal status
  - show liquidation map
  - bitcoin halving cycle indicator
  - manda señal bitcoin
  - señal btc
  - backtest bitcoin
  - liquidaciones bitcoin
  - estado del bot btc
---

# BTC Halving Signal Bot

Bot de señales Bitcoin basado en ciclos de halving con indicadores backtestados.
Envía una señal diaria por Telegram con régimen de mercado, indicadores técnicos
y mapa de liquidaciones en tiempo real.

## Estrategia validada

- **MA200 + Mayer Multiple** — Sharpe 0.97 OOS, maxDD -21% vs HODL -67%
- **Choppiness Index** — 65.4% precisión a 30 días (edge +11.4%)
- **Bollinger Bands Width** — 58.0% precisión (edge +3.9%)
- Horizonte óptimo: **30 días**

## Infraestructura

- VPS: root@5.250.189.184, clave `~/.ssh/portatilguapo_ed25519`
- Cron `0 8 * * *` hora Madrid → `/root/btc-bot/run_signal.sh`
- Colector liquidaciones: `btc-liq-collector.service` (WebSocket Binance 24/7)
- Datos: Kraken BTC/USD + Binance futuros + alternative.me Fear & Greed

## Comandos

### Enviar señal ahora

```bash
ssh -i ~/.ssh/portatilguapo_ed25519 root@5.250.189.184 \
  "cd /root/btc-bot && venv/bin/python src/daily_signal.py"
```

### Backtest horizonte N días

```bash
ssh -i ~/.ssh/portatilguapo_ed25519 root@5.250.189.184 "
cd /root/btc-bot
venv/bin/python -c \"
import sys; sys.path.insert(0,'src')
import indicator_backtest as bt
df = bt.build_dataset()
checks = bt.run_checks(df, n=500, horizon=30)
bt.print_checks(checks, horizon=30)
\""
```

### Mapa de liquidaciones (gráfico PNG a Telegram)

```bash
ssh -i ~/.ssh/portatilguapo_ed25519 root@5.250.189.184 "
cd /root/btc-bot
venv/bin/python -c \"
import os, sys, json
sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv('.env')
from liq_chart import build_chart, send_chart_telegram
token = os.getenv('TELEGRAM_BOT_TOKEN')
chat  = os.getenv('TELEGRAM_CHAT_ID')
s = json.load(open('reports/latest_signal.json'))
buf = build_chart(current_price=s['precio_usd'])
if buf:
    send_chart_telegram(token, chat, buf, 'Mapa liquidaciones BTC 24h')
    print('Enviado')
else:
    print('Sin datos de liquidaciones aun')
\""
```

### Survey completo de 21 indicadores

```bash
ssh -i ~/.ssh/portatilguapo_ed25519 root@5.250.189.184 \
  "cd /root/btc-bot && venv/bin/python src/signal_survey.py"
```

### Estado del sistema

```bash
ssh -i ~/.ssh/portatilguapo_ed25519 root@5.250.189.184 "
echo '=== COLECTOR ==='
systemctl status btc-liq-collector --no-pager | head -5
echo '=== BUFFER ==='
python3 -c \"
import json,os
d=json.load(open('/root/btc-bot/reports/liq_rolling.json')) if os.path.exists('/root/btc-bot/reports/liq_rolling.json') else []
print(f'Eventos liquidaciones: {len(d)}')
\"
echo '=== ULTIMA SEÑAL ==='
python3 -c \"
import json
s=json.load(open('/root/btc-bot/reports/latest_signal.json'))
print(f'Fecha: {s[\"fecha\"]} | Precio: \${s[\"precio_usd\"]:,} | {s[\"regimen\"]}')
\""
```

## Indicadores y backtest (500 fechas, 2018-2026, horizonte 30d)

| Indicador | Precisión | Edge | Estado |
|-----------|-----------|------|--------|
| Choppiness Index <38.2 | 65.4% | +11.4% | En bot |
| BB Width percentil ≥80 | 58.0% | +3.9% | En bot |
| Mayer Multiple <0.80/>2.20 | 56.2% | +2.4% | En bot |
| MACD cruce | 54.6% | +0.8% | No integrado |
| MA200 régimen | referencia | — | En bot |
| RSI extremos ≤20/≥80 | 38.2% | -15.9% | En bot (señal débil) |

## Archivos principales en VPS

```
/root/btc-bot/
├── src/daily_signal.py       ← señal principal (cron 08:00)
├── src/liq_collector.py      ← colector WebSocket 24/7
├── src/liq_map.py            ← análisis liquidaciones
├── src/liq_chart.py          ← gráfico PNG matplotlib
├── src/indicator_backtest.py ← backtest indicadores
├── src/signal_survey.py      ← survey 21 indicadores
├── reports/latest_signal.json
├── reports/liq_rolling.json  ← buffer 24h liquidaciones
└── .env                      ← TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

## Seguridad

- API keys Kraken: permisos solo **Query + Trade**, NUNCA Withdraw
- IP whitelist con IP del VPS en Kraken
- `.env` chmod 600, nunca en el código
