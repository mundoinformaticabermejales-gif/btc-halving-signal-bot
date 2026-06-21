# BTC Halving Signal Bot

A Bitcoin signal bot based on halving cycles, with backtested indicators and a real-time liquidation map. Sends a daily signal to Telegram.

---

## What the bot does

Every morning at 08:00 (Madrid time), the bot analyzes the Bitcoin market and sends a complete message to Telegram with:

### 1. Market Regime (MA200)
The bot determines whether Bitcoin is in a **bull** or **bear** trend by looking at its position relative to the 200-day moving average:
- Price > MA200 × 1.02 → **Bullish**
- Price < MA200 × 0.98 → **Bearish**
- Between both → **Neutral** (transition zone)

### 2. Mayer Multiple
Divides the current price by the MA200. Historically:
- < 0.8 → BTC deeply undervalued (cycle bottom)
- 1.0 → fair value relative to trend
- > 2.4 → euphoria zone, likely cycle top

The bot adjusts the recommended capital allocation based on the multiple (100% at cycle bottom, 0% at bubble zone).

### 3. Choppiness Index (backtested)
Measures whether the market is trending or ranging:
- < 38.2 → **Strong trend** (higher confidence signal)
- > 61.8 → **Sideways / choppy market** (reduce exposure)
- Validated accuracy: **65.4%** at 30-day horizon (edge +11.4% over random)

### 4. Bollinger Bands Width (backtested)
Measures whether volatility is at historical extremes (possible imminent breakout):
- Percentile ≥ 80 → extreme compression, breakout likely
- Validated accuracy: **58.0%** at 30-day horizon (edge +3.9%)

### 5. Sentiment Indicators
- **Fear & Greed Index** (alternative.me): market panic/euphoria thermometer
- **Funding Rate** (Binance futures): high positive = market overly long
- **Long/Short Ratio**: proportion of long vs short futures positions
- **Open Interest**: total volume of open contracts

### 6. Halving Cycle Phase
The bot places Bitcoin within its 4-year cycle:
- Calculates days since the last halving (April 2024)
- Compares the current trajectory with previous cycles (2012, 2016, 2020)
- Shows whether we are in accumulation, expansion, or distribution

### 7. Liquidation Map (real-time)
A 24/7 service running on the VPS captures every forced liquidation on Binance futures via WebSocket. Using this data the bot:
- Identifies which price levels have the most liquidation clusters
- Differentiates between liquidated longs (red) and liquidated shorts (green)
- Sends a **PNG chart** showing the highest concentration zones around the current price

> Liquidation clusters act as magnets — price tends to move toward them to sweep stops.

### 8. Actionable Conclusion
The bot synthesizes all indicators and gives a clear recommendation:
- Recommended portfolio allocation (0–100%)
- Primary reason for the recommendation
- Confidence level based on how many indicators agree

---

## Backtest Results

Tested on 500 random dates between 2018 and 2026, 30-day horizon:

| Indicator | Accuracy | Edge vs random |
|-----------|----------|----------------|
| Choppiness Index | 65.4% | +11.4% ★ |
| BB Width percentile | 58.0% | +3.9% ★ |
| Mayer Multiple | 56.2% | +2.4% |
| MA200 regime | reference | — |

**Combined strategy (MA200 + Mayer Multiple):**
- OOS Sharpe ratio: **0.97**
- Max drawdown: **-21%** vs -67% for pure HODL
- Optimal horizon: **30 days**

---

## Infrastructure

```
Linux VPS
├── Cron 08:00 Madrid → daily_signal.py    ← daily signal
├── btc-liq-collector.service              ← Binance WebSocket 24/7
│     └── wss://fstream.binance.com/ws/btcusdt@forceOrder
│     └── saves rolling 24h buffer to reports/liq_rolling.json
└── .env (chmod 600)
      ├── TELEGRAM_BOT_TOKEN
      └── TELEGRAM_CHAT_ID
```

**Data sources (all free):**
- Kraken REST API → BTC/USD price and OHLCV
- Binance Futures REST → funding rate, long/short ratio, open interest
- Binance Futures WebSocket → forced liquidations in real time
- alternative.me API → Fear & Greed Index

---

## Installation (Claude Code skill)

```bash
npx skills add mundoinformaticabermejales-gif/btc-halving-signal-bot
```

Or clone the repo directly:

```bash
git clone https://github.com/mundoinformaticabermejales-gif/btc-halving-signal-bot.git
cd btc-halving-signal-bot
python3 -m venv venv
venv/bin/pip install ccxt pandas numpy matplotlib websocket-client python-dotenv pyarrow
```

Create the `.env` file:
```
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Set up the cron and liquidation service:
```bash
# Daily cron
echo 'CRON_TZ=Europe/Madrid
0 8 * * * root cd /path/to/btc-bot && venv/bin/python src/daily_signal.py' > /etc/cron.d/btc-signal

# Liquidation collector as a service
systemctl enable btc-liq-collector
systemctl start btc-liq-collector
```

---

## Main Files

| File | Description |
|------|-------------|
| `src/daily_signal.py` | Main signal — runs all indicators and sends to Telegram |
| `src/liq_collector.py` | WebSocket service — captures Binance liquidations in real time |
| `src/liq_map.py` | Analyzes the liquidation buffer and finds clusters by price level |
| `src/liq_chart.py` | Generates the PNG liquidation map chart (dark matplotlib) |
| `src/indicator_backtest.py` | Backtests any indicator against 2018–2026 historical data |
| `src/signal_survey.py` | Survey of 21 indicators from open-source bots with edge ranking |
| `src/cycle.py` | Halving cycle logic and comparison with previous cycles |
| `src/fetch_data.py` | Downloads and caches OHLCV data from Kraken |

---

## Security

- Kraken API keys: **Query + Trade** permissions only, NEVER Withdraw
- IP whitelist with the VPS IP on Kraken
- `.env` with `chmod 600`, never in code or in the repo
- The repo contains no credentials

---

## Disclaimer

This software is for educational and personal research purposes only. Cryptocurrency trading involves risk of loss. The bot does not execute orders automatically — it generates informational signals only.
