# Hang Seng Social-Sentiment Agent: n+1 price prediction

A Python pipeline that collects what Hong Kong retail investors and the news say about five
Hang Seng stocks, turns it into an engagement-weighted sentiment signal, summarises each stock
(price, sentiment, direction, stated positions, support/resistance), ranks them in a visual
report, and predicts the next price bar (n+1) with an honest walk-forward test.

**Stocks:** Tencent (0700.HK), Alibaba (9988.HK), HSBC (0005.HK), Meituan (3690.HK), Xiaomi (1810.HK)
**Sources:** LIHKG forum + Telegram channels (Apify) · news (Exa) · prices (yfinance)
**Data window of the submitted run:** 2–12 Nov 2026 (UTC)

## Pipeline
```
Apify: LIHKG (search → threads → replies)  ┐
Apify: Telegram public channels            ├─► posts (UTC) ─► sentiment ─┐
Exa: recent news per ticker ───────────────┘                ─► sentiment ─┤
yfinance: hourly + daily bars ────────────────────────────────────────────┴─► align to next bar
   ─► per-ticker summary (price, sentiment, direction, positions, S/R)
   ─► Ridge model on next-bar return, expanding walk-forward vs baselines
   ─► results/report.html
```

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece protobuf   # optional but recommended (multilingual sentiment model)
copy .env.example .env          # macOS/Linux: cp .env.example .env ; then add APIFY_TOKEN and EXA_API_KEY
python -m src.check_keys        # verifies Apify, Exa and yfinance
python run_all.py --refresh     # scrape + news + prices + sentiment + features + predict + report
```
Open `results/report.html`. Every step can also be run alone, e.g. `python -m src.scrape_lihkg --test`.
`python run_all.py` (without `--refresh`) reuses the saved scrape data and does not spend Apify credit.
A full refresh with the limits in `config.yaml` cost about **$1.84** of Apify credit (free plan = $5/month).

## Apify actors used
| Source | Actor ID | Input used |
|---|---|---|
| LIHKG | `claude_code_reviewer/lihkg-scraper-en` | `mode: search`, `keywords`, `dateFrom: "10d"`, then `mode: thread`, `threadIds`, `maxReplies` |
| Telegram | `automation-lab/telegram-scraper` | `channels`, `maxMessages`, `oldestMessageDate`, `includeChannelInfo` |

Telegram channels used: `hkej_news` (信報即時財經) and `FinanceNewsDaily` (華爾街見聞).
Channels and all limits are in `config.yaml`.

## Data collected (submitted run)
| Ticker | LIHKG posts | Telegram posts | Total social | Exa news |
|---|---|---|---|---|
| Meituan 3690.HK | 39 | 25 | 64 | 10 |
| Xiaomi 1810.HK | 24 | 30 | 54 | 10 |
| Tencent 0700.HK | 15 | 32 | 47 | 10 |
| Alibaba 9988.HK | 0 | 44 | 44 | 10 |
| HSBC 0005.HK | 12 | 20 | 32 | 10 |
| **Total** | **90** | **151** | **241** | **50** |

Each post has: UTC timestamp, text, likes/reactions, views, reposts (where the platform has them),
author/channel followers (Telegram subscriber count), URL. Prices: hourly bars (60 days) and daily bars (6 months).

## Method
- **Time alignment:** all timestamps are converted to UTC. Each post is attached to the **first price bar that starts at or after it**, so a post can only influence later prices (no look-ahead). Posts older than the 10-day window are dropped (the LIHKG actor filters by thread activity, so old replies in active threads would otherwise slip in).
- **Sentiment:** multilingual `cardiffnlp/twitter-xlm-roberta-base-sentiment` (Chinese/Cantonese/English), score = P(positive) − P(negative), from −1 to +1. If `transformers` is not installed, a transparent trading-slang lexicon is used instead (`src/sentiment.py`); the report states which engine produced the results. The submitted results use **XLM-RoBERTa**.
- **Ticker focus:** long posts (Telegram news digests cover many stories) are scored only on the text around the ticker's name, so an unrelated story in the same digest does not count as that ticker's sentiment.
- **Weighting:** `w = 1 + log(1+likes) + log(1+reposts) + 0.5·log(1+views) + 0.5·log(1+followers)`.
- **Per-ticker score:** 0.7 × weighted social sentiment + 0.3 × mean news sentiment → Bullish (≥ 0.10) / Bearish (≤ −0.10) / Neutral.
- **Positions:** regex for stated entries like "買入 380" / "bought at 380" / "沽 …", kept only if within ±40% of the current price.
- **Support/resistance:** nearest daily swing low/high (5-day window) around the current price, plus the classic pivot (H+L+C)/3 with S1/R1.
- **Prediction:** Ridge regression on the next hourly bar's return, pooled across tickers. Features: engagement-weighted sentiment over the trailing 24 hours, log mentions over 24 hours, this bar's sentiment, and the last two returns. The regularisation strength is chosen by time-ordered cross-validation **on the training data only**.
- **Evaluation:** expanding-window walk-forward. Every test bar is predicted by a model trained only on earlier bars. It is compared with three baselines: (1) naive "price stays the same", (2) the same model **without** sentiment (price-only), (3) always guessing the direction that was most common in the test period (a hindsight baseline for direction accuracy).

## Results
Test period: 6 Nov 07:30 → 12 Nov 06:30 UTC, **140 out-of-sample predictions** (28 per ticker).

| Model | Mean abs. error (% of price) | Mean abs. error (HKD, pooled) | Direction accuracy |
|---|---|---|---|
| **Sentiment + price model** | **0.537%** | 0.648 | **57.7%** |
| Price-only model (no sentiment) | 0.551% | 0.661 | 41.5% |
| Naive: price stays the same | 0.526% | 0.616 | n/a |
| Always guess the most common direction (hindsight) | n/a | n/a | 58.5% |

**Plain reading:**
- Adding social sentiment **improves on the same model without sentiment** (lower error, higher direction accuracy).
- The model **does not beat the naive "no change" baseline** (0.537% vs 0.526%), though the gap is small.
- 57.7% direction accuracy on 140 predictions is **not statistically distinguishable from chance**, and is below the 58.5% hindsight baseline.
- Per ticker, the model beat the naive baseline for **Meituan and Alibaba** and lost for HSBC, Tencent and Xiaomi (`results/metrics.json → per_ticker`).
- Conclusion: the pipeline and the evaluation work end to end, and there is a weak hint that sentiment carries information, but **10 days of data cannot show a reliable predictive edge.**

**Per-ticker summary (close of 12 Nov 2026):**

| Ticker | Price (HKD) | Mentions | Social sent. | News sent. | Score | Direction | Support | Resistance | n+1 prediction |
|---|---|---|---|---|---|---|---|---|---|
| Meituan | 72.10 | 64 | −0.129 | +0.068 | −0.069 | Neutral | 68.85 | 77.90 | 72.12 (Up) |
| Xiaomi | 29.98 | 54 | −0.045 | +0.014 | −0.027 | Neutral | 27.50 | 32.40 | 30.01 (Up) |
| Tencent | 398.80 | 47 | −0.048 | +0.035 | −0.023 | Neutral | 375.20 | 430.40 | 398.12 (Down) |
| Alibaba | 93.80 | 44 | −0.072 | +0.083 | −0.025 | Neutral | 88.65 | 109.40 | 93.85 (Up) |
| HSBC | 140.70 | 32 | −0.085 | +0.145 | −0.016 | Neutral | 133.50 | 148.80 | 140.82 (Up) |

Social chatter leaned slightly negative and news slightly positive for every stock. Combined, all five stay inside the Neutral band.
No stated positions ("bought at …") were found in this sample.

**Model history (reported for transparency):** the first version (per-bar sentiment, fixed Ridge alpha = 1) had 0.681% error and 53.8% direction accuracy, worse than every baseline. Two changes were decided **before** re-testing, and the test was run once: 24-hour smoothing of sparse sentiment, and regularisation chosen by cross-validation on training data. No further tuning was done on the test period, to avoid overfitting to it.

## Limitations (honest notes)
- **Too little data for a trading signal.** About 10 days, 241 posts and 140 test predictions. Treat this as a working, reproducible pipeline and evaluation framework, not a profitable model. The obvious next step is to run the scrapers daily for 2–3 months and re-test.
- **LIHKG volume is low for stocks,** and there were **no LIHKG threads about Alibaba** in the window (Alibaba's social data comes from Telegram only). Some LIHKG replies drift off-topic (politics, entertainment). Keyword matching cannot tell company news (e.g. Tencent Video deals) from stock discussion.
- **LIHKG has no follower counts or reposts.** Those fields are empty for LIHKG, and weighting falls back to likes.
- **The Telegram channels are news broadcasts, not retail chat.** About 60 candidate channel names were checked at `t.me/s/<name>`. No public HK retail-trader channel was found (that chat happens in private groups). `hkej_news` looks like an automated mirror (15 subscribers, about 1 view per post). `FinanceNewsDaily` has real engagement (59K subscribers, about 1.7K views per post) but is mainland-China focused.
- **`hkej_news` covers 4–12 Nov only.** It posts about 60 messages a day, and the 700-message cap (to stay on the free Apify plan) did not reach back to 2 Nov. `hket_news` (about 600 posts per day) was skipped for cost.
- **One digest message can mention several tickers.** It then counts once for each (the scorer only reads the text around each ticker's name).
- **The sentiment model is trained on tweets.** It is not finance-specific and not tuned for Cantonese slang, and sarcasm is often missed.
- **Hourly bars include the overnight gap.** The first bar of each day mixes overnight news with the open.
- **Scope:** Hong Kong was built fully. The brief also mentions Korean sources. The code is exchange-agnostic (`config.yaml` includes a ready Korea block), and a Naver discussion actor exists (`hgservices/naver-stock-discussion-scraper`), but Korea was not run.

## Project structure
```
config.yaml            all settings (tickers, keywords, actors, channels, limits, thresholds)
src/check_keys.py      verifies Apify + Exa + yfinance
src/scrape_lihkg.py    source 1 → data/posts_lihkg.csv
src/scrape_telegram.py source 2 → data/posts_telegram.csv
src/fetch_news.py      Exa → data/news.csv
src/fetch_prices.py    yfinance → data/prices_hourly.csv, data/prices_daily.csv
src/sentiment.py       XLM-RoBERTa (or lexicon) → posts_scored.csv, news_scored.csv
src/features.py        align posts to bars, weights, per-ticker summary → results/ticker_summary.csv
src/predict.py         walk-forward test vs baselines → results/metrics.json, walkforward.csv, predictions.csv
src/report.py          results/report.html
run_all.py             runs everything (--refresh re-scrapes)
```
