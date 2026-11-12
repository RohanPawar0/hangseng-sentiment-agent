"""Run the whole pipeline in order.

  python run_all.py              # uses saved scrape data if present (saves Apify credit)
  python run_all.py --refresh    # re-scrape LIHKG + Telegram and re-fetch news/prices
"""
import argparse

from src import fetch_news, fetch_prices, features, predict, report, scrape_lihkg, scrape_telegram, sentiment
from src.common import DATA


def step(name, fn):
    print(f"\n==== {name} ====")
    fn()


def main(refresh):
    if refresh or not (DATA / "posts_lihkg.csv").exists():
        step("Scrape LIHKG (Apify)", scrape_lihkg.main)
    if refresh or not (DATA / "posts_telegram.csv").exists():
        step("Scrape Telegram (Apify)", scrape_telegram.main)
    if refresh or not (DATA / "news.csv").exists():
        step("News (Exa)", fetch_news.main)
    step("Prices (yfinance)", fetch_prices.main)
    step("Sentiment", sentiment.main)
    step("Features + per-ticker summary", features.main)
    step("Predict n+1 + walk-forward test", predict.main)
    step("Visual report", report.main)
    print("\nDone. Open results/report.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    main(p.parse_args().refresh)
