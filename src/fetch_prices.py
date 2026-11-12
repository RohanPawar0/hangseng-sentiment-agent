"""Hourly + daily price bars from yfinance, in UTC.  Run: python -m src.fetch_prices"""
import pandas as pd
import yfinance as yf

from src.common import DATA, iter_tickers, load_config


def bars(symbol, period, interval):
    h = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if h.empty:
        return h
    h = h.reset_index().rename(columns={"Datetime": "ts", "Date": "ts"})
    h["timestamp_utc"] = pd.to_datetime(h["ts"], utc=True)
    h = h.rename(columns=str.lower)[["timestamp_utc", "open", "high", "low", "close", "volume"]]
    h.insert(0, "ticker", symbol)
    return h


def main():
    cfg = load_config()
    hourly, daily = [], []
    for _, t in iter_tickers(cfg):
        h = bars(t["symbol"], cfg["prices"]["hourly_period"], "1h")
        d = bars(t["symbol"], cfg["prices"]["daily_period"], "1d")
        hourly.append(h)
        daily.append(d)
        last = f"{d['close'].iloc[-1]:.2f}" if len(d) else "n/a"
        print(f"[prices] {t['symbol']}: {len(h)} hourly, {len(d)} daily bars, last close {last}")
    pd.concat(hourly).to_csv(DATA / "prices_hourly.csv", index=False)
    pd.concat(daily).to_csv(DATA / "prices_daily.csv", index=False)
    print("Saved data/prices_hourly.csv and data/prices_daily.csv")


if __name__ == "__main__":
    main()
