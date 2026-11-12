"""Align posts to price bars and build the per-ticker summary.

Outputs:
  data/bar_features.csv        one row per ticker per hourly bar (mentions, weighted sentiment)
  results/ticker_summary.csv   ticker, name, price, sentiment, direction, positions, support/resistance

Run: python -m src.features
"""
import re

import numpy as np
import pandas as pd

from src.common import DATA, RESULTS, iter_tickers, load_config


def engagement_weight(df):
    """Posts with more reactions, views and followers count more (log-scaled)."""
    f = lambda c: np.log1p(df[c].fillna(0).clip(lower=0))
    return 1 + f("likes") + f("reposts") + 0.5 * f("views") + 0.5 * f("author_followers")


def align_to_bars(posts, hourly):
    """Each post is attached to the FIRST bar that starts at or after it,
    so a post can only influence prices that come later (no look-ahead)."""
    out = []
    for sym, p in posts.groupby("ticker"):
        b = hourly[hourly["ticker"] == sym][["timestamp_utc"]].sort_values("timestamp_utc")
        if b.empty:
            continue
        p = p.sort_values("timestamp_utc")
        m = pd.merge_asof(p, b.rename(columns={"timestamp_utc": "bar_utc"}),
                          left_on="timestamp_utc", right_on="bar_utc", direction="forward")
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else posts.assign(bar_utc=pd.NaT)


POS_LONG = re.compile(r"(買入|买入|入咗|入了|揸|加倉|bought|buy|long)[^\d]{0,8}(\d+(?:\.\d+)?)", re.I)
POS_SHORT = re.compile(r"(沽|賣出|卖出|short|sold|sell)[^\d]{0,8}(\d+(?:\.\d+)?)", re.I)


def extract_positions(texts, price):
    """Find stated positions like 'bought at 380' and keep plausible prices (±40% of current)."""
    longs, shorts = [], []
    for t in texts:
        for rx, bucket in ((POS_LONG, longs), (POS_SHORT, shorts)):
            for _, num in rx.findall(str(t)):
                v = float(num)
                if price and 0.6 * price <= v <= 1.4 * price:
                    bucket.append(v)
    def desc(xs):
        return f"{len(xs)} mentions, median {np.median(xs):.2f}" if xs else "none"
    return desc(longs), desc(shorts)


def support_resistance(daily, price, window=5):
    """Swing lows/highs on daily bars + classic floor-trader pivot of the last day."""
    lows, highs = daily["low"].values, daily["high"].values
    sw_low = [lows[i] for i in range(window, len(lows) - window) if lows[i] == lows[i - window:i + window + 1].min()]
    sw_high = [highs[i] for i in range(window, len(highs) - window) if highs[i] == highs[i - window:i + window + 1].max()]
    support = max([x for x in sw_low if x < price], default=float(daily["low"].tail(20).min()))
    resistance = min([x for x in sw_high if x > price], default=float(daily["high"].tail(20).max()))
    last = daily.iloc[-1]
    pivot = (last["high"] + last["low"] + last["close"]) / 3
    return round(support, 2), round(resistance, 2), round(pivot, 2), round(2 * pivot - last["high"], 2), round(2 * pivot - last["low"], 2)


def main():
    cfg = load_config()
    s = cfg["sentiment"]
    posts = pd.read_csv(DATA / "posts_scored.csv", parse_dates=["timestamp_utc"])
    hourly = pd.read_csv(DATA / "prices_hourly.csv", parse_dates=["timestamp_utc"])
    daily = pd.read_csv(DATA / "prices_daily.csv", parse_dates=["timestamp_utc"])
    news = pd.read_csv(DATA / "news_scored.csv") if (DATA / "news_scored.csv").exists() else pd.DataFrame(columns=["ticker", "sentiment"])

    posts["w"] = engagement_weight(posts)
    posts = align_to_bars(posts, hourly)
    posts.to_csv(DATA / "posts_aligned.csv", index=False, encoding="utf-8-sig")

    # per ticker per bar
    g = posts.dropna(subset=["bar_utc"]).assign(ws=lambda d: d["sentiment"] * d["w"]).groupby(["ticker", "bar_utc"])
    per_bar = pd.DataFrame({"mentions": g.size(), "sent_w": g["ws"].sum() / g["w"].sum(), "engagement": g["w"].sum()}).reset_index()
    feats = hourly.merge(per_bar, left_on=["ticker", "timestamp_utc"], right_on=["ticker", "bar_utc"], how="left").drop(columns="bar_utc")
    feats[["mentions", "engagement"]] = feats[["mentions", "engagement"]].fillna(0)
    feats["sent_w"] = feats["sent_w"].fillna(0)
    feats.to_csv(DATA / "bar_features.csv", index=False)

    rows = []
    for ex_code, t in iter_tickers(cfg):
        sym = t["symbol"]
        p = posts[posts["ticker"] == sym]
        d = daily[daily["ticker"] == sym].sort_values("timestamp_utc")
        if d.empty:
            continue
        h = hourly[hourly["ticker"] == sym].sort_values("timestamp_utc")
        price = float(h["close"].iloc[-1]) if len(h) else float(d["close"].iloc[-1])  # latest available price
        social = float((p["sentiment"] * p["w"]).sum() / p["w"].sum()) if len(p) else 0.0
        n = news[news["ticker"] == sym]
        news_s = float(n["sentiment"].mean()) if len(n) else 0.0
        combined = s["social_weight"] * social + s["news_weight"] * news_s
        direction = "Bullish" if combined >= s["bullish_threshold"] else "Bearish" if combined <= s["bearish_threshold"] else "Neutral"
        longs, shorts = extract_positions(p["text"], price)
        sup, res, piv, s1, r1 = support_resistance(d, price)
        rows.append({
            "exchange": ex_code, "ticker": sym, "name": t["name"], "price": round(price, 2),
            "mentions": int(len(p)), "sources": ", ".join(sorted(p["source"].unique())) if len(p) else "",
            "social_sentiment": round(social, 3), "news_sentiment": round(news_s, 3), "news_articles": int(len(n)),
            "sentiment_score": round(combined, 3), "direction": direction,
            "long_positions": longs, "short_positions": shorts,
            "support": sup, "resistance": res, "pivot": piv, "pivot_S1": s1, "pivot_R1": r1,
        })
    summary = pd.DataFrame(rows).sort_values(["exchange", "mentions", "sentiment_score"], ascending=[True, False, False])
    summary.to_csv(RESULTS / "ticker_summary.csv", index=False, encoding="utf-8-sig")
    print(summary[["ticker", "name", "price", "mentions", "sentiment_score", "direction", "support", "resistance"]].to_string(index=False))


if __name__ == "__main__":
    main()
