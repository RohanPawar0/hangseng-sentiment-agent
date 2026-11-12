"""Source 2: public Telegram channels via Apify.

Telegram channels aren't per stock, so we scrape whole HK-stock channels for the
lookback window and keep only messages that mention a ticker keyword.
Followers = the channel's subscriber count.

Run:  python -m src.scrape_telegram            (all channels in config.yaml)
      python -m src.scrape_telegram --test     (first channel, 30 messages, prints fields)
"""
import argparse
import json

import pandas as pd
from apify_client import ApifyClient

from src.common import RAW, first, run_items, iter_tickers, load_config, mentions_ticker, save_posts, secret, since_utc, to_num, to_utc


def main(test=False):
    cfg = load_config()
    a = cfg["apify"]
    channels = a.get("telegram_channels") or []
    if not channels:
        raise SystemExit("No Telegram channels in config.yaml (apify.telegram_channels). See CLAUDE.md step 4.")
    if test:
        channels = channels[:1]
    client = ApifyClient(secret("APIFY_TOKEN"))
    run = client.actor(a["telegram_actor"]).call(run_input={
        "channels": channels,
        "maxMessages": 30 if test else a["telegram_max_messages"],
        "oldestMessageDate": since_utc(cfg).strftime("%Y-%m-%d"),
        "includeChannelInfo": True,
    })
    items = run_items(client, run)
    (RAW / "telegram_raw.json").write_text(json.dumps(items, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[Telegram] {len(items)} items from {channels}")
    if test and items:
        for it in items[:2]:
            print("  Fields:", sorted(it.keys()))

    # subscriber counts from the channel-info items
    subs = {}
    for it in items:
        n = to_num(first(it, "subscribers", "subscriberCount", "subscribersCount", "participantsCount",
                         "membersCount", "members", "channelInfo.subscribers"))
        name = str(first(it, "username", "channel", "channelUsername", "channelName", default="")).lstrip("@").lower()
        if name and n == n:  # not NaN
            subs[name] = max(subs.get(name, 0), n)

    rows, n_msgs = [], 0
    tickers = list(iter_tickers(cfg))
    for it in items:
        text = first(it, "text", "message", "messageText", "content", "caption")
        if not text:
            continue
        n_msgs += 1
        ch = str(first(it, "channel", "channelUsername", "username", "channelName", default="")).lstrip("@").lower()
        mid = first(it, "messageId", "id", "message_id", "postId")
        for ex_code, t in tickers:
            if "telegram" not in cfg["exchanges"][ex_code]["sources"] or not mentions_ticker(text, t):
                continue
            rows.append({
                "exchange": ex_code, "source": "telegram", "ticker": t["symbol"],
                "post_id": f"{ch}-{mid}",
                "timestamp_utc": to_utc(first(it, "date", "datetime", "timestamp", "publishedAt", "time")),
                "text": str(text).replace("\n", " ").strip(),
                "likes": to_num(first(it, "reactions", "reactionsCount", "totalReactions", "likes")),
                "reposts": to_num(first(it, "forwards", "forwardsCount", "shares")),
                "views": to_num(first(it, "views", "viewsCount", "viewCount")),
                "author_followers": subs.get(ch, float("nan")),
                "url": first(it, "url", "link", default=f"https://t.me/{ch}/{mid}"),
            })
    out = save_posts(rows, "telegram")
    df = pd.read_csv(out)
    print(f"{n_msgs} messages read, {len(df)} mention one of our tickers -> {out}")
    if len(df):
        print(df.groupby("ticker").size().rename("posts"))
    else:
        print("No ticker mentions. Add more/better channels, or more keywords per ticker in config.yaml.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    main(p.parse_args().test)
