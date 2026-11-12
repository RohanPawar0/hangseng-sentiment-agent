"""Source 1: LIHKG forum (Hong Kong retail traders) via Apify.

Two Apify runs per ticker, kept small for the free plan:
  1. search mode  -> recent threads mentioning the ticker keywords
  2. thread mode  -> replies (the actual posts) inside the most active threads
Raw output is saved to data/raw/ so nothing is lost; normalised posts go to
data/posts_lihkg.csv.

Run:  python -m src.scrape_lihkg            (all tickers)
      python -m src.scrape_lihkg --test     (1 ticker, tiny, prints the fields)
"""
import argparse
import json

from apify_client import ApifyClient

from src.common import RAW, first, run_items, iter_tickers, load_config, save_posts, secret, to_num, to_utc, since_utc

TZ = "Asia/Hong_Kong"


def run_actor(client, actor_id, run_input):
    run = client.actor(actor_id).call(run_input=run_input)
    return run_items(client, run)


def thread_id_of(item):
    tid = first(item, "thread_id", "threadId", "id", "tid")
    if tid is None:
        url = first(item, "url", "threadUrl", "thread_url", default="")
        if "/thread/" in str(url):
            tid = str(url).split("/thread/")[1].split("/")[0]
    return str(tid) if tid is not None else None


def reply_rows(item):
    """A thread-mode item may hold replies in a list, or be one reply itself."""
    replies = first(item, "replies", "posts", "items", "comments")
    if isinstance(replies, list):
        return [{**r, "_thread": item} for r in replies if isinstance(r, dict)]
    return [item]


def normalise(row, ex_code, symbol):
    thread = row.get("_thread", {})
    # own_text = the reply without the quoted earlier post (avoids counting the same text twice)
    text = first(row, "own_text", "msg", "content", "text", "message", "body", "msg_text", "title")
    ts = first(row, "reply_time", "replyTime", "create_time", "createTime", "created_at",
               "createdAt", "timestamp", "time", "date")
    tid = thread_id_of(thread) or thread_id_of(row)
    floor = first(row, "msg_num", "floor", "msgNum", "post_id", "postId", "id")
    likes = to_num(first(row, "like_count", "likeCount", "likes", "upvotes", "vote_score", "reactions"))
    return {
        "exchange": ex_code, "source": "lihkg", "ticker": symbol,
        "post_id": f"{tid}-{floor}" if floor is not None else f"{tid}-{abs(hash((str(text), str(ts)))) % 10**10}",
        "timestamp_utc": to_utc(ts, TZ),
        "text": str(text).replace("\n", " ").strip() if text else None,
        "likes": likes,
        "reposts": 0.0,  # LIHKG has no reposts
        "views": to_num(first(thread, "view_count", "viewCount", "views")),
        "author_followers": float("nan"),  # LIHKG has no follower counts
        "url": f"https://lihkg.com/thread/{tid}/page/1" if tid else None,
    }


def main(test=False):
    cfg = load_config()
    a = cfg["apify"]
    client = ApifyClient(secret("APIFY_TOKEN"))
    rows = []
    for ex_code, t in iter_tickers(cfg):
        if "lihkg" not in cfg["exchanges"][ex_code]["sources"]:
            continue
        kws = t["keywords"][: a["lihkg_keywords_per_ticker"]]
        print(f"\n[LIHKG] {t['symbol']} {t['name']}: searching {kws}")
        found = run_actor(client, a["lihkg_actor"], {
            "mode": "search", "keywords": kws,
            "maxItems": 5 if test else a["lihkg_search_per_keyword"],
            "order": "desc_reply_time", "dateFrom": f"{cfg['lookback_days']}d",
        })
        (RAW / f"lihkg_search_{t['symbol']}.json").write_text(json.dumps(found, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        if test and found:
            print("  Fields in a search result:", sorted(found[0].keys()))
        # most active threads first
        found.sort(key=lambda x: to_num(first(x, "no_of_reply", "reply_count", "replyCount", "replies_count", default=0)) or 0, reverse=True)
        ids = []
        for it in found:
            tid = thread_id_of(it)
            if tid and tid not in ids:
                ids.append(tid)
        ids = ids[: 1 if test else a["lihkg_threads_per_ticker"]]
        print(f"  {len(found)} threads found, opening {len(ids)}: {ids}")
        if not ids:
            continue
        threads = run_actor(client, a["lihkg_actor"], {
            "mode": "thread", "threadIds": ids,
            "maxReplies": 5 if test else a["lihkg_max_replies"],
        })
        (RAW / f"lihkg_threads_{t['symbol']}.json").write_text(json.dumps(threads, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        if test and threads:
            print("  Fields in a thread result:", sorted(threads[0].keys()))
        for item in threads:
            for r in reply_rows(item):
                rows.append(normalise(r, ex_code, t["symbol"]))
        if test:
            break
    out = save_posts(rows, "lihkg")
    import pandas as pd
    df = pd.read_csv(out)
    print(f"\nSaved {len(df)} LIHKG posts to {out}")
    if len(df):
        print(df.groupby("ticker").agg(posts=("text", "size"), first=("timestamp_utc", "min"), last=("timestamp_utc", "max")))
        print("\nSample:", df[["ticker", "timestamp_utc", "likes", "text"]].head(3).to_string())
    print("\nIf timestamps or likes are empty, open data/raw/lihkg_threads_*.json, find the real field names,"
          " and add them to normalise() in src/scrape_lihkg.py.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="1 ticker, tiny run, prints field names")
    main(p.parse_args().test)
