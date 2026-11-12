"""News per ticker from Exa (last `lookback_days`).  Run: python -m src.fetch_news"""
import pandas as pd
from exa_py import Exa
from urllib.parse import urlparse

from src.common import DATA, iter_tickers, load_config, secret, since_utc, to_utc


def main():
    cfg = load_config()
    exa = Exa(secret("EXA_API_KEY"))
    start = since_utc(cfg).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for ex_code, t in iter_tickers(cfg):
        query = f"{t['name']} {t['symbol']} stock news"
        res = exa.search(query, num_results=cfg["exa"]["results_per_ticker"], category="news",
                         start_published_date=start, contents={"text": {"max_characters": 1500}})
        for r in res.results:
            rows.append({
                "exchange": ex_code, "ticker": t["symbol"], "title": r.title, "url": r.url,
                "published_utc": to_utc(r.published_date),
                "source_domain": urlparse(r.url).netloc,
                "summary": (getattr(r, "text", "") or "").replace("\n", " ")[:500],
            })
        print(f"[Exa] {t['symbol']}: {len(res.results)} articles")
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "news.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} articles to data/news.csv")
    for sym, g in df.groupby("ticker"):
        print(f"  {sym}: " + " | ".join(g["title"].head(2).astype(str)))


if __name__ == "__main__":
    main()
