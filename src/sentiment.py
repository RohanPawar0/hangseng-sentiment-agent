"""Score every post and headline from -1 (bearish) to +1 (bullish).

engine "auto": uses the multilingual XLM-RoBERTa sentiment model if
`transformers` + `torch` are installed (handles Chinese/Cantonese/Korean/English);
otherwise falls back to a transparent keyword lexicon of trading slang.
The engine used is recorded in the output so results are reproducible.

Run: python -m src.sentiment
"""
import re

import pandas as pd

from src.common import DATA, iter_tickers, load_config

BULL = ["升", "漲", "涨", "買入", "买入", "加倉", "加仓", "抄底", "睇好", "看好", "突破", "新高", "牛", "反彈", "反弹",
        "爆升", "上車", "上车", "揸住", "bull", "bullish", "buy", "long", "moon", "rally", "breakout", "upgrade", "beat",
        "상승", "매수", "호재"]
BEAR = ["跌", "冧", "沽", "賣出", "卖出", "減倉", "减仓", "唱淡", "看淡", "睇淡", "熊", "爆煲", "新低", "蝕", "亏", "插水",
        "走佬", "bear", "bearish", "sell", "short", "dump", "crash", "downgrade", "miss", "plunge",
        "하락", "매도", "악재"]


def lexicon_score(text: str) -> float:
    t = str(text).lower()
    b = sum(t.count(w.lower()) for w in BULL)
    s = sum(t.count(w.lower()) for w in BEAR)
    return 0.0 if b + s == 0 else (b - s) / (b + s)


def focus_text(text: str, keywords: list, width: int = 120, long_after: int = 300) -> str:
    """Long posts (e.g. Telegram news digests) often cover many stories; only the
    text around the ticker's name is about that ticker. Short posts (LIHKG replies,
    which often don't repeat the name) are kept whole."""
    text = str(text)
    if len(text) <= long_after:
        return text
    low = text.lower()
    spans = []
    for k in keywords:
        start = low.find(k.lower())
        while start != -1:
            spans.append((max(0, start - width), min(len(text), start + len(k) + width)))
            start = low.find(k.lower(), start + 1)
    if not spans:
        return text[:500]
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return " … ".join(text[a:b] for a, b in merged)[:500]


def build_scorer(cfg):
    engine = cfg["sentiment"]["engine"]
    if engine in ("auto", "transformer"):
        try:
            from transformers import pipeline
            clf = pipeline("sentiment-analysis", model=cfg["sentiment"]["model"], top_k=None, truncation=True)

            def score(texts):
                out = []
                for res in clf([str(x)[:512] for x in texts], batch_size=16):
                    p = {d["label"].lower(): d["score"] for d in res}
                    out.append(p.get("positive", 0) - p.get("negative", 0))
                return out
            return "xlm-roberta", score
        except Exception as e:  # not installed or no internet
            if engine == "transformer":
                raise
            print(f"[sentiment] transformer unavailable ({type(e).__name__}); using lexicon")
    return "lexicon", lambda texts: [lexicon_score(x) for x in texts]


def main():
    cfg = load_config()
    name, score = build_scorer(cfg)
    posts = pd.concat([pd.read_csv(p) for p in sorted(DATA.glob("posts_*.csv"))], ignore_index=True)
    posts = posts.drop_duplicates(subset=["ticker", "text", "timestamp_utc"])
    kw = {t["symbol"]: t["keywords"] + [t["name"]] for _, t in iter_tickers(cfg)}
    posts["scored_text"] = [focus_text(x, kw.get(tk, [])) for x, tk in zip(posts["text"].fillna(""), posts["ticker"])]
    posts["sentiment"] = score(posts["scored_text"].tolist())
    posts["sentiment_engine"] = name
    posts.to_csv(DATA / "posts_scored.csv", index=False, encoding="utf-8-sig")

    news_path = DATA / "news.csv"
    if news_path.exists():
        news = pd.read_csv(news_path)
        news["scored_text"] = [focus_text(f"{ti}. {su}", kw.get(tk, []), long_after=500)
                               for ti, su, tk in zip(news["title"].fillna(""), news["summary"].fillna(""), news["ticker"])]
        news["sentiment"] = score(news["scored_text"].tolist())
        news["sentiment_engine"] = name
        news.to_csv(DATA / "news_scored.csv", index=False, encoding="utf-8-sig")
    print(f"[sentiment] engine={name}; scored {len(posts)} posts")
    print(posts.groupby(["source", "ticker"])["sentiment"].agg(["count", "mean"]).round(3))


if __name__ == "__main__":
    main()
