"""Shared helpers: paths, config, secrets, robust field access, time parsing."""
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
for _p in (DATA, RAW, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# Windows cmd uses cp1252 by default and crashes on Chinese text; print UTF-8 instead.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_items(client, run) -> list:
    """Items from an Apify run's dataset. apify-client 3.x returns a pydantic
    object (run.default_dataset_id); older versions return a dict."""
    if not run:
        raise RuntimeError("Apify actor did not return a run")
    if isinstance(run, dict):
        ds = run.get("defaultDatasetId")
    else:
        ds = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
    items = client.dataset(ds).iterate_items()
    return [i if isinstance(i, dict) else (i.model_dump() if hasattr(i, "model_dump") else dict(i)) for i in items]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def secret(name: str) -> str:
    """Read an API key from .env. Never print the value."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_"):
        raise SystemExit(f"Missing {name}. Put it in the .env file (see .env.example).")
    return value


def iter_tickers(cfg: dict):
    """Yield (exchange_code, ticker_dict) for every configured ticker."""
    for ex_code, ex in cfg["exchanges"].items():
        for t in ex["tickers"]:
            yield ex_code, t


def since_utc(cfg: dict) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=int(cfg["lookback_days"]))


def first(d, *keys, default=None):
    """Return the first non-empty value among several possible keys.
    Keys may be dotted ("author.followers"). Scraper outputs differ between
    actors and versions, so we look for several names instead of one."""
    for key in keys:
        cur, ok = d, True
        for part in key.split("."):
            if isinstance(cur, dict) and cur.get(part) not in (None, "", []):
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def to_utc(value, assume_tz: str = "UTC"):
    """Parse unix seconds/ms or a date string into a UTC pandas Timestamp."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NaT
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            v = float(value)
            return pd.to_datetime(v, unit="ms" if v > 1e12 else "s", utc=True)
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(assume_tz)
        return ts.tz_convert("UTC")
    except Exception:
        return pd.NaT


def to_num(value) -> float:
    """Numbers like 1200, '1.2K', '3M', or a list of reactions -> float."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):  # e.g. [{"emoji": "👍", "count": 5}, ...]
        return float(sum(to_num(first(x, "count", "total", default=0)) if isinstance(x, dict) else 0 for x in value))
    if isinstance(value, dict):
        return to_num(first(value, "count", "total", "value"))
    s = str(value).strip().replace(",", "").upper()
    mult = 1.0
    if s.endswith("K"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return float("nan")


def mentions_ticker(text: str, ticker: dict) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in ticker["keywords"])


POST_COLUMNS = ["exchange", "source", "ticker", "post_id", "timestamp_utc", "text",
                "likes", "reposts", "views", "author_followers", "url"]


def save_posts(rows: list, source: str) -> Path:
    """Write normalised posts for one source to data/posts_<source>.csv."""
    df = pd.DataFrame(rows, columns=POST_COLUMNS)
    df = df.dropna(subset=["timestamp_utc", "text"])
    df = df.drop_duplicates(subset=["source", "ticker", "post_id"])
    # Strict lookback window: scrapers' own date filters work per thread/channel,
    # so old replies inside an active thread can slip through. Keep last N days only.
    cutoff = since_utc(load_config())
    before = len(df)
    df = df[pd.to_datetime(df["timestamp_utc"], utc=True) >= cutoff]
    if before != len(df):
        print(f"[{source}] dropped {before - len(df)} posts older than {cutoff:%Y-%m-%d %H:%M} UTC")
    out = DATA / f"posts_{source}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out
