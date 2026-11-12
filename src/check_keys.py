"""Step 1 check: are the Apify and Exa keys valid, and can we download prices?
Run:  python -m src.check_keys
"""
import yfinance as yf
from apify_client import ApifyClient
from exa_py import Exa

from src.common import secret


def main():
    user = ApifyClient(secret("APIFY_TOKEN")).user().get()
    # newer apify-client returns a pydantic object, older versions a dict
    username = user.get("username") if isinstance(user, dict) else getattr(user, "username", "?")
    print(f"[Apify] OK, logged in as: {username}")

    exa = Exa(secret("EXA_API_KEY"))
    res = exa.search("Tencent 0700.HK stock news", num_results=3, contents={"text": {"max_characters": 200}})
    print("[Exa] OK, sample headlines:")
    for r in res.results:
        print(f"   - {r.title}  ({r.published_date})  {r.url}")

    hist = yf.Ticker("0700.HK").history(period="10d", interval="1h")
    print(f"[yfinance] OK, {len(hist)} hourly bars for 0700.HK, last close = {hist['Close'].iloc[-1]:.2f} HKD")


if __name__ == "__main__":
    main()
