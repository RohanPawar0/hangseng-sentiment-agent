# Assignment brief (summary)

**CWT HengSeng exchange sentiment Agent: intern task** (CrowdWisdomTrading, via Internshala as TZURONI LTD.)
Original: https://docs.google.com/document/d/1t-SSYC_thT-PEUYDyta4zqmuYTjk9kPlYmK5g4gYE2Y

- Python sentiment pipeline predicting price at **n+1** from **minimum 2 sentiment sources**; social posts via **Apify**, recent news via **exa.com**; predictive model with **out-of-sample validation**.
- Social sources: **2 different sources per exchange** from the list. Hong Kong: LIHKG, Telegram, Futu/Moomoo community, AAStocks & ETNet. Korea: Naver Finance boards, Daum Finance, KakaoTalk/Telegram, Blind.
- Scope:
  1. Scrape posts about **5 assets per exchange** for the **last 10 days** (ticker, cashtag, keyword searches). Capture timestamps (UTC-aligned), text, engagement (likes, reposts, views), author follower counts; align with price bars. Add the most recent relevant news (Exa).
  2. Per ticker: ticker, asset name, current price; sentiment, direction; positions if any; support/resistance. ("Let's see your creativity.")
  3. Report sorting tickers by social mentions / sentiment score and direction **per exchange**: a visual report.
  4. Predict the price at n+1 and check the results.
- Submission **by email** to **gilad@crowdwisdomtrading.com** (they also mention Tavily in passing): (1) GitHub/GitLab repo link; (2) Apify / Exa tokens and actor IDs used, so they can reproduce the pipeline.
- Timeline: 5–7 days. Internshala deadline: **14 Nov 2026**.
