"""Visual report: results/report.html (open it in any browser).
Ranks tickers by mentions and sentiment per exchange, shows direction, support/
resistance, the n+1 prediction and the honest walk-forward test.
Run: python -m src.report
"""
import json
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.common import DATA, RESULTS, load_config

COLORS = {"Bullish": "#1a9e5c", "Bearish": "#d6453d", "Neutral": "#8a94a6"}


def fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def main():
    cfg = load_config()
    summ = pd.read_csv(RESULTS / "ticker_summary.csv")
    preds = pd.read_csv(RESULTS / "predictions.csv") if (RESULTS / "predictions.csv").exists() else pd.DataFrame()
    metrics = json.loads((RESULTS / "metrics.json").read_text()) if (RESULTS / "metrics.json").exists() else {}
    feats = pd.read_csv(DATA / "bar_features.csv", parse_dates=["timestamp_utc"])
    posts = pd.read_csv(DATA / "posts_scored.csv")
    engine = posts["sentiment_engine"].iloc[0] if len(posts) else "n/a"

    sections = []
    for ex_code, ex in cfg["exchanges"].items():
        s = summ[summ["exchange"] == ex_code].sort_values("mentions", ascending=False)
        if s.empty:
            continue
        label = s["ticker"] + " " + s["name"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Social mentions (last %d days)" % cfg["lookback_days"], "Sentiment score (-1 bearish … +1 bullish)"))
        fig.add_bar(x=label, y=s["mentions"], marker_color="#3b5bdb", row=1, col=1, name="mentions")
        s2 = s.sort_values("sentiment_score", ascending=False)
        fig.add_bar(x=s2["ticker"] + " " + s2["name"], y=s2["sentiment_score"], marker_color=[COLORS[d] for d in s2["direction"]], row=1, col=2, name="sentiment")
        fig.update_layout(height=380, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))

        trend = go.Figure()
        for sym in s["ticker"]:
            f = feats[(feats["ticker"] == sym) & (feats["mentions"] > 0)]
            daily = f.set_index("timestamp_utc").resample("1D")["sent_w"].mean().dropna()
            trend.add_scatter(x=daily.index, y=daily.values, mode="lines+markers", name=sym)
        trend.update_layout(title="Daily weighted social sentiment", height=320, margin=dict(t=40, b=10, l=10, r=10))

        table = s.merge(preds, on="ticker", how="left") if len(preds) else s
        cols = ["ticker", "name", "price", "mentions", "sources", "social_sentiment", "news_sentiment", "sentiment_score",
                "direction", "long_positions", "short_positions", "support", "resistance", "pivot"] + \
               [c for c in ["pred_next_close", "pred_direction"] if c in table.columns]
        rows_html = "".join(
            "<tr>" + "".join(f"<td class='{r['direction'] if c == 'direction' else ''}'>{r[c]}</td>" for c in cols) + "</tr>"
            for _, r in table.iterrows())
        sections.append(f"""
        <h2>{ex['name']}</h2>
        {fig_html(fig)}{fig_html(trend)}
        <div class='scroll'><table><tr>{''.join(f'<th>{c.replace("_", " ")}</th>' for c in cols)}</tr>{rows_html}</table></div>""")

    m = metrics
    metric_html = "" if not m else f"""
    <h2>n+1 prediction: honest out-of-sample test</h2>
    <p>{m['method']}. {m['test_predictions']} test predictions, each made using only earlier data.</p>
    <p>Test period (UTC): {m['test_period_utc'][0][:16]} → {m['test_period_utc'][1][:16]}.</p>
    <table><tr><th>Model</th><th>Mean abs. error (% of price)</th><th>Mean abs. error (HKD, pooled)</th><th>Direction accuracy</th></tr>
    <tr><td><b>Sentiment + price model</b></td><td>{m['mae_return_pct_model']:.3f}%</td><td>{m['mae_model']}</td><td>{m['direction_accuracy']:.1%}</td></tr>
    <tr><td>Price-only model (no sentiment)</td><td>{m['mae_return_pct_price_only']:.3f}%</td><td>{m['mae_price_only_model']}</td><td>{m['direction_accuracy_price_only']:.1%}</td></tr>
    <tr><td>Naive: price stays the same</td><td>{m['mae_return_pct_naive']:.3f}%</td><td>{m['mae_naive_baseline']}</td><td>n/a</td></tr>
    <tr><td>Always guess the most common direction (hindsight)</td><td>n/a</td><td>n/a</td><td>{m['direction_accuracy_majority_baseline']:.1%}</td></tr></table>
    <p class='note'><b>Result:</b> the sentiment model {'beats' if m['model_beats_baseline'] else 'does <b>not</b> beat'} the naive baseline,
    and sentiment {'improves' if m['sentiment_adds_value'] else 'does <b>not</b> improve'} on the price-only model.
    {m['note']}</p>"""

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Hang Seng Sentiment Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:system-ui,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#16202c}}
h1{{margin-bottom:4px}} .meta{{color:#5a6573}} table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{border-bottom:1px solid #dde2e8;padding:6px 8px;text-align:left;white-space:nowrap}} th{{background:#f3f5f8}}
.scroll{{overflow-x:auto}} .Bullish{{color:#1a9e5c;font-weight:600}} .Bearish{{color:#d6453d;font-weight:600}}
.note{{background:#fff4dc;padding:8px 12px;border-radius:6px}}</style></head><body>
<h1>Social Sentiment &amp; n+1 Price Report</h1>
<p class='meta'>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · sources: LIHKG, Telegram (Apify) + news (Exa) · sentiment engine: {engine}</p>
{''.join(sections)}{metric_html}
<p class='meta'>Support/resistance = nearest daily swing low/high around the current price (5-day window); pivot = classic (H+L+C)/3 of the last session.</p>
</body></html>"""
    out = RESULTS / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
