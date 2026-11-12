"""Predict the next bar's price (n+1) and test it honestly.

Model: Ridge regression on next-bar return, pooled across tickers.
Features (all known at the close of bar n): weighted sentiment, mentions, engagement,
3-bar rolling sentiment, last two returns.
Evaluation: expanding-window walk-forward. For every test timestamp the model is
trained ONLY on earlier bars. Compared with the naive baseline "price stays the same".

Outputs: results/metrics.json, results/walkforward.csv, results/predictions.csv
Run: python -m src.predict
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import DATA, RESULTS, load_config, since_utc

# v2 (12 Nov): posts are sparse (most hourly bars have none), so sentiment is
# smoothed over the trailing 24 hours instead of per bar.
FEATURES = ["sent_24h", "log_mentions_24h", "sent_w", "ret_0", "ret_1"]
PRICE_ONLY = ["ret_0", "ret_1"]  # ablation: same model without any sentiment features
ALPHAS = np.logspace(-1, 4, 12)   # candidate regularisation strengths


def make_model():
    """Ridge whose strength is picked by time-ordered cross-validation on the
    TRAINING rows only (rows must be sorted by time before fitting)."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=4)))


def rolling_24h(df):
    """Engagement-weighted sentiment and mention count over the previous 24 hours,
    per ticker. Uses only bars at or before the current one (no look-ahead)."""
    out = []
    for _, d in df.groupby("ticker"):
        s = d.set_index("timestamp_utc")
        wsum = (s["sent_w"] * s["engagement"]).rolling("24h").sum()
        w = s["engagement"].rolling("24h").sum()
        d = d.copy()
        d["sent_24h"] = (wsum / w.replace(0, np.nan)).fillna(0).values
        d["log_mentions_24h"] = np.log1p(s["mentions"].rolling("24h").sum()).values
        out.append(d)
    return pd.concat(out)


def build(df):
    df = df.sort_values(["ticker", "timestamp_utc"]).copy()
    df = rolling_24h(df)
    g = df.groupby("ticker")
    df["ret_0"] = g["close"].pct_change()
    df["ret_1"] = g["ret_0"].shift(1)
    df["next_close"] = g["close"].shift(-1)
    df["target"] = df["next_close"] / df["close"] - 1
    return df


def main():
    cfg = load_config()
    pc = cfg["predict"]
    df = build(pd.read_csv(DATA / "bar_features.csv", parse_dates=["timestamp_utc"]))
    # only use bars inside the social-data window (before it, sentiment is unknown, not neutral)
    window = df[df["timestamp_utc"] >= since_utc(cfg)].dropna(subset=FEATURES)
    train_all = window.dropna(subset=["target"]).sort_values(["timestamp_utc", "ticker"])  # time order for CV
    times = np.sort(train_all["timestamp_utc"].unique())
    start = int(len(times) * pc["min_train_fraction"])
    preds = []
    for ts in times[start:]:
        tr, te = train_all[train_all["timestamp_utc"] < ts], train_all[train_all["timestamp_utc"] == ts]
        if len(tr) < 30:
            continue
        m = make_model().fit(tr[FEATURES], tr["target"])
        m0 = make_model().fit(tr[PRICE_ONLY], tr["target"])
        te = te.assign(pred_ret=m.predict(te[FEATURES]), pred_ret_price_only=m0.predict(te[PRICE_ONLY]))
        preds.append(te)
    if not preds:
        raise SystemExit("Not enough bars for walk-forward testing. Need more posts/bars.")
    wf = pd.concat(preds)
    wf["pred_close"] = wf["close"] * (1 + wf["pred_ret"])
    wf["naive_close"] = wf["close"]
    mae_model = float((wf["pred_close"] - wf["next_close"]).abs().mean())
    mae_naive = float((wf["naive_close"] - wf["next_close"]).abs().mean())
    mae_price_only = float((wf["close"] * (1 + wf["pred_ret_price_only"]) - wf["next_close"]).abs().mean())
    moved = wf[wf["target"] != 0]
    hit = float((np.sign(moved["pred_ret"]) == np.sign(moved["target"])).mean()) if len(moved) else float("nan")
    hit_price_only = float((np.sign(moved["pred_ret_price_only"]) == np.sign(moved["target"])).mean()) if len(moved) else float("nan")
    up_share = float((moved["target"] > 0).mean()) if len(moved) else float("nan")
    # Scale-free errors in % of price (HKD errors are dominated by high-priced stocks like Tencent)
    mae_pct = lambda pred: round(float((pred - wf["target"]).abs().mean() * 100), 4)
    # Hindsight baseline: always guess the direction that turned out most common in the test period
    majority_acc = max(up_share, 1 - up_share) if len(moved) else float("nan")
    per_ticker = {sym: {
        "test_bars": int(len(g)),
        "mae_model": round(float((g["pred_close"] - g["next_close"]).abs().mean()), 4),
        "mae_naive": round(float((g["naive_close"] - g["next_close"]).abs().mean()), 4),
        "direction_accuracy": round(float((np.sign(g["pred_ret"]) == np.sign(g["target"]))[g["target"] != 0].mean()), 3),
    } for sym, g in wf.groupby("ticker")}

    final = make_model().fit(train_all[FEATURES], train_all["target"])
    coefs = dict(zip(FEATURES, np.round(final[-1].coef_, 6).tolist()))
    last = window.sort_values("timestamp_utc").groupby("ticker").tail(1)
    nxt = last.assign(pred_ret=final.predict(last[FEATURES]))
    nxt["pred_next_close"] = (nxt["close"] * (1 + nxt["pred_ret"])).round(2)
    nxt["pred_direction"] = np.where(nxt["pred_ret"] > 0, "Up", np.where(nxt["pred_ret"] < 0, "Down", "Flat"))
    nxt[["ticker", "timestamp_utc", "close", "pred_ret", "pred_next_close", "pred_direction"]].to_csv(RESULTS / "predictions.csv", index=False)
    wf[["ticker", "timestamp_utc", "close", "next_close", "pred_close", "pred_ret", "target"]].to_csv(RESULTS / "walkforward.csv", index=False)

    metrics = {
        "method": ("Ridge on next-bar return (24h-smoothed sentiment; regularisation chosen by time-series CV "
                   "on training data only), expanding-window walk-forward, pooled tickers"),
        "chosen_alpha_final_model": round(float(final[-1].alpha_), 3),
        "test_predictions": int(len(wf)),
        "mae_model": round(mae_model, 4), "mae_naive_baseline": round(mae_naive, 4),
        "model_beats_baseline": bool(mae_model < mae_naive),
        "direction_accuracy": round(hit, 3),
        "mae_price_only_model": round(mae_price_only, 4),
        "direction_accuracy_price_only": round(hit_price_only, 3),
        "sentiment_adds_value": bool(mae_model < mae_price_only),
        "share_of_up_bars": round(up_share, 3),
        "direction_accuracy_majority_baseline": round(majority_acc, 3),
        "mae_return_pct_model": mae_pct(wf["pred_ret"]),
        "mae_return_pct_naive": mae_pct(0.0),
        "mae_return_pct_price_only": mae_pct(wf["pred_ret_price_only"]),
        "test_period_utc": [str(wf["timestamp_utc"].min()), str(wf["timestamp_utc"].max())],
        "per_ticker": per_ticker, "coefficients_standardised": coefs,
        "note": "Only ~10 days of social data: treat results as a pipeline demonstration, not a trading signal.",
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_ticker"}, indent=2))
    print(nxt[["ticker", "close", "pred_next_close", "pred_direction"]].to_string(index=False))


if __name__ == "__main__":
    main()
