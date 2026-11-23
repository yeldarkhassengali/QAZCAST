"""Simulation-based pilot experiment: daily retail sales with Kazakhstani calendar effects.

Models: Seasonal naive, ETS, LightGBM M0 (no local calendar), LightGBM M1 (with local calendar).
Rolling-origin evaluation, 28-day horizon, test period Jan-Jun 2025.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- calendar
dates = pd.date_range("2022-01-01", "2025-06-30", freq="D")

FIXED = {  # (month, day): holiday type
    (1, 1): "new_year", (1, 2): "new_year", (1, 7): "orthodox_christmas",
    (3, 8): "womens_day", (3, 21): "nauryz", (3, 22): "nauryz", (3, 23): "nauryz",
    (5, 1): "unity_day", (5, 7): "defender_day", (5, 9): "victory_day",
    (7, 6): "capital_day", (8, 30): "constitution_day", (10, 25): "republic_day",
    (12, 16): "independence_day",
}
KURBAN = pd.to_datetime(["2022-07-09", "2023-06-28", "2024-06-16", "2025-06-06"])
# official transferred days off creating long weekends (bridge days), illustrative per year
BRIDGES = pd.to_datetime(["2022-03-07", "2022-03-24", "2023-03-24", "2023-05-08",
                          "2024-03-25", "2024-07-08", "2025-01-03", "2025-05-08"])

cal = pd.DataFrame({"date": dates})
cal["hol_type"] = [FIXED.get((d.month, d.day), "none") for d in dates]
cal.loc[cal.date.isin(KURBAN), "hol_type"] = "kurban_ait"
cal["is_holiday"] = (cal.hol_type != "none").astype(int)
cal["is_bridge"] = cal.date.isin(BRIDGES).astype(int)
cal["is_off"] = ((cal.is_holiday == 1) | (cal.is_bridge == 1) | (cal.date.dt.dayofweek >= 5)).astype(int)
# long weekend: run of >=3 consecutive days off
run = (cal.is_off != cal.is_off.shift()).cumsum()
cal["long_weekend"] = ((cal.is_off == 1) & (cal.groupby(run).is_off.transform("size") >= 3)).astype(int)

hol_dates = cal.loc[cal.is_holiday == 1, "date"].values


def days_to(series_dates, events, cap=30):
    ev = np.sort(np.asarray(events, dtype="datetime64[D]"))
    d = np.asarray(series_dates, dtype="datetime64[D]")
    idx = np.searchsorted(ev, d)
    nxt = np.where(idx < len(ev), (ev[np.minimum(idx, len(ev) - 1)] - d).astype(int), cap)
    prv_i = np.searchsorted(ev, d, side="right") - 1
    prv = np.where(prv_i >= 0, (d - ev[np.maximum(prv_i, 0)]).astype(int), cap)
    return np.minimum(nxt, cap), np.minimum(prv, cap)


nauryz = cal.loc[cal.hol_type == "nauryz"].groupby(cal.date.dt.year).date.min().values
newyear = cal.loc[(cal.date.dt.month == 1) & (cal.date.dt.day == 1), "date"].values
womens = cal.loc[cal.hol_type == "womens_day", "date"].values
cal["d_to_hol"], cal["d_after_hol"] = days_to(cal.date, hol_dates)
cal["d_to_nauryz"], cal["d_after_nauryz"] = days_to(cal.date, nauryz)
cal["d_to_kurban"], cal["d_after_kurban"] = days_to(cal.date, KURBAN)
cal["d_to_newyear"], _ = days_to(cal.date, newyear)
cal["d_to_womens"], _ = days_to(cal.date, womens)

# ---------------------------------------------------------------- products
CATS = {  # category: (base level range, holiday sensitivity profile)
    "meat": (20, 60), "dairy": (30, 80), "bakery": (40, 100), "sweets": (10, 40),
    "beverages": (25, 70), "household": (8, 25),
}
# multiplicative uplift windows: (anchor dates, days before, days after, peak uplift per category)
EFFECTS = {
    "nauryz": (nauryz, 5, 2, {"meat": 1.8, "dairy": 1.0, "bakery": 0.9, "sweets": 0.8, "beverages": 0.6, "household": 0.1}),
    "kurban": (KURBAN.values, 4, 1, {"meat": 2.2, "dairy": 0.3, "bakery": 0.5, "sweets": 0.4, "beverages": 0.3, "household": 0.05}),
    "newyear": (newyear, 6, 0, {"meat": 0.9, "dairy": 0.5, "bakery": 0.4, "sweets": 1.4, "beverages": 1.5, "household": 0.3}),
    "womens": (womens, 3, 0, {"meat": 0.2, "dairy": 0.2, "bakery": 0.4, "sweets": 1.6, "beverages": 0.5, "household": 0.1}),
}

items = []
store_ids = ["S1", "S2", "S3", "S4"]
for cat, (lo, hi) in CATS.items():
    for k in range(5):
        for s in store_ids:
            items.append({"item": f"{cat}_{k}", "store": s, "cat": cat,
                          "base": rng.uniform(lo, hi) * {"S1": 1.2, "S2": 1.0, "S3": 0.8, "S4": 0.6}[s]})
items = pd.DataFrame(items)
items["sid"] = items.item + "_" + items.store
print("series:", len(items))

rows = []
t = (dates - dates[0]).days.values
dow = dates.dayofweek.values
doy = dates.dayofyear.values
for it in items.itertuples():
    weekly = np.array([0.92, 0.9, 0.93, 0.97, 1.05, 1.18, 1.1])[dow]
    yearly = 1 + 0.08 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    trend = 1 + 0.0002 * t
    price0 = rng.uniform(300, 3000)
    promo = (rng.random(len(dates)) < 0.06).astype(int)
    disc = promo * rng.uniform(0.1, 0.3, len(dates))
    price = price0 * (1 - disc) * (1 + 0.00015 * t)
    promo_eff = 1 + 1.8 * disc
    hol = np.ones(len(dates))
    for name, (anchors, before, after, mult) in EFFECTS.items():
        for a in np.asarray(anchors, dtype="datetime64[D]"):
            off = (np.asarray(dates, dtype="datetime64[D]") - a).astype(int)
            pre = (off >= -before) & (off < 0)
            hol[pre] *= 1 + mult[it.cat] * (1 - np.abs(off[pre]) / (before + 1)) ** 0.5
            on = (off >= 0) & (off <= after)
            hol[on] *= 1 + 0.35 * mult[it.cat] if name != "kurban" else 1 + 0.15 * mult[it.cat]
    # public holiday itself: fewer shoppers for non-festive goods; long weekends lift
    hol *= np.where(cal.is_holiday.values == 1, 0.85, 1.0)
    hol *= np.where(cal.long_weekend.values == 1, 1.07, 1.0)
    mu = it.base * weekly * yearly * trend * promo_eff * hol
    n = 12.0
    y = rng.negative_binomial(n, n / (n + mu))
    rows.append(pd.DataFrame({"sid": it.sid, "item": it.item, "store": it.store, "cat": it.cat,
                              "date": dates, "sales": y, "price": price.round(1), "promo": promo}))
df = pd.concat(rows, ignore_index=True).merge(cal, on="date")
df.to_csv(OUT / "sim_sales.csv.gz", index=False, compression="gzip")
print("rows:", len(df), "total sales:", df.sales.sum())

# ---------------------------------------------------------------- features
df = df.sort_values(["sid", "date"]).reset_index(drop=True)
g = df.groupby("sid").sales
for L in (28, 35, 42, 364):
    df[f"lag{L}"] = g.shift(L)
for w in (7, 28, 90):
    df[f"rmean{w}"] = g.transform(lambda s, w=w: s.shift(28).rolling(w).mean())
df["rstd28"] = g.transform(lambda s: s.shift(28).rolling(28).std())
df["dow"] = df.date.dt.dayofweek
df["dom"] = df.date.dt.day
df["month"] = df.date.dt.month
df["woy"] = df.date.dt.isocalendar().week.astype(int)
df["price_rel"] = df.price / df.groupby("sid").price.transform(lambda s: s.shift(28).rolling(90, min_periods=1).mean())
for c in ("sid", "item", "store", "cat", "hol_type"):
    df[c + "_c"] = df[c].astype("category")

BASE = ["lag28", "lag35", "lag42", "lag364", "rmean7", "rmean28", "rmean90", "rstd28",
        "dow", "dom", "month", "woy", "price", "price_rel", "promo", "item_c", "store_c", "cat_c"]
LOCAL = ["is_holiday", "hol_type_c", "is_bridge", "long_weekend", "d_to_hol", "d_after_hol",
         "d_to_nauryz", "d_after_nauryz", "d_to_kurban", "d_after_kurban", "d_to_newyear", "d_to_womens"]
FEATS = {"M0": BASE, "M1": BASE + LOCAL}
PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=7, num_threads=4)
N_ROUNDS = 600

# ---------------------------------------------------------------- rolling origin
origins = pd.date_range("2025-01-01", periods=6, freq="28D")  # covers Jan .. mid-June 2025
H = 28
preds = []
pivot = df.pivot(index="date", columns="sid", values="sales")
for o in origins:
    tr = df[(df.date < o) & (df.date >= "2023-01-01")]
    te = df[(df.date >= o) & (df.date < o + pd.Timedelta(days=H))].copy()
    print("origin", o.date(), "train", len(tr), "test", len(te))
    for m, feats in FEATS.items():
        ds = lgb.Dataset(tr[feats], tr.sales, categorical_feature=[f for f in feats if f.endswith("_c")])
        bst = lgb.train(PARAMS, ds, N_ROUNDS)
        te[m] = np.clip(bst.predict(te[feats]), 0, None)
        if o == origins[-1]:
            bst.save_model(str(OUT / f"lgb_{m}.txt"))
            if m == "M1":
                contrib = bst.predict(te[feats], pred_contrib=True)
                shap_imp = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=feats).sort_values(ascending=False)
                shap_imp.to_csv(OUT / "shap_importance_M1.csv")
    # seasonal naive: same weekday one week before origin, repeated
    hist = pivot[pivot.index < o]
    last_week = hist.iloc[-7:]
    sn = {sid: np.tile(last_week[sid].values, 4)[:H] for sid in pivot.columns}
    # ETS on log1p, additive weekly seasonality, damped trend, fitted per series on last 2 years
    ets = {}
    for sid in pivot.columns:
        yv = np.log1p(hist[sid].iloc[-730:].values.astype(float))
        try:
            fit = ExponentialSmoothing(yv, trend="add", damped_trend=True, seasonal="add",
                                       seasonal_periods=7, initialization_method="estimated").fit(optimized=True)
            ets[sid] = np.clip(np.expm1(fit.forecast(H)), 0, None)
        except Exception:
            ets[sid] = sn[sid]
    hpos = (te.date - o).dt.days.values
    te["SNaive"] = [sn[s][h] for s, h in zip(te.sid, hpos)]
    te["ETS"] = [ets[s][h] for s, h in zip(te.sid, hpos)]
    te["origin"] = o
    preds.append(te[["sid", "cat", "date", "origin", "sales", "SNaive", "ETS", "M0", "M1",
                     "hol_type", "is_holiday", "d_to_hol", "d_after_hol"]])
P = pd.concat(preds, ignore_index=True)
P.to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- metrics
MODELS = ["SNaive", "ETS", "M0", "M1"]
# MASE scale: in-sample seasonal-naive (lag 7) MAE per series before test period
train_hist = pivot[pivot.index < origins[0]].iloc[-730:]
scale = (train_hist - train_hist.shift(7)).abs().mean()
P["scale"] = P.sid.map(scale)
P["hol_win"] = ((P.d_to_hol <= 7) | (P.d_after_hol <= 3)).astype(int)


def metrics(d):
    out = {}
    for m in MODELS:
        e = d.sales - d[m]
        out[m] = {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                  "WAPE": e.abs().sum() / d.sales.sum() * 100,
                  "MASE": (e.abs() / d.scale).mean(),
                  "Bias%": e.sum() / d.sales.sum() * -100}
    return pd.DataFrame(out).T


res = {"full": metrics(P), "holiday_window": metrics(P[P.hol_win == 1]),
       "regular_days": metrics(P[P.hol_win == 0])}
for k, v in res.items():
    v.round(3).to_csv(OUT / f"metrics_{k}.csv")
    print("\n==", k, f"(n={len(P) if k=='full' else int((P.hol_win==(1 if k=='holiday_window' else 0)).sum())})")
    print(v.round(3))

# per-event windows WAPE (pre-holiday 7 days)
ev_rows = []
for name, anchors in {"New Year": newyear, "8 March": womens, "Nauryz": nauryz, "Kurban Ait": KURBAN.values}.items():
    for a in pd.to_datetime(anchors):
        if a < origins[0] or a > P.date.max():
            continue
        w = P[(P.date >= a - pd.Timedelta(days=7)) & (P.date <= a + pd.Timedelta(days=2))]
        ev_rows.append({"event": name, **{m: (w.sales - w[m]).abs().sum() / w.sales.sum() * 100 for m in MODELS}})
ev = pd.DataFrame(ev_rows).set_index("event")
ev.round(2).to_csv(OUT / "metrics_events.csv")
print("\n== events WAPE\n", ev.round(2))

# per-category WAPE in holiday window
cw = P[P.hol_win == 1].groupby("cat").apply(lambda d: pd.Series({m: (d.sales - d[m]).abs().sum() / d.sales.sum() * 100 for m in MODELS}))
cw.round(2).to_csv(OUT / "metrics_cat_holiday.csv")
print("\n== category holiday WAPE\n", cw.round(2))

# Diebold-Mariano on daily total absolute error (h=1 style, HAC with lag 6)
daily = P.groupby("date").apply(lambda d: pd.Series({m: (d.sales - d[m]).abs().sum() for m in MODELS}))


def dm(a, b, lag=6):
    d = (daily[a] - daily[b]).values
    n = len(d)
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(lag + 1)]
    v = gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))
    stat = dbar / np.sqrt(v / n)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


dmr = {f"M1 vs {b}": dict(zip(["DM", "p"], dm(b, "M1"))) for b in ["SNaive", "ETS", "M0"]}
pd.DataFrame(dmr).T.to_csv(OUT / "dm_tests.csv")
print("\n== DM\n", pd.DataFrame(dmr).T)

# inventory simulation: daily stock = forecast * 1.15 (safety), perishable -> excess discarded
inv = {}
for m in MODELS:
    stock = P[m] * 1.15
    lost = np.clip(P.sales - stock, 0, None)
    excess = np.clip(stock - P.sales, 0, None)
    so_days = (P.sales > stock).mean() * 100
    inv[m] = {"stockout_rate%": so_days, "lost_units%": lost.sum() / P.sales.sum() * 100,
              "excess_units%": excess.sum() / P.sales.sum() * 100}
invh = {}
H1 = P[P.hol_win == 1]
for m in MODELS:
    stock = H1[m] * 1.15
    invh[m] = {"stockout_rate%": (H1.sales > stock).mean() * 100,
               "lost_units%": np.clip(H1.sales - stock, 0, None).sum() / H1.sales.sum() * 100,
               "excess_units%": np.clip(stock - H1.sales, 0, None).sum() / H1.sales.sum() * 100}
pd.DataFrame(inv).T.round(2).to_csv(OUT / "inventory_full.csv")
pd.DataFrame(invh).T.round(2).to_csv(OUT / "inventory_holiday.csv")
print("\n== inventory full\n", pd.DataFrame(inv).T.round(2))
print("\n== inventory holiday\n", pd.DataFrame(invh).T.round(2))
print("\n== SHAP top\n", pd.read_csv(OUT / "shap_importance_M1.csv", index_col=0).head(12))
json.dump({"n_series": len(items), "n_test_rows": len(P), "origins": [str(o.date()) for o in origins]},
          open(OUT / "meta.json", "w"))
