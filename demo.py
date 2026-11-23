"""1-minute demo for the QAZCAST prototype video.
Run experiment.py first (or use the saved out/predictions.csv.gz), then: python demo.py
Prints accuracy per model and around each holiday, then shows the Kurban Ait forecast plot.
"""
import pandas as pd, matplotlib.pyplot as plt
P = pd.read_csv("out/predictions.csv.gz", parse_dates=["date"])
M = ["SNaive", "ETS", "M0", "M1"]
wape = lambda d, m: (d.sales - d[m]).abs().sum() / d.sales.sum() * 100
print("\nWAPE (%) on 20,160 test forecasts, Jan-Jun 2025")
print(pd.Series({m: round(wape(P, m), 2) for m in M}).to_string())
for name, day in [("Nauryz", "2025-03-21"), ("Kurban Ait", "2025-06-06")]:
    a = pd.Timestamp(day); w = P[(P.date >= a - pd.Timedelta(days=7)) & (P.date <= a + pd.Timedelta(days=2))]
    print(f"\n{name} window WAPE:", {m: float(round(wape(w, m), 1)) for m in M})
a = pd.Timestamp("2025-06-06")
w = P[(P.cat == "meat") & (P.date >= a - pd.Timedelta(days=12)) & (P.date <= a + pd.Timedelta(days=5))]
agg = w.groupby("date")[["sales", "M0", "M1"]].sum()
agg.plot(title="Meat sales around Kurban Ait 2025: actual vs M0 (no local calendar) vs M1 (QAZCAST)")
plt.axvline(a, ls=":", c="gray"); plt.ylabel("units/day"); plt.tight_layout(); plt.show()
