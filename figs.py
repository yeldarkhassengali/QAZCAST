import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
O=Path("out"); F=Path("figs"); F.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
C={"sales":"#222222","SNaive":"#9e9e9e","ETS":"#e69f00","M0":"#56b4e9","M1":"#d55e00"}
L={"sales":"Actual","SNaive":"Seasonal naive","ETS":"ETS","M0":"LightGBM M0","M1":"LightGBM M1"}
P=pd.read_csv(O/"predictions.csv.gz",parse_dates=["date"])
fig,axs=plt.subplots(1,2,figsize=(7.2,2.4))
for ax,(name,a) in zip(axs,[("Nauryz (21 Mar 2025)","2025-03-21"),("Kurban Ait (6 Jun 2025)","2025-06-06")]):
    a=pd.Timestamp(a); w=P[(P.cat=="meat")&(P.date>=a-pd.Timedelta(days=12))&(P.date<=a+pd.Timedelta(days=5))]
    agg=w.groupby("date")[["sales","ETS","M0","M1"]].sum()
    for c in ["sales","ETS","M0","M1"]:
        ax.plot(agg.index,agg[c],color=C[c],lw=1.8 if c in("sales","M1") else 1.1,label=L[c],ls="-" if c!="ETS" else "--")
    ax.axvline(a,color="#888",lw=0.6,ls=":"); ax.set_title(name+", meat category",fontsize=8)
    ax.set_ylabel("Units/day"); ax.tick_params(axis="x",rotation=30,labelsize=7)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3)); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
axs[0].legend(frameon=False,fontsize=7,loc="upper left")
fig.tight_layout(); fig.savefig(F/"holiday_forecasts.pdf"); fig.savefig(F/"holiday_forecasts.png",dpi=150)
ev=pd.read_csv(O/"metrics_events.csv",index_col=0)
fig,ax=plt.subplots(figsize=(3.5,2.3)); x=np.arange(len(ev)); wdt=0.2
for i,m in enumerate(["SNaive","ETS","M0","M1"]):
    ax.bar(x+(i-1.5)*wdt,ev[m],wdt,color=C[m],label=L[m])
ax.set_xticks(x); ax.set_xticklabels(ev.index); ax.set_ylabel("WAPE (%)"); ax.legend(frameon=False,fontsize=6.5,ncol=2)
fig.tight_layout(); fig.savefig(F/"event_wape.pdf"); fig.savefig(F/"event_wape.png",dpi=150)
s=pd.read_csv(O/"shap_importance_M1.csv",index_col=0).iloc[:,0].head(12)[::-1]
local=set("is_holiday hol_type_c is_bridge long_weekend d_to_hol d_after_hol d_to_nauryz d_after_nauryz d_to_kurban d_after_kurban d_to_newyear d_to_womens".split())
fig,ax=plt.subplots(figsize=(3.5,2.6))
ax.barh(s.index.str.replace("_c",""),s.values,color=["#d55e00" if i in local else "#56b4e9" for i in s.index])
ax.set_xlabel("Mean |SHAP value|"); ax.tick_params(axis="y",labelsize=7)
fig.tight_layout(); fig.savefig(F/"shap.pdf"); fig.savefig(F/"shap.png",dpi=150)
print("ok")
