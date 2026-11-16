# -*- coding: utf-8 -*-
"""成员D · 第二轮 · 数据理解：行政区划与企业。

口径：招聘量与企业统计用 load_flagged()（未过滤 430,664 条）；城市分组（H2）用 load_enriched()（过滤后 429,086 条）。
产出：output/figs/D2_core_rank.png
      data/processed/D2_city_group_table.csv、D2_core_rank.csv
旧图 D_city_rank_hist.png、D_company_bins.png 由 D_understand.py 生成，本脚本只复核其中引用的数字。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D2_lib import *

plt = apply_style()
f = load_flagged().reset_index(drop=True)
f["batch"] = infer_batches(f)
df = load_enriched()

# ---------------------------------------------------------------------------
# 1. 招聘量的“断崖 + 平台”与采集批次（H3）
# ---------------------------------------------------------------------------
vc = f["city"].value_counts()
print("总行数", len(f), "城市数", len(vc))
print("前 14 城:", vc.iloc[13], "~", vc.iloc[0], "| 第 15~18:", vc.iloc[14:18].to_dict())
rest = vc.iloc[18:][vc.iloc[18:] >= 200]
print("其后 ≥200 条的城市数", len(rest), "范围", rest.min(), "~", rest.max(), "中位数", rest.median(),
      "290~300 条的城市数", int(rest.between(290, 300).sum()), "<200 条城市数", int((vc < 200).sum()))
print("各批次行数:", f["batch"].value_counts().to_dict())
a = f[f["batch"] == "A"]
blk = (a["city"] != a["city"].shift()).cumsum()
bl = a.groupby(blk).agg(city=("city", "first"), n=("city", "size"))
print("批次 A：城市数", a["city"].nunique(), "块数", len(bl), "块长中位数", bl["n"].median(), "最大块", bl["n"].max(),
      "| 每城条数中位数", a["city"].value_counts().median())
print("批次 A 中出现多个块的城市:", bl[bl["n"] >= 100]["city"].value_counts().loc[lambda s: s > 1].to_dict(),
      "| 超过 300 行的块:", bl[bl["n"] > 300].to_dict("records"))
bc = f[f["batch"] != "A"]
print("批次 B、C 中核心城市行数占比", round(bc["city"].isin(CORE).mean(), 5), "核心城市每城 B/C 行数范围:",
      bc[bc["city"].isin(CORE)].groupby(["batch", "city"]).size().groupby("batch").agg(["min", "max"]).to_dict("index"))

# ---------------------------------------------------------------------------
# 2. H2：14 个核心城市的薪酬与学历排序
# ---------------------------------------------------------------------------
core = df[df["city"].isin(CORE)]
g = core.groupby("city")
rank = pd.DataFrame({
    "city_group": g["city_group"].first(), "n": g.size(),
    "salary_mean": g["salary_mid_k"].mean(), "salary_median": g["salary_mid_k"].median(),
    "bachelor_share": g["edu_level"].apply(lambda s: (s >= 4).mean()),
}).sort_values("salary_mean", ascending=False)
rank.round(4).to_csv(os.path.join(PROCESSED_DIR, "D2_core_rank.csv"), encoding="utf-8-sig")
print(rank.round(3).to_string())
gaps = -rank["salary_mean"].diff().dropna()
print("相邻名次的均值差：第4-5名", round(gaps.iloc[3], 2), "其余最大", round(gaps.drop(gaps.index[3]).max(), 2))
bs = rank["bachelor_share"].sort_values(ascending=False)
print("本科+占比降序:", bs.round(3).to_dict())
print("本科+占比 第4-5名差", round(bs.iloc[3] - bs.iloc[4], 3), "一线型范围", rank[rank.city_group == "一线型"]["bachelor_share"].agg(["min", "max"]).round(3).tolist(),
      "新一线型范围", rank[rank.city_group == "新一线型"]["bachelor_share"].agg(["min", "max"]).round(3).tolist())

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True, sharey=True)
order = rank.index[::-1]
cols = [HIGHLIGHT if rank.loc[c, "city_group"] == "一线型" else NEUTRAL for c in order]
ax = axes[0]
ax.barh(order, rank.loc[order, "salary_mean"], color=cols, height=0.66)
ax.scatter(rank.loc[order, "salary_median"], order, marker="|", s=170, color=INK, linewidths=1.8, zorder=3)
ax.axhline(9.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax.set_xlim(0, 15.4)
ax.set_xlabel("月薪（千元/月）")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=HIGHLIGHT, label="一线型：均值"), Patch(color=NEUTRAL, label="新一线型：均值"),
                   Line2D([], [], marker="|", color=INK, ls="", markersize=11, markeredgewidth=1.8, label="中位数")],
          loc="lower right", fontsize=9)
ax = axes[1]
ax.barh(order, rank.loc[order, "bachelor_share"] * 100, color=cols, height=0.66)
ax.axhline(9.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax.set_xlabel("本科及以上占比（%）")
for a_ in axes:
    a_.grid(axis="y", visible=False)
axes[0].set_ylabel("核心城市（按薪酬均值排序）")
save_fig(fig, "output/figs/D2_core_rank.png")

# ---------------------------------------------------------------------------
# 3. 更新后的城市分组表
# ---------------------------------------------------------------------------
g = df.groupby("city_group")
tab = pd.DataFrame({
    "城市数": g["city"].nunique(), "记录数": g.size(), "占比": (g.size() / len(df)).map("{:.2%}".format),
    "薪酬中位数(千元/月)": g["salary_mid_k"].median(),
    "本科+占比": g["edu_level"].apply(lambda s: (s >= 4).mean()).map("{:.1%}".format),
}).reindex(CITY_GROUP_ORDER).rename_axis("分组")
tab.to_csv(os.path.join(PROCESSED_DIR, "D2_city_group_table.csv"), encoding="utf-8-sig")
print(tab.to_string())

# ---------------------------------------------------------------------------
# 4. 企业稀疏性（复核旧图 D_company_bins.png 引用的数字）
# ---------------------------------------------------------------------------
cc = f["company"].value_counts()
ncity = f.groupby("company")["city"].nunique()
print("企业数", len(cc), "平均每家", round(cc.mean(), 2), "只发 1 条", int((cc == 1).sum()), f"{(cc == 1).mean():.2%}",
      "这些企业的记录占比", f"{(cc == 1).sum() / len(f):.1%}", "≤2 条", f"{(cc <= 2).mean():.2%}",
      ">50 条", int((cc > 50).sum()), "最多", int(cc.max()), "| 跨 ≥2 城", int((ncity >= 2).sum()), f"{(ncity >= 2).mean():.2%}")
print("done")
