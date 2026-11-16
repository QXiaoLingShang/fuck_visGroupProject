# -*- coding: utf-8 -*-
"""成员C · 数据处理：职位与行业维度的统计与分析（基于 load_clean）。

产出表：
  data/processed/C_industry_profile.csv   行业级汇总
  data/processed/C_job_profile.csv        职位级汇总
产出图：
  output/figs/C_industry_scatter.png      行业 招聘量 × 薪酬中位数
  output/figs/C_industry_salary_box.png   Top20 行业薪酬分布
  output/figs/C_industry_structure.png    Top12 行业的经验/学历结构
  output/figs/C_job_longtail_top15.png    职位长尾 + Top15 高频职位画像
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

plt = apply_style()
df = load_clean()
N = len(df)
print("清洗后行数:", N)

# ------------------------------------------------------------ 0. 记录级派生列
df["exp_rank"] = df["experience"].map(EXP_RANK)
df["edu_rank"] = df["education"].map(EDU_RANK)
df["is_core_city"] = df["city"].map(city_tier_map(df)).eq("核心城市")
# 薪酬溢价：记录月薪 / 同“经验×学历”组合的月薪中位数 - 1（剔除经验学历差异后的相对薪酬）
cell_med = df.groupby(["experience", "education"])["salary_mid_k"].transform("median")
df["premium"] = df["salary_mid_k"] / cell_med - 1
ALL_MED = df["salary_mid_k"].median()
print("全体月薪中位数:", ALL_MED, " 核心城市记录占比:", round(df["is_core_city"].mean(), 4),
      " 13薪占比:", round(df["has_bonus_months"].mean(), 4))


def shannon(s):
    """香农熵（bit）：职位名称在组内分布得越分散，熵越大。"""
    p = s.value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())


def zscore(s):
    return (s - s.mean()) / s.std()


# ------------------------------------------------------------ 1. 行业级汇总表
g = df.groupby("company_type")
ind = pd.DataFrame({
    "n_posts": g.size(),
    "n_companies": g["company"].nunique(),
    "n_job_titles": g["job_title"].nunique(),
    "salary_median": g["salary_mid_k"].median(),
    "salary_mean": g["salary_mid_k"].mean(),
    "salary_p25": g["salary_mid_k"].quantile(.25),
    "salary_p75": g["salary_mid_k"].quantile(.75),
    "exp_rank_mean": g["exp_rank"].mean(),
    "edu_rank_mean": g["edu_rank"].mean(),
    "bonus_share": g["has_bonus_months"].mean(),
    "core_city_share": g["is_core_city"].mean(),
    "title_per_post": g["job_title"].nunique() / g.size(),
    "title_entropy": g["job_title"].agg(shannon),
    "premium_mean": g["premium"].mean(),          # 控制经验×学历后的平均薪酬溢价
})
ind["post_share"] = ind["n_posts"] / N
# 问题5 的“急需人才”截面代理（初步）：溢价高 + 企业覆盖广 + 经验要求低
ind["demand_proxy"] = (zscore(ind["premium_mean"]) + zscore(np.log10(ind["n_companies"]))
                       - zscore(ind["exp_rank_mean"]))
ind = ind.sort_values("n_posts", ascending=False)
ind.round(4).to_csv(os.path.join(PROCESSED_DIR, "C_industry_profile.csv"), encoding="utf-8-sig",
                    index_label="company_type")
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\n行业数:", len(ind))
print(ind.head(12).round(3))
print("\n[行业指标描述]\n", ind[["n_posts", "n_companies", "salary_median", "exp_rank_mean", "edu_rank_mean",
                           "bonus_share", "core_city_share", "title_per_post", "title_entropy",
                           "premium_mean"]].describe().round(3))
print("\n[薪酬中位数最高 5]\n", ind.nlargest(5, "salary_median")[["n_posts", "n_companies", "salary_median", "bonus_share", "edu_rank_mean"]].round(3))
print("[薪酬中位数最低 5]\n", ind.nsmallest(5, "salary_median")[["n_posts", "n_companies", "salary_median", "bonus_share", "edu_rank_mean"]].round(3))
print("[demand_proxy 最高 8]\n", ind.nlargest(8, "demand_proxy")[["n_posts", "n_companies", "salary_median", "premium_mean", "exp_rank_mean", "demand_proxy"]].round(3))
cc = ind[["n_posts", "salary_median", "exp_rank_mean", "edu_rank_mean", "bonus_share", "core_city_share"]].corr(method="spearman")
print("[行业指标 Spearman]\n", cc.round(2))

# ---- 图1：招聘量 × 薪酬中位数
fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
lab = list(dict.fromkeys(list(ind.nlargest(3, "salary_median").index) + list(ind.nlargest(3, "n_posts").index)
                         + list(ind.nsmallest(2, "salary_median").index)))
is_lab = ind.index.isin(lab)
sz = 12 + ind["n_companies"] / ind["n_companies"].max() * 420
ax.scatter(ind.loc[~is_lab, "n_posts"], ind.loc[~is_lab, "salary_median"], s=sz[~is_lab],
           color=PALETTE[0], alpha=0.45, edgecolor=SURFACE, linewidth=0.8)
ax.scatter(ind.loc[is_lab, "n_posts"], ind.loc[is_lab, "salary_median"], s=sz[is_lab],
           color=HIGHLIGHT, alpha=0.85, edgecolor=SURFACE, linewidth=0.8, zorder=3)
top_sal = list(ind.nlargest(3, "salary_median").index)
offs = dict(zip(top_sal, [(8, 5), (-52, 8), (-22, 9)]))   # 三个并列最高薪行业的标签错开摆放
for c in lab:
    ax.annotate(c.replace("type_", ""), (ind.loc[c, "n_posts"], ind.loc[c, "salary_median"]),
                xytext=offs.get(c, (7, 7)), textcoords="offset points", fontsize=9, color=INK)
ax.set_ylim(3.8, 13.3)
ax.axhline(ALL_MED, color=MUTED, linewidth=1)
ax.text(ind["n_posts"].min(), ALL_MED + 0.15, f"全体中位数 {ALL_MED:.1f}", fontsize=9, color=INK_2)
ax.set_xscale("log")
ax.set_xlabel("行业招聘量（条，对数刻度）")
ax.set_ylabel("月薪中值的行业中位数（千元/月）")
ax.set_title("行业招聘量与薪酬水平仅弱负相关，最高薪行业均为中小体量")
for v in (500, 5000, 15000):   # 点大小图例
    ax.scatter([], [], s=12 + v / ind["n_companies"].max() * 420, color=NEUTRAL, label=f"{v:,} 家企业")
ax.legend(loc="lower left", labelspacing=1.1, borderpad=0.8)
save_fig(fig, "output/figs/C_industry_scatter.png")

# ---- 图2：Top20 行业薪酬分布（按中位数排序）
top20 = ind.head(20).sort_values("salary_median").index
fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
data = [df.loc[df["company_type"] == c, "salary_mid_k"].values for c in top20]
ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, widths=0.62, patch_artist=True,
           boxprops=dict(facecolor=PALETTE[0], alpha=0.35, linewidth=0), medianprops=dict(color=PALETTE[0], linewidth=2.4),
           whiskerprops=dict(color=AXIS, linewidth=1.2), capprops=dict(color=AXIS, linewidth=1.2))
ax.set_yticks(range(1, 21)); ax.set_yticklabels([c.replace("type_", "") for c in top20])
ax.axvline(ALL_MED, color=MUTED, linewidth=1)
ax.text(ALL_MED + 0.3, 0.45, f"全体中位数 {ALL_MED:.1f}", fontsize=9, color=INK_2)
ax.set_ylim(0.2, 20.7)
ax.set_xlabel("月薪中值（千元/月）；箱体为 P25–P75，须线为 P5–P95")
ax.set_title("招聘量 Top 20 行业的薪酬分布：中位数 4.5～11.5 千元/月，高薪行业内部更分散")
ax.grid(axis="y", visible=False)
save_fig(fig, "output/figs/C_industry_salary_box.png")
t = ind.loc[top20, ["n_posts", "salary_p25", "salary_median", "salary_p75"]]
t["iqr"] = t["salary_p75"] - t["salary_p25"]
print("\n[Top20 行业薪酬，按中位数升序]\n", t.round(2))

# ---- 图3：Top12 行业的经验/学历结构（行归一化热力图，行按薪酬中位数降序）
top12 = ind.head(12).sort_values("salary_median", ascending=False).index
sub = df[df["company_type"].isin(top12)]
e = pd.crosstab(sub["company_type"], sub["experience"], normalize="index").reindex(index=top12, columns=EXP_ORDER).fillna(0) * 100
d = pd.crosstab(sub["company_type"], sub["education"], normalize="index").reindex(index=top12, columns=EDU_ORDER).fillna(0) * 100
print("\n[Top12 经验结构 %]\n", e.round(1), "\n[Top12 学历结构 %]\n", d.round(1))
print(ind.loc[top12, ["salary_median", "exp_rank_mean", "edu_rank_mean"]].round(2))
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [7, 9]})
for ax, m, title, xl in ((axes[0], e, "经验结构", "经验编码（推断等级 低 → 高）"),
                         (axes[1], d, "学历结构", "学历编码（推断等级 低 → 高）")):
    im = ax.imshow(m.values, cmap=seq_cmap(), aspect="auto", vmin=0, vmax=70)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(m.shape[0]))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if v >= 10:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 35 else INK)
    ax.set_xlabel(xl); ax.set_title(f"Top 12 行业的{title}"); ax.grid(False)
axes[0].set_yticklabels([f"{c.replace('type_', '')}  {ind.loc[c, 'salary_median']:.1f}" for c in top12])
axes[1].set_yticklabels([])
axes[0].set_ylabel("行业（按薪酬中位数降序，数字为千元/月）")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
cb.set_label("占该行业记录的比例（%）"); cb.outline.set_visible(False)
save_fig(fig, "output/figs/C_industry_structure.png")

# ------------------------------------------------------------ 2. 职位级汇总表
g = df.groupby("job_title")
job = pd.DataFrame({
    "n_posts": g.size(),
    "n_companies": g["company"].nunique(),
    "n_cities": g["city"].nunique(),
    "n_industries": g["company_type"].nunique(),
    "salary_median": g["salary_mid_k"].median(),
    "salary_p25": g["salary_mid_k"].quantile(.25),
    "salary_p75": g["salary_mid_k"].quantile(.75),
    "exp_rank_mean": g["exp_rank"].mean(),
    "edu_rank_mean": g["edu_rank"].mean(),
    "bonus_share": g["has_bonus_months"].mean(),
    "premium_mean": g["premium"].mean(),
}).sort_values("n_posts", ascending=False)
# 急需人才代理只对样本充足的职位计算（≥30 条），其余留空
hf = job["n_posts"] >= 30
job.loc[hf, "demand_proxy"] = (zscore(job.loc[hf, "premium_mean"]) + zscore(np.log10(job.loc[hf, "n_companies"]))
                               - zscore(job.loc[hf, "exp_rank_mean"]))
job.round(4).to_csv(os.path.join(PROCESSED_DIR, "C_job_profile.csv"), encoding="utf-8-sig", index_label="job_title")

J = len(job)
print("\n职位种数:", J, " 只出现1次:", int((job.n_posts == 1).sum()), f"({(job.n_posts == 1).mean():.2%})",
      " 这些职位的记录占比:", f"{(job.n_posts == 1).sum() / N:.2%}")
for thr in (2, 5, 10, 30, 50, 100, 300, 1000):
    m = job["n_posts"] >= thr
    print(f"  出现≥{thr}: {int(m.sum())} 种 ({m.mean():.3%})，覆盖招聘量 {job.loc[m, 'n_posts'].sum() / N:.2%}")
print("[≥30 职位的企业数/城市数/行业数 中位数]", job.loc[hf, ["n_companies", "n_cities", "n_industries"]].median().to_dict())
print("[≥30 职位薪酬中位数的分位]", job.loc[hf, "salary_median"].quantile([.05, .25, .5, .75, .95]).round(2).to_dict())
top15 = job.head(15)
print("\n[Top15 职位]\n", top15.round(3))
print("Top15 覆盖招聘量:", f"{top15.n_posts.sum() / N:.2%}")
print("[≥100 职位 demand_proxy 最高 8]\n", job[job.n_posts >= 100].nlargest(8, "demand_proxy")[
    ["n_posts", "n_companies", "salary_median", "premium_mean", "exp_rank_mean", "demand_proxy"]].round(3))

# ---- 图4：职位长尾（分箱条形） + Top15 高频职位画像
bins = [0, 1, 2, 4, 9, 29, 99, np.inf]
labels = ["1", "2", "3–4", "5–9", "10–29", "30–99", "≥100"]
b = pd.cut(job["n_posts"], bins=bins, labels=labels)
n_titles = b.value_counts().reindex(labels)
post_share = job.groupby(b, observed=False)["n_posts"].sum().reindex(labels) / N * 100
print("\n[分箱] 职位种数:\n", n_titles.to_dict(), "\n 招聘量占比%:\n", post_share.round(2).to_dict())

fig = plt.figure(figsize=(11, 6.4), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1])
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
ax1.bar(labels, n_titles.values, color=PALETTE[0], width=0.7)
ax1.set_yscale("log")
for x, v in enumerate(n_titles.values):
    ax1.text(x, v * 1.15, f"{v:,}", ha="center", fontsize=8.5, color=INK_2)
ax1.set_ylim(top=n_titles.max() * 4)
ax1.set_title("职位种数：约九成只出现 1 次" if (job.n_posts == 1).mean() > 0.85 else "职位种数：绝大多数只出现 1 次")
ax1.set_ylabel("职位种数（对数刻度）"); ax1.grid(axis="x", visible=False)
ax2.bar(labels, post_share.values, color=PALETTE[0], width=0.7)
for x, v in enumerate(post_share.values):
    ax2.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5, color=INK_2)
ax2.set_ylim(0, post_share.max() * 1.15)
ax2.set_title("对应的招聘量占比")
ax2.set_xlabel("职位出现次数（条）"); ax2.set_ylabel("占全部记录的比例（%）"); ax2.grid(axis="x", visible=False)

t = top15.iloc[::-1]
names = [f"{i[:6]}…" for i in t.index]
ax3 = fig.add_subplot(gs[:, 1])
ax3.barh(names, t["n_posts"], color=PALETTE[0], height=0.68)
for y in (14, 0):
    ax3.text(t["n_posts"].iloc[y] + t["n_posts"].max() * 0.02, y, f"{t['n_posts'].iloc[y]:,}", va="center", fontsize=9, color=INK_2)
ax3.set_xlim(0, t["n_posts"].max() * 1.2)
ax3.set_title("Top 15 职位：招聘量"); ax3.set_xlabel("招聘记录数（条）"); ax3.set_ylabel("职位编码（前 6 位）")
ax3.grid(axis="y", visible=False)
ax4 = fig.add_subplot(gs[:, 2], sharey=ax3)
y = np.arange(len(t))
ax4.hlines(y, t["salary_p25"], t["salary_p75"], color=PALETTE[0], linewidth=5, alpha=0.35)
ax4.plot(t["salary_median"], y, "o", color=PALETTE[0], markersize=7.5)
ax4.axvline(ALL_MED, color=MUTED, linewidth=1)
ax4.set_xlim(left=0)
ax4.tick_params(labelleft=False)
ax4.set_title("薪酬中位数与 P25–P75"); ax4.set_xlabel("月薪中值（千元/月），竖线为全体中位数")
ax4.grid(axis="y", visible=False)
save_fig(fig, "output/figs/C_job_longtail_top15.png")
