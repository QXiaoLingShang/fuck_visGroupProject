# -*- coding: utf-8 -*-
"""成员C · 数据理解：经验、学历与行业类别（基于 load_flagged，未过滤）。

产出图：
  output/figs/C_exp_edu_freq.png      经验/学历编码频数
  output/figs/C_exp_edu_salary.png    各编码月薪中位数与四分位（按推断顺序）
  output/figs/C_exp_edu_heatmap.png   经验 × 学历 行归一化热力图
  output/figs/C_industry_longtail.png 行业招聘量 Top20 + 累计占比
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

plt = apply_style()
df = load_flagged()
N = len(df)
print("总行数:", N)

# ---------------------------------------------------------------- 1. 频数分布
exp_vc = df["experience"].value_counts()
edu_vc = df["education"].value_counts()
print("\n[经验频数]\n", exp_vc, "\n[学历频数]\n", edu_vc)
print("经验噪声编码(<100):", exp_vc[exp_vc < RARE_CATEGORY_MIN].to_dict(),
      "合计", exp_vc[exp_vc < RARE_CATEGORY_MIN].sum())
print("学历噪声编码(<100):", edu_vc[edu_vc < RARE_CATEGORY_MIN].to_dict(),
      "合计", edu_vc[edu_vc < RARE_CATEGORY_MIN].sum())
print("经验前三占比:", (exp_vc.head(3).sum() / N).round(4), " 学历前三占比:", (edu_vc.head(3).sum() / N).round(4))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
for ax, vc, title in ((axes[0], exp_vc, "经验要求：10 种编码，3 种为噪声编码"),
                      (axes[1], edu_vc, "学历要求：12 种编码，3 种为噪声编码")):
    vc = vc.sort_values()
    ax.barh(vc.index, vc.values, color=PALETTE[0], height=0.68)
    for y, v in enumerate(vc.values):   # 噪声编码的条形短到看不见，直接在数值后注明
        note = "（噪声编码）" if v < RARE_CATEGORY_MIN else ""
        ax.text(v + vc.max() * 0.012, y, f"{v:,}{note}", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, vc.max() * 1.17)
    ax.set_title(title)
    ax.set_xlabel("招聘记录数（条）")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e4:.0f}万" if x else "0"))
    ax.grid(axis="y", visible=False)
save_fig(fig, "output/figs/C_exp_edu_freq.png")

# ---------------------------------------------------------------- 2. 编码与薪酬
def sal_stat(col):
    g = df.groupby(col)["salary_mid_k"]
    out = pd.DataFrame({"n": df[col].value_counts(), "mean": g.mean(), "p25": g.quantile(.25),
                        "median": g.median(), "p75": g.quantile(.75)})
    form = pd.crosstab(df[col], df["salary_form"], normalize="index")
    out["day_hour_share"] = form.reindex(columns=["day", "hour"]).fillna(0).sum(axis=1)
    return out

exp_s, edu_s = sal_stat("experience"), sal_stat("education")
print("\n[经验×薪酬]\n", exp_s.sort_values("mean").round(3))
print("\n[学历×薪酬]\n", edu_s.sort_values("mean").round(3))
print("\n[Eby 薪酬写法]\n", df.loc[df.experience == "Eby", "salary_form"].value_counts())
print("[Gw 薪酬写法]\n", df.loc[df.education == "Gw", "salary_form"].value_counts())
print("[Gw 的经验分布]\n", df.loc[df.education == "Gw", "experience"].value_counts())
print("[日结(day)岗位的经验分布]\n", df.loc[df.salary_form == "day", "experience"].value_counts())
# 单调性检查：按推断顺序，中位数是否不减
print("经验中位数序列:", exp_s.loc[EXP_ORDER, "median"].tolist(),
      "单调不减:", exp_s.loc[EXP_ORDER, "median"].is_monotonic_increasing)
print("学历中位数序列:", edu_s.loc[EDU_ORDER, "median"].tolist(),
      "单调不减:", edu_s.loc[EDU_ORDER, "median"].is_monotonic_increasing)
print("经验均值单调:", exp_s.loc[EXP_ORDER, "mean"].is_monotonic_increasing,
      " 学历均值单调:", edu_s.loc[EDU_ORDER, "mean"].is_monotonic_increasing)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True, sharey=True)
for ax, s, order, title in ((axes[0], exp_s, EXP_ORDER, "经验编码：薪酬随推断等级上升"),
                            (axes[1], edu_s, EDU_ORDER, "学历编码：薪酬随推断等级上升")):
    s = s.loc[order]
    x = np.arange(len(order))
    ax.vlines(x, s["p25"], s["p75"], color=PALETTE[0], linewidth=5, alpha=0.35)   # 四分位区间
    ax.plot(x, s["median"], "o", color=PALETTE[0], markersize=7.5, zorder=3)      # 中位数
    for xi, m in zip(x, s["median"]):
        ax.text(xi + 0.16, m, f"{m:.1f}", va="center", fontsize=9, color=INK_2)
    ax.set_xticks(x)
    fmt = lambda n: f"{n/1e4:.1f}万" if n >= 1e4 else f"{n:,}"
    ax.set_xticklabels([f"{c}\n{fmt(s.loc[c, 'n'])}" for c in order], fontsize=9)
    ax.set_xlim(-0.5, len(order) - 0.3)
    ax.set_title(title)
    ax.set_xlabel("按平均月薪推断的等级顺序（低 → 高），编码下方为记录数")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("折算月薪中值（千元/月）")
axes[0].set_ylim(0, 40)
axes[0].text(0.03, 0.93, "圆点：中位数；竖条：P25–P75", transform=axes[0].transAxes, fontsize=9, color=INK_2)
save_fig(fig, "output/figs/C_exp_edu_salary.png")

# ---------------------------------------------------------------- 3. 经验 × 学历
ct = pd.crosstab(df["experience"], df["education"]).reindex(index=EXP_ORDER, columns=EDU_ORDER)
ct_row = ct.div(ct.sum(axis=1), axis=0)
print("\n[经验×学历 行占比 %]\n", (ct_row * 100).round(1))
df["_er"], df["_dr"] = df["experience"].map(EXP_RANK), df["education"].map(EDU_RANK)
print("经验等级与学历等级 Spearman:", df[["_er", "_dr"]].corr(method="spearman").iloc[0, 1].round(3))

fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
im = ax.imshow(ct_row.values * 100, cmap=seq_cmap(), aspect="auto", vmin=0, vmax=60)
ax.set_xticks(range(len(EDU_ORDER))); ax.set_xticklabels(EDU_ORDER)
ax.set_yticks(range(len(EXP_ORDER))); ax.set_yticklabels(EXP_ORDER)
for i in range(ct_row.shape[0]):
    for j in range(ct_row.shape[1]):
        v = ct_row.values[i, j] * 100
        if v >= 5:   # 只标注占比 ≥5% 的格子
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 30 else INK)
ax.set_xlabel("学历编码（推断等级 低 → 高）")
ax.set_ylabel("经验编码（推断等级 低 → 高）")
ax.set_title("经验等级越高，学历要求越向高等级编码集中")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("占该经验编码记录的比例（%）")
cb.outline.set_visible(False)
save_fig(fig, "output/figs/C_exp_edu_heatmap.png")

# ---------------------------------------------------------------- 4. 行业类别
ind = df["company_type"].value_counts()
cum = ind.cumsum() / N
print("\n行业数:", len(ind), " Top1:", ind.index[0], ind.iloc[0], f"{ind.iloc[0]/N:.2%}")
print(f"Top10 占比 {cum.iloc[9]:.2%}  Top20 占比 {cum.iloc[19]:.2%}  Top50 占比 {cum.iloc[49]:.2%}")
print("覆盖 80% 需行业数:", int((cum < 0.8).sum() + 1), " 覆盖 90%:", int((cum < 0.9).sum() + 1))
rare = ind[ind < RARE_INDUSTRY_MIN]
print("稀有行业(<30):", len(rare), rare.to_dict(), "合计记录", rare.sum())
print("招聘量中位数:", ind.median(), " 最小:", ind.min())
# 占位编码检查：形态、空白、常见缺失词
odd = [c for c in ind.index if not re.fullmatch(r"type_[A-Za-z]{6}", c)]
kw = [c for c in ind.index if re.search(r"(?i)nan|null|none|miss|unknown|缺|无|^\s*$", c)]
print("形态异常的行业编码:", odd, " 含缺失关键词:", kw, " 空值:", df["company_type"].isna().sum(),
      " 含首尾空白:", int((df["company_type"] != df["company_type"].str.strip()).sum()))
print("仅大小写不同的编码组数:", int((pd.Series(ind.index).str.lower().value_counts() > 1).sum()))

fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.15, 1]})
top = ind.head(20).sort_values()
ax = axes[0]
ax.barh([c.replace("type_", "") for c in top.index], top.values, color=PALETTE[0], height=0.7)
for y in (19, 18, 10, 0):   # 只标关键值：第 1、2、10、20 名
    ax.text(top.values[y] + 250, y, f"{top.values[y]:,}", va="center", fontsize=9, color=INK_2)
ax.set_xlim(0, top.max() * 1.13)
ax.set_title("招聘量 Top 20 行业")
ax.set_xlabel("招聘记录数（条）")
ax.grid(axis="y", visible=False)
ax = axes[1]
xs = np.arange(1, len(ind) + 1)
ax.plot(xs, cum.values * 100, color=PALETTE[0])
for k in (10, 20, 50):
    ax.plot(k, cum.iloc[k - 1] * 100, "o", color=HIGHLIGHT, markersize=7, zorder=3)
    ax.annotate(f"Top{k}：{cum.iloc[k-1]:.1%}", (k, cum.iloc[k - 1] * 100),
                xytext=(9, -13), textcoords="offset points", fontsize=9.5, color=INK)
ax.set_xlim(0, 160); ax.set_ylim(0, 102)
ax.set_title("158 个行业的累计招聘量占比")
ax.set_xlabel("行业按招聘量降序的名次")
ax.set_ylabel("累计占全部记录的比例（%）")
save_fig(fig, "output/figs/C_industry_longtail.png")

# ---------------------------------------------------------------- 5. 企业—行业一致性
k = df.groupby("company")["company_type"].nunique()
print("\n企业数:", len(k), f" ≥2 个行业的企业: {(k >= 2).sum()} ({(k >= 2).mean():.2%})",
      f" ≥3: {(k >= 3).sum()} ({(k >= 3).mean():.2%})", " 最大:", k.max())
posts = df["company"].value_counts()
multi_post = posts[posts >= 2]
print("发布 ≥2 条的企业数:", len(multi_post),
      f" 其中 ≥2 行业占比: {(k[multi_post.index] >= 2).mean():.2%}")
print("≥2 行业企业涉及的记录数:", int(posts[k[k >= 2].index].sum()),
      f"({posts[k[k >= 2].index].sum() / N:.2%})")
# 按发布量分组看多行业比例：若随发布量稳定上升，更像多元经营/多标签而非随机错标
grp = pd.cut(posts, [0, 1, 2, 4, 9, 1e9], labels=["1", "2", "3-4", "5-9", "≥10"])
print("[按发布量分组的多行业企业占比]\n", (k >= 2).groupby(grp, observed=True).agg(["size", "mean"]).round(4))
# 多行业企业中，主行业（最多的那个）占该企业记录的平均比例
m = df[df["company"].isin(k[k >= 2].index)]
dom = m.groupby("company")["company_type"].agg(lambda s: s.value_counts().iloc[0] / len(s))
print("多行业企业的主行业记录占比 均值/中位数:", dom.mean().round(3), dom.median().round(3))
