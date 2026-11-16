# -*- coding: utf-8 -*-
"""成员D · 数据理解：行政区划（city）与企业（company）两个字段。

口径：load_flagged()（未过滤的全部 430,664 条）。
产出：output/figs/D_city_rank_hist.png、D_city_tier.png、D_city_letter.png、D_company_bins.png
      data/processed/D_city_tier_table.csv、D_city_letter_table.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

plt = apply_style()
df = load_flagged()
N = len(df)
print("总行数:", N, " 城市数:", df["city"].nunique(), " 企业数:", df["company"].nunique())

# ---------------------------------------------------------------------------
# 1. 城市招聘量分布与分级
# ---------------------------------------------------------------------------
vc = df["city"].value_counts()                 # 每个城市的招聘条数（降序）
tier = city_tier_map(df).reindex(vc.index)     # 全组统一的城市分级
print("城市编码格式:", vc.index.str.replace(r"\d", "9", regex=True).str.replace(r"[A-Z]", "A", regex=True).value_counts().to_dict())

THRESH = {"核心城市": ">5000 条", "中等城市": "401~5000 条", "普通城市": "200~400 条", "小样本城市": "<200 条"}
rows = []
for t in CITY_TIER_ORDER:
    s = vc[tier == t]
    rows.append({"等级": t, "阈值": THRESH[t], "城市数": len(s), "招聘量范围": f"{s.min()}~{s.max()}",
                 "招聘量": int(s.sum()), "占比": f"{s.sum() / N:.2%}"})
tier_tab = pd.DataFrame(rows)
tier_tab.to_csv(os.path.join(PROCESSED_DIR, "D_city_tier_table.csv"), index=False, encoding="utf-8-sig")
print(tier_tab.to_string(index=False))
print("核心14城:", vc[tier == "核心城市"].to_dict())
print("中等4城:", vc[tier == "中等城市"].to_dict())
normal = vc[tier == "普通城市"]
print("普通城市 招聘量 描述:", normal.describe().round(1).to_dict())
print("普通城市 290~300 条的城市数:", int(((normal >= 290) & (normal <= 300)).sum()),
      " 众数:", normal.mode().tolist(), " 280~300:", int(((normal >= 280) & (normal <= 300)).sum()),
      " 最大值及个数:", normal.max(), int((normal == normal.max()).sum()))
print("普通城市取值频数 Top8:", normal.value_counts().head(8).to_dict())
print("小样本城市:", vc[tier == "小样本城市"].to_dict())

TIER_COLOR = dict(zip(CITY_TIER_ORDER, PALETTE[:4]))

# 图1：rank-size（对数轴）+ 200~400 条区间直方图
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
rank = np.arange(1, len(vc) + 1)
for t in CITY_TIER_ORDER:
    m = (tier == t).values
    ax.scatter(rank[m], vc.values[m], s=16, color=TIER_COLOR[t], label=f"{t}（{m.sum()} 个）", zorder=3)
ax.set_yscale("log")
ax.set_xlabel("城市按招聘量的排名")
ax.set_ylabel("招聘条数（对数轴）")
ax.set_title("371 个城市的招聘量呈“断崖 + 平台”形态")
ax.legend(loc="upper right")
ax.annotate("14 个城市各 2.2~2.6 万条", xy=(14, vc.iloc[13]), xytext=(60, 9000),
            fontsize=9.5, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate(f"约 {len(normal)} 个城市停在 300 条左右", xy=(200, 300), xytext=(110, 1100),
            fontsize=9.5, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

ax = axes[1]
bins = np.arange(200, 405, 5)
ax.hist(normal.values, bins=bins, color=PALETTE[0])
ax.axvline(normal.median(), color=HIGHLIGHT, lw=1.5)
ax.text(normal.median() - 4, ax.get_ylim()[1] * 0.92, f"中位数 {normal.median():.0f} 条",
        color=HIGHLIGHT, ha="right", fontsize=9.5)
ax.set_xlabel("招聘条数（每 5 条一组）")
ax.set_ylabel("城市数（个）")
ax.set_title("普通城市的招聘量高度集中")
save_fig(fig, "output/figs/D_city_rank_hist.png")

# 图2：城市分级的城市数与招聘量占比（并排子图）
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, sharey=True)
ylab = CITY_TIER_ORDER[::-1]
cnt = tier_tab.set_index("等级")["城市数"].reindex(ylab)
shr = (tier_tab.set_index("等级")["招聘量"] / N * 100).reindex(ylab)
cols = [HIGHLIGHT if t == "核心城市" else NEUTRAL for t in ylab]
axes[0].barh(ylab, cnt.values, color=cols, height=0.62)
for y, v in enumerate(cnt.values):
    axes[0].text(v + 5, y, f"{v} 个", va="center", fontsize=9.5, color=INK_2)
axes[0].set_xlim(0, 380); axes[0].set_xlabel("城市数（个）")
axes[0].set_title("城市数：九成是普通城市")
axes[1].barh(ylab, shr.values, color=cols, height=0.62)
for y, v in enumerate(shr.values):
    axes[1].text(v + 1.2, y, f"{v:.1f}%", va="center", fontsize=9.5, color=INK_2)
axes[1].set_xlim(0, 100); axes[1].set_xlabel("占全部招聘记录的比例（%）")
axes[1].set_title("招聘量：14 个核心城市占四分之三")
for a in axes:
    a.grid(axis="y", visible=False)
save_fig(fig, "output/figs/D_city_tier.png")

# ---------------------------------------------------------------------------
# 2. 城市编码首字母是否有含义
# ---------------------------------------------------------------------------
letter = pd.Series(vc.index.str[0], index=vc.index)
lt = pd.DataFrame({"城市数": letter.value_counts(),
                   "招聘量": vc.groupby(letter).sum(),
                   "核心城市数": (tier == "核心城市").groupby(letter).sum(),
                   "招聘量中位数": vc.groupby(letter).median()}).sort_index()
lt["招聘量占比"] = (lt["招聘量"] / N).map("{:.2%}".format)
lt.to_csv(os.path.join(PROCESSED_DIR, "D_city_letter_table.csv"), encoding="utf-8-sig")
print(lt.to_string())
print("核心城市编码:", sorted(vc.index[tier == "核心城市"]))
# 首字母与城市等级是否独立（卡方）
from scipy.stats import chi2_contingency
ct = pd.crosstab(letter, tier.replace({"中等城市": "非核心", "普通城市": "非核心", "小样本城市": "非核心"}))
chi2, p, dof, _ = chi2_contingency(ct)
print(f"首字母×是否核心 卡方: chi2={chi2:.2f}, dof={dof}, p={p:.3f}")

# 城市×行业 占比矩阵（全部行业），计算城市间余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
ci = pd.crosstab(df["city"], df["company_type"])
share = ci.div(ci.sum(axis=1), axis=0)


def within_between(cities, n_perm=2000, seed=0):
    """同首字母（组内）与不同首字母（组间）城市对的平均行业结构余弦相似度，及置换检验。"""
    S = cosine_similarity(share.loc[cities].values)
    lab = np.array([c[0] for c in cities])
    iu = np.triu_indices(len(cities), 1)
    sim = S[iu]
    same = lab[iu[0]] == lab[iu[1]]
    obs = sim[same].mean() - sim[~same].mean()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):          # 随机打乱首字母标签，得到“首字母无含义”时差值的分布
        pl = rng.permutation(lab)
        ps = pl[iu[0]] == pl[iu[1]]
        null[i] = sim[ps].mean() - sim[~ps].mean()
    pval = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return sim[same].mean(), sim[~same].mean(), obs, pval, null, int(same.sum()), int((~same).sum())


res = {}
for name, cities in (("招聘量≥200 的城市", list(vc.index[vc >= 200])),
                     ("普通城市", list(vc.index[tier == "普通城市"])),
                     ("核心城市", list(vc.index[tier == "核心城市"]))):
    w, b, d, p, null, nw, nb = within_between(cities)
    res[name] = (w, b, d, p, null)
    print(f"[{name}] n={len(cities)} 组内对数={nw} 组间对数={nb} 组内均值={w:.4f} 组间均值={b:.4f} 差={d:+.4f} 置换p={p:.3f}"
          f"  null范围(2.5%,97.5%)=({np.percentile(null, 2.5):+.4f},{np.percentile(null, 97.5):+.4f})")

# 图3：首字母分组的城市数（按等级堆叠）+ 置换检验
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
st = pd.crosstab(letter, tier).reindex(columns=CITY_TIER_ORDER, fill_value=0).sort_index()
bottom = np.zeros(len(st))
for t in CITY_TIER_ORDER:
    ax.bar(st.index, st[t].values, bottom=bottom, color=TIER_COLOR[t], label=t, width=0.72)
    bottom += st[t].values
ax.set_xlabel("城市编码首字母"); ax.set_ylabel("城市数（个）"); ax.set_yticks(range(0, 26, 5)); ax.set_ylim(0, 24)
ax.set_title("各首字母下的城市数大致均匀，核心城市分散其中")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), columnspacing=1.2, handlelength=1.2)
ax.grid(axis="x", visible=False)

ax = axes[1]
w, b, d, p, null = res["普通城市"]
ax.hist(null, bins=40, color=NEUTRAL)
ax.axvline(d, color=HIGHLIGHT, lw=2)
ax.set_xlabel("组内 − 组间 平均余弦相似度")
ax.set_ylabel("置换次数")
ax.set_title("同首字母城市的行业结构并不更相似")
ymax = ax.get_ylim()[1]
ax.set_ylim(0, ymax * 1.18)
ax.text(d, ymax * 1.06, f"实际差值 {d:+.4f}（p={p:.2f}）", color=HIGHLIGHT, fontsize=9.5,
        ha="left" if d < np.median(null) else "right")
ax.text(0.02, 0.80, "灰色：随机打乱首字母\n2000 次的差值分布", transform=ax.transAxes, fontsize=9, color=INK_2, va="top")
save_fig(fig, "output/figs/D_city_letter.png")

# ---------------------------------------------------------------------------
# 3. 企业（company）
# ---------------------------------------------------------------------------
cc = df["company"].value_counts()
print("企业数:", len(cc), " 只发1条:", int((cc == 1).sum()), f"{(cc == 1).mean():.2%}",
      " ≤2条:", f"{(cc <= 2).mean():.2%}", " 最多:", int(cc.max()), " 均值:", round(cc.mean(), 3), " 中位:", cc.median())
ncity = df.groupby("company")["city"].nunique()
print("跨城市: ≥2城企业数", int((ncity >= 2).sum()), f"{(ncity >= 2).mean():.2%}", " 最多城市数:", int(ncity.max()))
multi = cc[cc >= 2].index
print("发布≥2条的企业数:", len(multi), " 其中跨≥2城占比:", f"{(ncity.loc[multi] >= 2).mean():.2%}")
print("每家企业的行业种数>1 的占比:", f"{(df.groupby('company')['company_type'].nunique() > 1).mean():.2%}")

labels = ["1", "2", "3-5", "6-10", "11-50", ">50"]
cb = pd.cut(cc, bins=[0, 1, 2, 5, 10, 50, np.inf], labels=labels)
comp_n = cb.value_counts().reindex(labels)
post_n = cc.groupby(cb, observed=False).sum().reindex(labels)
bins_tab = pd.DataFrame({"企业数": comp_n, "企业占比": comp_n / len(cc), "招聘条数": post_n, "招聘占比": post_n / N})
print(bins_tab.to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True, sharey=True)
for ax, col, raw, unit, title in ((axes[0], "企业占比", "企业数", "家", "企业数占比：七成企业只发布 1 条"),
                                  (axes[1], "招聘占比", "招聘条数", "条", "招聘量占比：这些企业贡献约 44% 的记录")):
    v = bins_tab[col].values * 100
    ax.bar(labels, v, color=PALETTE[0], width=0.66)
    for x, (p_, r_) in enumerate(zip(v, bins_tab[raw].values)):
        ax.text(x, p_ + 1.2, f"{p_:.1f}%\n{int(r_):,} {unit}", ha="center", va="bottom", fontsize=8.8, color=INK_2)
    ax.set_xlabel("每家企业发布的招聘条数")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("占比（%）")
axes[0].set_ylim(0, 95)
save_fig(fig, "output/figs/D_company_bins.png")
print("done")
