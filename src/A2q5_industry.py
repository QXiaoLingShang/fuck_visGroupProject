# -*- coding: utf-8 -*-
"""问题 5（上）：行业/职位族的截面指标、行业类型划分、批次差异（H4）的检验。

输出
    data/processed/A2q5_industry_indicators.csv   152 个行业的指标与类型
    data/processed/A2q5_family_indicators.csv     51 个职位族的指标（供 A2q5_emerging.py 使用）
    data/processed/A2q5_industry_types.csv        各行业类型的个数与代表行业
    data/processed/A2q5_h4_check.csv              H4 检验的关键数字
    output/figs/A2q5_industry_types.png           规模 × 薪酬溢价散点
    output/figs/A2q5_segment_matrix.png           城市 × 职位族的“批次 B 占比”矩阵（按采集组排序）
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *
from scipy import stats
import itertools

plt = apply_style()
MIN_N = 300          # 行业参与分类的最小记录数（核心城市批次）
PREM_CUT = 0.05      # 溢价的分类阈值 ±5%

df = load_enriched()
# 口径：14 个核心城市的批次 B、C 合并（每城覆盖完整的类目列表；批次 A 是默认排序的前 300 条，不可比）
core = df[df["batch"].isin(["B", "C"]) & df["city_group"].isin(["一线型", "新一线型"])].copy()
print("核心城市批次记录数:", len(core), " 有职位族:", core["job_family"].notna().sum())

# ---------------------------------------------------------------------------
# 1) 薪酬溢价：对数月薪 ~ 学历 + 经验 + 城市 + 是否兼职类，取残差
# ---------------------------------------------------------------------------
X = pd.get_dummies(core[["education", "experience", "city"]], drop_first=True).astype(float)
X["parttime"] = core["is_parttime"].astype(float).values
X.insert(0, "const", 1.0)
y = np.log(core["salary_mid_k"].values)
beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
core["resid"] = y - X.values @ beta
R2 = 1 - core["resid"].var() / y.var()
print(f"回归 R2 = {R2:.4f}")

BACH_ALL = (core["edu_level"] >= 4).mean()
# 各查询段全体记录的一线型占比：一线型 4 城都在采集组 G1、G2，查询段 S2 恰好被这两组覆盖，所以一线占比偏高
T1_SEG = core.groupby("query_segment")["city_group"].apply(lambda s: (s == "一线型").mean())
print(f"本科及以上占比 {BACH_ALL:.4f}；各查询段的一线型占比 {T1_SEG.round(4).to_dict()}")


def indicators(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """按 key（行业或职位族）汇总各项截面指标。"""
    g = d.groupby(key)
    # 按查询段标准化的区位熵：期望一线占比 = 该对象的记录在各查询段的分布 × 各段全体记录的一线型占比
    seg = d.groupby([key, "query_segment"]).size().unstack(fill_value=0)
    t1_exp = seg.div(seg.sum(1), axis=0) @ T1_SEG
    t1_act = g["city_group"].apply(lambda s: (s == "一线型").mean())
    out = pd.DataFrame({
        "n": g.size(),
        "n_company": g["company"].nunique(),
        "share": g.size() / len(d),
        "resid_mean": g["resid"].mean(),
        "resid_se": g["resid"].std() / np.sqrt(g.size()),
        "bachelor_share": g["edu_level"].apply(lambda s: (s >= 4).mean()),
        "senior_share": g["exp_level"].apply(lambda s: (s >= 4).mean()),
        "edu_any_share": g["edu_any"].mean(),
        "exp_any_share": g["exp_any"].mean(),
        "fresh_share": g["exp_level"].apply(lambda s: s.isin([1, 2]).mean()),
        "tier1_share": t1_act,
        "tier1_expected": t1_exp,
        "tier1_lq": t1_act / t1_exp,
        "bonus_share": g["has_bonus_months"].mean(),
        "salary_median": g["salary_mid_k"].median(),
        "salary_mean": g["salary_mid_k"].mean(),
    })
    out["premium"] = np.exp(out["resid_mean"]) - 1          # 同等学历、经验、城市下多付的比例
    out["premium_t"] = out["resid_mean"] / out["resid_se"]
    return out


ind = indicators(core, "company_type")
ind["n_city"] = df.groupby("company_type")["city"].nunique()      # 覆盖城市数用全部清洗数据（371 城）
fam = core[core["job_family"].notna()]
famind = indicators(fam, "job_family")
famind["family_label"] = pd.Series(family_labels())
fseg = fam.groupby(["job_family", "query_segment"]).size().unstack(fill_value=0)
fseg = fseg.div(fseg.sum(1), axis=0)
famind["main_segment"], famind["main_segment_share"] = fseg.idxmax(axis=1), fseg.max(axis=1)
famind["n_city"] = fam.groupby("job_family")["city"].nunique()
famind.to_csv(os.path.join(PROCESSED_DIR, "A2q5_family_indicators.csv"), encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 2) 行业类型（我们的命名）：溢价方向 × 学历门槛
# ---------------------------------------------------------------------------
TYPES = ["知识密集高薪型", "低门槛高薪型", "平价型", "高学历低薪型", "低门槛低薪型"]
TYPE_COLOR = dict(zip(TYPES, [PALETTE[0], PALETTE[1], NEUTRAL, PALETTE[2], PALETTE[3]]))


def classify(r):
    if r["n"] < MIN_N:
        return "样本不足"
    high_edu = r["bachelor_share"] >= BACH_ALL
    if r["premium"] >= PREM_CUT:
        return TYPES[0] if high_edu else TYPES[1]
    if r["premium"] <= -PREM_CUT:
        return TYPES[3] if high_edu else TYPES[4]
    return TYPES[2]


ind["type"] = ind.apply(classify, axis=1)
ind.index = ind.index.str.replace("type_", "")
ind.sort_values("n_company", ascending=False).to_csv(
    os.path.join(PROCESSED_DIR, "A2q5_industry_indicators.csv"), encoding="utf-8-sig")
cls = ind[ind["type"] != "样本不足"]
print("参与分类的行业数:", len(cls), " 样本不足:", (ind["type"] == "样本不足").sum())
print("有溢价类行业中 |t| 的最小值:", cls.loc[cls["type"] != "平价型", "premium_t"].abs().min().round(2))

rows = []
for t in TYPES:
    s = cls[cls["type"] == t].sort_values("n_company", ascending=False)
    rows.append({
        "type": t, "n_industry": len(s), "record_share": s["share"].sum(),
        "premium_median": s["premium"].median(), "bachelor_median": s["bachelor_share"].median(),
        "senior_median": s["senior_share"].median(), "bonus_median": s["bonus_share"].median(),
        "tier1_lq_median": s["tier1_lq"].median(),
        "top3": "、".join(f"{i}({p:+.0%})" for i, p in s["premium"].head(3).items()),
    })
types = pd.DataFrame(rows).set_index("type")
types.to_csv(os.path.join(PROCESSED_DIR, "A2q5_industry_types.csv"), encoding="utf-8-sig")
print(types.round(3).to_string())
big20 = cls.sort_values("n_company", ascending=False).head(20)
print("企业数前 20 的行业类型:", big20["type"].value_counts().to_dict())
print("规模与溢价的秩相关:", stats.spearmanr(cls["n_company"], cls["premium"]))
print("溢价最高 5 个:", cls["premium"].nlargest(5).round(3).to_dict())
print("溢价最低 5 个:", cls["premium"].nsmallest(5).round(3).to_dict())

# 图 1：规模 × 薪酬溢价，颜色为类型，点大小为 13 薪及以上占比
from matplotlib.lines import Line2D
fig, (ax, lg) = plt.subplots(1, 2, figsize=(10.5, 5.8), constrained_layout=True, gridspec_kw={"width_ratios": [5, 1]})
size = lambda b: 18 + 520 * b
for t in TYPES:
    s = cls[cls["type"] == t]
    ax.scatter(s["n_company"], s["premium"] * 100, s=size(s["bonus_share"]), color=TYPE_COLOR[t],
               alpha=0.85, edgecolor=SURFACE, linewidth=0.8, zorder=3)
for v in (-PREM_CUT * 100, PREM_CUT * 100):
    ax.axhline(v, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
# 只标注各类型中企业数最多的 2 个行业，以及溢价最高/最低的行业
lab = {}
for t in TYPES:
    if t != "平价型":
        for i in cls[cls["type"] == t].nlargest(2, "n_company").index:
            lab[i] = (5, -13) if t == "高学历低薪型" else (5, 5)
for i in list(cls["premium"].nlargest(3).index) + list(cls["premium"].nsmallest(2).index):
    lab.setdefault(i, (5, 5))
for i, off in lab.items():
    r = cls.loc[i]
    ax.annotate(i, (r["n_company"], r["premium"] * 100), xytext=off, textcoords="offset points",
                fontsize=8.5, color=INK_2)
ax.set_xscale("log")
ax.set_xlabel("招聘企业数（个，对数刻度）")
ax.set_ylabel("薪酬溢价（%，同等学历、经验、城市下）")
lg.axis("off")
ht = [Line2D([], [], marker="o", ls="", ms=8, color=TYPE_COLOR[t]) for t in TYPES]
l1 = lg.legend(ht, [f"{t}（{(cls['type'] == t).sum()}）" for t in TYPES], loc="upper left",
               bbox_to_anchor=(-0.25, 1.0), title="行业类型（个数）", alignment="left")
lg.add_artist(l1)
hs = [lg.scatter([], [], s=size(b), color=NEUTRAL, edgecolor=SURFACE) for b in (0.05, 0.2, 0.4)]
lg.legend(hs, ["5%", "20%", "40%"], loc="lower left", bbox_to_anchor=(-0.25, 0.05), title="13 薪及以上占比",
          alignment="left", labelspacing=1.1)
save_fig(fig, "output/figs/A2q5_industry_types.png")

# ---------------------------------------------------------------------------
# 3) H4 检验：批次 B、C 的差异是查询段的差异，不是时间上的变化
#    机制（见 enrich.py 的 query_segment / collect_group）：类目表分三段 S1、S2、S3，
#    采集组 G1：B=S1、C=S2；G2：B=S2、C=S3；G3：B=S3、C=S1，每城只覆盖两段
# ---------------------------------------------------------------------------
# (a) 同一城市内，一个职位是否基本只出现在一个批次
cell = core.groupby(["city", "job_title"])["batch"].agg(n="size", b_share=lambda s: (s == "B").mean())
cell = cell[cell["n"] >= 20]
one_batch = ((cell["b_share"] <= 0.1) | (cell["b_share"] >= 0.9)).mean()   # 九成以上记录落在同一批次
print(f"(城市, 职位) 单元 {len(cell)} 个，九成以上落在同一批次的占 {one_batch:.3f}")

# (b) 6 个“采集组 × 查询段”块的职位族份额向量：同段异组 vs 同组异段 的相关系数
blk = fam.groupby(["collect_group", "query_segment", "job_family"]).size().unstack(fill_value=0)
blk = blk.div(blk.sum(1), axis=0)
C = np.corrcoef(blk.values)
pairs = list(itertools.combinations(range(len(blk)), 2))
same_seg = [C[i, j] for i, j in pairs if blk.index[i][1] == blk.index[j][1]]
same_grp = [C[i, j] for i, j in pairs if blk.index[i][0] == blk.index[j][0]]
print("职位族份额的相关：同段异组", np.round(same_seg, 3), " 同组异段", np.round(same_grp, 3))

# (c) 只属于一个查询段的职位族（该段占比 >= 95%）
single = famind[famind["main_segment_share"] >= 0.95].sort_values(["main_segment", "n"], ascending=[True, False])
print("单段职位族:", single.groupby("main_segment").apply(lambda d: list(d.index), include_groups=False).to_dict())
pd.Series({
    "title_city_cells": len(cell), "title_cells_single_batch_share": one_batch,
    "family_share_corr_same_segment_min": min(same_seg), "family_share_corr_same_segment_max": max(same_seg),
    "family_share_corr_same_group_min": min(same_grp), "family_share_corr_same_group_max": max(same_grp),
    "n_single_segment_families": len(single),
    **{f"tier1_share_{k}": v for k, v in T1_SEG.items()},
}).to_csv(os.path.join(PROCESSED_DIR, "A2q5_h4_check.csv"), header=["value"], encoding="utf-8-sig")

# 图 2：14 城 × 单段职位族的“批次 B 占比”矩阵，城市按采集组排序，职位族按查询段排序
grp = core.groupby("city")["collect_group"].first()
cities = grp.sort_values(kind="stable").index.tolist()
cnt = fam.groupby(["city", "job_family"]).size().unstack(fill_value=0).reindex(index=cities, columns=single.index)
bsh = fam.groupby(["city", "job_family"])["batch"].agg(lambda s: (s == "B").mean()).unstack()
bsh = bsh.reindex(index=cities, columns=single.index).where(cnt >= 10)       # 不足 10 条视为未采到

fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
im = ax.imshow(np.ma.masked_invalid(bsh.values), cmap=div_cmap(), vmin=0, vmax=1, aspect="auto")
for i, j in zip(*np.where(bsh.isna().values)):
    ax.text(j, i, "×", ha="center", va="center", fontsize=9, color=AXIS)
ax.set_xticks(range(bsh.shape[1]), bsh.columns)
ax.set_yticks(range(len(cities)), [f"{c}（{grp[c]}）" for c in cities])
ax.grid(False)
ax.set_xticks(np.arange(-.5, bsh.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, len(cities)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="both", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
# 采集组、查询段之间的分隔线与查询段标注
gb = np.flatnonzero(grp[cities].values[1:] != grp[cities].values[:-1]) + 0.5
sb = np.flatnonzero(single["main_segment"].values[1:] != single["main_segment"].values[:-1]) + 0.5
for v in gb:
    ax.axhline(v, color=INK_2, lw=1.2)
for v in sb:
    ax.axvline(v, color=INK_2, lw=1.2)
edges = np.r_[-0.5, sb, bsh.shape[1] - 0.5]
for k, sname in enumerate(single["main_segment"].unique()):
    ax.text((edges[k] + edges[k + 1]) / 2, -0.75, sname, ha="center", va="bottom", fontsize=9.5, color=INK_2)
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, ticks=[0, 0.5, 1])
cb.ax.set_yticklabels(["0（全在批次 C）", "0.5", "1（全在批次 B）"])
cb.set_label("批次 B 占比（× 为未采到）")
cb.outline.set_visible(False)
ax.set_xlabel("只属于一个查询段（S1、S2、S3）的职位族")
ax.set_title("同一职位族落在哪个批次由采集组决定，另有一组城市完全没采到", pad=24)
save_fig(fig, "output/figs/A2q5_segment_matrix.png")
