# -*- coding: utf-8 -*-
"""成员C · 第二轮 · 数据处理：职位族的恢复验证与职位画像（对应赛题问题 2，依赖假设 H1、H3）。

职位族的恢复算法在 enrich.py（_build_families），本脚本只做验证、画像与类型归纳。
产出表：
  data/processed/C2_family_validation.csv   职位族有效性检验的数字
  data/processed/C2_family_types.csv        51 个职位族的画像指标与类型
  data/processed/C2_type_summary.csv        类型汇总表
  data/processed/C2_title_profile_example.csv 示例职位族内高频职位的画像
产出图：
  output/figs/C2_family_ribbon.png          核心城市大块按行号展开的职位族色带
  output/figs/C2_family_heatmap.png         51 族 × 关键指标热力图
  output/figs/C2_family_examples.png        4 个职位族的画像小多图
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *
from scipy.stats import spearmanr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt = apply_style()
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 80)
df = load_enriched()
N = len(df)
prof = family_profile()
LABEL = prof["family_label"].to_dict()
diag = family_diagnostics()
d = df[df["job_family"].notna()].copy()
d["ls"] = np.log(d["salary_mid_k"])
tax = json.load(open(os.path.join(ROOT, "data", "reference", "platform_taxonomy.json"), encoding="utf-8"))

# ------------------------------------------------------------ 1. 为什么需要职位族
vc = df["job_title"].value_counts()
print(f"职位 {len(vc)} 种；只出现 1 次 {(vc == 1).mean():.1%}；≥30 次 {(vc >= 30).sum()} 种，覆盖 {vc[vc >= 30].sum() / N:.1%}；"
      f"行业 {df['company_type'].nunique()} 个")
print("[职位族诊断]\n", diag.round(4).to_string())
print(f"有职位族的记录 {len(d)}，占清洗后全部 {len(d) / N:.2%}；其中只出现 1 次的职位的记录占 {d['job_title'].map(vc).eq(1).mean():.1%}")
print("平台职位类目：一级", tax["职位类目"]["一级个数"], "二级", tax["职位类目"]["二级个数"])

# ------------------------------------------------------------ 2. 有效性检验
def eta2(y, g):
    """分组对 y 的方差解释比例（组间平方和 / 总平方和）。"""
    m = y.groupby(g).transform("mean")
    return float(1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum())

val = {}
# 2.1 记录级：各分类变量对对数月薪的解释力
for col, name in (("job_family", "职位族"), ("company_type", "行业"), ("city", "城市"),
                  ("education", "学历"), ("experience", "经验")):
    val[f"记录级_对数月薪_{name}({d[col].nunique()}级)"] = eta2(d["ls"], d[col])

# 2.2 职位级：族内职位的画像是否比随机分组更一致（恢复过程只用了行序，没有用薪酬、学历、经验）
tf = pd.read_csv(os.path.join(PROCESSED_DIR, "job_family_titles.csv")).set_index("job_title")["job_family"]
core = df[df["batch"].isin(["B", "C"]) & df["job_title"].isin(tf.index)]
g = core.groupby("job_title")
tp = pd.DataFrame({
    "对数薪酬中位数": np.log(g["salary_mid_k"].median()),
    "本科+占比": g["edu_level"].apply(lambda s: (s >= 4).mean()),
    "学历不限占比": g["edu_any"].mean(),
    "经验不限占比": g["exp_any"].mean(),
    "13薪占比": g["has_bonus_months"].mean(),
})
fam_of = tf.reindex(tp.index)
rng = np.random.default_rng(0)
print(f"\n[职位级检验] 锚点职位 {len(tp)} 个、{fam_of.nunique()} 个族；随机分组=保持各族大小不变打乱 200 次")
for c in tp.columns:
    real = eta2(tp[c], fam_of)
    perm = [eta2(tp[c], pd.Series(rng.permutation(fam_of.values), index=tp.index)) for _ in range(200)]
    val[f"职位级_{c}_职位族"] = real
    val[f"职位级_{c}_随机分组均值"] = float(np.mean(perm))
    val[f"职位级_{c}_随机分组最大"] = float(np.max(perm))
    print(f"  {c}: 职位族解释 {real:.1%}；随机分组 均值 {np.mean(perm):.1%}、最大 {np.max(perm):.1%}")

# 2.3 稳定性：职位族薪酬中位数排序跨城市、跨批次的秩相关
med_city = d.groupby(["job_family", "city"])["salary_mid_k"].median().unstack()
print("职位族×城市 缺失格数:", int(med_city.isna().sum().sum()), " 城市数:", med_city.shape[1])   # 缺失格按成对删除处理
r = med_city.corr(method="spearman").values
iu = np.triu_indices_from(r, 1)
mb = d.groupby(["job_family", "batch"])["salary_mid_k"].median().unstack()
val["薪酬排序_城市两两秩相关中位数"] = float(np.median(r[iu]))
val["薪酬排序_城市两两秩相关最小"] = float(r[iu].min())
val["薪酬排序_查询段两两秩相关中位数"] = float(np.median(d.groupby(["job_family", "query_segment"])["salary_mid_k"].median().unstack().corr(method="spearman").values[np.triu_indices(3, 1)]))
val["薪酬排序_批次B与C秩相关"] = float(spearmanr(mb["B"], mb["C"])[0])
bb = d.assign(b=d["edu_level"] >= 4).groupby(["job_family", "batch"])["b"].mean().unstack()
val["本科+占比排序_批次B与C秩相关"] = float(spearmanr(bb["B"], bb["C"])[0])
val = pd.Series(val, name="value")
val.round(4).to_csv(os.path.join(PROCESSED_DIR, "C2_family_validation.csv"), encoding="utf-8-sig")
print("\n[有效性检验]\n", val.round(4).to_string())

# 2.4 色带图：一线型城市 F047 批次 B 的大块，按行号展开
blk = df[(df["city"] == "F047") & (df["batch"] == "B")].reset_index(drop=True)
top7 = blk["job_family"].value_counts().head(7).index.tolist()
fam_row = blk["job_family"]                                   # 行级归属（多数票）
fam_own = blk["job_title"].map(tf)                            # 锚点职位自身所属的族（非锚点为空）
same = (fam_row == fam_row.shift()) & fam_row.notna() & fam_row.shift().notna()
both = fam_row.notna() & fam_row.shift().notna()
print(f"\n[色带] F047 批次 B：{len(blk)} 行，得到职位族 {fam_row.notna().mean():.1%}，锚点职位记录占 {fam_own.notna().mean():.1%}；"
      f"前 7 个族占 {fam_row.isin(top7).mean():.1%}")
seg_start = np.flatnonzero(np.r_[True, fam_row.fillna("NA").values[1:] != fam_row.fillna("NA").values[:-1]])
seg_len = np.diff(np.r_[seg_start, len(blk)])
seg_fam = fam_row.fillna("NA").values[seg_start]
print(f"  连续段 {len(seg_len)} 个；前 7 个族各自的段数与最长段：",
      {f: (int((seg_fam == f).sum()), int(seg_len[seg_fam == f].max())) for f in top7})
own7 = fam_own[fam_own.isin(top7)]
print(f"  前 7 个族的锚点记录中，自身族与行级归属一致的比例 {(own7 == fam_row[own7.index]).mean():.1%}")

def to_code(s):
    """族 -> 颜色序号：0 无归属，1 其余族，2~8 前 7 个族。"""
    out = np.where(s.isna(), 0, 1)
    for i, f in enumerate(top7):
        out[(s == f).values] = i + 2
    return out

cmap = ListedColormap([SURFACE, NEUTRAL] + PALETTE[:7])
W = 3600                                                       # 每排展示的行数
n_strip = int(np.ceil(len(blk) / W))
fig, axes = plt.subplots(n_strip, 1, figsize=(11, 5.6), constrained_layout=True)
for k, ax in enumerate(axes):
    a, b = k * W, min((k + 1) * W, len(blk))
    img = np.zeros((2, W))
    img[0, :b - a] = to_code(fam_own.iloc[a:b])
    img[1, :b - a] = to_code(fam_row.iloc[a:b])
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=8, interpolation="nearest", extent=(a, a + W, 2, 0))
    ax.axhline(1, color=SURFACE, linewidth=2)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["锚点职位", "行级归属"], fontsize=8.8)
    ax.set_xlim(a, a + W)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[-1].set_xlabel("记录在该城市块内的行号")
handles = [Patch(color=PALETTE[i], label=f) for i, f in enumerate(top7)]
handles += [Patch(color=NEUTRAL, label="其余 44 个职位族")]
fig.legend(handles=handles, loc="outside lower center", ncol=8, fontsize=8.8, handlelength=1.2, columnspacing=1.4)
save_fig(fig, "output/figs/C2_family_ribbon.png")

# ------------------------------------------------------------ 3. 职位族画像指标（数据无技能文本，以“任职门槛画像”替代关键技能）
def entropy(s):
    p = s.value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())

g = d.groupby("job_family")
TIER1_BASE = (d["city_group"] == "一线型").mean()
ind = prof[["family_label", "n", "n_title", "n_company", "salary_median", "salary_p25", "salary_p75",
            "bachelor_share", "edu_any_share", "exp_any_share", "senior_share", "bonus_share",
            "parttime_share", "top_industry", "top_industry_share", "tier1_share"]].copy()
ind["record_share"] = ind["n"] / len(d)
ind["salary_band"] = ind["salary_p75"] / ind["salary_p25"]          # 薪酬带宽：P75/P25
ind["college_share"] = g["edu_level"].apply(lambda s: (s == 3).mean())
ind["industry_entropy"] = g["company_type"].agg(entropy)             # 行业熵（bit），越小越专属
# 城市偏好：职位类目表分三个查询段 S1~S3 采集，每个核心城市只覆盖其中两段（见 enrich.py 的 query_segment），
# 某族在某城没有记录可能只是该段没采到。因此按查询段标准化：
#   期望记录数 E(族, 城) = Σ_段 该族在该段的记录数 × 该段全部记录中该城的占比；区位熵 = 实际 / 期望
core14 = sorted(d["city"].value_counts().loc[lambda s: s > 1000].index)   # 14 个核心城市
t1 = [c for c in core14 if c in TIER1_CITIES]
seg_city = pd.crosstab(d["query_segment"], d["city"], normalize="index")[core14]   # 各段内的城市构成

def city_lq_table(key):
    """按查询段标准化的 14 城区位熵表（行：key 的取值；没有记录或期望不足 5 条的城市为 NaN）与一线型区位熵。"""
    obs = pd.crosstab(key, d.loc[key.index, "city"]).reindex(columns=core14, fill_value=0)
    exp = pd.crosstab(key, d.loc[key.index, "query_segment"]).reindex(columns=seg_city.index, fill_value=0) @ seg_city
    return (obs / exp).where((obs > 0) & (exp >= 5)), obs[t1].sum(axis=1) / exp[t1].sum(axis=1)

city_lq, ind["tier1_lq"] = city_lq_table(d["job_family"])
ind["tier1_lq_raw"] = ind["tier1_share"] / TIER1_BASE               # 未标准化，仅作对照
ind["n_city_covered"] = (pd.crosstab(d["job_family"], d["city"])[core14] > 0).sum(axis=1)   # 有记录的核心城市数
ind["city_lq_max"] = city_lq.max(axis=1)
fam_seg = pd.crosstab(d["job_family"], d["query_segment"], normalize="index")
ind["n_segment"] = (fam_seg >= 0.02).sum(axis=1)                     # 该族出现在几个查询段（占比≥2% 才计）
print(f"\n有职位族记录的总体：薪酬中位数 {d['salary_mid_k'].median()}，本科+ {(d['edu_level'] >= 4).mean():.1%}，"
      f"学历不限 {d['edu_any'].mean():.1%}，经验不限 {d['exp_any'].mean():.1%}，一线型 {TIER1_BASE:.1%}")
print("[采集组×批次×查询段 的记录数]\n", pd.crosstab(d["collect_group"], [d["batch"], d["query_segment"]]).to_string())
print("职位族出现的查询段个数：", ind["n_segment"].value_counts().sort_index().to_dict())
part = ind["n_city_covered"] < 14
print(f"未覆盖全部 14 城的族 {int(part.sum())} 个（覆盖 {ind.loc[part, 'n_city_covered'].min()}~{ind.loc[part, 'n_city_covered'].max()} 城），"
      f"占有职位族记录的 {ind.loc[part, 'n'].sum() / len(d):.1%}：", ind.loc[part, "n_city_covered"].to_dict())
print(f"一线型区位熵（按查询段标准化）范围 {ind['tier1_lq'].min():.2f}~{ind['tier1_lq'].max():.2f}；"
      f"未标准化时 {ind['tier1_lq_raw'].min():.2f}~{ind['tier1_lq_raw'].max():.2f}；两者秩相关 "
      f"{spearmanr(ind['tier1_lq'], ind['tier1_lq_raw'])[0]:.3f}")
print("  标准化前后差异最大的族：\n", ind.assign(diff=(ind["tier1_lq"] - ind["tier1_lq_raw"]).abs()).nlargest(5, "diff")[
    ["n", "n_segment", "n_city_covered", "tier1_lq_raw", "tier1_lq"]].round(3).to_string())
print("  一线型区位熵最高/最低的族：\n", ind.sort_values("tier1_lq")[["family_label", "n", "tier1_lq"]].iloc[[0, 1, 2, -3, -2, -1]].round(3).to_string())
print("[指标范围]\n", ind[["salary_median", "salary_band", "bachelor_share", "edu_any_share", "exp_any_share",
                        "senior_share", "bonus_share", "parttime_share", "top_industry_share",
                        "industry_entropy", "tier1_lq", "city_lq_max"]].describe().loc[["min", "50%", "max"]].round(3).to_string())

# ------------------------------------------------------------ 4. 类型归纳（我们的命名；按顺序判定，先满足者优先）
TYPE_ORDER = ["高学历高薪型", "跨行业·本科为主型", "跨行业·中等门槛型", "跨行业·低门槛型", "行业专属·有门槛型", "行业专属·低门槛型"]
TYPE_RULE = {
    "高学历高薪型": "本科+占比≥50% 且薪酬中位数≥10",
    "行业专属·低门槛型": "Top1 行业占比≥30% 且学历不限占比≥1/3",
    "行业专属·有门槛型": "Top1 行业占比≥30% 且学历不限占比<1/3",
    "跨行业·低门槛型": "Top1 行业占比<30% 且学历不限占比≥1/3",
    "跨行业·本科为主型": "Top1 行业占比<30%、学历不限占比<1/3 且本科+占比≥40%",
    "跨行业·中等门槛型": "Top1 行业占比<30%、学历不限占比<1/3 且本科+占比<40%",
}

def family_type(r):
    if r["bachelor_share"] >= 0.5 and r["salary_median"] >= 10:
        return "高学历高薪型"
    if r["top_industry_share"] >= 0.30:
        return "行业专属·低门槛型" if r["edu_any_share"] >= 1 / 3 else "行业专属·有门槛型"
    if r["edu_any_share"] >= 1 / 3:
        return "跨行业·低门槛型"
    return "跨行业·本科为主型" if r["bachelor_share"] >= 0.40 else "跨行业·中等门槛型"

ind["type"] = ind.apply(family_type, axis=1)
ind.round(4).to_csv(os.path.join(PROCESSED_DIR, "C2_family_types.csv"), encoding="utf-8-sig", index_label="job_family")
d["type"] = d["job_family"].map(ind["type"])
tg = d.groupby("type")
tsum = pd.DataFrame({
    "rule": pd.Series(TYPE_RULE), "n_family": ind["type"].value_counts(), "record_share": tg.size() / len(d),
    "salary_median": tg["salary_mid_k"].median(), "bachelor_share": tg["edu_level"].apply(lambda s: (s >= 4).mean()),
    "edu_any_share": tg["edu_any"].mean(), "exp_any_share": tg["exp_any"].mean(),
    "senior_share": tg["exp_level"].apply(lambda s: (s >= 4).mean()), "bonus_share": tg["has_bonus_months"].mean(),
    "tier1_share": tg["city_group"].apply(lambda s: (s == "一线型").mean()),
    "families": ind.groupby("type").apply(lambda x: " ".join(x.index)),
}).reindex(TYPE_ORDER)
tsum.round(4).to_csv(os.path.join(PROCESSED_DIR, "C2_type_summary.csv"), encoding="utf-8-sig", index_label="type")
print("\n[类型汇总]\n", tsum.drop(columns="rule").round(3).to_string())
print(f"类型（6 级）对对数月薪的解释力 {eta2(d['ls'], d['type']):.1%}")
near = ind[(ind["top_industry_share"].between(0.28, 0.32)) | (ind["edu_any_share"].between(0.31, 0.355))
           | (ind["bachelor_share"].between(0.38, 0.42))]
print("[靠近阈值的族]\n", near[["top_industry_share", "edu_any_share", "bachelor_share", "salary_median", "type"]].round(3).to_string())

# ------------------------------------------------------------ 热力图：51 族 × 10 项指标（按列标准化）
COLS = [("salary_median", "薪酬\n中位数"), ("salary_band", "薪酬\n带宽"), ("bachelor_share", "本科+\n占比"),
        ("edu_any_share", "学历\n不限"), ("exp_any_share", "经验\n不限"), ("senior_share", "3年+\n经验"),
        ("bonus_share", "13薪\n占比"), ("parttime_share", "兼职类\n占比"), ("top_industry_share", "Top1\n行业"),
        ("tier1_lq", "一线型\n区位熵")]
keys = [c for c, _ in COLS]
z = (ind[keys] - ind[keys].mean()) / ind[keys].std()
ind["_t"] = ind["type"].map({t: i for i, t in enumerate(TYPE_ORDER)})
order = ind.sort_values(["_t", "salary_median", "n"], ascending=[True, False, False]).index.tolist()
half = 26
type_cmap = ListedColormap(PALETTE[:len(TYPE_ORDER)])
fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), constrained_layout=True)
for ax, rows in zip(axes, (order[:half], order[half:])):
    im = ax.imshow(z.loc[rows].values, cmap=div_cmap(), vmin=-2.5, vmax=2.5, aspect="auto",
                   extent=(0, len(keys), len(rows), 0))
    ax.imshow(ind.loc[rows, ["_t"]].values, cmap=type_cmap, vmin=-0.5, vmax=len(TYPE_ORDER) - 0.5, aspect="auto",
              extent=(-0.75, -0.2, len(rows), 0))
    ax.set_xlim(-0.75, len(keys)); ax.set_ylim(half, 0)
    for x in range(1, len(keys)):
        ax.axvline(x, color=SURFACE, linewidth=1.2)
    for y in range(1, len(rows)):
        ax.axhline(y, color=SURFACE, linewidth=1.2)
    t = ind.loc[rows, "_t"].values
    for y in np.flatnonzero(t[1:] != t[:-1]) + 1:               # 类型之间的分隔
        ax.axhline(y, color=INK_2, linewidth=1.1, xmin=0)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=8.6)
    ax.set_xticks(np.arange(len(keys)) + 0.5)
    ax.set_xticklabels([n for _, n in COLS], fontsize=7.8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.32, aspect=30, pad=0.02, anchor=(0.97, 0.5))
cb.set_label("指标按列标准化后的 z 值", fontsize=9)
cb.outline.set_visible(False)
fig.legend(handles=[Patch(color=PALETTE[i], label=t) for i, t in enumerate(TYPE_ORDER)], loc="outside lower left",
           ncol=3, fontsize=8.8, handlelength=1.2, columnspacing=1.4, title="职位族类型", title_fontsize=8.8,
           alignment="left")
save_fig(fig, "output/figs/C2_family_heatmap.png")
print("[族间指标 Spearman]\n", ind[keys].corr(method="spearman").round(2).to_string())

# ------------------------------------------------------------ 5. 画像示例：4 个差异鲜明的职位族
EX = ["J03", "J19", "J09", "J38"]
print("\n[示例族]\n", ind.loc[EX, ["family_label", "type", "n", "salary_median", "salary_band", "bachelor_share", "college_share",
                               "edu_any_share", "exp_any_share", "senior_share", "top_industry_share", "tier1_lq",
                               "n_segment", "n_city_covered", "city_lq_max", "bonus_share"]].round(3).T.to_string())
EDU_SHORT = ["不限", "初中", "中专\n高中", "大专", "本科", "硕士", "博士"]
EXP_SHORT = ["不限", "在校\n应届", "<1年", "1-3", "3-5", "5-10", "10+"]
city_order = [c for c in TIER1_CITIES] + sorted(c for c in core14 if c not in TIER1_CITIES)
bins = np.logspace(np.log10(2), np.log10(60), 30)
fig, axes = plt.subplots(len(EX), 4, figsize=(11, 6.5), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1, 1.05, 1.5]})
for i, f in enumerate(EX):
    x = d[d["job_family"] == f]
    e = x["edu_level"].value_counts(normalize=True).reindex(range(7), fill_value=0) * 100
    p = x["exp_level"].value_counts(normalize=True).reindex(range(7), fill_value=0) * 100
    print(f, "学历构成%", e.round(1).tolist(), "经验构成%", p.round(1).tolist(),
          "薪酬 P25/P50/P75", x["salary_mid_k"].quantile([.25, .5, .75]).tolist())
    ax = axes[i, 0]
    ax.bar(range(7), e.values, color=[NEUTRAL] + [PALETTE[0]] * 6, width=0.72)
    ax.set_ylim(0, 80); ax.set_yticks([0, 40, 80])
    ax.set_ylabel(f"{f}\n占比（%）", fontsize=9.5)
    ax = axes[i, 1]
    ax.bar(range(7), p.values, color=[NEUTRAL] + [PALETTE[0]] * 6, width=0.72)
    ax.set_ylim(0, 80); ax.set_yticks([0, 40, 80]); ax.tick_params(labelleft=False)
    ax = axes[i, 2]
    h, _ = np.histogram(x["salary_mid_k"], bins=bins)
    ax.bar(bins[:-1], h / h.sum() * 100, width=np.diff(bins) * 0.9, align="edge", color=PALETTE[0])
    ax.axvline(x["salary_mid_k"].median(), color=INK, linewidth=1.2)
    ax.set_xscale("log"); ax.set_xlim(2, 60); ax.set_ylim(0, 24); ax.set_yticks([0, 10, 20])
    ax.set_xticks([3, 5, 10, 20, 40]); ax.set_xticklabels(["3", "5", "10", "20", "40"]); ax.minorticks_off()
    ax = axes[i, 3]
    lq = city_lq.loc[f, city_order].fillna(1)                # 未采到的城市不画条
    ax.bar(range(14), lq.values - 1, bottom=1, width=0.72,
           color=[HIGHLIGHT if c in TIER1_CITIES else NEUTRAL for c in city_order])
    ax.axhline(1, color=INK_2, linewidth=0.9)
    ax.set_ylim(0, 2.9); ax.set_yticks([0, 1, 2])
    ax.set_xticks(range(14))
    for j, ax in enumerate(axes[i]):
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", labelsize=8.3)
        if i < len(EX) - 1:
            ax.tick_params(labelbottom=False)
for ax, labs in ((axes[-1, 0], EDU_SHORT), (axes[-1, 1], EXP_SHORT)):
    ax.set_xticks(range(7)); ax.set_xticklabels(labs, fontsize=8)
axes[-1, 3].set_xticklabels(city_order, fontsize=7.6, rotation=90)
axes[-1, 0].set_xlabel("学历要求构成（%）")
axes[-1, 1].set_xlabel("经验要求构成（%）")
axes[-1, 2].set_xlabel("月薪中值（千元/月）的分布（%）")
axes[-1, 3].set_xlabel("14 个核心城市的区位熵")
save_fig(fig, "output/figs/C2_family_examples.png")

# 族内高频职位的画像表（以 J03 为例）
F0 = "J03"
x = d[d["job_family"] == F0]
gt = x.groupby("job_title")
tt = pd.DataFrame({
    "n": gt.size(), "n_company": gt["company"].nunique(), "salary_median": gt["salary_mid_k"].median(),
    "edu_mode": gt["edu_label"].agg(lambda s: s.value_counts().index[0]),
    "edu_mode_share": gt["edu_label"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
    "exp_mode": gt["exp_label"].agg(lambda s: s.value_counts().index[0]),
    "exp_mode_share": gt["exp_label"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
}).sort_values("n", ascending=False).head(5)
tt["tier1_lq"] = city_lq_table(x["job_title"])[1].reindex(tt.index)   # 一线型区位熵（按查询段标准化）
tt["segments"] = pd.crosstab(x["job_title"], x["query_segment"]).reindex(tt.index).apply(
    lambda r: "/".join(r.index[r >= 0.02 * r.sum()]), axis=1)
tt.insert(0, "title8", tt.index.str[:8])
tt.round(4).to_csv(os.path.join(PROCESSED_DIR, "C2_title_profile_example.csv"), encoding="utf-8-sig", index_label="job_title")
print(f"\n[{F0} 族内高频职位]（族内职位 {x['job_title'].nunique()} 种，一线型区位熵 {ind.loc[F0, 'tier1_lq']:.2f}）\n",
      tt.round(3).to_string(index=False))
