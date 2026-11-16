# -*- coding: utf-8 -*-
"""成员D · 第二轮 · 全国口径：批次 A 的可比性、城市画像表、全国城市分型与异常城市 H610（问题 4）。

口径：load_enriched() 的批次 A（每城一次查询、约 300 条），剔除 6 个采集条件异常的块。
先运行 D2_core.py（本脚本会把它输出的核心城市偏好并入画像表）。
产出：output/figs/D2_batchA_blocks.png、D2_nation_types.png
      data/processed/D2_city_profile.csv（两套口径）、D2_nation_types.csv、D2_batchA_vs_core.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D2_lib import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt = apply_style()
df = load_d2()
T1 = TIER1_CITIES
LAB = family_labels()
A = df[df["batch"] == "A"]
BC = df[df["seg"].notna()]
HI2 = ["type_BLfSmG", "type_CbBLwi"]

# ---------------------------------------------------------------------------
# 1. 批次 A 的块：按文件顺序看“学历不限”占比，识别采集条件异常的块
# ---------------------------------------------------------------------------
g = A.groupby("a_block")
blk = pd.DataFrame({"city": g["city"].first(), "n": g.size(), "edu_any": g["edu_any"].mean(),
                    "salary_median": g["salary_mid_k"].median(), "bachelor": g["edu_level"].apply(lambda s: (s >= 4).mean()),
                    "hi2": g["company_type"].apply(lambda s: s.isin(HI2).mean()), "anomaly": g["a_anomaly"].first()})
blk = blk[blk["n"] >= 100].reset_index(drop=True)
blk["order"] = np.arange(1, len(blk) + 1)
bad = blk[blk["anomaly"]]
print("批次 A 不少于 100 条的块数", len(blk), "异常块:\n", bad.round(3).to_string())
print("其余块 学历不限占比 最小值", round(blk[~blk.anomaly]["edu_any"].min(), 3), "中位数", round(blk[~blk.anomaly]["edu_any"].median(), 3),
      "| BLfSmG+CbBLwi 占比中位数", round(blk[~blk.anomaly]["hi2"].median(), 3))

fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
ok = blk[~blk.anomaly]
ax.scatter(ok["order"], ok["edu_any"] * 100, s=14, color=NEUTRAL, label=f"其余 {len(ok)} 个块", zorder=2)
ax.scatter(bad["order"], bad["edu_any"] * 100, s=42, color=HIGHLIGHT, label=f"异常块（{len(bad)} 个，学历不限占比 <15%）", zorder=3)
ax.axhline(ANOMALY_EDU_ANY * 100, color=MUTED, lw=0.9, ls=(0, (4, 3)))
head = bad[bad["order"] <= 10]
ax.annotate("文件最前面的 5 个块：\n" + "、".join(head["city"]) + "\n薪酬中位数 " + "、".join(f"{v:g}" for v in head["salary_median"]) + " 千元/月",
            xy=(head["order"].max() + 1, head["edu_any"].mean() * 100), xytext=(32, 20.5), fontsize=9.5, color=INK_2, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
h = bad[bad["city"] == "H610"].iloc[0]
ax.annotate(f"H610（第 {int(h.order)} 块）\n薪酬中位数 {h.salary_median:g} 千元/月", xy=(h.order - 1, h.edu_any * 100), xytext=(h.order - 14, 20),
            fontsize=9.5, color=INK_2, ha="right", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlabel("批次 A 中城市块在原始文件里的先后次序")
ax.set_ylabel("块内“学历不限”岗位占比（%）")
ax.set_ylim(0, 70)
ax.set_title("批次 A 有 6 个块的采集条件与其余不同：H610 是其中之一")
ax.legend(loc="upper right", ncol=2)
save_fig(fig, "output/figs/D2_batchA_blocks.png")

# ---------------------------------------------------------------------------
# 2. 核心城市：批次 A 画像与 B+C 整体的差别
# ---------------------------------------------------------------------------
PA_core = city_metrics(A[A["city"].isin(CORE) & (A.groupby("a_block")["city"].transform("size") >= 100)])
PB_core = city_metrics(BC)
ciA = pd.crosstab(A["city"], A["company_type"])
ciB = pd.crosstab(BC["city"], BC["company_type"]).reindex(columns=ciA.columns, fill_value=0)
rng = np.random.default_rng(0)


def cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


rows = []
for c in CORE:
    a_, b_ = ciA.loc[c].values / ciA.loc[c].sum(), ciB.loc[c].values / ciB.loc[c].sum()
    base = np.mean([cos(rng.multinomial(int(ciA.loc[c].sum()), b_) / ciA.loc[c].sum(), b_) for _ in range(200)])
    rows.append({"city": c, "a_anomaly": c in set(bad["city"]),
                 "median_A": PA_core.loc[c, "salary_median"], "median_BC": PB_core.loc[c, "salary_median"],
                 "bachelor_A": PA_core.loc[c, "bachelor_share"], "bachelor_BC": PB_core.loc[c, "bachelor_share"],
                 "edu_any_A": PA_core.loc[c, "edu_any_share"], "edu_any_BC": PB_core.loc[c, "edu_any_share"],
                 "industry_cos": cos(a_, b_), "industry_cos_sampling": base})
cmp_ = pd.DataFrame(rows).set_index("city")
cmp_.round(4).to_csv(os.path.join(PROCESSED_DIR, "D2_batchA_vs_core.csv"), encoding="utf-8-sig")
print(cmp_.round(3).to_string())
for flag, nm in ((True, "异常块所在的 4 个核心城市"), (False, "其余 10 个核心城市")):
    s = cmp_[cmp_.a_anomaly == flag]
    print(f"[{nm}] 薪酬中位数 A−BC: {(s.median_A - s.median_BC).min():+.1f}~{(s.median_A - s.median_BC).max():+.1f}；"
          f"本科+ A−BC: {(s.bachelor_A - s.bachelor_BC).min():+.3f}~{(s.bachelor_A - s.bachelor_BC).max():+.3f}；"
          f"学历不限 A−BC: {(s.edu_any_A - s.edu_any_BC).min():+.3f}~{(s.edu_any_A - s.edu_any_BC).max():+.3f}；"
          f"行业余弦 {s.industry_cos.min():.3f}~{s.industry_cos.max():.3f}（抽样基准 {s.industry_cos_sampling.mean():.3f}）")

# ---------------------------------------------------------------------------
# 3. 全国口径的城市画像（剔除异常块）
# ---------------------------------------------------------------------------
drop_city = sorted(set(bad["city"]) - {"K445"})                 # K445 另有一个正常块，保留
N = A[~A["a_anomaly"] & ~A["city"].isin(drop_city)]
P = city_metrics(N)
P["city_group"] = N.groupby("city")["city_group"].first()
big = P[P["n"] >= MIN_CITY_N].copy()
N2 = N[N["city"].isin(big.index)]
print("全国口径城市数", len(P), "其中 ≥200 条", len(big), "<200 条", int((P.n < MIN_CITY_N).sum()), "| 剔除的城市", drop_city)
print("参与分型城市的原分组:", big["city_group"].value_counts().to_dict(), "每城条数中位数", big["n"].median())
ci = pd.crosstab(N2["city"], N2["company_type"])
sh = ci.div(ci.sum(1), axis=0)
nat = ci.sum() / ci.values.sum()
lq = sh / nat

# 3.1 抽样噪声与结构强弱
n_med = big["n"].median()
for p_ in (0.10, 0.05):
    print(f"占比 {p_:.0%} 的指标在 n={n_med:.0f} 时标准误 {np.sqrt(p_ * (1 - p_) / n_med) * 100:.2f} 个百分点")
for col, nm in (("bachelor_share", "本科+"), ("edu_any_share", "学历不限"), ("exp_any_share", "经验不限"), ("exp3plus_share", "3年以上"), ("bonus_share", "13薪")):
    p_ = (big[col] * big["n"]).sum() / big["n"].sum()
    se = np.sqrt(p_ * (1 - p_) / n_med)
    print(f"  {nm}: 合并占比 {p_:.3f} 标准误 {se:.4f} 城市间标准差 {big[col].std():.4f} 信度≈{1 - se ** 2 / big[col].var():.2f}")
ls_se = N2.groupby("city")["salary_mid_k"].apply(lambda s: np.log(s).std() / np.sqrt(len(s))).median()
lgm = np.log(big["salary_gmean"])
print(f"  对数薪酬均值: 标准误 {ls_se:.4f} 城市间标准差 {lgm.std():.4f} 信度≈{1 - ls_se ** 2 / lgm.var():.2f}")
rng = np.random.default_rng(1)
med_se = np.median([np.std([np.median(rng.choice(v, len(v))) for _ in range(200)])
                    for v in (N2.loc[N2.city == c, "salary_mid_k"].values for c in rng.choice(big.index, 40, replace=False))])
print("  薪酬中位数的自助标准误（40 城中位数）", round(float(med_se), 3), "千元/月")

feat = pd.DataFrame({"对数薪酬": lgm, "本科+": big["bachelor_share"], "学历不限": big["edu_any_share"], "经验不限": big["exp_any_share"],
                     "3年以上": big["exp3plus_share"], "13薪": big["bonus_share"]})
Zs = ((feat - feat.mean()) / feat.std()).values
ev = np.linalg.eigvalsh(np.corrcoef(Zs.T))[::-1]
print("6 个薪酬/门槛指标的主成分解释比:", (ev / ev.sum()).round(3).tolist())
print("指标相关矩阵:\n", feat.corr().round(2).to_string())
for k in range(2, 7):
    lab_ = KMeans(k, n_init=20, random_state=0).fit_predict(Zs)
    print(f"  KMeans k={k} 轮廓系数 {silhouette_score(Zs, lab_):.3f} 簇大小 {sorted(np.bincount(lab_).tolist(), reverse=True)}")

# 3.2 复核“两种行业底色”（lOdYUb 主导 / uTAWZv 主导）
top1 = sh.idxmax(1).map(short)
print("第一大行业频数:", top1.value_counts().head(6).to_dict())
lo, ut = sh["type_lOdYUb"], sh["type_uTAWZv"]
d = lo - ut
se_d = np.sqrt((lo + ut - d ** 2) / big["n"])
print(f"全国占比 lOdYUb {nat['type_lOdYUb']:.3f} uTAWZv {nat['type_uTAWZv']:.3f}；两者份额相关 {np.corrcoef(lo, ut)[0, 1]:.3f}")
print(f"份额差（lOdYUb − uTAWZv）: 均值 {d.mean():+.3f} 标准差 {d.std():.3f}；与 0 无显著差异（|差|<1.96 标准误）的城市占 {(d.abs() < 1.96 * se_d).mean():.1%}；"
      f"lOdYUb 显著更高 {int((d > 1.96 * se_d).sum())} 城，uTAWZv 显著更高 {int((d < -1.96 * se_d).sum())} 城")
print("份额差直方图（每 2.5 个百分点）:", np.histogram(d, bins=np.arange(-0.175, 0.126, 0.025))[0].tolist())
sim_top = []
for _ in range(50):
    S = np.vstack([rng.multinomial(int(k), nat.values) for k in big["n"]])
    w = nat.index[S.argmax(1)]
    sim_top.append([(w == "type_lOdYUb").sum(), (w == "type_uTAWZv").sum()])
print("若所有城市行业结构完全相同，仅抽样波动下第一大行业为 lOdYUb / uTAWZv 的城市数:", np.mean(sim_top, 0).round(1).tolist())
for t in ("type_uTAWZv", "type_lOdYUb", "type_BLfSmG"):
    p_ = nat[t]
    print(f"  {short(t)} 份额: 城市间标准差 {sh[t].std():.4f} 抽样标准误 {np.sqrt(p_ * (1 - p_) / n_med):.4f} 与对数薪酬相关 {np.corrcoef(sh[t], lgm)[0, 1]:+.2f} 与学历不限占比相关 {np.corrcoef(sh[t], big.edu_any_share)[0, 1]:+.2f}")

# 3.3 规则分型（我们的命名）：薪酬层级 × 是否有特色行业
GM_LOW, GM_HIGH, SPEC_SHARE, SPEC_LQ = 5.1, 6.0, 0.10, 2.0
TYPES = ["较高薪型", "特色行业型", "中薪型", "低薪型"]


def assign(gm, spec):
    return np.where(gm >= GM_HIGH, "较高薪型", np.where(spec, "特色行业型", np.where(gm >= GM_LOW, "中薪型", "低薪型")))


spec_m = (sh >= SPEC_SHARE) & (lq >= SPEC_LQ)
big["spec_industry"] = [short(lq.loc[c][spec_m.loc[c]].idxmax()) if spec_m.loc[c].any() else "" for c in big.index]
big["type"] = assign(big["salary_gmean"].values, spec_m.any(axis=1).values)
print("各类型城市数:", big["type"].value_counts().reindex(TYPES).to_dict())
print("特色行业型的特色行业:", big.loc[big.type == "特色行业型", "spec_industry"].value_counts().to_dict())
print("较高薪型中同时满足特色行业条件的城市数:", int(((big.type == "较高薪型") & (big.spec_industry != "")).sum()))
print("规则分型在 6 个指标空间的轮廓系数:", round(silhouette_score(Zs, big["type"].values), 3))

# 分型的抽样稳定性：每城记录自助抽样 200 次后重新套用规则
cc = pd.Categorical(N2["city"], categories=big.index).codes.astype(np.int64)
ic = pd.Categorical(N2["company_type"]).codes.astype(np.int64)
K = ic.max() + 1
lsal = np.log(N2["salary_mid_k"].values)
order = np.argsort(cc, kind="stable")
bounds = np.r_[0, np.cumsum(np.bincount(cc))]
keep = np.zeros(len(big))
B = 200
for _ in range(B):
    idx = np.concatenate([order[rng.integers(bounds[i], bounds[i + 1], bounds[i + 1] - bounds[i])] for i in range(len(big))])
    n_ = np.bincount(cc[idx], minlength=len(big))
    gm_ = np.exp(np.bincount(cc[idx], weights=lsal[idx], minlength=len(big)) / n_)
    M = np.bincount(cc[idx] * K + ic[idx], minlength=len(big) * K).reshape(len(big), K)
    s_ = M / M.sum(1, keepdims=True)
    nt_ = M.sum(0) / M.sum()
    sp_ = ((s_ >= SPEC_SHARE) & (s_ >= SPEC_LQ * nt_)).any(1)
    keep += assign(gm_, sp_) == big["type"].values
big["type_stability"] = keep / B
print("自助抽样中类型保持不变的比例：全体均值", round(big["type_stability"].mean(), 3),
      "各类型:", big.groupby("type")["type_stability"].mean().reindex(TYPES).round(3).to_dict(),
      "| 稳定度 ≥0.8 的城市占", f"{(big.type_stability >= 0.8).mean():.1%}")

# 各类型的画像：合并样本后计算偏好行业与职位族（职位族由锚点职位映射，仅覆盖部分记录）
N2 = N2.assign(type=N2["city"].map(big["type"]))
tf = pd.read_csv(os.path.join(PROCESSED_DIR, "job_family_titles.csv")).set_index("job_title")["job_family"]
fam = N2["job_title"].map(tf)
print("全国口径记录中可由锚点职位映射到职位族的比例", f"{fam.notna().mean():.1%}")
ct = pd.crosstab(N2["type"], N2["company_type"])
tlq = ct.div(ct.sum(1), axis=0) / nat
cf = pd.crosstab(N2["type"][fam.notna()], fam.dropna())
flq = cf.div(cf.sum(1), axis=0) / (cf.sum() / cf.values.sum())
rows = []
for t in TYPES:
    s = N2[N2["type"] == t]
    ti = tlq.loc[t][nat >= 0.015].sort_values(ascending=False)
    tfm = flq.loc[t][cf.sum() >= 800].sort_values(ascending=False)
    rows.append({"类型": t, "城市数": int((big.type == t).sum()), "薪酬中位数": s["salary_mid_k"].median(),
                 "薪酬几何均值": round(float(np.exp(np.log(s["salary_mid_k"]).mean())), 2),
                 "本科+占比": round(float((s["edu_level"] >= 4).mean()), 4), "学历不限占比": round(float(s["edu_any"].mean()), 4),
                 "经验不限占比": round(float(s["exp_any"].mean()), 4), "13薪占比": round(float(s["has_bonus_months"].mean()), 4),
                 "偏好行业": "、".join(f"{short(k)}({v:.2f})" for k, v in ti.head(3).items()),
                 "少见行业": "、".join(f"{short(k)}({v:.2f})" for k, v in ti.tail(2).items()),
                 "偏好职位族": "、".join(f"{LAB[k]}({v:.2f})" for k, v in tfm.head(2).items()),
                 "少见职位族": "、".join(f"{LAB[k]}({v:.2f})" for k, v in tfm.tail(2).items()),
                 "类型稳定度": round(float(big.loc[big.type == t, "type_stability"].mean()), 3),
                 "代表城市": "、".join(big[big.type == t].sort_values("type_stability", ascending=False).index[:3])})
types_tab = pd.DataFrame(rows)
types_tab.to_csv(os.path.join(PROCESSED_DIR, "D2_nation_types.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 90)
print(types_tab.T.to_string())
print("各类型 uTAWZv / lOdYUb / BLfSmG 份额(%):\n", (ct.div(ct.sum(1), axis=0)[["type_uTAWZv", "type_lOdYUb", "type_BLfSmG"]] * 100).round(2).reindex(TYPES).to_string())
print("核心城市（批次 A 正常块）的类型:", big.loc[[c for c in CORE if c in big.index], "type"].to_dict())
print("较高薪型的原分组:", big[big.type == "较高薪型"]["city_group"].value_counts().to_dict(),
      "| 薪酬几何均值最高的 6 城:", big.sort_values("salary_gmean", ascending=False).head(6)["salary_gmean"].round(2).to_dict())

# 3.4 抽象布局：6 个薪酬/门槛指标 + 行业份额前 3 个主成分 → PCA 二维
top20 = nat.sort_values(ascending=False).index[:20]
Xi = np.sqrt(sh[top20].values); Xi = Xi - Xi.mean(0)
U, s_, Vt = np.linalg.svd(Xi, full_matrices=False)
print("行业份额（前 20 行业合计占 %.1f%%）前 3 个主成分解释比:" % (nat[top20].sum() * 100), (s_[:3] ** 2 / (s_ ** 2).sum()).round(3).tolist())
ind_pc = U[:, :3] * s_[:3]
Zall = np.column_stack([Zs, (ind_pc - ind_pc.mean(0)) / ind_pc.std(0, ddof=1)])
Uc, sc, Vc = np.linalg.svd(Zall - Zall.mean(0), full_matrices=False)
xy = Uc[:, :2] * sc[:2]
evr = sc ** 2 / (sc ** 2).sum()
if np.corrcoef(xy[:, 0], lgm)[0, 1] < 0:
    xy[:, 0] *= -1
print("二维布局解释方差:", evr[:2].round(3).tolist(), "PC1 与对数薪酬相关", round(np.corrcoef(xy[:, 0], lgm)[0, 1], 3),
      "PC2 载荷:", dict(zip(list(feat.columns) + ["行业PC1", "行业PC2", "行业PC3"], Vc[1].round(2))))
big["x"], big["y"] = xy[:, 0], xy[:, 1]

fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
TYPE_COLOR = dict(zip(TYPES, PALETTE[:4]))
for t in TYPES[::-1]:
    s = big[big.type == t]
    ax.scatter(s["x"], s["y"], s=24, color=TYPE_COLOR[t], alpha=0.85, edgecolors=SURFACE, linewidths=0.5, label=f"{t}（{len(s)} 个）", zorder=3)
cores = big[big.index.isin(CORE)]
ax.scatter(cores["x"], cores["y"], s=70, facecolors="none", edgecolors=INK, linewidths=0.9, zorder=4, label="核心城市")
handles, labels = ax.get_legend_handles_labels()
o = [labels.index(next(l for l in labels if l.startswith(t))) for t in TYPES] + [len(labels) - 1]
ax.legend([handles[i] for i in o], [labels[i] for i in o], loc="lower right", fontsize=9)
ax.set_xlabel(f"主成分 1（解释 {evr[0]:.1%} 方差）")
ax.set_ylabel(f"主成分 2（解释 {evr[1]:.1%} 方差）")
save_fig(fig, "output/figs/D2_nation_types.png")

# ---------------------------------------------------------------------------
# 4. 异常城市 H610
# ---------------------------------------------------------------------------
hb = A[A["a_anomaly"]]
comp_t1 = {c: set(df.loc[df["city"].isin([t for t in T1 if t != c]), "company"]) for c in set(hb["city"]) | {"_"}}
t1_all = set(df.loc[df["city"].isin(T1), "company"])


def overlap(d, city):
    u = pd.Series(d["company"].unique())
    return u.isin(comp_t1.get(city, t1_all)).mean()


ov_bad = {c: round(overlap(d, c), 3) for c, d in hb.groupby("city")}
ov_norm = N2[N2["city"].isin(big.index[big.city_group == "普通城市"])].groupby("city").apply(lambda d: overlap(d, "_"))
print("企业同时出现在（其他）一线型城市的比例：异常块", ov_bad, "| 普通城市中位数", round(ov_norm.median(), 3), "最大值", round(ov_norm.max(), 3))
print("异常块内部的薪酬中位数:", bad.set_index("city")["salary_median"].to_dict(), "本科+占比:", bad.set_index("city")["bachelor"].round(3).to_dict(),
      "BLfSmG+CbBLwi 占比:", bad.set_index("city")["hi2"].round(3).to_dict())
hh = hb[hb["city"] == "H610"]
print("H610：记录", len(hh), "企业", hh["company"].nunique(), "前三大行业", hh["company_type"].map(short).value_counts(normalize=True).head(3).round(3).to_dict(),
      "13薪占比", round(hh["has_bonus_months"].mean(), 3), "经验 3 年以上占比", round((hh["exp_level"] >= 4).mean(), 3))

# ---------------------------------------------------------------------------
# 5. 城市画像表（两套口径）
# ---------------------------------------------------------------------------
def pref_industry(sh_row, lq_row, min_share=0.03):
    s = lq_row[sh_row >= min_share].sort_values(ascending=False).head(3)
    return "；".join(f"{short(k)}({v:.2f},{sh_row[k]:.1%})" for k, v in s.items())


ci_all = pd.crosstab(N["city"], N["company_type"])
sh_all = ci_all.div(ci_all.sum(1), axis=0)
lq_all = sh_all / (ci_all.sum() / ci_all.values.sum())
nat_prof = P.copy()
nat_prof["pref_industry"] = [pref_industry(sh_all.loc[c], lq_all.loc[c]) for c in nat_prof.index]
nat_prof["region_type"] = big["type"].reindex(nat_prof.index)
nat_prof["type_stability"] = big["type_stability"].reindex(nat_prof.index)
nat_prof["note"] = np.where(nat_prof["n"] < MIN_CITY_N, "小样本，只展示不参与分型", "")
an = city_metrics(hb[hb["city"].isin(drop_city)])
an["city_group"] = hb.groupby("city")["city_group"].first()
ci_an = pd.crosstab(hb["city"], hb["company_type"]).reindex(columns=ci_all.columns, fill_value=0).loc[an.index]
sh_an = ci_an.div(ci_an.sum(1), axis=0)
an["pref_industry"] = [pref_industry(sh_an.loc[c], sh_an.loc[c] / (ci_all.sum() / ci_all.values.sum())) for c in an.index]
an["note"] = "批次 A 的块采集条件异常，不参与分型"
nat_prof = pd.concat([nat_prof, an])
nat_prof.insert(0, "caliber", "全国口径（批次A）")
core_prof = city_metrics(BC).loc[CORE]
core_prof["city_group"] = BC.groupby("city")["city_group"].first()
pref = pd.read_csv(os.path.join(PROCESSED_DIR, "D2_core_pref.csv")).set_index("city")
core_prof = core_prof.join(pref)
core_prof["note"] = "偏好项为同口径区位熵（见 D2_core.py）"
core_prof.insert(0, "caliber", "核心城市口径（批次B+C）")
prof = pd.concat([nat_prof, core_prof]).rename_axis("city").reset_index()
num = prof.select_dtypes("number").columns
prof[num] = prof[num].round(4)
prof.to_csv(os.path.join(PROCESSED_DIR, "D2_city_profile.csv"), index=False, encoding="utf-8-sig")
print("画像表行数:", prof["caliber"].value_counts().to_dict(), "列:", prof.columns.tolist())
print(core_prof[["salary_median", "salary_p25", "salary_p75", "bachelor_share", "edu_any_share", "exp_any_share", "exp3plus_share", "bonus_share", "parttime_share", "n_company", "industry_hhi"]].round(3).to_string())
print("done")
