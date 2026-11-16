# -*- coding: utf-8 -*-
"""成员D · 数据处理：地域维度的统计与分析。

口径：load_clean()（过滤后的 429,086 条）。
产出：data/processed/D_city_profile.csv、D_city_industry_share.csv、D_city_cluster.csv
      output/figs/D_city_salary.png、D_core_industry_lq.png、D_city_simmap.png、D_core_similarity.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

plt = apply_style()
df = load_clean()
N = len(df)
df["exp_rank"] = df["experience"].map(EXP_RANK)      # 经验等级（按平均月薪推断的顺序）
df["edu_rank"] = df["education"].map(EDU_RANK)       # 学历等级（同上，仅为推断）
df["is_dayhour"] = df["salary_form"].isin(["day", "hour"])
print("clean 行数:", N, " exp_rank缺失:", int(df["exp_rank"].isna().sum()), " edu_rank缺失:", int(df["edu_rank"].isna().sum()))
print("salary_form:", df["salary_form"].value_counts().to_dict())

tier = city_tier_map(df)                              # 在清洗后数据上重新分级
TIER_COLOR = dict(zip(CITY_TIER_ORDER, PALETTE[:4]))
print("清洗后分级:", tier.value_counts().to_dict())

# ---------------------------------------------------------------------------
# 1. 城市级汇总表
# ---------------------------------------------------------------------------
g = df.groupby("city")
ci = pd.crosstab(df["city"], df["company_type"])      # 城市 × 行业 计数（全部行业）
share_all = ci.div(ci.sum(axis=1), axis=0)
prof = pd.DataFrame({
    "n_posts": g.size(),
    "n_company": g["company"].nunique(),
    "n_job_title": g["job_title"].nunique(),
    "n_industry": g["company_type"].nunique(),
    "salary_median": g["salary_mid_k"].median(),
    "salary_mean": g["salary_mid_k"].mean(),
    "salary_p25": g["salary_mid_k"].quantile(0.25),
    "salary_p75": g["salary_mid_k"].quantile(0.75),
    "exp_rank_mean": g["exp_rank"].mean(),
    "edu_rank_mean": g["edu_rank"].mean(),
    "bonus_share": g["has_bonus_months"].mean(),
    "dayhour_share": g["is_dayhour"].mean(),
})
prof["industry_hhi"] = (share_all ** 2).sum(axis=1)                       # 行业集中度 HHI
prof["industry_entropy"] = -(share_all * np.log(share_all.where(share_all > 0, 1))).sum(axis=1)
prof["top1_industry"] = share_all.idxmax(axis=1)
prof["top1_share"] = share_all.max(axis=1)
prof["tier"] = tier.reindex(prof.index)
prof = prof.sort_values("n_posts", ascending=False)
prof.round(4).to_csv(os.path.join(PROCESSED_DIR, "D_city_profile.csv"), encoding="utf-8-sig")
print("城市数:", len(prof))
num_cols = ["n_posts", "n_company", "n_job_title", "n_industry", "salary_median", "salary_mean",
            "exp_rank_mean", "edu_rank_mean", "bonus_share", "dayhour_share", "industry_hhi", "top1_share"]
print("按等级的中位数画像:\n", prof.groupby("tier")[num_cols].median().reindex(CITY_TIER_ORDER).round(3).to_string())
print("按等级合并样本的薪酬中位数:", df.groupby(df["city"].map(tier))["salary_mid_k"].median().to_dict())
print("全体薪酬中位数:", df["salary_mid_k"].median())
core = prof.index[prof["tier"] == "核心城市"].tolist()
print("核心城市画像:\n", prof.loc[core, ["n_posts", "n_company", "salary_median", "salary_p25", "salary_p75", "salary_mean",
                                   "exp_rank_mean", "edu_rank_mean", "bonus_share", "dayhour_share",
                                   "industry_hhi", "top1_industry", "top1_share"]].round(3).to_string())
nm = prof[prof["tier"] == "普通城市"]
print("普通城市 薪酬中位数 分布:", nm["salary_median"].describe().round(2).to_dict())
print("普通城市 Top1 行业频数:", nm["top1_industry"].value_counts().head(5).to_dict())
print("核心城市 Top1 行业频数:", prof.loc[core, "top1_industry"].value_counts().to_dict())
big = prof[prof["n_posts"] >= 200]
print("≥200条城市: 招聘量(log10) 与 薪酬中位数 Spearman:",
      round(big["n_posts"].corr(big["salary_median"], method="spearman"), 3), " n=", len(big))

# ---------------------------------------------------------------------------
# 2. 图：核心城市薪酬箱线图 + 招聘量×薪酬散点
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
order = prof.loc[core].sort_values(["salary_median", "salary_mean"]).index.tolist()
data = [df.loc[df["city"] == c, "salary_mid_k"].values for c in order]
bp = ax.boxplot(data, vert=False, showfliers=False, widths=0.62, patch_artist=True,
                medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=MUTED, lw=0.9),
                capprops=dict(color=MUTED, lw=0.9), boxprops=dict(facecolor=PALETTE[0], linewidth=0))
ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order)
ax.set_xlabel("月薪中值（千元/月，不显示离群点）")
ax.set_title("14 个核心城市的薪酬分布（按中位数排序）")
ax.grid(axis="y", visible=False)

ax = axes[1]
for t in CITY_TIER_ORDER:
    s = prof[prof["tier"] == t]
    ax.scatter(s["n_posts"], s["salary_median"], s=20, color=TIER_COLOR[t], alpha=0.75,
               label=f"{t}（{len(s)} 个）", zorder=3 if t == "核心城市" else 2)
ax.set_xscale("log")
ax.set_xlabel("招聘条数（对数轴）"); ax.set_ylabel("月薪中值的中位数（千元/月）")
ax.set_title("城市招聘量 × 薪酬中位数")
ax.legend(loc="upper right", fontsize=9)
save_fig(fig, "output/figs/D_city_salary.png")

# ---------------------------------------------------------------------------
# 3. 图：核心城市 × Top15 行业 的区位熵 log2(LQ)
# ---------------------------------------------------------------------------
ind_rank = df["company_type"].value_counts()
top15 = ind_rank.index[:15].tolist()
nat = ind_rank / N                                   # 全国（全体样本）行业占比
lq = share_all.loc[core, top15] / nat[top15].values
log_lq = np.log2(lq)
row_order = [core[i] for i in leaves_list(linkage(log_lq.values, "average"))]
log_lq = log_lq.loc[row_order]
print("Top15 行业合计占比:", round(nat[top15].sum(), 4))
print("Top15 行业全体占比:", nat[top15].round(4).to_dict())
print("log2(LQ) 范围:", round(log_lq.values.min(), 3), round(log_lq.values.max(), 3))
st = log_lq.stack()
print("LQ 最高5:", (2 ** st.sort_values(ascending=False).head(5)).round(2).to_dict())
print("LQ 最低5:", (2 ** st.sort_values().head(5)).round(2).to_dict())
print("核心城市 Top15 行业占比（%）:\n", (share_all.loc[row_order, top15] * 100).round(1).rename(columns=lambda c: c.replace("type_", "")).to_string())

fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
vmax = float(np.abs(log_lq.values).max())
im = ax.imshow(log_lq.values, cmap=div_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(15)); ax.set_xticklabels([c.replace("type_", "") for c in top15], rotation=45, ha="right")
ax.set_yticks(range(len(row_order))); ax.set_yticklabels(row_order)
ax.grid(False)
for i in range(log_lq.shape[0]):
    for j in range(15):
        v = lq.loc[row_order[i], top15[j]]
        if abs(log_lq.values[i, j]) >= 0.5:          # 只标注偏离明显的格子（LQ≥1.41 或 ≤0.71）
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(log_lq.values[i, j]) > vmax * 0.6 else INK)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("log2(区位熵 LQ)：>0 高于全体占比，<0 低于全体占比")
ax.set_xlabel("行业类别（招聘量 Top 15，按招聘量从左到右）"); ax.set_ylabel("核心城市（按行业结构聚类排序）")
ax.set_title("核心城市的行业专业化程度（格内数字为 LQ，仅标注偏离明显者）")
save_fig(fig, "output/figs/D_core_industry_lq.png")

# ---------------------------------------------------------------------------
# 4. 城市 × 行业（Top40 + 其他）占比矩阵
# ---------------------------------------------------------------------------
top40 = ind_rank.index[:40].tolist()
share40 = share_all[top40].copy()
share40["其他"] = 1 - share40.sum(axis=1)
share40.insert(0, "n_posts", prof["n_posts"].reindex(share40.index))
share40.insert(1, "tier", tier.reindex(share40.index))
share40 = share40.sort_values("n_posts", ascending=False)
share40.round(5).to_csv(os.path.join(PROCESSED_DIR, "D_city_industry_share.csv"), encoding="utf-8-sig")
print("Top40 行业合计占比:", round(nat[top40].sum(), 4), " 行业总数:", len(ind_rank))

# 小样本噪声评估：若所有普通城市的“真实”行业结构完全相同，仅因每城约 296 条的抽样波动，城市间余弦相似度会是多少？
normal = prof.index[prof["tier"] == "普通城市"]
obs_S = cosine_similarity(share_all.loc[normal].values)
iu = np.triu_indices(len(normal), 1)
pooled = ci.loc[normal].sum().values / ci.loc[normal].values.sum()
rng = np.random.default_rng(0)
sim_means = []
for _ in range(20):
    X = np.vstack([rng.multinomial(int(n), pooled) for n in prof.loc[normal, "n_posts"]])
    sim_means.append(cosine_similarity(X / X.sum(axis=1, keepdims=True))[iu].mean())
print(f"普通城市间行业余弦相似度: 实际均值={obs_S[iu].mean():.4f}  纯抽样噪声模拟均值={np.mean(sim_means):.4f}±{np.std(sim_means):.4f}")
core_S = cosine_similarity(share_all.loc[core].values)
print("核心城市间行业余弦相似度均值:", round(core_S[np.triu_indices(14, 1)].mean(), 4))
print("普通城市合并 vs 核心城市合并 行业分布余弦:", round(float(cosine_similarity([pooled], [ci.loc[core].sum().values / ci.loc[core].values.sum()])[0, 0]), 4))
# 300 条样本下占比估计的标准误示例
for p_ in (0.10, 0.02):
    print(f"占比 {p_:.0%} 的行业在 n=296 时标准误 = {np.sqrt(p_ * (1 - p_) / 296) * 100:.2f} 个百分点")

# ---------------------------------------------------------------------------
# 5. 城市相似度：特征矩阵 -> 标准化 -> KMeans（轮廓系数选 k）-> PCA 二维
# ---------------------------------------------------------------------------
STRUCT = ["salary_median", "salary_p25", "salary_p75", "exp_rank_mean", "edu_rank_mean", "bonus_share", "dayhour_share"]
keep = prof.index[prof["n_posts"] >= 200]             # 只对招聘量 ≥200 条的城市聚类
feat = share40.loc[keep, top40 + ["其他"]].join(prof.loc[keep, STRUCT])
Z = StandardScaler().fit_transform(feat.values)
print("参与聚类城市数:", len(keep), " 特征数:", Z.shape[1])
sil = {}
for k in range(2, 9):
    lab = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Z)
    sil[k] = silhouette_score(Z, lab)
    print(f"k={k} 轮廓系数={sil[k]:.4f} 各簇大小={np.bincount(lab).tolist()}")
best_k = max(sil, key=sil.get)
km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(Z)
lab = km.labels_
# 簇按规模从大到小重新编号为 C1、C2…
remap = {old: new for new, old in enumerate(np.argsort(-np.bincount(lab)))}
lab = np.array([remap[l] for l in lab])
ss = silhouette_samples(Z, lab)
pca = PCA(n_components=2, random_state=0)
P = pca.fit_transform(Z)
print("最佳 k:", best_k, " 轮廓系数:", round(sil[best_k], 4), " PCA 方差解释:", pca.explained_variance_ratio_.round(4).tolist())
clu = pd.DataFrame({"city": keep, "tier": prof.loc[keep, "tier"].values, "n_posts": prof.loc[keep, "n_posts"].values,
                    "cluster": [f"C{l + 1}" for l in lab], "silhouette": ss.round(4),
                    "pc1": P[:, 0].round(4), "pc2": P[:, 1].round(4)})
clu.to_csv(os.path.join(PROCESSED_DIR, "D_city_cluster.csv"), index=False, encoding="utf-8-sig")
print("簇 × 等级:\n", pd.crosstab(clu["cluster"], clu["tier"]).to_string())
print("各簇画像（中位数）:\n", clu.set_index("city").join(prof[STRUCT + ["industry_hhi"]]).groupby("cluster")[STRUCT + ["industry_hhi"]].median().round(3).to_string())
print("各簇平均轮廓:", clu.groupby("cluster")["silhouette"].mean().round(4).to_dict())
# 去掉核心城市后，仅普通+中等城市内部是否还有结构？
m_nc = clu["tier"].values != "核心城市"
Z_nc = StandardScaler().fit_transform(feat.values[m_nc])
for k in (2, 3, 4, 5):
    l2 = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Z_nc)
    print(f"[非核心城市内部] k={k} 轮廓系数={silhouette_score(Z_nc, l2):.4f} 簇大小={np.bincount(l2).tolist()}")

fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
for l in range(best_k):
    m = lab == l
    ax.scatter(P[m, 0], P[m, 1], s=22, color=PALETTE[l], alpha=0.8, label=f"簇 C{l + 1}（{m.sum()} 个城市）")
is_core = clu["tier"].values == "核心城市"
ax.scatter(P[is_core, 0], P[is_core, 1], s=60, facecolors="none", edgecolors=INK, linewidths=0.9, label="核心城市（黑圈）")
# 核心城市标签：按纵坐标排开，避免重叠
idx = np.where(is_core)[0]
idx = idx[np.argsort(P[idx, 1])]
x_lab = P[idx, 0].max() + 2.5                       # 标签列放在核心城市群右侧
ys = np.linspace(P[idx, 1].min() - 1.5, P[idx, 1].max() + 2.0, len(idx))
for i, y in zip(idx, ys):
    ax.annotate(clu["city"].iloc[i], xy=(P[i, 0], P[i, 1]), xytext=(x_lab, y), fontsize=8.5, color=INK_2, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlabel(f"主成分 1（解释 {pca.explained_variance_ratio_[0]:.1%} 方差）")
ax.set_ylabel(f"主成分 2（解释 {pca.explained_variance_ratio_[1]:.1%} 方差）")
ax.set_title(f"城市相似度地图：{len(keep)} 个城市的 PCA 投影（KMeans k={best_k}，轮廓系数 {sil[best_k]:.2f}）")
ax.legend(loc="lower right", fontsize=9)
print("PC1 最大的非核心城市:", clu.loc[~is_core].nlargest(3, "pc1")[["city", "tier", "n_posts", "cluster", "pc1"]].to_dict("records"))
print(prof.loc[clu.loc[~is_core].nlargest(1, "pc1")["city"], ["salary_median", "salary_mean", "edu_rank_mean", "bonus_share", "top1_industry", "top1_share"]].round(3).to_dict("records"))
save_fig(fig, "output/figs/D_city_simmap.png")

# ---------------------------------------------------------------------------
# 6. 14 个核心城市之间的相似度矩阵（在核心城市内部重新标准化，层次聚类排序）
# ---------------------------------------------------------------------------
feat_core = share40.loc[core, top40 + ["其他"]].join(prof.loc[core, STRUCT])
Zc = StandardScaler().fit_transform(feat_core.values)
Sc = cosine_similarity(Zc)
o = leaves_list(linkage(Zc, "average", metric="cosine"))
Sc_o = Sc[np.ix_(o, o)]
names = [core[i] for i in o]
off = Sc[np.triu_indices(14, 1)]
print("核心城市相似度(内部标准化后的余弦) 非对角: min/mean/max =", off.min().round(3), off.mean().round(3), off.max().round(3))
pairs = pd.Series({(core[i], core[j]): Sc[i, j] for i in range(14) for j in range(i + 1, 14)})
print("最相似5对:", pairs.sort_values(ascending=False).head(5).round(3).to_dict())
print("最不相似5对:", pairs.sort_values().head(5).round(3).to_dict())
print("排序后顺序:", names)
for k in (2, 3, 4):
    from scipy.cluster.hierarchy import fcluster
    lc = fcluster(linkage(Zc, "average", metric="cosine"), k, "maxclust")
    print(f"[核心城市层次聚类] k={k} 轮廓(cosine)={silhouette_score(Zc, lc, metric='cosine'):.4f} 分组=", dict(zip(core, lc.tolist())))

fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
vmax = float(np.abs(off).max())
im = ax.imshow(Sc_o, cmap=div_cmap(), vmin=-vmax, vmax=vmax)
ax.set_xticks(range(14)); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticks(range(14)); ax.set_yticklabels(names)
ax.grid(False)
for i in range(14):
    for j in range(14):
        if i != j and abs(Sc_o[i, j]) >= 0.3:
            ax.text(j, i, f"{Sc_o[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=7.5,
                    color="white" if abs(Sc_o[i, j]) > vmax * 0.6 else INK)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("余弦相似度（特征在 14 城内部标准化；对角线为 1，超出色标）")
ax.set_title("14 个核心城市的两两相似度（层次聚类排序）")
save_fig(fig, "output/figs/D_core_similarity.png")
print("done")
