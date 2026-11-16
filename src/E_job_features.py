# -*- coding: utf-8 -*-
"""成员E · 第二部分：多维度职位特征的构建与差异度的第一轮度量。

基于 load_clean()：筛选高频职位 -> 六个维度的职位级特征 -> 对数/标准化 ->
相关性、PCA、KMeans、职位对距离、分维度贡献。
输出：data/processed/E_job_features.csv、E_job_embedding.csv 与 3 张图。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *   # noqa
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt = apply_style()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)

MIN_COUNT = 30      # 高频职位阈值：出现次数 >= 30
K_CLUSTER = 4       # KMeans 类别数（见下方轮廓系数比较）
SEED = 42

c = load_clean()
N = len(c)

# ---------------------------------------------------------------- 1. 分析粒度
vc = c["job_title"].value_counts()
print("== 分析粒度 ==")
print(f"清洗后职位种数 {vc.size:,}；只出现 1 次 {int((vc == 1).sum()):,}（{(vc == 1).mean():.2%}）；中位出现次数 {vc.median():.0f}")
for t in (10, 20, 30, 50, 100):
    s = vc[vc >= t]
    print(f"  >= {t:3d}: {s.size:5d} 种，覆盖 {int(s.sum()):7,d} 条（{s.sum() / N:.2%}）")
jobs = vc[vc >= MIN_COUNT].index
d = c[c["job_title"].isin(jobs)].copy()
d["exp_rank"] = d["experience"].map(EXP_RANK)
d["edu_rank"] = d["education"].map(EDU_RANK)
d["is_core"] = d["city"].map(city_tier_map(c)).eq("核心城市")   # 城市分级按全量清洗数据确定


def entropy(counts: pd.Series) -> float:
    """香农熵（自然对数）。"""
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def dist_feats(col: str, prefix: str) -> pd.DataFrame:
    """某类别字段在每个职位内的分布：类别数、Top1 占比、熵。"""
    ct = d.groupby(["job_title", col]).size()
    g = ct.groupby(level=0)
    return pd.DataFrame({
        f"{prefix}_n": g.size(),
        f"{prefix}_top1_share": g.max() / g.sum(),
        f"{prefix}_entropy": g.apply(entropy),
    })


# ---------------------------------------------------------------- 2. 特征构建
g = d.groupby("job_title")
feat = pd.DataFrame({
    # 薪酬维度
    "sal_median": g["salary_mid_k"].median(),
    "sal_p25": g["salary_mid_k"].quantile(0.25),
    "sal_p75": g["salary_mid_k"].quantile(0.75),
    "sal_span_mean": g["salary_span_k"].mean(),
    "sal_bonus_share": g["has_bonus_months"].mean(),
    # 经验 / 学历维度（等级均值）
    "exp_rank_mean": g["exp_rank"].mean(),
    "edu_rank_mean": g["edu_rank"].mean(),
})
exp_share = pd.crosstab(d["job_title"], d["experience"], normalize="index").reindex(columns=EXP_ORDER, fill_value=0)
exp_share.columns = [f"exp_share_{k}" for k in EXP_ORDER]
edu_share = pd.crosstab(d["job_title"], d["education"], normalize="index").reindex(columns=EDU_ORDER, fill_value=0)
edu_share.columns = [f"edu_share_{k}" for k in EDU_ORDER]
ind = dist_feats("company_type", "ind")
city = dist_feats("city", "city").drop(columns="city_top1_share")
feat = feat.join(exp_share).join(edu_share).join(ind).join(city)
feat["city_core_share"] = g["is_core"].mean()
feat["demand_posts"] = g.size()
feat["demand_companies"] = g["company"].nunique()
feat = feat.loc[jobs]            # 按招聘量降序

GROUPS = {   # 维度 -> 特征列
    "薪酬": ["sal_median", "sal_p25", "sal_p75", "sal_span_mean", "sal_bonus_share"],
    "经验": ["exp_rank_mean"] + list(exp_share.columns),
    "学历": ["edu_rank_mean"] + list(edu_share.columns),
    "行业": ["ind_entropy", "ind_top1_share", "ind_n"],
    "地域": ["city_core_share", "city_n", "city_entropy"],
    "需求": ["demand_posts", "demand_companies"],
}
FEATS = [f for fs in GROUPS.values() for f in fs]
feat = feat[FEATS]
feat.round(4).to_csv(os.path.join(PROCESSED_DIR, "E_job_features.csv"), encoding="utf-8-sig", index_label="job_title")
print(f"\n== 特征矩阵 == {feat.shape[0]} 个职位 × {feat.shape[1]} 个特征；各维度特征数:", {k: len(v) for k, v in GROUPS.items()})
print(feat.describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())
print("偏度（取对数前）:", feat[["sal_median", "sal_p25", "sal_p75", "sal_span_mean", "demand_posts", "demand_companies", "ind_n", "city_n"]].skew().round(2).to_dict())

# ---------------------------------------------------------------- 3. 预处理
LOG_COLS = ["sal_median", "sal_p25", "sal_p75", "sal_span_mean", "demand_posts", "demand_companies", "ind_n", "city_n"]
X = feat.copy()
X[LOG_COLS] = np.log(X[LOG_COLS])
print("偏度（取对数后）:", X[LOG_COLS].skew().round(2).to_dict())
Z = (X - X.mean()) / X.std(ddof=0)
print("零方差特征:", list(X.columns[X.std() == 0]))

corr = Z.corr()
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=np.abs, ascending=False)
print("\n== |r| >= 0.85 的特征对 ==")
print(pairs[pairs.abs() >= 0.85].round(3).to_string())
print("|r|>=0.85 对数:", int((pairs.abs() >= 0.85).sum()), " |r|>=0.7 对数:", int((pairs.abs() >= 0.7).sum()))

# ---------------------------------------------------------------- 4. PCA
pca = PCA(random_state=SEED).fit(Z.values)
evr = pca.explained_variance_ratio_
print("\n== PCA ==")
print("前 8 个主成分方差解释率:", np.round(evr[:8], 4).tolist())
print("累计:", np.round(np.cumsum(evr)[:8], 4).tolist())
print("累计达 80% / 90% 所需主成分数:", int(np.searchsorted(np.cumsum(evr), 0.8) + 1), int(np.searchsorted(np.cumsum(evr), 0.9) + 1))
load = pd.DataFrame(pca.components_[:4].T, index=FEATS, columns=["PC1", "PC2", "PC3", "PC4"])
for pc in load.columns:
    top = load[pc].sort_values(key=np.abs, ascending=False).head(6).round(2)
    print(f"  {pc} 载荷 Top6:", top.to_dict())
P = pca.transform(Z.values)
# 让 PC1 正向 = 高薪，便于解读
if load.loc["sal_median", "PC1"] < 0:
    P[:, 0] *= -1; load["PC1"] *= -1

# ---------------------------------------------------------------- 5. 聚类
print("\n== KMeans 轮廓系数（标准化特征空间，欧氏距离） ==")
sil = {}
for k in range(2, 9):
    lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z.values)
    sil[k] = silhouette_score(Z.values, lab)
    print(f"  k={k}: {sil[k]:.3f}  各类大小 {np.bincount(lab).tolist()}")
km = KMeans(n_clusters=K_CLUSTER, n_init=20, random_state=SEED).fit(Z.values)
# 类别按月薪中位数由低到高重新编号为 C1..Ck
order = pd.Series(feat["sal_median"].values).groupby(km.labels_).median().sort_values().index
remap = {old: new + 1 for new, old in enumerate(order)}
cluster = np.array([remap[l] for l in km.labels_])
prof = feat.assign(cluster=cluster).groupby("cluster").agg(
    n=("sal_median", "size"), sal_median=("sal_median", "median"), exp_rank=("exp_rank_mean", "mean"),
    edu_rank=("edu_rank_mean", "mean"), ind_entropy=("ind_entropy", "mean"),
    core_share=("city_core_share", "mean"), posts=("demand_posts", "median"), bonus=("sal_bonus_share", "mean"))
print(f"选用 k={K_CLUSTER}，轮廓系数 {sil[K_CLUSTER]:.3f}；各类画像:")
print(prof.round(3).to_string())
print("仅用前 2 个主成分坐标计算的轮廓系数:", round(silhouette_score(P[:, :2], cluster), 3))

emb = pd.DataFrame({"job_title": feat.index, "pc1": P[:, 0], "pc2": P[:, 1], "pc3": P[:, 2], "pc4": P[:, 3],
                    "cluster": cluster, "sal_median": feat["sal_median"].values, "demand_posts": feat["demand_posts"].values})
emb.round(4).to_csv(os.path.join(PROCESSED_DIR, "E_job_embedding.csv"), encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------- 6. 职位对距离
D = squareform(pdist(Z.values, "euclidean"))
Dc = squareform(pdist(Z.values, "cosine"))
iu = np.triu_indices_from(D, 1)
dv = D[iu]
print("\n== 两两距离 ==")
print(f"职位对数 {dv.size:,}；欧氏距离 均值 {dv.mean():.2f} 中位 {np.median(dv):.2f} 最小 {dv.min():.2f} 最大 {dv.max():.2f}")
print("欧氏与余弦距离的 Spearman 相关:", round(pd.Series(dv).corr(pd.Series(Dc[iu]), method="spearman"), 3))
short = [j[:8] for j in feat.index]


def show_pairs(idx, title):
    print(title)
    for n_ in idx:
        i, j = iu[0][n_], iu[1][n_]
        a, b = feat.iloc[i], feat.iloc[j]
        print(f"  {short[i]} | {short[j]} | 欧氏 {D[i, j]:.2f} | 余弦 {Dc[i, j]:.3f} | 月薪中位 {a.sal_median:.1f} vs {b.sal_median:.1f}"
              f" | 经验 {a.exp_rank_mean:.2f} vs {b.exp_rank_mean:.2f} | 学历 {a.edu_rank_mean:.2f} vs {b.edu_rank_mean:.2f}"
              f" | 招聘量 {int(a.demand_posts)} vs {int(b.demand_posts)} | 行业熵 {a.ind_entropy:.2f} vs {b.ind_entropy:.2f}")


srt = np.argsort(dv)
show_pairs(srt[::-1][:5], "差异最大的 5 对:")
show_pairs(srt[:5], "最相似的 5 对:")


def distinct_pairs(sorted_idx, n_pairs=5):
    """按顺序取职位对，但每个职位最多出现一次（避免被单个离群职位占满）。"""
    used, out = set(), []
    for n_ in sorted_idx:
        i, j = iu[0][n_], iu[1][n_]
        if i in used or j in used:
            continue
        used.update((i, j)); out.append(n_)
        if len(out) == n_pairs:
            break
    return out


show_pairs(distinct_pairs(srt[::-1]), "差异最大的 5 对（每个职位只出现一次）:")
# 离群职位的成因：哪些标准化特征绝对值最大
far = int(np.argmax(D.mean(axis=1)))
print(f"平均距离最大的职位 {short[far]}（平均欧氏距离 {D[far].mean():.2f}），|z| 最大的特征:",
      Z.iloc[far].sort_values(key=np.abs, ascending=False).head(4).round(1).to_dict())
print("  其原始取值:", feat.iloc[far][Z.iloc[far].abs().sort_values(ascending=False).head(4).index].round(3).to_dict())
print("各特征 |z| 最大值 Top6:", Z.abs().max().sort_values(ascending=False).head(6).round(1).to_dict())

# ---------------------------------------------------------------- 7. 分维度差异贡献
# 标准化后每个特征方差均为 1，直接按方差求和只反映特征个数；因此用三个口径：
# (a) 对平方欧氏距离的份额（= 特征数份额，作为基准）
# (b) 前 3 个主成分对该维度特征方差的平均解释比例（该维度与主要差异方向的一致程度）
# (c) 聚类对该维度特征的平均解释比例 eta^2（该维度区分职位群的能力）
K_PC = 3
comm = (pca.components_[:K_PC].T ** 2 * pca.explained_variance_[:K_PC]).sum(axis=1) / Z.var(ddof=1).values
comm = pd.Series(comm, index=FEATS)
zc = Z.assign(cluster=cluster)
eta = pd.Series({f_: 1 - zc.groupby("cluster")[f_].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum() / (zc[f_] ** 2).sum() for f_ in FEATS})
contrib = pd.DataFrame({
    "特征数": {k: len(v) for k, v in GROUPS.items()},
    "距离份额": {k: len(v) / len(FEATS) for k, v in GROUPS.items()},
    "前3主成分解释比例": {k: comm[v].mean() for k, v in GROUPS.items()},
    "聚类解释比例": {k: eta[v].mean() for k, v in GROUPS.items()},
})
# 前 3 主成分的方差中各维度所占份额（平方载荷 × 特征值）
pc_share = pd.DataFrame({k: (pca.components_[:K_PC, [FEATS.index(f_) for f_ in v]] ** 2).sum(axis=1) for k, v in GROUPS.items()},
                        index=[f"PC{i + 1}" for i in range(K_PC)])
print("\n== 分维度贡献 ==")
print(contrib.round(3).to_string())
print("各主成分平方载荷在各维度的分配:\n", pc_share.round(3).to_string())
print("eta^2 最高的 6 个特征:", eta.sort_values(ascending=False).head(6).round(3).to_dict())

# ================================================================ 图 1：相关性热力图
CN = {
    "sal_median": "月薪中位数", "sal_p25": "月薪P25", "sal_p75": "月薪P75", "sal_span_mean": "带宽均值", "sal_bonus_share": "13薪占比",
    "exp_rank_mean": "经验等级均值", "edu_rank_mean": "学历等级均值",
    "ind_entropy": "行业熵", "ind_top1_share": "Top1行业占比", "ind_n": "行业数",
    "city_core_share": "核心城市占比", "city_n": "城市数", "city_entropy": "城市熵",
    "demand_posts": "招聘量", "demand_companies": "企业数",
}
lab = [CN.get(f_, f_.replace("exp_share_", "经验·").replace("edu_share_", "学历·")) for f_ in FEATS]
fig, ax = plt.subplots(figsize=(10.5, 8.6), constrained_layout=True)
im = ax.imshow(corr.values, cmap=div_cmap(), vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATS)), lab, rotation=90, fontsize=8.5)
ax.set_yticks(range(len(FEATS)), lab, fontsize=8.5)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
pos = 0
for k, v in GROUPS.items():   # 用细线框出各维度
    ax.add_patch(plt.Rectangle((pos - .5, pos - .5), len(v), len(v), fill=False, ec=INK_2, lw=1.0))
    ax.text(pos + len(v) - .45, pos - .62, k, fontsize=9, color=INK, ha="left", va="bottom", fontweight="bold")
    pos += len(v)
cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cb.set_label("Pearson 相关系数"); cb.outline.set_visible(False)
ax.set_title("特征相关性：薪酬组内部、招聘量与企业数高度冗余", pad=22)
save_fig(fig, "output/figs/E_feature_corr.png")

# ================================================================ 图 2：方差解释率 + 分维度贡献
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
ax = axes[0]
n_show = 10
ax.bar(range(1, n_show + 1), evr[:n_show] * 100, color=PALETTE[0], width=0.68)
for i in range(4):
    ax.text(i + 1, evr[i] * 100 + 0.5, f"{evr[i]:.1%}", ha="center", fontsize=9.5, color=INK_2)
ax.set_xticks(range(1, n_show + 1), [f"PC{i}" for i in range(1, n_show + 1)])
ax.set_ylabel("方差解释率（%）")
ax.set_title(f"前 4 个主成分累计解释 {evr[:4].sum():.1%} 的方差", fontsize=11.5)
ax.grid(axis="x", visible=False)
ax = axes[1]
cs = contrib["聚类解释比例"].sort_values()
ax.barh(cs.index, cs.values * 100, color=[HIGHLIGHT if k == cs.idxmax() else NEUTRAL for k in cs.index], height=0.62)
for i, v in enumerate(cs.values):
    ax.text(v * 100 + 0.8, i, f"{v:.1%}", va="center", fontsize=9.5, color=INK_2)
ax.set_xlim(0, cs.max() * 100 * 1.18)
ax.set_xlabel(f"k={K_CLUSTER} 聚类对该维度特征方差的平均解释比例（%）")
ax.set_title(f"{cs.idxmax()}维度最能区分职位群", fontsize=11.5)
ax.grid(axis="y", visible=False)
save_fig(fig, "output/figs/E_pca_variance.png")

# ================================================================ 图 3：PC1-PC2 散点
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), constrained_layout=True, sharex=True, sharey=True)
ax = axes[0]
sc = ax.scatter(P[:, 0], P[:, 1], c=np.log10(feat["sal_median"]), cmap=seq_cmap(), s=13, linewidths=0)
cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
ticks = [t for t in (3, 5, 8, 12, 20, 30, 50) if feat["sal_median"].min() <= t <= feat["sal_median"].max()]
cb.set_ticks(np.log10(ticks)); cb.set_ticklabels([str(t) for t in ticks])
cb.set_label("月薪中位数（千元/月，对数色阶）"); cb.outline.set_visible(False)
ax.set_title("按月薪中位数着色：PC1 即薪酬—资历轴", fontsize=11.5)
ax.set_xlabel(f"PC1（{evr[0]:.1%}）"); ax.set_ylabel(f"PC2（{evr[1]:.1%}）")
ax = axes[1]
for k in range(1, K_CLUSTER + 1):
    m = cluster == k
    ax.scatter(P[m, 0], P[m, 1], color=PALETTE[k - 1], s=13, linewidths=0, alpha=0.85, label=f"C{k}（{int(m.sum())} 个）")
ax.legend(loc="best", markerscale=1.8, handletextpad=0.2)
ax.set_title(f"按 KMeans（k={K_CLUSTER}）着色：轮廓系数 {sil[K_CLUSTER]:.2f}，边界连续", fontsize=11.5)
ax.set_xlabel(f"PC1（{evr[0]:.1%}）")
save_fig(fig, "output/figs/E_pca_scatter.png")
print("\n完成。")
