# -*- coding: utf-8 -*-
"""成员E · 第二轮 · 职位差异度的量化评估（赛题问题 1）。

差异度 = 两个对象（职位族 / 高频职位）在薪酬、经验、学历、行业、地域五个维度上“要求分布”的总变差距离，
每个维度都在 0~1，综合差异度为加权平均。默认薪酬、经验、学历、行业四维等权；地域维度照常计算并保存，
但默认权重为 0（职位族在 14 个城市的覆盖受采集结构影响，见第 2 节的核对）。
口径：核心城市口径（批次 B+C 且有职位族）并剔除兼职类记录。
输出：data/processed/E2_family_distance.csv、E2_title_features.csv 与 4 张图；正文数字均由本脚本打印。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *   # noqa
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import cdist, squareform
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.manifold import MDS

plt = apply_style()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 100)

DIMS = ["薪酬", "经验", "学历", "行业", "地域"]
DEFAULT_W = [1, 1, 1, 1, 0]   # 默认权重：地域不计入综合差异度
N_SALARY_BIN = 10      # 对数月薪按全体十分位分档
N_INDUSTRY = 30        # 行业取记录数最多的 30 个 + 其他
TITLE_MIN = 30         # 高频职位阈值
SEED = 0
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. 口径与各维度的类别编码
df = load_enriched()
d = df[df.job_family.notna() & ~df.is_parttime].reset_index(drop=True)
print(f"== 口径 == 核心城市口径 {int(df.job_family.notna().sum()):,} 条，剔除兼职类后 {len(d):,} 条，{d.city.nunique()} 城，{d.job_family.nunique()} 族")

logs = np.log(d.salary_mid_k.values)
edges = np.unique(np.quantile(logs, np.linspace(0, 1, N_SALARY_BIN + 1))[1:-1])
top_ind = d.company_type.value_counts().index[:N_INDUSTRY]
print("薪酬分档边界（千元/月）:", np.exp(edges).round(2).tolist())
print(f"行业 Top{N_INDUSTRY} 覆盖 {d.company_type.isin(top_ind).mean():.1%}")
codes = {
    "薪酬": np.searchsorted(edges, logs, side="right"),
    "经验": d.exp_level.values.astype(int),
    "学历": d.edu_level.values.astype(int),
    "行业": pd.Categorical(d.company_type.where(d.company_type.isin(top_ind), "其他")).codes.astype(int),
    "地域": pd.Categorical(d.city).codes.astype(int),
}
K = {k: int(v.max()) + 1 for k, v in codes.items()}
print("各维度类别数:", K)
print("薪酬各档占比:", (np.bincount(codes["薪酬"]) / len(d)).round(3).tolist())


def profiles(g, G, idx=None):
    """各对象在每个维度上的分布矩阵（行归一）。g 为对象编号数组，idx 为所用记录的下标。"""
    out = {}
    for k in DIMS:
        c = codes[k] if idx is None else codes[k][idx]
        m = np.zeros((G, K[k]))
        np.add.at(m, (g, c), 1)
        out[k] = m / np.maximum(m.sum(1, keepdims=True), 1)
    return out


def tv_matrices(P, Q=None):
    """分维度的总变差距离矩阵 dict。"""
    return {k: 0.5 * cdist(P[k], P[k] if Q is None else Q[k], "cityblock") for k in DIMS}


def combine(D, w=None):
    w = np.asarray(DEFAULT_W if w is None else w, float)
    return sum(wi * D[k] for wi, k in zip(w / w.sum(), DIMS))


# ---------------------------------------------------------------- 2. 职位族粒度：51×51
prof = family_profile()
fams = sorted(d.job_family.unique())
F = len(fams)
fam_id = pd.Categorical(d.job_family, categories=fams).codes.astype(int)
fam_n = np.bincount(fam_id)
PF = profiles(fam_id, F)
DF = tv_matrices(PF)
DF["综合"] = combine(DF)
DF["五维等权"] = combine(DF, [1, 1, 1, 1, 1])
iu = np.triu_indices(F, 1)
pairs = pd.DataFrame({"family_a": np.array(fams)[iu[0]], "family_b": np.array(fams)[iu[1]]})
for k in DIMS + ["综合", "五维等权"]:
    pairs[k] = DF[k][iu]
pairs.rename(columns={"薪酬": "d_salary", "经验": "d_experience", "学历": "d_education", "行业": "d_industry",
                      "地域": "d_region", "综合": "d_overall", "五维等权": "d_overall_with_region"}).round(4).to_csv(
    os.path.join(PROCESSED_DIR, "E2_family_distance.csv"), index=False, encoding="utf-8-sig")
print(f"\n== 职位族 == {F} 族，{len(pairs)} 对；族记录数 {fam_n.min()}~{fam_n.max()}")
summ = pairs[DIMS + ["综合"]].agg(["mean", "std", "min", "median", "max"]).T
summ["cv"] = summ["std"] / summ["mean"]
summ["占综合比重"] = summ["mean"] / summ.loc[DIMS[:4], "mean"].sum()
summ["与综合的秩相关"] = [spearmanr(pairs[k], pairs["综合"])[0] for k in summ.index]
print(summ.round(3).to_string())
print("维度间秩相关:\n", pairs[DIMS].corr(method="spearman").round(2).to_string())

# 地域维度的核对：职位族在 14 个城市的覆盖
ct = pd.crosstab(d.job_family, d.city)
miss = (ct.div(ct.sum(1), axis=0) < 0.005).sum(1)
print(f"在 >=3 个核心城市几乎没有记录（份额 <0.5%）的职位族 {int((miss >= 3).sum())} 个，缺失 {miss[miss >= 3].min()}~{miss[miss >= 3].max()} 城，"
      f"占记录 {ct.sum(1)[miss >= 3].sum() / len(d):.1%}：{miss[miss >= 3].to_dict()}")
full = np.isin(pairs.family_a, miss[miss == 0].index) & np.isin(pairs.family_b, miss[miss == 0].index)
print(f"地域差异度均值：两族都覆盖 14 城的族对 {pairs['地域'][full].mean():.3f}（{int(full.sum())} 对），其余族对 {pairs['地域'][~full].mean():.3f}")
# 行业粒度的敏感性：Top10 + 其他
top10 = d.company_type.value_counts().index[:10]
c10 = pd.Categorical(d.company_type.where(d.company_type.isin(top10), "其他")).codes
m10 = np.zeros((F, 11)); np.add.at(m10, (fam_id, c10), 1); m10 /= m10.sum(1, keepdims=True)
print(f"行业改用 Top10+其他：差异度均值 {(0.5 * cdist(m10, m10, 'cityblock'))[iu].mean():.3f}（Top30 为 {pairs['行业'].mean():.3f}）")

lab = family_labels()
show = pairs.sort_values("综合")
cols = ["family_a", "family_b", "综合"] + DIMS
print("\n最相似 8 对:\n", show.head(8)[cols].round(3).to_string())
print("最不相似 8 对:\n", show.tail(8)[cols].round(3).to_string())


def dedupe(tab, k=4):
    """每个职位族只出现一次的前 k 对。"""
    seen, keep = set(), []
    for i, r in tab.iterrows():
        if r.family_a not in seen and r.family_b not in seen:
            keep.append(i); seen |= {r.family_a, r.family_b}
        if len(keep) == k:
            break
    return tab.loc[keep]


for name, tab in (("最相似（每族一次）", dedupe(show)), ("最不相似（每族一次）", dedupe(show.iloc[::-1]))):
    tab = tab.assign(a=tab.family_a.map(lab), b=tab.family_b.map(lab))
    print(name + ":\n", tab[["a", "b", "综合"] + DIMS].round(3).to_string())
used = sorted(set(show.head(6).family_a) | set(show.head(6).family_b) | set(show.tail(6).family_a) | set(show.tail(6).family_b))
print(prof.loc[used, ["n", "salary_median", "bachelor_share", "edu_any_share", "exp_any_share", "top_industry", "top_industry_share", "family_label"]].round(3).to_string())
mean_d = pd.Series((DF["综合"].sum(1)) / (F - 1), index=fams)
print("平均综合差异度最大的 6 族:", mean_d.sort_values().tail(6).round(3).to_dict())
print("平均综合差异度最小的 6 族:", mean_d.sort_values().head(6).round(3).to_dict())

# ---------------------------------------------------------------- 3. 稳健性
print("\n== 稳健性 ==")
rows_by_fam = [np.flatnonzero(fam_id == i) for i in range(F)]


def split_half(rows_list, reps):
    """每个对象的记录随机对半拆分，返回各维度与综合差异度（对象 × 重复 的中位数）。"""
    res = {k: [] for k in DIMS + ["综合"]}
    for _ in range(reps):
        ia, ib, ga, gb = [], [], [], []
        for i, r in enumerate(rows_list):
            p = rng.permutation(r); h = len(p) // 2
            ia.append(p[:h]); ib.append(p[h:2 * h]); ga.append(np.full(h, i)); gb.append(np.full(h, i))
        G = len(rows_list)
        A = profiles(np.concatenate(ga), G, np.concatenate(ia)); B = profiles(np.concatenate(gb), G, np.concatenate(ib))
        dd = {k: 0.5 * np.abs(A[k] - B[k]).sum(1) for k in DIMS}
        dd["综合"] = combine(dd)
        for k in res:
            res[k].append(dd[k])
    return {k: np.median(np.array(v), axis=0) for k, v in res.items()}


sh = split_half(rows_by_fam, 20)
base_f = pd.Series({k: np.median(v) for k, v in sh.items()})
print("职位族对半拆分基准（各族中位数）:", base_f.round(4).to_dict(), " 最大:", {k: round(float(v.max()), 3) for k, v in sh.items()})
print("  最小的族对综合差异度 / 基准中位数:", round(pairs["综合"].min() / base_f["综合"], 1),
      " 综合差异度低于 2 倍最大基准的族对数:", int((pairs["综合"] < 2 * sh["综合"].max()).sum()))
print("  各维度 均值/基准:", (summ.loc[DIMS + ["综合"], "mean"] / base_f).round(1).to_dict())

# 基准随样本量的变化：同族内抽两份各 m 条（噪声），异族各抽 m 条（信号+噪声）
M_LIST = [15, 30, 50, 100, 200, 500, 1000]
curve = []
for m in M_LIST:
    ok = [i for i in range(F) if fam_n[i] >= 2 * m]
    within, between = [], []
    for _ in range(30):
        ia, ib, g = [], [], []
        for j, i in enumerate(ok):
            p = rng.choice(rows_by_fam[i], 2 * m, replace=False)
            ia.append(p[:m]); ib.append(p[m:]); g.append(np.full(m, j))
        g = np.concatenate(g)
        A = profiles(g, len(ok), np.concatenate(ia)); B = profiles(g, len(ok), np.concatenate(ib))
        X = combine(tv_matrices(A, B))
        within.append(np.diag(X)); between.append(X[~np.eye(len(ok), dtype=bool)])
    w_, b_ = np.concatenate(within), np.concatenate(between)
    curve.append({"m": m, "n_fam": len(ok), "within_med": np.median(w_), "within_p90": np.quantile(w_, .9),
                  "between_med": np.median(b_), "between_p10": np.quantile(b_, .1),
                  "auc": mannwhitneyu(b_, w_).statistic / (len(b_) * len(w_))})
curve = pd.DataFrame(curve)
curve["noise_ratio"] = curve.within_med / curve.between_med
print("基准随样本量（综合差异度）:\n", curve.round(3).to_string())

# 权重方案与度量替换
schemes = {"五维等权(含地域)": [1, 1, 1, 1, 1], "赛题三维(薪酬/经验/行业)": [1, 1, 0, 1, 0], "薪酬加倍": [2, 1, 1, 1, 0],
           "行业减半": [2, 2, 2, 1, 0], "仅门槛(经验+学历)": [0, 1, 1, 0, 0], "仅薪酬": [1, 0, 0, 0, 0], "仅行业": [0, 0, 0, 1, 0]}
top20 = set(pairs["综合"].nsmallest(20).index)
for name, w in schemes.items():
    v = combine(DF, w)[iu]
    print(f"  权重方案 {name}: 与默认的秩相关 {spearmanr(v, pairs['综合'])[0]:.3f}，最相似 20 对重合 {len(top20 & set(pd.Series(v).nsmallest(20).index))}")
JS = {k: cdist(PF[k], PF[k], "jensenshannon") / np.sqrt(np.log(2)) for k in DIMS}
print(f"  度量替换 JS 距离: 与 TV 综合差异度的秩相关 {spearmanr(combine(JS)[iu], pairs['综合'])[0]:.3f}")
# 城市内计算再平均：消除各族城市覆盖不同带来的混杂（两族在该城都不少于 100 条才计入）
city_id = codes["地域"]
acc, cnt = np.zeros((F, F)), np.zeros((F, F))
for c in range(K["地域"]):
    idx = np.flatnonzero(city_id == c)
    okf = np.bincount(fam_id[idx], minlength=F) >= 100
    sub = idx[okf[fam_id[idx]]]
    Dc = combine(tv_matrices(profiles(fam_id[sub], F, sub)))
    m = okf[:, None] & okf[None, :]
    acc[m] += Dc[m]; cnt[m] += 1
okp = cnt[iu] >= 3
within_city = acc[iu][okp] / cnt[iu][okp]
print(f"  城市内计算再平均（{int(okp.sum())} 对有 >=3 个共同城市）: 与默认综合差异度的秩相关 {spearmanr(within_city, pairs['综合'][okp])[0]:.3f}；"
      f"均值 {within_city.mean():.3f} 对 {pairs['综合'][okp].mean():.3f}")
qs = np.linspace(.005, .995, 100)
Q = np.vstack([np.quantile(logs[r], qs) for r in rows_by_fam])
W = cdist(Q, Q, "cityblock") / len(qs) / (np.quantile(logs, .95) - np.quantile(logs, .05))
print(f"  薪酬维度改用标准化 Wasserstein 距离: 与分档 TV 的秩相关 {spearmanr(W[iu], pairs['薪酬'])[0]:.3f}；W 均值 {W[iu].mean():.3f}")
med_gap = np.abs(np.log(prof.loc[fams, 'salary_median'].values[:, None] / prof.loc[fams, 'salary_median'].values[None, :]))[iu]
print(f"  薪酬分布距离与中位数对数差的秩相关 {spearmanr(med_gap, pairs['薪酬'])[0]:.3f}；中位数相同（差为 0）的族对 {int((med_gap == 0).sum())} 对，其薪酬差异度 {pairs['薪酬'][med_gap == 0].min():.3f}~{pairs['薪酬'][med_gap == 0].max():.3f}")

# ---------------------------------------------------------------- 4. 高频职位粒度
print("\n== 高频职位 ==")
vc = d.job_title.value_counts()
titles = vc[vc >= TITLE_MIN].index
t = d[d.job_title.isin(titles)]
print(f"核心城市口径（不含兼职）中出现 >= {TITLE_MIN} 次的职位 {len(titles)} 种，覆盖 {len(t):,} 条（{len(t) / len(d):.1%}）；"
      f"全量口径 >= {TITLE_MIN} 次的职位 {int((df.job_title.value_counts() >= TITLE_MIN).sum())} 种")
tid = pd.Categorical(t.job_title, categories=titles).codes.astype(int)
tidx = t.index.values
T = len(titles)
PT = profiles(tid, T, tidx)
DT = tv_matrices(PT)
DT["综合"] = combine(DT)
g = t.groupby("job_title")
tf = pd.DataFrame({
    "n": g.size(),
    "job_family": g.job_family.agg(lambda s: s.value_counts().index[0]),
    "family_purity": g.job_family.agg(lambda s: s.value_counts(normalize=True).iloc[0]),
    "salary_median": g.salary_mid_k.median(), "salary_p25": g.salary_mid_k.quantile(.25), "salary_p75": g.salary_mid_k.quantile(.75),
    "exp_any_share": g.exp_any.mean(), "exp_mode": g.exp_label.agg(lambda s: s.value_counts().index[0]),
    "senior_share": g.exp_level.apply(lambda s: (s >= 4).mean()),
    "edu_any_share": g.edu_any.mean(), "edu_mode": g.edu_label.agg(lambda s: s.value_counts().index[0]),
    "bachelor_share": g.edu_level.apply(lambda s: (s >= 4).mean()),
    "top_industry": g.company_type.agg(lambda s: s.value_counts().index[0]),
    "top_industry_share": g.company_type.agg(lambda s: s.value_counts(normalize=True).iloc[0]),
    "n_industry": g.company_type.nunique(), "n_city": g.city.nunique(),
    "tier1_share": g.city_group.apply(lambda s: (s == "一线型").mean()),
}).loc[titles]
rows_by_title = [tidx[tid == i] for i in range(T)]
sht = split_half(rows_by_title, 20)
tf["split_half_noise"] = sht["综合"]
tf["family_label"] = tf.job_family.map(lab)
tf["mean_distance"] = DT["综合"].sum(1) / (T - 1)
tf.round(4).rename_axis("job_title").to_csv(os.path.join(PROCESSED_DIR, "E2_title_features.csv"), encoding="utf-8-sig")
print("族纯度中位数:", round(tf.family_purity.median(), 3), " 覆盖族数:", tf.job_family.nunique())
nb = pd.cut(tf.n, [29, 49, 99, 199, 1e9], labels=["30-49", "50-99", "100-199", ">=200"])
print("职位对半拆分基准（综合，按记录数分组的中位数）:\n", tf.groupby(nb, observed=True).split_half_noise.agg(["size", "median"]).round(3).to_string())
print("  各维度基准中位数:", {k: round(float(np.median(v)), 3) for k, v in sht.items()})

ju = np.triu_indices(T, 1)
same = (tf.job_family.values[:, None] == tf.job_family.values[None, :])[ju]


def compare(mask=None, tag="全部"):
    out = {}
    for k in DIMS + ["综合"]:
        v = DT[k][ju]; s = same
        if mask is not None:
            v, s = v[mask], s[mask]
        a, b = v[s], v[~s]
        auc = mannwhitneyu(b, a).statistic / (len(a) * len(b))
        dcoh = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
        out[k] = {"同族中位数": np.median(a), "异族中位数": np.median(b), "AUC": auc, "Cohen_d": dcoh, "同族对数": len(a), "异族对数": len(b)}
    print(f"同族 vs 异族（{tag}）:\n", pd.DataFrame(out).T.round(3).to_string())
    return pd.DataFrame(out).T


cmp_all = compare()
big = tf.n.values >= 100
cmp_big = compare((big[:, None] & big[None, :])[ju], f"仅记录数>=100 的 {int(big.sum())} 种职位")
# 最近邻是否同族
Dn = DT["综合"].copy(); np.fill_diagonal(Dn, np.inf)
nn = Dn.argmin(1)
fam_share = tf.job_family.value_counts(normalize=True)
print(f"最近邻职位同族的比例 {np.mean(tf.job_family.values[nn] == tf.job_family.values):.1%}；随机期望 {(fam_share ** 2).sum():.1%}")

# 案例：同族、门槛分布接近、薪酬差异大的职位对（两者记录数都 >= 100）
sal_med = tf.salary_median.values
cand = pd.DataFrame({"a": ju[0], "b": ju[1], "综合": DT["综合"][ju], **{k: DT[k][ju] for k in DIMS}})
cand = cand[same & (big[:, None] & big[None, :])[ju]]
case = cand[(cand["经验"] < 0.15) & (cand["学历"] < 0.15)].sort_values("薪酬", ascending=False).head(6)
case2 = cand[(cand["薪酬"] < 0.12)].sort_values("学历", ascending=False).head(4)
cc = ["n", "job_family", "salary_median", "salary_p25", "salary_p75", "exp_mode", "exp_any_share", "edu_mode", "edu_any_share", "bachelor_share", "top_industry", "top_industry_share"]
for name, tab in (("同族·门槛相同·薪酬差异大", case), ("同族·薪酬相同·学历差异大", case2)):
    print(f"\n案例 {name}:")
    for _, r in tab.iterrows():
        a, b = titles[int(r.a)], titles[int(r.b)]
        print(f"  {a[:8]} vs {b[:8]} 综合 {r['综合']:.3f} " + " ".join(f"{k}{r[k]:.3f}" for k in DIMS))
        print(tf.loc[[a, b], cc].round(3).to_string(header=False))

# ---------------------------------------------------------------- 5. MDS 布局
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=20, max_iter=1000)
xy = mds.fit_transform(DF["综合"])
d2 = squareform(cdist(xy, xy), checks=False); d0 = squareform(DF["综合"], checks=False)
print(f"\n== MDS == stress-1 {np.sqrt(((d2 - d0) ** 2).sum() / (d0 ** 2).sum()):.3f}；二维距离与原差异度的秩相关 {spearmanr(d2, d0)[0]:.3f}")
sal = prof.loc[fams, "salary_median"].values
bach = prof.loc[fams, "bachelor_share"].values
for i in (0, 1):
    print(f"  MDS 轴 {i + 1} 与 薪酬中位数/本科占比/Top1行业占比 的秩相关:", round(spearmanr(xy[:, i], sal)[0], 2), round(spearmanr(xy[:, i], bach)[0], 2),
          round(spearmanr(xy[:, i], prof.loc[fams, 'top_industry_share'])[0], 2))
Z = linkage(squareform(DF["综合"], checks=False), "average")
order = leaves_list(Z)
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
print("  层次聚类切成 k 类的轮廓系数:", {k: round(silhouette_score(DF["综合"], fcluster(Z, k, "maxclust"), metric="precomputed"), 3) for k in range(2, 9)})
print("  k=2、3 时各类的族数:", np.bincount(fcluster(Z, 2, "maxclust"))[1:].tolist(), np.bincount(fcluster(Z, 3, "maxclust"))[1:].tolist())
c2 = fcluster(Z, 2, "maxclust"); c3 = fcluster(Z, 3, "maxclust")
print("  k=2 的小类:", [fams[i] for i in np.flatnonzero(c2 == np.bincount(c2)[1:].argmin() + 1)], " k=3 各类:", {int(k): [fams[i] for i in np.flatnonzero(c3 == k)] for k in np.unique(c3) if (c3 == k).sum() < 10})

# ================================================================ 图 1：51×51 综合差异度热力图
fig, ax = plt.subplots(figsize=(8.6, 6.5), constrained_layout=True)
M = DF["综合"][np.ix_(order, order)].copy()
np.fill_diagonal(M, np.nan)
cmap = seq_cmap(); cmap.set_bad("#ecebe6")
im = ax.imshow(M, cmap=cmap, vmin=np.nanmin(M), vmax=np.nanmax(M))
tick = [fams[i] for i in order]
ax.set_xticks(range(F), tick, rotation=90, fontsize=6.5); ax.set_yticks(range(F), tick, fontsize=6.5)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("综合差异度"); cb.outline.set_visible(False)
ax.set_xlabel("职位族"); ax.set_ylabel("职位族")
save_fig(fig, "output/figs/E2_family_heatmap.png")

# ================================================================ 图 2：各维度差异度分布 + 噪声基准
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
od = summ.loc[DIMS[:4], "mean"].sort_values(ascending=False).index.tolist() + ["地域", "综合"]
bp = ax.boxplot([pairs[k] for k in od], vert=False, widths=0.6, patch_artist=True, whis=(5, 95), showfliers=False,
                boxprops=dict(linewidth=0), medianprops=dict(color="white", lw=1.6), whiskerprops=dict(color=MUTED, lw=1), capprops=dict(color=MUTED, lw=1))
for patch, k in zip(bp["boxes"], od):
    patch.set_facecolor(NEUTRAL if k in ("综合", "地域") else PALETTE[0])
ax.scatter([base_f[k] for k in od], range(1, len(od) + 1), marker="D", s=34, color=HIGHLIGHT, zorder=3, label="噪声基准")
ax.set_yticks(range(1, len(od) + 1), od); ax.invert_yaxis()
ax.set_xlabel("职位族对的差异度")
ax.set_xlim(0, None); ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
ax = axes[1]
ax.plot(curve.m, curve.between_med, marker="o", color=PALETTE[0], label="不同职位族")
ax.plot(curve.m, curve.within_med, marker="o", color=HIGHLIGHT, label="同一职位族（噪声基准）")
ax.set_xscale("log"); ax.set_xticks(M_LIST, [str(m) for m in M_LIST]); ax.minorticks_off()
ax.set_ylim(0, None)
ax.set_xlabel("每个对象的记录数（条，对数轴）"); ax.set_ylabel("综合差异度")
ax.legend(loc="upper right")
save_fig(fig, "output/figs/E2_dimension_noise.png")

# ================================================================ 图 3：高频职位 同族 vs 异族
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
v = DT["综合"][ju]
bins = np.linspace(0, v.max(), 61)
ax.hist(v[~same], bins=bins, density=True, color=NEUTRAL, label="异族职位对")
ax.hist(v[same], bins=bins, density=True, color=PALETTE[0], alpha=0.8, label="同族职位对")
ax.set_xlabel("综合差异度"); ax.set_ylabel("密度"); ax.legend(loc="upper right")
ax = axes[1]
od2 = cmp_all.loc[DIMS].sort_values("AUC").index.tolist()
y = np.arange(len(od2))
ax.hlines(y, cmp_all.loc[od2, "同族中位数"], cmp_all.loc[od2, "异族中位数"], color=NEUTRAL, lw=2, zorder=1)
ax.scatter(cmp_all.loc[od2, "异族中位数"], y, s=46, color=INK_2, zorder=3, label="异族中位数")
ax.scatter(cmp_all.loc[od2, "同族中位数"], y, s=46, color=PALETTE[0], zorder=3, label="同族中位数")
ax.set_yticks(y, od2)
ax.set_xlabel("分维度差异度的中位数"); ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
save_fig(fig, "output/figs/E2_title_same_family.png")

# ================================================================ 图 4：MDS 布局
fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
sc = ax.scatter(xy[:, 0], xy[:, 1], c=sal, cmap=seq_cmap(), s=70, vmin=sal.min(), vmax=sal.max(),
                edgecolor=MUTED, linewidth=0.5, zorder=3)
# 只标注正文点名的两端各 3 个职位族（编号）
pick = {"J03": (-8, 4, "right"), "J50": (-4, -14, "right"), "J44": (8, -4, "left"),
        "J09": (8, -12, "left"), "J51": (-8, 4, "right"), "J40": (0, -16, "center")}
for i, fcode in enumerate(fams):
    if fcode in pick:
        dx, dy, ha = pick[fcode]
        ax.annotate(fcode, (xy[i, 0], xy[i, 1]), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK, ha=ha)
cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02); cb.set_label("职位族薪酬中位数（千元/月）"); cb.outline.set_visible(False)
ax.set_xlabel("MDS 维度 1"); ax.set_ylabel("MDS 维度 2"); ax.set_aspect("equal", adjustable="datalim")
save_fig(fig, "output/figs/E2_family_mds.png")
print("\n图已保存。")
