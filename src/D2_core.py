# -*- coding: utf-8 -*-
"""成员D · 第二轮 · 核心城市口径：14 个核心城市的地域特征与相似地域（问题 4）。

口径：load_enriched() 中 14 个核心城市的批次 B、C（有职位族的记录）。
步骤：1) 复核“三个查询段、每城只覆盖两段”的采集结构；2) 在同口径下计算职位族厚度、行业偏好、门槛/薪酬结构；
      3) 两两相似度 + 层次聚类 + 自助抽样；4) 职位族区位熵热力图。
产出：output/figs/D2_core_family_lq.png、D2_core_similarity.png
      data/processed/D2_core_segments.csv、D2_core_pref.csv、D2_core_similarity.csv、D2_core_family_lq.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D2_lib import *
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

plt = apply_style()
df = load_d2()
T1 = TIER1_CITIES
BC = df[df["seg"].notna()].copy()
LAB = family_labels()
N_BOOT = 200

# ---------------------------------------------------------------------------
# 1. 采集结构复核：28 个“城市 × 批次”块按锚点职位集合聚成 3 个查询段
# ---------------------------------------------------------------------------
BC["blk"] = BC["city"] + "." + BC["batch"]
blks = [c + "." + b for c in CORE for b in "BC"]
vc = BC["job_title"].value_counts()
m = BC[BC["job_title"].isin(vc[vc >= 5].index)]
pres = (pd.crosstab(m["blk"], m["job_title"]).loc[blks] > 0).values.astype(float)
inter = pres @ pres.T
jac = inter / (pres.sum(1)[:, None] + pres.sum(1)[None, :] - inter)
lab3 = fcluster(linkage(pres, "average", metric="jaccard"), 3, "maxclust")
seg_of_blk = BC.groupby("blk")["seg"].first().loc[blks].values
print("锚点职位集合层次聚类（k=3）与查询段标注的列联表:\n", pd.crosstab(lab3, seg_of_blk).to_string())
iu28 = np.triu_indices(28, 1)
same_city = np.array([blks[a][:4] == blks[b][:4] for a, b in zip(*iu28)])
same_seg = seg_of_blk[iu28[0]] == seg_of_blk[iu28[1]]
jv = jac[iu28]
print(f"锚点职位 Jaccard：同城两批次 {jv[same_city].mean():.3f} | 异城同段 {jv[same_seg & ~same_city].mean():.3f} | 异城异段 {jv[~same_seg & ~same_city].mean():.3f}")
segtab = BC.groupby(["crawl_group", "batch"]).agg(seg=("seg", "first"), n=("city", "size")).reset_index()
segtab["cities"] = segtab["crawl_group"].map(lambda g: "、".join(CRAWL_GROUP[g]))
segtab.to_csv(os.path.join(PROCESSED_DIR, "D2_core_segments.csv"), index=False, encoding="utf-8-sig")
print(segtab.to_string(index=False))

R = BC[BC["job_family"].notna()].copy()
print("核心城市批次记录", len(BC), "其中有职位族", len(R), f"{len(R) / len(BC):.2%}")


def r2_between(S, labels):
    """城市间方差中由分组解释的比例（份额开方后）。"""
    labels = np.array(labels)
    tot = ((S - S.mean(0)) ** 2).sum()
    within = sum(((S[labels == u] - S[labels == u].mean(0)) ** 2).sum() for u in set(labels))
    return 1 - within / tot


cf = pd.crosstab(R["city"], R["job_family"]).loc[CORE]
S_ = np.sqrt(cf.div(cf.sum(1), axis=0).values)
print(f"不做校正时，城市×职位族份额的城市间方差：采集组解释 {r2_between(S_, [CITY2CG[c] for c in CORE]):.1%}，"
      f"一线型/新一线型解释 {r2_between(S_, [c in T1 for c in CORE]):.1%}")
ci_ = pd.crosstab(BC["city"], BC["company_type"]).loc[CORE]
S_ = np.sqrt(ci_.div(ci_.sum(1), axis=0).values)
print(f"城市×行业份额：采集组解释 {r2_between(S_, [CITY2CG[c] for c in CORE]):.1%}，一线型/新一线型解释 {r2_between(S_, [c in T1 for c in CORE]):.1%}")
# H2 在查询段内是否成立：各城市在两个查询段上的薪酬均值
sm = BC.groupby(["city", "seg"])["salary_mid_k"].mean().unstack().loc[CORE]
print("各城两段薪酬均值之差的最大绝对值:", round((sm.max(1) - sm.min(1)).max(), 2))
print("查询段 S2（覆盖 G1、G2 共 10 城）内薪酬均值降序:", sm["S2"].dropna().sort_values(ascending=False).round(2).to_dict())

# ---------------------------------------------------------------------------
# 2. 同口径特征
# ---------------------------------------------------------------------------
fams = sorted(R["job_family"].unique())
top_ind = R["company_type"].value_counts().index[:30].tolist()
print("前 30 个行业占有职位族记录的", f"{R['company_type'].isin(top_ind).mean():.1%}")
c_code = R["city"].map({c: i for i, c in enumerate(CORE)}).values
s_code = R["seg"].map({s: i for i, s in enumerate(SEGS)}).values
f_code = R["job_family"].map({f_: i for i, f_ in enumerate(fams)}).values
i_code = R["company_type"].map({t: i for i, t in enumerate(top_ind)}).fillna(30).astype(int).values
NC, NS, NF, NI = 14, 3, len(fams), 31
csf = (c_code * NS + s_code) * NF + f_code
STRUCT = {"薪酬(对数)": np.log(R["salary_mid_k"].values), "本科+": (R["edu_level"] >= 4).values.astype(float),
          "学历不限": R["edu_any"].values.astype(float), "经验不限": R["exp_any"].values.astype(float),
          "3年以上": (R["exp_level"] >= 4).values.astype(float), "13薪": R["has_bonus_months"].values.astype(float)}
V_ALL = np.column_stack(list(STRUCT.values()))
grp_idx = {g: [CORE.index(c) for c in cs] for g, cs in CRAWL_GROUP.items()}
base_idx = {g: [CORE.index(c) for c in cs if c not in T1] for g, cs in CRAWL_GROUP.items()}   # 同组新一线型城市
MIN_POOLED, GAP_LQ, CLIP = 0.005, 0.2, 1.5


def features(idx):
    """给定记录下标（可为重抽样），返回 职位族厚度 log2LQ[c,s,f]、合并 LQ[c,f]、行业 log2LQ[c,i]、结构[c,k]。"""
    T = np.bincount(csf[idx], minlength=NC * NS * NF).reshape(NC, NS, NF).astype(float)
    Ti = np.bincount(csf[idx] * NI + i_code[idx], minlength=NC * NS * NF * NI).reshape(NC, NS, NF, NI).astype(float)
    n_cs = T.sum(2)
    share = np.divide(T, n_cs[..., None], out=np.zeros_like(T), where=n_cs[..., None] > 0)
    # (F) 职位族厚度：相对“同采集组、同查询段内新一线型城市平均份额”的区位熵
    base, pooled = np.zeros_like(T), np.zeros_like(T)
    for g, members in grp_idx.items():
        base[members] = share[base_idx[g]].mean(0)
        pooled[members] = T[members].sum(0) / np.maximum(n_cs[members].sum(0), 1)[:, None]
    expect = n_cs[..., None] * base
    raw = np.divide(T, expect, out=np.full_like(T, np.nan), where=expect > 0)
    covered = (pooled >= MIN_POOLED) & (expect > 0)                    # 该采集组在该查询段确实采到了这个职位族
    valid = covered & (raw >= GAP_LQ)                                  # LQ<0.2 视为疑似漏采，不参与计算
    LF = np.where(valid, np.clip(np.log2((T + 0.5) / (expect + 0.5)), -CLIP, CLIP), np.nan)
    num, den = np.where(valid, T, 0).sum(1), np.where(valid, expect, 0).sum(1)
    LQ = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    features.n_gap = int((covered & (raw < GAP_LQ)).sum())
    # (I) 行业偏好：按“查询段 × 职位族”标准化后的区位熵
    pool = T.sum(0)
    p_i = np.divide(Ti.sum(0), pool[..., None], out=np.zeros((NS, NF, NI)), where=pool[..., None] > 0)
    E_i = np.einsum("csf,sfi->ci", T, p_i)
    O_i = Ti.sum((1, 2))
    LI = np.log2((O_i + 0.5) / (E_i + 0.5))
    # (S) 门槛/薪酬结构：同样按“查询段 × 职位族”标准化
    St = np.zeros((NC, V_ALL.shape[1]))
    for k in range(V_ALL.shape[1]):
        Vk = np.bincount(csf[idx], weights=V_ALL[idx, k], minlength=NC * NS * NF).reshape(NC, NS, NF)
        mean_sf = np.divide(Vk.sum(0), pool, out=np.zeros((NS, NF)), where=pool > 0)
        O, E = Vk.sum((1, 2)), (T * mean_sf).sum((1, 2))
        St[:, k] = (O - E) / T.sum((1, 2)) if k == 0 else np.log2(O / E)
    return LF, LQ, LI, St, O_i / T.sum((1, 2))[:, None], T


IU = np.triu_indices(NC, 1)


def distances(LF, LI, St):
    """三类特征的两两距离（各 91 对）。"""
    dF = np.empty(len(IU[0]))
    for k, (a, b) in enumerate(zip(*IU)):
        d = (LF[a] - LF[b]).ravel()
        dF[k] = np.sqrt(np.nanmean(d ** 2))
    dI = np.sqrt(((LI[IU[0], :30] - LI[IU[1], :30]) ** 2).mean(1))
    Z = (St - St.mean(0)) / St.std(0, ddof=1)
    dS = np.sqrt(((Z[IU[0]] - Z[IU[1]]) ** 2).mean(1))
    return dF, dI, dS


def similarity(dF, dI, dS):
    """相似度 = 1 − 距离/全部城市对的平均距离；三类特征等权平均。"""
    return ((1 - dF / dF.mean()) + (1 - dI / dI.mean()) + (1 - dS / dS.mean())) / 3


all_idx = np.arange(len(R))
LF, LQ, LI, St, ind_share, T = features(all_idx)
dF, dI, dS = distances(LF, LI, St)
sim = similarity(dF, dI, dS)
pairs = [(CORE[a], CORE[b]) for a, b in zip(*IU)]
same_cg = np.array([CITY2CG[a] == CITY2CG[b] for a, b in pairs])
n_t1 = np.array([(a in T1) + (b in T1) for a, b in pairs])

n_cell = np.isfinite(LF).sum()
print("疑似漏采（LQ<0.2）的格子数", features.n_gap)
print("职位族厚度有效格子数", int(n_cell), "；其中 |log2LQ|<0.26（与基准相差不到 20%）的比例", f"{(np.abs(LF[np.isfinite(LF)]) < 0.263).mean():.1%}")
print("结构特征（按族标准化后）:\n", pd.DataFrame(St, index=CORE, columns=list(STRUCT)).round(3).to_string())


def show(name, v):
    print(f"[{name}] 同采集组 {v[same_cg].mean():.3f} / 跨组 {v[~same_cg].mean():.3f} | 一线型内部 {v[n_t1 == 2].mean():.3f}"
          f" / 一线型-新一线型 {v[n_t1 == 1].mean():.3f} / 新一线型内部 {v[n_t1 == 0].mean():.3f}")


for nm, v in (("职位族距离", dF), ("行业距离", dI), ("结构距离", dS), ("综合相似度", sim)):
    show(nm, v)
print("三类距离两两相关: 族-行业", round(np.corrcoef(dF, dI)[0, 1], 3), "族-结构", round(np.corrcoef(dF, dS)[0, 1], 3),
      "行业-结构", round(np.corrcoef(dI, dS)[0, 1], 3))
for nm, v in (("职位族", 1 - dF / dF.mean()), ("行业", 1 - dI / dI.mean()), ("结构", 1 - dS / dS.mean())):
    print(f"   [{nm}相似度] 一线型内部 {v[n_t1 == 2].mean():+.3f} 混合 {v[n_t1 == 1].mean():+.3f} 新一线型内部 {v[n_t1 == 0].mean():+.3f}")

SIM = pd.DataFrame(squareform(sim), index=CORE, columns=CORE)
np.fill_diagonal(SIM.values, np.nan)
ps = pd.Series(sim, index=pd.MultiIndex.from_tuples(pairs)).sort_values(ascending=False)
print("最相似 10 对:\n", ps.head(10).round(3).to_string())
print("最不相似 5 对:\n", ps.tail(5).round(3).to_string())
nn = SIM.idxmax(1)
print("各城最近邻:", nn.to_dict())
Zl = linkage(sim.max() - sim, "average")
for k in (2, 3, 4):
    lab_k = pd.Series(fcluster(Zl, k, "maxclust"), index=CORE)
    print("层次聚类 k =", k, [list(v.index) for _, v in lab_k.groupby(lab_k)])
to_t1 = SIM.loc[[c for c in CORE if c not in T1], T1].mean(1).sort_values(ascending=False)
print("新一线型城市与四个一线型城市的平均相似度:", to_t1.round(3).to_dict())
hub = SIM.loc[[c for c in CORE if c not in T1], [c for c in CORE if c not in T1]].mean(1).sort_values(ascending=False)
print("新一线型城市与其余新一线型城市的平均相似度:", hub.round(3).to_dict())

# ---------------------------------------------------------------------------
# 3. 自助抽样（每个“城市 × 查询段”块内对记录有放回重抽样）
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
blk_code = c_code * NS + s_code
order = np.argsort(blk_code, kind="stable")
bounds = np.r_[0, np.cumsum(np.bincount(blk_code, minlength=NC * NS))]
boot = np.empty((N_BOOT, len(sim)))
boot_nn = np.empty((N_BOOT, NC), dtype=int)
boot_t1 = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = np.concatenate([order[rng.integers(bounds[i], bounds[i + 1], bounds[i + 1] - bounds[i])]
                          for i in range(NC * NS) if bounds[i + 1] > bounds[i]])
    f_ = features(idx)
    sb = similarity(*distances(f_[0], f_[2], f_[3]))
    boot[b] = sb
    M = squareform(sb); np.fill_diagonal(M, -np.inf)
    boot_nn[b] = M.argmax(1)
    boot_t1[b] = sb[n_t1 == 2].mean() - sb[n_t1 == 1].mean()
nn_stab = pd.Series([(boot_nn[:, i] == CORE.index(nn[c])).mean() for i, c in enumerate(CORE)], index=CORE)
print("最近邻在 200 次抽样中保持不变的比例:\n", pd.DataFrame({"最近邻": nn, "稳定度": nn_stab}).to_string())
print("稳定度 ≥0.9 的城市数", int((nn_stab >= 0.9).sum()), "中位数", nn_stab.median())
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
ci = pd.DataFrame({"sim": sim, "lo": lo, "hi": hi}, index=pd.MultiIndex.from_tuples(pairs)).sort_values("sim", ascending=False)
print("最相似 8 对的 95% 抽样区间:\n", ci.head(8).round(3).to_string())
print("抽样区间半宽中位数", round(np.median((hi - lo) / 2), 3))
print(f"一线型内部相似度 − 一线型与新一线型之间相似度：{boot_t1.mean():.3f}，95% 区间 [{np.percentile(boot_t1, 2.5):.3f}, {np.percentile(boot_t1, 97.5):.3f}]")
t1_nn_t1 = np.mean([[CORE[boot_nn[b, CORE.index(c)]] in T1 for c in T1] for b in range(N_BOOT)], axis=0)
print("四个一线型城市的最近邻仍是一线型的比例:", dict(zip(T1, t1_nn_t1.round(2))))
out = ci.reset_index().rename(columns={"level_0": "city_a", "level_1": "city_b", "sim": "similarity", "lo": "ci_low", "hi": "ci_high"})
out["d_family"], out["d_industry"], out["d_structure"] = [pd.Series(v, index=pd.MultiIndex.from_tuples(pairs)).reindex(ci.index).values for v in (dF, dI, dS)]
out["same_crawl_group"] = [CITY2CG[a] == CITY2CG[b] for a, b in zip(out.city_a, out.city_b)]
out.round(4).to_csv(os.path.join(PROCESSED_DIR, "D2_core_similarity.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 4. 各城市偏好的职位族与行业（写入画像表的核心城市部分）
# ---------------------------------------------------------------------------
LQf = pd.DataFrame(LQ, index=CORE, columns=fams)
cnt_cf = pd.DataFrame(T.sum(1), index=CORE, columns=fams)
LQi = pd.DataFrame(2 ** LI[:, :30], index=CORE, columns=[short(t) for t in top_ind])
shi = pd.DataFrame(ind_share[:, :30], index=CORE, columns=LQi.columns)
LQf.round(3).to_csv(os.path.join(PROCESSED_DIR, "D2_core_family_lq.csv"), encoding="utf-8-sig")
rows = []
for c in CORE:
    lf = LQf.loc[c][cnt_cf.loc[c] >= 100].dropna().sort_values(ascending=False).head(3)
    li = LQi.loc[c][shi.loc[c] >= 0.01].sort_values(ascending=False).head(3)
    rows.append({"city": c, "crawl_group": CITY2CG[c],
                 "pref_family": "；".join(f"{LAB[k]}({v:.2f})" for k, v in lf.items()),
                 "pref_industry": "；".join(f"{k}({v:.2f},{shi.loc[c, k]:.1%})" for k, v in li.items()),
                 "nearest_city": nn[c], "nearest_stability": nn_stab[c]})
    print(c, "| 族:", rows[-1]["pref_family"], "| 行业:", rows[-1]["pref_industry"])
pd.DataFrame(rows).to_csv(os.path.join(PROCESSED_DIR, "D2_core_pref.csv"), index=False, encoding="utf-8-sig")

# 一线型相对同组新一线型的偏好：两个采集组（G1、G2）是否一致
t1_lq = {g: np.log2(LQf.loc[[c for c in CRAWL_GROUP[g] if c in T1]]).mean() for g in ("G1", "G2")}
both = pd.DataFrame(t1_lq).dropna()
print("一线型 log2LQ（相对同组新一线型）：G1 与 G2 在", len(both), "个共有职位族上的相关", round(both.corr().iloc[0, 1], 3))
t1_mean = np.log2(LQf.loc[T1]).mean().dropna()
valid_n = np.log2(LQf.loc[T1]).notna().sum()
t1_mean = t1_mean[valid_n >= 2]
print("一线型平均 LQ 最高:", {LAB[k]: round(2 ** v, 2) for k, v in t1_mean.sort_values(ascending=False).head(8).items()})
print("一线型平均 LQ 最低:", {LAB[k]: round(2 ** v, 2) for k, v in t1_mean.sort_values().head(5).items()})
print("上述族在四个一线型城市的 LQ:\n", LQf.loc[T1, t1_mean.sort_values(ascending=False).head(8).index].round(2).to_string())
ti = np.log2(LQi.loc[T1]).mean().sort_values(ascending=False)
print("一线型行业 LQ（标准化后）最高:", (2 ** ti.head(5)).round(2).to_dict(), "最低:", (2 ** ti.tail(4)).round(2).to_dict())

# ---------------------------------------------------------------------------
# 5. 图：职位族区位熵热力图 + 相似度热力图
# ---------------------------------------------------------------------------
leaf = [CORE[i] for i in leaves_list(Zl)]
L2 = np.log2(LQf)
n_valid = L2.notna().sum()
sd = L2.std()[n_valid >= 8].sort_values(ascending=False)
sel = sd.index[:16].tolist()
sel = sorted(sel, key=lambda k: -t1_mean.get(k, 0))                    # 行按“一线型偏好程度”排序
col_order = [c for c in leaf if c in T1] + [c for c in leaf if c not in T1]
H = L2.loc[col_order, sel].T
print("热力图选入的职位族:", [LAB[k] for k in sel])
print("热力图取值范围", round(np.nanmin(H.values), 2), round(np.nanmax(H.values), 2), "缺失格", int(H.isna().sum().sum()))

fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
vmax = 1.5
cm = div_cmap().copy(); cm.set_bad("#ffffff")
im = ax.imshow(np.ma.masked_invalid(H.values), cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(col_order))); ax.set_xticklabels(col_order)
ax.set_yticks(range(len(sel))); ax.set_yticklabels([LAB[k] for k in sel])
ax.axvline(len(T1) - 0.5, color=INK_2, lw=1.2)
ax.set_xticks(np.arange(-.5, len(col_order), 1), minor=True); ax.set_yticks(np.arange(-.5, len(sel), 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=1.5); ax.grid(which="major", visible=False)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("核心城市")
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("log2(区位熵)")
save_fig(fig, "output/figs/D2_core_family_lq.png")

Sm = SIM.loc[leaf, leaf].values
fig, ax = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
vmax = float(np.nanmax(np.abs(Sm)))
cm = div_cmap().copy(); cm.set_bad(SURFACE)
im = ax.imshow(np.ma.masked_invalid(Sm), cmap=cm, vmin=-vmax, vmax=vmax)
ticks = [f"{c}*" if c in T1 else c for c in leaf]
ax.set_xticks(range(NC)); ax.set_xticklabels(ticks, rotation=45, ha="right")
ax.set_yticks(range(NC)); ax.set_yticklabels(ticks)
ax.set_xticks(np.arange(-.5, NC, 1), minor=True); ax.set_yticks(np.arange(-.5, NC, 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=1.5); ax.grid(which="major", visible=False)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("相似度")
save_fig(fig, "output/figs/D2_core_similarity.png")
print("聚类顺序:", leaf)
print("done")
