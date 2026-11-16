# -*- coding: utf-8 -*-
"""问题 5（下）：新兴/急需职位族的合成得分、稳健性检查、与朴素指标的对比、代表性高频职位。

先运行 A2q5_industry.py（生成 A2q5_family_indicators.csv）。
输出
    data/processed/A2q5_family_score.csv        51 个职位族的分量 z 值、得分、排名、稳健性
    data/processed/A2q5_emerging_titles.csv     排名靠前职位族内的代表性高频职位
    data/processed/A2q5_score_checks.csv        稳健性与对比的关键数字
    output/figs/A2q5_family_score.png           Top 12 职位族的分量热力图 + Top 10 入选频率
    output/figs/A2q5_naive_compare.png          合成得分排名 vs 朴素指标排名
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *
from scipy import stats

plt = apply_style()
TOPK = 10
N_DRAW = 500

F = pd.read_csv(os.path.join(PROCESSED_DIR, "A2q5_family_indicators.csv"), index_col=0)

# ---------------------------------------------------------------------------
# 1) 合成得分：5 个分量先标准化（z 值）
#    新兴得分 = (溢价 + 本科及以上占比 + 应届友好度 + 一线集中度) / 4
#    急需得分 = (溢价 + 招聘企业数) / 2
#    合成得分 = (新兴得分 + 急需得分) / 2，即权重 0.375 / 0.125 / 0.125 / 0.125 / 0.25
# ---------------------------------------------------------------------------
COMP = {
    "薪酬溢价": F["resid_mean"],
    "本科及以上占比": F["bachelor_share"],
    "应届友好度": F["fresh_share"],
    "一线集中度": F["tier1_lq"],
    "招聘企业数": np.log(F["n_company"]),
}
zs = lambda v: (v - v.mean()) / v.std()
Z = pd.DataFrame({k: zs(v) for k, v in COMP.items()})
EMERGE, URGENT = list(Z.columns[:4]), ["薪酬溢价", "招聘企业数"]
s_emerge, s_urgent = Z[EMERGE].mean(axis=1), Z[URGENT].mean(axis=1)
score = (s_emerge + s_urgent) / 2
WEIGHT = np.array([0.375, 0.125, 0.125, 0.125, 0.25])
assert np.allclose(score, Z.values @ WEIGHT)
rank = score.rank(ascending=False).astype(int)
top = score.nlargest(TOPK).index

# 稳健性 (a)：权重围绕基准权重随机扰动，Dirichlet(10·w) 抽 500 次，统计进入 Top 10 的频率
rng = np.random.default_rng(0)
W = rng.dirichlet(10 * WEIGHT, N_DRAW)
print("扰动后各权重的标准差:", W.std(axis=0).round(3))
S = Z.values @ W.T                                  # 51 × 500
in_top = (np.argsort(np.argsort(-S, axis=0), axis=0) < TOPK).mean(axis=1)
freq = pd.Series(in_top, index=Z.index)

# 批次变化（H4）经检验是查询段差异而非时间变化，不构造该分量

# 朴素指标：记录数 × 平均薪酬
naive = F["n"] * F["salary_mean"]
rank_naive = naive.rank(ascending=False).astype(int)
top_naive = naive.nlargest(TOPK).index
rho_naive, p_naive = stats.spearmanr(score, naive)

out = Z.round(3).copy()
out.insert(0, "family_label", F["family_label"])
out["score_emerge"], out["score_urgent"] = s_emerge.round(3), s_urgent.round(3)
out["score"], out["rank"] = score.round(3), rank
out["top10_freq"] = freq.round(3)
out["naive"], out["rank_naive"] = naive.round(0), rank_naive
for c in ["n", "n_company", "n_city", "main_segment", "main_segment_share", "premium", "bachelor_share",
          "senior_share", "fresh_share", "tier1_share", "tier1_expected", "tier1_lq", "bonus_share", "salary_median"]:
    out[c] = F[c].round(4) if F[c].dtype != object else F[c]
out.sort_values("rank").to_csv(os.path.join(PROCESSED_DIR, "A2q5_family_score.csv"), encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.sort_values("rank").head(15)[["family_label", "score_emerge", "score_urgent", "score", "top10_freq", "rank_naive", "premium",
                                         "bachelor_share", "fresh_share", "tier1_share", "tier1_expected", "tier1_lq", "n_company",
                                         "n_city", "main_segment", "salary_median"]].to_string())
print("Top10:", list(top))
print("入选频率 >= 0.8 的职位族:", list(freq[freq >= 0.8].sort_values(ascending=False).index),
      " Top10 中频率最低:", freq[top].min().round(3))
print(f"朴素指标：秩相关 {rho_naive:.3f} (p={p_naive:.3f})，Top10 重合 {len(set(top) & set(top_naive))}")
g = lambda idx, c: F.loc[list(idx), c].median()
only_s, only_n = [i for i in top if i not in top_naive], [i for i in top_naive if i not in top]
print("仅合成得分 Top10:", only_s, " 仅朴素 Top10:", only_n)
for name, idx in (("合成 Top10", top), ("朴素 Top10", top_naive)):
    print(name, {c: round(g(idx, c), 3) for c in ["n", "premium", "bachelor_share", "fresh_share", "tier1_lq", "salary_median"]})
print("朴素指标与记录数的秩相关:", stats.spearmanr(naive, F["n"])[0].round(3))
pd.Series({
    "top10": " ".join(top), "n_freq_ge_0.8": int((freq >= 0.8).sum()), "min_freq_in_top10": freq[top].min(),
    "spearman_naive": rho_naive, "top10_overlap_naive": len(set(top) & set(top_naive)),
    "naive_top10_median_n": g(top_naive, "n"), "score_top10_median_n": g(top, "n"),
    "naive_top10_median_premium": g(top_naive, "premium"), "score_top10_median_premium": g(top, "premium"),
}).to_csv(os.path.join(PROCESSED_DIR, "A2q5_score_checks.csv"), header=["value"], encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 2) 排名靠前职位族内的代表性高频职位（职位编码取前 8 位）
# ---------------------------------------------------------------------------
df = load_enriched()
fam = df[df["batch"].isin(["B", "C"]) & df["city_group"].isin(["一线型", "新一线型"]) & df["job_family"].notna()]
mode = lambda s: s.value_counts().index[0]
rows = []
for j in score.nlargest(5).index:
    d = fam[fam["job_family"] == j]
    for t in d["job_title"].value_counts().head(2).index:
        s = d[d["job_title"] == t]
        rows.append({
            "job_family": j, "family_label": F.loc[j, "family_label"], "title": t[:8], "n": len(s),
            "n_company": s["company"].nunique(), "n_city": s["city"].nunique(),
            "salary_median": s["salary_mid_k"].median(),
            "edu_mode": mode(s["edu_label"]), "edu_mode_share": round((s["edu_label"] == mode(s["edu_label"])).mean(), 3),
            "exp_mode": mode(s["exp_label"]), "exp_mode_share": round((s["exp_label"] == mode(s["exp_label"])).mean(), 3),
            "bachelor_share": round((s["edu_level"] >= 4).mean(), 3),
        })
titles = pd.DataFrame(rows)
titles.to_csv(os.path.join(PROCESSED_DIR, "A2q5_emerging_titles.csv"), index=False, encoding="utf-8-sig")
print(titles.to_string())

# ---------------------------------------------------------------------------
# 图 3：左 Top 12 职位族的分量 z 值热力图；右 权重扰动下进入 Top 10 的频率
# ---------------------------------------------------------------------------
show = score.nlargest(12).index
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
M = Z.loc[show]
im = a1.imshow(M.values, cmap=div_cmap(), vmin=-3, vmax=3, aspect="auto")
a1.set_xticks(range(M.shape[1]), [c.replace("及以上", "+") for c in M.columns], fontsize=9)
a1.set_yticks(range(len(show)), [F.loc[j, "family_label"] for j in show])
a1.xaxis.tick_top()
a1.grid(False)
a1.set_xticks(np.arange(-.5, M.shape[1]), minor=True); a1.set_yticks(np.arange(-.5, len(show)), minor=True)
a1.grid(which="minor", color=SURFACE, linewidth=2); a1.tick_params(which="both", length=0)
for s in a1.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=a1, location="bottom", shrink=0.6, pad=0.02, aspect=30)
cb.set_label("分量的标准化值（z）")
cb.outline.set_visible(False)

yy = np.arange(len(show))[::-1]
a2.barh(yy, freq[show] * 100, color=[HIGHLIGHT if freq[j] >= 0.8 else NEUTRAL for j in show], height=0.62)
a2.axvline(80, color=AXIS, lw=0.8, ls=(0, (4, 3)))
a2.set_yticks(yy, [j for j in show])
a2.set_xlim(0, 100)
a2.set_xlabel(f"权重扰动 {N_DRAW} 次中进入 Top {TOPK} 的频率（%）")
a2.grid(axis="y", visible=False)
save_fig(fig, "output/figs/A2q5_family_score.png")

# ---------------------------------------------------------------------------
# 图 4：合成得分排名 vs 朴素指标排名
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
cat = pd.Series("其余", index=F.index)
cat[list(top)] = "仅合成得分 Top 10"
cat[list(top_naive)] = "仅朴素指标 Top 10"
cat[list(set(top) & set(top_naive))] = "两者均入选"
COL = {"其余": NEUTRAL, "仅合成得分 Top 10": HIGHLIGHT, "仅朴素指标 Top 10": PALETTE[0], "两者均入选": PALETTE[2]}
ax.axvspan(0.5, TOPK + 0.5, color=GRID, alpha=0.45, lw=0, zorder=0)
ax.axhspan(0.5, TOPK + 0.5, color=GRID, alpha=0.45, lw=0, zorder=0)
for c in COL:
    s = cat[cat == c].index
    ax.scatter(rank_naive[s], rank[s], s=18 + F.loc[s, "n"] / 45, color=COL[c], alpha=0.9,
               edgecolor=SURFACE, linewidth=0.8, zorder=3)
for j in cat[cat != "其余"].index:   # 标签按点的半径向右偏移，避免压住圆点
    off = (-4, 9) if j == "J03" else (np.sqrt(18 + F.loc[j, "n"] / 45) / 2 + 3, -3)
    ax.annotate(j, (rank_naive[j], rank[j]), xytext=off, textcoords="offset points", fontsize=8.5, color=INK_2)
ax.set_xlim(0, 52.5); ax.set_ylim(52.5, 0)
ax.set_xlabel("朴素指标“记录数 × 平均薪酬”的排名")
ax.set_ylabel("合成得分的排名")
from matplotlib.lines import Line2D
hd = [Line2D([], [], marker="o", ls="", ms=8, color=COL[c]) for c in COL]
ax.legend(hd, [f"{c}（{(cat == c).sum()}）" for c in COL], loc="upper left", bbox_to_anchor=(1.01, 1.0),
          title="职位族（个数）", alignment="left")
save_fig(fig, "output/figs/A2q5_naive_compare.png")
