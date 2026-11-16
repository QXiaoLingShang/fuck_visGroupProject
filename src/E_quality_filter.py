# -*- coding: utf-8 -*-
"""成员E · 第一部分：数据质量问题识别与过滤。

基于 load_flagged()：核实常规检查、统计 7 条规则命中、规则重叠、阈值敏感性、
过滤前后对比。所有写进小节稿的数字都由本脚本打印。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *   # noqa
from matplotlib.colors import LogNorm

plt = apply_style()
pd.set_option("display.width", 250)

raw = load_raw()
f = load_flagged()
c = load_clean()
COLS = list(COLUMNS_CN)
FLAGS = list(FLAG_COLUMNS)
SHORT = {  # 图中使用的规则简称
    "flag_duplicate": "完全重复",
    "flag_salary_unparsed": "薪酬无法折算",
    "flag_salary_range": "薪酬数值异常",
    "flag_salary_ratio": "区间过宽",
    "flag_rare_experience": "稀有经验编码",
    "flag_rare_education": "稀有学历编码",
    "flag_rare_industry": "稀有行业",
}
N = len(f)

# ---------------------------------------------------------------- 1. 常规检查
print("== 常规检查 ==")
print("行列:", raw.shape, "空值:", int(raw.isna().sum().sum()))
print("首尾空格:", {k: int((raw[k] != raw[k].str.strip()).sum()) for k in COLS})
print("空字符串:", {k: int((raw[k].str.len() == 0).sum()) for k in COLS})
PATTERNS = {
    "job_title": r"^[0-9a-f]{32}[A-Za-z]{2}$", "city": r"^[A-Z]\d{3}$",
    "experience": r"^E[A-Za-z]{2}$", "education": r"^G[A-Za-z]$",
    "company": r"^company_\d{6}$", "company_type": r"^type_[A-Za-z]{6}$",
}
for k, pat in PATTERNS.items():
    print(f"  {k:13s} 符合 {pat:28s}: {int(raw[k].str.match(pat).sum()):,} / {N:,}")
print("薪酬写法:", f["salary_form"].value_counts().to_dict())

# ---------------------------------------------------------------- 2. 规则命中
print("\n== 规则命中 ==")
hits = f[FLAGS].sum().astype(int)
for k in FLAGS:
    print(f"  {SHORT[k]:8s} {hits[k]:5d}  {hits[k] / N:.4%}")
n_any = int(f["flag_any"].sum())
print(f"  去重合计 {n_any}  {n_any / N:.4%}；保留 {len(c):,}（{len(c) / N:.4%}）")
nflag = f[FLAGS].sum(axis=1)
print("  每条记录命中规则数:", nflag.value_counts().sort_index().to_dict())

# 典型样例
print("\n== 典型样例 ==")
print("无法折算:", f.loc[f.flag_salary_unparsed, "salary_form"].value_counts().to_dict())
r = f[f.flag_salary_range]
print("数值异常 Top:", r["salary"].value_counts().head(12).to_dict())
print("  中值<1K:", int((r.salary_mid_k < 1).sum()), " 中值>100K:", int((r.salary_mid_k > 100).sum()),
      " 下限>上限:", int((r.salary_min_k > r.salary_max_k).sum()), " 非正:", int((r.salary_min_k <= 0).sum()))
q = f[f.flag_salary_ratio]
print("区间过宽 Top:", q["salary"].value_counts().head(8).to_dict())
print("  按写法:", q["salary_form"].value_counts().to_dict())
print("经验编码:", f["experience"].value_counts().to_dict())
print("学历编码:", f["education"].value_counts().to_dict())
vc_ind = f["company_type"].value_counts()
print("稀有行业:", vc_ind[vc_ind < RARE_INDUSTRY_MIN].to_dict())

# ---------------------------------------------------------------- 3. 阈值敏感性
print("\n== 阈值敏感性 ==")
ok = ~f["flag_salary_unparsed"]
ratio_thr = [2, 2.5, 3, 4, 5, 6, 8, 10]
ratio_hit = [int(((f.salary_ratio > t) & ok).sum()) for t in ratio_thr]
max_thr = [30, 50, 80, 100, 150, 200]
max_hit = [int((f.salary_mid_k > t).sum()) for t in max_thr]
min_thr = [0.5, 1, 1.5, 2]
min_hit = [int((f.salary_mid_k < t).sum()) for t in min_thr]
ind_thr = [10, 20, 30, 50, 100]
ind_hit = [(int((vc_ind < t).sum()), int(vc_ind[vc_ind < t].sum())) for t in ind_thr]
print("ratio >", dict(zip(ratio_thr, ratio_hit)))
print("月薪中值 >", dict(zip(max_thr, max_hit)))
print("月薪中值 <", dict(zip(min_thr, min_hit)))
print("行业阈值 -> (行业数, 记录数)", dict(zip(ind_thr, ind_hit)))
rr = f.loc[ok, "salary_ratio"].round(2)
print("ratio 恰为 3 / 3.5 / 4 / 5 / 6 的条数:", {v: int((rr == v).sum()) for v in (3, 3.5, 4, 5, 6)})

# ---------------------------------------------------------------- 4. 过滤影响
print("\n== 过滤前后 ==")
for k in COLS:
    print(f"  {COLUMNS_CN[k]:5s} {f[k].nunique():7,d} -> {c[k].nunique():7,d}")
cv = f["city"].value_counts()
cc = c["city"].value_counts().reindex(cv.index).fillna(0)
loss = 1 - cc / cv
print("  被整体删除的城市数:", int((cc == 0).sum()), " 城市最大损失率:", f"{loss.max():.2%}", loss.idxmax(), int(cv[loss.idxmax()]))
print("  核心14城最大损失率:", f"{loss[cv > 5000].max():.3%}")
QS = [0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
qt = pd.DataFrame({"前": f["salary_mid_k"].quantile(QS), "后": c["salary_mid_k"].quantile(QS)}).round(2)
print(qt.T.to_string())
print("  均值 前/后:", round(f.salary_mid_k.mean(), 3), round(c.salary_mid_k.mean(), 3),
      " 最小 前/后:", round(f.salary_mid_k.min(), 3), round(c.salary_mid_k.min(), 3))
for k, order in (("experience", EXP_ORDER), ("education", EDU_ORDER)):
    d = (c[k].value_counts(normalize=True) - f[k].value_counts(normalize=True)).reindex(order).abs().max()
    print(f"  {k} 各编码占比最大变化: {d * 100:.3f} 个百分点")

# 遗留问题
print("\n== 遗留问题 ==")
multi = f.groupby("company")["company_type"].nunique()
print("同一企业对应多个行业类别:", int((multi > 1).sum()), "/", multi.size, f"{(multi > 1).mean():.2%}")
print("各写法清洗后月薪中位数:", c.groupby("salary_form")["salary_mid_k"].median().round(2).to_dict())
print("招聘量在 290~310 条的城市数:", int(((cv >= 290) & (cv <= 310)).sum()), "/", cv.size)

# ================================================================ 图 1：命中条数
order = hits.sort_values().index
fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
ax.barh([SHORT[k] for k in order], hits[order].values, color=PALETTE[0], height=0.62)
for i, k in enumerate(order):
    ax.text(hits[k] + 12, i, f"{hits[k]:,}（{hits[k] / N:.3%}）", va="center", fontsize=9.5, color=INK_2)
ax.set_xlim(0, hits.max() * 1.28)
ax.set_xlabel("命中记录数（条）")
ax.set_title(f"7 条规则共标记 {n_any:,} 条（{n_any / N:.2%}），区间过宽占一半以上")
ax.grid(axis="y", visible=False)
save_fig(fig, "output/figs/E_rule_hits.png")

# ================================================================ 图 2：规则共现矩阵
M = f[FLAGS].astype(int)
co = (M.T @ M).values
labels = [SHORT[k] for k in FLAGS]
fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
off = np.where(np.eye(len(FLAGS), dtype=bool), np.nan, co).astype(float)
off[off == 0] = np.nan
cmap = seq_cmap(); cmap.set_bad(SURFACE)
im = ax.imshow(off, cmap=cmap, vmin=0, vmax=np.nanmax(off))
for i in range(len(FLAGS)):
    for j in range(len(FLAGS)):
        v = co[i, j]
        if i == j:   # 对角线：该规则自身命中数，灰底
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#ecebe6"))
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
        else:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    color=(MUTED if v == 0 else ("white" if v > np.nanmax(off) * .55 else INK)))
ax.set_xticks(range(len(FLAGS)), labels, rotation=30, ha="right")
ax.set_yticks(range(len(FLAGS)), labels)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"规则之间几乎不重叠：两两共现最多 {int(np.nanmax(off))} 条（对角线为自身命中数）")
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("同时命中两条规则的记录数（条）")
cb.outline.set_visible(False)
save_fig(fig, "output/figs/E_rule_overlap.png")
print("\n共现矩阵:\n", pd.DataFrame(co, index=labels, columns=labels).to_string())

# ================================================================ 图 3：阈值敏感性
from matplotlib.patches import Patch
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), constrained_layout=True)
specs = [
    (axes[0], ratio_thr, ratio_hit, SALARY_RATIO_MAX, "上限/下限 倍数阈值"),
    (axes[1], max_thr, max_hit, SALARY_MAX_K, "月薪中值上限（千元/月）"),
    (axes[2], min_thr, min_hit, SALARY_MIN_K, "月薪中值下限（千元/月）"),
]
for ax, xs, ys, cur, xl in specs:
    cols = [HIGHLIGHT if x == cur else NEUTRAL for x in xs]
    ax.bar([f"{x:g}" for x in xs], ys, color=cols, width=0.68)
    for i, y in enumerate(ys):
        ax.text(i, y * 1.12, f"{y:,}", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_yscale("log")
    ax.set_ylim(top=max(ys) * 4)
    ax.set_xlabel(xl)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("将被剔除的记录数（条，对数轴）")
axes[2].legend(handles=[Patch(color=HIGHLIGHT, label="采用的阈值"), Patch(color=NEUTRAL, label="其他取值")], loc="upper left")
save_fig(fig, "output/figs/E_threshold_sensitivity.png")

# ================================================================ 图 4：过滤前后月薪分布
bins = np.logspace(np.log10(0.1), np.log10(300), 61)
fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
ax.hist(f["salary_mid_k"].dropna(), bins=bins, color=NEUTRAL, label="过滤前")
ax.hist(c["salary_mid_k"], bins=bins, color=PALETTE[0], label="过滤后")
ax.set_xscale("log"); ax.set_yscale("log")
for i, x in enumerate((SALARY_MIN_K, SALARY_MAX_K)):
    ax.axvline(x, color=HIGHLIGHT, lw=1.2, label="月薪阈值（1K、100K）" if i == 0 else None)
ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.set_xlabel("折算月薪中值（千元/月，对数轴）")
ax.set_ylabel("记录数（条，对数轴）")
ax.set_ylim(0.7, 6e5)
ax.legend(loc="upper left")
save_fig(fig, "output/figs/E_filter_before_after.png")
print("\n图已保存。")
