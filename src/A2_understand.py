# -*- coding: utf-8 -*-
"""成员 A（第二轮，上半）：赛题解读与分析假设 + 数据集整体概览与采集结构。

产出：
  output/figs/A2_structure_match.png    数据结构与真实平台公开配置的对照
  output/figs/A2_batch_blocks.png       原始行序上的城市块结构与三个采集批次
  output/figs/A2_within_block.png       大块内部的职位聚集与薪酬自相关
  output/figs/A2_core_batch_gap.png     核心城市批次 A 与 B+C 的差异
  data/processed/A2_structure_match.csv / A2_batch_summary.csv / A2_city_blocks.csv
  data/processed/A2_core_batch_gap.csv / A2_block_autocorr.csv / A2_title_position.csv
用法：python src/A2_understand.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
from scipy.stats import binomtest
from enrich import *   # noqa

plt = apply_style()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)


def p(*a):
    print(*a, flush=True)


def out_csv(d, name, index=False):
    d.to_csv(os.path.join(PROCESSED_DIR, name), index=index, encoding="utf-8-sig")


f = load_flagged().reset_index(drop=True)      # 未过滤、保持原始行序
f["batch"] = infer_batches(f)
df = load_enriched()                           # 清洗后 + 语义列
raw = load_raw()
tax = json.load(open(os.path.join(ROOT, "data", "reference", "platform_taxonomy.json"), encoding="utf-8"))
tier = city_tier_map(f)
CORE = sorted(tier[tier == "核心城市"].index)

# 职位的长尾：高频职位与只出现一次的职位占比
vc_title = raw["job_title"].value_counts()
p(f"== 招聘数>=10 的职位 {int((vc_title >= 10).sum()):,} 种，占职位种类 {(vc_title >= 10).mean():.2%}，"
  f"覆盖记录 {vc_title[vc_title >= 10].sum() / len(raw):.2%}；只出现 1 次的职位占 {(vc_title == 1).mean():.2%}")

# ---------------------------------------------------------------------------
# 1. 数据结构与真实平台公开配置的对照
# ---------------------------------------------------------------------------
edu_valid = [c for c in df["education"].unique() if c != "Gw"]      # Gw 按 H1 视为噪声
exp_valid = list(df["experience"].unique())
n_tier1 = df.loc[df["city_group"] == "一线型", "city"].nunique()
n_newt1 = df.loc[df["city_group"] == "新一线型", "city"].nunique()
match = pd.DataFrame([
    ("学历要求选项", len(edu_valid), "有效编码", len(tax["学历选项"]), "筛选项", None, ""),
    ("经验要求选项", len(exp_valid), "有效编码", len(tax["经验选项"]), "筛选项", None, ""),
    ("核心（热门）城市", n_tier1 + n_newt1, f"{n_tier1}+{n_newt1} 两组", len(tax["热门城市"]), "一线 4 + 其他 10", None, ""),
    ("城市总数", raw["city"].nunique(), "行政区划编码", tax["城市总数"], "城市", None, ""),
    ("职位分类", df["job_family"].nunique(), "职位族", tax["职位类目"]["一级个数"], "一级类目",
     tax["职位类目"]["二级个数"], "二级类目"),
    ("行业分类", raw["company_type"].nunique(), "行业编码", tax["行业"]["二级个数"], "二级行业", None, ""),
], columns=["结构", "本数据", "本数据口径", "平台", "平台口径", "平台2", "平台2口径"])
out_csv(match, "A2_structure_match.csv")
p("== 结构对照 ==\n", match.to_string(index=False))
p("清洗后行业编码数", df["company_type"].nunique(), "平台行业一级", tax["行业"]["一级个数"],
  "平台职位三级", tax["职位类目"]["三级个数"])

fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
n = len(match)
dy = 0.17
for i, r in enumerate(match.itertuples(index=False)):
    y = n - 1 - i
    xs = [r.本数据, r.平台] + ([r.平台2] if pd.notna(r.平台2) else [])
    ax.plot([min(xs), max(xs)], [y, y], color=NEUTRAL, lw=1, zorder=1)
    ax.scatter([r.本数据], [y + dy], s=70, color=PALETTE[0], zorder=3, label="本数据集（脱敏）" if i == 0 else None)
    ax.scatter([r.平台], [y - dy], s=70, color=PALETTE[1], zorder=3, label="真实平台公开配置" if i == 0 else None)
    # 对数轴上 371 与 373 这类差别读不出来，保留数值标签
    ax.text(r.本数据 * 1.13, y + dy, f"{r.本数据:,}", va="center", fontsize=9.5, color=INK_2)
    if pd.notna(r.平台2):       # 职位：平台的一级、二级类目各一个点
        ax.scatter([r.平台2], [y - dy], s=70, color=PALETTE[1], zorder=3)
        ax.text(r.平台 / 1.13, y - dy, f"{r.平台:,}", va="center", ha="right", fontsize=9.5, color=INK_2)
        ax.text(r.平台2 * 1.13, y - dy, f"{int(r.平台2):,}", va="center", fontsize=9.5, color=INK_2)
    elif r.平台 < r.本数据:     # 平台值较小时标在左侧，避免与连线重叠
        ax.text(r.平台 / 1.13, y - dy, f"{r.平台:,}", va="center", ha="right", fontsize=9.5, color=INK_2)
    else:
        ax.text(r.平台 * 1.13, y - dy, f"{r.平台:,}", va="center", fontsize=9.5, color=INK_2)
ax.set_xscale("log"); ax.set_xlim(3, 1500); ax.set_ylim(-0.6, n - 0.4)
ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1,000"])
ylab = {"核心（热门）城市": "核心城市 / 热门城市", "职位分类": "职位族 / 一、二级职位类目",
        "行业分类": "行业编码 / 二级行业"}
ax.set_yticks(range(n)); ax.set_yticklabels([ylab.get(s, s) for s in match["结构"][::-1]])
ax.set_xlabel("取值个数（对数轴）"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
save_fig(fig, "output/figs/A2_structure_match.png")

# ---------------------------------------------------------------------------
# 2. 原始行序上的城市块与采集批次
# ---------------------------------------------------------------------------
city = f["city"].values
start = np.r_[0, np.flatnonzero(city[1:] != city[:-1]) + 1]
end = np.r_[start[1:], len(city)]
runs = pd.DataFrame({"city": city[start], "start": start, "end": end, "batch": f["batch"].values[start]})
# 同一城市的相邻行段若只被不超过 10 行的零星记录隔开，合并为一块
blocks = []
for (b, c), g in runs.groupby(["batch", "city"], sort=False):
    cur = None
    for s, e in zip(g["start"], g["end"]):
        if cur is not None and s - cur[1] <= 10:
            cur[1] = e
        else:
            if cur is not None:
                blocks.append((b, c, cur[0], cur[1]))
            cur = [s, e]
    blocks.append((b, c, cur[0], cur[1]))
blocks = pd.DataFrame(blocks, columns=["batch", "city", "start", "end"]).sort_values("start").reset_index(drop=True)
blocks["rows"] = [int((city[s:e] == c).sum()) for c, s, e in zip(blocks["city"], blocks["start"], blocks["end"])]
out_csv(blocks, "A2_city_blocks.csv")
p2, p3 = int((f["batch"] == "A").sum()), int((f["batch"] != "C").sum())
p(f"== 批次分界：A/B 行号 {p2:,}，B/C 行号 {p3:,}；连续行段 {len(runs)} 个，合并后城市块 {len(blocks)} 个")

per_city = f.groupby(["batch", "city"]).size().rename("rows").reset_index()
a_city = per_city[per_city["batch"] == "A"].set_index("city")["rows"]
a_runs = runs[runs["batch"] == "A"].assign(rows=lambda d: d["end"] - d["start"])
p("批次 A：行段数", len(a_runs), "单行段最大的 3 个", a_runs.nlargest(3, "rows")[["city", "rows"]].values.tolist(),
  "| 每城行数：中位数", a_city.median(), "不超过300的城市", int((a_city <= 300).sum()),
  "281~300的城市", int(a_city.between(281, 300).sum()), "超过300的城市", a_city[a_city > 300].to_dict())
stray = per_city[(per_city["batch"] != "A") & ~per_city["city"].isin(CORE)]
p("批次 B、C 中的非核心城市零星记录：\n", stray.to_string(index=False), "\n合计", int(stray["rows"].sum()))

clean_b = df.groupby("batch").agg(薪酬中位数=("salary_mid_k", "median"),
                                  本科及以上占比=("edu_level", lambda s: (s >= 4).mean()))
rows = []
for b in "ABC":
    pc = per_city[per_city["batch"] == b]
    main = pc if b == "A" else pc[pc["city"].isin(CORE)]
    lo, hi = np.flatnonzero(f["batch"].values == b)[[0, -1]]
    rows.append({"批次": b, "原始行号范围": f"{lo + 1:,}~{hi + 1:,}", "行数": int(pc["rows"].sum()),
                 "占比%": round(pc["rows"].sum() / len(f) * 100, 2), "城市数": len(main),
                 "零星城市数": len(pc) - len(main), "零星记录数": int(pc["rows"].sum() - main["rows"].sum()),
                 "每城行数中位数": float(main["rows"].median()),
                 "每城行数最小": int(main["rows"].min()), "每城行数最大": int(main["rows"].max()),
                 "薪酬中位数_千元月": clean_b.loc[b, "薪酬中位数"],
                 "本科及以上占比%": round(clean_b.loc[b, "本科及以上占比"] * 100, 2)})
batch_tab = pd.DataFrame(rows)
out_csv(batch_tab, "A2_batch_summary.csv")
p("== 批次汇总 ==\n", batch_tab.to_string(index=False))
p("核心城市批次 B+C 行数", int(f[(f.batch != "A") & f.city.isin(CORE)].shape[0]),
  f"占全部 {f[(f.batch != 'A') & f.city.isin(CORE)].shape[0] / len(f):.2%}")

BC = {"A": PALETTE[0], "B": PALETTE[1], "C": PALETTE[2]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.7, 1]})
for b, g in blocks.groupby("batch"):
    big = g[g["rows"] > 1000]
    small = g[g["rows"] <= 1000]
    a1.hlines(big["rows"], big["start"], big["end"], color=BC[b], lw=4, label=f"批次 {b}")
    a1.scatter((small["start"] + small["end"]) / 2, small["rows"], s=9, color=BC[b], alpha=0.75, linewidths=0)
for x in (p2, p3):
    a1.axvline(x, color=MUTED, lw=1)
a1.axhline(300, color=MUTED, lw=0.8, ls=(0, (4, 3)), label="300 行")
a1.set_yscale("log"); a1.set_ylim(0.7, 4e4); a1.set_xlim(-4000, len(f) + 4000)
a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, borderaxespad=0.2)
a1.set_xticks([0, p2, p3, len(f)])
a1.set_xticklabels(["1", f"{p2:,}", f"{p3:,}", f"{len(f):,}"])
a1.set_xlabel("原始文件行号"); a1.set_ylabel("城市块的行数（对数轴）")
a1.grid(axis="x", visible=False)
bins = np.arange(0.5, 621, 20)           # 每箱 (k, k+20]，使 300 落在 281~300 这一箱
a2.hist(a_city.values, bins=bins, color=PALETTE[0], rwidth=0.9)
a2.set_xlim(0, 620); a2.set_xticks([0, 150, 300, 450, 600])
a2.set_xlabel("批次 A 中每个城市的行数"); a2.set_ylabel("城市个数")
a2.grid(axis="x", visible=False)
save_fig(fig, "output/figs/A2_batch_blocks.png")

# ---------------------------------------------------------------------------
# 3. 块内结构：职位位置与薪酬自相关（F047 的批次 B 大块）
# ---------------------------------------------------------------------------
DEMO = "F047"
d = f[(f["batch"] == "B") & (f["city"] == DEMO)]
assert d.index.max() - d.index.min() + 1 == len(d)      # 该块在行序上连续
pos = np.arange(len(d))
vc = d["job_title"].value_counts()


def dense_share(pp, w=300):
    """落在最密的 w 行窗口内的记录比例。"""
    pp = np.sort(pp)
    j = np.searchsorted(pp, pp + w, side="left")
    return (j - np.arange(len(pp))).max() / len(pp)


tp = []
for t in vc[vc >= 20].index:
    pp = pos[(d["job_title"] == t).values]
    tp.append({"job_title": t, "条数": len(pp), "首行": int(pp.min()), "末行": int(pp.max()),
               "中间80%跨度": float(np.percentile(pp, 90) - np.percentile(pp, 10)),
               "最密300行窗口占比": dense_share(pp)})
tp = pd.DataFrame(tp)
out_csv(tp, "A2_title_position.csv")
TOPN = 10
top = tp.head(TOPN)
p(f"== {DEMO} 批次 B 大块 {len(d):,} 行；条数>=20 的职位 {len(tp)} 个：最密 300 行窗口占比中位数 "
  f"{tp['最密300行窗口占比'].median():.3f}，占比>=0.9 的职位 {(tp['最密300行窗口占比'] >= 0.9).mean():.1%}，"
  f"中间80%跨度中位数 {tp['中间80%跨度'].median():.0f} 行")
p(top.round(3).to_string(index=False))


def acf(y, k):
    a, b = y[:-k], y[k:]
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[m], b[m])[0, 1]


LAGS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
y = np.log(d["salary_mid_k"].where(~d["flag_any"])).values      # 被过滤的记录记为缺失
ac_real = np.array([acf(y, k) for k in LAGS])
rng = np.random.default_rng(0)
ac_shuf = np.array([[acf(rng.permutation(y), k) for k in LAGS] for _ in range(50)])
ac_tab = pd.DataFrame({"滞后行数": LAGS, "实际行序": ac_real.round(4), "打乱行序均值": ac_shuf.mean(0).round(4),
                       "打乱行序最小": ac_shuf.min(0).round(4), "打乱行序最大": ac_shuf.max(0).round(4)})
p("== 自相关 ==\n", ac_tab.to_string(index=False))
# 28 个大块（14 城 × B、C）的滞后 1 行、300 行自相关
ac_all = []
for (b, c), g in f[(f["batch"] != "A") & f["city"].isin(CORE)].groupby(["batch", "city"]):
    yy = np.log(g["salary_mid_k"].where(~g["flag_any"])).values
    ac_all.append({"batch": b, "city": c, "rows": len(g), "滞后1": acf(yy, 1), "滞后300": acf(yy, 300)})
ac_all = pd.DataFrame(ac_all)
out_csv(ac_all.round(4), "A2_block_autocorr.csv")
p("28 个大块 滞后1：", ac_all["滞后1"].describe().round(3).to_dict(), "\n滞后300：", ac_all["滞后300"].describe().round(3).to_dict())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.45, 1]})
for i, r in enumerate(top.itertuples(index=False)):
    pp = pos[(d["job_title"] == r.job_title).values]
    a1.scatter(pp, np.full(len(pp), TOPN - 1 - i), marker="|", s=170, color=PALETTE[0], linewidths=1.1)
a1.set_yticks(range(TOPN))
a1.set_yticklabels([f"{t[:6]}…" for t in top["job_title"]][::-1])
a1.set_ylabel("记录数最多的 10 个职位（编码前 6 位）")
a1.set_xlim(-200, len(d) + 200); a1.set_ylim(-0.7, TOPN - 0.3)
a1.set_xlabel(f"在 {DEMO} 批次 B 大块内的行位置（共 {len(d):,} 行）")
a1.grid(axis="y", visible=False)
n_tight = int((top["最密300行窗口占比"] >= 0.9).sum())
p(f"前 {TOPN} 个高频职位中 {n_tight} 个把九成以上记录集中在 300 行窗口内")
a2.fill_between(LAGS, ac_shuf.min(0), ac_shuf.max(0), color=NEUTRAL, alpha=0.5, linewidth=0)
a2.plot(LAGS, ac_real, color=PALETTE[0], lw=2, marker="o", ms=4.5, label="实际行序")
a2.plot(LAGS, ac_shuf.mean(0), color=MUTED, lw=1.5, label="打乱行序（50 次）")
a2.set_xscale("log"); a2.set_xlim(0.8, 1300); a2.set_ylim(-0.06, 0.4)
a2.set_xticks([1, 10, 100, 1000]); a2.set_xticklabels(["1", "10", "100", "1,000"])
a2.set_xlabel("滞后行数（对数轴）"); a2.set_ylabel("对数月薪的自相关系数")
a2.legend(loc="upper right")
save_fig(fig, "output/figs/A2_within_block.png")

# ---------------------------------------------------------------------------
# 4. 核心城市：批次 A 与 B+C 的差异（清洗后口径）
# ---------------------------------------------------------------------------
core = df[df["city"].isin(CORE)].assign(part=lambda x: np.where(x["batch"] == "A", "A", "BC"),
                                        ba=lambda x: x["edu_level"] >= 4)
gap = core.groupby(["city", "part"]).agg(n=("ba", "size"), sal=("salary_mid_k", "median"), ba=("ba", "mean")).unstack()
gap.columns = [f"{a}_{b}" for a, b in gap.columns]
gap["city_group"] = core.groupby("city")["city_group"].first()
# 若批次 A 是 B+C 的随机样本，其本科及以上人数应服从二项分布：做双侧检验
gap["binom_p"] = [binomtest(int(round(r.ba_A * r.n_A)), int(r.n_A), r.ba_BC).pvalue for r in gap.itertuples()]
gap = gap.sort_values(["sal_BC", "ba_BC"])
out_csv(gap.round(4), "A2_core_batch_gap.csv", index=True)
p("== 核心城市 批次 A vs B+C ==\n", gap.round(4).to_string())
p("本科及以上占比与 B+C 有显著差异（p<0.05）的城市数", int((gap["binom_p"] < 0.05).sum()),
  gap.index[gap["binom_p"] < 0.05].tolist())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True, sharey=True)
yy = np.arange(len(gap))
for ax, ca, cb, xl in ((a1, "sal_A", "sal_BC", "折算月薪中位数（千元/月）"),
                       (a2, "ba_A", "ba_BC", "本科及以上占比（%）")):
    k = 100 if ca == "ba_A" else 1
    ax.hlines(yy, gap[cb] * k, gap[ca] * k, color=NEUTRAL, lw=1.5, zorder=1)
    ax.scatter(gap[cb] * k, yy, s=55, color=PALETTE[0], zorder=3, label="批次 B+C")
    ax.scatter(gap[ca] * k, yy, s=55, color=PALETTE[1], zorder=3, label="批次 A")
    ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
a1.set_yticks(yy); a1.set_yticklabels([f"{c}（{g}）" for c, g in zip(gap.index, gap["city_group"])])
a1.set_xlim(0, 28); a2.set_xlim(0, 92)
a1.legend(loc="lower right")
save_fig(fig, "output/figs/A2_core_batch_gap.png")

# ---------------------------------------------------------------------------
# 5. 假设汇总表中引用的证据数字（核对用）
# ---------------------------------------------------------------------------
ok = df[df["education"] != "Gw"]
p("== H1 == 学历等级薪酬中位数", ok.groupby("edu_level")["salary_mid_k"].median().to_dict(),
  "\n经验等级薪酬中位数", df.groupby("exp_level")["salary_mid_k"].median().to_dict(),
  f"\n学历不限中经验不限占 {ok.loc[ok.edu_any, 'exp_any'].mean():.1%}，有学历要求中占 {ok.loc[~ok.edu_any, 'exp_any'].mean():.1%}")
cs = core.groupby("city").agg(组=("city_group", "first"), 中位数=("salary_mid_k", "median"),
                              均值=("salary_mid_k", "mean"), 本科及以上=("ba", "mean")).sort_values("均值", ascending=False)
p("== H2 ==\n", cs.round(3).to_string())
p("== 后果 == 清洗后行业数", df["company_type"].nunique(), "职位数", df["job_title"].nunique(),
  f"有职位族的记录占清洗后 {df['job_family'].notna().mean():.2%}")
p("外部对照文件大小（字节）", os.path.getsize(os.path.join(ROOT, "data", "reference", "platform_taxonomy.json")))
p("done")
