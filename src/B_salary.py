# -*- coding: utf-8 -*-
"""成员 B：薪酬数据的理解与标准化处理。

运行：python src/B_salary.py
产出：output/figs/B_*.png、data/processed/B_*.csv，并在控制台打印小节稿用到的全部数字。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from scipy import stats

plt = apply_style()
FIG = "output/figs/"
FORM_ORDER = ["month_k", "month_k_n", "day", "hour", "month_yuan", "piece", "week", "negotiable", "open_k"]
BAND_BINS = [0, 5, 8, 12, 20, 35, np.inf]
BAND_LABELS = ["<5K", "5-8K", "8-12K", "12-20K", "20-35K", "≥35K"]


def fmt(n):
    return f"{int(n):,}"


# =============================================================================
# 第一部分：薪酬数据的理解（未过滤数据 load_flagged）
# =============================================================================
f = load_flagged()
N = len(f)
print("总行数", N, "薪酬写法数", f["salary"].nunique())

# ---- 1.1 写法类别统计表 ----
g = f.groupby("salary_form")
form_tab = pd.DataFrame({
    "写法类别": [SALARY_FORMS[k] for k in FORM_ORDER],
    "示例": [g.get_group(k)["salary"].value_counts().index[0] for k in FORM_ORDER],
    "条数": [len(g.get_group(k)) for k in FORM_ORDER],
    "不同写法数": [g.get_group(k)["salary"].nunique() for k in FORM_ORDER],
}, index=FORM_ORDER)
form_tab["占比(%)"] = (form_tab["条数"] / N * 100).round(4)
form_tab["能否折算月薪"] = ["能" if g.get_group(k)["salary_mid_k"].notna().all() else "否" for k in FORM_ORDER]
print(form_tab.to_string())

# ---- 图1：写法类别数量（对数轴） + N薪月数分布 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
ax = axes[0]
t = form_tab.sort_values("条数")
ax.barh(t["写法类别"], t["条数"], color=PALETTE[0])
ax.set_xscale("log"); ax.set_xlim(0.6, 1e6)
ax.set_xlabel("记录数（条，对数轴）")
ax.grid(axis="y", visible=False)

ax = axes[1]
mon = f.loc[f["salary_form"] == "month_k_n", "salary_months"].value_counts().sort_index()
ax.bar(mon.index.astype(str), mon.values, color=PALETTE[0])
ax.set_yscale("log"); ax.set_ylim(0.6, 1e5)
ax.set_xlabel("年发薪月数（薪）"); ax.set_ylabel("记录数（条，对数轴）")
ax.grid(axis="x", visible=False)
save_fig(fig, FIG + "B_form_counts.png")
print("N薪月数分布\n", mon, "\n13薪占N薪比例", mon[13] / mon.sum(), "13-16薪占比", mon.loc[13:16].sum() / mon.sum())

# ---- 1.2 原始月薪区间的下限/上限特点（K 制两类） ----
mk = f[f["salary_form"].isin(["month_k", "month_k_n"])]
top15 = f["salary"].value_counts().head(15)
print("最常见15种写法\n", top15, "\n合计占比", top15.sum() / N)
print("下限为整数占比", (mk["salary_raw_min"] % 1 == 0).mean(), "上限为整数占比", (mk["salary_raw_max"] % 1 == 0).mean())
print("下限top5", mk["salary_raw_min"].value_counts().head(5).to_dict())
print("上限top5", mk["salary_raw_max"].value_counts().head(5).to_dict())
print("下限≤10K占比", (mk["salary_raw_min"] <= 10).mean(), "上限≤15K占比", (mk["salary_raw_max"] <= 15).mean())
print("K制写法数", mk["salary"].nunique(), "top15占全部", top15.sum() / N)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
ax = axes[0]
t = top15.sort_values()
ax.barh(t.index, t.values, color=PALETTE[0])
for y, v in enumerate(t.values):
    ax.text(v + 300, y, fmt(v), va="center", fontsize=9, color=INK_2)
ax.set_xlim(0, 29500); ax.set_xlabel("记录数（条）")
ax.set_title("最常见的 15 种薪酬写法")
ax.grid(axis="y", visible=False)

ax = axes[1]
ks = np.arange(1, 31)
cmin = mk["salary_raw_min"].value_counts().reindex(ks, fill_value=0)
cmax = mk["salary_raw_max"].value_counts().reindex(ks, fill_value=0)
w = 0.42
ax.bar(ks - w / 2, cmin.values / 1e4, width=w, color=PALETTE[0], label="区间下限")
ax.bar(ks + w / 2, cmax.values / 1e4, width=w, color=PALETTE[1], label="区间上限")
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("原始数值（K，仅“月薪(K)”与“N薪”两类，1~30K）"); ax.set_ylabel("记录数（万条）")
ax.set_title("下限集中在 3~8K，上限偏好 10/15/20 等整数")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
save_fig(fig, FIG + "B_raw_preference.png")

# ---- 1.3 薪酬带宽（上限/下限倍数） ----
r = f["salary_ratio"].dropna()
print("ratio 描述\n", r.describe(percentiles=[.05, .25, .5, .75, .95, .99]))
print("ratio≤2占比", (r <= 2).mean(), "ratio>4条数", (r > 4).sum(), "ratio=1.5", np.isclose(r, 1.5).sum(), "ratio=2", np.isclose(r, 2).sum())
print("ratio>4 最常见\n", f.loc[f["salary_ratio"] > 4, "salary"].value_counts().head(8))
print("带宽过宽按类别\n", f.loc[f["salary_ratio"] > 4, "salary_form"].value_counts())

fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
ax.hist(r.clip(upper=4.1), bins=np.arange(1.0, 4.2001, 0.05), color=PALETTE[0])
med = r.median()
ax.set_xlim(1, 4.2)
ax.set_xlabel("薪酬带宽：上限 / 下限（倍，大于 4 的并入最右一格）"); ax.set_ylabel("记录数（条）")
save_fig(fig, FIG + "B_ratio_dist.png")

# ---- 1.4 极端值与日薪/时薪岗位的编码集中度 ----
for s in ["250-260K", "500-1000元/月", "1-6K", "1-5K"]:
    print("极端写法", s, (f["salary"] == s).sum())
print("折算月薪中值>100K", (f["salary_mid_k"] > 100).sum(), "<1K", (f["salary_mid_k"] < 1).sum(),
      "最大", f["salary_mid_k"].max(), "最小", f["salary_mid_k"].min())
print("无法折算", f["flag_salary_unparsed"].sum(), "数值异常", f["flag_salary_range"].sum(), "区间过宽", f["flag_salary_ratio"].sum())

f["is_daily"] = f["salary_form"].isin(["day", "hour"])   # 日结类 = 元/天 + 元/时
print("日结类总数", f["is_daily"].sum(), f["is_daily"].mean())
code_share = {}
for col in ("experience", "education"):
    t = f.groupby(col)["is_daily"].agg(总数="size", 日结类="sum", 占比="mean")
    t["元/天"] = f[f["salary_form"] == "day"].groupby(col).size().reindex(t.index, fill_value=0)
    t["元/时"] = f[f["salary_form"] == "hour"].groupby(col).size().reindex(t.index, fill_value=0)
    code_share[col] = t
    print(col, "\n", t.to_string())
e = code_share["experience"]
print("Eby 的日薪占全部日薪", e.loc["Eby", "元/天"] / e["元/天"].sum(), "EdD 时薪占全部时薪", e.loc["EdD", "元/时"] / e["元/时"].sum())
d = code_share["education"]
print("GP 日结类占全部日结", d.loc["GP", "日结类"] / d["日结类"].sum())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
for ax, col, cn, hot in ((axes[0], "experience", "经验", ["Eby"]), (axes[1], "education", "学历", ["Gw"])):
    t = code_share[col]
    t = t[t["总数"] >= 100].sort_values("占比")      # 只画样本 ≥100 的编码，噪声编码在文中说明
    colors = [HIGHLIGHT if c in hot else NEUTRAL for c in t.index]
    ax.barh(t.index, t["占比"] * 100, color=colors)
    for y, (c, row) in enumerate(t.iterrows()):
        ax.text(row["占比"] * 100 + 1.5, y, f"{row['占比'] * 100:.1f}%（{fmt(row['日结类'])}/{fmt(row['总数'])}）",
                va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, 150); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("日薪、时薪岗位占该编码全部岗位的比例（%）"); ax.set_ylabel(f"{cn}要求编码")
    ax.grid(axis="y", visible=False)
axes[0].set_title("经验编码 Eby 中约一半是日结类岗位")
axes[1].set_title("学历编码 Gw 全部为日薪岗位")
save_fig(fig, FIG + "B_daily_codes.png")

# =============================================================================
# 第二部分：薪酬数据的标准化处理（过程用 flagged，结果用 load_clean）
# =============================================================================
c = load_clean()
NC = len(c)
print("\n===== 清洗后 =====", NC)
c["salary_band"] = pd.cut(c["salary_mid_k"], BAND_BINS, right=False, labels=BAND_LABELS)   # 左闭右开

# ---- 2.1 按写法类别汇总（处理前后） ----
rows = []
for k in FORM_ORDER:
    a = f[f["salary_form"] == k]; b = c[c["salary_form"] == k]
    raw_mid = (a["salary_raw_min"] + a["salary_raw_max"]) / 2
    rows.append({
        "salary_form": k, "写法类别": SALARY_FORMS[k], "示例": form_tab.loc[k, "示例"],
        "原始条数": len(a), "原始占比(%)": round(len(a) / N * 100, 4), "不同写法数": a["salary"].nunique(),
        "能否折算月薪": form_tab.loc[k, "能否折算月薪"],
        "原始数值中值_中位数(原单位)": raw_mid.median(),
        "清洗后条数": len(b), "保留率(%)": round(len(b) / len(a) * 100, 2),
        "折算月薪中值_P25": b["salary_mid_k"].quantile(.25), "折算月薪中值_中位数": b["salary_mid_k"].median(),
        "折算月薪中值_P75": b["salary_mid_k"].quantile(.75), "折算月薪中值_均值": round(b["salary_mid_k"].mean(), 3),
    })
form_sum = pd.DataFrame(rows)
form_sum.to_csv(os.path.join(PROCESSED_DIR, "B_salary_form_summary.csv"), index=False, encoding="utf-8-sig")
print(form_sum.drop(columns=["示例"]).to_string())

# ---- 图5：处理前后对比（按写法类别的箱线图，对数轴） ----
CONV = ["month_k", "month_k_n", "month_yuan", "day", "hour", "week"]
labels = [SALARY_FORMS[k] for k in CONV]
bp_kw = dict(vert=True, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
             boxprops=dict(facecolor=PALETTE[0], alpha=0.85), medianprops=dict(color=INK, lw=1.4),
             whiskerprops=dict(color=INK_2, lw=1), capprops=dict(color=INK_2, lw=1))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), constrained_layout=True)
before = [((f.loc[f["salary_form"] == k, "salary_raw_min"] + f.loc[f["salary_form"] == k, "salary_raw_max"]) / 2).values for k in CONV]
after = [c.loc[c["salary_form"] == k, "salary_mid_k"].values for k in CONV]
for ax, data, yl, tt in ((axes[0], before, "原始区间中值（各自的原单位，对数轴）", "折算前：各类中位数相差近 3 个数量级，不可比"),
                         (axes[1], after, "月薪中值（千元/月，对数轴）", "折算后：6 类写法落在同一尺度")):
    ax.boxplot(data, **bp_kw)
    ax.set_xticks(range(1, len(CONV) + 1)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log"); ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
axes[0].set_ylim(1, 3e4); axes[1].set_ylim(1, 100)
axes[1].set_yticks([1, 2, 5, 10, 20, 50, 100]); axes[1].set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
save_fig(fig, FIG + "B_before_after.png")

# ---- 图6：折算后月薪中值整体分布 ----
s = c["salary_mid_k"]
q = s.quantile([.01, .05, .25, .5, .75, .95, .99])
print("月薪中值分布\n", s.describe(), "\n", q, "\n偏度", stats.skew(s), "log偏度", stats.skew(np.log(s)))
p99, med, mean = q[.99], s.median(), s.mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
ax = axes[0]
ax.hist(s[s <= p99], bins=np.arange(1, p99 + 0.5001, 0.5), color=PALETTE[0])
ax.set_xlabel(f"月薪中值（千元/月，截断到 P99={p99:g}）"); ax.set_ylabel("岗位数（条）")
ax = axes[1]
ax.hist(s, bins=np.logspace(0, 2, 41), color=PALETTE[0])
ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_xlabel("月薪中值（千元/月，对数轴，全量）"); ax.set_ylabel("岗位数（条）")
save_fig(fig, FIG + "B_mid_dist.png")

# ---- 2.3 月薪 vs 年薪（N 薪） ----
bon, nob = c[c["has_bonus_months"]], c[~c["has_bonus_months"]]
print("13薪及以上占比", len(bon), len(bon) / NC)
print(c.groupby("has_bonus_months")[["salary_mid_k", "salary_annual_k"]].agg(["median", "mean"]).to_string())
u = stats.mannwhitneyu(bon["salary_mid_k"], nob["salary_mid_k"], alternative="greater")
print("Mann-Whitney U", u.statistic, "p", u.pvalue, "AUC(效应量)", u.statistic / (len(bon) * len(nob)))
mon_sum = c.groupby("salary_months").agg(岗位数=("salary_mid_k", "size"), 月薪中值_中位数=("salary_mid_k", "median"),
                                         月薪中值_均值=("salary_mid_k", "mean"), 年薪_中位数=("salary_annual_k", "median")).round(3)
mon_sum.to_csv(os.path.join(PROCESSED_DIR, "B_salary_months_summary.csv"), encoding="utf-8-sig")
print(mon_sum.to_string())
print("年薪描述\n", c["salary_annual_k"].describe())
print("年薪/（月薪×12）在有N薪岗位的平均抬升", (bon["salary_months"] / 12).mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
ms = [12, 13, 14, 15, 16]
for ax, col, yl, tt in ((axes[0], "salary_mid_k", "月薪中值（千元/月，对数轴）", "发薪月数越多，月薪本身也越高（12~16 薪）"),
                        (axes[1], "salary_annual_k", "年薪 = 月薪中值 × 发薪月数（千元/年，对数轴）", "计入 N 薪后，年薪差距进一步拉大")):
    data = [c.loc[c["salary_months"] == m, col].values for m in ms]
    kw = dict(bp_kw); kw.pop("boxprops")
    bp = ax.boxplot(data, boxprops=dict(alpha=0.9), **kw)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(NEUTRAL if i == 0 else PALETTE[0])
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(["未注明(12薪)", "13薪", "14薪", "15薪", "16薪"])
    ax.set_yscale("log"); ax.set_ylabel(yl); ax.set_title(tt); ax.grid(axis="x", visible=False)
    for i, dd in enumerate(data):
        m = np.median(dd)
        ax.text(i + 1.33, m, f"{round(m, 2):g}", va="center", fontsize=8.5, color=INK_2)
axes[0].set_yticks([2, 5, 10, 20, 50]); axes[0].set_yticklabels(["2", "5", "10", "20", "50"])
axes[1].set_yticks([20, 50, 100, 200, 500, 1000]); axes[1].set_yticklabels(["20", "50", "100", "200", "500", "1000"])
fig.supxlabel("箱体为 P25~P75，须线为 P5~P95，数字为中位数；17 薪及以上样本少（共 1,028 条）未画出", fontsize=9.5, color=MUTED)
save_fig(fig, FIG + "B_months_salary.png")
print("17薪及以上条数", (c["salary_months"] >= 17).sum())

# ---- 2.4 薪酬分档 ----
band = c.groupby("salary_band", observed=False).agg(
    岗位数=("salary_mid_k", "size"), 月薪中值_中位数=("salary_mid_k", "median"),
    含N薪岗位占比=("has_bonus_months", "mean"), 年薪_中位数=("salary_annual_k", "median"))
band.insert(0, "区间定义(千元/月,左闭右开)", ["[1,5)", "[5,8)", "[8,12)", "[12,20)", "[20,35)", "[35,100]"])
band.insert(2, "占比(%)", (band["岗位数"] / NC * 100).round(2))
band["含N薪岗位占比"] = (band["含N薪岗位占比"] * 100).round(2)
band = band.rename(columns={"含N薪岗位占比": "含N薪岗位占比(%)"})
band.to_csv(os.path.join(PROCESSED_DIR, "B_salary_band_summary.csv"), encoding="utf-8-sig")
print(band.to_string())

fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
ax.bar(BAND_LABELS, band["岗位数"] / 1e4, color=PALETTE[0], width=0.65)
for i, (n, p) in enumerate(zip(band["岗位数"], band["占比(%)"])):
    ax.text(i, n / 1e4 + 0.25, f"{fmt(n)}\n{p:.1f}%", ha="center", fontsize=9.5, color=INK_2)
ax.set_ylim(0, band["岗位数"].max() / 1e4 * 1.18)
ax.set_xlabel("薪酬分档 salary_band（按折算月薪中值，左闭右开）"); ax.set_ylabel("岗位数（万条）")
ax.set_title("5-8K 档岗位最多，8K 以下合计占 54.5%")
ax.grid(axis="x", visible=False)
save_fig(fig, FIG + "B_salary_band.png")
print("8K以下占比", band["占比(%)"].iloc[:2].sum(), "20K以上占比", band["占比(%)"].iloc[4:].sum())

# ---- 2.5 全勤假设的局限：日薪/时薪折算结果 ----
for k in ("day", "hour"):
    b = c[c["salary_form"] == k]
    print(k, len(b), "折算月薪中位数", b["salary_mid_k"].median(), "≥20K条数", (b["salary_mid_k"] >= 20).sum(), (b["salary_mid_k"] >= 20).mean())
print("清洗后日结类合计", c["salary_form"].isin(["day", "hour"]).sum(), c["salary_form"].isin(["day", "hour"]).mean())
print("时薪≥100元/时 条数（清洗后）", ((c["salary_form"] == "hour") & (c["salary_raw_min"] >= 100)).sum())
print("完成")
