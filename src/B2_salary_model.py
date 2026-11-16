# -*- coding: utf-8 -*-
"""成员 B：薪酬模式的建模与差异识别，对应赛题问题 3。

运行：python src/B2_salary_model.py
产出：output/figs/B2_*.png、data/processed/B2_*.csv，并在控制台打印小节稿用到的全部数字。
模型：因变量为对数月薪中值，自变量全部为类别哑变量的 OLS；标准误按企业聚类。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *  # noqa
import scipy.sparse as sp

plt = apply_style()
FIG = "output/figs/"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)


def out_csv(t, name, index=True):
    t.to_csv(os.path.join(PROCESSED_DIR, name), index=index, encoding="utf-8-sig")


# =============================================================================
# 0. 数据与口径
# =============================================================================
df = load_enriched()
prof = family_profile()
FLAB = prof["family_label"].to_dict()
df["ly"] = np.log(df["salary_mid_k"])
df["edu"] = df["edu_level"].map(dict(enumerate(EDU_LEVEL_NAME)))      # 7 个学历等级（H1）
df["exp"] = df["exp_label"]
df["pt"] = np.where(df["is_parttime"], "是", "否")                    # 日结类（日薪/时薪/周薪）标记
df["ind"] = df["company_type"].str.replace("type_", "")
core = df[df["job_family"].notna()].copy()                            # 核心城市口径：有职位族的记录
core["t1"] = core["city_group"].eq("一线型")
print("全量口径", len(df), "核心城市口径", len(core), "日结类", df["is_parttime"].sum(), core["is_parttime"].sum())
print(pd.crosstab(df["city_group"], df["batch"]).to_string())

mode = lambda s: s.value_counts().index[0]
REFS = {"edu": "大专", "exp": "1-3年", "pt": "否", "batch": "A", "city_group": "普通城市",
        "pattern": "常规区间型", "edu_t1": "_0", "exp_t1": "_0", "fam_t1": "_0"}


def ols(d, cats, cluster="company"):
    """类别哑变量 OLS。返回 (系数 Series, 聚类稳健协方差 DataFrame, R²)；索引为 (变量, 水平)。"""
    n = len(d)
    mats, names = [sp.csr_matrix(np.ones((n, 1)))], [("const", "")]
    for c in cats:
        ref = REFS.get(c, mode(d[c]))                                  # 未指定基准的变量以众数为基准
        lv = [v for v in pd.unique(d[c]) if v != ref]
        code = d[c].map(pd.Series(np.arange(len(lv)), index=lv)).values
        ok = ~np.isnan(code)
        mats.append(sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), code[ok].astype(int))), shape=(n, len(lv))))
        names += [(c, v) for v in lv]
    X = sp.hstack(mats).tocsr()
    y = d["ly"].values
    inv = np.linalg.pinv((X.T @ X).toarray(), hermitian=True)
    b = inv @ (X.T @ y)
    e = y - X @ b
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    g, _ = pd.factorize(d[cluster])                                    # 同一企业的岗位不独立：按企业聚类
    G = sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(g.max() + 1, n))
    S = (G @ X.multiply(e[:, None]).tocsr())
    V = inv @ (S.T @ S).toarray() @ inv
    idx = pd.MultiIndex.from_tuples(names)
    return pd.Series(b, index=idx), pd.DataFrame(V, index=idx, columns=idx), r2


def pct(b, se):
    """对数系数 -> 百分比溢价及 95% 置信区间。"""
    return pd.DataFrame({"溢价(%)": (np.exp(b) - 1) * 100, "下限(%)": (np.exp(b - 1.96 * se) - 1) * 100,
                         "上限(%)": (np.exp(b + 1.96 * se) - 1) * 100, "标准误(对数)": se})


def prem(b, V, var):
    """某个变量各水平相对基准水平的溢价。"""
    return pct(b[var], pd.Series(np.sqrt(np.diag(V.loc[var, var].values)), index=b[var].index))


def prem_centered(b, V, var, w):
    """各水平相对“记录数加权平均水平”的溢价（避免依赖任意选定的基准水平）。w 为各水平记录数。"""
    lv = list(b[var].index)
    allv = lv + [k for k in w.index if k not in lv]
    bb = np.r_[b[var].values, np.zeros(len(allv) - len(lv))]
    VV = np.zeros((len(allv), len(allv))); VV[:len(lv), :len(lv)] = V.loc[var, var].values
    A = np.eye(len(allv)) - np.outer(np.ones(len(allv)), w.reindex(allv).values / w.sum())
    t = pct(pd.Series(A @ bb, index=allv), pd.Series(np.sqrt(np.diag(A @ VV @ A.T)), index=allv))
    t.insert(0, "记录数", w.reindex(allv).values)
    return t.sort_values("溢价(%)", ascending=False)


def eta2(d, c):
    """单因素解释的方差比例（组间平方和 / 总平方和）。"""
    g = d.groupby(c)["ly"]; m = d["ly"].mean()
    return (g.size() * (g.mean() - m) ** 2).sum() / ((d["ly"] - m) ** 2).sum()


# =============================================================================
# 1. 各因素的解释力：单因素方差比例 + 逐步加入的 R²
# =============================================================================
FACTORS = [("经验", "exp"), ("学历", "edu"), ("职位族", "job_family"), ("行业", "company_type"), ("城市", "city")]
single = pd.DataFrame({
    "核心城市口径": {cn: eta2(core, c) for cn, c in FACTORS},
    "全量口径": {cn: eta2(df, c) for cn, c in FACTORS if c != "job_family"},
}) * 100
print("\n单因素解释的方差比例(%)\n", single.round(2).to_string())
print("原始学历编码(9种) 核心", eta2(core, "education"), "全国", eta2(df, "education"))
print("全量口径：城市组", eta2(df, "city_group"), "批次", eta2(df, "batch"), "日结类", eta2(df, "pt"), "核心日结类", eta2(core, "pt"))

STEPS_CORE = [("学历+经验", ["edu", "exp"]), ("+城市", ["edu", "exp", "city"]),
              ("+行业", ["edu", "exp", "city", "company_type"]),
              ("+职位族", ["edu", "exp", "city", "company_type", "job_family"]),
              ("+日结类标记", ["edu", "exp", "city", "company_type", "job_family", "pt"])]
STEPS_NAT = [("学历+经验", ["edu", "exp"]), ("+城市", ["edu", "exp", "batch", "city"]),      # 全量口径的城市一步同时控制批次
             ("+行业", ["edu", "exp", "batch", "city", "company_type"]),
             ("+日结类标记", ["edu", "exp", "batch", "city", "company_type", "pt"])]
step = pd.DataFrame({"核心城市口径": {k: ols(core, c)[2] for k, c in STEPS_CORE},
                     "全量口径": {k: ols(df, c)[2] for k, c in STEPS_NAT}}).reindex([k for k, _ in STEPS_CORE])
print("\n逐步加入的累计 R²\n", step.round(4).to_string(), "\n增量\n", step.diff().round(4).to_string())
core_np, bA = core[~core["is_parttime"]], df[df["batch"] == "A"]
print("稳健性：核心口径剔除日结类 五因素 R²", ols(core_np, STEPS_CORE[3][1])[2],
      "；仅批次A 学历+经验+城市+行业+日结类 R²", ols(bA, ["edu", "exp", "city", "company_type", "pt"])[2], len(bA))
ev = pd.concat([single.round(2).add_prefix("单因素方差比例(%)_"), step.round(4).add_prefix("累计R2_")], axis=1)
out_csv(ev, "B2_explained_variance.csv")

# ---- 图1：解释力 ----
SER = [("核心城市口径", PALETTE[0]), ("全量口径", PALETTE[1])]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
h = 0.38
for ax, tab, scale, fm in ((axes[0], single.sort_values("核心城市口径"), 1, "{:.1f}"), (axes[1], step.iloc[::-1] * 100, 1, "{:.1f}")):
    y = np.arange(len(tab))
    for k, (nm, col) in enumerate(SER):
        v = tab[nm].values
        ax.barh(y + (0.5 - k) * h, v, height=h - 0.03, color=col, label=nm)
    ax.set_yticks(y); ax.set_yticklabels(tab.index); ax.grid(axis="y", visible=False)
axes[0].set_xlim(0, 25); axes[0].set_xlabel("单个因素解释的对数月薪方差比例（%）")
axes[1].set_xlim(0, 55); axes[1].set_xlabel("逐步加入因素后的累计 R²（%）")
axes[0].legend(loc="center right")
save_fig(fig, FIG + "B2_explain.png")

# =============================================================================
# 2. 溢价结构
# =============================================================================
CORE_FULL = ["edu", "exp", "pt", "city", "company_type", "job_family"]
NAT_FULL = ["edu", "exp", "pt", "batch", "city", "company_type"]
bC, VC, r2C = ols(core, CORE_FULL)
bN, VN, r2N = ols(df, NAT_FULL)
print("\n核心城市口径全模型 R²", r2C, "全量口径全模型 R²", r2N)
coef = []
for nm, b, V in (("核心城市口径", bC, VC), ("全量口径", bN, VN)):
    for var, cn in (("edu", "学历"), ("exp", "经验"), ("pt", "日结类"), ("batch", "批次")):
        if var in b.index.get_level_values(0):
            t = prem(b, V, var); t.insert(0, "变量", cn); t.insert(0, "模型", nm); coef.append(t)
# 城市组溢价（全部记录，相对普通城市）：未控制 / 控制学历经验 / 仅批次 A
CG = {"未控制": (df, ["city_group"]), "控制学历、经验": (df, ["edu", "exp", "pt", "city_group"]),
      "控制学历、经验、行业": (df, ["edu", "exp", "pt", "company_type", "city_group"]),
      "仅批次A·未控制": (bA, ["city_group"]), "仅批次A·控制学历、经验": (bA, ["edu", "exp", "pt", "city_group"])}
cg = {}
for nm, (d, cats) in CG.items():
    b, V, r2 = ols(d, cats); cg[nm] = prem(b, V, "city_group")
    t = cg[nm].copy(); t.insert(0, "变量", "城市组"); t.insert(0, "模型", "全量口径·" + nm); coef.append(t)
    print("城市组溢价 |", nm, "R²=%.4f\n" % r2, cg[nm].round(2).to_string())
c2 = cg["控制学历、经验"]["溢价(%)"]
print("控制后 一线型相对新一线型", ((100 + c2["一线型"]) / (100 + c2["新一线型"]) - 1) * 100)
coef = pd.concat(coef).rename_axis("水平").reset_index()
out_csv(coef.round(4), "B2_premium_coefs.csv", index=False)
print(coef[coef["变量"].isin(["学历", "经验", "日结类", "批次"])].round(2).to_string())

# 行业、职位族溢价（核心城市口径全模型，相对记录数加权平均）
ind_p = prem_centered(bC, VC, "company_type", core["company_type"].value_counts())
ind_p.index = ind_p.index.str.replace("type_", "")
fam_p = prem_centered(bC, VC, "job_family", core["job_family"].value_counts())
fam_p.insert(0, "职位族名称", fam_p.index.map(FLAB))
out_csv(ind_p.round(3).rename_axis("行业"), "B2_premium_industry.csv")
out_csv(fam_p.round(3).rename_axis("职位族"), "B2_premium_family.csv")
ind_big = ind_p[ind_p["记录数"] >= 500]
print("\n行业数", len(ind_p), "记录数≥500 的行业", len(ind_big))
print(ind_big.head(6).round(1).to_string(), "\n", ind_big.tail(6).round(1).to_string())
print(fam_p.head(6).round(1).to_string(), "\n", fam_p.tail(6).round(1).to_string())
print("职位族溢价范围", fam_p["溢价(%)"].min(), fam_p["溢价(%)"].max(), "行业(≥500)", ind_big["溢价(%)"].min(), ind_big["溢价(%)"].max())


def dots(ax, y, t, color, label=None, hollow=False):
    """系数图：点为溢价，横线为 95% 置信区间。"""
    ax.hlines(y, t["下限(%)"], t["上限(%)"], color=color, lw=1.6)
    ax.plot(t["溢价(%)"], y, "o", ms=6.5, color=color, mfc=SURFACE if hollow else color, mew=1.6, label=label)


# ---- 图2：学历阶梯、经验阶梯、城市组溢价 ----
EDU_ROWS = ["学历不限", None, "初中及以下", "中专/高中", "大专", "本科", "硕士", "博士"]
EXP_ROWS = ["经验不限", None, "在校/应届", "1年以内", "1-3年", "3-5年", "5-10年", "10年以上"]
fig, axes = plt.subplots(1, 3, figsize=(11, 4.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1, 0.9]})
for ax, var, rows, ref, tt in ((axes[0], "edu", EDU_ROWS, "大专", "学历：本科起逐级加速"), (axes[1], "exp", EXP_ROWS, "1-3年", "经验：5 年以上回报最大")):
    lab = [r for r in rows if r]
    y = pd.Series([i for i, r in enumerate(rows) if r], index=lab)
    for k, (nm, b, V) in enumerate((("核心城市口径", bC, VC), ("全量口径", bN, VN))):
        t = prem(b, V, var).reindex([r for r in lab if r != ref])
        dots(ax, y[t.index] + (0.5 - k) * 0.3, t, SER[k][1], nm)
    ax.plot(0, y[ref], "o", ms=6.5, color=INK_2)
    ax.axvline(0, color=AXIS, lw=1); ax.axhline(1, color=GRID, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(y.values); ax.set_yticklabels([f"{r}（基准）" if r == ref else r for r in lab])
    ax.set_xlabel(f"相对{ref}的月薪溢价（%）"); ax.grid(axis="y", visible=False)
axes[0].legend(loc="lower right")
ax = axes[2]
rows = ["普通城市", "中等城市", "新一线型", "一线型"]
y = pd.Series(np.arange(4), index=rows)
dots(ax, y[rows[1:]] + 0.15, cg["未控制"].reindex(rows[1:]), MUTED, "未控制", hollow=True)
dots(ax, y[rows[1:]] - 0.15, cg["控制学历、经验"].reindex(rows[1:]), PALETTE[1], "控制学历、经验后")
ax.plot(0, 0, "o", ms=6.5, color=INK_2); ax.axvline(0, color=AXIS, lw=1)
ax.set_yticks(y.values); ax.set_yticklabels(["普通城市（基准）"] + rows[1:]); ax.set_ylim(-0.6, 3.9)
ax.set_xlabel("相对普通城市的月薪溢价（%）"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
save_fig(fig, FIG + "B2_premium_ladder.png")

# ---- 图3：行业与职位族溢价 Top / Bottom ----
K = 6
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [0.8, 1]})
for ax, t, lab, tt in ((axes[0], ind_big, lambda s: s.index, f"行业：最高 +{ind_big['溢价(%)'].max():.0f}%，最低 {ind_big['溢价(%)'].min():.0f}%"),
                       (axes[1], fam_p, lambda s: s["职位族名称"], f"职位族：最高 +{fam_p['溢价(%)'].max():.0f}%，最低 {fam_p['溢价(%)'].min():.0f}%")):
    s = pd.concat([t.head(K), t.tail(K)])
    y = np.r_[np.arange(2 * K + 1, K + 1, -1), np.arange(K, 0, -1)] - 1.0      # 高、低两组之间空一行
    dots(ax, y, s, PALETTE[0])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(list(lab(s))); ax.grid(axis="y", visible=False)
    ax.set_xlabel("相对全体加权平均的月薪溢价（%）")
save_fig(fig, FIG + "B2_premium_top.png")

# =============================================================================
# 3. 薪酬模式类型（“怎么付”）
# =============================================================================
NARROW, WIDE = 1.25, 2.0          # 带宽倍数阈值：≤1.25 为窄带，≥2 为宽带
PATTERNS = ["固定窄带型", "常规区间型", "宽带浮动型", "多薪激励型", "日结时薪型"]
r = df["salary_ratio"]
print("\n月薪制且未注明N薪 的带宽分布：≤1.25 占", (r[~df.is_parttime & (df.salary_months == 12)] <= NARROW + 1e-9).mean(),
      "≥2 占", (r[~df.is_parttime & (df.salary_months == 12)] >= WIDE - 1e-9).mean())
df["pattern"] = np.select([df["is_parttime"], df["salary_months"] > 12, r <= NARROW + 1e-9, r >= WIDE - 1e-9],
                          ["日结时薪型", "多薪激励型", "固定窄带型", "宽带浮动型"], "常规区间型")
core["pattern"] = df.loc[core.index, "pattern"]
g = df.groupby("pattern")
pat = pd.DataFrame({"记录数": g.size(), "占比(%)": g.size() / len(df) * 100, "月薪中位数(千元)": g["salary_mid_k"].median(),
                    "带宽倍数中位数": g["salary_ratio"].median(), "本科及以上占比(%)": g["edu_level"].apply(lambda s: (s >= 4).mean() * 100),
                    "经验不限占比(%)": g["exp_any"].mean() * 100}).reindex(PATTERNS)
b, V, r2 = ols(core, ["edu", "exp", "city", "company_type", "job_family", "pattern"])
pp = prem(b, V, "pattern")
pat["控制后相对常规区间型(%)"] = pp["溢价(%)"]; pat["控制后下限(%)"] = pp["下限(%)"]; pat["控制后上限(%)"] = pp["上限(%)"]
print("薪酬模式概况（控制后溢价来自核心城市口径，R²=%.4f）\n" % r2, pat.round(2).to_string())
print("模式单因素方差比例 全国", eta2(df, "pattern"), "核心", eta2(core, "pattern"))
out_csv(pat.round(3).rename_axis("薪酬模式"), "B2_pay_pattern_summary.csv")

share = lambda d, c: (pd.crosstab(d[c], d["pattern"], normalize="index") * 100).reindex(columns=PATTERNS)
by_city, by_fam, by_ind = share(df, "city_group"), share(core, "job_family"), share(df, "ind")
by_ind = by_ind[df["ind"].value_counts().reindex(by_ind.index) >= 1000]
overall = (df["pattern"].value_counts(normalize=True) * 100).reindex(PATTERNS)
print("城市组\n", by_city.round(1).to_string())
print("各职位族占比最高的模式\n", by_fam.idxmax(axis=1).value_counts().to_string())
for p in ("固定窄带型", "宽带浮动型", "多薪激励型"):
    print(p, "职位族前3\n", by_fam.sort_values(p, ascending=False).head(3).round(1).assign(名称=lambda t: t.index.map(FLAB)).to_string())
    print(p, "行业前3（记录数≥1000）\n", by_ind.sort_values(p, ascending=False).head(3).round(1).to_string())
print("学历\n", share(df, "edu").round(1).to_string(), "\n经验\n", share(df, "exp").round(1).to_string())
allg = pd.concat([by_city.assign(维度="城市组"), by_fam.assign(维度="职位族"), by_ind.assign(维度="行业")]).rename_axis("分组")
out_csv(allg.round(2), "B2_pay_pattern_by_group.csv")

# ---- 图4：薪酬模式构成（堆叠条形） ----
left = pd.concat([overall.to_frame("全部记录").T, by_city.reindex(["一线型", "新一线型", "中等城市", "普通城市"])])
picks = []
for p in ("多薪激励型", "宽带浮动型", "固定窄带型"):
    picks += [f for f in by_fam.sort_values(p, ascending=False).index if f not in picks][:3]
right = by_fam.reindex(picks)                      # 图中只写职位族编号
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), constrained_layout=True, gridspec_kw={"width_ratios": [0.85, 1]})
for ax, t, tt in ((axes[0], left, "城市等级越高，多薪越多、宽带越少"), (axes[1], right, "各模式占比最高的 3 个职位族")):
    t = t.iloc[::-1]; acc = np.zeros(len(t))
    for k, p in enumerate(PATTERNS):
        ax.barh(t.index, t[p], left=acc, color=PALETTE[k], height=0.68, label=p, edgecolor=SURFACE, linewidth=1.2)
        acc += t[p].values
    ax.set_xlim(0, 100); ax.set_xlabel("各薪酬模式的记录数占比（%）"); ax.grid(axis="y", visible=False)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside upper center", ncol=5)
save_fig(fig, FIG + "B2_pattern.png")

# =============================================================================
# 4. 潜在差异（交互）
# =============================================================================
# 4.1 学历、经验回报是否因城市组而异：同一设定下分城市组回归
SHOW = [("edu", "本科"), ("edu", "硕士"), ("exp", "3-5年"), ("exp", "5-10年"), ("exp", "10年以上")]
ret = {}
for grp in ("一线型", "新一线型", "普通城市"):
    d = df[df["city_group"] == grp]
    b, V, r2 = ols(d, ["edu", "exp", "pt", "city", "company_type"] + ([] if grp == "普通城市" else ["batch"]))
    t = pd.concat([prem(b, V, "edu"), prem(b, V, "exp")], keys=["edu", "exp"]).loc[SHOW]
    ret[grp] = t["溢价(%)"]; ret[grp + "_下限"] = t["下限(%)"]; ret[grp + "_上限"] = t["上限(%)"]
    print("分城市组回归", grp, len(d), "R²=%.4f" % r2)
ret = pd.DataFrame(ret)
# 正式检验：核心城市口径全模型加入 学历×一线型、经验×一线型 交互项（基准水平并入 _0，主效应由城市吸收）
core["edu_t1"] = np.where(core["t1"] & (core["edu"] != "大专"), core["edu"], "_0")
core["exp_t1"] = np.where(core["t1"] & (core["exp"] != "1-3年"), core["exp"], "_0")
b, V, r2 = ols(core, CORE_FULL + ["edu_t1", "exp_t1"])
it = pd.concat([prem(b, V, "edu_t1"), prem(b, V, "exp_t1")], keys=["edu", "exp"])
print(f"交互模型 R²={r2:.4f}；一线型的回报相对新一线型高出的比例（%）\n", it.round(2).to_string())
ret["交互项(%)"] = it["溢价(%)"].reindex(ret.index); ret["交互项下限(%)"] = it["下限(%)"].reindex(ret.index)
ret["交互项上限(%)"] = it["上限(%)"].reindex(ret.index)
ret.index = [f"{v}（相对{'大专' if k == 'edu' else '1-3年'}）" for k, v in ret.index]
print(ret.round(1).to_string())
out_csv(ret.round(3).rename_axis("回报"), "B2_interaction_returns.csv")

# 4.2 同一职位族在一线型与新一线型之间的差距（控制学历、经验、日结类）
core["fam_t1"] = np.where(core["t1"], core["job_family"], "_0")
gap_raw = prem(*ols(core, ["job_family", "fam_t1"])[:2], "fam_t1")
b, V, r2 = ols(core, ["edu", "exp", "pt", "job_family", "fam_t1"])
gap = prem(b, V, "fam_t1")
med = core.groupby(["job_family", "t1"])["salary_mid_k"].median().unstack()
cnt = core.groupby(["job_family", "t1"]).size().unstack()
gap = pd.DataFrame({"一线型记录数": cnt[True], "新一线型记录数": cnt[False],
                    "一线型中位数(千元)": med[True], "新一线型中位数(千元)": med[False],
                    "未控制差距(%)": gap_raw["溢价(%)"], "控制后差距(%)": gap["溢价(%)"],
                    "下限(%)": gap["下限(%)"], "上限(%)": gap["上限(%)"]}).sort_values("控制后差距(%)", ascending=False)
gap.insert(0, "职位族名称", gap.index.map(FLAB))      # 按索引对齐后再补名称
out_csv(gap.round(3).rename_axis("职位族"), "B2_family_tier_gap.csv")
print("\n职位族的一线型-新一线型差距（R²=%.4f）" % r2)
print(gap.head(5).round(1).to_string(), "\n", gap.tail(5).round(1).to_string())
print(gap[["未控制差距(%)", "控制后差距(%)"]].describe().round(2).to_string())
print("全部为正且区间不含 0：", (gap["下限(%)"] > 0).all(), "；差距与职位族薪酬中位数的秩相关",
      gap["控制后差距(%)"].corr(prof["salary_median"], method="spearman"),
      "；与本科及以上占比的秩相关", gap["控制后差距(%)"].corr(prof["bachelor_share"], method="spearman"))
print("完成")
