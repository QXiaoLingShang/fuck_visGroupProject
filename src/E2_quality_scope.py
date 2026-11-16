# -*- coding: utf-8 -*-
"""成员E · 第二轮 · 数据质量问题识别与过滤（补充部分）。

第一轮的规则命中、阈值敏感性、过滤前后对比仍由 E_quality_filter.py 产出；本脚本只做新增内容：
1) 稀有编码与 Gw 的语义核对；2) 分析口径表；3) 同一指标在不同口径下的差异图；4) 遗留问题状态。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *   # noqa

plt = apply_style()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

f = load_flagged().reset_index(drop=True)
f["batch"] = infer_batches(f)
df = load_enriched()

# ---------------------------------------------------------------- 1. 稀有编码核对
print("== 稀有编码与 Gw：日薪占比与经验构成（未过滤数据）==")
for col, codes in (("experience", ["ESu", "EkP", "Emc"]), ("education", ["GJ", "Gi", "GW", "Gw"])):
    for c in codes:
        d = f[f[col] == c]
        print(f"  {c:3s} n={len(d):3d} 日薪占比={d.salary_form.eq('day').mean():.1%} "
              f"写法={d.salary_form.value_counts().to_dict()} 经验={d.experience.value_counts().to_dict()} "
              f"学历={d.education.value_counts().to_dict()} 月薪中位数={d.salary_mid_k.median():.2f}")
day = f[f.salary_form == "day"]
print("  全体日薪占比:", f"{len(day) / len(f):.2%}", " 日薪记录中在校/应届占比:", f"{day.experience.eq('Eby').mean():.1%}")
print("  ESu 薪酬写法 Top:", f[f.experience == "ESu"].salary.value_counts().head(4).to_dict())
print("  有效经验编码月薪中位数:", f.groupby("experience").salary_mid_k.median().round(2).to_dict())
gw = df[df.education == "Gw"]
print("  清洗后 Gw:", len(gw), f"占 {len(gw) / len(df):.3%}", gw.salary_form.value_counts().to_dict(),
      gw.experience.value_counts().to_dict(), "有职位族:", int(gw.job_family.notna().sum()))

# ---------------------------------------------------------------- 2. 采集结构与分析口径
print("\n== 采集结构 ==")
city = f["city"].values
start = np.r_[0, np.flatnonzero(city[1:] != city[:-1]) + 1]
blocks = pd.DataFrame({"city": city[start], "len": np.diff(np.r_[start, len(city)]), "batch": f.batch.values[start]})
a = blocks[blocks.batch == "A"]
print("  批次 A：块数", len(a), "城市数", a.city.nunique(), "块长中位数", a.len.median(), "块长>300 的块数", int((a.len > 300).sum()),
      "最大", int(a.len.max()))
acity = f[f.batch == "A"].city.value_counts()
print("  批次 A 每城记录数：中位数", acity.median(), ">310 的城市", acity[acity > 310].to_dict())
tier = city_tier_map(f)
x = f[f.batch.isin(["B", "C"]) & ~f.city.map(tier).eq("核心城市")]
print("  批次 B、C 中非核心城市记录:", len(x), x.groupby("city").size().to_dict())

print("\n== 分析口径 ==")
is_core = df.city_group.isin(["一线型", "新一线型"])
scopes = {
    "全国可比口径": df.batch.eq("A") & ~df.batch_a_anomaly,
    "核心城市口径": df.batch.isin(["B", "C"]) & df.job_family.notna(),
    "全量口径": pd.Series(True, index=df.index),
}
for k, m in scopes.items():
    d = df[m]
    print(f"  {k}: 记录 {len(d):,}（{len(d) / len(df):.2%}） 城市 {d.city.nunique()} "
          f"薪酬中位数 {d.salary_mid_k.median():.1f} 本科及以上 {(d.edu_level >= 4).mean():.1%} 兼职类 {d.is_parttime.mean():.2%}")
bc = df[df.batch.isin(["B", "C"])]
print("  批次 B+C 全部:", len(bc), " 核心城市:", int((df.batch.isin(["B", "C"]) & is_core).sum()),
      " 无职位族:", int(bc.job_family.isna().sum()), f"有族占比 {bc.job_family.notna().mean():.2%}")
pt = df[df.is_parttime]
print("  兼职类:", len(pt), f"{len(pt) / len(df):.2%}", pt.salary_form.value_counts().to_dict(),
      "月薪中位数 兼职/非兼职:", pt.salary_mid_k.median().round(2), df[~df.is_parttime].salary_mid_k.median(),
      "在校/应届占比:", f"{pt.exp_level.eq(1).mean():.1%}", "对照", f"{df[~df.is_parttime].exp_level.eq(1).mean():.1%}")
print("  批次 A 异常块记录:", int(df.batch_a_anomaly.sum()), " 涉及城市:", sorted(df[df.batch_a_anomaly].city.unique()))
ok_a = df[scopes["全国可比口径"]]
print("  一线型/普通城市 薪酬中位数之比（全国可比口径）:",
      round(ok_a[ok_a.city_group == "一线型"].salary_mid_k.median() / ok_a[ok_a.city_group == "普通城市"].salary_mid_k.median(), 2))
print("  兼职类在各口径:", {k: int(df.is_parttime[m].sum()) for k, m in scopes.items()})

# 同一指标在不同口径下的差异
df["口径"] = np.where(df.batch.eq("A"), "批次 A", "批次 B+C")
stat = lambda g: pd.Series({"n": len(g), "薪酬中位数": g.salary_mid_k.median(), "本科占比": (g.edu_level >= 4).mean()})
grp = df.groupby(["city_group", "口径"]).apply(stat).unstack()
print("\n城市组 × 批次:\n", grp.round(3).to_string())
print("城市组（全量混用）:\n", df.groupby("city_group").apply(stat).round(3).to_string())
core = df[is_core]
cc = core.groupby(["city", "口径"]).apply(stat).unstack()
cc["组"] = df.drop_duplicates("city").set_index("city").city_group.reindex(cc.index)
print("\n14 个核心城市 × 批次:\n", cc.round(3).to_string())
dsal = (cc[("薪酬中位数", "批次 A")] - cc[("薪酬中位数", "批次 B+C")])
print("  薪酬中位数 A - (B+C)：", dsal.round(1).to_dict())
print("  秩相关（14 城，A 与 B+C）：薪酬", round(cc[("薪酬中位数", "批次 A")].corr(cc[("薪酬中位数", "批次 B+C")], method="spearman"), 3),
      " 本科占比", round(cc[("本科占比", "批次 A")].corr(cc[("本科占比", "批次 B+C")], method="spearman"), 3))
r = grp["薪酬中位数"]
print("  一线型/普通城市 薪酬中位数之比：批次 A", round(r.loc["一线型", "批次 A"] / r.loc["普通城市", "批次 A"], 2),
      " 混用（一线型全量/普通城市）", round(df[df.city_group == "一线型"].salary_mid_k.median() / r.loc["普通城市", "批次 A"], 2))

# ---------------------------------------------------------------- 3. 遗留问题状态
print("\n== 遗留问题 ==")
multi = df.groupby("company")["company_type"].nunique()
print("  多行业企业:", int((multi > 1).sum()), f"{(multi > 1).mean():.2%}")
pair = df.groupby(["company", "job_title"])["company_type"].nunique()
print("  同一企业同一职位对应多个行业的组合:", int((pair > 1).sum()), "/", int(pair.size), f"{(pair > 1).mean():.2%}")
print("  各写法月薪中位数:", df.groupby("salary_form").salary_mid_k.median().round(2).to_dict())

# ================================================================ 图：14 个核心城市在两种口径下的同一指标
order = cc.sort_values(("薪酬中位数", "批次 B+C")).index.tolist()
ylab = [f"{c}（{'一线型' if c in TIER1_CITIES else '新一线型'}）" for c in order]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True, sharey=True)
for ax, key, xl, scale in ((axes[0], "薪酬中位数", "折算月薪中位数（千元/月）", 1), (axes[1], "本科占比", "本科及以上占比（%）", 100)):
    va = cc.loc[order, (key, "批次 A")].values * scale
    vb = cc.loc[order, (key, "批次 B+C")].values * scale
    y = np.arange(len(order))
    ax.hlines(y, np.minimum(va, vb), np.maximum(va, vb), color=NEUTRAL, lw=2, zorder=1)
    ax.scatter(vb, y, s=46, color=PALETTE[0], zorder=3, label="批次 B+C（每城约 2.2~2.5 万条）")
    ax.scatter(va, y, s=46, color=PALETTE[1], zorder=3, label="批次 A（每城约 300 条）")
    ax.set_yticks(y, ylab)
    ax.set_xlabel(xl)
    ax.grid(axis="y", visible=False)
axes[0].legend(loc="lower right")
axes[0].set_title("薪酬：F047、K115 在批次 A 中高出一倍以上", fontsize=11)
axes[1].set_title("学历：同样两城的本科占比高出约 35~38 个百分点", fontsize=11)
fig.suptitle("同一城市、同一指标，两种采集口径给出不同的结论", x=0.01, ha="left", fontsize=13, fontweight="bold")
save_fig(fig, "output/figs/E2_scope_compare.png")
print("\n图已保存。")
