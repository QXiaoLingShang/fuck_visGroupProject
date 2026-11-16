# -*- coding: utf-8 -*-
"""成员C · 第二轮 · 数据理解：经验/学历编码“从编码到语义”（假设 H1）的证据链。

频数与行业长尾沿用第一轮的 C_understand.py（未过滤口径）；本脚本只做 H1 的检验（过滤后口径）。
产出表：
  data/processed/C2_code_semantics.csv        编码—语义对照表（学历 9 种、经验 7 种）
  data/processed/C2_industry_level_compare.csv 行业平均门槛等级：第一轮顺序与本轮等级的对比
产出图：
  output/figs/C2_h1_salary_order.png          按语义顺序排列的薪酬中位数与四分位
  output/figs/C2_h1_any_evidence.png          “不限”与“在校/应届”的特殊性
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *
from scipy.stats import spearmanr

plt = apply_style()
df = load_enriched()
N = len(df)
df["ls"] = np.log(df["salary_mid_k"])
print("过滤后行数:", N)

# ------------------------------------------------------------ (a) 编码个数与平台选项个数
tax = json.load(open(os.path.join(ROOT, "data", "reference", "platform_taxonomy.json"), encoding="utf-8"))
raw = load_flagged()
for col, key in (("education", "学历选项"), ("experience", "经验选项")):
    vc = raw[col].value_counts()
    print(f"{col}: 原始编码 {len(vc)} 种，稀有编码(<100) {int((vc < RARE_CATEGORY_MIN).sum())} 种，"
          f"有效 {int((vc >= RARE_CATEGORY_MIN).sum())} 种；平台{key} {len(tax[key])} 个")
print("有效学历编码中剔除 Gw 后:", int((raw["education"].value_counts() >= RARE_CATEGORY_MIN).sum()) - 1)
gw = raw[raw["education"] == "Gw"]
print("Gw:", len(gw), gw["salary_form"].value_counts().to_dict(), gw["experience"].value_counts().to_dict())
print("行业编码数:", raw["company_type"].nunique(), " 平台二级行业:", tax["行业"]["二级个数"])

# 编码频数图（未过滤口径）
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
for ax, col, name in ((axes[0], "experience", "经验要求编码"), (axes[1], "education", "学历要求编码")):
    vc = raw[col].value_counts().sort_values()
    ax.barh(vc.index, vc.values / 1e4, color=PALETTE[0], height=0.68)
    ax.set_xlabel("招聘记录数（万条）")
    ax.set_ylabel(name)
    ax.grid(axis="y", visible=False)
save_fig(fig, "output/figs/C2_code_freq.png")

# ------------------------------------------------------------ 编码—语义对照表
def sem_table(col, label_map, level_map, kind):
    g = df.groupby(col)
    t = pd.DataFrame({
        "kind": kind, "label": pd.Series(label_map), "level": pd.Series(level_map),
        "n": g.size(), "share": g.size() / N,
        "salary_median": g["salary_mid_k"].median(), "salary_p25": g["salary_mid_k"].quantile(.25),
        "salary_p75": g["salary_mid_k"].quantile(.75), "log_sd": g["ls"].std(),
        "parttime_share": g["is_parttime"].mean(),
        "core_city_share": g["city_group"].apply(lambda s: s.isin(["一线型", "新一线型"]).mean()),
    })
    return t.sort_values(["level", "n"], ascending=[True, False]).rename_axis("code")

edu_t = sem_table("education", EDU_LABEL, EDU_LEVEL, "学历")
exp_t = sem_table("experience", EXP_LABEL, EXP_LEVEL, "经验")
sem = pd.concat([edu_t, exp_t])
sem.round(4).to_csv(os.path.join(PROCESSED_DIR, "C2_code_semantics.csv"), encoding="utf-8-sig")
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\n[编码—语义对照]\n", sem.round(3).to_string())
print("全体核心城市占比:", round(df["city_group"].isin(["一线型", "新一线型"]).mean(), 4),
      " 全体兼职类占比:", round(df["is_parttime"].mean(), 4))

# ------------------------------------------------------------ (b) 语义顺序下的单调性
edu_order = ["Gh", "Gy", "Go", "GI", "Gx", "GZ", "Gz"]            # 不含“不限”与噪声 Gw
exp_order = ["Eby", "EKk", "Eas", "Eqh", "EzN", "EaZ"]
print("\n学历中位数（初中→博士）:", edu_t.loc[edu_order, "salary_median"].tolist(),
      "单调不减:", edu_t.loc[edu_order, "salary_median"].is_monotonic_increasing)
print("经验中位数（在校/应届→10年以上）:", exp_t.loc[exp_order, "salary_median"].tolist(),
      "单调不减:", exp_t.loc[exp_order, "salary_median"].is_monotonic_increasing)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True, sharey=True,
                         gridspec_kw={"width_ratios": [8, 7]})
short = {"中专/高中(一)": "中专/高中", "中专/高中(二)": "中专/高中"}
for ax, t, any_code, order, title in (
        (axes[0], edu_t, "GP", edu_order, "学历：去掉“不限”后，薪酬随等级单调上升"),
        (axes[1], exp_t, "EdD", exp_order, "经验：同样单调，“不限”落在中间")):
    codes = [any_code] + order
    x = np.r_[0, np.arange(len(order)) + 1.6]                      # “不限”与等级序列之间留空
    col = [HIGHLIGHT] + [PALETTE[0]] * len(order)
    s = t.loc[codes]
    ax.vlines(x, s["salary_p25"], s["salary_p75"], color=col, linewidth=6, alpha=0.35)
    ax.scatter(x, s["salary_median"], color=col, s=55, zorder=3)
    ax.plot(x[1:], s["salary_median"].values[1:], color=PALETTE[0], linewidth=1.2, alpha=0.6)
    ax.axvline(0.8, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short.get(s.loc[c, 'label'], s.loc[c, 'label'])}\n({c})" for c in codes], fontsize=8.6)
    ax.set_xlim(-0.6, x[-1] + 0.5)
    ax.set_xlabel("学历要求（括号内为原编码）" if any_code == "GP" else "经验要求（括号内为原编码）")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("折算月薪中值（千元/月）")
axes[0].set_ylim(0, 40)
save_fig(fig, "output/figs/C2_h1_salary_order.png")

# ------------------------------------------------------------ (c) “不限”的特殊性
p_any_given_any = df.loc[df["edu_any"], "exp_any"].mean()
p_any_given_req = df.loc[~df["edu_any"], "exp_any"].mean()
print(f"\n学历不限中经验不限占比 {p_any_given_any:.2%}；有学历要求的记录中 {p_any_given_req:.2%}")
print(f"经验不限中学历不限占比 {df.loc[df['exp_any'], 'edu_any'].mean():.2%}；"
      f"有经验要求的记录中 {df.loc[~df['exp_any'], 'edu_any'].mean():.2%}")
co = df[df["education"] != "Gw"].groupby("education")["exp_any"].mean().reindex(["GP"] + edu_order)
print("[各学历编码中经验不限占比]\n", co.round(4).to_dict())

# (d) 在校/应届（Eby）
eby = df[df["experience"] == "Eby"]
print("Eby 记录数:", len(eby), " 薪酬写法:", eby["salary_form"].value_counts(normalize=True).round(4).to_dict())
print("Eby 城市组:", eby["city_group"].value_counts(normalize=True).round(4).to_dict())
pt = df.groupby("experience")["is_parttime"].mean().reindex(["EdD"] + exp_order)
print("[各经验编码的日薪/时薪占比]\n", pt.round(4).to_dict())

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1, 1]})
# 左：各学历编码中“经验不限”的占比
ax = axes[0]
labs = [short.get(EDU_LABEL[c], EDU_LABEL[c]) + f" ({c})" for c in co.index][::-1]
ax.barh(labs, co.values[::-1] * 100, color=([HIGHLIGHT] + [NEUTRAL] * len(edu_order))[::-1], height=0.68)
ax.text(co.iloc[0] * 100 - 1.5, len(co) - 1, f"{co.iloc[0]:.1%}", va="center", ha="right", fontsize=9.5, color="white")
ax.set_xlim(0, 70)
ax.set_xlabel("其中“经验不限”(EdD) 的占比（%）")
ax.grid(axis="y", visible=False)
# 中：薪酬中位数相近（5.5~7.5 千元/月）的编码的离散度
ax = axes[1]
near = sem[(sem["salary_median"] >= 5.5) & (sem["salary_median"] <= 7.5)].sort_values("log_sd")
print("[薪酬中位数 5.5~7.5 的编码的对数月薪标准差]", near["log_sd"].round(3).to_dict())
labs = [short.get(r["label"], r["label"]) + f" ({c})" for c, r in near.iterrows()]
ax.barh(labs, near["log_sd"], color=[HIGHLIGHT if c in ("GP", "EdD") else NEUTRAL for c in near.index], height=0.68)
for y, (c, v) in enumerate(near["log_sd"].items()):
    if c in ("GP", "EdD"):
        ax.text(v - 0.008, y, f"{v:.3f}", va="center", ha="right", fontsize=9.5, color="white")
ax.set_xlim(0, 0.6)
ax.set_xlabel("对数月薪的标准差")
ax.grid(axis="y", visible=False)
# 右：各经验编码的日薪/时薪占比
ax = axes[2]
labs = [EXP_LABEL[c] + f" ({c})" for c in pt.index][::-1]
ax.barh(labs, pt.values[::-1] * 100, color=[HIGHLIGHT if c == "Eby" else NEUTRAL for c in pt.index][::-1], height=0.68)
ax.text(pt["Eby"] * 100 + 1, len(pt) - 2, f"{pt['Eby']:.1%}", va="center", fontsize=9.5, color=INK)
ax.set_xlim(0, 60)
ax.set_xlabel("日薪/时薪岗位的占比（%）")
ax.grid(axis="y", visible=False)
save_fig(fig, "output/figs/C2_h1_any_evidence.png")

# ------------------------------------------------------------ 第一轮顺序与本轮等级的差异（行业平均等级）
df["edu_rank_old"] = df["education"].map(EDU_RANK)
df["exp_rank_old"] = df["experience"].map(EXP_RANK)
g = df.groupby("company_type")
cmp_ = pd.DataFrame({
    "n": g.size(), "edu_old": g["edu_rank_old"].mean(), "edu_new": g["edu_level"].mean(),
    "exp_old": g["exp_rank_old"].mean(), "exp_new": g["exp_level"].mean(),
    "edu_any_share": g["edu_any"].mean(), "exp_any_share": g["exp_any"].mean(),
})
for k in ("edu", "exp"):
    ro, rn = cmp_[f"{k}_old"].rank(ascending=False), cmp_[f"{k}_new"].rank(ascending=False)
    cmp_[f"{k}_rank_old"], cmp_[f"{k}_rank_new"] = ro, rn
    d = (ro - rn).abs()
    top = d.idxmax()
    print(f"\n{k}: 新旧行业平均等级 Spearman {spearmanr(cmp_[f'{k}_old'], cmp_[f'{k}_new'])[0]:.3f}；"
          f"名次变化≥10 的行业 {int((d >= 10).sum())}/{len(d)}，最大 {d.max():.0f} 位（{top}: {ro[top]:.0f}→{rn[top]:.0f}，"
          f"n={cmp_.loc[top, 'n']}）")
    print(f"   新平均等级与“不限”占比的 Spearman: {spearmanr(cmp_[f'{k}_new'], cmp_[f'{k}_any_share'])[0]:.3f}")
cmp_.round(4).to_csv(os.path.join(PROCESSED_DIR, "C2_industry_level_compare.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------ 行业长尾与企业—行业（未过滤口径，数字与第一轮一致）
ind = raw["company_type"].value_counts()
cum = ind.cumsum() / len(raw)
k = raw.groupby("company")["company_type"].nunique()
print(f"\n行业 {len(ind)} 种；Top10 {cum.iloc[9]:.1%}，Top20 {cum.iloc[19]:.1%}；"
      f"对应 ≥2 个行业的企业 {(k >= 2).sum()}/{len(k)} = {(k >= 2).mean():.2%}")
