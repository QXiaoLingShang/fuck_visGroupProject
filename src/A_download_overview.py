# -*- coding: utf-8 -*-
"""成员 A：数据下载 + 数据集整体概览。

产出：
  output/figs/A_seven_steps.png        七步设计方法与本阶段范围
  output/figs/A_file_read.png          文件大小与读取耗时
  output/figs/A_cardinality.png        各字段唯一值个数（对数轴）
  output/figs/A_concentration.png      各字段取值集中度
  output/figs/A_field_question.png     字段—分析问题对应矩阵
  data/processed/A_file_inventory.csv  下载文件清单
  data/processed/A_read_timing.csv     读取耗时实测
  data/processed/A_field_overview.csv  字段概览表
  data/processed/A_concentration.csv   字段集中度指标
用法：python src/A_download_overview.py [--retime]   （--retime 重新实测读取耗时）
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from common import *   # noqa
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap

plt = apply_style()
RAW_DIR = os.path.join(ROOT, "data", "raw")
MB = 1024 * 1024


def p(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. 下载文件清单
# ---------------------------------------------------------------------------
purpose = {
    "JobWanted.xlsx": "招聘数据主文件",
    "JobWanted_description.docx": "官方数据说明（字段含义）",
    "second_answerSheet.docx": "答卷模板（5 个问题）",
    "QA.docx": "问答文档",
}
inv = []
for fn in sorted(os.listdir(RAW_DIR)):
    size = os.path.getsize(os.path.join(RAW_DIR, fn))
    with open(os.path.join(RAW_DIR, fn), "rb") as f:
        head = f.read(4)
    inv.append({"文件名": fn, "字节数": size, "大小MB": round(size / MB, 3),
                "大小KB": round(size / 1024, 1), "是否有效文件": head[:2] == b"PK",
                "用途": purpose.get(fn, "")})
inv = pd.DataFrame(inv)
inv.to_csv(os.path.join(PROCESSED_DIR, "A_file_inventory.csv"), index=False, encoding="utf-8-sig")
p("== 文件清单 ==\n", inv.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. 读取耗时实测：xlsx（1 次） vs pickle（5 次取中位数）
# ---------------------------------------------------------------------------
timing_csv = os.path.join(PROCESSED_DIR, "A_read_timing.csv")
raw_pkl = os.path.join(CACHE_DIR, "raw.pkl")
df = load_raw()                               # 保证缓存已存在
if "--retime" in sys.argv or not os.path.exists(timing_csv):
    t0 = time.perf_counter()
    _x = pd.read_excel(RAW_XLSX, dtype=str)   # 直接解析 xlsx
    t_xlsx = time.perf_counter() - t0
    assert _x.equals(df), "xlsx 与 pickle 缓存内容不一致"
    del _x
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        pd.read_pickle(raw_pkl)
        ts.append(time.perf_counter() - t0)
    timing = pd.DataFrame({"方式": ["read_excel 读取 xlsx", "read_pickle 读取缓存"],
                           "耗时秒": [round(t_xlsx, 2), round(float(np.median(ts)), 3)],
                           "重复次数": [1, 5]})
    timing.to_csv(timing_csv, index=False, encoding="utf-8-sig")
timing = pd.read_csv(timing_csv)
t_xlsx, t_pkl = timing["耗时秒"]
p("== 读取耗时 ==\n", timing.to_string(index=False), f"\n加速倍数 {t_xlsx / t_pkl:.1f}")

# ---------------------------------------------------------------------------
# 3. 数据集整体概览
# ---------------------------------------------------------------------------
mem_mb = df.memory_usage(deep=True).sum() / MB
p(f"== 规模 == 行 {len(df):,} 列 {df.shape[1]} 内存 {mem_mb:.1f} MB "
  f"完全重复行 {df.duplicated().sum():,}")
role = {"job_title": "名义", "city": "名义", "salary": "数量（文本区间）",
        "experience": "有序", "education": "有序", "company": "名义", "company_type": "名义"}
rows, conc = [], []
for c in df.columns:
    s = df[c]
    vc = s.value_counts()
    share = vc / len(df)
    rows.append({"字段": c, "中文含义": COLUMNS_CN[c], "类型角色": role[c],
                 "唯一值个数": s.nunique(), "示例值": vc.index[0],
                 "空值数": int(s.isna().sum()),
                 "空串数": int((s.fillna("").str.strip() == "").sum()),
                 "字符长度范围": f"{s.str.len().min()}-{s.str.len().max()}"})
    half = int((share.cumsum() < 0.5).sum()) + 1      # 覆盖 50% 记录所需的取值个数
    conc.append({"字段": c, "中文含义": COLUMNS_CN[c], "唯一值个数": len(vc),
                 "Top1占比%": round(share.iloc[0] * 100, 2),
                 "Top10占比%": round(share.iloc[:10].sum() * 100, 2),
                 "覆盖50%记录所需取值数": half,
                 "仅出现1次的取值数": int((vc == 1).sum()),
                 "仅出现1次的取值占比%": round((vc == 1).mean() * 100, 2),
                 "平均每个取值记录数": round(len(df) / len(vc), 2)})
ov = pd.DataFrame(rows)
conc = pd.DataFrame(conc)
ov.to_csv(os.path.join(PROCESSED_DIR, "A_field_overview.csv"), index=False, encoding="utf-8-sig")
conc.to_csv(os.path.join(PROCESSED_DIR, "A_concentration.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
p("== 字段概览 ==\n", ov.to_string(index=False))
p("== 集中度 ==\n", conc.to_string(index=False))
p("== 前 6 行 ==\n", df.head(6).to_string())
fl = load_flagged()
keep = 1 - fl["flag_any"].mean()
p(f"== 过滤后保留比例 {keep:.2%}（{int((~fl['flag_any']).sum()):,} 行）")

# ---------------------------------------------------------------------------
# 图 1：七步设计方法与本阶段范围
# ---------------------------------------------------------------------------
steps = ["数据下载", "数据理解", "数据过滤", "数据处理", "图形映射", "美化装饰", "交互"]
notes = [
    "官网下载 xlsx\n与说明文档\npickle 缓存加速",
    f"{len(df):,} 行 × 7 列\n字段基数与分布\n薪酬 {ov.set_index('字段').loc['salary', '唯一值个数']:,} 种写法",
    f"7 类质量标记\n剔除噪声记录\n保留 {keep:.2%}",
    "薪酬折算为\n千元/月\n多维统计汇总表",
]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.set_xlim(0, 7 * 1.5 + 0.1); ax.set_ylim(0.65, 3.6); ax.axis("off")
bw, bh, gap, y0 = 1.14, 0.78, 0.36, 2.0
for i, st in enumerate(steps):
    x = 0.08 + i * (bw + gap)
    done = i < 4
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=PALETTE[0] if done else NEUTRAL, ec="none"))
    ax.text(x + bw / 2, y0 + bh / 2 + 0.13, f"第 {i + 1} 步", ha="center", va="center",
            fontsize=8.5, color="white" if done else INK_2)
    ax.text(x + bw / 2, y0 + bh / 2 - 0.13, st, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white" if done else INK)
    if i < 6:   # 步骤之间的箭头
        ax.annotate("", xy=(x + bw + gap - 0.05, y0 + bh / 2), xytext=(x + bw + 0.05, y0 + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    if done:
        ax.plot([x + bw / 2] * 2, [y0 - 0.08, y0 - 0.3], color=AXIS, lw=0.8)
        ax.text(x + bw / 2, y0 - 0.38, notes[i], ha="center", va="top", fontsize=9,
                color=INK_2, linespacing=1.55)
# 阶段范围括注
x_end4 = 0.08 + 3 * (bw + gap) + bw
ax.plot([0.08, x_end4], [3.05, 3.05], color=PALETTE[0], lw=2)
ax.text(0.08, 3.14, "第一阶段：本次作业完成（第一轮设计）", fontsize=10, color=INK, va="bottom")
x5 = 0.08 + 4 * (bw + gap)
ax.plot([x5, x5 + 2 * (bw + gap) + bw], [3.05, 3.05], color=NEUTRAL, lw=2)
ax.text(x5, 3.14, "后续阶段：可视化设计与系统实现", fontsize=10, color=INK_2, va="bottom")
ax.text(x5 + (2 * (bw + gap) + bw) / 2, y0 - 0.38, "基于本阶段产出的清洗数据\n与统计表继续开展",
        ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.55)
ax.set_title("可视化七步设计方法与本阶段范围")
save_fig(fig, "output/figs/A_seven_steps.png")

# ---------------------------------------------------------------------------
# 图 2：文件大小与读取耗时（两个量纲 -> 并排子图）
# ---------------------------------------------------------------------------
xlsx_mb = os.path.getsize(RAW_XLSX) / MB
pkl_mb = os.path.getsize(raw_pkl) / MB
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
lab = ["原始 xlsx 文件", "pickle 缓存文件", "读入内存后的 DataFrame"]
val = [xlsx_mb, pkl_mb, mem_mb]
a1.barh(lab[::-1], val[::-1], color=PALETTE[0], height=0.55)
for yy, v in zip(range(3), val[::-1]):
    a1.text(v + max(val) * 0.015, yy, f"{v:.1f} MB", va="center", fontsize=9.5, color=INK_2)
a1.set_xlim(0, max(val) * 1.2); a1.set_xlabel("占用空间（MB）"); a1.grid(axis="y", visible=False)
a1.set_title("同一份数据在三种形态下的体积")
lab2 = ["read_excel 解析 xlsx\n（首次）", "read_pickle 读缓存\n（之后每次）"]
a2.barh(lab2[::-1], [t_pkl, t_xlsx], color=[HIGHLIGHT, NEUTRAL], height=0.5)
a2.text(t_xlsx + t_xlsx * 0.015, 1, f"{t_xlsx:.1f} 秒", va="center", fontsize=9.5, color=INK_2)
a2.text(t_pkl + t_xlsx * 0.015, 0, f"{t_pkl:.2f} 秒，快约 {t_xlsx / t_pkl:.0f} 倍", va="center",
        fontsize=9.5, color=INK_2)
a2.set_xlim(0, t_xlsx * 1.22); a2.set_xlabel("读取耗时（秒）"); a2.grid(axis="y", visible=False)
a2.set_title("缓存后读取耗时大幅缩短")
save_fig(fig, "output/figs/A_file_read.png")

# ---------------------------------------------------------------------------
# 图 3：各字段唯一值个数（对数轴横向条形）
# ---------------------------------------------------------------------------
o = ov.sort_values("唯一值个数")
fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
ylab = [f"{c}（{n}）" for c, n in zip(o["字段"], o["中文含义"])]
ax.barh(ylab, o["唯一值个数"], color=PALETTE[0], height=0.62)
for i, v in enumerate(o["唯一值个数"]):
    if v > 1e5:   # 大数值标在条内，避免与参考线重叠
        ax.text(v / 1.12, i, f"{v:,}", va="center", ha="right", fontsize=9.5, color="white")
    else:
        ax.text(v * 1.12, i, f"{v:,}", va="center", fontsize=9.5, color=INK_2)
ax.set_xscale("log"); ax.set_xlim(1, 1e6)
ax.set_xlabel("唯一值个数（对数轴）"); ax.grid(axis="y", visible=False)
save_fig(fig, "output/figs/A_cardinality.png")

# ---------------------------------------------------------------------------
# 图 4：取值集中度：Top1/Top10 占比 + 累计占比曲线
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.05]})
cc = conc.sort_values("Top10占比%")
y = np.arange(len(cc)); h = 0.36
a1.barh(y + h / 2 + 0.02, cc["Top10占比%"], height=h, color=PALETTE[0], label="前 10 个取值合计")
a1.barh(y - h / 2 - 0.02, cc["Top1占比%"], height=h, color=PALETTE[1], label="第 1 大取值")
for i, (t1, t10) in enumerate(zip(cc["Top1占比%"], cc["Top10占比%"])):
    a1.text(t10 + 1.5, i + h / 2 + 0.02, f"{t10:.1f}%", va="center", fontsize=8.5, color=INK_2)
    a1.text(t1 + 1.5, i - h / 2 - 0.02, f"{t1:.1f}%", va="center", fontsize=8.5, color=INK_2)
a1.set_yticks(y); a1.set_yticklabels(cc["字段"])
a1.set_xlim(0, 118); a1.set_xticks(range(0, 101, 20))
a1.set_xlabel("占全部记录的比例（%）"); a1.grid(axis="y", visible=False)
a1.legend(loc="lower right")
a1.set_title("头部取值占比")
curve_cols = ["company_type", "city", "salary", "job_title", "company"]
ends = []
for k, c in enumerate(curve_cols):
    cum = df[c].value_counts().cumsum().to_numpy() / len(df) * 100
    xs = np.arange(1, len(cum) + 1)
    a2.plot(xs, cum, color=PALETTE[k], lw=2, label=c)
a2.set_xscale("log"); a2.set_xlim(1, 4e5); a2.set_ylim(0, 102)
a2.axhline(50, color=MUTED, lw=1)
a2.text(120, 48, "覆盖 50% 记录", fontsize=8.5, color=MUTED, va="top")
a2.set_xlabel("按频次从高到低排序的取值个数（对数轴）"); a2.set_ylabel("累计记录占比（%）")
a2.legend(loc="upper left", bbox_to_anchor=(0.43, 0.88))
a2.set_title("累计占比曲线：越靠右尾部越长")
save_fig(fig, "output/figs/A_concentration.png")

# ---------------------------------------------------------------------------
# 图 5：字段—分析问题对应矩阵（2=核心字段，1=辅助字段）
# ---------------------------------------------------------------------------
qs = ["问题1\n职位差异度", "问题2\n职位画像", "问题3\n薪酬模式", "问题4\n地域画像", "问题5\n行业动态"]
mat = pd.DataFrame(
    [[2, 2, 2, 2, 2],    # job_title
     [1, 2, 2, 2, 1],    # city
     [2, 2, 2, 1, 2],    # salary
     [2, 2, 1, 1, 1],    # experience
     [2, 2, 1, 1, 1],    # education
     [1, 1, 1, 1, 2],    # company
     [2, 1, 2, 2, 2]],   # company_type
    index=list(df.columns), columns=qs)
mat.to_csv(os.path.join(PROCESSED_DIR, "A_field_question_matrix.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8.6, 5), constrained_layout=True)
cmap = ListedColormap([SEQ_BLUES[0], SEQ_BLUES[7]])
ax.pcolormesh(mat.to_numpy(), cmap=cmap, vmin=1, vmax=2, edgecolors=SURFACE, linewidth=3)
ax.invert_yaxis()
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.iat[i, j]
        ax.text(j + 0.5, i + 0.5, "核心" if v == 2 else "辅助", ha="center", va="center", fontsize=10,
                color="white" if v == 2 else INK_2)
ax.set_xticks(np.arange(5) + 0.5); ax.set_xticklabels(qs)
ax.set_yticks(np.arange(7) + 0.5); ax.set_yticklabels([f"{c}（{COLUMNS_CN[c]}）" for c in mat.index])
ax.xaxis.tick_top(); ax.tick_params(length=0)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("字段与 5 个分析问题的对应关系（深色为核心字段，浅色为辅助字段）", pad=14)
save_fig(fig, "output/figs/A_field_question.png")
p("== 矩阵核心字段数/问题 ==", (mat == 2).sum().tolist(), "每字段核心次数", (mat == 2).sum(axis=1).to_dict())
p("done")
