# -*- coding: utf-8 -*-
"""全组共享的基础模块：数据加载、薪酬解析、质量标记、统一图表风格。

所有成员（子任务）的脚本都必须从这里取数与取样式，保证口径一致：
    from common import load_raw, load_flagged, load_clean, apply_style, PALETTE, save_fig
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_XLSX = os.path.join(ROOT, "data", "raw", "JobWanted.xlsx")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(ROOT, "output")
for _d in (CACHE_DIR, PROCESSED_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------
# 1. 数据加载
# ----------------------------------------------------------------------------
COLUMNS_CN = {
    "job_title": "职位名称", "city": "行政区划", "salary": "薪酬",
    "experience": "经验要求", "education": "学历要求",
    "company": "企业", "company_type": "行业类别",
}


def load_raw() -> pd.DataFrame:
    """读取原始数据（430,664 行 × 7 列，全部为字符串）。首次读取 xlsx 后缓存为 pickle。"""
    cache = os.path.join(CACHE_DIR, "raw.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    df = pd.read_excel(RAW_XLSX, dtype=str)
    df.to_pickle(cache)
    return df


# ----------------------------------------------------------------------------
# 2. 薪酬解析（统一折算为 “千元/月”）
# ----------------------------------------------------------------------------
WORK_DAYS_PER_MONTH = 21.75   # 法定月计薪天数
WORK_HOURS_PER_DAY = 8
WEEKS_PER_MONTH = 4.35

# 薪酬写法类别 -> 中文名
SALARY_FORMS = {
    "month_k": "月薪(K)", "month_k_n": "月薪(K)·N薪", "month_yuan": "元/月",
    "day": "元/天", "hour": "元/时", "week": "元/周", "piece": "元/单",
    "open_k": "NK以上", "negotiable": "面议", "other": "其他",
}

_RE_K_N = re.compile(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)K·(\d+)薪$")
_RE_K = re.compile(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)K$")
_RE_UNIT = re.compile(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)元/(月|天|时|周|单)$")
_RE_OPEN = re.compile(r"^(\d+(?:\.\d+)?)K以上$")
_UNIT_FORM = {"月": "month_yuan", "天": "day", "时": "hour", "周": "week", "单": "piece"}
_UNIT_FACTOR = {  # 原单位(元) -> 千元/月
    "月": 1 / 1000,
    "天": WORK_DAYS_PER_MONTH / 1000,
    "时": WORK_HOURS_PER_DAY * WORK_DAYS_PER_MONTH / 1000,
    "周": WEEKS_PER_MONTH / 1000,
}


def parse_salary_str(s: str):
    """解析单个薪酬字符串。

    返回 (form, raw_min, raw_max, months, min_k, max_k)
      form     : 写法类别（见 SALARY_FORMS）
      raw_min/raw_max : 原始数值（原单位）
      months   : 年发薪月数（未注明为 12）
      min_k/max_k : 折算后的月薪下/上限（千元/月）；无法折算为 NaN
    """
    nan = np.nan
    s = str(s).strip()
    m = _RE_K.match(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return "month_k", a, b, 12, a, b
    m = _RE_K_N.match(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return "month_k_n", a, b, int(m.group(3)), a, b
    m = _RE_UNIT.match(s)
    if m:
        a, b, u = float(m.group(1)), float(m.group(2)), m.group(3)
        if u == "单":  # 计件，无法折算为月薪
            return "piece", a, b, 12, nan, nan
        f = _UNIT_FACTOR[u]
        return _UNIT_FORM[u], a, b, 12, a * f, b * f
    m = _RE_OPEN.match(s)
    if m:
        a = float(m.group(1))
        return "open_k", a, nan, 12, a, nan
    if s == "面议":
        return "negotiable", nan, nan, 12, nan, nan
    return "other", nan, nan, 12, nan, nan


def add_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 df 上追加薪酬派生列（按唯一值解析后映射，速度快）。"""
    uniq = df["salary"].dropna().unique()
    parsed = pd.DataFrame(
        [parse_salary_str(u) for u in uniq], index=uniq,
        columns=["salary_form", "salary_raw_min", "salary_raw_max",
                 "salary_months", "salary_min_k", "salary_max_k"],
    )
    out = df.join(parsed, on="salary")
    out["salary_mid_k"] = (out["salary_min_k"] + out["salary_max_k"]) / 2      # 月薪中值
    out["salary_span_k"] = out["salary_max_k"] - out["salary_min_k"]           # 薪酬带宽
    out["salary_ratio"] = out["salary_max_k"] / out["salary_min_k"]            # 上限/下限
    out["salary_annual_k"] = out["salary_mid_k"] * out["salary_months"]        # 年薪(千元)
    out["has_bonus_months"] = out["salary_months"] > 12                        # 是否 13 薪及以上
    return out


# ----------------------------------------------------------------------------
# 3. 质量标记与过滤规则（第一轮设计，全组统一口径）
# ----------------------------------------------------------------------------
RARE_CATEGORY_MIN = 100     # 经验/学历编码出现次数 < 100 视为噪声编码
RARE_INDUSTRY_MIN = 30      # 行业类别出现次数 < 30 视为样本不足的稀有行业
SALARY_MIN_K = 1.0          # 折算月薪中值下限（千元/月）
SALARY_MAX_K = 100.0        # 折算月薪中值上限（千元/月）
SALARY_RATIO_MAX = 4.0      # 薪酬上限/下限 的最大合理倍数

FLAG_COLUMNS = {
    "flag_duplicate": "完全重复记录",
    "flag_salary_unparsed": "薪酬无法折算为月薪（面议/计件/开放区间）",
    "flag_salary_range": "薪酬数值异常（下限>上限、非正、折算月薪越界）",
    "flag_salary_ratio": "薪酬区间过宽（上限/下限 > 4 倍）",
    "flag_rare_experience": "经验要求为噪声编码（出现 < 100 次）",
    "flag_rare_education": "学历要求为噪声编码（出现 < 100 次）",
    "flag_rare_industry": "行业类别样本不足（出现 < 30 次）",
}


def load_flagged() -> pd.DataFrame:
    """原始数据 + 薪酬派生列 + 各类质量标记列（不删除任何行）。"""
    cache = os.path.join(CACHE_DIR, "flagged.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    df = add_salary_features(load_raw())
    df["flag_duplicate"] = df.duplicated(subset=list(COLUMNS_CN), keep="first")
    df["flag_salary_unparsed"] = df["salary_mid_k"].isna()
    bad_range = (
        (df["salary_min_k"] <= 0)
        | (df["salary_min_k"] > df["salary_max_k"])
        | (df["salary_mid_k"] < SALARY_MIN_K)
        | (df["salary_mid_k"] > SALARY_MAX_K)
    )
    df["flag_salary_range"] = bad_range & ~df["flag_salary_unparsed"]
    df["flag_salary_ratio"] = (df["salary_ratio"] > SALARY_RATIO_MAX) & ~df["flag_salary_unparsed"]
    for col, flag, thr in (
        ("experience", "flag_rare_experience", RARE_CATEGORY_MIN),
        ("education", "flag_rare_education", RARE_CATEGORY_MIN),
        ("company_type", "flag_rare_industry", RARE_INDUSTRY_MIN),
    ):
        vc = df[col].map(df[col].value_counts())
        df[flag] = vc < thr
    df["flag_any"] = df[list(FLAG_COLUMNS)].any(axis=1)
    df.to_pickle(cache)
    return df


def load_clean() -> pd.DataFrame:
    """过滤后的分析数据集：去掉任一质量标记为 True 的记录，并丢弃标记列。"""
    df = load_flagged()
    clean = df.loc[~df["flag_any"]].drop(columns=list(FLAG_COLUMNS) + ["flag_any"])
    return clean.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. 统一图表风格
# ----------------------------------------------------------------------------
# 分类色（固定顺序使用，不循环；超过 8 类请合并为“其他”或用小多图）
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
# 顺序色（单色相，浅->深），用于热力图等数值大小编码
SEQ_BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# 发散色两极 + 中性灰中点
DIV_COLORS = ["#2a78d6", "#f0efec", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
HIGHLIGHT = "#eb6834"      # 强调色（突出某一类，其余用 NEUTRAL）
NEUTRAL = "#c3c2b7"


def seq_cmap(name="seq_blues"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQ_BLUES)


def div_cmap(name="div_blue_red"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, DIV_COLORS)


def apply_style():
    """设置全局 matplotlib 样式（中文字体、细线、弱化网格）。在每个绘图脚本开头调用一次。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10.5, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2, "patch.linewidth": 0,
        "text.color": INK,
    })
    return plt


def save_fig(fig, path):
    """保存图片（200 dpi PNG）并关闭。path 为绝对路径或相对于项目根目录的路径。"""
    import matplotlib.pyplot as plt
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path

# ----------------------------------------------------------------------------
# 5. 全组统一的有序编码与城市分级（第一轮设计的约定）
# ----------------------------------------------------------------------------
# 经验/学历均为脱敏编码，真实含义未知。这里按“清洗后数据的平均折算月薪由低到高”
# 给出一个数据驱动的等级顺序（rank 从 1 开始），仅是推断，文档中须注明。
EXP_ORDER = ["Eby", "EKk", "EdD", "Eas", "Eqh", "EzN", "EaZ"]
EDU_ORDER = ["Gw", "Gh", "Gy", "GP", "Go", "GI", "Gx", "GZ", "Gz"]
EXP_RANK = {c: i + 1 for i, c in enumerate(EXP_ORDER)}
EDU_RANK = {c: i + 1 for i, c in enumerate(EDU_ORDER)}

CITY_TIER_ORDER = ["核心城市", "中等城市", "普通城市", "小样本城市"]


def city_tier_map(df: pd.DataFrame) -> pd.Series:
    """按招聘量给行政区划分级：>5000 核心；400~5000 中等；200~400 普通；<200 小样本。"""
    vc = df["city"].value_counts()
    tier = pd.cut(vc, bins=[0, 199, 400, 5000, np.inf],
                  labels=["小样本城市", "普通城市", "中等城市", "核心城市"])
    return tier.astype(str)


if __name__ == "__main__":
    f = load_flagged()
    print("raw rows:", len(f))
    print(f["salary_form"].value_counts())
    print(f[list(FLAG_COLUMNS) + ["flag_any"]].sum())
    c = load_clean()
    print("clean rows:", len(c), f"({len(c) / len(f):.2%})")
    print(c["salary_mid_k"].describe())
