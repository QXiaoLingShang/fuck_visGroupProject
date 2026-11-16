# -*- coding: utf-8 -*-
"""成员D · 第二轮公共函数：采集块/查询段的标注与城市画像指标（供 D2_*.py 共用）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich import *  # noqa: F401,F403

# 14 个核心城市按“两个批次各覆盖哪一个查询段”分成三个采集组（由 D2_core.py 第 1 步的聚类结果核对）
CRAWL_GROUP = {"G1": ["F047", "P949", "F379", "O198", "S959"],
               "G2": ["K115", "B170", "N806", "R892", "V449"],
               "G3": ["Y232", "Y633", "S542", "V932"]}
CITY2CG = {c: g for g, cs in CRAWL_GROUP.items() for c in cs}
CORE = sum(CRAWL_GROUP.values(), [])
SEG_OF = {("G1", "B"): "S1", ("G1", "C"): "S2", ("G2", "B"): "S2",
          ("G2", "C"): "S3", ("G3", "B"): "S3", ("G3", "C"): "S1"}
SEGS = ["S1", "S2", "S3"]
ANOMALY_EDU_ANY = 0.15     # 批次 A 中“学历不限”占比低于 15% 的块判为采集条件异常
MIN_CITY_N = 200           # 少于 200 条的城市只展示、不参与分型


def load_d2():
    """load_enriched() + 批次 A 的块编号与异常块标记 + 核心城市批次的采集组、查询段。"""
    df = load_enriched().copy()
    a = df["batch"].eq("A")
    df["a_block"] = np.where(a, (df["city"] != df["city"].shift()).cumsum(), -1)   # 清洗不改变行序
    g = df[a].groupby("a_block")
    blk = pd.DataFrame({"city": g["city"].first(), "n": g.size(), "edu_any": g["edu_any"].mean()})
    bad = blk.index[(blk["n"] >= 100) & (blk["edu_any"] < ANOMALY_EDU_ANY)]
    df["a_anomaly"] = df["a_block"].isin(bad)
    core_bc = ~a & df["city"].isin(CORE)
    df["crawl_group"] = df["city"].map(CITY2CG).where(core_bc)
    df["seg"] = pd.Series(list(zip(df["crawl_group"], df["batch"])), index=df.index).map(SEG_OF).where(core_bc)
    return df


def city_metrics(d: pd.DataFrame) -> pd.DataFrame:
    """按城市汇总画像指标（d 为某一口径下的记录）。"""
    g = d.groupby("city")
    ci = pd.crosstab(d["city"], d["company_type"])
    sh = ci.div(ci.sum(1), axis=0)
    return pd.DataFrame({
        "n": g.size(),
        "salary_median": g["salary_mid_k"].median(),
        "salary_p25": g["salary_mid_k"].quantile(.25),
        "salary_p75": g["salary_mid_k"].quantile(.75),
        "salary_gmean": np.exp(g["salary_mid_k"].apply(lambda s: np.log(s).mean())),   # 几何均值
        "bachelor_share": g["edu_level"].apply(lambda s: (s >= 4).mean()),
        "edu_any_share": g["edu_any"].mean(),
        "exp_any_share": g["exp_any"].mean(),
        "exp3plus_share": g["exp_level"].apply(lambda s: (s >= 4).mean()),             # 3-5 年及以上
        "bonus_share": g["has_bonus_months"].mean(),
        "parttime_share": g["is_parttime"].mean(),
        "n_company": g["company"].nunique(),
        "industry_hhi": (sh ** 2).sum(1),
    })


def short(ind: str) -> str:
    return ind.replace("type_", "")
