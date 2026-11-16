# -*- coding: utf-8 -*-
"""第二轮设计的公共模块：在 common.py 的清洗结果上追加“语义假设”与“隐含结构”。

全组第二轮脚本统一从这里取数：
    from enrich import *        # 同时带出 common.py 的全部内容
    df = load_enriched()        # 清洗后的 429,086 行 + 下列新增列

新增列（都建立在假设之上，假设及证据见 output/sections/A.md 的“分析假设”一节）
    batch        采集批次 A / B / C（由原始文件行序推断，见 infer_batches）
    edu_label    学历要求的语义标签（假设 H1）      edu_level  学历门槛等级 0~6
    exp_label    经验要求的语义标签（假设 H1）      exp_level  经验门槛等级 0~6
    edu_any / exp_any   是否为“学历不限”/“经验不限”
    city_group   城市分组：一线型 / 新一线型 / 中等城市 / 普通城市 / 小样本城市（假设 H2）
    is_parttime  是否日薪/时薪/周薪岗位（视为兼职、实习类）
    job_family   职位族编号 J01、J02…（假设 H3，仅批次 B、C 的核心城市记录有值，其余为 None）
    query_segment  查询段 S1/S2/S3：职位类目表被分成三段采集，每个核心城市的批次 B、C 各对应一段，
                   即每城只覆盖三段中的两段（仅批次 B、C 的核心城市记录有值）
    collect_group  采集组 G1/G2/G3：覆盖相同两段的核心城市为一组（非核心城市为 None）
    batch_a_anomaly  批次 A 中采集条件明显不同的异常块（学历不限占比低于 15%，共 6 块）
"""
import os

import numpy as np
import pandas as pd

from common import *  # noqa: F401,F403
from common import CACHE_DIR, PROCESSED_DIR, load_flagged, FLAG_COLUMNS, city_tier_map

# ----------------------------------------------------------------------------
# H1 经验 / 学历编码的语义假设
# ----------------------------------------------------------------------------
# 依据：有效编码个数（经验 7、学历 8）与主流招聘平台的选项个数一致；薪酬随等级单调；
# “不限”类编码的记录数最多、薪酬离散度最大，且“学历不限”与“经验不限”高度共现。
# 这是我们的推断，不是官方含义。Go 与 Gy 无法区分先后，合并为同一门槛等级。
EDU_LABEL = {
    "GP": "学历不限", "Gh": "初中及以下", "Gy": "中专/高中(一)", "Go": "中专/高中(二)",
    "GI": "大专", "Gx": "本科", "GZ": "硕士", "Gz": "博士", "Gw": "实习标签(噪声)",
}
EDU_LEVEL = {"GP": 0, "Gh": 1, "Gy": 2, "Go": 2, "GI": 3, "Gx": 4, "GZ": 5, "Gz": 6, "Gw": 0}
EDU_LEVEL_NAME = ["学历不限", "初中及以下", "中专/高中", "大专", "本科", "硕士", "博士"]

EXP_LABEL = {
    "EdD": "经验不限", "Eby": "在校/应届", "EKk": "1年以内", "Eas": "1-3年",
    "Eqh": "3-5年", "EzN": "5-10年", "EaZ": "10年以上",
}
EXP_LEVEL = {"EdD": 0, "Eby": 1, "EKk": 2, "Eas": 3, "Eqh": 4, "EzN": 5, "EaZ": 6}
EXP_LEVEL_NAME = ["经验不限", "在校/应届", "1年以内", "1-3年", "3-5年", "5-10年", "10年以上"]

# ----------------------------------------------------------------------------
# H2 城市分组
# ----------------------------------------------------------------------------
# 14 个核心城市按薪酬中位数与本科占比分成两组（两项指标在第 4 与第 5 名之间都有断层）。
TIER1_CITIES = ["F047", "K115", "P949", "B170"]
CITY_GROUP_ORDER = ["一线型", "新一线型", "中等城市", "普通城市", "小样本城市"]

# ----------------------------------------------------------------------------
# 采集批次与职位族（H3）
# ----------------------------------------------------------------------------
FAMILY_MIN_TITLE_COUNT = 5     # 在核心城市批次中出现 >=5 次的职位作为“锚点职位”
FAMILY_BIN = 40                # 行序分箱宽度（半重叠），同箱出现记为一次共现
FAMILY_EDGE_MIN_CO = 3         # 共现次数下限
FAMILY_EDGE_MIN_W = 0.08       # 余弦归一化共现强度下限
FAMILY_RESOLUTION = 1.0        # Louvain 分辨率
FAMILY_MIN_SIZE = 5            # 少于 5 个锚点职位的社区并入“未归类”
FAMILY_WINDOW = 15             # 行级归属：取前后各 15 行的锚点职位投票


def infer_batches(df: pd.DataFrame) -> pd.Series:
    """按原始行序把记录分成三个采集批次。

    原始文件中同一城市的记录连续成块：前一段是 371 个城市各一块（每块不超过 300 行），
    之后 14 个核心城市的大块依次出现两轮。第一段记为 A，两轮分别记为 B、C。
    """
    city = df["city"].values
    start = np.r_[0, np.flatnonzero(city[1:] != city[:-1]) + 1]
    length = np.diff(np.r_[start, len(city)])
    big = np.flatnonzero(length > 1000)
    p2 = start[big[0]]
    first_city = city[p2]
    second = [i for i in big[1:] if city[start[i]] == first_city]
    p3 = start[second[0]]
    out = np.where(np.arange(len(city)) < p2, "A", np.where(np.arange(len(city)) < p3, "B", "C"))
    return pd.Series(out, index=df.index, name="batch")


BATCH_A_ANOMALY_EDUANY_MAX = 0.15   # 批次 A 正常块的学历不限占比最低为 24.4%，异常块不超过 9.7%


def infer_segments(f: pd.DataFrame, core) -> pd.Series:
    """核心城市的每个（城市, 批次）大块属于哪个查询段。

    用各块包含的锚点职位集合的 Jaccard 相似度做层次聚类（3 类），按文件中首次出现的顺序命名 S1~S3。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    d = f[f["batch"].isin(["B", "C"]) & f["city"].isin(core)]
    vc = d["job_title"].value_counts()
    anchors = set(vc[vc >= FAMILY_MIN_TITLE_COUNT].index)
    first_row = d.groupby(["city", "batch"]).apply(lambda g: g.index[0])
    keys = list(first_row.sort_values().index)
    sets = [set(d.loc[(d["city"] == c) & (d["batch"] == b), "job_title"]) & anchors for c, b in keys]
    n = len(keys)
    dist = [1 - len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i in range(n) for j in range(i + 1, n)]
    lab = fcluster(linkage(np.array(dist), "average"), 3, "maxclust")
    order = list(dict.fromkeys(lab))                       # 按文件顺序首次出现
    name = {old: f"S{k + 1}" for k, old in enumerate(order)}
    return pd.Series([name[x] for x in lab], index=pd.MultiIndex.from_tuples(keys, names=["city", "batch"]))


def _build_families(d: pd.DataFrame):
    """在核心城市批次（B、C）上由行序相邻关系恢复职位族。

    返回 (行级职位族编号数组, 锚点职位->职位族 的 Series, 诊断信息 dict)。
    """
    import networkx as nx
    import scipy.sparse as sp

    blk = (d["city"] != d["city"].shift()).cumsum().values
    pos = d.groupby(blk).cumcount().values
    vc = d["job_title"].value_counts()
    anchors = vc[vc >= FAMILY_MIN_TITLE_COUNT].index
    tid = d["job_title"].map(pd.Series(np.arange(len(anchors)), index=anchors)).values
    has = ~np.isnan(tid)

    # 1) 锚点职位的共现图：同一城市块内、行序相邻（同一分箱）即共现
    rows, cols, base = [], [], 0
    for off in (0, FAMILY_BIN // 2):
        key = blk[has].astype(np.int64) * 1_000_000 + (pos[has] + off) // FAMILY_BIN
        codes, uniq = pd.factorize(key)
        rows.append(codes + base)
        cols.append(tid[has].astype(int))
        base += len(uniq)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(base, len(anchors)))
    X.data[:] = 1
    C = (X.T @ X).tocoo()
    freq = np.asarray(X.sum(0)).ravel()
    w = C.data / np.sqrt(freq[C.row] * freq[C.col])
    keep = (C.row < C.col) & (C.data >= FAMILY_EDGE_MIN_CO) & (w >= FAMILY_EDGE_MIN_W)
    G = nx.Graph()
    G.add_nodes_from(range(len(anchors)))
    G.add_weighted_edges_from(zip(C.row[keep].tolist(), C.col[keep].tolist(), w[keep].tolist()))

    # 2) Louvain 社区发现
    com = nx.community.louvain_communities(G, weight="weight", resolution=FAMILY_RESOLUTION, seed=0)
    modularity = nx.community.modularity(G, com, weight="weight")
    com = [c for c in com if len(c) >= FAMILY_MIN_SIZE]
    title_lab = np.full(len(anchors), -1)
    for k, c in enumerate(com):
        title_lab[list(c)] = k
    K = len(com)

    # 3) 行级归属：前后各 FAMILY_WINDOW 行内锚点职位的多数票（同一城市块内）
    own = np.full(len(d), -1)
    own[has] = title_lab[tid[has].astype(int)]
    onehot = np.zeros((len(d), K), dtype=np.float32)
    ok = own >= 0
    onehot[np.flatnonzero(ok), own[ok]] = 1
    row_lab = np.full(len(d), -1)
    loo_hit = loo_n = 0
    bstart = np.r_[0, np.flatnonzero(blk[1:] != blk[:-1]) + 1, len(d)]
    for a, b in zip(bstart[:-1], bstart[1:]):
        cs = np.vstack([np.zeros((1, K), np.float32), onehot[a:b].cumsum(0)])
        n = b - a
        lo = np.clip(np.arange(n) - FAMILY_WINDOW, 0, n)
        hi = np.clip(np.arange(n) + FAMILY_WINDOW + 1, 0, n)
        cnt = cs[hi] - cs[lo]
        best = cnt.argmax(1)
        top = cnt.max(1)
        tot = cnt.sum(1)
        good = (top >= 2) & (top >= 0.5 * tot)
        row_lab[a:b] = np.where(good, best, -1)
        # 留一检验：去掉自身后，邻居多数票是否与自身职位族一致
        o = own[a:b]
        m = o >= 0
        cnt2 = cnt[m].copy()
        cnt2[np.arange(m.sum()), o[m]] -= 1
        valid = cnt2.sum(1) > 0
        loo_hit += int((cnt2.argmax(1)[valid] == o[m][valid]).sum())
        loo_n += int(valid.sum())

    # 4) 按记录数从多到少重新编号 J01、J02…
    order = pd.Series(row_lab[row_lab >= 0]).value_counts().index.tolist()
    remap = {old: f"J{new + 1:02d}" for new, old in enumerate(order)}
    row_name = pd.Series(row_lab).map(remap).values
    title_family = pd.Series(pd.Series(title_lab).map(remap).values, index=anchors, name="job_family")
    info = {
        "n_anchor_titles": len(anchors), "n_edges": int(keep.sum()), "modularity": float(modularity),
        "n_families": len(order), "row_coverage": float((row_lab >= 0).mean()),
        "neighbor_agreement": loo_hit / max(loo_n, 1), "n_rows": len(d),
    }
    return row_name, title_family.dropna(), info


def load_enriched(rebuild: bool = False) -> pd.DataFrame:
    """清洗后的数据 + 语义标签 + 采集批次 + 职位族。首次运行约 1 分钟，之后读缓存。"""
    cache = os.path.join(CACHE_DIR, "enriched.pkl")
    if os.path.exists(cache) and not rebuild:
        return pd.read_pickle(cache)
    f = load_flagged().reset_index(drop=True)
    f["batch"] = infer_batches(f)
    tier = city_tier_map(f)
    core = set(tier[tier == "核心城市"].index)

    # 职位族在“未过滤、保持原始行序”的核心城市批次上恢复，再随清洗结果带出
    mask = f["batch"].isin(["B", "C"]) & f["city"].isin(core)
    row_name, title_family, info = _build_families(f.loc[mask])
    f["job_family"] = None
    f.loc[mask, "job_family"] = row_name
    title_family.rename_axis("job_title").reset_index().to_csv(
        os.path.join(PROCESSED_DIR, "job_family_titles.csv"), index=False, encoding="utf-8-sig")
    pd.Series(info).to_csv(os.path.join(PROCESSED_DIR, "job_family_diagnostics.csv"),
                           header=["value"], encoding="utf-8-sig")

    # 查询段与采集组
    seg = infer_segments(f, core)
    key = pd.MultiIndex.from_arrays([f["city"], f["batch"]])
    f["query_segment"] = pd.Series(seg.reindex(key).values, index=f.index).where(mask)
    grp_name = {"S1": "G1", "S2": "G2", "S3": "G3"}       # 以批次 B 所属的查询段命名采集组
    f["collect_group"] = f["city"].map(seg.xs("B", level="batch").map(grp_name))
    seg.rename("query_segment").reset_index().to_csv(
        os.path.join(PROCESSED_DIR, "query_segments.csv"), index=False, encoding="utf-8-sig")
    # 批次 A 的异常块
    blk = (f["city"] != f["city"].shift()).cumsum()
    g = f.groupby(blk)["education"].agg(n="size", eduany=lambda x: (x == "GP").mean())
    bad = g.index[(g["n"] >= 100) & (g["eduany"] < BATCH_A_ANOMALY_EDUANY_MAX)]
    f["batch_a_anomaly"] = blk.isin(bad) & f["batch"].eq("A")

    df = f.loc[~f["flag_any"]].drop(columns=list(FLAG_COLUMNS) + ["flag_any"]).reset_index(drop=True)
    df["edu_label"] = df["education"].map(EDU_LABEL)
    df["edu_level"] = df["education"].map(EDU_LEVEL)
    df["exp_label"] = df["experience"].map(EXP_LABEL)
    df["exp_level"] = df["experience"].map(EXP_LEVEL)
    df["edu_any"] = df["education"].eq("GP")
    df["exp_any"] = df["experience"].eq("EdD")
    grp = tier.replace({"核心城市": "新一线型"})
    grp[grp.index.isin(TIER1_CITIES)] = "一线型"
    df["city_group"] = df["city"].map(grp)
    df["is_parttime"] = df["salary_form"].isin(["day", "hour", "week"])
    df.to_pickle(cache)
    return df


def family_profile() -> pd.DataFrame:
    """职位族画像表（每族一行）及全组统一的描述性名称 family_label。

    名称按规则自动生成：编号 + 薪酬档 + 主流学历 + 行业属性，例如“J03 高薪·本科+·跨行业”。
    结果同时保存到 data/processed/job_family_profile.csv。
    """
    df = load_enriched()
    d = df[df["job_family"].notna()]
    edu_bin = pd.cut(d["edu_level"], [-1, 0, 2, 3, 6], labels=["学历不限", "高中及以下", "大专", "本科+"])
    g = d.groupby("job_family")
    prof = pd.DataFrame({
        "n": g.size(),
        "n_title": g["job_title"].nunique(),
        "n_company": g["company"].nunique(),
        "n_city": g["city"].nunique(),
        "salary_median": g["salary_mid_k"].median(),
        "salary_p25": g["salary_mid_k"].quantile(.25),
        "salary_p75": g["salary_mid_k"].quantile(.75),
        "bachelor_share": g["edu_level"].apply(lambda s: (s >= 4).mean()),
        "edu_any_share": g["edu_any"].mean(),
        "exp_any_share": g["exp_any"].mean(),
        "fresh_share": g["exp_level"].apply(lambda s: (s == 1).mean()),
        "senior_share": g["exp_level"].apply(lambda s: (s >= 4).mean()),
        "bonus_share": g["has_bonus_months"].mean(),
        "parttime_share": g["is_parttime"].mean(),
        "tier1_share": g["city_group"].apply(lambda s: (s == "一线型").mean()),
        "top_industry": g["company_type"].agg(lambda s: s.value_counts().index[0]),
        "top_industry_share": g["company_type"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
        "edu_mode": edu_bin.groupby(d["job_family"]).agg(lambda s: s.value_counts().index[0]),
        "batch_b_share": g["batch"].apply(lambda s: (s == "B").mean()),
    })
    tier = pd.cut(prof["salary_median"], [0, 7, 9, 12, 1e9], right=False, labels=["基础薪", "中薪", "中高薪", "高薪"])
    ind = np.where(prof["top_industry_share"] >= 0.30, prof["top_industry"].str.replace("type_", "") + "行业", "跨行业")
    prof["family_label"] = prof.index + " " + tier.astype(str) + "·" + prof["edu_mode"].astype(str) + "·" + ind
    prof = prof.sort_index()
    prof.to_csv(os.path.join(PROCESSED_DIR, "job_family_profile.csv"), encoding="utf-8-sig")
    return prof


def family_labels() -> dict:
    """职位族编号 -> 描述性名称。"""
    return family_profile()["family_label"].to_dict()


def family_diagnostics() -> pd.Series:
    load_enriched()
    return pd.read_csv(os.path.join(PROCESSED_DIR, "job_family_diagnostics.csv"), index_col=0)["value"]


if __name__ == "__main__":
    df = load_enriched(rebuild=True)
    print(family_diagnostics().round(4).to_string())
    print(df["batch"].value_counts().to_string())
    print(df["city_group"].value_counts().to_string())
    print(pd.crosstab(df["collect_group"], [df["batch"], df["query_segment"]]).to_string())
    print("批次 A 异常块：", df.loc[df["batch_a_anomaly"]].groupby("city").size().to_dict())
    print("job_family 覆盖：", df["job_family"].notna().sum(), f"{df['job_family'].notna().mean():.2%}")
    print(family_profile()[["n", "salary_median", "family_label"]].head(12).to_string())
