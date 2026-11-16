# -*- coding: utf-8 -*-
"""组信息（姓名、学号等个人信息）统一从环境变量 / .env 读取。

个人信息属于敏感数据，不进版本库：
    .env          真实值（已被 .gitignore 忽略，仅本机存在）
    .env.example  占位模板（随仓库发布，供他人复制）

读取优先级：真实环境变量 > 项目根目录 .env > 内置占位值。
没有 .env 时仍可生成文档（署名为占位符），因此克隆仓库后构建不会失败。

用法：
    from config import load_config
    CFG = load_config()
    CFG["group_no"], CFG["project_name"], CFG["members"]["A"]["name"] ...
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, ".env")
ENV_EXAMPLE = os.path.join(ROOT, ".env.example")
MEMBER_KEYS = ["A", "B", "C", "D", "E"]

DEFAULT_PROJECT_NAME = "职数洞见多变量招聘数据可视分析"


def parse_env(path):
    """极简 .env 解析器：支持注释、空行、export 前缀、引号包裹、行尾注释。"""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, encoding="utf-8-sig") as f:
        for line in f.read().splitlines():
            s = line.strip()
            if not s or s[0] in "#;":
                continue
            if s.startswith("export "):
                s = s[7:].lstrip()
            if "=" not in s:
                continue
            key, val = s.split("=", 1)
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            elif " #" in val:
                val = val.split(" #", 1)[0].rstrip()
            data[key] = val
    return data


def load_config(env_file=ENV_FILE, warn=True):
    """返回 {group_no, project_name, members:{A..E: {name, id, leader}}}。"""
    file_vals = parse_env(env_file)

    def get(key, default=""):
        if key in os.environ and os.environ[key].strip():
            return os.environ[key].strip()
        return file_vals.get(key, default).strip() or default

    if warn and not file_vals:
        print(f"[config] 未找到 {os.path.basename(env_file)}，使用占位组信息；"
              f"正式提交前请复制 {os.path.basename(ENV_EXAMPLE)} 为 .env 并填写。")

    members = {}
    for k in MEMBER_KEYS:
        members[k] = {
            "name": get(f"MEMBER_{k}_NAME", f"成员{k}姓名"),
            "id": get(f"MEMBER_{k}_ID", "学号"),
            "leader": get(f"MEMBER_{k}_LEADER", "false").lower() in ("1", "true", "yes", "y"),
            "role": get(f"MEMBER_{k}_ROLE", ""),
        }
    if not any(m["leader"] for m in members.values()):
        members[MEMBER_KEYS[0]]["leader"] = True   # 兜底：A 默认为组长

    return {
        "group_no": get("GROUP_NO", "X"),
        "project_name": get("PROJECT_NAME", DEFAULT_PROJECT_NAME),
        "members": members,
    }


def has_real_members(cfg):
    """判断配置是否已填入真实姓名学号（占位值里含“姓名/学号”字样）。"""
    return all(("姓名" not in m["name"]) and ("学号" not in m["id"])
               for m in cfg["members"].values())


if __name__ == "__main__":
    c = load_config()
    print("group_no:", c["group_no"])
    print("project_name:", c["project_name"])
    print("已填入真实信息:", has_real_members(c))
    for k, m in c["members"].items():
        masked = (m["id"][:4] + "*" * max(len(m["id"]) - 4, 0)) if m["id"] else m["id"]
        print(f"  {k}: {m['name']} / {masked} / leader={m['leader']}")
