# -*- coding: utf-8 -*-
"""提交前敏感信息自检：确认即将进入版本库的文件里没有个人信息、本机路径和密钥。

用法：
    python src/check_sensitive.py            # 检查“将要提交”的文件（已跟踪 + 未忽略的新文件）
    python src/check_sensitive.py --all      # 额外检查已跟踪文件的全部内容（含历史遗留路径）
    python src/check_sensitive.py --strict   # 出现任何告警都返回非零（默认仅 ERROR 返回非零）

检查项：
    1. 个人信息：.env 中配置的真实姓名、学号（以及 10 位连续数字的学号样式）
    2. 本机路径：盘符开头的用户目录、MSYS 风格的工具目录、类 Unix 家目录等绝对路径
    3. 凭据：常见 API Key / Token / 私钥 / 口令赋值
    4. 应当被忽略却仍被跟踪的文件：.env、build/ 下的产物、根目录作业文档

退出码：0 通过；1 命中 ERROR（或 --strict 下命中 WARN）。
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config, has_real_members  # noqa: E402

TEXT_EXT = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml",
            ".cfg", ".ini", ".gitignore", ".example", ".sh", ".ps1", ""}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".xlsx", ".pkl", ".zip", ".gif", ".webp", ".pdf"}
# 需要解包内文再检查的容器格式
DOCX_EXT = {".docx"}
MAX_BYTES = 8 * 1024 * 1024          # 单文件最多读 8MB 做内容检查
BIG_FILE_WARN = 5 * 1024 * 1024      # 超过 5MB 的文本文件给体积告警

# ---------------------------------------------------------------- 规则
PATH_RULES = [
    (re.compile(r"[A-Za-z]:[\\/]{1,2}(?:Users|WorkSpace|Projects|Documents)[\\/]", re.I),
     "本机绝对路径（Windows 盘符）"),
    (re.compile(r"(?<![\w/])/[a-z]/WorkSpace/", re.I), "本机绝对路径（MSYS 风格 /d/WorkSpace）"),
    (re.compile(r"(?<![\w/])/(?:Users|home)/[\w.-]+/"), "本机绝对路径（类 Unix 家目录）"),
]
SECRET_RULES = [
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "疑似 GitHub Token"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "疑似 OpenAI 风格 API Key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "疑似 AWS Access Key"),
    (re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|PGP) PRIVATE KEY-----"), "私钥内容"),
    (re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd)\b\s*[:=]\s*"
                r"['\"]?(?![\s'\"<>\$%{]|your|YOUR|xxx|xxx|占位|示例|示例)[\w\-/+=.]{8,}"),
     "疑似口令/密钥赋值"),
]
# 学号样式：本项目学号是 10 位连续数字；单独给出，防止 .env 缺失时漏检。
# 前后不能是任何单词字符，避免把哈希/长浮点数误判为学号。
STUDENT_ID_RE = re.compile(r"(?<![0-9A-Za-z_])\d{10}(?![0-9A-Za-z_])")
# 个人信息检查用：把 .env 中的姓名逐字拆成可匹配序列（防止只匹配到半个名字）
NAME_MIN_LEN = 2

# 应当被忽略却仍被跟踪的路径
FORBIDDEN_TRACKED = [
    (re.compile(r"(^|/)\.env$"), ".env 不应进入版本库"),
    (re.compile(r"(^|/)\.env\.(?!example$)"), "除 .env.example 外的 .env.* 不应进入版本库"),
    (re.compile(r"^build/"), "build/ 下的生成产物不应进入版本库"),
    (re.compile(r"^src/members\.json$"), "旧的 members.json（明文姓名学号）不应进入版本库"),
    (re.compile(r"^[^/]+\.(docx|pdf)$"), "根目录的作业文档不应进入版本库（统一放 build/）"),
]

WARN_TRACKED = [
    (re.compile(r"^data/(raw|cache)/"), "原始数据/缓存体积较大，确认是否必须入库"),
]


def git(*args):
    out = subprocess.run(["git", "-C", ROOT, *args], capture_output=True)
    if out.returncode != 0:
        raise SystemExit("git 命令失败：" + out.stderr.decode("utf-8", "replace").strip())
    return out.stdout.decode("utf-8", "surrogateescape")


def target_files(all_tracked=False):
    """默认取“将被提交”的文件集合：已跟踪 + 未忽略的新文件。"""
    if all_tracked:
        names = git("ls-files").splitlines()
    else:
        names = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    return sorted({n.strip() for n in names if n.strip()})


def docx_text(path):
    """提取 docx 内文（段落 + 表格）；不是合法 docx 时返回 None。"""
    try:
        from docx import Document
    except ImportError:
        return None
    try:
        doc = Document(path)
    except Exception:
        return None
    parts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


def pii_literals(cfg):
    """返回 [(标签, 字面量)]：只收集已填真实值（占位值自动跳过）。"""
    items = []
    for key, m in cfg["members"].items():
        if "姓名" not in m["name"] and len(m["name"]) >= NAME_MIN_LEN:
            items.append((f"成员{key}姓名", m["name"]))
        if "学号" not in m["id"] and m["id"]:
            items.append((f"成员{key}学号", m["id"]))
    return items


def scan_text(rel, text, pii, hits, allow_pii_file=None):
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, lit in pii:
            if lit and lit in line:
                hits.append(("ERROR", rel, lineno, f"命中{label}", line.strip()[:120]))
        for rx, label in PATH_RULES:
            if rx.search(line):
                hits.append(("ERROR", rel, lineno, label, line.strip()[:120]))
        for rx, label in SECRET_RULES:
            if rx.search(line):
                hits.append(("ERROR", rel, lineno, label, line.strip()[:120]))
        m = STUDENT_ID_RE.search(line)
        if m:  # 10 位连续数字：学号样式（也可能是数据中的其他长数字），提醒人工确认
            hits.append(("WARN", rel, lineno, "疑似 10 位学号", line.strip()[:120]))


def main(argv):
    strict = "--strict" in argv
    all_tracked = "--all" in argv
    cfg = load_config(warn=False)
    pii = pii_literals(cfg)
    if not has_real_members(cfg):
        print("[1/3] 未找到 .env 中的真实个人信息：仅用学号样式规则检查\n")
    else:
        print(f"[1/3] 已载入 .env 中的 {len(pii)} 条个人信息字面量（仅用于比对，不会打印）\n")

    files = target_files(all_tracked)
    print(f"[2/3] 待检查文件 {len(files)} 个（{'全部已跟踪' if all_tracked else '已跟踪 + 未忽略的新文件'}）")

    hits = []
    for rel in files:
        for rx, label in FORBIDDEN_TRACKED:
            if rx.search(rel):
                hits.append(("ERROR", rel, 0, label, ""))
        for rx, label in WARN_TRACKED:
            if rx.search(rel):
                hits.append(("WARN", rel, 0, label, ""))

        full = os.path.join(ROOT, rel)
        if not os.path.exists(full):
            continue
        ext = os.path.splitext(rel)[1].lower()
        size = os.path.getsize(full)
        if ext in DOCX_EXT:
            text = docx_text(full)
            if text is None:
                hits.append(("WARN", rel, 0, "无法解析的 docx（非标准包或缺少 python-docx），需人工确认", ""))
                continue
            scan_text(rel, text, pii, hits)
            continue
        if ext in SKIP_EXT:
            continue
        if size > MAX_BYTES:
            hits.append(("WARN", rel, 0, f"文本文件过大（{size/1048576:.1f}MB），已跳过内容检查", ""))
            continue
        if size > BIG_FILE_WARN:
            hits.append(("WARN", rel, 0, f"文本文件较大（{size/1048576:.1f}MB）", ""))
        try:
            text = open(full, "rb").read().decode("utf-8", "ignore")
        except OSError as e:
            hits.append(("WARN", rel, 0, f"读取失败：{e}", ""))
            continue
        scan_text(rel, text, pii, hits)

    errs = [h for h in hits if h[0] == "ERROR"]
    warns = [h for h in hits if h[0] == "WARN"]
    print(f"      ERROR {len(errs)} 条，WARN {len(warns)} 条\n")

    print("[3/3] 结果")
    if errs:
        print("  必须修复（ERROR）：")
        for level, rel, lineno, label, snippet in errs:
            loc = f"{rel}:{lineno}" if lineno else rel
            print(f"    - {loc}  {label}")
            if snippet:
                print(f"        > {snippet}")
    if warns:
        # WARN 按规则聚合，只列前几个例子，避免刷屏
        groups = {}
        for level, rel, lineno, label, snippet in warns:
            groups.setdefault(label, []).append((rel, lineno, snippet))
        print("  建议确认（WARN）：")
        for label, items in groups.items():
            files = sorted({r for r, _, _ in items})
            print(f"    - {label}：{len(items)} 处，涉及 {len(files)} 个文件（如 {files[0]}）")
            for rel, lineno, snippet in items[:2]:
                loc = f"{rel}:{lineno}" if lineno else rel
                print(f"        > {loc} {snippet}")
    if not hits:
        print("  ✔ 未发现个人信息、本机绝对路径或凭据。")

    ok = not errs and not (strict and warns)
    print("\n结论：" + ("通过，可以提交。" if ok else "未通过，请先处理上面的问题。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
