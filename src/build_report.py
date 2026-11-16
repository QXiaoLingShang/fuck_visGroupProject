# -*- coding: utf-8 -*-
"""把各成员的小节稿（output/sections/X.md）汇总为第一阶段作业 Word 文档。

用法：python src/build_report.py

署名来源：组号、姓名、学号等个人信息全部来自项目根目录的 .env（不进版本库），
模板见 .env.example；缺少 .env 时以占位署名生成，方便他人克隆后直接跑通。
生成产物：build/ 目录（已被 .gitignore 忽略），文件名含姓名学号，请勿入库。
"""
import os
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config, has_real_members  # noqa: E402

SEC_DIR = os.path.join(ROOT, "output", "sections")
# 生成产物统一放在 build/（含姓名学号的文档不进版本库）；可用 BUILD_DIR 环境变量覆盖
BUILD_DIR = os.environ.get("BUILD_DIR") or os.path.join(ROOT, "build")
CFG = load_config()
MEMBERS = CFG["members"]

# 文档大纲：(章标题, [(成员字母, 小节稿中的二级标题), ...])；章下只有一个小节时不再设二级编号
OUTLINE = [
    ("「职数洞见」多变量招聘数据可视分析挑战赛的数据下载",
     [("A", "「职数洞见」多变量招聘数据可视分析挑战赛的数据下载")]),
    ("数据理解", [
        ("A", "数据集整体概览"),
        ("A", "数据的采集结构"),
        ("C", "经验、学历与行业类别数据的理解"),
        ("D", "行政区划与企业数据的理解"),
        ("B", "薪酬数据的理解"),
        ("A", "分析假设与问题拆解"),
    ]),
    ("数据过滤", [("E", "数据质量问题识别与过滤")]),
    ("数据处理", [
        ("B", "薪酬数据的标准化处理"),
        ("C", "职位族的恢复与职位画像"),
        ("E", "职位差异度的量化评估"),
        ("B", "薪酬模式的建模与差异识别"),
        ("D", "地域招聘画像与相似地域识别"),
        ("A", "行业发展动态与新兴职位"),
    ]),
    ("小结与下一阶段计划", [("ALL", "小结与下一阶段计划")]),
]


def member_label(key):
    if key == "ALL":
        return "全体成员"
    name = MEMBERS[key]["name"]
    return f"成员{key}" if "姓名" in name else name


def load_sections():
    """读取所有小节稿，返回 {(成员, 二级标题): [行, ...]}"""
    out = {}
    for fn in sorted(os.listdir(SEC_DIR)):
        if not fn.endswith(".md"):
            continue
        key = os.path.splitext(fn)[0].split("_")[0]   # A.md、A_q5.md 都归成员 A
        cur = None
        in_code = False
        for line in open(os.path.join(SEC_DIR, fn), encoding="utf-8").read().splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
            if not in_code and line.startswith("## "):
                cur = (key, line[3:].strip())
                out[cur] = []
            elif cur is not None:
                out[cur].append(line)
    return out


# ---------------------------------------------------------------- 样式工具
def set_font(run, size=None, bold=None, east="宋体", west="Times New Roman", color=None):
    run.font.name = west
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, hex_fill):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


class Builder:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.6)
        sec.top_margin = sec.bottom_margin = Cm(2.5)
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        for lvl, size in ((1, 16), (2, 14), (3, 12)):
            h = self.doc.styles[f"Heading {lvl}"]
            h.font.name = "Arial"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
            h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
            h.paragraph_format.space_after = Pt(6)
        self.fig_no = 0
        self.tab_no = 0
        self.missing = []

    # -- 基本块
    def heading(self, text, level):
        self.doc.add_heading(text, level=level)

    def para(self, text, indent=True, align=None, size=11, bold=False, color=None, after=4):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.35
        pf.space_after = Pt(after)
        if indent:
            pf.first_line_indent = Pt(size * 2)
        if align:
            p.alignment = align
        self._inline(p, text, size, bold, color)
        return p

    def _inline(self, p, text, size=11, bold=False, color=None):
        """支持行内 `code`，去掉残留的 ** 标记。"""
        text = text.replace("**", "")
        for i, part in enumerate(re.split(r"`([^`]+)`", text)):
            if not part:
                continue
            r = p.add_run(part)
            if i % 2:
                set_font(r, size - 0.5, bold, east="宋体", west="Consolas", color=color)
            else:
                set_font(r, size, bold, color=color)

    def bullet(self, text):
        p = self.doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(2)
        self._inline(p, text)

    def code(self, lines):
        t = self.doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = t.rows[0].cells[0]
        shade(c, "F3F3F1")
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        for i, ln in enumerate(lines):
            r = p.add_run(ln + ("\n" if i < len(lines) - 1 else ""))
            set_font(r, 9, east="宋体", west="Consolas")
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def figure(self, caption, path, width_cm=15.5):
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if not os.path.exists(full):
            self.missing.append(path)
            return
        self.fig_no += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(full, width=Cm(width_cm))
        self.para(f"图 {self.fig_no}  {caption}", indent=False,
                  align=WD_ALIGN_PARAGRAPH.CENTER, size=10, color="52514E", after=8)

    def table(self, caption, rows):
        self.tab_no += 1
        cp = self.para(f"表 {self.tab_no}  {caption}", indent=False,
                       align=WD_ALIGN_PARAGRAPH.CENTER, size=10, color="52514E", after=3)
        cp.paragraph_format.keep_with_next = True
        cp.paragraph_format.space_before = Pt(6)
        ncol = max(len(r) for r in rows)
        t = self.doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(rows):
            for j in range(ncol):
                cell = t.rows[i].cells[j]
                txt = row[j] if j < len(row) else ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (i == 0 or len(txt) <= 14) else WD_ALIGN_PARAGRAPH.LEFT
                self._inline(p, txt, size=9.5, bold=(i == 0))
                if i == 0:
                    shade(cell, "E8EEF7")
        # 列宽按各列最长文本分配（中文按 2 个字符宽计），避免长名称被挤成多行
        def vis(x):
            return sum(2 if ord(ch) > 127 else 1 for ch in x)
        wts = [min(max(max(vis(r[j]) if j < len(r) else 0 for r in rows), 10), 40) for j in range(ncol)]
        total = 15.6
        t.autofit = False
        for row in t.rows:
            for j, cell in enumerate(row.cells):
                cell.width = Cm(total * wts[j] / sum(wts))
        # 表头跨页重复
        trpr = t.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader"); hdr.set(qn("w:val"), "true"); trpr.append(hdr)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # -- 渲染一个小节稿
    def render(self, lines, sub_level):
        i, n = 0, len(lines)
        pending_caption = None
        while i < n:
            ln = lines[i].rstrip()
            s = ln.strip()
            if not s:
                i += 1
                continue
            if s.startswith("```"):
                block = []
                i += 1
                while i < n and not lines[i].strip().startswith("```"):
                    block.append(lines[i].rstrip())
                    i += 1
                i += 1
                self.code(block)
                continue
            m = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
            if m:
                self.figure(m.group(1), m.group(2))
                i += 1
                continue
            if s.startswith("### "):
                self.heading(s[4:].strip(), sub_level)
                i += 1
                continue
            if re.match(r"^表题[:：]", s):
                pending_caption = re.sub(r"^表题[:：]\s*", "", s)
                i += 1
                continue
            if s.startswith("|"):
                rows = []
                while i < n and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                        rows.append([c.replace("`", "") for c in cells])
                    i += 1
                self.table(pending_caption or "", rows)
                pending_caption = None
                continue
            if re.match(r"^[-*] ", s):
                self.bullet(s[2:].strip())
                i += 1
                continue
            self.para(s)
            i += 1


def _pretty(path):
    """尽量显示为相对项目根目录的路径（跨盘符时退回绝对路径）。"""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def build():
    secs = load_sections()
    b = Builder()
    doc = b.doc
    leader = next(k for k, v in MEMBERS.items() if v.get("leader"))
    order = [leader] + [k for k in MEMBERS if k != leader]

    # ---------------- 封面
    for _ in range(4):
        doc.add_paragraph()
    b.para("数据可视化课程项目 · 作业1", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=16, color="52514E")
    b.para("「职数洞见」多变量招聘数据可视分析", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=24, bold=True, after=6)
    b.para("可视化设计前四步：数据下载 · 数据理解 · 数据过滤 · 数据处理",
           indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, after=6)
    b.para("选题来源：ChinaVis 2024 数据可视化竞赛 赛道1-Ⅱ 多变量招聘数据可视分析挑战赛",
           indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=11, color="52514E", after=40)
    b.para(f"第 {CFG['group_no']} 组", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, after=10)

    duty = {k: [] for k in MEMBERS}
    for ci, (chap, items) in enumerate(OUTLINE, 1):
        for si, (k, title) in enumerate(items, 1):
            if k in duty:
                num = f"{ci}" if len(items) == 1 else f"{ci}.{si}"
                duty[k].append(f"{num} {title if len(items) > 1 else chap}")
    rows = [["角色", "姓名", "学号", "主要完成的章节"]]
    for k in order:
        m = MEMBERS[k]
        rows.append(["组长" if m.get("leader") else "组员", member_label(k), m["id"], "；".join(duty[k])])
    b.tab_no -= 1  # 封面分工表不占正文表号
    t0 = len(doc.tables)
    b.table("", rows)
    # 去掉封面表的“表 0”题注段落文字
    for p in doc.paragraphs:
        if p.text.strip().startswith("表 0"):
            p._element.getparent().remove(p._element)
    widths = [Cm(1.6), Cm(2.6), Cm(2.8), Cm(8.8)]
    for row in doc.tables[t0].rows:
        for c, w in zip(row.cells, widths):
            c.width = w
    doc.add_page_break()

    # ---------------- 正文
    for ci, (chap, items) in enumerate(OUTLINE, 1):
        single = len(items) == 1
        if single:
            k, title = items[0]
            b.heading(f"{ci} {chap}（{member_label(k)}）", 1)
            body = secs.get((k, title))
            if body is None:
                b.missing.append(f"section {k}/{title}")
            else:
                b.render(body, sub_level=2)
            continue
        b.heading(f"{ci} {chap}", 1)
        for si, (k, title) in enumerate(items, 1):
            b.heading(f"{ci}.{si} {title}（{member_label(k)}）", 2)
            body = secs.get((k, title))
            if body is None:
                b.missing.append(f"section {k}/{title}")
                continue
            b.render(body, sub_level=3)

    names = "-".join(f"{member_label(k)}({MEMBERS[k]['id']})" for k in order)
    fname = f"第{CFG['group_no']}组-作业1-{CFG['project_name']}-{names}.docx"
    os.makedirs(BUILD_DIR, exist_ok=True)
    out = os.path.join(BUILD_DIR, fname)
    try:
        doc.save(out)
    except PermissionError:
        raise SystemExit(
            f"无法写入 {_pretty(out)}：该文件正被 Word/WPS 等程序打开。\n"
            "请关闭文档后重试，或用 BUILD_DIR 指定另一个输出目录。"
        )
    print("saved:", _pretty(out))
    print("figures:", b.fig_no, "tables:", b.tab_no)
    if not has_real_members(CFG):
        print("WARNING: 当前为占位署名（未配置 .env），文档不计为正式提交稿。")
    if b.missing:
        print("MISSING:", b.missing)
    return out


if __name__ == "__main__":
    build()
