# 职数洞见 · 多变量招聘数据可视分析

数据可视化课程小组作业 1（第一阶段：数据下载 → 数据理解 → 数据过滤 → 数据处理）。
选题来自 **ChinaVis 2024 数据可视化竞赛 赛道 1-Ⅱ「职数洞见」多变量招聘数据可视分析挑战赛**。

仓库内容 = 可复现的分析脚本 + 图表 + 图表所用数据表 + 小节稿，以及一个把小节稿汇总成 Word 作业文档的构建脚本。

---

## 目录结构

```
.
├── .env.example              # 组信息模板（复制为 .env 后填写真实姓名学号）
├── requirements.txt          # Python 依赖
├── src/                      # 分析脚本（公共模块 common.py 统一取数与图表风格）
│   ├── common.py             # 数据加载、薪酬折算、质量标记、全组图表样式
│   ├── config.py             # 从 .env 读取组信息（个人信息不入库）
│   ├── check_sensitive.py    # 提交前敏感信息自检
│   ├── build_report.py       # 汇总 output/sections/*.md → Word 文档（输出到 build/）
│   ├── A*.py B*.py C*.py D*.py E*.py
│   └── enrich.py D2_lib.py
├── data/
│   ├── raw/                  # 挑战赛原始数据（脱敏虚拟编码）
│   ├── processed/            # 脚本生成的中间结果表
│   ├── reference/            # 外部对照资料（平台公开筛选项快照）
│   └── cache/                # 可重建缓存（不入库）
├── output/
│   ├── figs/                 # 全部图表（脚本生成）
│   └── sections/             # 各成员的小节稿（Markdown，构建文档的输入）
├── docs/
│   ├── 第一阶段要求.txt        # 课程作业要求
│   ├── 我们小组情况与选题.txt
│   ├── template/             # 作业文档模板（占位版）
│   └── internal/             # 内部协作说明（已脱敏，见其 README）
└── build/                    # 生成的 Word 作业文档（不入库，见下）
```

## 环境准备

```bash
pip install -r requirements.txt
```

原始数据 `data/raw/JobWanted.xlsx` 首次读取后会缓存为 `data/cache/*.pkl`（约 160MB，可随时删除重建）。

## 配置组信息（个人信息不进版本库）

文档署名的姓名、学号属于个人信息，本仓库**不保存**，只保存占位模板：

```bash
cp .env.example .env    # Windows: copy .env.example .env
# 然后编辑 .env，填写 GROUP_NO / PROJECT_NAME / MEMBER_X_NAME / MEMBER_X_ID / MEMBER_X_LEADER
```

- `.env` 已被 `.gitignore` 忽略，不会被提交或公开。
- 没有 `.env` 时脚本仍能跑通，但文档署名为占位符（脚本会打印警告），不能作为正式提交稿。
- 读取优先级：真实环境变量 > `.env` > 内置占位值，因此 CI 里也可以用环境变量注入。

## 生成作业文档

```bash
python src/build_report.py
```

- 输入：`output/sections/*.md`（小节稿）与 `output/figs/*.png`（图表）
- 输出：`build/第X组-作业1-<项目名>-<组长姓名(学号)>-....docx`（严格按课程命名规则）
- `build/` 已被忽略：文档内含姓名学号，请从本地 `build/` 取件提交，不要入库。
- 可用环境变量 `BUILD_DIR` 指定其它输出目录。

## 提交前自检

```bash
python src/check_sensitive.py
```

扫描「已跟踪 + 未忽略的新文件」，检查：个人信息（`.env` 中的真实姓名学号、学号样式数字）、本机绝对路径、常见凭据/私钥、以及不该入库的路径（`.env`、`build/`、根目录 `.docx`）。有任何 ERROR 时退出码非零；`--strict` 让 WARN 也失败。

## 分析脚本一览

| 阶段 | 脚本 |
| --- | --- |
| 数据下载 | `A_download_overview.py` |
| 数据理解 | `A2_understand.py`、`A2q5_emerging.py`、`A2q5_industry.py`、`C_understand.py`、`C2_understand.py`、`D_understand.py`、`D2_understand.py`、`B_salary.py` |
| 数据过滤 | `E_quality_filter.py`、`E_job_features.py` |
| 数据处理 | `B2_salary_model.py`、`C2_family.py`、`E2_job_distance.py`、`E2_quality_scope.py`、`D2_core.py`、`D2_nation.py` |
| 文档构建 | `build_report.py` |

所有脚本都通过 `from common import *` 取数与取样式，保证口径一致；脚本可重复运行。

## 数据说明

- 数据为挑战赛提供的脱敏数据（430,664 行 × 7 列，全部为字符串）：职位名称、行政区划、薪酬、经验要求、学历要求、企业、行业类别均以哈希/虚拟编码给出，真实含义未知，文档中的“经验/学历等级顺序”“职位族编号”等均为**基于数据的推断或我们的命名**，不代表官方含义。
- `data/raw/JobWanted.xlsx`（约 21MB）与 `data/processed/C_job_profile.csv`（约 12.5MB）体积较大但属于复现必需，故保留在仓库中。
- `data/reference/platform_taxonomy.json` 为某招聘平台网页端公开筛选项快照，仅作理解脱敏编码的外部对照，不含任何个人数据。
