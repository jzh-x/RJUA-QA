# RJUA-QA Eval

> 仁济泌尿外科智能体（RJUA）知识库问答评测框架 — 公开透明、可复现的端到端评测流水线

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## 项目背景

仁济泌尿外科智能体（RJUA）是一个面向泌尿外科临床场景的医疗知识库问答系统。本仓库提供了该智能体的完整评测方案，覆盖 **50 道真实临床问答题**，从多个维度量化评估 AI 回答的质量。

---

## 评测维度

| 维度 | 方法 | 说明 |
|------|------|------|
| **完整度** | 关键词匹配 + LLM Judge 兜底 | 评估回答是否覆盖了标准答案中的所有评估要点。PASS → 100%, PARTIAL → ≥50% |
| **忠实度** | RAGAS Faithfulness | 从回答中提取所有断言，逐一验证是否被源文档支持 |
| **相关度** | RAGAS Relevancy | 从回答倒推 3 个逆向问题，由 LLM 评估语义相似度（0–1 连续分） |
| **正确度** | DeepSeek-V3 多轮投票 | 独立 LLM Judge 对答案的医学正确性进行 3 轮独立判定，取多数意见 |

#
## 评测方法

1. **完整度**：标准化后提取中文词组，匹配评估要点中的关键词。为了避免字面差异导致的漏判，加入了 RJUA 医疗同义词表。同时由 LLM Judge 对完整性进行语义验证。
2. **忠实度**：要求 LLM 从回答中提取所有独立断言，逐条判断每个断言是否能从源文档推断出来。最终分数 = 被支持的断言数 ÷ 总断言数。
3. **相关度**：LLM 从回答倒推生成 3 个逆向问题，逐一评估每个逆向问题与原问题的语义相似度，取 0–1 连续分（均值）。与 RAGAS 原论文方法一致。
4. **正确度**：独立部署的 DeepSeek-V3 作为评委，对每道题的答案进行 PASS / PARTIAL / FAIL 判定，3 轮取多数意见，计算加权正确分数。

---

## 项目结构

```
rjua-qa-eval/
├── rjua_qa_eval_standalone.py  # 主评测脚本（一键运行）
├── .env.example                # 环境变量模板
├── RJUA-仁济QA-50题评测集.xlsx   # 评测数据集
├── RJUA-仁济源文档-全部50篇.txt  # 源文档
├── rjua_report.html            # 完整版评测报告
├── rjua_report_2.html     # 简明版评测报告
└── README.md                   # 本文件
```

---

## 快速开始

#
## 环境要求

- Python 3.10+
- 可用的 LLM API Key（SiliconFlow DeepSeek-V3）
- 被测智能体平台访问权限（KB UUID）

#
## 安装步骤

> **环境要求**：安装 [Python 3.10+](https://www.python.org/downloads/)（装好后 `pip` 自动可用）

```bash
# 1. 克隆仓库
git clone https://github.com/jzh-x/RJUA-QA.git
cd RJUA-QA

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
# macOS / Linux:
cp .env.example .env
# Windows 用户：
copy .env.example .env

# 4. 编辑 .env 文件，填写以下信息：
#    PLATFORM_URL    = 平台地址
#    PLATFORM_USERNAME = 用户名
#    PLATFORM_PASSWORD = 密码
#    KB_UUID         = 被测智能体的知识库ID
#    LLM_API_KEY     = SiliconFlow API Key（如使用LLM Judge）

# 5. 运行评测
python rjua_qa_eval_standalone.py
```

> **注意**：评测数据文件已包含在仓库中，无需额外下载。


## 运行说明

- 评测过程自动保存中间结果到 `results_partial.json`，支持断点续跑
- 完整评测预计耗时约 30-50 分钟（取决于网络和平台响应速度）
- 评测结果输出为 JSON 文件并保存到 `rjua_results/` 目录
- 可使用 `gen_full.py` 生成 HTML 格式的评测报告

---

## 数据集

本评测使用的数据集及源文档来自公开论文：**仁济医院泌尿科 × 蚂蚁集团**（arXiv:2312.09785）。

- **评测数据集**：[RJUA-仁济QA-50题评测集.xlsx](RJUA-仁济QA-50题评测集.xlsx) — 50 道真实临床问答题
- **源文档**：[RJUA-仁济源文档-全部50篇.txt](RJUA-仁济源文档-全部50篇.txt) — 50 篇泌尿外科参考文档

---
## 评测报告

- [rjua_report.html](https://htmlpreview.github.io/?https://raw.githubusercontent.com/jzh-x/RJUA-QA/main/rjua_report.html)
- [rjua_report_2.html](https://htmlpreview.github.io/?https://raw.githubusercontent.com/jzh-x/RJUA-QA/main/rjua_report_2.html)

---


## 联系方式

如有疑问或建议，欢迎提交 Issue 或 Pull Request。
## 许可证

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 开源。

---

## 联系方式

如有疑问或建议，欢迎提交 Issue 或 Pull Request。
