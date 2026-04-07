---
title: style-transfer-mvp
app_file: app.py
sdk: gradio
sdk_version: 5.22.0
---
# 基于小样本的叙事风格迁移研究（MVP）

本项目实现一个可复现实验流水线：
- 生成侧对比三种基线：Prompt Engineering（Zero-shot）、Context Engineering（Static Few-shot）、RAG + ICL（Dynamic Few-shot）
- 评估侧输出五类指标：语义保真、LLM 风格裁判、风格向量距离、语言学统计、流利度（100/NLL）
- 工程侧支持中间态逐条落盘（JSONL）与 Markdown 对比报告导出
- 当前主输出已切换为 `case_summary.jsonl` / `case_summary.md`

---

## 1. 项目结构

```text
style-transfer-mvp/
├── configs/config.yaml
├── data/
│   ├── raw/
│   │   ├── test_cases.json
│   │   ├── target_styles.json
│   │   └── style_corpus.json
│   └── outputs/
│       ├── case_summary.jsonl
│       └── case_summary.md
├── scripts/
│   ├── download_modelscope_models.py
│   └── export_case_summary_md.py
├── src/
│   ├── generator/
│   ├── evaluator/
│   └── utils/
├── requirements.txt
└── main_pipeline.py
```

---

## 2. 环境准备

### 2.1 Python 版本
- 推荐 Python `3.10+`（当前项目已在 3.11 环境验证依赖安装）

### 2.2 配置 API Key
1. 复制环境变量模板为 `.env`
2. 在 `.env` 中填写 `DEEPSEEK_API_KEY`

示例：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

> 安全建议：不要把真实 Key 提交到仓库；如已泄露请立即在服务商后台轮换。

### 2.3 模型下载说明（国内网络）
项目当前采用“**ModelScope 优先 + HuggingFace 国内镜像兜底**”的方式加载评估模型：
- HuggingFace 镜像：`https://hf-mirror.com`
- 本地缓存目录：`.modelscope_cache/`、`.hf_cache/`（如已存在会复用，不会每次完整重下）

首次运行前建议先执行：

```powershell
python scripts/download_modelscope_models.py
```

说明：
- 如果某模型在魔搭没有同名仓库，会自动退回到 HuggingFace 国内镜像下载。
- 已下载完成的模型下次运行通常不会重复下载，只会复用缓存或补齐缺失文件。

---

## 3. 运行说明

### 3.1 首次运行（推荐）

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env 填写真实 DEEPSEEK_API_KEY
python scripts/download_modelscope_models.py
python main_pipeline.py
python scripts/export_case_summary_md.py
```

### 3.2 虚拟环境下的高效运行

如果你已经进入项目虚拟环境，且依赖已安装，无需每次重复安装，直接运行：

```powershell
python main_pipeline.py
python scripts/export_case_summary_md.py
```

仅在以下情况需要重新执行依赖安装：
- 首次拉取项目
- `requirements.txt` 发生变更
- 切换到新的 Python 环境

### 3.3 当前主产物
- 主中间结果：`data/outputs/case_summary.jsonl`
- 主展示报告：`data/outputs/case_summary.md`

当前主流程默认：
- 读取 `data/raw/test_cases.json`
- 对每个 case 跑 3 种策略
- 将每条结果实时落盘到 `case_summary.jsonl`
- 再由导出脚本生成 `case_summary.md`

---

## 4. 输入数据字段说明

### 4.1 `data/raw/test_cases.json`
顶层字段：
- `cases`: 测试样本列表

每个 case 字段：
- `case_id`: 样本唯一ID
- `style_name`: 目标风格标签（如 `luxun`）
- `source_text`: 待改写事实文本

### 4.2 `data/raw/target_styles.json`
- 键：风格标签（如 `luxun`）
- 值：该风格的代表文本数组（2-3段或更多）
- 用途：
  - Baseline B 作为静态 few-shot 参考
  - Zero-shot 评估阶段作为 Linguistic 的隐藏 target 标尺

### 4.3 `data/raw/style_corpus.json`
- 键：风格标签
- 值：该风格检索语料数组
- 用途：Baseline C（RAG + ICL）动态检索 Top-K 参考片段

---

## 5. 输出文件说明

### 5.1 当前主中间态日志 `data/outputs/case_summary.jsonl`
每行一个完整实验记录，核心字段包括：
- `timestamp`
- `case_id`
- `strategy`
- `style_name`
- `source_text`
- `style_references`
- `generated_text`
- `metrics`

`metrics` 子字段：
- `bert_score`
- `llm_judge_score`
- `style_vector_score`
- `linguistic_stats`
- `fluency_score`
- `details`

其中 `metrics.details` 当前包含：
- `semantic`：语义指标明细
- `llm_judge`：单裁判明细（DeepSeek-V3.2）
- `style_vector`：风格向量指标明细
- `fluency`：流利度原始细节

### 5.2 当前主展示报告 `data/outputs/case_summary.md`
- 以 Markdown 形式展示全部样本结果
- 表1：内容保真 → 风格相似 → 语言流利 + 语言学特征
- 表2：双裁判模型逐维评分 + 评语
- 适合直接用于汇报、答辩、人工复核

### 5.3 历史产物说明
仓库内可能仍保留以下旧产物，用于历史对比或留档：
- `data/outputs/intermediate_results.jsonl`
- `data/outputs/final_report.csv`
- `data/outputs/intermediate_results_compare.md`

但**当前主流程不再依赖它们作为最终交付物**。

---

## 6. 评估指标解释

### 6.1 Semantic Similarity（BERTScore）
- 对比：`source_text` vs `generated_text`
- 指标：BERTScore F1
- 作用：约束事实保真，惩罚信息丢失与幻觉

> 当前已固定表一语义指标方法为 **标准 `BERTScore-F1`**，避免实验间方法漂移。

### 6.2 LLM-as-a-Judge（风格裁判）
- 模型：`DeepSeek-V3.2`、`Qwen3-Next-80B-A3B-Instruct`
- 输入：`style_references` + `generated_text`
- 输出：`lexical/syntax/emotion/overall/comment` JSON
- 作用：以大模型主观判别风格贴合度

主流程中：
- `llm_judge_score` 为单裁判总分（DeepSeek-V3.2）
- `details.llm_judge` 保留单裁判维度明细与评语

### 6.3 Style Embedding Distance（风格向量距离）
- 做法：
  - 计算目标参考文本向量均值（style centroid）
  - 计算生成文本向量
  - 求余弦相似度
- 作用：提供客观向量空间的风格接近度信号

> 当前已固定表一风格相似指标方法为 **SentenceTransformer + 余弦相似度**。

### 6.4 Linguistic Feature Stats（语言学统计）
输出四个核心维度的三方原始值：
- `avg_len`: 平均句长（字符数/句子数）
- `ttr`: 词汇丰富度（Type-Token Ratio）
- `adj_density`: 形容词密度
- `adv_density`: 副词密度

三方比较对象：
- `source`（原文）
- `target`（目标风格标尺）
- `generated`（生成文本）

> 当前已固定表一语言学指标方法为 **jieba/POS 统计链路**。

### 6.5 Fluency（100/NLL）
- 使用 Causal LM 计算生成文本 NLL
- 采用 `100 / NLL` 转换为正向分值
- 分值越高通常表示语言更流畅

> 当前已固定表一流利度指标方法为 **标准 `100 / NLL`**。

---

## 7. 三种基线策略

- `Baseline_A_Prompt_Eng`：`style_references=[]`（Zero-shot）
- `Baseline_B_Context_Eng`：固定 few-shot 参考（Static）
- `Baseline_C_RAG_ICL`：从风格语料动态检索 Top-K（Dynamic）

---

## 8. 当前实验状态（2026-03-12）

- 当前样本规模：**5 个 case × 3 个策略 = 15 条结果**
- 最新结论：
  - **Baseline B（Static Few-shot）最稳**：保真与流利度综合最好
  - **Baseline C（RAG+ICL）主观风格感最强**
- 当前文档建议优先查看：
  - `summary.md`
  - `summary_defense.md`
  - `data/outputs/case_summary.md`

---

## 9. 常见问题（FAQ）

1. **报错 `Missing DEEPSEEK_API_KEY`**  
   检查项目根目录 `.env` 是否存在、变量名是否为 `DEEPSEEK_API_KEY`。

2. **首次运行很慢**  
   `bert-score`、`transformers`、`sentence-transformers` 可能会下载模型权重，属正常现象；建议先执行 `python scripts/download_modelscope_models.py`。

3. **以后会不会重复下载模型？**  
   一般不会。模型文件会缓存在本地目录中，后续运行通常直接复用缓存。

4. **输出文件不断变大**  
   中间态是追加写入。新一轮实验前可手动清理 `data/outputs/case_summary.jsonl`。

5. **Zero-shot 场景 target 值为 0**  
   请确认 `data/raw/target_styles.json` 中已配置对应 `style_name` 的示例文本。

6. **为什么 BERTScore 会突然变低？**  
   旧版本中如果模型下载失败，可能触发回退方法。当前已固定评估方法并禁用默认回退，确保多轮结果可比。

---

## 10. 最小实验流程建议

1. 先只保留 1-2 个 case 做冒烟测试
2. 检查 `case_summary.jsonl` 字段完整性
3. 运行 `python scripts/export_case_summary_md.py`
4. 再扩充 case 与风格库，比较三策略趋势
5. 最后基于 `case_summary.md`、`summary.md`、`summary_defense.md` 做人工复核与汇报展示
