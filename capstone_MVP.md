#  小样本叙事风格迁移 MVP 项目 (Narrative Style Transfer) - Copilot 开发任务书

## 1. 顶层设计与目录结构规范

为了保证项目的可扩展性和 GitHub 开源规范，请严格按照以下目录树生成和组织代码：

```
style-transfer-mvp/
├── configs/                  # 配置文件夹
│   └── config.yaml           # 全局配置 (模型参数、评估权重、API URL等)
├── data/                     # 数据文件夹
│   ├── raw/                  # 原始测试 case (事实文本、风格参考文本、语料库)
│   └── outputs/              # 生成的文本及评估报告 (JSON/CSV)
├── src/                      # 核心源代码 (面向对象设计)
│   ├── __init__.py
│   ├── generator/            # 文本生成模块 (支持三种迁移策略)
│   │   ├── __init__.py
│   │   ├── base_generator.py # 生成器抽象基类
│   │   ├── prompt_eng.py     # 策略一: Prompt Engineering
│   │   ├── context_eng.py    # 策略二: Context Engineering
│   │   └── rag_icl.py        # 策略三: RAG + In-context Learning
│   ├── evaluator/            # 核心评估模块 (继承体系)
│   │   ├── __init__.py
│   │   ├── metrics_base.py   # 评估指标抽象基类 (定义 evaluate 接口)
│   │   ├── semantic.py       # 内容保真度 (BERTScore)
│   │   ├── style_llm.py      # 风格打分 (LLM-as-a-Judge)
│   │   ├── style_vector.py   # 风格向量距离 (Embedding)
│   │   ├── linguistic.py     # 语言学特征统计 (句长、标点等)
│   │   └── fluency_nll.py    # 语言流利度 (100/NLL)
│   └── utils/                # 工具函数
│       ├── __init__.py
│       ├── api_client.py     # 统一的 API 请求封装 (处理重试、并发)
│       └── file_io.py        # 文件读写工具(重点实现 .jsonl 的 Append 追加写入，用于中间态落盘)
├── .env.example              # 环境变量示例 (API Keys 存放处)
├── .gitignore                # 忽略 data/, .env, __pycache__ 等
├── requirements.txt          # 依赖包列表
└── main_pipeline.py          # 主流程脚本：串联加载、多策略生成、多维度评估并输出报告
```

---

## 2. 核心模块计划框架 (面向对象与配置化)

### 2.1 环境与配置模块 (`configs/config.yaml` & `.env`)
* **作用：** 将代码与配置解耦。
* **设计：** * `.env` 仅存储敏感的 API Keys（如 `DEEPSEEK_API_KEY`）。
    * `config.yaml` 存储非敏感参数（如 `judge_model: "DeepSeek-V3.2-Instruct"`）。

### 2.2 生成器模块 (`src/generator/`) -> **核心探究基线 (Baselines)**
* **作用：** 负责拼接 Prompt 并调用大模型进行风格重写。通过控制输入的 `style_references` (参考文本) 和前置检索逻辑，实现三大对比基线的高效代码复用。
* **设计：** 实现一个统一的 `StyleTransferGenerator` 类，核心接口为 `generate(source_text, target_style_name, style_references=None)`。设计好数据结构，强制要求数据以分离的字段传入，方便后续指标计算等流程
* **三大探究基线逻辑 (在主流程 main_pipeline 中通过传参控制)：**
    1. **Baseline A: Prompt Engineering (Zero-shot)**
       * **参数控制：** `style_references` 传入空列表 `[]`。
       * **探究目的：** 纯靠指令（如“请将其重写为鲁迅风格”），测试大模型自身参数化知识 (Parametric Knowledge) 中对该风格的掌握底线。
    2. **Baseline B: Context Engineering (Static Few-shot)**
       * **参数控制：** `style_references` 传入固定的 2-3 篇目标风格的代表性文本片段。
       * **探究目的：** 测试大模型的上下文学习 (In-context Learning) 能力，观察硬塞入固定特征样本后，指标比 Baseline A 提升多少。
    3. **Baseline C: RAG + In-context Learning (Dynamic Few-shot)**
       * **参数控制：** 增加前置检索步骤。从目标风格语料库中，利用向量检索找出与 `source_text` 语义/情境最相关的 3 个片段，作为 `style_references` 传入。
       * **探究目的：** 测试“动态且语义精准的上下文”是否能最大限度地降低“事实幻觉 (Hallucination)”，同时保持极高的风格融合度。

### 2.3 评估器模块 (`src/evaluator/`)
* **作用：** 规范所有评估指标的接口，方便流水线调用。
* **设计：** 创建 `BaseMetric` 抽象基类。所有具体指标必须继承此类，并实现 `evaluate(source_text, generated_text, style_references=None)` 方法，统一返回字典格式（如 `{"score": 0.85, "details": "..."}`）。

### 2.4 主流程管道 (`main_pipeline.py`) -> **含中间态落盘机制 (Persistence)**
* **作用：** MVP 的入口，执行自动化评测流程，并负责数据的持久化落盘，确保实验过程可追溯、防中断。
* **流程与落盘机制设计：**
    1. **加载测试集：** 读取 `data/raw/` 下的结构化测试用例。
    2. **策略循环生成与评估：** 遍历三种基线策略（Prompt, Context, RAG+ICL）。每次生成 `generated_text` 并完成 `evaluator` 打分后，**必须立即触发单次落盘**。
    3. **中间落盘文件格式 (Intermediate Logging)：** * **依赖模块：** 调用 `src/utils/file_io.py` 中的 `save_intermediate_result()` 方法。
       * **文件路径：** 统一保存在 `data/outputs/intermediate_results.jsonl` (JSON Lines 格式，方便逐行追加写入，防止程序崩溃导致数据丢失)。
       * **落盘字段要求 (必须包含完整的前后对比与指标)：**
         ```json
         {
           "timestamp": "2024-05-20T10:00:00",
           "case_id": "case_001",
           "strategy": "Baseline_B_Context_Eng",
           "style_name": "luxun",
           "source_text": "今天早上下大雨，我上班迟到被老板骂了。",
           "style_references": ["参考文本1...", "参考文本2..."], 
           "generated_text": "今日骤雨，吾赴工迟滞，遭掌柜一通痛斥...",
           "metrics": {
               "bert_score": 0.85,
               "llm_judge_score": 8.5,
               "linguistic_stats": {
                   "avg_len": {"source": 18, "target": 12, "generated": 14},
                   "ttr": {"source": 0.4, "target": 0.6, "generated": 0.55}
               }
           }
         }
         ```
    4. **最终聚合报告：** 整个 Test Suite 跑完后，读取上述 JSONL 中间文件，利用 Pandas 或 csv 模块，生成一份直观的 `final_report.csv` 或 Markdown 对比表格，供人类 Review。

---

## 3. 评估指标定义与注释 (请 Copilot 严格按照此逻辑实现)

在实现 `src/evaluator/` 目录下的具体指标时，请遵循以下学术/工程定义：

1.  **内容保真度 - Semantic Similarity (BERTScore)**
    * **文件：** `semantic.py`
    * **逻辑：** 比较 `source_text` (原事实文本) 和 `generated_text` (生成文本)。调用 `bert-score` 库提取 F1 值，用于惩罚“事实幻觉”或“核心信息丢失”。

2.  **风格相似度 - 大模型裁判 (LLM-as-a-Judge)**
    * **文件：** `style_llm.py`
    * **指定模型：** **DeepSeek-V3.2-Instruct**
    * **逻辑：** 构建裁判 Prompt，将 `style_references` (目标风格文本) 和 `generated_text` 一起输入给大模型。要求其从词汇偏好、句式结构、情绪表达三个维度进行 0-10 分定量打分，并输出定性评价 (严格要求返回 JSON 格式解析)。

3.  **风格相似度 - 向量距离 (Style Embedding Distance)**
    * **文件：** `style_vector.py`
    * **逻辑：** 使用开源中文 Embedding 模型(微调 Style Encoder 或 Style Embedding 模型 或 其他方法)，分别计算目标风格文本的平均向量和 `generated_text` 的向量，求余弦相似度 (Cosine Similarity)。

4.  **风格相似度 - 语言学特征统计 (Linguistic Feature Stats)**
    * **文件：** `linguistic.py`
    * **核心逻辑：** 弃用主观的加权单一得分和绝对误差，直接统计并输出“原文本 (source)”、“目标风格参考文本 (target)”和“生成文本 (generated)”在 4 个核心维度上的【原始真实数值】，以观察风格特征的迁移方向和程度。
    * **Target 标尺的动态解析与 Zero-shot 运行流 ：**
        * **Few-shot / RAG 场景：** 评估器直接接收并解析生成器动态传入的 `style_references` ，计算其特征值作为本次任务的 Target 标尺。
        * **Zero-shot 场景 (引入隐藏 Ground Truth)：** 生成器阶段的 `style_references` 为空。在评估器阶段，代码需根据传入的风格标签（如 "luxun"），主动去 `data/raw/target_styles.json` (隐藏数据库) 中读取预设的真实文本，计算其特征平均值作为 Target 标尺。（**注意：** 该预设风格库目前仍在梳理中，跑 MVP 测试前需先在 JSON 文件中填入待测风格及 2-3 篇示例）。
    * **依赖工具：** 强制使用 `jieba` 进行分词，使用 `jieba.posseg` 进行词性标注。
    * **需计算的 4 个核心维度及口径：**
        1. **平均句长 (avg_len)：** `文本总字符数 / 句子总数`（基于正则匹配 `[。！？…]` 作为断句符）。反映文本的“呼吸感”。
        2. **词汇丰富度 (ttr)：** Type-Token Ratio (TTR) = `去重后的独立词汇数 / 总分词数`。反映用词变化率。
        3. **形容词密度 (adj_density)：** `被标注为 'a' (形容词) 的词汇数 / 总分词数`。反映描写浓度。
        4. **副词密度 (adv_density)：** `被标注为 'd' (副词) 的词汇数 / 总分词数`。反映情绪烈度和语气修饰习惯。
    * **最终输出格式：** 必须返回一个嵌套字典，记录三者的原始数值，以便主程序生成迁移轨迹报告。
      ```json
      {
        "avg_len": {"source": 18.5, "target": 6.2, "generated": 8.1},
        "ttr": {"source": 0.35, "target": 0.65, "generated": 0.52},
        "adj_density": {"source": 0.05, "target": 0.12, "generated": 0.09},
        "adv_density": {"source": 0.02, "target": 0.08, "generated": 0.06}
      }
      ```

5.  **语言流利度 - Fluency (100/NLL)**
    * **文件：** `fluency_nll.py`
    * **参考来源：** LMStyle Benchmark
    * **逻辑：** 使用一个预训练的因果语言模型 (Causal LM，例如中文项目可选用开源的 `gpt2-chinese`) 计算 `generated_text` 的负对数似然 (Negative Log Likelihood, NLL)。由于 NLL 越小表示文本越流利自然，因此采用 `100 / NLL` 作为最终得分，使其变为“分数越高、流利度越好”的正向指标。