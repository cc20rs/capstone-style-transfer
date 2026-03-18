## 2) 本次实际使用的模型 / 接口 / 参数

### 2.1 生成与裁判接口
- **接口协议**：OpenAI Compatible Chat Completions
- **API 网关**：`https://llmapi.paratera.com/v1`
- **生成模型**：`DeepSeek-V3.2-Instruct`
- **裁判模型**：`DeepSeek-V3.2`、`Qwen3-Next-80B-A3B-Instruct`

### 2.2 评估相关模型
- **BERTScore**：`bert-base-chinese`
- **Style Embedding**：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Fluency NLL**：`uer/gpt2-chinese-cluecorpussmall`

### 2.3 关键运行参数（来自 config）
- `generation.temperature = 0.7`
- `generation.max_tokens = 1024`
- `rag.top_k = 3`
- `api.timeout_seconds = 60`
- `api.max_retries = 3`
- `api.retry_backoff_seconds = 1.5`
- `huggingface.endpoint = https://hf-mirror.com`

### 2.4 本次基线策略
- `Baseline_A_Prompt_Eng`（Zero-shot）
- `Baseline_B_Context_Eng`（Static Few-shot）
- `Baseline_C_RAG_ICL`（Dynamic Few-shot）

---

## 3) 指标含义（直观版）

### 3.1 `bert_score`（语义保真）
- 含义：生成文本与原事实文本在语义上的一致程度（越高越好）
- 直观理解：越高表示“事实没丢、偏离更少”
- 本轮实际方法：**标准 `BERTScore-F1`**，15 条结果全部一致

### 3.2 `llm_judge_score`（风格主观评分）
- 含义：LLM 从词汇、句法、情绪三个维度综合给分（0~10，越高越好）
- 直观理解：越高表示“更像目标文风”
- 当前同时记录双裁判明细，但兼容字段仍保留主裁判总分

### 3.3 `style_vector_score`（风格向量相似度）
- 含义：生成文本向量与目标风格参考向量的余弦相似度（越高越好）
- 直观理解：越高表示“在向量空间里更接近目标风格”

### 3.4 `fluency_score = 100 / NLL`（流利度）
- 含义：由语言模型困惑程度反推的流畅度分数（越高越好）
- 直观理解：越高表示“读起来更顺、更自然”
- 本轮实际方法：**标准 `100 / NLL`**，15 条结果全部一致

### 3.5 语言学统计（`linguistic_stats`）
- 维度：`avg_len`、`ttr`、`adj_density`、`adv_density`
- 含义：不是单一好坏分，而是看 `generated` 与 `target` 的“迁移方向和接近程度”

---

## 4) 指标差异怎么解读（对比口径）

- `bert_score` 高 + `style_vector_score` 低：保真不错，但风格迁移不充分
- `style_vector_score` 高 + `bert_score` 低：风格像了，但事实偏移风险更大
- `llm_judge_score` 高而向量分一般：主观风格感强，但客观向量不一定一致
- `fluency_score` 高不代表风格准：只说明语句更顺，不直接保证风格/事实

建议：**四类核心分数联合看，不看单项冠军**。

---

## 5) 本次结果分析

## 5.1 全局（按策略均值）
- `Baseline_A_Prompt_Eng`：
  - `bert_score ≈ 0.7498`
  - `llm_judge_score ≈ 5.66`
  - `style_vector_score ≈ 0.5038`
  - `fluency_score ≈ 21.19`
- `Baseline_B_Context_Eng`：
  - `bert_score ≈ 0.7512`（三者最高）
  - `llm_judge_score ≈ 6.64`
  - `style_vector_score ≈ 0.4995`
  - `fluency_score ≈ 22.82`（三者最高）
- `Baseline_C_RAG_ICL`：
  - `bert_score ≈ 0.7255`
  - `llm_judge_score ≈ 8.08`（主观评分最高）
  - `style_vector_score ≈ 0.4899`
  - `fluency_score ≈ 22.66`

### 5.2 样本级观察
- `case_001 (luxun)`：RAG 的 `llm_judge_score` 与 `style_vector_score` 都最高，说明该样本中动态检索帮助明显。
- `case_002 (qianzhongshu)`：Static Few-shot 的主观风格分与向量分都最高，说明固定样例对该风格更稳定。
- `case_003 (luxun)`：Static Few-shot 的 `bert_score` 与 `style_vector_score` 更优，但 RAG 的主观风格评分最高。
- `case_004 (luxun)`：Static Few-shot 的保真最高，而 Zero-shot 的向量分最高，说明该样本存在指标分歧。
- `case_005 (luxun)`：RAG 的保真和主观风格都最好，但 Zero-shot 的向量分更高，说明风格主观感受与客观向量并不总一致。

### 5.3 小结
- 当前 5 个 case 上，**Baseline B（Static Few-shot）最稳健**，在保真与流利度上综合领先。
- **Baseline C（RAG+ICL）在主观风格感上优势最明显**，但稳定性仍依赖检索片段质量。
- Zero-shot 可作为低成本基线，但在主观风格上明显弱于 B/C。

---

## 6) 风险与下一步建议

- 样本量已从 2 个扩到 5 个，但统计显著性仍不足；建议扩展到 30~100 个 case 再定最终策略结论。
- RAG 结果波动仍较大，建议：
  1) 提升检索相关性（语义相似度 + 风格约束联合排序）
  2) 约束参考片段长度/风格纯度
  3) 增加检索结果重排与风格纯度过滤
  4) 为各策略做分层报告（按风格标签、句长区间、事实密度）
- 当前双裁判明细已完整落盘，后续可考虑做双裁判融合总分，减少单模型主观偏差。
- 可将最终评分做归一化加权总分（保留单项分）便于一眼比较。
