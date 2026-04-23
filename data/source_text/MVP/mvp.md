请帮我编写一个完整的 Python MVP 验证脚本，用于演示“基于人类偏好学习提取 LLM 风格向量”的流程。
脚本不仅需要执行偏好学习优化，还需要读取真实本地文件，并最终自动生成一份图文并茂的 `MVP_summary.md` 汇报文档。

核心要求与执行步骤如下，请在一个完整的 Python 文件中实现：

**1. 真实数据加载与规则 Mock (Data Loading & Simulation):**
- 读取本地 `source_text.json`，仅截取前 5 条 cases 作为待转化文本（Query）。
- 目标风格设定为 2 种：“余华”和“汪曾祺”。
- 策略设定为 3 种（A: Prompt, B: Context, C: RAG）。总计生成样本数为 5 * 2 * 3 = 30 个。生成文本的方法请严格按照`main_pipeline.py`中的方法。
- 加载 `corpus_text.json`。对于每一个目标作家，统计其在语料库中的真实条数 $N$。
- **特征**：加载本项目指定的本地 Embedding 模型。请根据统计出的条数，生成作家的语料库多层特征矩阵 `corpus_features`，形状为 (N, 24, 1024)。同样，为 30 个生成文本生成 `gen_features`，形状为 (30, 24, 1024)。
- **Mock 人类打分 (human_scores)**：范围 1-10。请务必遵循规则 `A < B ≈ C` 来随机生成。例如：策略 A 的分数在 4-6 之间，策略 B 和 C 的分数在 6-9 之间。
- **Mock 或计算 BERTScore**：使用本项目指定的本地模型为这 30 个生成样本计算 BERTScore 用于展示。

**2. 核心模型定义 (WeightedStyleModel):**
- 使用 `nn.Module`，定义可学习参数 `weights`，形状为 (24,)。
- **强制初始化**：将最后一层（index -1）初始化为 1.0，其余所有层初始化为 0.0。
- `forward` 函数：使用 `weights` 对特征矩阵在层数维度进行加权求和。

**3. 前置评估与 RAG 召回记录 (Pre-tuning & Retrieval):**
- 计算各作家的初始“风格质心”（corpus_features 的加权平均）。
- 对于策略 C 的样本，从 `corpus_text.json` 中抽取 3 条属于该作家的文本，记录为“召回的 Top-3 文本”。抽取匹配规则是（级联召回 + 风格精排），具体请查阅`data\source_text\scene_and_recall.md`。
- 计算 30 个 `gen_features` 与对应作家质心的初始余弦相似度 `pre_sims`，并计算其与 `human_scores` 的斯皮尔曼相关系数。

**4. 训练数据构建 (Triplet Generation):**
- 在同一个 Case 和同一个目标作家的 3 个文本（A, B, C）中，基于 `human_scores` 两两组合，构建 (High_Score_Idx, Low_Score_Idx) 偏好对。分数相同的文本则跳过。

**5. 偏好对齐训练与可视化 (Training & Plotting):**
- 优化器：Adam (lr=0.01)。
- 损失函数：`Total_Loss = F.margin_ranking_loss(sim_high, sim_low, target=1, margin=0.1) + 0.005 * torch.norm(weights, p=1)`。
- 迭代 100 个 Epoch，记录每个 Epoch 的 Loss。
- 训练结束后，使用 `matplotlib` 绘制 Loss 下降曲线，并保存为本地图片 `alignment_loss_curve.png`。

**6. 后置评估 (Post-tuning Evaluation):**
- 使用优化后的 `weights` 重新计算质心和 `post_sims`。
- 计算 `post_sims` 与 `human_scores` 的斯皮尔曼相关系数。

**7. 自动生成 MVP_summary.md 报告:**
- 脚本最后，使用 Python 自动写入生成一份 Markdown 文档，必须包含以下内容板块：
  - **1. 对齐优化思路概述**：简述如何通过同一 Case 下不同策略的分数差组成三元数据对，写出 Margin Ranking Loss 和 L1 正则化的公式，解释优化的目的是为了让加权向量更贴近人类直觉。
  - **2. 整体指标变化**：展示对齐前 vs 对齐后的斯皮尔曼相关性系数对比，以及初始权重（[0,0,...,1]）到优化后权重的变化数组。
  - **3. 损失函数下降图**：插入生成的 `![Loss Curve](alignment_loss_curve.png)`。
  - **4. Case 抽样细节展示**：抽取 30 个样本中的 1 个具体 Case（例如 Case 1 转化为余华）。详细列出：
    - 源文本 (Source Text)
    - 策略 B 的参考文本来源说明
    - 策略 C 具体召回的 Top-3 语料文本
    - 策略 A, B, C 各自的 Mock BERTScore 和 Human Score
    - 策略 A, B, C 调优前后的风格向量余弦相似度变化对比（展示低分文本相似度变低，高分文本相似度变高的趋势）。