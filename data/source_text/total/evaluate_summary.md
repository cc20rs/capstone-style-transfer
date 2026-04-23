目前完整评估体系（字段名 + 方法/模型）

结果主字段（在 metrics 下）
bert_score
style_label
style_comment 
linguistic_stats
fluency_score
details

定义位置：
main_pipeline.py
app.py

各指标方法与模型
bert_score
方法：BERTScore-F1
实现：SemanticBERTScoreMetric
模型：bert_score_model（当前配置为 bert-base-chinese，本地可映射）
代码： semantic.py

style_vector_score
旧方法：sentence-transformers 余弦相似度
新方法：LLM-as-a-Judge 四档标签
实现：StyleVectorDistanceMetric
模型：judge_model（当前配置 DeepSeek-V3.2）
核心 system prompt 来源： style_system_prompt.md
输出标签严格限制：完全符合 / 比较符合 / 比较不符合 / 完全不符合
代码： style_vector.py

linguistic_stats
方法：jieba + jieba.posseg 统计特征
字段：avg_len, ttr, adj_density, adv_density（各含 source/target/generated）
实现：LinguisticFeatureMetric
代码： linguistic.py

fluency_score
方法：语言模型 NLL，得分为 100 / NLL
实现：FluencyNLLMetric
模型：fluency_model（当前配置 uer/gpt2-chinese-cluecorpussmall，本地可映射）
代码： fluency_nll.py



v1用正常prompt
v2少了第三层 ＋ 有规则
v3正常prompt ＋ 有规则

data\source_text\total\style-evaluate.jsonl 和
data\source_text\total\style-evaluate_badcase.jsonl
使用的prompt比目前使用的 data\source_text\total\style_system_prompt.md
多了第三层
```
按“字词结构—情感修辞—宏观背景”三层级风格特征抓取方法提取特征，各层级分析方法与核心要点如下：
1.字词结构层（浅层分析）：以词和句子为单位，量化分析遣词造句规律。重点统计字符组合、字频、标点及表情符号使用情况；分析词汇组合模式、功能词偏好、词汇丰富度及非标准词汇使用；梳理词性组合、句子成分关联及词长、句长、段落长度的分布规律。该模块能够识别作者写作时偏 好的词汇、句式和语法结构，从而理解作者的写作技巧和语言习惯。

2.情感修辞层（中层分析）：以文本片段为单位，聚焦语言艺术与情感表达。通过算法识别并统计积极、消极、中性情感的分布及切换频率，判断情感倾向；统计比喻、拟人等修辞手法的使用频次与组合能力，挖掘语言表达特色。

3.宏观背景层（深层分析）：以整篇文本为单位，结合外部语境分析。梳理文本段落排布、逻辑脉络等篇章结构；提炼核心主题及表达模式；挖掘文本蕴含的文化传统、价值观念，结合时代环境、场景设定等外部因素，解读深层风格内涵。
具体可提取主题特征、摘要特征、文化语境特征和创作背景特征，其中主题和摘要特征通过相应分析工具提取，文化语境和创作背景特征通过分类算法 提取。
```



我刚才确认到真正的风格评估调用只在解耦脚本里，主流程那边是别的 metrics 封装；这次只改判别器和这一个评测入口，范围就够了。