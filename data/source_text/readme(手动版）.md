使用`data\source_text\data_scripts\build_corpus_text_json.py`批处理脚本：解析指定的 docx 的【编号】条目、清理结构噪声、逐条调用 DeepSeek-V3.2 标注三维标签并做白名单校验，最后合并写到 `data/source_text/corpus_text.json`

`data\source_text\scene_and_recall.md`的作用：记录了多维标签的取值范围，以及制定了级联召回的方案




`data\source_text\语料库（未清洗）\data_clean_prompt.md`的作用：用来描述 docx 文档的结构，方便 agent 写`build_corpus_text_json.py`脚本清洗数据（非必要不再使用，归为一次性文档）

`data\source_text\data_scripts\clean_scene_tags_with_llm.py`脚本的作用：`data\source_text\语料库（未清洗）\测试文本.docx` case数据初次解析得到的场景不规范，用此脚本来清洗，属于一次性脚本