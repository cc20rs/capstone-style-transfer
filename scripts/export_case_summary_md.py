from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def df_to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        vals = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    input_path = Path("data/outputs/case_summary.jsonl")
    md_path = Path("data/outputs/case_summary.md")

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("case_summary.jsonl is empty")

    judge_models: List[str] = []
    first_details = rows[0].get("metrics", {}).get("details", {}).get("llm_judge_by_model", {})
    if isinstance(first_details, dict):
        judge_models = list(first_details.keys())

    records: List[Dict[str, Any]] = []
    for item in rows:
        metrics = item.get("metrics", {})
        linguistic = metrics.get("linguistic_stats", {})
        details = metrics.get("details", {})
        by_model = details.get("llm_judge_by_model", {})

        rec: Dict[str, Any] = {
            "时间戳(timestamp)": item.get("timestamp", ""),
            "样本ID(case_id)": item.get("case_id", ""),
            "策略(strategy)": item.get("strategy", ""),
            "目标风格(style_name)": item.get("style_name", ""),
            "原文(source_text)": item.get("source_text", ""),
            "参考文本(style_references)": " | ".join(item.get("style_references", [])),
            "生成文本(generated_text)": item.get("generated_text", ""),
            "语义保真(BERTScore-F1)": metrics.get("bert_score", ""),
            "风格向量相似度(style_vector)": metrics.get("style_vector_score", ""),
            "流利度(100/NLL)": metrics.get("fluency_score", ""),
            "句长-source": linguistic.get("avg_len", {}).get("source", ""),
            "句长-target": linguistic.get("avg_len", {}).get("target", ""),
            "句长-generated": linguistic.get("avg_len", {}).get("generated", ""),
            "TTR-source": linguistic.get("ttr", {}).get("source", ""),
            "TTR-target": linguistic.get("ttr", {}).get("target", ""),
            "TTR-generated": linguistic.get("ttr", {}).get("generated", ""),
            "形容词密度-source": linguistic.get("adj_density", {}).get("source", ""),
            "形容词密度-target": linguistic.get("adj_density", {}).get("target", ""),
            "形容词密度-generated": linguistic.get("adj_density", {}).get("generated", ""),
            "副词密度-source": linguistic.get("adv_density", {}).get("source", ""),
            "副词密度-target": linguistic.get("adv_density", {}).get("target", ""),
            "副词密度-generated": linguistic.get("adv_density", {}).get("generated", ""),
        }

        for model_name in judge_models:
            model_result = by_model.get(model_name, {})
            model_details = model_result.get("details", {}) if isinstance(model_result, dict) else {}
            rec[f"{model_name}-词汇"] = model_details.get("lexical", "")
            rec[f"{model_name}-句法"] = model_details.get("syntax", "")
            rec[f"{model_name}-情绪"] = model_details.get("emotion", "")
            rec[f"{model_name}-总分"] = model_details.get("overall", model_result.get("score", "") if isinstance(model_result, dict) else "")
            rec[f"{model_name}-评语"] = model_details.get("comment", "")

        records.append(rec)

    df = pd.DataFrame(records)

    table1_cols = [
        "样本ID(case_id)",
        "策略(strategy)",
        "目标风格(style_name)",
        "语义保真(BERTScore-F1)",
        "风格向量相似度(style_vector)",
        "流利度(100/NLL)",
        "句长-source",
        "句长-target",
        "句长-generated",
        "TTR-source",
        "TTR-target",
        "TTR-generated",
        "形容词密度-source",
        "形容词密度-target",
        "形容词密度-generated",
        "副词密度-source",
        "副词密度-target",
        "副词密度-generated",
    ]

    table2_cols = ["样本ID(case_id)", "策略(strategy)", "目标风格(style_name)"]
    for model_name in judge_models:
        table2_cols.extend([
            f"{model_name}-词汇",
            f"{model_name}-句法",
            f"{model_name}-情绪",
            f"{model_name}-总分",
        ])
    for model_name in judge_models:
        table2_cols.append(f"{model_name}-评语")

    lines: List[str] = [
        "# case_summary 样本对比（中文释义版）",
        "",
        "## 一、字段中文释义",
        "",
        "| 字段路径 | 中文含义 |",
        "|---|---|",
        "| timestamp | 记录产生时间（ISO 格式） |",
        "| case_id | 测试样本唯一编号 |",
        "| strategy | 基线策略名称（A/B/C） |",
        "| style_name | 目标风格标签 |",
        "| source_text | 原始事实文本 |",
        "| style_references | 风格参考片段列表（Zero-shot 为空） |",
        "| generated_text | 模型生成文本 |",
        "| metrics.bert_score | 语义保真分，越高越好 |",
        "| metrics.style_vector_score | 风格向量相似度，越高越好 |",
        "| metrics.fluency_score | 流利度分（100/NLL），越高越好 |",
        "| metrics.linguistic_stats.* | 语言学迁移特征（source/target/generated） |",
        "| metrics.details.llm_judge_by_model | 多裁判模型明细（维度分+评语） |",
        "",
        "## 二、表格一（内容保真 → 风格相似 → 语言流利 + 语言学特征）",
        "",
        "### 表格一指标参考范围（建议值，用于快速判读）",
        "",
        "| 指标 | 建议可接受范围 | 判读说明 |",
        "|---|---|---|",
        "| 语义保真(BERTScore-F1) | >= 0.60（中文短文本任务） | 越高表示事实语义保留更好 |",
        "| 风格向量相似度(style_vector) | >= 0.50（经验阈值） | 越高表示向量空间更接近目标风格 |",
        "| 流利度(100/NLL) | >= 20（当前模型经验值） | 越高表示语句更流畅自然 |",
        "| 句长(avg_len) | generated 与 target 相对差 <= 30% | 越接近目标风格句长节奏越好 |",
        "| 词汇丰富度(TTR) | generated 与 target 绝对差 <= 0.15 | 过低可能单调，过高可能离散 |",
        "| 形容词/副词密度 | generated 与 target 绝对差 <= 0.05 | 观察修饰强度是否向目标迁移 |",
        "",
        df_to_markdown_table(df[table1_cols]),
        "",
        "## 三、表格二（仅大模型裁判评分对比，最右侧为评语）",
        "",
        df_to_markdown_table(df[table2_cols]),
        "",
        "## 四、原文与生成文本对照",
        "",
    ]

    for idx, row in df.iterrows():
        lines.append(f"### {idx + 1}. {row['样本ID(case_id)']} / {row['策略(strategy)']} / {row['目标风格(style_name)']}")
        lines.append(f"- 原文：{row['原文(source_text)']}")
        lines.append(f"- 生成：{row['生成文本(generated_text)']}")
        refs = row["参考文本(style_references)"]
        lines.append(f"- 参考：{refs if refs else '（无，Zero-shot）'}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {md_path}")


if __name__ == "__main__":
    main()
