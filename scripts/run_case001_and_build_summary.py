from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import EXPERIMENT_JSONL_PATH, StyleTransferWebUI


STRATEGY_MAP = {
    "Prompt_Eng": "Baseline_A_Prompt_Eng",
    "Context_Eng": "Baseline_B_Context_Eng",
    "RAG_ICL": "Baseline_C_RAG_ICL",
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _pick_latest_per_method(rows: List[Dict[str, Any]], case_id: str) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("case_id") != case_id:
            continue
        method = row.get("method")
        if method not in STRATEGY_MAP:
            continue
        current = latest.get(method)
        if current is None:
            latest[method] = row
            continue
        if int(row.get("version", 0)) >= int(current.get("version", 0)):
            latest[method] = row
    ordered = [latest[m] for m in ["Prompt_Eng", "Context_Eng", "RAG_ICL"] if m in latest]
    return ordered


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def _build_summary_md(records: List[Dict[str, Any]]) -> str:
    header = [
        "# case001_summary（三种方法对比）",
        "",
        "## 一、表格一（内容保真 → 风格相似 → 语言流利 + 语言学特征）",
        "",
        "| 样本ID(case_id) | 策略(strategy) | 目标风格(style_name) | 语义保真(BERTScore-F1) | 风格向量相似度(style_vector) | 流利度(100/NLL) | 句长-source | 句长-target | 句长-generated | TTR-source | TTR-target | TTR-generated | 形容词密度-source | 形容词密度-target | 形容词密度-generated | 副词密度-source | 副词密度-target | 副词密度-generated |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for rec in records:
        method = rec.get("method", "")
        strategy = STRATEGY_MAP.get(method, method)
        ling = rec.get("linguistic_stats", {})
        avg = ling.get("avg_len", {})
        ttr = ling.get("ttr", {})
        adj = ling.get("adj_density", {})
        adv = ling.get("adv_density", {})
        rows.append(
            "| "
            + " | ".join(
                [
                    str(rec.get("case_id", "")),
                    strategy,
                    str(rec.get("target_style", "")),
                    _fmt(float(rec.get("bert_score", 0.0))),
                    _fmt(float(rec.get("style_vector_score", 0.0))),
                    _fmt(float(rec.get("fluency_score", 0.0))),
                    _fmt(float(avg.get("source", 0.0))),
                    _fmt(float(avg.get("target", 0.0))),
                    _fmt(float(avg.get("generated", 0.0))),
                    _fmt(float(ttr.get("source", 0.0))),
                    _fmt(float(ttr.get("target", 0.0))),
                    _fmt(float(ttr.get("generated", 0.0))),
                    _fmt(float(adj.get("source", 0.0))),
                    _fmt(float(adj.get("target", 0.0))),
                    _fmt(float(adj.get("generated", 0.0))),
                    _fmt(float(adv.get("source", 0.0))),
                    _fmt(float(adv.get("target", 0.0))),
                    _fmt(float(adv.get("generated", 0.0))),
                ]
            )
            + " |"
        )

    section2 = [
        "",
        "## 二、表格二（仅大模型裁判评分对比，最右侧为评语）",
        "",
        "| 样本ID(case_id) | 策略(strategy) | 目标风格(style_name) | DeepSeek-V3.2-词汇 | DeepSeek-V3.2-句法 | DeepSeek-V3.2-情绪 | DeepSeek-V3.2-总分 | DeepSeek-V3.2-评语 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    section2_rows = []
    for rec in records:
        method = rec.get("method", "")
        strategy = STRATEGY_MAP.get(method, method)
        details = rec.get("details", {})
        deepseek = details.get("llm_judge", {}) if isinstance(details.get("llm_judge", {}), dict) else {}
        section2_rows.append(
            "| "
            + " | ".join(
                [
                    str(rec.get("case_id", "")),
                    strategy,
                    str(rec.get("target_style", "")),
                    _fmt(float(deepseek.get("lexical", 0.0))),
                    _fmt(float(deepseek.get("syntax", 0.0))),
                    _fmt(float(deepseek.get("emotion", 0.0))),
                    _fmt(float(deepseek.get("overall", 0.0))),
                    str(deepseek.get("comment", "")).replace("\n", " "),
                ]
            )
            + " |"
        )

    section3 = ["", "## 三、原文与生成文本对照", ""]
    for idx, rec in enumerate(records, start=1):
        method = rec.get("method", "")
        strategy = STRATEGY_MAP.get(method, method)
        refs = rec.get("style_references", [])
        if isinstance(refs, list) and refs:
            refs_text = " | ".join(str(x) for x in refs)
        else:
            refs_text = "（无，Zero-shot）"
        section3.extend(
            [
                f"### {idx}. {rec.get('case_id', '')} / {strategy} / {rec.get('target_style', '')}",
                f"- 原文：{rec.get('source_text', '')}",
                f"- 生成：{rec.get('generated_text', '')}",
                f"- 参考：{refs_text}",
                "",
            ]
        )

    return "\n".join(header + rows + section2 + section2_rows + section3)


def main() -> None:
    ui = StyleTransferWebUI()

    case_id = "case_001"
    scenario = "叙事"
    case_source = ui.repo.get_source_text_by_case_id(case_id)
    if not case_source:
        raise RuntimeError("case_001 source_text not found")

    target_style = ui.style_labels[0]

    methods = ["Prompt_Eng", "Context_Eng", "RAG_ICL"]
    for method in methods:
        refs_update = ui._get_style_refs_update(method, target_style, scenario, case_source)
        style_refs = refs_update.get("value", "") if isinstance(refs_update, dict) else ""
        system_prompt = ui.DEFAULT_SYSTEM_PROMPT.format(target_style=target_style, scenario=scenario)
        generated_text, metrics, runtime_prompt, runtime_refs = ui._run_experiment(
            case_id=case_id,
            source_text=case_source,
            target_style=target_style,
            scenario=scenario,
            method=method,
            system_prompt=system_prompt,
            style_references=style_refs,
        )
        if isinstance(metrics, dict) and "error" in metrics:
            raise RuntimeError(f"{method} failed: {metrics['error']}")

        status = ui._save_record(
            case_id=case_id,
            source_text=case_source,
            target_style=target_style,
            scenario=scenario,
            method=method,
            system_prompt=system_prompt,
            style_references=style_refs,
            runtime_system_prompt=runtime_prompt,
            runtime_style_references=runtime_refs,
            generated_text=generated_text,
            metrics=metrics,
        )
        print(status)

    all_rows = _read_jsonl(EXPERIMENT_JSONL_PATH)
    selected = _pick_latest_per_method(all_rows, case_id="case_001")
    if len(selected) != 3:
        raise RuntimeError(f"Expected 3 records for case_001 methods, got {len(selected)}")

    summary_md = _build_summary_md(selected)
    output_path = Path("data/outputs/case001_summary.md")
    output_path.write_text(summary_md, encoding="utf-8")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
