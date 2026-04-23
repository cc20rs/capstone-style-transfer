from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.fluency_nll import FluencyNLLMetric
from src.evaluator.semantic import SemanticBERTScoreMetric
from src.evaluator.style_vector import StyleVectorDistanceMetric
from src.generator.context_eng import ContextEngineeringGenerator
from src.generator.prompt_eng import PromptEngineeringGenerator
from src.generator.rag_icl import RAGICLGenerator
from src.generator.scene_recall import CascadeSceneRetriever
from src.utils.api_client import APIClient
from src.utils.file_io import load_json, load_yaml, read_jsonl
from src.utils.model_hub import configure_huggingface_mirror


def normalize_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        merged: List[str] = []
        for nested in value.values():
            if isinstance(nested, list):
                merged.extend([item.strip() for item in nested if isinstance(item, str) and item.strip()])
            elif isinstance(nested, str) and nested.strip():
                merged.append(nested.strip())
        return merged
    return []


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_runtime() -> Dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_yaml(PROJECT_ROOT / "configs" / "config.yaml")
    configure_huggingface_mirror(cfg)

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment variables")

    api_client = APIClient(cfg, api_key)

    return {
        "config": cfg,
        "api_client": api_client,
        "source_cases": load_json(PROJECT_ROOT / "data" / "source_text" / "source_text.json").get("cases", []),
        "corpus_rows": load_json(PROJECT_ROOT / "data" / "source_text" / "corpus_text.json").get("corpus", []),
        "context_refs": load_json(PROJECT_ROOT / "data" / "source_text" / "data_to_generate" / "context_eng_references.json"),
    }


def run_generate(output_path: Path) -> None:
    rt = load_runtime()
    cfg = rt["config"]
    api_client = rt["api_client"]
    source_cases = rt["source_cases"]
    corpus_rows = rt["corpus_rows"]
    context_refs = rt["context_refs"]

    prompt_gen = PromptEngineeringGenerator(cfg, api_client)
    context_gen = ContextEngineeringGenerator(cfg, api_client)
    rag_gen = RAGICLGenerator(cfg, api_client)

    author_corpus: Dict[str, List[str]] = {}
    for row in corpus_rows:
        author = str(row.get("author", "")).strip()
        content = str(row.get("content", "")).strip()
        if not author or not content:
            continue
        author_corpus.setdefault(author, []).append(content)

    rows: List[Dict[str, Any]] = []

    for case in source_cases:
        case_id = str(case.get("case_id", "")).strip()
        source_text = str(case.get("source_text", "")).strip()
        target_style = str(case.get("style_name", "")).strip()
        stage = case.get("stage")
        location = case.get("location")
        event = case.get("event")
        if not case_id or not source_text or not target_style:
            continue

        strategy_map = [
            ("A", prompt_gen, []),
            ("B", context_gen, normalize_text_list(context_refs.get(target_style, []))[:3]),
            ("C", rag_gen, author_corpus.get(target_style, [])),
        ]

        for strategy, generator, refs in strategy_map:
            if strategy == "C":
                generated = generator.generate(
                    source_text=source_text,
                    target_style_name=target_style,
                    style_references=refs,
                    stage=stage,
                    location=location,
                    event=event,
                )
            else:
                generated = generator.generate(
                    source_text=source_text,
                    target_style_name=target_style,
                    style_references=refs,
                )

            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "case": case_id,
                    "目标风格": target_style,
                    "策略": strategy,
                    "source_text": source_text,
                    "style_references": generated.get("style_references", []),
                    "生成文本": generated.get("generated_text", ""),
                }
            )

    write_jsonl(output_path, rows)
    print(f"[OK] generation written: {output_path} rows={len(rows)}")


def run_evaluate(dataset_path: Path) -> None:
    rt = load_runtime()
    cfg = rt["config"]
    api_client = rt["api_client"]
    source_cases = rt["source_cases"]
    corpus_rows = rt["corpus_rows"]

    semantic_metric = SemanticBERTScoreMetric(cfg)
    style_metric = StyleVectorDistanceMetric(cfg, api_client)
    fluency_metric = FluencyNLLMetric(cfg)
    retriever = CascadeSceneRetriever(cfg, corpus_rows)

    case_map: Dict[str, Dict[str, Any]] = {
        str(c.get("case_id", "")).strip(): c for c in source_cases if str(c.get("case_id", "")).strip()
    }

    rows = read_jsonl(dataset_path)
    if not rows:
        print(f"[WARN] empty dataset: {dataset_path}")
        return

    errors: List[str] = []
    success = 0

    for i, row in enumerate(rows):
        case_id = str(row.get("case", "")).strip()
        target_style = str(row.get("目标风格", "")).strip()
        generated_text = str(row.get("生成文本", "")).strip()

        case_obj = case_map.get(case_id, {})
        source_text = str(case_obj.get("source_text", "")).strip()
        stage = case_obj.get("stage")
        location = case_obj.get("location")
        event = case_obj.get("event")

        if not source_text or not generated_text or not target_style:
            errors.append(f"row={i} missing required fields")
            continue

        try:
            refs, _ = retriever.retrieve_topk_refs(
                source_text=source_text,
                target_author=target_style,
                stage=stage,
                location=location,
                event=event,
                candidate_k=10,
                top_k=3,
            )

            semantic = semantic_metric.evaluate(source_text, generated_text, refs)
            style = style_metric.evaluate(source_text, generated_text, target_style=target_style, style_references=refs)
            fluency = fluency_metric.evaluate(source_text, generated_text, refs)

            row["bert_score"] = float(semantic.get("score", 0.0))
            row["style_label"] = str(style.get("details", {}).get("label", ""))
            row["style_comment"] = str(style.get("details", {}).get("comment", ""))
            row["fluency_score"] = float(fluency.get("score", 0.0))
            row.setdefault("human_label", None)
            row.setdefault("human_comment", None)
            success += 1
        except Exception as exc:
            errors.append(f"row={i}, case={case_id}, style={target_style}, error={exc}")

    write_jsonl(dataset_path, rows)
    print(f"[OK] evaluated rows={success}/{len(rows)} -> {dataset_path}")
    if errors:
        print("[WARN] errors:")
        for msg in errors:
            print(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decoupled generation/evaluation pipeline for source_text data")
    sub = parser.add_subparsers(dest="mode", required=True)

    gen = sub.add_parser("generate", help="Run generation stage only")
    gen.add_argument(
        "--output",
        default="data/source_text/total/generated_cases_decoupled.jsonl",
        help="Path to generation output jsonl",
    )

    eva = sub.add_parser("evaluate", help="Run evaluation stage only")
    eva.add_argument(
        "--dataset",
        default="data/source_text/total/style-evaluate.jsonl",
        help="Path to style-evaluate jsonl",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "generate":
        run_generate(PROJECT_ROOT / args.output)
    elif args.mode == "evaluate":
        run_evaluate(PROJECT_ROOT / args.dataset)


if __name__ == "__main__":
    main()
