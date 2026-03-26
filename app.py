from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPERIMENT_JSONL_PATH = PROJECT_ROOT / "experiment_records.jsonl"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CONTEXT_ENG_STYLE_REFERENCES_PATH = DATA_RAW_DIR / "Context_Eng_style_references.json"


@dataclass
class PipelineResult:
    generated_text: str
    metrics: Dict[str, Any]


class DataRepository:
    def __init__(self) -> None:
        self.test_cases = self._load_json(DATA_RAW_DIR / "test_cases.json").get("cases", [])
        self.target_styles = self._load_json(DATA_RAW_DIR / "target_styles.json")
        self.style_corpus = self._load_json(DATA_RAW_DIR / "style_corpus.json")
        self.context_eng_style_references = self._load_json(CONTEXT_ENG_STYLE_REFERENCES_PATH)
        self.config = self._load_config(CONFIG_PATH)
        self._rag_index_cache: Dict[str, Tuple[TfidfVectorizer, Any, List[str]]] = {}

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def mock_fetch_case_ids(self) -> List[str]:
        case_ids: List[str] = []
        for item in self.test_cases:
            case_id = item.get("case_id")
            if isinstance(case_id, str) and case_id.strip():
                case_ids.append(case_id.strip())
        seen = set()
        unique_case_ids: List[str] = []
        for case_id in case_ids:
            if case_id in seen:
                continue
            seen.add(case_id)
            unique_case_ids.append(case_id)
        return unique_case_ids

    def get_source_text_by_case_id(self, case_id: str) -> str:
        for item in self.test_cases:
            if item.get("case_id") == case_id:
                return item.get("source_text", "")
        return ""

    def get_context_references(self, style_key: str) -> List[str]:
        refs = self._normalize_text_list(self.target_styles.get(style_key, []))
        return refs[:3]

    def get_context_references_by_scenario(self, style_key: str, scenario: str) -> List[str]:
        scenario_block = self.context_eng_style_references.get(style_key, {})
        refs = self._normalize_text_list(scenario_block.get(scenario, []) if isinstance(scenario_block, dict) else [])
        return refs[:3]

    @staticmethod
    def _normalize_text_list(value: Any) -> List[str]:
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

    def _build_rag_index(self, style_key: str) -> Tuple[TfidfVectorizer, Any, List[str]] | None:
        if style_key in self._rag_index_cache:
            return self._rag_index_cache[style_key]

        corpus = self._normalize_text_list(self.style_corpus.get(style_key, []))
        if not corpus:
            return None

        vectorizer = TfidfVectorizer()
        try:
            docs_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return None

        bundle = (vectorizer, docs_matrix, corpus)
        self._rag_index_cache[style_key] = bundle
        return bundle

    def get_rag_references(self, source_text: str, style_key: str) -> List[str]:
        source = (source_text or "").strip()
        index_bundle = self._build_rag_index(style_key)
        if index_bundle is None:
            return []

        vectorizer, docs_matrix, corpus = index_bundle

        top_k = int(self.config.get("rag", {}).get("top_k", 3))
        top_k = max(1, min(top_k, len(corpus)))

        if not source:
            return corpus[:top_k]

        try:
            query = vectorizer.transform([source])
        except ValueError:
            return corpus[:top_k]

        sims = cosine_similarity(query, docs_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [corpus[idx] for idx in top_indices.tolist()]


class RealMainPipelineAdapter:
    def __init__(self, project_root: Path, repo: DataRepository) -> None:
        self.project_root = project_root
        self.repo = repo
        self._pipeline = None

    @staticmethod
    def _parse_references(style_references_text: str) -> List[str]:
        text = (style_references_text or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split("\n\n") if part.strip()]
        if parts:
            return parts
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def _bind_prompt_to_style(system_prompt: str, target_style: str, scenario: str) -> str:
        prompt = (system_prompt or "").strip()
        if "{target_style}" in prompt or "{scenario}" in prompt:
            return prompt.format(target_style=target_style, scenario=scenario)
        return prompt

    @staticmethod
    def _build_generation_prompt(
        source_text: str,
        bound_system_prompt: str,
        style_references: List[str],
    ) -> str:
        refs_text = "\n\n".join([f"参考片段{i + 1}: {item}" for i, item in enumerate(style_references)])
        reference_block = (
            f"以下是目标风格参考文本，请提炼其写作特征并迁移到改写中：\n{refs_text}\n\n"
            if refs_text
            else ""
        )
        return f"{bound_system_prompt}\n\n{reference_block}原文本：\n{source_text}\n"

    def _get_pipeline(self):
        if self._pipeline is None:
            from main_pipeline import MainPipeline

            self._pipeline = MainPipeline(self.project_root)
        return self._pipeline

    def run(
        self,
        source_text: str,
        target_style_key: str,
        target_style_label: str,
        scenario: str,
        method: str,
        system_prompt: str,
        style_references_text: str,
    ) -> PipelineResult:
        pipeline = self._get_pipeline()

        refs_for_generation = [] if method == "Prompt_Eng" else self._parse_references(style_references_text)
        bound_system_prompt = self._bind_prompt_to_style(
            system_prompt=system_prompt,
            target_style=target_style_label,
            scenario=scenario,
        )
        final_prompt = self._build_generation_prompt(
            source_text=source_text,
            bound_system_prompt=bound_system_prompt,
            style_references=refs_for_generation,
        )

        generated_text = pipeline.api_client.chat_completion(
            model=pipeline.config["models"]["generation_model"],
            prompt=final_prompt,
            temperature=pipeline.config["generation"]["temperature"],
            max_tokens=pipeline.config["generation"]["max_tokens"],
        )

        if method == "Prompt_Eng":
            eval_refs: List[str] = []
        else:
            eval_refs = refs_for_generation

        semantic = pipeline.semantic_metric.evaluate(source_text, generated_text, refs_for_generation)
        primary_judge = pipeline._evaluate_single_judge(generated_text, eval_refs)
        style_vector = pipeline.style_vector_metric.evaluate(source_text, generated_text, eval_refs)
        linguistic = pipeline.linguistic_metric.evaluate(
            source_text,
            generated_text,
            eval_refs,
            target_style_name=target_style_key,
            target_scenario=scenario,
        )
        fluency = pipeline.fluency_metric.evaluate(source_text, generated_text, eval_refs)

        metrics = {
            "bert_score": float(semantic.get("score", 0.0)),
            "llm_judge_score": float(primary_judge.get("score", 0.0)),
            "style_vector_score": float(style_vector.get("score", 0.0)),
            "linguistic_stats": linguistic.get("details", {}),
            "fluency_score": float(fluency.get("score", 0.0)),
            "details": {
                "semantic": semantic.get("details", {}),
                "llm_judge": primary_judge.get("details", {}),
                "style_vector": style_vector.get("details", {}),
                "fluency": fluency.get("details", {}),
            },
        }

        return PipelineResult(
            generated_text=generated_text,
            metrics=metrics,
        )


class ExperimentRecordStore:
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path

    def _read_records(self) -> List[Dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _next_version(self, case_id: str, target_style: str, scenario: str, method: str) -> int:
        records = self._read_records()
        max_version = 0
        for row in records:
            if (
                row.get("case_id") == case_id
                and row.get("target_style") == target_style
                and row.get("scenario") == scenario
                and row.get("method") == method
            ):
                try:
                    max_version = max(max_version, int(row.get("version", 0)))
                except ValueError:
                    pass
        return max_version + 1

    def save(
        self,
        case_id: str,
        target_style: str,
        scenario: str,
        method: str,
        source_text: str,
        system_prompt: str,
        style_references: List[str],
        generated_text: str,
        metrics: Dict[str, Any],
    ) -> int:
        version = self._next_version(
            case_id=case_id,
            target_style=target_style,
            scenario=scenario,
            method=method,
        )

        bert_score = float(metrics.get("bert_score", 0.0))
        style_vector_score = float(metrics.get("style_vector_score", 0.0))
        fluency_score = float(metrics.get("fluency_score", 0.0))

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "case_id": case_id,
            "target_style": target_style,
            "scenario": scenario,
            "method": method,
            "version": version,
            "source_text": source_text,
            "system_prompt": system_prompt,
            "style_references": style_references,
            "generated_text": generated_text,
            "bert_score": bert_score,
            "style_vector_score": style_vector_score,
            "fluency_score": fluency_score,
            "llm_judge_score": float(metrics.get("llm_judge_score", 0.0)),
            "linguistic_stats": metrics.get("linguistic_stats", {}),
            "details": metrics.get("details", {}),
            "metrics": metrics,
        }

        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return version


class StyleTransferWebUI:
    STYLE_OPTIONS: Dict[str, str] = {}

    DEFAULT_SYSTEM_PROMPT = (
        "你是一名叙事风格迁移助手。请将给定事实文本重写为“{target_style}”风格。\n"
        "要求：\n"
        "1) 保持事实内容与关键信息不变，避免幻觉。\n"
        "2) 在词汇偏好、句式结构、情绪表达上体现目标风格。\n"
        "3) 仅输出改写结果，不要解释。"
    )

    BEST_DEMO_PROMPT = (
        "你是一名资深中文改写助手。请在保留事实的前提下，输出更自然、更连贯、"
        "更有文学叙事感的改写文本，只输出改写结果。"
    )

    METHOD_OPTIONS = ["Prompt_Eng", "Context_Eng", "RAG_ICL"]
    SCENARIO_OPTIONS = ["叙事", "抒情", "讽刺", "议论"]

    DEMO_ROUTE_MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {
        "鲁迅风": {
            "叙事": {"method": "Context_Eng"},
            "抒情": {"method": "RAG_ICL"},
            "讽刺": {"method": "RAG_ICL"},
            "议论": {"method": "Context_Eng"},
        },
        "钱钟书风": {
            "叙事": {"method": "Context_Eng"},
            "抒情": {"method": "Prompt_Eng"},
            "讽刺": {"method": "RAG_ICL"},
            "议论": {"method": "RAG_ICL"},
        },
        "知名传记风": {
            "叙事": {"method": "Context_Eng"},
            "抒情": {"method": "Prompt_Eng"},
            "讽刺": {"method": "Prompt_Eng"},
            "议论": {"method": "Context_Eng"},
        },
    }

    PREVIEW_BOX_CSS = """
    .route-preview textarea {
        background: rgba(128, 128, 128, 0.12) !important;
        border: 1px dashed rgba(120, 120, 120, 0.8) !important;
        color: rgba(20, 20, 20, 0.88) !important;
    }
    .route-preview label {
        opacity: 0.9;
    }
    """

    def __init__(self) -> None:
        self.repo = DataRepository()
        self.store = ExperimentRecordStore(EXPERIMENT_JSONL_PATH)
        self.pipeline_adapter = RealMainPipelineAdapter(PROJECT_ROOT, self.repo)
        context_styles = list(self.repo.context_eng_style_references.keys())
        if context_styles:
            self.style_labels = context_styles
        elif self.repo.target_styles:
            self.style_labels = list(self.repo.target_styles.keys())
        else:
            self.style_labels = ["luxun"]
        self.style_options = {label: label for label in self.style_labels}

    def _style_label_to_key(self, style_label: str) -> str:
        return self.style_options.get(style_label, style_label)

    @staticmethod
    def _join_references(refs: Any) -> str:
        if isinstance(refs, str):
            return refs
        if isinstance(refs, list):
            return "\n\n".join([item for item in refs if isinstance(item, str)])
        if isinstance(refs, dict):
            merged: List[str] = []
            for value in refs.values():
                if isinstance(value, list):
                    merged.extend([item for item in value if isinstance(item, str)])
                elif isinstance(value, str):
                    merged.append(value)
            return "\n\n".join(merged)
        return ""

    @staticmethod
    def _parse_references_text(style_references_text: str) -> List[str]:
        text = (style_references_text or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split("\n\n") if part.strip()]
        if parts:
            return parts
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _get_style_refs_update(self, method: str, style_label: str, scenario: str, source_text: str) -> gr.update:
        style_key = self._style_label_to_key(style_label)
        try:
            if method == "Prompt_Eng":
                return gr.update(visible=False, value=""), False
            if method == "Context_Eng":
                refs = self.repo.get_context_references_by_scenario(style_key, scenario)
                return gr.update(visible=True, value=self._join_references(refs)), False
            refs = self.repo.get_rag_references(source_text=source_text, style_key=style_key)
            return gr.update(visible=True, value=self._join_references(refs)), False
        except Exception:
            return gr.update(visible=True, value=""), False

    def _on_case_change(self, case_id: str, method: str, style_label: str, scenario: str) -> Tuple[str, gr.update, bool]:
        source_text = self.repo.get_source_text_by_case_id(case_id)
        refs_update = self._get_style_refs_update(
            method=method,
            style_label=style_label,
            scenario=scenario,
            source_text=source_text,
        )
        # _get_style_refs_update now returns (gr.update, loading_flag)
        if isinstance(refs_update, tuple):
            refs, loading = refs_update
        else:
            refs, loading = refs_update, False
        return source_text, refs, loading

    def _on_method_or_style_change(self, method: str, style_label: str, scenario: str, source_text: str) -> Tuple[gr.update, bool]:
        refs_update = self._get_style_refs_update(
            method=method,
            style_label=style_label,
            scenario=scenario,
            source_text=source_text,
        )
        if isinstance(refs_update, tuple):
            return refs_update
        return refs_update, False

    def _on_style_change_update_prompt_and_refs(
        self,
        method: str,
        style_label: str,
        scenario: str,
        source_text: str,
    ) -> Tuple[str, gr.update, bool]:
        # 关键：system_prompt 与 target_style 强绑定
        prompt = self.DEFAULT_SYSTEM_PROMPT.format(target_style=style_label, scenario=scenario)
        refs_update = self._get_style_refs_update(
            method=method,
            style_label=style_label,
            scenario=scenario,
            source_text=source_text,
        )
        if isinstance(refs_update, tuple):
            refs, loading = refs_update
        else:
            refs, loading = refs_update, False
        return prompt, refs, loading

    def _on_scenario_change_update_prompt_and_refs(
        self,
        method: str,
        target_style: str,
        scenario: str,
        source_text: str,
    ) -> Tuple[str, gr.update, bool]:
        prompt = self.DEFAULT_SYSTEM_PROMPT.format(target_style=target_style, scenario=scenario)
        refs_update = self._get_style_refs_update(
            method=method,
            style_label=target_style,
            scenario=scenario,
            source_text=source_text,
        )
        if isinstance(refs_update, tuple):
            refs, loading = refs_update
        else:
            refs, loading = refs_update, False
        return prompt, refs, loading

    def _scenario_few_shot_templates(self, target_style: str, scenario: str) -> List[str]:
        style_key = self._style_label_to_key(target_style)
        return self.repo.get_context_references_by_scenario(style_key, scenario)

    def _resolve_demo_route(self, target_style: str, scenario: str, source_text: str) -> Dict[str, Any]:
        style_routes = self.DEMO_ROUTE_MATRIX.get(target_style, {})
        route = style_routes.get(scenario)
        if route is None:
            route = {"method": "Prompt_Eng"}

        method = route.get("method", "Prompt_Eng")
        demo_prompt = (
            "你是一名长文本风格迁移助手。请将给定事实文本改写为“{target_style}-{scenario}”风格，"
            "保持事实不变、语言自然、有可读性，只输出改写结果。"
        )

        few_shot_list = []
        if method == "Context_Eng":
            # Context_Eng：使用写死的最优模板
            few_shot_list = self._scenario_few_shot_templates(target_style=target_style, scenario=scenario)
            if not few_shot_list:
                few_shot_list = self.repo.get_context_references(self._style_label_to_key(target_style))
        elif method == "RAG_ICL":
            # RAG_ICL：根据输入文本实时进行知识库匹配更新
            few_shot_list = self.repo.get_rag_references(source_text, self._style_label_to_key(target_style))

        reason = f"经过迭代测试，{target_style}的{scenario}场景使用{method}策略效果最好。"

        return {
            "method": method,
            "system_prompt": demo_prompt,
            "few_shot_list": few_shot_list,
            "reason": reason,
        }

    def _run_experiment(
        self,
        case_id: str,
        source_text: str,
        target_style: str,
        scenario: str,
        method: str,
        system_prompt: str,
        style_references: str,
    ) -> Tuple[str, Dict[str, Any], str, str]:
        style_key = self._style_label_to_key(target_style)
        effective_refs = "" if method == "Prompt_Eng" else (style_references or "")

        final_system_prompt = (system_prompt or self.DEFAULT_SYSTEM_PROMPT).strip()
        try:
            result = self.pipeline_adapter.run(
                source_text=source_text,
                target_style_key=style_key,
                target_style_label=target_style,
                scenario=scenario,
                method=method,
                system_prompt=final_system_prompt,
                style_references_text=effective_refs,
            )
        except Exception as exc:
            gr.Warning(f"运行失败：{exc}")
            return "", {"error": str(exc)}, final_system_prompt, effective_refs

        metrics = dict(result.metrics)
        metrics["target_style"] = style_key
        metrics["scenario"] = scenario
        metrics["method"] = method
        metrics["case_id"] = case_id
        return result.generated_text, metrics, final_system_prompt, effective_refs

    def _save_record(
        self,
        case_id: str,
        source_text: str,
        target_style: str,
        scenario: str,
        method: str,
        system_prompt: str,
        style_references: str,
        runtime_system_prompt: str,
        runtime_style_references: str,
        generated_text: str,
        metrics: Dict[str, Any],
    ) -> str:
        if not generated_text.strip():
            gr.Warning("请先点击“运行生成与评估”，再执行保存。")
            return "保存失败：请先运行。"

        if not isinstance(metrics, dict):
            gr.Warning("指标结果为空或格式异常，请重新运行。")
            return "保存失败：metrics 异常。"

        if "error" in metrics:
            gr.Warning("当前结果包含错误信息，请先修复后再保存。")
            return "保存失败：运行报错。"

        effective_system_prompt = (runtime_system_prompt or system_prompt or self.DEFAULT_SYSTEM_PROMPT).strip()
        effective_refs = "" if method == "Prompt_Eng" else (runtime_style_references or style_references or "")
        refs_list = [] if method == "Prompt_Eng" else self._parse_references_text(effective_refs)
        version = self.store.save(
            case_id=case_id,
            target_style=self._style_label_to_key(target_style),
            scenario=scenario,
            method=method,
            source_text=source_text,
            system_prompt=effective_system_prompt,
            style_references=refs_list,
            generated_text=generated_text,
            metrics=metrics,
        )
        gr.Info(f"保存成功，当前 Version = {version}")
        return f"保存成功：case_id={case_id}, target_style={target_style}, scenario={scenario}, method={method}, version={version}"

    def _demo_generate(self, source_text: str, target_style: str, scenario: str) -> str:
        style_key = self._style_label_to_key(target_style)
        route = self._resolve_demo_route(target_style=target_style, scenario=scenario, source_text=source_text)
        few_shot_text = self._join_references(route["few_shot_list"])
        try:
            result = self.pipeline_adapter.run(
                source_text=source_text,
                target_style_key=style_key,
                target_style_label=target_style,
                scenario=scenario,
                method=route["method"],
                system_prompt=route["system_prompt"],
                style_references_text=few_shot_text,
            )
            return result.generated_text
        except Exception as exc:
            gr.Warning(f"生成失败：{exc}")
            return ""

    def _preview_demo_route(self, source_text: str, target_style: str, scenario: str) -> Tuple[str, str, str]:
        route = self._resolve_demo_route(target_style=target_style, scenario=scenario, source_text=source_text)
        method = route.get("method", "Prompt_Eng")
        few_shot_list = route.get("few_shot_list", [])
        few_shot_text = self._join_references(few_shot_list)
        if not few_shot_text.strip():
            few_shot_text = "（无，当前自动策略不使用 few-shot）"
        reason = route.get("reason", "")
        return method, few_shot_text, reason

    def build(self) -> gr.Blocks:
        case_id_choices = self.repo.mock_fetch_case_ids()
        default_case_id = case_id_choices[0] if case_id_choices else "case_001"
        default_source = self.repo.get_source_text_by_case_id(default_case_id)
        style_labels = list(self.style_labels)
        default_style = style_labels[0]
        default_scenario = self.SCENARIO_OPTIONS[0]
        default_method = self.METHOD_OPTIONS[0]

        with gr.Blocks() as demo:
            gr.Markdown("# 长文本风格迁移 Web UI（实验版）")

            with gr.Tab("实验调试区"):
                with gr.Row():
                    with gr.Column(scale=5):
                        case_id = gr.Dropdown(
                            label="case_id",
                            choices=case_id_choices,
                            value=default_case_id,
                            interactive=True,
                        )
                        source_text = gr.Textbox(
                            label="source_text",
                            lines=8,
                            value=default_source,
                            interactive=True,
                        )
                        target_style = gr.Dropdown(
                            label="target_style",
                            choices=style_labels,
                            value=default_style,
                            interactive=True,
                        )
                        scenario = gr.Dropdown(
                            label="scenario",
                            choices=self.SCENARIO_OPTIONS,
                            value=default_scenario,
                            visible=False,
                            interactive=True,
                        )
                        method = gr.Radio(
                            label="method",
                            choices=self.METHOD_OPTIONS,
                            value=default_method,
                            interactive=True,
                        )
                        system_prompt = gr.Textbox(
                            label="system_prompt",
                            lines=8,
                            value=self.DEFAULT_SYSTEM_PROMPT.format(target_style=default_style, scenario=default_scenario),
                            interactive=True,
                        )
                        style_references = gr.Textbox(
                            label="style_references",
                            lines=8,
                            value="",
                            visible=False,
                            interactive=True,
                        )
                        run_btn = gr.Button("🚀 运行生成与评估", variant="primary")

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=6):
                        generated_text = gr.Textbox(label="generated_text", lines=12, interactive=False)
                    with gr.Column(scale=4):
                        metrics_json = gr.JSON(label="metrics（对齐 case_summary）")
                        save_btn = gr.Button("保存当前策略与结果落盘")
                        save_status = gr.Textbox(label="保存状态", interactive=False)
                last_runtime_system_prompt = gr.State(value="")
                last_runtime_style_references = gr.State(value="")
                style_refs_loading = gr.State(value=False)

                # 关键事件绑定1：切换 case_id 时，自动联动 source_text 与 style_references
                case_id.change(
                    fn=self._on_case_change,
                    inputs=[case_id, method, target_style, scenario],
                    outputs=[source_text, style_references, style_refs_loading],
                    queue=False,
                )

                # 关键事件绑定2：method 变化时，动态控制 style_references 显示与自动填充
                method.change(
                    fn=self._on_method_or_style_change,
                    inputs=[method, target_style, scenario, source_text],
                    outputs=[style_references, style_refs_loading],
                    queue=False,
                )

                # 关键事件绑定3：target_style 变化时，联动刷新 system_prompt 与 style_references
                target_style.change(
                    fn=self._on_style_change_update_prompt_and_refs,
                    inputs=[method, target_style, scenario, source_text],
                    outputs=[system_prompt, style_references, style_refs_loading],
                    queue=False,
                )

                # 关键事件绑定4：scenario 变化时，刷新 system_prompt
                scenario.change(
                    fn=self._on_scenario_change_update_prompt_and_refs,
                    inputs=[method, target_style, scenario, source_text],
                    outputs=[system_prompt, style_references, style_refs_loading],
                    queue=False,
                )

                run_btn.click(
                    fn=self._run_experiment,
                    inputs=[case_id, source_text, target_style, scenario, method, system_prompt, style_references],
                    outputs=[generated_text, metrics_json, last_runtime_system_prompt, last_runtime_style_references],
                )

                save_btn.click(
                    fn=self._save_record,
                    inputs=[
                        case_id,
                        source_text,
                        target_style,
                        scenario,
                        method,
                        system_prompt,
                        style_references,
                        last_runtime_system_prompt,
                        last_runtime_style_references,
                        generated_text,
                        metrics_json,
                    ],
                    outputs=[save_status],
                )

            with gr.Tab("演示区"):
                demo_source = gr.Textbox(label="source_text", lines=10)
                demo_style = gr.Dropdown(label="target_style", choices=style_labels, value=default_style)
                demo_scenario = gr.Dropdown(label="scenario", choices=self.SCENARIO_OPTIONS, value=default_scenario, visible=False)
                demo_auto_method = gr.Textbox(
                    label="自动选择策略（路由结果）",
                    interactive=False,
                    elem_classes=["route-preview"],
                )
                demo_auto_few_shot = gr.Textbox(
                    label="自动选择 few-shot 模板（路由结果）",
                    lines=6,
                    interactive=False,
                    elem_classes=["route-preview"],
                )
                demo_auto_reason = gr.Textbox(
                    label="自动选择原因（路由说明）",
                    lines=2,
                    interactive=False,
                    elem_classes=["route-preview"],
                )
                demo_btn = gr.Button("一键生成", variant="primary")
                demo_output = gr.Textbox(label="generated_text", lines=12, interactive=False)

                demo_source.change(
                    fn=self._preview_demo_route,
                    inputs=[demo_source, demo_style, demo_scenario],
                    outputs=[demo_auto_method, demo_auto_few_shot, demo_auto_reason],
                    queue=False,
                )
                demo_style.change(
                    fn=self._preview_demo_route,
                    inputs=[demo_source, demo_style, demo_scenario],
                    outputs=[demo_auto_method, demo_auto_few_shot, demo_auto_reason],
                    queue=False,
                )
                demo_scenario.change(
                    fn=self._preview_demo_route,
                    inputs=[demo_source, demo_style, demo_scenario],
                    outputs=[demo_auto_method, demo_auto_few_shot, demo_auto_reason],
                    queue=False,
                )

                demo.load(
                    fn=self._preview_demo_route,
                    inputs=[demo_source, demo_style, demo_scenario],
                    outputs=[demo_auto_method, demo_auto_few_shot, demo_auto_reason],
                    queue=False,
                )

                demo_btn.click(
                    fn=self._demo_generate,
                    inputs=[demo_source, demo_style, demo_scenario],
                    outputs=[demo_output],
                )

        return demo


def main() -> None:
    ui = StyleTransferWebUI()
    app = ui.build()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        auth=("capstone", "cuhksz"),
        theme=gr.themes.Soft(),
        css=StyleTransferWebUI.PREVIEW_BOX_CSS,
    )


if __name__ == "__main__":
    main()
