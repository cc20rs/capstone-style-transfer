from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

from src.evaluator.fluency_nll import FluencyNLLMetric
from src.evaluator.linguistic import LinguisticFeatureMetric
from src.evaluator.semantic import SemanticBERTScoreMetric
from src.evaluator.style_llm import LLMStyleJudgeMetric
from src.evaluator.style_vector import StyleVectorDistanceMetric
from src.generator.context_eng import ContextEngineeringGenerator
from src.generator.prompt_eng import PromptEngineeringGenerator
from src.generator.rag_icl import RAGICLGenerator
from src.utils.api_client import APIClient
from src.utils.file_io import load_json, load_yaml, save_intermediate_result
from src.utils.model_hub import configure_huggingface_mirror


class MainPipeline:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        load_dotenv(self.project_root / ".env")
        self.config = load_yaml(self.project_root / "configs" / "config.yaml")
        configure_huggingface_mirror(self.config)
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY in environment variables")
        self.api_client = APIClient(self.config, api_key)

        self.prompt_generator = PromptEngineeringGenerator(self.config, self.api_client)
        self.context_generator = ContextEngineeringGenerator(self.config, self.api_client)
        self.rag_generator = RAGICLGenerator(self.config, self.api_client)

        self.semantic_metric = SemanticBERTScoreMetric(self.config)
        self.style_vector_metric = StyleVectorDistanceMetric(self.config)
        self.linguistic_metric = LinguisticFeatureMetric(self.config, self.project_root)
        self.fluency_metric = FluencyNLLMetric(self.config)
        self.judge_models = self.config.get("judge_models", [self.config["models"]["judge_model"]])

    def _paths(self) -> Dict[str, Path]:
        p = self.config["paths"]
        return {
            "raw_cases": self.project_root / p["raw_cases"],
            "target_styles": self.project_root / p["target_styles"],
            "style_corpus": self.project_root / p["style_corpus"],
            "case_summary": self.project_root / p["case_summary"],
        }

    def _evaluate_judge_models(self, generated_text: str, target_refs_for_eval: List[str]) -> Dict[str, Any]:
        by_model: Dict[str, Any] = {}
        for model_name in self.judge_models:
            judge_config = dict(self.config)
            judge_config["models"] = dict(self.config["models"])
            judge_config["models"]["judge_model"] = model_name
            judge_metric = LLMStyleJudgeMetric(judge_config, self.api_client)
            result = judge_metric.evaluate("", generated_text, target_refs_for_eval)
            by_model[model_name] = {
                "score": result.get("score", 0.0),
                "details": result.get("details", {}),
            }
        return by_model

    def _build_metrics(
        self,
        source_text: str,
        generated_text: str,
        style_name: str,
        style_references: List[str],
        target_refs_for_eval: List[str],
    ) -> Dict[str, Any]:
        semantic = self.semantic_metric.evaluate(source_text, generated_text, style_references)
        judge_by_model = self._evaluate_judge_models(generated_text, target_refs_for_eval)
        primary_judge_model = self.judge_models[0]
        primary_judge = judge_by_model.get(primary_judge_model, {"score": 0.0, "details": {}})
        style_vector = self.style_vector_metric.evaluate(source_text, generated_text, target_refs_for_eval)
        linguistic = self.linguistic_metric.evaluate(
            source_text,
            generated_text,
            style_references,
            target_style_name=style_name,
        )
        fluency = self.fluency_metric.evaluate(source_text, generated_text, style_references)

        return {
            "bert_score": semantic["score"],
            "llm_judge_score": primary_judge["score"],
            "style_vector_score": style_vector["score"],
            "linguistic_stats": linguistic["details"],
            "fluency_score": fluency["score"],
            "details": {
                "semantic": semantic["details"],
                "llm_judge": primary_judge["details"],
                "llm_judge_by_model": judge_by_model,
                "style_vector": style_vector["details"],
                "fluency": fluency["details"],
            },
        }

    def run(self) -> None:
        paths = self._paths()
        test_cases = load_json(paths["raw_cases"]).get("cases", [])
        target_styles = load_json(paths["target_styles"])
        style_corpus = load_json(paths["style_corpus"])

        strategies = [
            ("Baseline_A_Prompt_Eng", self.prompt_generator),
            ("Baseline_B_Context_Eng", self.context_generator),
            ("Baseline_C_RAG_ICL", self.rag_generator),
        ]

        for case in test_cases:
            case_id = case["case_id"]
            source_text = case["source_text"]
            style_name = case["style_name"]
            static_refs = target_styles.get(style_name, [])[:3]
            rag_corpus = style_corpus.get(style_name, [])

            for strategy_name, generator in strategies:
                if strategy_name == "Baseline_A_Prompt_Eng":
                    refs_for_generation: List[str] = []
                elif strategy_name == "Baseline_B_Context_Eng":
                    refs_for_generation = static_refs
                else:
                    refs_for_generation = rag_corpus

                generated = generator.generate(
                    source_text=source_text,
                    target_style_name=style_name,
                    style_references=refs_for_generation,
                )

                if strategy_name == "Baseline_C_RAG_ICL":
                    refs_logged = generated["style_references"]
                else:
                    refs_logged = refs_for_generation

                target_refs_for_eval = refs_logged if refs_logged else static_refs
                metrics = self._build_metrics(
                    source_text=source_text,
                    generated_text=generated["generated_text"],
                    style_name=style_name,
                    style_references=refs_logged,
                    target_refs_for_eval=target_refs_for_eval,
                )

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "case_id": case_id,
                    "strategy": strategy_name,
                    "style_name": style_name,
                    "source_text": source_text,
                    "style_references": refs_logged,
                    "generated_text": generated["generated_text"],
                    "metrics": metrics,
                }
                save_intermediate_result(paths["case_summary"], row)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    pipeline = MainPipeline(root)
    pipeline.run()
