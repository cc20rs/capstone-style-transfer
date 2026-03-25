from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from bert_score import score
from transformers import AutoConfig

from src.evaluator.metrics_base import BaseMetric
from src.utils.model_hub import resolve_model_name_or_path


class SemanticBERTScoreMetric(BaseMetric):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_type = resolve_model_name_or_path(self.config, "bert_score_model")
        self.num_layers: Optional[int] = None
        model_path = Path(self.model_type)
        if model_path.exists() and model_path.is_dir():
            try:
                model_cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
                layers = getattr(model_cfg, "num_hidden_layers", None)
                if isinstance(layers, int) and layers > 0:
                    self.num_layers = layers
            except Exception:
                self.num_layers = None
        method_cfg = self.config.get("evaluation", {}).get("methods", {})
        self.allow_fallback = bool(method_cfg.get("allow_fallback", False))

    def evaluate(
        self,
        source_text: str,
        generated_text: str,
        style_references: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            _, _, f1 = score(
                [generated_text],
                [source_text],
                model_type=self.model_type,
                num_layers=self.num_layers,
                lang="zh",
                verbose=False,
            )
            value = float(f1.mean().item())
            details = {"metric": "BERTScore-F1", "lang": "zh", "model_type": self.model_type}
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "BERTScore evaluation failed under fixed-method mode. "
                    f"model_type={self.model_type}; error={exc}"
                ) from exc
            src_tokens = set(source_text)
            gen_tokens = set(generated_text)
            overlap = len(src_tokens & gen_tokens)
            denom = max(len(src_tokens | gen_tokens), 1)
            value = float(overlap / denom)
            details = {
                "metric": "char_jaccard_fallback",
                "lang": "zh",
                "model_type": self.model_type,
                "reason": "bert_score_unavailable",
                "error": str(exc),
            }
        return {
            "score": value,
            "details": details,
        }
