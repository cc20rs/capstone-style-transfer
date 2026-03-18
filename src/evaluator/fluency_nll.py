from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluator.metrics_base import BaseMetric
from src.utils.model_hub import resolve_model_name_or_path


class FluencyNLLMetric(BaseMetric):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        model_name = resolve_model_name_or_path(self.config, "fluency_model")
        method_cfg = self.config.get("evaluation", {}).get("methods", {})
        self.allow_fallback = bool(method_cfg.get("allow_fallback", False))
        self._load_error = ""
        self._available = True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
        except Exception as exc:
            self._available = False
            self._load_error = str(exc)
            self.tokenizer = None
            self.model = None

    def evaluate(
        self,
        source_text: str,
        generated_text: str,
        style_references: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self._available:
            if not self.allow_fallback:
                raise RuntimeError(
                    "Fluency model is not available locally under fixed-method mode. "
                    "Run scripts/download_modelscope_models.py first or enable huggingface mirror."
                )
            text_len = max(len(generated_text.strip()), 1)
            proxy_score = min(100.0, 30.0 + text_len / 6.0)
            return {
                "score": float(proxy_score),
                "details": {
                    "fallback": "length_proxy",
                    "reason": "fluency_model_unavailable",
                    "error": self._load_error,
                },
            }

        inputs = self.tokenizer(generated_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            nll = float(outputs.loss.item())
        score = 100.0 / max(nll, 1e-6)
        return {"score": score, "details": {"nll": nll}}
