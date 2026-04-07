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

    def _get_effective_max_length(self) -> int:
        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        model_max = getattr(getattr(self.model, "config", None), "n_positions", None)

        candidates: List[int] = []
        for value in (tokenizer_max, model_max):
            if isinstance(value, int) and value > 0 and value < 1_000_000:
                candidates.append(value)

        if candidates:
            return int(min(candidates))
        return 1024

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

        effective_max_length = self._get_effective_max_length()
        raw_inputs = self.tokenizer(generated_text, return_tensors="pt", verbose=False)
        raw_token_len = int(raw_inputs["input_ids"].shape[1])

        inputs = self.tokenizer(
            generated_text,
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_length,
            verbose=False,
        )
        used_token_len = int(inputs["input_ids"].shape[1])
        was_truncated = raw_token_len > used_token_len

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            nll = float(outputs.loss.item())
        score = 100.0 / max(nll, 1e-6)
        return {
            "score": score,
            "details": {
                "nll": nll,
                "token_len": used_token_len,
                "max_length": effective_max_length,
                "truncated": was_truncated,
                "raw_token_len": raw_token_len,
            },
        }
