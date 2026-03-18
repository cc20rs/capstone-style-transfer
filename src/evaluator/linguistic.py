from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import jieba
import jieba.posseg as pseg

from src.evaluator.metrics_base import BaseMetric
from src.utils.file_io import load_json


class LinguisticFeatureMetric(BaseMetric):
    def __init__(self, config: Dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"[。！？…]+", text)
        return [item.strip() for item in parts if item.strip()]

    def _compute_features(self, text: str) -> Dict[str, float]:
        sentences = self._split_sentences(text)
        sentence_count = max(len(sentences), 1)
        avg_len = len(text) / sentence_count

        tokens = [tok for tok in jieba.lcut(text) if tok.strip()]
        total_tokens = max(len(tokens), 1)
        ttr = len(set(tokens)) / total_tokens

        tagged = list(pseg.cut(text))
        adj_count = sum(1 for item in tagged if item.flag.startswith("a"))
        adv_count = sum(1 for item in tagged if item.flag.startswith("d"))
        adj_density = adj_count / total_tokens
        adv_density = adv_count / total_tokens

        return {
            "avg_len": float(avg_len),
            "ttr": float(ttr),
            "adj_density": float(adj_density),
            "adv_density": float(adv_density),
        }

    def _resolve_target_refs(
        self,
        target_style_name: str,
        style_references: Optional[List[str]],
    ) -> List[str]:
        refs = style_references or []
        if refs:
            return refs
        target_styles_path = self.project_root / self.config["paths"]["target_styles"]
        target_styles = load_json(target_styles_path)
        values = target_styles.get(target_style_name, [])
        return values if isinstance(values, list) else []

    def evaluate(
        self,
        source_text: str,
        generated_text: str,
        style_references: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target_style_name = kwargs.get("target_style_name", "")
        target_refs = self._resolve_target_refs(target_style_name, style_references)
        target_text = "\n".join(target_refs) if target_refs else ""

        source_feat = self._compute_features(source_text)
        generated_feat = self._compute_features(generated_text)
        target_feat = self._compute_features(target_text) if target_text else {
            "avg_len": 0.0,
            "ttr": 0.0,
            "adj_density": 0.0,
            "adv_density": 0.0,
        }

        details = {
            "avg_len": {
                "source": source_feat["avg_len"],
                "target": target_feat["avg_len"],
                "generated": generated_feat["avg_len"],
            },
            "ttr": {
                "source": source_feat["ttr"],
                "target": target_feat["ttr"],
                "generated": generated_feat["ttr"],
            },
            "adj_density": {
                "source": source_feat["adj_density"],
                "target": target_feat["adj_density"],
                "generated": generated_feat["adj_density"],
            },
            "adv_density": {
                "source": source_feat["adv_density"],
                "target": target_feat["adv_density"],
                "generated": generated_feat["adv_density"],
            },
        }
        return {"score": 0.0, "details": details}
