from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluator.metrics_base import BaseMetric
from src.utils.model_hub import resolve_model_name_or_path


class StyleVectorDistanceMetric(BaseMetric):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        model_name = resolve_model_name_or_path(self.config, "style_embedding_model")
        self.model = SentenceTransformer(model_name)

    def evaluate(
        self,
        source_text: str,
        generated_text: str,
        style_references: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        refs = style_references or []
        if not refs:
            return {"score": 0.0, "details": {"reason": "style_references is empty"}}
        ref_vectors = self.model.encode(refs)
        ref_centroid = np.mean(ref_vectors, axis=0, keepdims=True)
        gen_vector = self.model.encode([generated_text])
        similarity = float(cosine_similarity(ref_centroid, gen_vector)[0][0])
        return {"score": similarity, "details": {"metric": "cosine_similarity"}}
