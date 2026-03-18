from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.generator.base_generator import BaseGenerator, StyleTransferGenerator


class RAGICLGenerator(BaseGenerator):
    strategy_name = "Baseline_C_RAG_ICL"

    def __init__(self, config: Dict[str, Any], api_client) -> None:
        super().__init__(config, api_client)
        self.generator = StyleTransferGenerator(config, api_client)

    def retrieve_style_references(
        self,
        source_text: str,
        style_corpus: List[str],
        top_k: Optional[int] = None,
    ) -> List[str]:
        if not style_corpus:
            return []
        k = top_k or self.config["rag"]["top_k"]
        corpus = [source_text] + style_corpus
        matrix = TfidfVectorizer().fit_transform(corpus)
        query_vec = matrix[0:1]
        corpus_vecs = matrix[1:]
        sims = cosine_similarity(query_vec, corpus_vecs).flatten()
        top_indices = np.argsort(sims)[::-1][:k]
        return [style_corpus[idx] for idx in top_indices.tolist()]

    def generate(
        self,
        source_text: str,
        target_style_name: str,
        style_references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        dynamic_refs = self.retrieve_style_references(
            source_text=source_text,
            style_corpus=style_references or [],
            top_k=self.config["rag"]["top_k"],
        )
        return self.generator.generate(
            source_text=source_text,
            target_style_name=target_style_name,
            style_references=dynamic_refs,
        )
