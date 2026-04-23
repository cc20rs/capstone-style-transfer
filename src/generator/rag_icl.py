from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generator.base_generator import BaseGenerator, StyleTransferGenerator
from src.generator.scene_recall import CascadeSceneRetriever
from src.utils.file_io import load_json


class RAGICLGenerator(BaseGenerator):
    strategy_name = "Baseline_C_RAG_ICL"

    def __init__(self, config: Dict[str, Any], api_client) -> None:
        super().__init__(config, api_client)
        self.generator = StyleTransferGenerator(config, api_client)
        self.project_root = Path(__file__).resolve().parents[2]

        corpus_path = self.project_root / "data" / "source_text" / "corpus_text.json"
        corpus_rows = load_json(corpus_path).get("corpus", []) if corpus_path.exists() else []
        self.scene_retriever: Optional[CascadeSceneRetriever] = None
        if isinstance(corpus_rows, list) and corpus_rows:
            self.scene_retriever = CascadeSceneRetriever(config, corpus_rows)

    def retrieve_style_references(
        self,
        source_text: str,
        style_corpus: List[str],
        target_style_name: Optional[str] = None,
        stage: Optional[str] = None,
        location: Optional[str] = None,
        event: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        k = top_k or self.config["rag"]["top_k"]

        # Preferred path: cascade recall with metadata + centroid rerank.
        if self.scene_retriever and target_style_name and self.scene_retriever.has_author(target_style_name):
            refs, _ = self.scene_retriever.retrieve_topk_refs(
                source_text=source_text,
                target_author=target_style_name,
                stage=stage,
                location=location,
                event=event,
                candidate_k=10,
                top_k=k,
            )
            if refs:
                return refs

        # Fallback path: rank given candidate refs by embedding cosine to query text.
        if self.scene_retriever:
            ranked = self.scene_retriever.rank_plain_refs(source_text=source_text, refs=style_corpus, top_k=k)
            if ranked:
                return ranked

        return [x for x in style_corpus if isinstance(x, str) and x.strip()][:k]

    def generate(
        self,
        source_text: str,
        target_style_name: str,
        style_references: Optional[List[str]] = None,
        stage: Optional[str] = None,
        location: Optional[str] = None,
        event: Optional[str] = None,
    ) -> Dict[str, Any]:
        dynamic_refs = self.retrieve_style_references(
            source_text=source_text,
            style_corpus=style_references or [],
            target_style_name=target_style_name,
            stage=stage,
            location=location,
            event=event,
            top_k=self.config["rag"]["top_k"],
        )
        return self.generator.generate(
            source_text=source_text,
            target_style_name=target_style_name,
            style_references=dynamic_refs,
        )
