from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.model_hub import resolve_model_name_or_path


@dataclass
class RecallInfo:
    level: str
    candidate_count: int
    selected_count: int


class CascadeSceneRetriever:
    def __init__(self, config: Dict[str, Any], corpus_rows: List[Dict[str, Any]]) -> None:
        self.config = config
        model_name = resolve_model_name_or_path(config, "style_embedding_model")
        self.embedder = SentenceTransformer(model_name)

        self.author_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in corpus_rows:
            author = self._norm_label(row.get("author"))
            content = str(row.get("content", "")).strip()
            if not author or not content:
                continue
            cleaned = {
                "author": author,
                "content": content,
                "stage": self._norm_label(row.get("stage")),
                "location": self._norm_label(row.get("location")),
                "event": self._norm_label(row.get("event")),
            }
            self.author_rows.setdefault(author, []).append(cleaned)

        self.author_embeddings: Dict[str, np.ndarray] = {}
        self.author_centroids: Dict[str, np.ndarray] = {}

    @staticmethod
    def _norm_label(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def has_author(self, author: str) -> bool:
        a = self._norm_label(author)
        return bool(a and a in self.author_rows and self.author_rows[a])

    def _ensure_author_matrix(self, author: str) -> None:
        if author in self.author_embeddings:
            return
        rows = self.author_rows.get(author, [])
        texts = [r["content"] for r in rows]
        if not texts:
            self.author_embeddings[author] = np.zeros((0, 1), dtype=np.float32)
            self.author_centroids[author] = np.zeros((1,), dtype=np.float32)
            return

        emb = self.embedder.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        centroid = emb.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        self.author_embeddings[author] = emb
        self.author_centroids[author] = centroid.astype(np.float32)

    @staticmethod
    def _match_count(row: Dict[str, Any], stage: Optional[str], location: Optional[str], event: Optional[str]) -> int:
        hit = 0
        if stage is not None and row.get("stage") == stage:
            hit += 1
        if location is not None and row.get("location") == location:
            hit += 1
        if event is not None and row.get("event") == event:
            hit += 1
        return hit

    def _cascade_indices(
        self,
        author: str,
        stage: Optional[str],
        location: Optional[str],
        event: Optional[str],
        candidate_k: int,
    ) -> Tuple[List[int], str]:
        rows = self.author_rows.get(author, [])
        if not rows:
            return [], "empty"

        strict = [
            i
            for i, row in enumerate(rows)
            if self._match_count(row, stage=stage, location=location, event=event) >= 2
        ]
        if len(strict) >= candidate_k:
            return strict, "strict"

        core = []
        for i, row in enumerate(rows):
            loc_hit = location is not None and row.get("location") == location
            evt_hit = event is not None and row.get("event") == event
            if loc_hit or evt_hit:
                core.append(i)
        if len(core) >= candidate_k:
            return core, "core"

        return list(range(len(rows))), "author_fallback"

    def retrieve_topk_refs(
        self,
        *,
        source_text: str,
        target_author: str,
        stage: Optional[str],
        location: Optional[str],
        event: Optional[str],
        candidate_k: int = 10,
        top_k: int = 3,
    ) -> Tuple[List[str], RecallInfo]:
        author = self._norm_label(target_author)
        if author is None or not self.has_author(author):
            return [], RecallInfo(level="empty", candidate_count=0, selected_count=0)

        stage = self._norm_label(stage)
        location = self._norm_label(location)
        event = self._norm_label(event)

        candidate_indices, level = self._cascade_indices(author, stage, location, event, candidate_k=candidate_k)
        if not candidate_indices:
            return [], RecallInfo(level=level, candidate_count=0, selected_count=0)

        self._ensure_author_matrix(author)
        emb = self.author_embeddings[author]
        centroid = self.author_centroids[author]

        selected_mat = emb[candidate_indices]
        style_sims = np.dot(selected_mat, centroid)
        top_local = np.argsort(style_sims)[::-1][: min(top_k, len(candidate_indices))]
        chosen_indices = [candidate_indices[int(i)] for i in top_local]
        refs = [self.author_rows[author][idx]["content"] for idx in chosen_indices]

        return refs, RecallInfo(level=level, candidate_count=len(candidate_indices), selected_count=len(refs))

    def rank_plain_refs(self, source_text: str, refs: List[str], top_k: int = 3) -> List[str]:
        cleaned = [x.strip() for x in refs if isinstance(x, str) and x.strip()]
        if not cleaned:
            return []
        if len(cleaned) <= top_k:
            return cleaned

        query = self.embedder.encode([source_text], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(query, dtype=np.float32)[0]
        ref_emb = self.embedder.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
        ref_mat = np.asarray(ref_emb, dtype=np.float32)
        sims = np.dot(ref_mat, q)
        top = np.argsort(sims)[::-1][:top_k]
        return [cleaned[int(i)] for i in top]
