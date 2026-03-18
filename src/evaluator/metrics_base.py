from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMetric(ABC):
    @abstractmethod
    def evaluate(
        self,
        source_text: str,
        generated_text: str,
        style_references: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError
