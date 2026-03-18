from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generator.base_generator import BaseGenerator, StyleTransferGenerator


class PromptEngineeringGenerator(BaseGenerator):
    strategy_name = "Baseline_A_Prompt_Eng"

    def __init__(self, config: Dict[str, Any], api_client) -> None:
        super().__init__(config, api_client)
        self.generator = StyleTransferGenerator(config, api_client)

    def generate(
        self,
        source_text: str,
        target_style_name: str,
        style_references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self.generator.generate(
            source_text=source_text,
            target_style_name=target_style_name,
            style_references=[],
        )
