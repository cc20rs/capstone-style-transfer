from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.utils.api_client import APIClient


class BaseGenerator(ABC):
    def __init__(self, config: Dict[str, Any], api_client: APIClient) -> None:
        self.config = config
        self.api_client = api_client

    @abstractmethod
    def generate(
        self,
        source_text: str,
        target_style_name: str,
        style_references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class StyleTransferGenerator:
    def __init__(self, config: Dict[str, Any], api_client: APIClient) -> None:
        self.config = config
        self.api_client = api_client

    def _build_prompt(
        self,
        source_text: str,
        target_style_name: str,
        style_references: Optional[List[str]],
    ) -> str:
        refs = style_references or []
        refs_text = "\n\n".join([f"参考片段{i+1}: {item}" for i, item in enumerate(refs)])
        reference_block = (
            "以下是目标风格参考文本，请先在心中提炼稳定风格特征，再进行改写：\n"
            f"{refs_text}\n\n"
            if refs_text
            else ""
        )
        return (
            f"你是一名叙事风格迁移助手。请将给定事实文本重写为“{target_style_name}”风格。\n"
            "要求：\n"
            "1) 保持事实内容与关键事件不变，避免幻觉与新增设定。\n"
            "2) 在词汇偏好、句式节奏、情绪表达上体现目标风格。\n"
            "3) 可以借鉴参考语气，但不要照抄原句，不要出现连续照搬片段。\n"
            "4) 仅输出改写后的正文，不要解释。\n\n"
            f"{reference_block}"
            f"原文本：\n{source_text}\n"
        )

    def generate(
        self,
        source_text: str,
        target_style_name: str,
        style_references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(source_text, target_style_name, style_references)
        response_text = self.api_client.chat_completion(
            model=self.config["models"]["generation_model"],
            prompt=prompt,
            temperature=self.config["generation"]["temperature"],
            max_tokens=self.config["generation"]["max_tokens"],
        )
        return {
            "source_text": source_text,
            "target_style_name": target_style_name,
            "style_references": style_references or [],
            "generated_text": response_text,
        }
