from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.evaluator.metrics_base import BaseMetric
from src.utils.api_client import APIClient


class LLMStyleJudgeMetric(BaseMetric):
    def __init__(self, config: Dict[str, Any], api_client: APIClient) -> None:
        self.config = config
        self.api_client = api_client

    def _build_prompt(self, generated_text: str, style_references: Optional[List[str]]) -> str:
        refs = style_references or []
        refs_text = "\n\n".join([f"参考{i+1}: {text}" for i, text in enumerate(refs)])
        return (
            "你是风格评估裁判。请基于目标风格参考文本评估候选文本。\n"
            "请先在心中概括目标风格的核心特征，再进行评分，但最终只输出 JSON，不要额外输出分析过程。\n"
            "注意：本轮只评估风格相似度，不评估事实是否正确，不因为文本更长、更华丽或信息更多而直接加分。\n"
            "请从以下 3 个维度分别打 0-10 分：词汇偏好、句式结构、情绪表达。\n"
            "评分 rubric：0-3 表示基本不像目标风格；4-6 表示局部有相似但整体不足；7-8 表示整体较像且风格特征较明显；9-10 表示高度贴合、风格特征稳定且多维一致。\n"
            "overall 必须根据 lexical、syntax、emotion 三项综合给出，并尽量与三项平均值保持一致。\n"
            "必须严格输出 JSON，格式如下：\n"
            '{"lexical": 0.0, "syntax": 0.0, "emotion": 0.0, "overall": 0.0, "comment": "..."}\n\n'
            f"目标风格参考文本:\n{refs_text}\n\n"
            f"候选文本:\n{generated_text}\n"
        )

    def evaluate(
        self,
        source_text: str,
        generated_text: str,
        style_references: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(generated_text, style_references)
        raw = self.api_client.chat_completion(
            model=self.config["models"]["judge_model"],
            prompt=prompt,
            temperature=0.0,
            max_tokens=512,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            parsed = json.loads(raw[start : end + 1]) if start != -1 and end != -1 else {}
        overall = float(parsed.get("overall", 0.0))
        return {"score": overall, "details": parsed}
