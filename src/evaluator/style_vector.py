from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluator.metrics_base import BaseMetric
from src.utils.api_client import APIClient


class StyleVectorDistanceMetric(BaseMetric):
    LABELS: List[str] = ["完全符合", "比较符合", "比较不符合", "完全不符合"]
    LABEL_TO_RANGE: Dict[str, List[int]] = {
        "完全符合": [8, 9, 10],
        "比较符合": [6, 7],
        "比较不符合": [4, 5],
        "完全不符合": [1, 2, 3],
    }

    def __init__(self, config: Dict[str, Any], api_client: APIClient) -> None:
        self.config = config
        self.api_client = api_client
        self.project_root = Path(__file__).resolve().parents[2]
        prompt_path = self.project_root / "data" / "source_text" / "total" / "style_system_prompt.md"
        if not prompt_path.exists():
            raise RuntimeError(f"Missing style judge prompt file: {prompt_path}")
        self.core_prompt = prompt_path.read_text(encoding="utf-8").strip()

        self.system_prompt = (
            "你是风格一致性评估裁判。请先完整理解并严格遵循以下风格特征抓取框架：\n"
            f"{self.core_prompt}\n\n"
            "接下来，请你仅评估候选文本在风格层面与目标风格的匹配程度，"
            "不要评估事实正确性、信息量多少或文采华丽程度。\n"
            "完成判断后，请严格输出 JSON，且只允许包含以下两个字段：label 与 comment。\n"
            "其中 label 必须且只能是以下四个标签之一：\n"
            "1) 完全符合（对应整数评分 9-10）\n"
            "2) 比较符合（对应整数评分 7-8）\n"
            "3) 比较不符合（对应整数评分 4-6）\n"
            "4) 完全不符合（对应整数评分 1-3）\n\n"
            "comment 需为简短评语（不超过30字），用于说明依据，禁止输出长段分析。\n"
            "若判断存在犹豫，请采用保守原则选择较低一档。\n\n"
        )

    @staticmethod
    def _extract_label(raw: str) -> Optional[str]:
        text = (raw or "").strip()
        for label in StyleVectorDistanceMetric.LABELS:
            if text == label:
                return label
        for label in StyleVectorDistanceMetric.LABELS:
            if label in text:
                return label
        return None

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                return {}

    @classmethod
    def _label_to_score(cls, label: str) -> float:
        values = cls.LABEL_TO_RANGE.get(label, [0])
        return float(sum(values) / len(values))

    def _build_user_prompt(self, generated_text: str, target_style: str) -> str:
        return (
            f"目标风格：{target_style}\n\n"
            f"候选文本：\n{generated_text}\n\n"
            "请严格输出 JSON：{\"label\": \"完全符合|比较符合|比较不符合|完全不符合\", \"comment\": \"简短评语\"}"
        )

    def evaluate(
        self,
        source_text: str,
        generated_text: str,
        target_style: Optional[str] = None,
        style_references: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target_style_text = (target_style or kwargs.get("target_style_name") or kwargs.get("style_name") or "").strip()
        if not target_style_text:
            target_style_text = "未知风格"

        user_prompt = self._build_user_prompt(generated_text, target_style_text)
        raw = self.api_client.chat_completion(
            model=self.config["models"]["judge_model"],
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            temperature=0.0,
            max_tokens=80,
        )
        parsed = self._extract_json(raw)
        label = self._extract_label(str(parsed.get("label", ""))) or self._extract_label(raw)
        if label is None:
            label = "比较不符合"
        comment = str(parsed.get("comment", "")).strip()
        if not comment:
            comment = "风格特征匹配度有限"
        if len(comment) > 30:
            comment = comment[:30]
        score = self._label_to_score(label)

        return {
            "score": score,
            "details": {
                "metric": "llm_as_judge_4level",
                "label": label,
                "comment": comment,
                "score_range": self.LABEL_TO_RANGE[label],
                "raw_output": raw,
                "prompt_path": "data/source_text/total/style_system_prompt.md",
            },
        }
