from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import jieba
import jieba.posseg as pseg

from src.evaluator.metrics_base import BaseMetric
from src.utils.file_io import load_json


class LinguisticFeatureMetric(BaseMetric):
    SCENARIO_CONTEXT_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
        "luxun": {
            "叙事": [
                "他把雨伞收在门后，鞋底却带进来一地泥水，屋里的人都不作声。",
                "钟声一下一下敲着，像把白日里没说完的话都钉在墙上。",
            ],
            "抒情": [
                "风从窗纸缝里钻进来，像旧事的手，轻轻一碰就凉到心里。",
                "夜色并不深，只是人心被雨声压得低了下去。",
            ],
            "讽刺": [
                "人人都说公道，偏偏公道总要等散会后才肯露面。",
                "他把漂亮话说得很圆，圆得正好滚开了责任。",
            ],
            "议论": [
                "事情并不复杂，复杂的是人人都愿意把简单话说成雾。",
                "所谓体面，不过是把难堪折叠起来，暂借灯光照着。",
            ],
        },
        "qianzhongshu": {
            "叙事": [
                "他上楼时鞋声极轻，像怕惊动自己刚编好的理由。",
                "茶凉得很快，像会议纪要里的热情。",
            ],
            "抒情": [
                "黄昏像一封写了一半的信，落款总在天黑以后。",
                "人的惆怅常常很文明，连叹息都懂得排队。",
            ],
            "讽刺": [
                "他们把原则谈得如同瓷器，真正用时却当作一次性纸杯。",
                "掌声总是很准时，结论却总在路上堵车。",
            ],
            "议论": [
                "观点若没有代价，不过是语言对现实的客气。",
                "聪明未必通向真理，倒常先通向自我原谅。",
            ],
        },
        "biography": {
            "叙事": [
                "他在那一年做出决定，此后数十年的人生轨迹都由此改写。",
                "一次看似寻常的经历，后来成为其思想转向的重要节点。",
            ],
            "抒情": [
                "回望旧日，他把那些沉默时刻称作生命中最响亮的回声。",
                "岁月在他的叙述里不再抽象，而是一页页可以触摸的纹理。",
            ],
            "讽刺": [
                "外界常以头衔定义他，真正塑造他的却是那些无人喝彩的失败。",
                "历史喜欢整齐的结论，而他的经历偏偏总从意外处生长。",
            ],
            "议论": [
                "个体命运与时代结构并非平行线，它们总在关键处互相改写。",
                "若只看结果，便会错过一生中最具解释力的过程。",
            ],
        },
    }

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
        target_scenario: str,
        style_references: Optional[List[str]],
    ) -> List[str]:
        refs = style_references or []
        if refs:
            return refs
        scenario_refs = self.SCENARIO_CONTEXT_TEMPLATES.get(target_style_name, {}).get(target_scenario, [])
        if scenario_refs:
            return scenario_refs
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
        target_scenario = kwargs.get("target_scenario", "")
        target_refs = self._resolve_target_refs(target_style_name, target_scenario, style_references)
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
