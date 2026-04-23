from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.api_client import APIClient
from src.utils.file_io import load_yaml


def parse_allowed_values(md_text: str) -> Dict[str, List[str]]:
    def _extract(dim_name: str) -> List[str]:
        pattern = rf"维度\s*{dim_name}[^\[]*\[([^\]]+)\]"
        match = re.search(pattern, md_text)
        if not match:
            raise ValueError(f"Cannot parse allowed values for dimension {dim_name}")
        raw = match.group(1)
        values = [v.strip() for v in re.split(r"[，,]", raw) if v.strip()]
        return values

    return {
        "stage": _extract("A"),
        "location": _extract("B"),
        "event": _extract("C"),
    }


def extract_first_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_or_null(value: Any, allowed: List[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or v.lower() == "null":
        return None
    return v if v in allowed else None


def build_prompt(
    source_text: str,
    scene_tags: Optional[str],
    allowed_stage: List[str],
    allowed_location: List[str],
    allowed_event: List[str],
) -> str:
    text_for_model = source_text.strip()
    if len(text_for_model) > 2200:
        text_for_model = text_for_model[:2200]

    return (
        "你是标签标准化助手。请根据文本语义，为文本打三个维度标签：stage/location/event。\n"
        "必须严格从给定枚举中选择；如果无法判断，填 null。\n"
        "只输出 JSON 对象，不要输出任何解释。\n\n"
        f"允许的 stage 枚举: {allowed_stage}\n"
        f"允许的 location 枚举: {allowed_location}\n"
        f"允许的 event 枚举: {allowed_event}\n\n"
        "输出格式(严格一致):\n"
        '{"stage": "...或null", "location": "...或null", "event": "...或null"}\n\n'
        f"已有人工粗标签(scene_tags): {scene_tags if scene_tags else 'null'}\n"
        f"文本:\n{text_for_model}\n"
    )


def main() -> None:
    config = load_yaml(ROOT / "configs" / "config.yaml")
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

    md_path = ROOT / "data" / "source_text" / "scene_and_recall.md"
    json_path = ROOT / "data" / "source_text" / "source_text.json"

    allowed = parse_allowed_values(md_path.read_text(encoding="utf-8"))
    stage_allowed = allowed["stage"]
    location_allowed = allowed["location"]
    event_allowed = allowed["event"]

    data = json.loads(json_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError("Invalid source_text.json format: cases must be a list")

    client = APIClient(config, api_key)
    model = "DeepSeek-V3.2"

    cleaned_cases: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, 1):
        source_text = str(case.get("source_text", ""))
        rough_scene_tags = case.get("scene_tags")

        prompt = build_prompt(
            source_text=source_text,
            scene_tags=str(rough_scene_tags) if rough_scene_tags is not None else None,
            allowed_stage=stage_allowed,
            allowed_location=location_allowed,
            allowed_event=event_allowed,
        )

        response = client.chat_completion(
            model=model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=160,
        )

        parsed = extract_first_json_object(response)
        stage = validate_or_null(parsed.get("stage"), stage_allowed)
        location = validate_or_null(parsed.get("location"), location_allowed)
        event = validate_or_null(parsed.get("event"), event_allowed)

        new_case = {k: v for k, v in case.items() if k != "scene_tags"}
        new_case["stage"] = stage
        new_case["location"] = location
        new_case["event"] = event

        cleaned_cases.append(new_case)
        print(f"[{idx}/{len(cases)}] {new_case.get('case_id', f'case_{idx:03d}')} -> stage={stage}, location={location}, event={event}")

    data["cases"] = cleaned_cases
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {json_path}")


if __name__ == "__main__":
    main()
