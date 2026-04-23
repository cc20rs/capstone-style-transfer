from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.api_client import APIClient
from src.utils.file_io import load_yaml


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
ITEM_RE = re.compile(r"^【\s*(\d{3})\s*】\s*(.*)$")
SECTION_RE = re.compile(r"^[一二三四五六七八九十百千零]+、")
DESC_RE = re.compile(r"^（[^）]*）$")
CITATION_RE = re.compile(r"^[—\-]{2,}")


AUTHOR_META = {
    "余华": {
        "prefix": "yu_hua",
        "file": ROOT / "data" / "source_text" / "语料库（未清洗）" / "余华自传语料库.docx",
    },
    "汪曾祺": {
        "prefix": "wang_zeng_qi",
        "file": ROOT / "data" / "source_text" / "语料库（未清洗）" / "汪曾祺自传语料库.docx",
    },
}


def parse_allowed_values(md_text: str) -> Dict[str, List[str]]:
    def _extract(dim_name: str) -> List[str]:
        pattern = rf"维度\s*{dim_name}[^\[]*\[([^\]]+)\]"
        match = re.search(pattern, md_text)
        if not match:
            raise ValueError(f"Cannot parse allowed values for dimension {dim_name}")
        values = [v.strip() for v in re.split(r"[，,]", match.group(1)) if v.strip()]
        return values

    return {
        "stage": _extract("A"),
        "location": _extract("B"),
        "event": _extract("C"),
    }


def extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
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


def read_docx_paragraphs(docx_path: Path) -> List[str]:
    with zipfile.ZipFile(docx_path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs: List[str] = []
    for p in root.findall(".//w:body/w:p", NS):
        txt = "".join(t.text for t in p.findall(".//w:t", NS) if t.text).strip()
        paragraphs.append(txt)
    return paragraphs


def parse_entries_from_docx(docx_path: Path) -> List[Dict[str, Any]]:
    paragraphs = read_docx_paragraphs(docx_path)

    first_item_idx = -1
    for i, t in enumerate(paragraphs):
        if ITEM_RE.match(t):
            first_item_idx = i
            break
    if first_item_idx < 0:
        return []

    entries: List[Dict[str, Any]] = []
    current_no: Optional[str] = None
    current_label: str = ""
    current_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current_no, current_label, current_lines
        if current_no is None:
            return
        content_lines = [
            ln.strip()
            for ln in current_lines
            if ln.strip()
            and not SECTION_RE.match(ln.strip())
            and not DESC_RE.match(ln.strip())
            and not CITATION_RE.match(ln.strip())
            and not ITEM_RE.match(ln.strip())
        ]
        content = "\n\n ".join(content_lines)
        if content:
            entries.append({"num": current_no, "label": current_label, "content": content})
        current_no = None
        current_label = ""
        current_lines = []

    for raw in paragraphs[first_item_idx:]:
        text = raw.strip()
        if not text:
            continue
        item_match = ITEM_RE.match(text)
        if item_match:
            flush_current()
            current_no = item_match.group(1)
            current_label = item_match.group(2).strip()
            continue

        if current_no is not None:
            if CITATION_RE.match(text):
                # Drop citation and close current entry.
                flush_current()
                continue
            current_lines.append(text)

    flush_current()
    return entries


def build_batch_prompt(
    items: List[Dict[str, Any]],
    allowed_stage: List[str],
    allowed_location: List[str],
    allowed_event: List[str],
) -> str:
    lines: List[str] = []
    for item in items:
        content_for_model = item["content"] if len(item["content"]) <= 1200 else item["content"][:1200]
        lines.append(
            "- corpus_id: {corpus_id}\n"
            "  author: {author}\n"
            "  label_hint: {label_hint}\n"
            "  content: {content}".format(
                corpus_id=item["corpus_id"],
                author=item["author"],
                label_hint=item.get("label_hint") or "null",
                content=content_for_model.replace("\n", " "),
            )
        )

    return (
        "你是文本标签清洗助手。请根据给定文本，为每条数据推断 stage/location/event 三个标签。\n"
        "只允许从枚举中选取；无法判断填 null。\n"
        "严格只输出 JSON，不要输出解释。\n\n"
        f"stage 枚举: {allowed_stage}\n"
        f"location 枚举: {allowed_location}\n"
        f"event 枚举: {allowed_event}\n\n"
        "输出必须是 JSON 数组，每个元素包含: corpus_id, stage, location, event。\n"
        "示例: [{\"corpus_id\":\"yu_hua_001\",\"stage\":\"童年\",\"location\":\"医院\",\"event\":\"日常\"}]\n\n"
        "待标注数据:\n"
        + "\n\n".join(lines)
        + "\n"
    )


def main() -> None:
    config = load_yaml(ROOT / "configs" / "config.yaml")
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

    scene_md_path = ROOT / "data" / "source_text" / "scene_and_recall.md"
    output_path = ROOT / "data" / "source_text" / "corpus_text.json"

    allowed = parse_allowed_values(scene_md_path.read_text(encoding="utf-8"))
    stage_allowed = allowed["stage"]
    location_allowed = allowed["location"]
    event_allowed = allowed["event"]

    all_items: List[Dict[str, Any]] = []
    for author, meta in AUTHOR_META.items():
        entries = parse_entries_from_docx(meta["file"])
        for e in entries:
            corpus_id = f"{meta['prefix']}_{e['num']}"
            all_items.append(
                {
                    "corpus_id": corpus_id,
                    "author": author,
                    "label_hint": e["label"],
                    "content": e["content"],
                }
            )

    client = APIClient(config, api_key)
    model = "DeepSeek-V3.2"

    cleaned: List[Dict[str, Any]] = []
    batch_size = 8
    id_to_tag: Dict[str, Dict[str, Any]] = {}

    for start in range(0, len(all_items), batch_size):
        batch = all_items[start : start + batch_size]
        prompt = build_batch_prompt(
            items=batch,
            allowed_stage=stage_allowed,
            allowed_location=location_allowed,
            allowed_event=event_allowed,
        )

        response = client.chat_completion(
            model=model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=1200,
        )
        payload = extract_json_payload(response)
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        if not isinstance(rows, list):
            rows = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("corpus_id", "")).strip()
            if not cid:
                continue
            id_to_tag[cid] = {
                "stage": validate_or_null(row.get("stage"), stage_allowed),
                "location": validate_or_null(row.get("location"), location_allowed),
                "event": validate_or_null(row.get("event"), event_allowed),
            }

        for i, item in enumerate(batch, 1):
            tags = id_to_tag.get(item["corpus_id"], {})
            stage = validate_or_null(tags.get("stage"), stage_allowed)
            location = validate_or_null(tags.get("location"), location_allowed)
            event = validate_or_null(tags.get("event"), event_allowed)

            cleaned.append(
                {
                    "corpus_id": item["corpus_id"],
                    "author": item["author"],
                    "content": item["content"],
                    "stage": stage,
                    "location": location,
                    "event": event,
                }
            )

            print(
                f"[{start + i}/{len(all_items)}] {item['corpus_id']} -> "
                f"stage={stage}, location={location}, event={event}"
            )

        output_path.write_text(
            json.dumps({"corpus": cleaned}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    output_path.write_text(
        json.dumps({"corpus": cleaned}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(cleaned)} entries to {output_path}")


if __name__ == "__main__":
    main()
