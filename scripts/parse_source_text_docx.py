from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
HEADING_RE = re.compile(r"^[一二三四五六七八九十百千零]+、\s*([^（(]+)\s*[（(]([^）)]+)[）)]\s*$")


def _paragraph_text(paragraph: ET.Element) -> str:
    texts = [t.text for t in paragraph.findall(".//w:t", NS) if t.text]
    return "".join(texts).strip()


def _has_page_break(paragraph: ET.Element) -> bool:
    return paragraph.find('.//w:br[@w:type="page"]', NS) is not None


def _read_paragraphs(docx_path: Path) -> List[Tuple[str, bool]]:
    with zipfile.ZipFile(docx_path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = root.findall(".//w:body/w:p", NS)
    return [(_paragraph_text(p), _has_page_break(p)) for p in paragraphs]


def parse_docx_to_cases(docx_path: Path) -> List[dict]:
    paragraphs = _read_paragraphs(docx_path)

    # Only parse content after the first explicit page break (from page 2 onward).
    start_idx = 0
    for i, (_, has_break) in enumerate(paragraphs):
        if has_break:
            start_idx = i + 1
            break

    cases: List[dict] = []
    current_style = ""
    current_scene = ""
    current_paragraphs: List[str] = []

    def flush_current() -> None:
        if not current_style:
            return
        text = "\n\n ".join(p for p in current_paragraphs if p.strip())
        if not text:
            return
        case_id = f"case_{len(cases) + 1:03d}"
        cases.append(
            {
                "case_id": case_id,
                "style_name": current_style,
                "scene_tags": current_scene.replace("・", "·"),
                "source_text": text,
            }
        )

    for raw_text, _ in paragraphs[start_idx:]:
        text = raw_text.strip()
        if not text:
            continue

        heading = HEADING_RE.match(text)
        if heading:
            flush_current()
            current_style = heading.group(1).strip()
            current_scene = heading.group(2).strip()
            current_paragraphs = []
            continue

        if current_style:
            current_paragraphs.append(text)

    flush_current()
    return cases


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_docx = project_root / "data" / "source_text" / "测试文本.docx"
    output_json = source_docx.with_name("source_text.json")

    cases = parse_docx_to_cases(source_docx)

    output_json.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Parsed {len(cases)} cases -> {output_json}")


if __name__ == "__main__":
    main()
