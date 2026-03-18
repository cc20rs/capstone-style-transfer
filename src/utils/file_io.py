from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_intermediate_result(file_path: Path, result: Dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_csv_report(file_path: Path, records: List[Dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with file_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["message"])
            writer.writerow(["no records"])
        return
    fieldnames = sorted({key for row in records for key in row.keys()})
    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
