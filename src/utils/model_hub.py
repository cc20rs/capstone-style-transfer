from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def configure_huggingface_mirror(config: Dict[str, Any]) -> None:
    hf_cfg = config.get("huggingface", {})
    endpoint = hf_cfg.get("endpoint", "")
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)


def _is_local_model_ready(model_key: str, local_path: Path) -> bool:
    if not local_path.exists() or not local_path.is_dir():
        return False

    if model_key == "style_embedding_model":
        return (local_path / "modules.json").exists()

    if model_key in {"fluency_model", "bert_score_model"}:
        has_config = (local_path / "config.json").exists()
        has_weights = (local_path / "pytorch_model.bin").exists() or (local_path / "model.safetensors").exists()
        return has_config and has_weights

    return True


def resolve_model_name_or_path(config: Dict[str, Any], model_key: str) -> str:
    model_name = config["models"][model_key]
    hub_cfg = config.get("modelscope", {})
    enabled = bool(hub_cfg.get("enabled", False))
    if not enabled:
        return model_name

    local_map = hub_cfg.get("local_models", {})
    local_path_str = local_map.get(model_key)
    if not local_path_str:
        return model_name

    local_path = Path(local_path_str)
    if _is_local_model_ready(model_key, local_path):
        return str(local_path)
    return model_name
