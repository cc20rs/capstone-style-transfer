from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

from modelscope.hub.snapshot_download import snapshot_download
import yaml

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "configs" / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    hf_cfg = config.get("huggingface", {})
    if hf_cfg.get("endpoint"):
        os.environ["HF_ENDPOINT"] = str(hf_cfg["endpoint"])

    ms_cfg = config.get("modelscope", {})
    if not ms_cfg.get("enabled", False):
        print("ModelScope is disabled in config. Skip download.")
        return

    cache_root = project_root / ms_cfg.get("cache_dir", ".modelscope_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    default_mapping = {
        "style_embedding_model": config["models"]["style_embedding_model"],
        "fluency_model": config["models"]["fluency_model"],
        "bert_score_model": config["models"].get("bert_score_model", "bert-base-chinese"),
    }
    override_mapping = ms_cfg.get("model_id_overrides", {})
    model_mapping = {k: override_mapping.get(k, v) for k, v in default_mapping.items()}

    for key, repo_id in model_mapping.items():
        local_target_str = ms_cfg.get("local_models", {}).get(key, "")
        if not local_target_str:
            continue
        local_target = project_root / local_target_str
        local_target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ModelScope] downloading {repo_id} -> {local_target}")

        max_retries = 3
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                temp_dir = local_target / "._____temp"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                snapshot_download(model_id=repo_id, local_dir=str(local_target))
                success = True
                print(f"[ModelScope] success: {repo_id}")
                break
            except Exception as exc:
                print(f"[ModelScope] attempt {attempt}/{max_retries} failed for {repo_id}: {exc}")
                temp_dir = local_target / "._____temp"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                time.sleep(min(5 * attempt, 15))

        if not success:
            print(
                f"[ModelScope] skip after retries: {repo_id}. "
                "Pipeline will fallback to remote model id if local cache is incomplete."
            )

    print("ModelScope download completed.")


if __name__ == "__main__":
    main()
