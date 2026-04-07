import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import EXPERIMENT_JSONL_PATH, StyleTransferWebUI


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_tail(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    ui = StyleTransferWebUI()
    pipeline = ui.pipeline_adapter._get_pipeline()
    pipeline.api_client.timeout = 20
    pipeline.api_client.max_retries = 1
    pipeline.config["generation"]["max_tokens"] = 128
    if getattr(pipeline, "judge_models", None):
        pipeline.judge_models = [pipeline.judge_models[0]]
    case_id = "case_001"
    style = "余华"
    scenario = "叙事"
    methods = ["Prompt_Eng", "Context_Eng", "RAG_ICL"]

    source_text = ui.repo.get_source_text_by_case_id(case_id)
    if not source_text:
        print(json.dumps({"ok": False, "error": "case_001 source_text not found"}, ensure_ascii=False))
        return

    before_count = count_lines(EXPERIMENT_JSONL_PATH)
    results: list[dict] = []

    for method in methods:
        try:
            refs_update = ui._get_style_refs_update(
                method=method,
                style_label=style,
                scenario=scenario,
                source_text=source_text,
            )
            refs_text = refs_update.get("value", "") if isinstance(refs_update, dict) else ""

            generated_text, metrics, runtime_prompt, runtime_refs = ui._run_experiment(
                case_id=case_id,
                source_text=source_text,
                target_style=style,
                scenario=scenario,
                method=method,
                system_prompt=ui.DEFAULT_SYSTEM_PROMPT.format(target_style=style, scenario=scenario),
                style_references=refs_text,
            )
        except Exception as exc:
            results.append({"method": method, "ok": False, "error": str(exc)})
            continue

        if not isinstance(metrics, dict):
            results.append({"method": method, "ok": False, "error": "metrics_not_dict"})
            continue

        if "error" in metrics:
            results.append({"method": method, "ok": False, "error": metrics["error"]})
            continue

        save_status = ui._save_record(
            case_id=case_id,
            source_text=source_text,
            target_style=style,
            scenario=scenario,
            method=method,
            system_prompt=ui.DEFAULT_SYSTEM_PROMPT.format(target_style=style, scenario=scenario),
            style_references=refs_text,
            runtime_system_prompt=runtime_prompt,
            runtime_style_references=runtime_refs,
            generated_text=generated_text,
            metrics=metrics,
        )

        results.append(
            {
                "method": method,
                "ok": True,
                "generated_len": len((generated_text or "").strip()),
                "runtime_refs_len": len([x for x in (runtime_refs or "").split("\n\n") if x.strip()]),
                "style_vector_score": metrics.get("style_vector_score"),
                "save_status": save_status,
            }
        )

    after_count = count_lines(EXPERIMENT_JSONL_PATH)
    appended = max(after_count - before_count, 0)
    tail_rows = read_tail(EXPERIMENT_JSONL_PATH, appended if appended > 0 else 0)

    inspection = []
    for row in tail_rows:
        method = row.get("method")
        refs = row.get("style_references") or []
        refs_len = len(refs) if isinstance(refs, list) else 0
        sv = row.get("metrics", {}).get("style_vector_score")
        inspection.append(
            {
                "method": method,
                "style_refs_len": refs_len,
                "style_vector_score": sv,
                "prompt_should_be_zero": method != "Prompt_Eng" or (isinstance(sv, (int, float)) and abs(float(sv)) < 1e-12),
            }
        )

    print(
        json.dumps(
            {
                "ok": True,
                "before_count": before_count,
                "after_count": after_count,
                "appended": appended,
                "run_results": results,
                "inspection": inspection,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
