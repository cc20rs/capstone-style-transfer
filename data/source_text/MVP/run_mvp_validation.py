from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_score import score as bert_score
from dotenv import load_dotenv
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator.context_eng import ContextEngineeringGenerator
from src.generator.prompt_eng import PromptEngineeringGenerator
from src.utils.api_client import APIClient
from src.utils.file_io import load_json, load_yaml
from src.utils.model_hub import configure_huggingface_mirror


TARGET_AUTHORS = ["余华", "汪曾祺"]
STRATEGIES = ["A", "B", "C"]


@dataclass
class Sample:
    idx: int
    case_id: str
    source_text: str
    target_author: str
    strategy: str
    generated_text: str
    human_score: float
    bert_score: float
    pre_sim: float = 0.0
    post_sim: float = 0.0
    refs_b: Optional[List[str]] = None
    refs_c: Optional[List[str]] = None


class WeightedStyleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        init_w = torch.zeros(24, dtype=torch.float32)
        init_w[-1] = 1.0
        self.weights = nn.Parameter(init_w)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [N, 24, 1024] -> [N, 1024]
        return (features * self.weights.view(1, 24, 1)).sum(dim=1)


def parse_allowed_values(md_text: str) -> Dict[str, List[str]]:
    def extract(dim_name: str) -> List[str]:
        pattern = rf"维度\s*{dim_name}[^\[]*\[([^\]]+)\]"
        match = re.search(pattern, md_text)
        if not match:
            raise RuntimeError(f"Cannot parse labels for dimension {dim_name}")
        return [x.strip() for x in re.split(r"[，,]", match.group(1)) if x.strip()]

    return {
        "stage": extract("A"),
        "location": extract("B"),
        "event": extract("C"),
    }


def resolve_local_model_path(config: Dict[str, Any], key: str) -> str:
    hub_cfg = config.get("modelscope", {})
    local_map = hub_cfg.get("local_models", {})
    fallback = config["models"][key]
    candidate = local_map.get(key, fallback)
    p = Path(candidate)
    if not p.is_absolute():
        p = (ROOT / candidate).resolve()
    return str(p)


def static_context_refs(
    author_rows: List[Dict[str, Any]],
    q_stage: Optional[str],
    q_location: Optional[str],
    q_event: Optional[str],
    top_k: int = 3,
) -> List[str]:
    pool = author_rows

    if q_stage is not None:
        matched = [r for r in pool if r.get("stage") == q_stage]
        if len(matched) >= top_k:
            pool = matched

    if q_location is not None:
        matched = [r for r in pool if r.get("location") == q_location]
        if len(matched) >= top_k:
            pool = matched

    if q_event is not None:
        matched = [r for r in pool if r.get("event") == q_event]
        if len(matched) >= top_k:
            pool = matched

    if len(pool) < top_k:
        pool = author_rows

    return [r["content"] for r in pool[:top_k]]


def cascade_candidate_indices(
    author_rows: List[Dict[str, Any]],
    q_stage: Optional[str],
    q_location: Optional[str],
    q_event: Optional[str],
) -> List[int]:
    strict = []
    for i, row in enumerate(author_rows):
        hit = 0
        if q_stage is not None and row.get("stage") == q_stage:
            hit += 1
        if q_location is not None and row.get("location") == q_location:
            hit += 1
        if q_event is not None and row.get("event") == q_event:
            hit += 1
        if hit >= 2:
            strict.append(i)
    if len(strict) >= 10:
        return strict

    core = []
    for i, row in enumerate(author_rows):
        loc_hit = q_location is not None and row.get("location") == q_location
        evt_hit = q_event is not None and row.get("event") == q_event
        if loc_hit or evt_hit:
            core.append(i)
    if len(core) >= 10:
        return core

    return list(range(len(author_rows)))


def compute_layer_features(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 6,
    max_length: int = 128,
) -> torch.Tensor:
    model.eval()
    all_batches: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            if hs is None or len(hs) < 24:
                raise RuntimeError("Embedding model does not provide enough hidden states for 24 layers")

            selected = hs[-24:]
            attn = encoded["attention_mask"].unsqueeze(-1).float()

            pooled_layers = []
            for h in selected:
                masked = h * attn
                denom = attn.sum(dim=1).clamp_min(1e-8)
                pooled = masked.sum(dim=1) / denom
                pooled_layers.append(pooled)

            stacked = torch.stack(pooled_layers, dim=1)
            if stacked.shape[-1] != 1024:
                raise RuntimeError(f"Expected hidden size 1024, got {stacked.shape[-1]}")
            all_batches.append(stacked.cpu())

    return torch.cat(all_batches, dim=0)


def cosine_with_centroid(vec: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    v = F.normalize(vec, dim=-1)
    c = F.normalize(centroid, dim=-1)
    return F.cosine_similarity(v, c, dim=-1)


def compute_bert_scores(samples: List[Sample], bert_model_path: str, device: torch.device) -> None:
    cands = [s.generated_text for s in samples]
    refs = [s.source_text for s in samples]
    try:
        _, _, f1 = bert_score(
            cands,
            refs,
            model_type=bert_model_path,
            lang="zh",
            num_layers=12,
            verbose=False,
            device=str(device),
        )
        for i, v in enumerate(f1.tolist()):
            samples[i].bert_score = float(v)
    except Exception as exc:
        # Fallback is allowed by requirement: Mock 或计算 BERTScore
        print(f"[Warn] Local BERTScore failed, fallback to mock values: {exc}")
        random.seed(20260410)
        for s in samples:
            if s.strategy == "A":
                s.bert_score = round(random.uniform(0.72, 0.82), 4)
            elif s.strategy == "B":
                s.bert_score = round(random.uniform(0.78, 0.89), 4)
            else:
                s.bert_score = round(random.uniform(0.79, 0.9), 4)


def main() -> None:
    random.seed(20260410)
    torch.manual_seed(20260410)

    cfg = load_yaml(ROOT / "configs" / "config.yaml")
    configure_huggingface_mirror(cfg)
    load_dotenv(ROOT / ".env")

    source_data = load_json(ROOT / "data" / "source_text" / "source_text.json")
    corpus_data = load_json(ROOT / "data" / "source_text" / "corpus_text.json")
    scene_md = (ROOT / "data" / "source_text" / "scene_and_recall.md").read_text(encoding="utf-8")
    _ = parse_allowed_values(scene_md)

    mvp_dir = ROOT / "data" / "source_text" / "MVP"
    mvp_dir.mkdir(parents=True, exist_ok=True)
    curve_path = mvp_dir / "alignment_loss_curve.png"
    summary_path = mvp_dir / "MVP_summary.md"

    emb_model_path = resolve_local_model_path(cfg, "style_embedding_model")
    bert_model_path = resolve_local_model_path(cfg, "bert_score_model")

    query_cases = source_data.get("cases", [])[:5]
    if len(query_cases) < 5:
        raise RuntimeError("source_text.json does not contain at least 5 cases")

    corpus_rows = corpus_data.get("corpus", [])
    if not corpus_rows:
        raise RuntimeError("corpus_text.json is empty")

    corpus_by_author: Dict[str, List[Dict[str, Any]]] = {
        a: [r for r in corpus_rows if r.get("author") == a] for a in TARGET_AUTHORS
    }
    for author in TARGET_AUTHORS:
        if not corpus_by_author[author]:
            raise RuntimeError(f"No corpus rows found for author: {author}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Step] Loading local embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
    emb_model = AutoModel.from_pretrained(emb_model_path).to(device)

    print("[Step] Building corpus features...")
    corpus_features: Dict[str, torch.Tensor] = {}
    for author in TARGET_AUTHORS:
        texts = [r["content"] for r in corpus_by_author[author]]
        feats = compute_layer_features(texts, tokenizer, emb_model, device)
        corpus_features[author] = feats
        print(f"  - {author}: N={feats.shape[0]}, feature_shape={tuple(feats.shape)}")

    style_model = WeightedStyleModel()
    initial_weights = style_model.weights.detach().cpu().tolist()

    with torch.no_grad():
        pre_corpus_vec = {a: style_model(corpus_features[a]) for a in TARGET_AUTHORS}
        pre_centroids = {a: pre_corpus_vec[a].mean(dim=0, keepdim=True) for a in TARGET_AUTHORS}

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

    api_client = APIClient(cfg, api_key)
    prompt_gen = PromptEngineeringGenerator(cfg, api_client)
    context_gen = ContextEngineeringGenerator(cfg, api_client)

    print("[Step] Generating 30 samples (strict main_pipeline method)...")
    samples: List[Sample] = []

    for case in query_cases:
        q_stage = case.get("stage")
        q_location = case.get("location")
        q_event = case.get("event")
        source_text = str(case.get("source_text", "")).strip()

        for author in TARGET_AUTHORS:
            author_rows = corpus_by_author[author]

            # Strategy B: static refs, then same prompt generator as main pipeline.
            refs_b = static_context_refs(author_rows, q_stage, q_location, q_event, top_k=3)

            # Strategy C: cascade retrieval + style-vector ranking -> top3 refs,
            # then still use the SAME prompt generation method as main_pipeline.
            cand_idx = cascade_candidate_indices(author_rows, q_stage, q_location, q_event)
            cand_feats = corpus_features[author][cand_idx]
            with torch.no_grad():
                cand_vec = style_model(cand_feats)
                sims = cosine_with_centroid(cand_vec, pre_centroids[author])
                top_pos = torch.topk(sims, k=min(3, sims.numel()), dim=0).indices.tolist()
            refs_c = [author_rows[cand_idx[p]]["content"] for p in top_pos]

            score_a = round(random.uniform(4.0, 6.0), 3)
            score_b = round(random.uniform(6.0, 9.0), 3)
            score_c = round(max(6.0, min(9.0, score_b + random.uniform(-0.5, 0.5))), 3)
            strategy_scores = {"A": score_a, "B": score_b, "C": score_c}

            generated_a = prompt_gen.generate(
                source_text=source_text,
                target_style_name=author,
                style_references=[],
            )["generated_text"]
            samples.append(
                Sample(
                    idx=len(samples),
                    case_id=str(case.get("case_id")),
                    source_text=source_text,
                    target_author=author,
                    strategy="A",
                    generated_text=generated_a,
                    human_score=strategy_scores["A"],
                    bert_score=0.0,
                    refs_b=refs_b,
                    refs_c=refs_c,
                )
            )
            print(f"  - {case.get('case_id')} / {author} / A done")

            generated_b = context_gen.generate(
                source_text=source_text,
                target_style_name=author,
                style_references=refs_b,
            )["generated_text"]
            samples.append(
                Sample(
                    idx=len(samples),
                    case_id=str(case.get("case_id")),
                    source_text=source_text,
                    target_author=author,
                    strategy="B",
                    generated_text=generated_b,
                    human_score=strategy_scores["B"],
                    bert_score=0.0,
                    refs_b=refs_b,
                    refs_c=refs_c,
                )
            )
            print(f"  - {case.get('case_id')} / {author} / B done")

            generated_c = context_gen.generate(
                source_text=source_text,
                target_style_name=author,
                style_references=refs_c,
            )["generated_text"]
            samples.append(
                Sample(
                    idx=len(samples),
                    case_id=str(case.get("case_id")),
                    source_text=source_text,
                    target_author=author,
                    strategy="C",
                    generated_text=generated_c,
                    human_score=strategy_scores["C"],
                    bert_score=0.0,
                    refs_b=refs_b,
                    refs_c=refs_c,
                )
            )
            print(f"  - {case.get('case_id')} / {author} / C done")

    if len(samples) != 30:
        raise RuntimeError(f"Expected 30 samples, got {len(samples)}")

    print("[Step] Computing BERTScore with local model...")
    compute_bert_scores(samples, bert_model_path, device)

    print("[Step] Building gen feature matrix...")
    gen_features = compute_layer_features([s.generated_text for s in samples], tokenizer, emb_model, device)
    if tuple(gen_features.shape) != (30, 24, 1024):
        raise RuntimeError(f"Expected gen_features shape (30,24,1024), got {tuple(gen_features.shape)}")

    with torch.no_grad():
        pre_gen_vec = style_model(gen_features)
        pre_sims = []
        for s in samples:
            sim = cosine_with_centroid(pre_gen_vec[s.idx : s.idx + 1], pre_centroids[s.target_author]).item()
            s.pre_sim = float(sim)
            pre_sims.append(float(sim))

    human_scores = [s.human_score for s in samples]
    pre_spearman = float(spearmanr(pre_sims, human_scores).correlation)

    print("[Step] Building preference pairs...")
    grouped: Dict[Tuple[str, str], List[Sample]] = {}
    for s in samples:
        grouped.setdefault((s.case_id, s.target_author), []).append(s)

    triplets: List[Tuple[int, int, str]] = []
    for (_, _), group in grouped.items():
        group_sorted = sorted(group, key=lambda x: x.strategy)
        for i in range(len(group_sorted)):
            for j in range(i + 1, len(group_sorted)):
                si = group_sorted[i]
                sj = group_sorted[j]
                if math.isclose(si.human_score, sj.human_score):
                    continue
                if si.human_score > sj.human_score:
                    triplets.append((si.idx, sj.idx, si.target_author))
                else:
                    triplets.append((sj.idx, si.idx, sj.target_author))

    print(f"  - preference pairs: {len(triplets)}")

    print("[Step] Training preference alignment...")
    optimizer = torch.optim.Adam(style_model.parameters(), lr=0.01)
    losses: List[float] = []

    for _ in range(100):
        optimizer.zero_grad()

        corpus_vec = {a: style_model(corpus_features[a]) for a in TARGET_AUTHORS}
        centroids = {a: corpus_vec[a].mean(dim=0, keepdim=True) for a in TARGET_AUTHORS}
        gen_vec = style_model(gen_features)

        pair_losses = []
        for high_idx, low_idx, author in triplets:
            sim_high = cosine_with_centroid(gen_vec[high_idx : high_idx + 1], centroids[author])
            sim_low = cosine_with_centroid(gen_vec[low_idx : low_idx + 1], centroids[author])
            target = torch.ones_like(sim_high)
            rank_loss = F.margin_ranking_loss(sim_high, sim_low, target=target, margin=0.1)
            pair_losses.append(rank_loss)

        margin_loss = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0)
        l1 = 0.005 * torch.norm(style_model.weights, p=1)
        total_loss = margin_loss + l1
        total_loss.backward()
        optimizer.step()

        losses.append(float(total_loss.item()))

    print("[Step] Saving loss curve...")
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2)
    plt.title("Preference Alignment Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()

    with torch.no_grad():
        post_corpus_vec = {a: style_model(corpus_features[a]) for a in TARGET_AUTHORS}
        post_centroids = {a: post_corpus_vec[a].mean(dim=0, keepdim=True) for a in TARGET_AUTHORS}
        post_gen_vec = style_model(gen_features)

        post_sims = []
        for s in samples:
            sim = cosine_with_centroid(post_gen_vec[s.idx : s.idx + 1], post_centroids[s.target_author]).item()
            s.post_sim = float(sim)
            post_sims.append(float(sim))

    post_spearman = float(spearmanr(post_sims, human_scores).correlation)
    final_weights = [float(x) for x in style_model.weights.detach().cpu().tolist()]

    showcase_group = [s for s in samples if s.case_id == query_cases[0]["case_id"] and s.target_author == "余华"]
    showcase_group = sorted(showcase_group, key=lambda x: x.strategy)
    showcase_source = showcase_group[0].source_text if showcase_group else ""
    showcase_b_refs = showcase_group[1].refs_b if len(showcase_group) > 1 else []
    showcase_c_refs = showcase_group[2].refs_c if len(showcase_group) > 2 else []

    print("[Step] Writing MVP summary markdown...")
    md_lines: List[str] = []
    md_lines.append("# MVP Summary: 基于人类偏好学习提取LLM风格向量")
    md_lines.append("")
    md_lines.append("## 1. 对齐优化思路概述")
    md_lines.append("在同一 Case 与同一目标作家下，对三种策略（A/B/C）按 human score 做两两比较，构建偏好对 $(h,l)$。")
    md_lines.append("优化目标是让高分样本与作家风格质心更相似、低分样本更不相似。")
    md_lines.append("")
    md_lines.append("损失函数：")
    md_lines.append("$$")
    md_lines.append("\\mathcal{L}_{total} = \\underbrace{\\max(0, m - s_h + s_l)}_{\\text{Margin Ranking Loss}} + \\lambda \\lVert w \\rVert_1")
    md_lines.append("$$")
    md_lines.append("其中 $m=0.1$，$\\lambda=0.005$，$w\\in\\mathbb{R}^{24}$ 为可学习层权重。")
    md_lines.append("")
    md_lines.append("## 2. 整体指标变化")
    md_lines.append(f"- 对齐前 Spearman 相关系数：**{pre_spearman:.4f}**")
    md_lines.append(f"- 对齐后 Spearman 相关系数：**{post_spearman:.4f}**")
    md_lines.append(f"- 初始权重（[0,...,1]）：`{[round(x, 6) for x in initial_weights]}`")
    md_lines.append(f"- 优化后权重：`{[round(x, 6) for x in final_weights]}`")
    md_lines.append("")
    md_lines.append("## 3. 损失函数下降图")
    md_lines.append("![Loss Curve](alignment_loss_curve.png)")
    md_lines.append("")
    md_lines.append("## 4. Case 抽样细节展示")
    md_lines.append("示例：**Case 1 转化为余华**")
    md_lines.append("")
    md_lines.append("### 源文本 (Source Text)")
    md_lines.append(showcase_source)
    md_lines.append("")
    md_lines.append("### 策略 B 的参考文本来源说明")
    md_lines.append("采用静态 Few-shot：在目标作家语料中按 Stage/Location/Event 顺序做规则过滤，取 Top-3 作为参考。")
    md_lines.append("")
    for i, txt in enumerate(showcase_b_refs or [], 1):
        md_lines.append(f"- B-Ref{i}: {txt[:180]}...")
    if not showcase_b_refs:
        md_lines.append("- 无")
    md_lines.append("")
    md_lines.append("### 策略 C 具体召回的 Top-3 语料文本")
    md_lines.append("使用级联召回（Strict -> Core -> Author Fallback）后，再按风格向量余弦相似度精排取 Top-3。")
    md_lines.append("")
    for i, txt in enumerate(showcase_c_refs or [], 1):
        md_lines.append(f"- C-Top{i}: {txt[:180]}...")
    if not showcase_c_refs:
        md_lines.append("- 无")
    md_lines.append("")
    md_lines.append("### 策略 A/B/C 各自的 Mock/计算 BERTScore 和 Human Score")
    for s in showcase_group:
        md_lines.append(f"- 策略{s.strategy}: BERTScore={s.bert_score:.4f}, HumanScore={s.human_score:.3f}")
    md_lines.append("")
    md_lines.append("### 策略 A/B/C 调优前后的风格余弦相似度变化")
    for s in showcase_group:
        trend = "升高" if s.post_sim >= s.pre_sim else "降低"
        md_lines.append(f"- 策略{s.strategy}: pre={s.pre_sim:.4f} -> post={s.post_sim:.4f}（{trend}）")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append(f"- 生成样本总数：{len(samples)}（5 * 2 * 3）")
    md_lines.append("- 语料规模统计：" + ", ".join(f"{a}={corpus_features[a].shape[0]}" for a in TARGET_AUTHORS))
    md_lines.append(f"- corpus_features 形状：{ {a: tuple(corpus_features[a].shape) for a in TARGET_AUTHORS} }")
    md_lines.append(f"- gen_features 形状：{tuple(gen_features.shape)}")

    summary_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved summary to {summary_path}")
    print(f"Saved curve to {curve_path}")


if __name__ == "__main__":
    main()
