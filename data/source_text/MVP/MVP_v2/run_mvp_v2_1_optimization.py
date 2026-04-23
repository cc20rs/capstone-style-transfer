from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.file_io import load_json, load_yaml
from src.utils.model_hub import configure_huggingface_mirror


TARGET_AUTHORS = ["余华", "汪曾祺"]
STRATEGIES = ["A", "B", "C"]


@dataclass
class DatasetRow:
    row_id: str
    case_id: str
    target_style: str
    strategy: str
    positive_text: str
    positive_score: float
    negative_text: str
    negative_score: float
    negative_style: str


@dataclass
class FeatureSample:
    idx: int
    row_id: str
    case_id: str
    target_style: str
    strategy: str
    generated_style: str
    polarity: str
    text: str
    human_score: float
    eval_author: str
    pre_sim: float = 0.0
    post_sim: float = 0.0


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


def load_context_single_refs(path: Path, authors: List[str]) -> Dict[str, str]:
    data = load_json(path)
    refs: Dict[str, str] = {}
    for a in authors:
        v = data.get(a)
        if isinstance(v, str) and v.strip():
            refs[a] = v.strip()
        elif isinstance(v, list):
            merged = [x.strip() for x in v if isinstance(x, str) and x.strip()]
            refs[a] = merged[0] if merged else ""
        else:
            refs[a] = ""
    return refs


def load_v2_rows(dataset_path: Path) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                case_id = str(obj["case"])
                target = str(obj["目标风格"])
                strategy = str(obj["策略"])
                pos_txt = str(obj["生成目标风格文本（正例）"])
                pos_score = float(obj["人类打分_正例"])
                neg_txt = str(obj["生成另一种风格文本（负例）"])
                neg_score = float(obj["人类打分_负例"])
                neg_style = str(obj["负例风格"])
            except KeyError as exc:
                raise RuntimeError(f"Bad dataset schema at line {line_no}: missing {exc}") from exc

            row_id = f"{case_id}_{target}_{strategy}"
            rows.append(
                DatasetRow(
                    row_id=row_id,
                    case_id=case_id,
                    target_style=target,
                    strategy=strategy,
                    positive_text=pos_txt,
                    positive_score=pos_score,
                    negative_text=neg_txt,
                    negative_score=neg_score,
                    negative_style=neg_style,
                )
            )
    return rows


def compute_layer_features(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 10,
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


def topk_rag_refs(
    author_rows: List[Dict[str, Any]],
    author_features: torch.Tensor,
    style_model: WeightedStyleModel,
    centroid: torch.Tensor,
    q_stage: Optional[str],
    q_location: Optional[str],
    q_event: Optional[str],
    top_k: int = 3,
) -> List[str]:
    cand_idx = cascade_candidate_indices(author_rows, q_stage, q_location, q_event)
    cand_feats = author_features[cand_idx]
    with torch.no_grad():
        cand_vec = style_model(cand_feats)
        sims = cosine_with_centroid(cand_vec, centroid)
        top_pos = torch.topk(sims, k=min(top_k, sims.numel()), dim=0).indices.tolist()
    return [author_rows[cand_idx[p]]["content"] for p in top_pos]


def parse_spearman(summary_text: str) -> Tuple[Optional[float], Optional[float]]:
    pre_m = re.search(r"对齐前 Spearman(?: 相关系数)?[:：]\*\*([\-0-9.]+)\*\*", summary_text)
    post_m = re.search(r"对齐后 Spearman(?: 相关系数)?[:：]\*\*([\-0-9.]+)\*\*", summary_text)
    pre = float(pre_m.group(1)) if pre_m else None
    post = float(post_m.group(1)) if post_m else None
    return pre, post


def main() -> None:
    torch.manual_seed(20260413)

    cfg = load_yaml(ROOT / "configs" / "config.yaml")
    configure_huggingface_mirror(cfg)

    source_data = load_json(ROOT / "data" / "source_text" / "source_text.json")
    corpus_data = load_json(ROOT / "data" / "source_text" / "corpus_text.json")
    scene_md = (ROOT / "data" / "source_text" / "scene_and_recall.md").read_text(encoding="utf-8")
    _ = parse_allowed_values(scene_md)

    mvp_v2_dir = ROOT / "data" / "source_text" / "MVP" / "MVP_v2"
    dataset_path = mvp_v2_dir / "synth_dataset_v2.jsonl"
    curve_path = mvp_v2_dir / "alignment_loss_curve_v2_1.png"
    summary_path = mvp_v2_dir / "MVP_v2.1_summary.md"

    v1_summary_path = ROOT / "data" / "source_text" / "MVP" / "MVP_summary.md"
    v2_summary_path = ROOT / "data" / "source_text" / "MVP" / "MVP_v2" / "MVP_v2_summary.md"
    context_ref_path = ROOT / "data" / "source_text" / "data_to_generate" / "context_eng_references.json"

    emb_model_path = resolve_local_model_path(cfg, "style_embedding_model")

    rows = load_v2_rows(dataset_path)
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 rows in v2 dataset, got {len(rows)}")

    query_cases = source_data.get("cases", [])[:5]
    source_map = {str(c["case_id"]): str(c.get("source_text", "")) for c in query_cases}
    case_meta = {str(c["case_id"]): c for c in query_cases}

    corpus_rows = corpus_data.get("corpus", [])
    corpus_by_author: Dict[str, List[Dict[str, Any]]] = {
        a: [r for r in corpus_rows if r.get("author") == a] for a in TARGET_AUTHORS
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[V2.1] Loading local embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
    emb_model = AutoModel.from_pretrained(emb_model_path).to(device)

    print("[V2.1] Building corpus features...")
    corpus_features: Dict[str, torch.Tensor] = {}
    for author in TARGET_AUTHORS:
        texts = [r["content"] for r in corpus_by_author[author]]
        feats = compute_layer_features(texts, tokenizer, emb_model, device)
        corpus_features[author] = feats
        print(f"  - {author}: N={feats.shape[0]}, feature_shape={tuple(feats.shape)}")

    # Build samples from existing dataset (no re-generation).
    samples: List[FeatureSample] = []
    for r in rows:
        samples.append(
            FeatureSample(
                idx=len(samples),
                row_id=r.row_id,
                case_id=r.case_id,
                target_style=r.target_style,
                strategy=r.strategy,
                generated_style=r.target_style,
                polarity="pos",
                text=r.positive_text,
                human_score=r.positive_score,
                eval_author=r.target_style,
            )
        )
        samples.append(
            FeatureSample(
                idx=len(samples),
                row_id=r.row_id,
                case_id=r.case_id,
                target_style=r.target_style,
                strategy=r.strategy,
                generated_style=r.negative_style,
                polarity="neg",
                text=r.negative_text,
                human_score=r.negative_score,
                eval_author=r.target_style,
            )
        )

    if len(samples) != 60:
        raise RuntimeError(f"Expected 60 samples, got {len(samples)}")

    style_model = WeightedStyleModel()
    initial_weights = style_model.weights.detach().cpu().tolist()

    with torch.no_grad():
        pre_corpus_vec = {a: style_model(corpus_features[a]) for a in TARGET_AUTHORS}
        pre_centroids = {a: pre_corpus_vec[a].mean(dim=0, keepdim=True) for a in TARGET_AUTHORS}

    print("[V2.1] Building gen feature matrix from existing dataset...")
    gen_features = compute_layer_features([s.text for s in samples], tokenizer, emb_model, device)
    if tuple(gen_features.shape) != (60, 24, 1024):
        raise RuntimeError(f"Expected gen_features shape (60,24,1024), got {tuple(gen_features.shape)}")

    with torch.no_grad():
        pre_gen_vec = style_model(gen_features)
        pre_sims = []
        for s in samples:
            sim = cosine_with_centroid(pre_gen_vec[s.idx : s.idx + 1], pre_centroids[s.eval_author]).item()
            s.pre_sim = float(sim)
            pre_sims.append(float(sim))

    human_scores = [s.human_score for s in samples]
    pre_spearman = float(spearmanr(pre_sims, human_scores).correlation)

    print("[V2.1] Building expanded preference pairs...")
    pos_samples = [s for s in samples if s.polarity == "pos"]
    neg_samples = [s for s in samples if s.polarity == "neg"]

    by_case_target_pos: Dict[Tuple[str, str], List[FeatureSample]] = {}
    by_case_target_neg: Dict[Tuple[str, str], List[FeatureSample]] = {}
    by_target_pos: Dict[str, List[FeatureSample]] = {a: [] for a in TARGET_AUTHORS}
    by_target_neg: Dict[str, List[FeatureSample]] = {a: [] for a in TARGET_AUTHORS}

    for s in pos_samples:
        key = (s.case_id, s.target_style)
        by_case_target_pos.setdefault(key, []).append(s)
        by_target_pos[s.target_style].append(s)

    for s in neg_samples:
        key = (s.case_id, s.target_style)
        by_case_target_neg.setdefault(key, []).append(s)
        by_target_neg[s.target_style].append(s)

    pos_pos_intra: List[Tuple[int, int, str]] = []
    for key, group in by_case_target_pos.items():
        _ = key
        group_sorted = sorted(group, key=lambda x: x.strategy)
        for i in range(len(group_sorted)):
            for j in range(i + 1, len(group_sorted)):
                si = group_sorted[i]
                sj = group_sorted[j]
                if math.isclose(si.human_score, sj.human_score):
                    continue
                if si.human_score > sj.human_score:
                    pos_pos_intra.append((si.idx, sj.idx, si.target_style))
                else:
                    pos_pos_intra.append((sj.idx, si.idx, sj.target_style))

    # Global positive-positive pairs inside each target author (larger set).
    pos_pos_global: List[Tuple[int, int, str]] = []
    for author in TARGET_AUTHORS:
        g = by_target_pos[author]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                si = g[i]
                sj = g[j]
                if abs(si.human_score - sj.human_score) < 0.2:
                    continue
                if si.human_score > sj.human_score:
                    pos_pos_global.append((si.idx, sj.idx, author))
                else:
                    pos_pos_global.append((sj.idx, si.idx, author))

    # For each case+target: each positive can pair against all three negatives (3x3 = 9).
    pos_neg_case_cross: List[Tuple[int, int, str]] = []
    for key, pos_group in by_case_target_pos.items():
        neg_group = by_case_target_neg.get(key, [])
        for p in pos_group:
            for n in neg_group:
                if p.human_score <= n.human_score:
                    continue
                pos_neg_case_cross.append((p.idx, n.idx, p.target_style))

    # Hard negatives from same target but different rows.
    pos_neg_hard: List[Tuple[int, int, str]] = []
    for author in TARGET_AUTHORS:
        neg_sorted = sorted(by_target_neg[author], key=lambda x: x.human_score, reverse=True)
        for p in by_target_pos[author]:
            for n in neg_sorted[:3]:
                if p.row_id == n.row_id:
                    continue
                if p.human_score <= n.human_score:
                    continue
                pos_neg_hard.append((p.idx, n.idx, author))

    all_pairs = pos_pos_intra + pos_pos_global + pos_neg_case_cross + pos_neg_hard
    if not all_pairs:
        raise RuntimeError("No preference pairs were built for v2.1")

    print(
        "  - pair stats: "
        f"pos_pos_intra={len(pos_pos_intra)}, "
        f"pos_pos_global={len(pos_pos_global)}, "
        f"pos_neg_case_cross={len(pos_neg_case_cross)}, "
        f"pos_neg_hard={len(pos_neg_hard)}, "
        f"total={len(all_pairs)}"
    )

    author_to_idx = {a: i for i, a in enumerate(TARGET_AUTHORS)}
    pair_high_idx = torch.tensor([p[0] for p in all_pairs], dtype=torch.long)
    pair_low_idx = torch.tensor([p[1] for p in all_pairs], dtype=torch.long)
    pair_author_idx = torch.tensor([author_to_idx[p[2]] for p in all_pairs], dtype=torch.long)

    print("[V2.1] Training with scheduler + early stopping + anti-oscillation...")
    optimizer = torch.optim.Adam(style_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    max_epochs = 200
    min_delta = 1e-5
    patience = 20
    wait = 0
    best_epoch = 0
    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {}

    losses: List[float] = []

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()

        corpus_vec = {a: style_model(corpus_features[a]) for a in TARGET_AUTHORS}
        centroids = {a: corpus_vec[a].mean(dim=0, keepdim=True) for a in TARGET_AUTHORS}
        centroid_mat = torch.cat([centroids[a] for a in TARGET_AUTHORS], dim=0)
        gen_vec = style_model(gen_features)

        gen_norm = F.normalize(gen_vec, dim=-1)
        centroid_norm = F.normalize(centroid_mat, dim=-1)
        high_vec = gen_norm[pair_high_idx]
        low_vec = gen_norm[pair_low_idx]
        pair_centroid = centroid_norm[pair_author_idx]

        sim_high = (high_vec * pair_centroid).sum(dim=-1)
        sim_low = (low_vec * pair_centroid).sum(dim=-1)
        margin_loss = F.relu(0.1 - (sim_high - sim_low)).mean()
        l1 = 0.003 * torch.norm(style_model.weights, p=1)
        total_loss = margin_loss + l1
        total_loss.backward()

        # Anti-oscillation #1: clip gradient.
        torch.nn.utils.clip_grad_norm_(style_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Anti-oscillation #2: clamp parameter range.
        with torch.no_grad():
            style_model.weights.data.clamp_(-0.5, 1.5)

        scheduler.step()

        loss_val = float(total_loss.item())
        losses.append(loss_val)

        if epoch % 25 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  - epoch={epoch}, loss={loss_val:.6f}, lr={lr_now:.6e}")

        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in style_model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"  - early stop at epoch={epoch}, best_epoch={best_epoch}, best_loss={best_loss:.6f}")
            break

    if best_state:
        style_model.load_state_dict(best_state)

    trained_epochs = len(losses)

    print("[V2.1] Saving loss curve...")
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, trained_epochs + 1), losses, linewidth=2)
    plt.title("MVP v2.1 Preference Alignment Loss Curve")
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
            sim = cosine_with_centroid(post_gen_vec[s.idx : s.idx + 1], post_centroids[s.eval_author]).item()
            s.post_sim = float(sim)
            post_sims.append(float(sim))

    post_spearman = float(spearmanr(post_sims, human_scores).correlation)
    final_weights = [float(x) for x in style_model.weights.detach().cpu().tolist()]

    v1_pre = None
    v1_post = None
    if v1_summary_path.exists():
        v1_text = v1_summary_path.read_text(encoding="utf-8")
        v1_pre, v1_post = parse_spearman(v1_text)

    v2_pre = None
    v2_post = None
    if v2_summary_path.exists():
        v2_text = v2_summary_path.read_text(encoding="utf-8")
        v2_pre, v2_post = parse_spearman(v2_text)

    context_single_refs = load_context_single_refs(context_ref_path, TARGET_AUTHORS)

    showcase_case = query_cases[0]["case_id"]
    showcase_target = "余华"
    showcase_rows = [r for r in rows if r.case_id == showcase_case and r.target_style == showcase_target]
    showcase_rows = sorted(showcase_rows, key=lambda x: x.strategy)

    case_obj = case_meta.get(showcase_case, {})
    q_stage = case_obj.get("stage")
    q_location = case_obj.get("location")
    q_event = case_obj.get("event")
    refs_c_target = topk_rag_refs(
        author_rows=corpus_by_author[showcase_target],
        author_features=corpus_features[showcase_target],
        style_model=style_model,
        centroid=post_centroids[showcase_target],
        q_stage=q_stage,
        q_location=q_location,
        q_event=q_event,
        top_k=3,
    )

    md: List[str] = []
    md.append("# MVP v2.1 Summary: 扩展偏好对与稳健训练")
    md.append("")
    md.append("## 1. 实验目的与v2.1改动")
    md.append("- 基于既有 `synth_dataset_v2.jsonl` 直接训练，不重新合成文本。")
    md.append("- 扩展偏好对构建：增加正例之间偏好对、正负交叉偏好对，提升监督强度。")
    md.append("- 优化训练稳定性：降低学习率、加入学习率衰减、早停、梯度裁剪与权重约束。")
    md.append("")
    md.append("## 2. 数据格式与规模")
    md.append("目标格式：")
    md.append("`case | 目标风格 | 策略 | 生成目标风格文本（正例） | 人类打分 | 生成另一种风格文本（负例） | 人类打分`")
    md.append("")
    md.append(f"- 数据行数：{len(rows)}")
    md.append(f"- 生成样本数：{len(samples)}（每行正负各1）")
    md.append(f"- 数据文件：`{dataset_path.relative_to(ROOT).as_posix()}`")
    md.append("")
    md.append("## 3. 偏好对构建（扩展版）")
    md.append("- 正例之间偏好对（同 case）：A/B/C 两两比较。")
    md.append("- 正例之间偏好对（同目标风格全局）：跨 case 采样，扩大正例排序信号。")
    md.append("- 正负交叉偏好对（同 case）：每个正例分别与三个负例构建偏好对（3x3）。")
    md.append("- 正负困难对（同目标风格全局）：正例对比高分负例，增加 hard pairs。")
    md.append("")
    md.append(f"- pos_pos_intra={len(pos_pos_intra)}")
    md.append(f"- pos_pos_global={len(pos_pos_global)}")
    md.append(f"- pos_neg_case_cross={len(pos_neg_case_cross)}")
    md.append(f"- pos_neg_hard={len(pos_neg_hard)}")
    md.append(f"- 总偏好对={len(all_pairs)}")
    md.append("")
    md.append("## 4. 指标变化（v2.1）")
    md.append(f"- v2.1 对齐前 Spearman 相关系数：**{pre_spearman:.4f}**")
    md.append(f"- v2.1 对齐后 Spearman 相关系数：**{post_spearman:.4f}**")
    md.append(f"- v2.1 调优增益：`{post_spearman - pre_spearman:+.4f}`")
    md.append(f"- 训练轮次：{trained_epochs}（best_epoch={best_epoch}, early_stopping_patience={patience}）")
    md.append(f"- 初始权重（[0,...,1]）：`{[round(x, 6) for x in initial_weights]}`")
    md.append(f"- 优化后权重：`{[round(x, 6) for x in final_weights]}`")
    md.append("")
    md.append("## 5. 三次实验对比（v1 / v2 / v2.1）")
    if v1_pre is not None and v1_post is not None:
        md.append(f"- v1: pre={v1_pre:.4f}, post={v1_post:.4f}, gain={v1_post - v1_pre:+.4f}")
    else:
        md.append("- v1: 未解析到有效指标")

    if v2_pre is not None and v2_post is not None:
        md.append(f"- v2: pre={v2_pre:.4f}, post={v2_post:.4f}, gain={v2_post - v2_pre:+.4f}")
    else:
        md.append("- v2: 未解析到有效指标")

    md.append(f"- v2.1: pre={pre_spearman:.4f}, post={post_spearman:.4f}, gain={post_spearman - pre_spearman:+.4f}")

    if v2_post is not None:
        md.append(f"- v2.1 相对 v2 后置变化：`{post_spearman - v2_post:+.4f}`")
    if v1_post is not None:
        md.append(f"- v2.1 相对 v1 后置变化：`{post_spearman - v1_post:+.4f}`")

    md.append("")
    md.append("## 6. Case 抽样展示（Case 1 -> 余华）")
    md.append("### 源文本")
    md.append(source_map[str(showcase_case)])
    md.append("")
    md.append("### 策略B参考来源说明")
    md.append("- 目标风格单条参考（按作家名直接读取）：")
    md.append(f"  - B-Target: {context_single_refs.get(showcase_target, '')[:200]}...")
    neg_author = "汪曾祺" if showcase_target == "余华" else "余华"
    md.append("- 负例风格单条参考（按作家名直接读取）：")
    md.append(f"  - B-Negative: {context_single_refs.get(neg_author, '')[:200]}...")
    md.append("")
    md.append("### 策略C Top-3 召回示例（目标风格）")
    for i, txt in enumerate(refs_c_target, 1):
        md.append(f"- C-Top{i}: {txt[:180]}...")
    if not refs_c_target:
        md.append("- 无")
    md.append("")
    md.append("### 各策略正负打分")
    for r in showcase_rows:
        md.append(
            f"- 策略{r.strategy}: 正例分={r.positive_score:.3f}, "
            f"负例分={r.negative_score:.3f}, 负例风格={r.negative_style}"
        )
    md.append("")
    md.append("## 7. 损失函数下降图")
    md.append("![Loss Curve](alignment_loss_curve_v2_1.png)")
    md.append("")
    md.append("## 8. 防震荡措施与结论")
    md.append("- 防震荡措施：学习率降至 1e-3、CosineAnnealingLR 衰减、梯度裁剪（max_norm=1.0）、权重范围约束（clamp）、早停并回滚最佳权重。")
    if post_spearman >= pre_spearman:
        md.append("- 本次 v2.1 训练后相关性提升，说明扩展偏好对与稳健训练策略有效。")
    else:
        md.append("- 本次 v2.1 训练后相关性未提升，说明仍需继续调整偏好对组成与损失系数。")
    md.append("- 与 v2 相比，v2.1 重点在“数据监督增强 + 训练稳定性增强”的联合优化。")
    md.append("")
    md.append("---")
    md.append("- 语料规模统计：" + ", ".join(f"{a}={corpus_features[a].shape[0]}" for a in TARGET_AUTHORS))
    md.append(f"- corpus_features 形状：{ {a: tuple(corpus_features[a].shape) for a in TARGET_AUTHORS} }")
    md.append(f"- gen_features 形状：{tuple(gen_features.shape)}")

    summary_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[V2.1] Saved summary to {summary_path}")
    print(f"[V2.1] Saved curve to {curve_path}")


if __name__ == "__main__":
    main()
