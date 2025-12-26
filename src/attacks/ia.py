# 文件: src/attacks/ia.py

from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import torch
import faiss
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics import roc_auc_score, accuracy_score

from attacks.ia_utils import (
    generate_interrogation_materials,
    generate_ground_truths,
    compute_ia_score
)
from attacks.ia_pipeline import get_rag_responses
from attacks.s2mia_splits_and_index import load_full_mapping


# ==================== 辅助函数 (这部分无变化) ====================

def _load_cache(path: str) -> Dict[str, Any]:
    """加载缓存文件"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: Dict[str, Any]):
    """保存缓存文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_splits(splits_path: str) -> Dict[str, Any]:
    """加载数据集划分配置"""
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rag_cache_key(doc_id: str, defense: bool, defense_mode: str, num_questions: int) -> str:
    """生成 RAG 响应缓存键"""
    defense_str = f"def_{defense_mode}" if defense else "no_def"
    return f"ia_rag_{doc_id}_nq{num_questions}_{defense_str}_v2"


def _find_best_threshold(
        scores: List[float],
        labels: List[int]
) -> Tuple[float, float]:
    """在参考集上搜索最佳阈值（基于准确率）"""
    if not scores: return 0.0, 0.0
    min_score, max_score = min(scores), max(scores)
    thresholds = np.linspace(min_score, max_score, 101)
    best_acc, best_threshold = 0.0, 0.0
    for threshold in thresholds:
        predictions = [1 if s >= threshold else 0 for s in scores]
        acc = accuracy_score(labels, predictions)
        if acc > best_acc:
            best_acc, best_threshold = acc, threshold
    return best_threshold, best_acc


def _compute_metrics(
        scores: List[float],
        labels: List[int],
        threshold: float
) -> Dict[str, float]:
    """计算评估指标"""
    predictions = [1 if s >= threshold else 0 for s in scores]
    acc = accuracy_score(labels, predictions)
    member_correct = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    non_member_correct = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    member_total = sum(labels)
    non_member_total = len(labels) - member_total
    member_acc = member_correct / member_total if member_total > 0 else 0
    non_member_acc = non_member_correct / non_member_total if non_member_total > 0 else 0
    adj_acc = (member_acc + non_member_acc) / 2 - 0.5
    roc_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0
    return {"acc": acc, "adj_acc": adj_acc, "roc_auc": roc_auc, "member_acc": member_acc, "non_member_acc": non_member_acc}


def _export_results(
        out_dir: str,
        metrics: Dict[str, Any],
        details: List[Dict[str, Any]]
):
    """导出实验结果"""
    os.makedirs(out_dir, exist_ok=True)

    #统计 Recall
    print("\n" + "=" * 40)
    print("      Recall Statistics (Members Only)      ")
    print("=" * 40)
    member_details = [d for d in details if d['label'] == 1]
    recall_summary = {}

    if member_details:
        # 检查是否有 recall 字段
        if "recall@1" in member_details[0]:
            for k in [1, 2, 3]:
                key = f"recall@{k}"
                hits = [d.get(key, 0) for d in member_details]
                avg_recall = float(np.mean(hits))
                recall_summary[key] = avg_recall
                print(f"  > Session Recall@{k}: {avg_recall:.2%} ({sum(hits)}/{len(hits)})")
                # Note: Session Recall@k 意思是：在该文档的 N 个问题中，是否有任意一个问题在 Top-k 中找到了该文档
        else:
            print("  [Info] No recall data found in evaluation details.")
    else:
        print("  [Warn] No members in evaluation set.")

    metrics["recall_stats"] = recall_summary
    metrics_file = os.path.join(out_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    details_file = os.path.join(out_dir, "details.jsonl")
    with open(details_file, "w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[Results] Saved to {out_dir}/")

def _calculate_session_recall(target_doc_id: str, all_retrieved_ids_per_query: List[List[int]]) -> Dict[str, int]:
    """
    计算会话级召回率。
    如果在 N 个问题中，有任意一个问题在 Top-K 中检索到了目标 ID，则对应的 Recall@K 为 1。
    """
    target_id_int = int(target_doc_id)

    # 初始化最小排名为无穷大 (越小越好，0表示Rank 1)
    min_rank_found = 999

    found_at_all = False

    for ids in all_retrieved_ids_per_query:
        if target_id_int in ids:
            found_at_all = True
            rank = ids.index(target_id_int)  # 0-based index
            if rank < min_rank_found:
                min_rank_found = rank

    # 如果没找到，min_rank_found 保持 999

    return {
        "recall@1": 1 if min_rank_found < 1 else 0,
        "recall@2": 1 if min_rank_found < 2 else 0,
        "recall@3": 1 if min_rank_found < 3 else 0
    }


# ==================== 主流程 ====================

def run_ia_attack(args):

    from rag.generate_llamaindex import PADConfig

    """运行 IA 攻击的主流程"""

    # ... (前面的打印和数据加载部分无变化)
    print("=" * 80)
    print("IA (Interrogation Attack) with Cross-Encoder Reranking")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Questions Generate: {args.num_questions_generate}")
    print(f"  - Questions Select: {args.num_questions_select}")
    print(f"  - Use Cross-Encoder: {args.use_cross_encoder}")
    print(f"  - Lambda Penalty: {args.lambda_penalty}")
    print(f"  - Defense: {args.defense}")
    if args.defense:
        print(f"  - Defense Mode: {args.defense_mode}")
    print(f"  - Reference Set: {args.n_ref_m} members + {args.n_ref_n} non-members")
    print(f"  - Evaluation Set: {args.n_eval_m} members + {args.n_eval_n} non-members")
    print("=" * 80)

    # 1. 加载数据和配置
    print("\n[Step 1] Loading data...")

    splits = _load_splits(args.splits_path)
    doc_mapping = load_full_mapping("index")
    doc_list = list(doc_mapping.items())
    def indices_to_doc_ids(indices):
        return [str(i) for i in indices if str(i) in doc_mapping]
    m_ref_doc_ids = indices_to_doc_ids(splits["member_ref"][:args.n_ref_m])
    n_ref_doc_ids = indices_to_doc_ids(splits["non_ref"][:args.n_ref_n])
    m_eval_doc_ids = indices_to_doc_ids(splits["member_eval"][:args.n_eval_m])
    n_eval_doc_ids = indices_to_doc_ids(splits["non_eval"][:args.n_eval_n])

    print(f"  - Loaded {len(doc_mapping)} documents")
    print(f"  - Reference: {len(m_ref_doc_ids)} members, {len(n_ref_doc_ids)} non-members")
    print(f"  - Evaluation: {len(m_eval_doc_ids)} members, {len(n_eval_doc_ids)} non-members")

    cache = _load_cache(args.cache_path)
    print(f"  - Loaded {len(cache)} cached entries")

    # 2. 预加载模型和索引
    print("\n[Step 1.5] Pre-loading models and indexes...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_bge_path = "/root/autodl-tmp/pycharm_Mirabel/models/BAAI/bge-m3"
    print(f"  - Loading BGE-M3 from: {local_bge_path}")
    embedder = BGEM3FlagModel(local_bge_path, use_fp16=True, device=device)
    print(f"  - Loaded embedder on {device}")
    member_index_path = os.path.join(args.index_dir_member, "nf_member.index")
    member_map_path = os.path.join(args.index_dir_member, "nf_member_docs.json")
    member_index = faiss.read_index(member_index_path)
    with open(member_map_path, "r", encoding="utf-8") as f:
        member_mapping = json.load(f)
    print(f"  - Loaded member index: {member_index.ntotal} vectors")

    # 构建 Mirabel 配置
    mirabel_cfg = None
    if args.defense and args.defense_mode == 'mirabel':
        mirabel_cfg = {"rho": args.rho, "margin": args.margin, "gap_min": args.gap_min, "use_gap": args.use_gap}
        print(f"  - Mirabel defense configured")
    elif args.defense and args.defense_mode == 'prompt_guard':
        print(f"  - Prompt Guard defense enabled")
    elif args.defense_mode == 'rewrite':
        print(f"  - Rewrite-based defense enabled")

    # 构建 pad 配置
    pad_cfg = None
    pad_model_id_or_path = None
    if args.defense and args.defense_mode == "pad":
        pad_cfg = PADConfig(
            epsilon_base=args.pad_epsilon_base,
            delta=args.pad_delta,
            alpha=args.pad_alpha,
            lambda_amp=args.pad_lambda_amp,
            tau_conf=args.pad_tau_conf,
            tau_margin=args.pad_tau_margin,
            delta_min=args.pad_delta_min,
            sigma_min=args.pad_sigma_min,
            w_entropy=args.pad_w_entropy,
            w_pos=args.pad_w_pos,
            w_conf=args.pad_w_conf,
            pos_k=args.pad_pos_k,
            max_new_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        pad_model_id_or_path = args.pad_model_id_or_path or os.getenv("PAD_MODEL_ID", "")

    # 3. 生成材料 (无变化)
    print("\n[Step 2] Generating interrogation materials...")
    all_doc_ids = list(set(m_ref_doc_ids + n_ref_doc_ids + m_eval_doc_ids + n_eval_doc_ids))
    materials = {}
    for doc_id in tqdm(all_doc_ids, desc="Generating materials"):
        doc_dict = doc_mapping.get(doc_id)
        if not doc_dict: continue
        doc_text = (doc_dict.get("title", "") + " " + doc_dict.get("text", "")).strip()
        if not doc_text: continue
        summary, questions = generate_interrogation_materials(
            doc_id=doc_id, doc_text=doc_text, num_questions_generate=args.num_questions_generate,
            num_questions_select=args.num_questions_select, use_cross_encoder=args.use_cross_encoder, cache=cache)
        ground_truths = generate_ground_truths(
            doc_id=doc_id, doc_text=doc_text, summary=summary, questions=questions, cache=cache)
        materials[doc_id] = {"summary": summary, "questions": questions, "ground_truths": ground_truths}
        if len(materials) % args.save_every == 0:
            _save_cache(args.cache_path, cache)
    _save_cache(args.cache_path, cache)
    print(f"  - Generated materials for {len(materials)} documents")

    # 4. 处理参考集
    print("\n[Step 3] Processing reference set...")
    ref_scores, ref_labels = [], []
    ref_tasks = [(doc_id, 1) for doc_id in m_ref_doc_ids] + [(doc_id, 0) for doc_id in n_ref_doc_ids]
    for doc_id, label in tqdm(ref_tasks, desc="Reference set"):
        if doc_id not in materials: continue
        mat = materials[doc_id]
        questions, ground_truths = mat["questions"], mat["ground_truths"]
        rag_cache_key = _rag_cache_key(doc_id, args.defense, args.defense_mode, len(questions))

        parsed_responses = []
        debug_logs = [] # 参考集通常不需要详细日志，但为了代码一致性可以接收一下
        if cache and rag_cache_key in cache:
            parsed_responses = cache[rag_cache_key]["parsed_responses"]
        else:
            # ✅ [核心修改] 修正参数传递逻辑
            raw_responses, parsed_responses, all_retrieved_ids, debug_logs = get_rag_responses(
                queries=questions,
                defense=args.defense,
                defense_mode=args.defense_mode,
                pad_cfg=pad_cfg,
                pad_model_id_or_path=pad_model_id_or_path,
                oracle_context=None,
                # 始终传递核心RAG组件
                index=member_index,
                mapping=member_mapping,
                embedder=embedder,
                # 仅在需要时传递Mirabel特定参数
                index_dir_member=args.index_dir_member,
                mirabel_cfg=mirabel_cfg,
                top_k=args.top_k,
                per_ctx_clip=args.per_ctx_clip,
                max_tokens=args.max_tokens
            )

            recall_stats = _calculate_session_recall(doc_id, all_retrieved_ids)

            cache[rag_cache_key] = {
                "parsed_responses": parsed_responses,
                "recall": recall_stats,  # 保存 Recall
                "debug_logs": debug_logs # 如果想省空间，参考集可以不存 logs
            }
        if len(ref_scores) % 10 == 0 and len(ref_scores) > 0:
            _save_cache(args.cache_path, cache)
        score = compute_ia_score(parsed_responses, ground_truths, args.lambda_penalty)
        ref_scores.append(score)
        ref_labels.append(label)
    _save_cache(args.cache_path, cache)

    best_threshold, _ = _find_best_threshold(ref_scores, ref_labels)
    ref_metrics = _compute_metrics(ref_scores, ref_labels, best_threshold)
    print(f"\n  Reference Set Results:")
    print(f"    - Best Threshold: {best_threshold:.4f}")
    print(f"    - Accuracy: {ref_metrics['acc']:.4f}")
    print(f"    - Adjusted Accuracy: {ref_metrics['adj_acc']:.4f}")
    print(f"    - ROC AUC: {ref_metrics['roc_auc']:.4f}")

    # 5. 处理评估集
    print("\n[Step 4] Processing evaluation set...")
    eval_scores, eval_labels, eval_details = [], [], []
    eval_tasks = [(doc_id, 1) for doc_id in m_eval_doc_ids] + [(doc_id, 0) for doc_id in n_eval_doc_ids]
    for doc_id, label in tqdm(eval_tasks, desc="Evaluation set"):
        if doc_id not in materials: continue
        mat = materials[doc_id]
        questions, ground_truths = mat["questions"], mat["ground_truths"]
        rag_cache_key = _rag_cache_key(doc_id, args.defense, args.defense_mode, len(questions))

        parsed_responses = []
        recall_stats = {}
        qa_logs = [] # <--- [新增] 用于存放该文档的所有问答日志
        if cache and rag_cache_key in cache:
            entry = cache[rag_cache_key]
            parsed_responses = entry["parsed_responses"]
            recall_stats = entry.get("recall", {}) # 读取缓存的 recall
            qa_logs = entry.get("debug_logs", [])  # <--- [新增] 尝试从缓存读取日志
        else:
            # ✅ [核心修改] 修正参数传递逻辑
            raw_responses, parsed_responses, all_retrieved_ids, qa_logs = get_rag_responses(
                queries=questions,
                defense=args.defense,
                defense_mode=args.defense_mode,
                pad_cfg=pad_cfg,
                pad_model_id_or_path=pad_model_id_or_path,
                oracle_context=None,
                # 始终传递核心RAG组件
                index=member_index,
                mapping=member_mapping,
                embedder=embedder,
                # 仅在需要时传递Mirabel特定参数
                index_dir_member=args.index_dir_member,
                mirabel_cfg=mirabel_cfg,
                top_k=args.top_k,
                per_ctx_clip=args.per_ctx_clip,
                max_tokens=args.max_tokens
            )

            recall_stats = _calculate_session_recall(doc_id, all_retrieved_ids)

            cache[rag_cache_key] = {
                "parsed_responses": parsed_responses,
                "recall": recall_stats,
                "debug_logs": qa_logs  # <--- [新增] 将日志存入缓存
            }
        if len(eval_scores) % 10 == 0 and len(eval_scores) > 0:
            _save_cache(args.cache_path, cache)
        score = compute_ia_score(parsed_responses, ground_truths, args.lambda_penalty)
        eval_scores.append(score)
        eval_labels.append(label)
        # 构建详情字典，包含 recall
        detail_item = {
            "doc_id": doc_id,
            "label": label,
            "score": score,
            "num_questions": len(questions),
            "prediction": 1 if score >= best_threshold else 0,
            "qa_logs": qa_logs  # <--- [核心修改] 将完整日志放入 details 中
        }
        detail_item.update(recall_stats) # 合并 Recall
        eval_details.append(detail_item)
    _save_cache(args.cache_path, cache)

    eval_metrics = _compute_metrics(eval_scores, eval_labels, best_threshold)
    print(f"\n  Evaluation Set Results:")
    print(f"    - Accuracy: {eval_metrics['acc']:.4f}")
    print(f"    - Adjusted Accuracy: {eval_metrics['adj_acc']:.4f}")
    print(f"    - ROC AUC: {eval_metrics['roc_auc']:.4f}")

    # 6. 导出结果 (无变化)
    if args.out_dir:
        print("\n[Step 5] Exporting results...")
        final_metrics = {
            "attack": "IA", "defense": args.defense,
            "defense_mode": args.defense_mode if args.defense else "none",
            "use_cross_encoder": args.use_cross_encoder,
            "num_questions_generate": args.num_questions_generate,
            "num_questions_select": args.num_questions_select,
            "lambda_penalty": args.lambda_penalty,
            "best_threshold": best_threshold,
            "reference": _compute_metrics(ref_scores, ref_labels, best_threshold),
            "evaluation": eval_metrics,
            "params": vars(args)}
        _export_results(args.out_dir, final_metrics, eval_details)

    print("\n" + "=" * 80)
    print("IA Attack Completed!")
    print("=" * 80)


# ==================== 命令行参数 (无变化) ====================

def parse_args():
    parser = argparse.ArgumentParser(description="IA (Interrogation Attack) with Cross-Encoder Reranking")
    parser.add_argument("--splits_path", type=str, default="configs/s2mia_splits.json")
    parser.add_argument("--index_dir_member", type=str, default="index_member")
    parser.add_argument("--n_ref_m", type=int, default=50)
    parser.add_argument("--n_ref_n", type=int, default=50)
    parser.add_argument("--n_eval_m", type=int, default=100)
    parser.add_argument("--n_eval_n", type=int, default=100)
    parser.add_argument("--num_questions_generate", type=int, default=50)
    parser.add_argument("--num_questions_select", type=int, default=30)
    parser.add_argument("--use_cross_encoder", action="store_true", default=True)
    parser.add_argument("--no_cross_encoder", action="store_false", dest="use_cross_encoder")
    parser.add_argument("--lambda_penalty", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--per_ctx_clip", type=int, default=600)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--defense", action="store_true", help="Enable defense mechanism.")
    parser.add_argument(
        "--defense_mode", type=str, default="mirabel", choices=["mirabel", "prompt_guard", "rewrite", "pad"],
        help="Specify the defense mode to use when --defense is enabled.")
    parser.add_argument("--rho", type=float, default=0.005)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--gap_min", type=float, default=0.03)
    parser.add_argument("--use_gap", action="store_true")
    parser.add_argument("--cache_path", type=str, default="cache/ia_cross_encoder.json")
    parser.add_argument("--out_dir", type=str, default="results/ia_cross_encoder")
    parser.add_argument("--save_every", type=int, default=20)

    # ===========================================
    # pad防御方式相关参数
    # ===========================================
    parser.add_argument("--pad_model_id_or_path", type=str, default="")
    parser.add_argument("--pad_epsilon_base", type=float, default=0.2)
    parser.add_argument("--pad_delta", type=float, default=1e-5)
    parser.add_argument("--pad_alpha", type=float, default=10.0)
    parser.add_argument("--pad_lambda_amp", type=float, default=3.0)
    parser.add_argument("--pad_tau_conf", type=float, default=0.90)
    parser.add_argument("--pad_tau_margin", type=float, default=1.00)
    parser.add_argument("--pad_delta_min", type=float, default=0.4)
    parser.add_argument("--pad_sigma_min", type=float, default=0.01)
    parser.add_argument("--pad_w_entropy", type=float, default=0.3)
    parser.add_argument("--pad_w_pos", type=float, default=0.2)
    parser.add_argument("--pad_w_conf", type=float, default=0.5)
    parser.add_argument("--pad_pos_k", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    run_ia_attack(args)


if __name__ == "__main__":
    main()
