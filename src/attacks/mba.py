# 文件: src/attacks/mba.py

from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm
import faiss
from FlagEmbedding import BGEM3FlagModel

from attacks.mba_pipeline import (
    generate_mba_no_defense,
    generate_mba_with_defense,
    generate_mba_with_prompt_guard_defense,
    generate_mba_with_rewrite_defense,
    generate_mba_with_pad_defense,
)
# 修改导入，只导入 generate_masks
from attacks.mba_utils import generate_masks_tfidf
from attacks.s2mia_splits_and_index import load_full_mapping
from attacks.s2mia_utils import adjusted_accuracy, check_id_recall
from sklearn.metrics import roc_auc_score
from rag.pad_local_generate import load_local_llm

def _load_cache(path: str) -> Dict[str, Any]:
    """加载JSON缓存文件。"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: Dict[str, Any]):
    """保存JSON缓存文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_model_id_for_llm() -> str:
    """获取用于生成缓存键的LLM模型ID。"""
    return os.getenv("LLM_MODEL_ID", "unknown-model")


def _cache_key(row_id: int, defense: bool, defense_mode: str, num_masks: int) -> str:
    """为 MBA 攻击创建一个唯一的缓存键。"""
    defense_str = f"def_{defense_mode}" if defense else "no_def"
    return f"{row_id}|mba_tfidf|{defense_str}|{num_masks}|{_get_model_id_for_llm()}|v2"


def _maybe_export(out_dir: str, defense: bool, metrics: Dict[str, Any], eval_details: List[Dict[str, Any]]):
    """如果指定了输出目录，则保存所有实验结果。"""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 40)
    print("      Recall Statistics (Members Only)      ")
    print("=" * 40)

    member_details = [d for d in eval_details if d['label'] == 1]
    recall_summary = {}

    if member_details:
        if "recall@1" in member_details[0]:
            for k in [1, 2, 3]:
                key = f"recall@{k}"
                hits = [d.get(key, 0) for d in member_details]
                avg_recall = float(np.mean(hits))
                recall_summary[key] = avg_recall
                print(f"  > Recall@{k}: {avg_recall:.2%} ({sum(hits)}/{len(hits)})")
        else:
            print("  [Info] No recall data found in evaluation details.")
    else:
        print("  [Warn] No members in evaluation set.")

    metrics["recall_stats"] = recall_summary

    with open(os.path.join(out_dir, f"metrics_def{int(defense)}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"eval_details_def{int(defense)}.jsonl"), "w", encoding="utf-8") as f:
        for d in eval_details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n[Export] MBA results saved successfully to: {out_dir}")

    scores_member = [d['pred_acc'] for d in eval_details if d['label'] == 1]
    scores_non = [d['pred_acc'] for d in eval_details if d['label'] == 0]

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.hist(scores_member, bins=np.linspace(0, 1, 21), alpha=0.7, label="Member", color="#1f77b4")
        plt.hist(scores_non, bins=np.linspace(0, 1, 21), alpha=0.7, label="Non-Member", color="#ff7f0e")
        plt.title(f"MBA-Random Prediction Accuracy Distribution (defense={defense})")
        plt.xlabel("Prediction Accuracy");
        plt.ylabel("Count");
        plt.legend();
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"mba_random_acc_histogram_def{int(defense)}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Export] MBA accuracy distribution plot saved to {plot_path}")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")


def calculate_prediction_accuracy(ground_truth: Dict[str, str], predicted: Dict[str, str]) -> float:
    """计算掩码预测的准确率。"""
    if not ground_truth:
        return 0.0
    return np.mean([1 if predicted.get(tag, "").strip().lower() == true_ans.strip().lower() else 0 for tag, true_ans in
                    ground_truth.items()])


def eval_one_doc(
        row_id: int, text: str, defense: bool, defense_mode: str,  # ✅ 新增 defense_mode 参数
        args: argparse.Namespace,
        cache: Dict[str, Any], member_index: faiss.Index, member_mapping: Dict[str, Dict], embedder: BGEM3FlagModel,
        llm_model=None, llm_tokenizer=None
) -> Tuple[float, Dict[str, int]]:
    """对单个文档执行MBA攻击并返回预测准确率。"""
    # ✅ 更新缓存键的调用
    key = _cache_key(row_id, defense, defense_mode, args.num_masks)
    if key in cache:
        cached_data = cache[key]
        pred_acc = float(cached_data.get("pred_acc", 0.0))
        recall_stats = cached_data.get("recall", {})  # 如果是旧缓存，给空
        return pred_acc, recall_stats

    try:
        masked_document, ground_truth_answers = generate_masks_tfidf(
            text, num_masks=args.num_masks
        )

        common_args = {
            'index': member_index, 'mapping': member_mapping, 'embedder': embedder,
            'top_k': args.top_k, 'per_ctx_clip': args.per_ctx_clip, 'max_tokens': args.max_tokens
        }

        if defense:
            if defense_mode == 'prompt_guard':
                # 使用 **common_args
                out = generate_mba_with_prompt_guard_defense(masked_document, **common_args)
            elif defense_mode == 'rewrite':
                # 使用 **common_args
                out = generate_mba_with_rewrite_defense(masked_document, **common_args)
            elif defense_mode == 'pad':
                # ✅ 新增：PAD 分支
                if llm_model is None:
                    raise ValueError("Model not loaded for PAD defense!")
                out = generate_mba_with_pad_defense(
                    masked_document,
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    **common_args
                )
            else:  # 默认为 mirabel
                # Mirabel 防御有不同的参数签名，不使用 common_args
                out = generate_mba_with_defense(
                    masked_document, args.index_dir_member,
                    args.rho, args.margin, args.gap_min,
                    top_k=args.top_k, per_ctx_clip=args.per_ctx_clip, max_tokens=args.max_tokens
                )
        else:  # 无防御
            out = generate_mba_no_defense(masked_document, **common_args)

        pred_acc = calculate_prediction_accuracy(ground_truth_answers, out["predicted_answers"])

        # ✅ 计算 Recall
        retrieved_ids = out.get("final_topk_ids", [])
        recall_stats = check_id_recall(
            target_id=int(row_id),
            retrieved_ids=retrieved_ids,
            top_k_list=[1, 2, 3]
        )

    except Exception as e:
        print(f"\n[MBA EVAL FAIL] row={row_id}, defense={defense}, mode={defense_mode}, err={e}")
        return 0.0, {}

    cache[key] = {
        "pred_acc": pred_acc,
        "gt_answers": ground_truth_answers,
        "pred_answers": out.get("predicted_answers"),
        "recall": recall_stats
    }
    return pred_acc, recall_stats


def run(args: argparse.Namespace):
    """MBA攻击的主执行函数。"""
    if args.out_dir: os.makedirs(args.out_dir, exist_ok=True)
    with open(args.splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    mapping_full = load_full_mapping(index_dir="index")
    m_ref, n_ref = (splits["member_ref"][:args.n_ref_m], splits["non_ref"][:args.n_ref_n])
    m_eval, n_eval = (splits["member_eval"][:args.n_eval_m], splits["non_eval"][:args.n_eval_n])
    cache, SAVE_EVERY = _load_cache(args.cache_path), args.save_every

    print("\n[Main] Pre-loading RAG models and indexes (no proxy model needed)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_bge_path = "/root/autodl-tmp/pycharm_Mirabel/models/BAAI/bge-m3"
    print(f"[Main] Loading local BGE-M3 from: {local_bge_path}")
    embedder = BGEM3FlagModel(local_bge_path, use_fp16=True, device=device)
    member_index = faiss.read_index(os.path.join(args.index_dir_member, "nf_member.index"))
    with open(os.path.join(args.index_dir_member, "nf_member_docs.json"), "r", encoding="utf-8") as f:
        member_mapping = json.load(f)
    print("[Main] Models and indexes are ready.")

    llm_model, llm_tokenizer = None, None
    if args.defense and args.defense_mode == 'pad':
        model_path = os.getenv("LLM_MODEL_PATH", "/root/autodl-tmp/pycharm_Mirabel/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
        llm_tokenizer, llm_model = load_local_llm(model_path)
    common_eval_args = (args.defense_mode, args, cache, member_index, member_mapping, embedder, llm_model, llm_tokenizer)

    ref_tasks = [(r, 1) for r in m_ref] + [(r, 0) for r in n_ref];
    np.random.shuffle(ref_tasks)
    print(f"\n--- Processing Reference Set ({len(ref_tasks)} total) ---")
    ref_scores, ref_labels = [], []
    with tqdm(total=len(ref_tasks), desc=f"Ref Set (def={args.defense}, mode={args.defense_mode})") as pbar:
        for i, (row_id, label) in enumerate(ref_tasks):
            score, _ = eval_one_doc(row_id, mapping_full[str(row_id)]["text"], args.defense, *common_eval_args)
            ref_scores.append(score);
            ref_labels.append(label)
            pbar.update(1)
            if (i + 1) % SAVE_EVERY == 0: _save_cache(args.cache_path, cache)
    _save_cache(args.cache_path, cache)

    best_acc_ref, best_threshold = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        acc = np.mean((np.array(ref_scores) >= t) == np.array(ref_labels))
        if acc > best_acc_ref: best_acc_ref, best_threshold = acc, t
    print(f"\n[Threshold] Best threshold found: {best_threshold:.2f} (Ref Acc: {best_acc_ref:.3f})")

    eval_tasks = [(r, 1) for r in m_eval] + [(r, 0) for r in n_eval];
    np.random.shuffle(eval_tasks)
    print(f"\n--- Processing Evaluation Set ({len(eval_tasks)} total) ---")
    eval_details = []
    with tqdm(total=len(eval_tasks), desc=f"Eval Set (def={args.defense}, mode={args.defense_mode})") as pbar:
        for i, (row_id, label) in enumerate(eval_tasks):
            score, rec = eval_one_doc(row_id, mapping_full[str(row_id)]["text"], args.defense, *common_eval_args)
            item = {"row_id": int(row_id), "label": label, "pred_acc": score}
            item.update(rec)  # 合并 recall
            eval_details.append(item)
            pbar.update(1)
            if (i + 1) % SAVE_EVERY == 0: _save_cache(args.cache_path, cache)
    _save_cache(args.cache_path, cache)

    eval_scores, y_true = [d['pred_acc'] for d in eval_details], np.array([d['label'] for d in eval_details])
    print("\n--- Final Evaluation ---")
    preds = (np.array(eval_scores) >= best_threshold).astype(int)
    acc, adj_acc = np.mean(preds == y_true), adjusted_accuracy(np.mean(preds == y_true))
    roc_auc = roc_auc_score(y_true, eval_scores)

    for i in range(len(eval_details)): eval_details[i]['pred'] = int(preds[i])
    print(f"[MBA-Random] Accuracy: {acc:.4f}, Adj Acc: {adj_acc:.4f}, ROC AUC: {roc_auc:.4f}")

    metrics = {"acc": acc, "adj_acc": adj_acc, "roc_auc": roc_auc, "best_threshold": best_threshold,
               "params": vars(args)}
    if args.out_dir: _maybe_export(args.out_dir, args.defense, metrics, eval_details)


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser("MBA (Random Masking) Attack on NFCorpus")
    p.add_argument("--splits_path", default="configs/s2mia_splits.json");
    p.add_argument("--index_dir_member", default="index_member")
    p.add_argument("--cache_path", default="cache/mba_random.json");
    p.add_argument("--out_dir", default="results/mba_random_run")
    p.add_argument("--n_ref_m", type=int, default=60);
    p.add_argument("--n_ref_n", type=int, default=60)
    p.add_argument("--n_eval_m", type=int, default=100);
    p.add_argument("--n_eval_n", type=int, default=100)
    p.add_argument("--num_masks", type=int, default=15);
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--per_ctx_clip", type=int, default=600);
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--rho", type=float, default=0.005);
    p.add_argument("--margin", type=float, default=0.02);
    p.add_argument("--gap_min", type=float, default=0.03)
    p.add_argument("--defense", action="store_true");
    p.add_argument(
        "--defense_mode",
        type=str,
        default="mirabel",
        choices=["mirabel", "prompt_guard", "rewrite", "pad"],
        help="Specify the defense mode to use when --defense is enabled."
    )
    p.add_argument("--save_every", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = build_parser()
    run(args)