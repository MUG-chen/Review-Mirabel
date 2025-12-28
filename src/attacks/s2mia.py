# src/attacks/s2mia.py
from __future__ import annotations
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 引入修改后的 utils
from attacks.s2mia_utils import (
    split_doc, compute_bleu_single, compute_ppl_with_model,  # <--- 替换了 compute_ppl_gpt2
    grid_search_thresholds, adjusted_accuracy, ks_statistic,
    compute_bertscore_single, check_id_recall
)
from attacks.s2mia_pipeline import (
    generate_s2mia_no_defense,
    generate_s2mia_with_defense,
    generate_s2mia_with_prompt_guard_defense,
    generate_s2mia_with_rewrite_defense,
    generate_s2mia_with_pad_defense,
)
from attacks.s2mia_splits_and_index import load_full_mapping
from rag.pad_local_generate import load_local_llm


# ---------- 缓存 ----------
def _load_cache(path: str) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _get_model_id_for_cache() -> str:
    return os.getenv("LLM_MODEL_ID", "unknown-model")


def _cache_key(row_id: int, defense: bool, defense_mode: str, per_ctx_clip: int, max_tokens: int, cont_words: int,
               query_strategy: str) -> str:
    model_id = _get_model_id_for_cache()
    defense_str = f"def_{defense_mode}" if defense else "no_def"
    # 更新版本号 v4，因为 PPL 计算方式变了
    return f"{row_id}|{defense_str}|{per_ctx_clip}|{max_tokens}|{cont_words}|{model_id}|{query_strategy}|v4"


def eval_one_doc(
        row_id: int,
        text: str,
        defense: bool,
        defense_mode: str,
        index_dir_member: str,
        per_ctx_clip: int,
        max_tokens: int,
        cont_words: int,
        mirabel_cfg: Dict[str, Any],
        cache: Dict[str, Dict[str, Any]],
        query_strategy: str,
        llm_model=None,
        llm_tokenizer=None,
) -> Tuple[float, float, float, Dict[str, int]]:
    key = _cache_key(row_id, defense, defense_mode, per_ctx_clip, max_tokens, cont_words, query_strategy)
    if key in cache:
        cached_data = cache[key]
        recall_stats = cached_data.get("recall", {})
        return (
            float(cached_data.get("bleu", 0.0)),
            float(cached_data.get("ppl", -1.0)),  # 默认 -1
            float(cached_data.get("bertscore", 0.0)),
            recall_stats
        )

    # --- 查询构造逻辑 ---
    full_text_cleaned = (text or "").strip()
    title = full_text_cleaned.split('\n', 1)[0].strip()

    if query_strategy == 'title_only':
        x_q = title
    elif query_strategy == 'title_and_snippet':
        body_text = full_text_cleaned.split('\n', 1)[1] if '\n' in full_text_cleaned else full_text_cleaned
        snippet = " ".join(body_text.split()[:30])
        x_q = f"{title} {snippet}".strip()
    else:
        max_query_chars = 1200
        mid = min(len(full_text_cleaned) // 2, max_query_chars)
        x_q = full_text_cleaned[:mid].strip()

    ground_truth_reference = full_text_cleaned

    if not x_q:
        return 0.0, 1e9, 0.0, {}

    try:
        # 调用RAG流水线
        if defense:
            if defense_mode == 'prompt_guard':
                out = generate_s2mia_with_prompt_guard_defense(
                    excerpt=x_q, index_dir=index_dir_member,
                    top_k=3, per_ctx_clip=per_ctx_clip, max_tokens=max_tokens, cont_words=cont_words,
                )
            elif defense_mode == 'rewrite':
                out = generate_s2mia_with_rewrite_defense(
                    excerpt=x_q, index_dir=index_dir_member,
                    top_k=3, per_ctx_clip=per_ctx_clip, max_tokens=max_tokens, cont_words=cont_words,
                )
            elif defense_mode == 'pad':
                if llm_model is None:
                    raise ValueError("Local LLM model not loaded for PAD defense!")
                out = generate_s2mia_with_pad_defense(
                    excerpt=x_q,
                    index_dir=index_dir_member,
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    top_k=3,
                    per_ctx_clip=per_ctx_clip,
                    max_tokens=max_tokens,
                    cont_words=cont_words
                )
            else:  # mirabel
                out = generate_s2mia_with_defense(
                    excerpt=x_q, index_dir=index_dir_member,
                    rho=mirabel_cfg.get("rho", 0.005),
                    margin=mirabel_cfg.get("margin", 0.02),
                    gap_min=mirabel_cfg.get("gap_min", 0.03),
                    use_gap=mirabel_cfg.get("use_gap", True),
                    use_full_corpus=True, topM_if_not_full=200, top_k=3, use_precise_mu=True,
                    per_ctx_clip=per_ctx_clip, max_tokens=max_tokens, cont_words=cont_words,
                )
        else:
            out = generate_s2mia_no_defense(
                excerpt=x_q, index_dir=index_dir_member,
                topM=None, top_k=3,
                per_ctx_clip=per_ctx_clip, max_tokens=max_tokens, cont_words=cont_words,
            )

        if out is None:
            raise RuntimeError("RAG pipeline returned None.")

        answer = (out.get("answer") or "").strip()
        retrieved_ids = out.get("final_topk_ids", [])

        recall_stats = check_id_recall(
            target_id=int(row_id),
            retrieved_ids=retrieved_ids,
            top_k_list=[1, 2, 3]
        )

        if not answer:
            cache[key] = {"answer": "", "bleu": 0.0, "ppl": 1e9, "bertscore": 0.0, "recall": recall_stats}
            return 0.0, 1e9, 0.0, recall_stats

        # --- 计算指标 ---
        bleu = compute_bleu_single(answer, ground_truth_reference)
        bertscore = compute_bertscore_single(answer, ground_truth_reference)

        # [修改] PPL 计算逻辑
        if llm_model is not None and llm_tokenizer is not None:
            # 如果有本地模型，使用它计算 PPL (符合论文)
            ppl = compute_ppl_with_model(answer, llm_model, llm_tokenizer)
        else:
            # 如果是 API 模式且没加载本地模型，设为 -1 (后续逻辑会忽略 PPL，只用 BLEU/BERTScore)
            ppl = -1.0

    except Exception as e:
        print(f"\n[EVAL FAIL] row={row_id}, err={e}")
        return 0.0, 1e9, 0.0, {}

    cache[key] = {
        "answer": answer,
        "bleu": bleu,
        "ppl": ppl,
        "bertscore": bertscore,
        "recall": recall_stats
    }
    return bleu, ppl, bertscore, recall_stats


def _maybe_export(
        out_dir: str,
        defense: bool,
        thresholds: Dict[str, float],
        metrics: Dict[str, Any],
        eval_details: List[Dict[str, Any]],
):
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

    metrics["recall_stats"] = recall_summary

    with open(os.path.join(out_dir, f"thresholds_def{int(defense)}.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"metrics_def{int(defense)}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    out_jsonl = os.path.join(out_dir, f"eval_details_def{int(defense)}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for d in eval_details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[Export] Results saved to directory: {out_dir}")


def run(args: argparse.Namespace) -> None:
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    mapping_full = load_full_mapping(index_dir="index")
    with open(args.splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    m_ref_rows = splits["member_ref"][:args.n_ref_m] if args.n_ref_m > 0 else splits["member_ref"]
    n_ref_rows = splits["non_ref"][:args.n_ref_n] if args.n_ref_n > 0 else splits["non_ref"]
    m_eval_rows = splits["member_eval"][:args.n_eval_m] if args.n_eval_m > 0 else splits["member_eval"]
    n_eval_rows = splits["non_eval"][:args.n_eval_n] if args.n_eval_n > 0 else splits["non_eval"]

    mirabel_cfg = {
        "rho": args.rho,
        "margin": (None if args.margin < 0 else args.margin),
        "gap_min": (None if args.gap_min < 0 else args.gap_min),
        "use_gap": args.use_gap,
    }

    cache = _load_cache(args.cache_path)
    print(f"[Cache] Loaded {len(cache)} entries")
    SAVE_EVERY = args.save_every if args.save_every > 0 else 50

    # [修改] 模型加载逻辑
    # 只要是 PAD 模式，或者用户显式指定要加载本地模型用于 PPL 计算，就加载
    llm_model, llm_tokenizer = None, None

    # 默认路径
    model_path = os.getenv("LLM_MODEL_PATH",
                           "/root/autodl-tmp/pycharm_Mirabel/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")

    # 如果是 PAD 模式，必须加载。如果是其他模式但想算 PPL，也可以加载(这里为了简化，仅 PAD 强制加载)
    if args.defense and args.defense_mode == 'pad':
        print(f"[Main] Defense mode is PAD. Loading local model from: {model_path}")
        llm_tokenizer, llm_model = load_local_llm(model_path)
    elif args.force_load_model:  # 可选参数，用于 Baseline 也想算 PPL 的情况
        print(f"[Main] Force loading local model for PPL calc: {model_path}")
        llm_tokenizer, llm_model = load_local_llm(model_path)

    # 2. 参考集处理
    print(f"\n--- Processing Reference Set ---")
    ref_scores_3d: List[Tuple[float, float, float]] = []
    ref_labels: List[int] = []

    ref_tasks = [(r, 1) for r in m_ref_rows] + [(r, 0) for r in n_ref_rows]
    np.random.shuffle(ref_tasks)

    for idx, (row_id, label) in enumerate(tqdm(ref_tasks, desc=f"Ref Set")):
        text = mapping_full[str(row_id)]["text"]
        b, p, bs, rec = eval_one_doc(
            row_id=row_id, text=text, defense=args.defense,
            defense_mode=args.defense_mode,
            index_dir_member=args.index_dir_member,
            per_ctx_clip=args.per_ctx_clip, max_tokens=args.max_tokens, cont_words=args.cont_words,
            mirabel_cfg=mirabel_cfg, cache=cache, query_strategy=args.query_strategy,
            llm_model=llm_model, llm_tokenizer=llm_tokenizer
        )
        ref_scores_3d.append((b, p, bs))
        ref_labels.append(label)
        if (idx + 1) % SAVE_EVERY == 0:
            _save_cache(args.cache_path, cache)
    _save_cache(args.cache_path, cache)

    # 3. S2MIA-T: 搜索阈值
    # [修改] 传递 (BERTScore, PPL) 给 grid_search_thresholds
    # 函数内部会自动判断 PPL 是否有效 (-1)
    print("\n[S2MIA-T] Searching thresholds...")
    ref_scores_for_t = [(bs, p) for b, p, bs in ref_scores_3d]
    th_t = grid_search_thresholds(ref_scores_for_t, ref_labels)
    print(
        f"  > Best Thresholds: Metric >= {th_t['theta_metric']:.3f}, PPL <= {th_t['theta_ppl']:.1f} (Acc: {th_t['acc']:.3f})")

    # 4. 评估集处理
    print(f"\n--- Processing Evaluation Set ---")
    eval_scores_3d: List[Tuple[float, float, float]] = []
    eval_labels: List[int] = []
    eval_details: List[Dict[str, Any]] = []

    eval_tasks = [(r, 1) for r in m_eval_rows] + [(r, 0) for r in n_eval_rows]
    np.random.shuffle(eval_tasks)

    for idx, (row_id, label) in enumerate(tqdm(eval_tasks, desc=f"Eval Set")):
        text = mapping_full[str(row_id)]["text"]
        b, p, bs, rec = eval_one_doc(
            row_id=row_id, text=text, defense=args.defense,
            defense_mode=args.defense_mode,
            index_dir_member=args.index_dir_member,
            per_ctx_clip=args.per_ctx_clip, max_tokens=args.max_tokens, cont_words=args.cont_words,
            mirabel_cfg=mirabel_cfg, cache=cache, query_strategy=args.query_strategy,
            llm_model=llm_model, llm_tokenizer=llm_tokenizer
        )
        eval_scores_3d.append((b, p, bs))
        eval_labels.append(label)

        detail_item = {"row_id": int(row_id), "label": label, "bleu": b, "ppl": p, "bertscore": bs}
        detail_item.update(rec)
        eval_details.append(detail_item)

        if (idx + 1) % SAVE_EVERY == 0:
            _save_cache(args.cache_path, cache)
    _save_cache(args.cache_path, cache)

    # 5. 最终评估
    print("\n--- Final Evaluation ---")

    # 5.1 S2MIA-T 评估
    eval_scores_for_t = [(bs, p) for b, p, bs in eval_scores_3d]
    # 使用搜索到的阈值进行预测
    preds_t = np.array(
        [(1 if (bs >= th_t["theta_metric"] and p <= th_t["theta_ppl"]) else 0) for (bs, p) in eval_scores_for_t])
    acc_t = float((preds_t == np.array(eval_labels)).mean())
    adj_t = adjusted_accuracy(acc_t)

    bertscores_member = [d['bertscore'] for d in eval_details if d['label'] == 1]
    bertscores_non = [d['bertscore'] for d in eval_details if d['label'] == 0]
    ks_bertscore = ks_statistic(bertscores_member, bertscores_non)

    print(f"[S2MIA-T | BERTScore] Accuracy: {acc_t:.4f}, Adjusted: {adj_t:.4f}")
    print(f"  KS(BERTScore)={ks_bertscore:.3f}")

    # 5.2 S2MIA-M 评估 (XGBoost)
    # [修改] 只有当 PPL 有效时才使用 PPL 特征，否则只用 BERTScore
    has_valid_ppl = any(p > 0 for _, p, _ in ref_scores_3d)

    try:
        print(f"\n[S2MIA-M | Features: {'PPL, BERTScore' if has_valid_ppl else 'BERTScore Only'}]")

        if has_valid_ppl:
            X_ref = np.array([(p, bs) for b, p, bs in ref_scores_3d], dtype=np.float64)
            X_eval = np.array([(p, bs) for b, p, bs in eval_scores_3d], dtype=np.float64)
        else:
            # 如果没有 PPL，只用 BERTScore (reshape 为 2D 数组)
            X_ref = np.array([bs for b, p, bs in ref_scores_3d], dtype=np.float64).reshape(-1, 1)
            X_eval = np.array([bs for b, p, bs in eval_scores_3d], dtype=np.float64).reshape(-1, 1)

        y_ref = np.array(ref_labels)
        y_eval = np.array(eval_labels)

        # 数据清洗 (处理 NaN/Inf)
        X_ref[np.isinf(X_ref)] = np.nan
        X_eval[np.isinf(X_eval)] = np.nan

        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        X_ref = imputer.fit_transform(X_ref)
        X_eval = imputer.transform(X_eval)

        # 如果有 PPL (第一列)，做 log 变换使其分布更正态
        if has_valid_ppl:
            X_ref[:, 0] = np.log1p(X_ref[:, 0])
            X_eval[:, 0] = np.log1p(X_eval[:, 0])

        # GridSearch XGBoost
        param_grid = {
            'max_depth': [3, 4, 5],
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1]
        }
        xgb = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=1)
        grid_search.fit(X_ref, y_ref)

        best_model = grid_search.best_estimator_
        y_pred_m = best_model.predict(X_eval)
        y_prob_m = best_model.predict_proba(X_eval)[:, 1]

        acc_m = float((y_pred_m == y_eval).mean())
        adj_m = adjusted_accuracy(acc_m)
        from sklearn.metrics import roc_auc_score
        roc_auc_m = roc_auc_score(y_eval, y_prob_m)

        print(f"  > XGBoost Accuracy = {acc_m:.4f}")
        print(f"  > XGBoost ROC AUC = {roc_auc_m:.4f}")

    except Exception as e:
        print(f"[WARN] S2MIA-M failed: {e}")
        acc_m, adj_m, roc_auc_m = 0.0, 0.0, 0.0
        y_pred_m, y_prob_m = np.zeros_like(eval_labels), np.zeros_like(eval_labels, dtype=float)

    # 6. 导出
    for i in range(len(eval_details)):
        eval_details[i]["pred_t"] = int(preds_t[i])
        eval_details[i]["pred_m"] = int(y_pred_m[i])
        eval_details[i]["prob_m"] = float(y_prob_m[i])

    thresholds_dump = {
        "theta_bertscore": th_t["theta_metric"],
        "theta_ppl": th_t["theta_ppl"],
        "ref_acc_t": th_t["acc"],
        "used_ppl": has_valid_ppl
    }
    metrics_dump = {
        "defense": bool(args.defense),
        "s2mia_t": {"acc": acc_t, "adj_acc": adj_t, "ks_bertscore": ks_bertscore},
        "s2mia_m": {"acc": acc_m, "adj_acc": adj_m, "roc_auc": roc_auc_m},
        "params": vars(args),
    }

    _maybe_export(args.out_dir, args.defense, thresholds_dump, metrics_dump, eval_details)


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser("S2MIA (continuation BLEU+PPL) on NFCorpus")
    p.add_argument("--splits_path", type=str, default="configs/s2mia_splits.json")
    p.add_argument("--index_dir_member", type=str, default="index_member")
    p.add_argument("--cache_path", type=str, default="cache/s2mia_answers.json")
    p.add_argument("--out_dir", type=str, default="results/default_run")
    p.add_argument("--n_ref_m", type=int, default=60)
    p.add_argument("--n_ref_n", type=int, default=60)
    p.add_argument("--n_eval_m", type=int, default=100)
    p.add_argument("--n_eval_n", type=int, default=100)
    p.add_argument("--per_ctx_clip", type=int, default=300)
    p.add_argument("--max_tokens", type=int, default=96)
    p.add_argument("--cont_words", type=int, default=120)
    p.add_argument("--rho", type=float, default=0.005)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--gap_min", type=float, default=0.03)
    p.add_argument("--use_gap", action="store_true", default=True)
    p.add_argument("--defense", action="store_true", default=False)
    p.add_argument("--defense_mode", type=str, default="mirabel", choices=["mirabel", "prompt_guard", "rewrite", "pad"])
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--query_strategy", type=str, default="title_and_snippet",
                   choices=["half_text", "title_only", "title_and_snippet"])
    # [新增] 强制加载模型参数
    p.add_argument("--force_load_model", action="store_true",
                   help="Force load local LLM for PPL calculation even if not in PAD mode.")

    return p.parse_args()


if __name__ == "__main__":
    args = build_parser()
    run(args)
