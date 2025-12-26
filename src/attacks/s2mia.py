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

from attacks.s2mia_utils import (
    split_doc, compute_bleu_single, compute_ppl_gpt2,
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

def _cache_key(row_id: int, defense: bool, defense_mode: str, per_ctx_clip: int, max_tokens: int, cont_words: int, query_strategy: str) -> str:
    """为缓存创建一个唯一的键，包含所有影响结果的参数。"""
    model_id = _get_model_id_for_cache()
    # 增加 defense_mode 到键中，并更新版本号
    defense_str = f"def_{defense_mode}" if defense else "no_def"
    return f"{row_id}|{defense_str}|{per_ctx_clip}|{max_tokens}|{cont_words}|{model_id}|{query_strategy}|v3"


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
    """
    单文档评测：
    1. 根据策略构造查询 (x_q)。
    2. 调用RAG（有/无防御）进行续写式生成。
    3. 计算 BLEU, PPL, BERTScore 三个指标。
    4. 使用缓存避免重复计算。
    """
    key = _cache_key(row_id, defense, defense_mode, per_ctx_clip, max_tokens, cont_words, query_strategy)
    if key in cache:
        cached_data = cache[key]
        recall_stats = cached_data.get("recall", {})
        return (
            float(cached_data.get("bleu", 0.0)),
            float(cached_data.get("ppl", 1e9)),
            float(cached_data.get("bertscore", 0.0)),
            recall_stats
        )

    # --- 查询构造逻辑 ---
    # `text` 参数是原始的、完整的文档文本
    full_text_cleaned = (text or "").strip()
    title = full_text_cleaned.split('\n', 1)[0].strip()

    if query_strategy == 'title_only':
        x_q = title
    elif query_strategy == 'title_and_snippet':
        # 使用文档正文（排除标题行后）的前30个词作为片段
        body_text = full_text_cleaned.split('\n', 1)[1] if '\n' in full_text_cleaned else full_text_cleaned
        snippet = " ".join(body_text.split()[:30])
        x_q = f"{title} {snippet}".strip()
    else:  # 默认 'half_text'
        max_query_chars = 1200
        mid = min(len(full_text_cleaned) // 2, max_query_chars)
        x_q = full_text_cleaned[:mid].strip()

    # 参考文本定为完整的原文，这对BLEU和BERTScore都更公平鲁棒
    ground_truth_reference = full_text_cleaned

    if not x_q:
        return 0.0, 1e9, 0.0  # 返回3个值

    try:
        # 调用RAG流水线获取生成的续写
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
                    raise ValueError("Local LLM model not loaded for PAD defense! Check --defense_mode.")
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
            else:  # 默认为 mirabel
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
        answer = out["answer"]
        answer = (answer or "").strip()

        retrieved_ids = out.get("final_topk_ids", [])

        # 计算 Recall@1, Recall@2, Recall@3
        recall_stats = check_id_recall(
            target_id=int(row_id),
            retrieved_ids=retrieved_ids,
            top_k_list=[1, 2, 3]
        )

        # 如果LLM拒答或返回空，则视为失败
        if not answer:
            cache[key] = {"answer": "", "bleu": 0.0, "ppl": 1e9, "bertscore": 0.0, "recall": recall_stats}
            return 0.0, 1e9, 0.0

        # --- 计算所有三个分数 ---
        bleu = compute_bleu_single(answer, ground_truth_reference)
        ppl = compute_ppl_gpt2(answer)
        bertscore = compute_bertscore_single(answer, ground_truth_reference)

    except Exception as e:
        print(f"\n[EVAL FAIL] row={row_id}, defense={defense}, query_strategy={query_strategy}, err={e}")
        return 0.0, 1e9, 0.0, {}

    # 缓存所有结果
    cache[key] = {
        "answer": answer,
        "bleu": bleu,
        "ppl": ppl,
        "bertscore": bertscore,
        "recall": recall_stats  # <--- 保存 recall
    }
    return bleu, ppl, bertscore, recall_stats


def _maybe_export(
        out_dir: str,
        defense: bool,
        thresholds: Dict[str, float],
        metrics: Dict[str, Any],
        eval_details: List[Dict[str, Any]],
):
    """保存所有实验结果：阈值、指标、详细评估数据和分布图。"""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)

    # ✅ [新增] 统计整体召回率 (Recall Statistics)
    # 我们只关心 Member 的召回率，因为 Non-Member 本来就不在知识库里
    print("\n" + "=" * 40)
    print("      Recall Statistics (Members Only)      ")
    print("=" * 40)

    member_details = [d for d in eval_details if d['label'] == 1]
    recall_summary = {}

    if member_details:
        # 检查是否有 recall 数据 (确保不是脏数据)
        if "recall@1" in member_details[0]:
            for k in [1, 2, 3]:
                key = f"recall@{k}"
                # 提取列表
                hits = [d.get(key, 0) for d in member_details]
                # 计算平均值
                avg_recall = float(np.mean(hits))
                recall_summary[key] = avg_recall
                print(f"  > Recall@{k}: {avg_recall:.2%} ({sum(hits)}/{len(hits)})")
        else:
            print("  [Info] No recall data found in evaluation details.")
    else:
        print("  [Warn] No members in evaluation set.")

    # 将召回率统计放入 metrics 字典中，以便保存到 JSON
    metrics["recall_stats"] = recall_summary

    # 保存阈值和指标
    with open(os.path.join(out_dir, f"thresholds_def{int(defense)}.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"metrics_def{int(defense)}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存详细的逐条评估结果
    out_jsonl = os.path.join(out_dir, f"eval_details_def{int(defense)}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for d in eval_details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[Export] Results saved to directory: {out_dir}")

    # 提取分数用于绘图
    bleus_member = [d['bleu'] for d in eval_details if d['label'] == 1]
    bleus_non = [d['bleu'] for d in eval_details if d['label'] == 0]
    bertscores_member = [d['bertscore'] for d in eval_details if d['label'] == 1]
    bertscores_non = [d['bertscore'] for d in eval_details if d['label'] == 0]

    # 尝试绘图
    try:
        import matplotlib.pyplot as plt

        # 绘制 BLEU 分布图
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(bleus_member, bins=30, alpha=0.7, label="Member", color="#1f77b4")
        plt.hist(bleus_non, bins=30, alpha=0.7, label="Non-Member", color="#ff7f0e")
        plt.title(f"BLEU Distribution (defense={defense})")
        plt.xlabel("BLEU Score");
        plt.ylabel("Count");
        plt.legend()

        # 新增：绘制 BERTScore 分布图
        plt.subplot(1, 2, 2)
        plt.hist(bertscores_member, bins=30, alpha=0.7, label="Member", color="#1f77b4")
        plt.hist(bertscores_non, bins=30, alpha=0.7, label="Non-Member", color="#ff7f0e")
        plt.title(f"BERTScore Distribution (defense={defense})")
        plt.xlabel("BERTScore");
        plt.ylabel("Count");
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"score_histograms_def{int(defense)}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Export] Score distribution plots saved to {plot_path}")

    except ImportError:
        print("[WARN] Matplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'.")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")


def run(args: argparse.Namespace) -> None:
    """主执行函数，协调整个S2MIA攻击和评估流程。"""
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据
    mapping_full = load_full_mapping(index_dir="index")
    with open(args.splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # 根据参数截取数据集大小
    m_ref_rows = splits["member_ref"][:args.n_ref_m] if args.n_ref_m > 0 else splits["member_ref"]
    n_ref_rows = splits["non_ref"][:args.n_ref_n] if args.n_ref_n > 0 else splits["non_ref"]
    m_eval_rows = splits["member_eval"][:args.n_eval_m] if args.n_eval_m > 0 else splits["member_eval"]
    n_eval_rows = splits["non_eval"][:args.n_eval_n] if args.n_eval_n > 0 else splits["non_eval"]

    # Mirabel 防御配置
    mirabel_cfg = {
        "rho": args.rho,
        "margin": (None if args.margin < 0 else args.margin),
        "gap_min": (None if args.gap_min < 0 else args.gap_min),
        "use_gap": args.use_gap,
    }

    # 加载缓存
    cache = _load_cache(args.cache_path)
    print(f"[Cache] Loaded {len(cache)} entries from {args.cache_path}")
    SAVE_EVERY = args.save_every if args.save_every > 0 else 50

    #预加载本地 LLM (仅当模式为 pad 时)
    llm_model, llm_tokenizer = None, None
    if args.defense and args.defense_mode == 'pad':
        # 默认路径或从环境变量读取
        model_path = os.getenv("LLM_MODEL_PATH",
                               "/root/autodl-tmp/pycharm_Mirabel/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
        print(f"[Main] Defense mode is PAD. Pre-loading local model from: {model_path}")
        llm_tokenizer, llm_model = load_local_llm(model_path)
        print("[Main] Local model loaded successfully.")
    # 封装通用评估参数
    common_eval_kwargs = {
        "defense": args.defense,
        "defense_mode": args.defense_mode,
        "index_dir_member": args.index_dir_member,
        "per_ctx_clip": args.per_ctx_clip,
        "max_tokens": args.max_tokens,
        "cont_words": args.cont_words,
        "mirabel_cfg": mirabel_cfg,
        "cache": cache,
        "query_strategy": args.query_strategy,
        "llm_model": llm_model,  # 传入
        "llm_tokenizer": llm_tokenizer  # 传入
    }

    # 2. 参考集处理 (获取三维特征)
    print(f"\n--- Processing Reference Set ({len(m_ref_rows)} members, {len(n_ref_rows)} non-members) ---")
    ref_scores_3d: List[Tuple[float, float, float]] = []
    ref_labels: List[int] = []

    ref_tasks = [(r, 1) for r in m_ref_rows] + [(r, 0) for r in n_ref_rows]
    np.random.shuffle(ref_tasks)

    for idx, (row_id, label) in enumerate(tqdm(ref_tasks, desc=f"Reference Set (def={args.defense})")):
        text = mapping_full[str(row_id)]["text"]
        b, p, bs, rec = eval_one_doc(
            row_id=row_id, text=text, defense=args.defense,
            defense_mode=args.defense_mode,
            index_dir_member=args.index_dir_member,
            per_ctx_clip=args.per_ctx_clip, max_tokens=args.max_tokens, cont_words=args.cont_words,
            mirabel_cfg=mirabel_cfg, cache=cache, query_strategy=args.query_strategy,
            llm_model = llm_model,
            llm_tokenizer = llm_tokenizer
        )
        ref_scores_3d.append((b, p, bs))
        ref_labels.append(label)
        if (idx + 1) % SAVE_EVERY == 0:
            _save_cache(args.cache_path, cache)
    _save_cache(args.cache_path, cache)

    # 3. S2MIA-T: 基于 (BERTScore, PPL) 搜索阈值
    print("\n[S2MIA-T] Searching thresholds using (BERTScore, PPL)...")
    ref_scores_for_t = [(bs, p) for b, p, bs in ref_scores_3d]
    th_t = grid_search_thresholds(ref_scores_for_t, ref_labels)
    print(
        f"  > Best Thresholds: BERTScore >= {th_t['theta_bleu']:.3f}, PPL <= {th_t['theta_ppl']:.1f} (Ref Acc: {th_t['acc']:.3f})")

    # 4. 评估集处理 (获取三维特征)
    print(f"\n--- Processing Evaluation Set ({len(m_eval_rows)} members, {len(n_eval_rows)} non-members) ---")
    eval_scores_3d: List[Tuple[float, float, float]] = []
    eval_labels: List[int] = []
    eval_details: List[Dict[str, Any]] = []

    eval_tasks = [(r, 1) for r in m_eval_rows] + [(r, 0) for r in n_eval_rows]
    np.random.shuffle(eval_tasks)

    for idx, (row_id, label) in enumerate(tqdm(eval_tasks, desc=f"Evaluation Set (def={args.defense})")):
        text = mapping_full[str(row_id)]["text"]
        b, p, bs, rec = eval_one_doc(
            row_id=row_id, text=text, defense=args.defense,
            defense_mode=args.defense_mode,  # ✅ 传递参数
            index_dir_member=args.index_dir_member,
            per_ctx_clip=args.per_ctx_clip, max_tokens=args.max_tokens, cont_words=args.cont_words,
            mirabel_cfg=mirabel_cfg, cache=cache, query_strategy=args.query_strategy,
            llm_model = llm_model,
            llm_tokenizer = llm_tokenizer
        )
        eval_scores_3d.append((b, p, bs))
        eval_labels.append(label)

        # ✅ [修改] 将 recall 统计 (rec) 合并到详情字典中
        detail_item = {"row_id": int(row_id), "label": label, "bleu": b, "ppl": p, "bertscore": bs}
        detail_item.update(rec)
        eval_details.append(detail_item)

        if (idx + 1) % SAVE_EVERY == 0:
            _save_cache(args.cache_path, cache)
    _save_cache(args.cache_path, cache)

    # 5. 最终评估
    print("\n--- Final Evaluation ---")

    # 5.1 S2MIA-T 评估 (基于BERTScore)
    eval_scores_for_t = [(bs, p) for b, p, bs in eval_scores_3d]
    preds_t = np.array(
        [(1 if (bs >= th_t["theta_bleu"] and p <= th_t["theta_ppl"]) else 0) for (bs, p) in eval_scores_for_t])
    acc_t = float((preds_t == np.array(eval_labels)).mean())
    adj_t = adjusted_accuracy(acc_t)

    bertscores_member = [d['bertscore'] for d in eval_details if d['label'] == 1]
    bertscores_non = [d['bertscore'] for d in eval_details if d['label'] == 0]
    ks_bertscore = ks_statistic(bertscores_member, bertscores_non)

    print(f"[S2MIA-T | BERTScore] Accuracy: {acc_t:.4f}, Adjusted Accuracy: {adj_t:.4f}")
    print(f"  KS(BERTScore)={ks_bertscore:.3f}")

    # 5.2 S2MIA-M 评估 (使用 GridSearchCV 调优 XGBoost)
    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import GridSearchCV

        print("\n[S2MIA-M | Features: PPL, BERTScore | Model: XGBoost with GridSearchCV]")

        # 提取 (PPL, BERTScore) 作为特征
        X_ref_m = np.array([(p, bs) for b, p, bs in ref_scores_3d], dtype=np.float64)
        X_eval_m = np.array([(p, bs) for b, p, bs in eval_scores_3d], dtype=np.float64)
        y_ref = np.array(ref_labels)
        y_eval = np.array(eval_labels)

        # 数据清洗和PPL变换
        # ... (使用我们上一版中更新的、更清晰的预处理逻辑) ...
        X_ref_m[np.isinf(X_ref_m)] = np.nan
        X_eval_m[np.isinf(X_eval_m)] = np.nan

        imputer_for_ppl = SimpleImputer(missing_values=np.nan, strategy='median')
        X_ref_m[:, 0] = np.log1p(imputer_for_ppl.fit_transform(X_ref_m[:, 0].reshape(-1, 1))).flatten()
        X_eval_m[:, 0] = np.log1p(imputer_for_ppl.transform(X_eval_m[:, 0].reshape(-1, 1))).flatten()

        imputer_for_bs = SimpleImputer(missing_values=np.nan, strategy='median')
        X_ref_m[:, 1] = imputer_for_bs.fit_transform(X_ref_m[:, 1].reshape(-1, 1)).flatten()
        X_eval_m[:, 1] = imputer_for_bs.transform(X_eval_m[:, 1].reshape(-1, 1)).flatten()

        X_ref_cleaned = X_ref_m
        X_eval_cleaned = X_eval_m

        # 🌟🌟🌟 新增：使用 GridSearchCV 进行超参数调优 🌟🌟🌟
        print("  > Finding best XGBoost parameters with GridSearchCV...")

        # 定义要搜索的参数网格
        param_grid = {
            'max_depth': [3, 4, 5],
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1]
        }

        # 初始化基础模型
        xgb = XGBClassifier(eval_metric='logloss', random_state=42, subsample=0.7, colsample_bytree=0.7, gamma=0.1)

        # 初始化 GridSearchCV
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=1)

        # 在参考集上进行搜索
        grid_search.fit(X_ref_cleaned, y_ref)

        print(f"  > Best parameters found: {grid_search.best_params_}")

        # 使用找到的最佳模型进行评估
        best_model = grid_search.best_estimator_
        # 🌟🌟🌟 GridSearchCV 调优结束 🌟🌟🌟

        # 预测和评估
        y_pred_m = best_model.predict(X_eval_cleaned)
        y_prob_m = best_model.predict_proba(X_eval_cleaned)[:, 1]

        acc_m = float((y_pred_m == y_eval).mean())
        adj_m = adjusted_accuracy(acc_m)
        roc_auc_m = roc_auc_score(y_eval, y_prob_m)

        print(f"  > Tuned XGBoost-based Accuracy = {acc_m:.4f}")
        print(f"  > Tuned XGBoost-based Adjusted Accuracy = {adj_m:.4f}")
        print(f"  > Tuned XGBoost-based ROC AUC = {roc_auc_m:.4f}")

    except ImportError as e:
        print(f"[WARN] Required library not found: {e}. Skipping S2MIA-M evaluation.")
        acc_m, adj_m, roc_auc_m = 0.0, 0.0, 0.0
        y_pred_m, y_prob_m = np.zeros_like(eval_labels), np.zeros_like(eval_labels, dtype=float)

    # 6. 结果导出
    for i in range(len(eval_details)):
        eval_details[i]["pred_t"] = int(preds_t[i])
        eval_details[i]["pred_m"] = int(y_pred_m[i])
        eval_details[i]["prob_m"] = float(y_prob_m[i])

    thresholds_dump = {
        "s2mia_t_features": "(BERTScore, PPL)",
        "theta_bertscore": th_t["theta_bleu"],
        "theta_ppl": th_t["theta_ppl"],
        "ref_acc_t": th_t["acc"]
    }
    metrics_dump = {
        "defense": bool(args.defense),
        "s2mia_t": {"acc": acc_t, "adj_acc": adj_t, "ks_bertscore": ks_bertscore},
        "s2mia_m": {"features": "(PPL, BERTScore)", "acc": acc_m, "adj_acc": adj_m, "roc_auc": roc_auc_m},
        "params": vars(args),
    }

    _maybe_export(args.out_dir, args.defense, thresholds_dump, metrics_dump, eval_details)


def build_parser() -> argparse.Namespace:
    """构建命令行参数解析器"""
    p = argparse.ArgumentParser("S2MIA (continuation BLEU+PPL) on NFCorpus (member index)")
    p.add_argument("--splits_path", type=str, default="configs/s2mia_splits.json")
    p.add_argument("--index_dir_member", type=str, default="index_member")
    p.add_argument("--cache_path", type=str, default="cache/s2mia_answers.json")
    p.add_argument("--out_dir", type=str, default="results/default_run", help="Directory to save metrics, plots, etc.")
    p.add_argument("--n_ref_m", type=int, default=60, help="Number of member samples for reference set.")
    p.add_argument("--n_ref_n", type=int, default=60, help="Number of non-member samples for reference set.")
    p.add_argument("--n_eval_m", type=int, default=100, help="Number of member samples for evaluation set.")
    p.add_argument("--n_eval_n", type=int, default=100, help="Number of non-member samples for evaluation set.")
    p.add_argument("--per_ctx_clip", type=int, default=300)
    p.add_argument("--max_tokens", type=int, default=96)
    p.add_argument("--cont_words", type=int, default=120)
    p.add_argument("--rho", type=float, default=0.005)
    p.add_argument("--margin", type=float, default=0.02, help="Set to a negative value to disable margin.")
    p.add_argument("--gap_min", type=float, default=0.03, help="Set to a negative value to disable gap.")
    p.add_argument("--use_gap", action="store_true", default=True)
    p.add_argument("--defense", action="store_true", default=False, help="Enable Mirabel defense.")
    p.add_argument(
        "--defense_mode",
        type=str,
        default="mirabel",
        choices=["mirabel", "prompt_guard", "rewrite", "pad"],
        help="Specify the defense mode to use when --defense is enabled."
    )
    p.add_argument("--lambda_penalty", type=float, default=1.0,
                   help="IA-specific penalty for 'I don't know' responses. Primarily for IA compatibility, not directly used in S2MIA core metrics.")
    p.add_argument("--save_every", type=int, default=50, help="Save cache every N iterations.")
    p.add_argument(
        "--query_strategy",
        type=str,
        default="title_and_snippet",
        choices=["half_text", "title_only", "title_and_snippet"],
        help="Query construction strategy to use."
    )

    return p.parse_args()

if __name__ == "__main__":
    args = build_parser()
    run(args)