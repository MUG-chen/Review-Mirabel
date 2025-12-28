# src/attacks/s2mia_utils.py
from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from scipy.stats import ks_2samp


# ---------- 文本拆分 ----------
def split_doc(text: str, max_query_chars: int = 1200) -> Tuple[str, str]:
    """
    将文档拆为 (x_q, x_r)：前半作为查询，后半作为剩余。
    NFCorpus 无问答对情况下采用该策略。
    """
    t = (text or "").strip()
    if not t:
        return "", ""
    mid = max(len(t) // 2, min(max_query_chars, len(t) // 2))
    x_q = t[:mid].strip()
    x_r = t[mid:].strip()
    if not x_q or not x_r:  # 兜底
        half = len(t) // 2 or 1
        x_q, x_r = t[:half], t[half:]
    return x_q, x_r


# ---------- BLEU ----------
def compute_bleu_single(hyp: str, ref: str) -> float:
    import sacrebleu
    hyp = (hyp or "").strip().lower()
    ref = (ref or "").strip().lower()
    try:
        return sacrebleu.corpus_bleu([hyp], [[ref]]).score / 100.0  # 0~1
    except Exception:
        return 0.0


# ---------- 定义本地模型根目录 ----------
MODEL_ROOT = "/root/autodl-tmp/pycharm_Mirabel/models"

# ---------- PPL (使用生成模型本身计算，替代 GPT-2) ----------

def compute_ppl_with_model(text: str, model, tokenizer) -> float:
    """
    使用传入的 LLM (Victim Model) 计算文本的 Perplexity。
    符合 S2MIA 论文原意：利用生成模型的置信度。
    """
    if not text or model is None or tokenizer is None:
        return 1e9  # 返回一个极大的值代表高困惑度

    try:
        # 编码输入
        inputs = tokenizer(text, return_tensors="pt")
        # 移动到模型所在的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            # 计算 Loss (CrossEntropy)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        if loss is None:
            return 1e9

        # PPL = exp(loss)
        ppl = torch.exp(loss).item()

        # 处理数值溢出
        if np.isnan(ppl) or np.isinf(ppl):
            return 1e9

        return float(ppl)

    except Exception as e:
        print(f"[PPL Error] {e}")
        return 1e9


# ---------- 阈值搜索（S2MIA-T） ----------
def grid_search_thresholds(
        ref_scores: List[Tuple[float, float]],  # (Metric, PPL)
        ref_labels: List[int],
        metric_grid: Optional[List[float]] = None,  # 通常是 BERTScore 或 BLEU
        ppl_grid: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    在参考集上搜索最优阈值组合。
    支持自动检测 PPL 是否有效。如果 PPL 全是 -1，则只搜索 Metric 阈值。
    """
    # 检查 PPL 是否有效 (即是否使用了本地模型)
    ppls = [p for _, p in ref_scores]
    use_ppl = any(p > 0 for p in ppls)

    if metric_grid is None:
        metric_grid = list(np.linspace(0.10, 0.95, 50))

    if use_ppl:
        if ppl_grid is None:
            # 动态生成 PPL 网格，基于数据的百分位数
            valid_ppls = [p for p in ppls if p < 1e8 and p > 0]
            if valid_ppls:
                start = np.percentile(valid_ppls, 5)
                end = np.percentile(valid_ppls, 95)
                ppl_grid = list(np.linspace(start, end, 30))
            else:
                ppl_grid = [10, 20, 50, 100, 200]
    else:
        # 如果不使用 PPL，设置一个虚拟阈值，让所有 PPL 都通过 (假设 PPL=-1 <= 1000)
        ppl_grid = [1e9]

    best = {"theta_metric": 0.5, "theta_ppl": 100.0, "acc": 0.0}
    y = np.array(ref_labels)

    for tm in metric_grid:
        for tp in ppl_grid:
            # 预测逻辑：
            # 如果是 Member：Similarity 应该高 (>= tm)，PPL 应该低 (<= tp)
            # 注意：如果 use_ppl 为 False，PPL 都是 -1，肯定 <= 1e9，所以只取决于 Metric
            preds = np.array([(1 if (m >= tm and p <= tp) else 0) for (m, p) in ref_scores])
            acc = float((preds == y).mean())

            if acc > best["acc"]:
                best = {"theta_metric": float(tm), "theta_ppl": float(tp), "acc": acc}

    return best


# ---------- 评估指标 ----------
def adjusted_accuracy(acc: float) -> float:
    """adjusted acc = max(acc, 1-acc) - 0.5，范围0~0.5，越低越好。"""
    return max(acc, 1.0 - acc) - 0.5


def ks_statistic(values_member: List[float], values_nonmember: List[float]) -> float:
    """KS 统计量，越小越不可分。"""
    if not values_member or not values_nonmember:
        return 0.0
    return float(ks_2samp(values_member, values_nonmember).statistic)


# ---------- BERTScore (单例加载) ----------
_bert_scorer = None


def compute_bertscore_single(hyp: str, ref: str) -> float:
    global _bert_scorer
    from bert_score import BERTScorer

    if _bert_scorer is None:
        local_roberta = f"{MODEL_ROOT}/AI-ModelScope/roberta-large"
        print(f"[BERTScore] Loading local Roberta from: {local_roberta}")
        _bert_scorer = BERTScorer(lang="en", model_type=local_roberta, num_layers=17, rescale_with_baseline=False,
                                  device=None)

    hyp = (hyp or "").strip()
    ref = (ref or "").strip()
    if not hyp or not ref:
        return 0.0

    try:
        P, R, F1 = _bert_scorer.score([hyp], [ref])
        score = float(F1.mean().item())
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"\n[BERTScore ERROR] {e}")
        return 0.0


def check_id_recall(target_id: int, retrieved_ids: List[int], top_k_list: List[int] = [1, 2, 3]) -> Dict[str, int]:
    stats = {}
    for k in top_k_list:
        current_k_ids = retrieved_ids[:k]
        stats[f"recall@{k}"] = 1 if target_id in current_k_ids else 0
    return stats
