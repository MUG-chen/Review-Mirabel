# src/attacks/s2mia_utils.py
from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import numpy as np
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

# ---------- PPL（单例加载 GPT-2，避免反复加载） ----------

_gpt2_tok = None
_gpt2_mdl = None
def _get_gpt2():
    global _gpt2_tok, _gpt2_mdl
    if _gpt2_tok is None or _gpt2_mdl is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # [修改] 指向本地 gpt2
        local_gpt2 = f"{MODEL_ROOT}/openai-community/gpt2"
        print(f"[PPL] Loading local GPT2 from: {local_gpt2}")
        _gpt2_tok = AutoTokenizer.from_pretrained(local_gpt2)
        _gpt2_mdl = AutoModelForCausalLM.from_pretrained(local_gpt2)
        _gpt2_mdl.eval()
    return _gpt2_tok, _gpt2_mdl

def compute_ppl_gpt2(text: str) -> float:
    import torch
    tok, mdl = _get_gpt2()
    inputs = tok((text or ""), return_tensors="pt")
    with torch.no_grad():
        out = mdl(**inputs, labels=inputs["input_ids"])
        loss = float(out.loss.item())
    ppl = float(np.exp(loss))
    return ppl

# ---------- 阈值搜索（S2MIA-T） ----------
def grid_search_thresholds(
    ref_scores: List[Tuple[float, float]],
    ref_labels: List[int],
    bleu_grid: Optional[List[float]] = None,
    ppl_grid: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    在参考集上为 (BLEU, PPL) 搜索最优阈值组合。
    默认网格适合先跑通，可按需加密。
    """
    if bleu_grid is None:
        bleu_grid = list(np.linspace(0.10, 0.90, 41))  # 0.10~0.90，步长0.02
    if ppl_grid is None:
        ppl_grid = [20, 30, 50, 80, 100, 150, 200, 250, 300]

    best = {"theta_bleu": 0.5, "theta_ppl": 100.0, "acc": 0.0}
    y = np.array(ref_labels)
    for tb in bleu_grid:
        for tp in ppl_grid:
            preds = np.array([(1 if (b >= tb and p <= tp) else 0) for (b, p) in ref_scores])
            acc = float((preds == y).mean())
            if acc > best["acc"]:
                best = {"theta_bleu": float(tb), "theta_ppl": float(tp), "acc": acc}
    return best

# ---------- 评估指标 ----------
def adjusted_accuracy(acc: float) -> float:
    """adjusted acc = max(acc, 1-acc) - 0.5，范围0~0.5，越低越好。"""
    return max(acc, 1.0 - acc) - 0.5

def ks_statistic(values_member: List[float], values_nonmember: List[float]) -> float:
    """KS 统计量，越小越不可分。"""
    return float(ks_2samp(values_member, values_nonmember).statistic)


# ---------- BERTScore (单例加载，避免反复加载) ----------
_bert_scorer = None


def compute_bertscore_single(hyp: str, ref: str) -> float:
    """
    计算单个假设-参考对的BERTScore F1值。
    使用单例模式来懒加载模型，提高效率。
    """
    global _bert_scorer
    from bert_score import BERTScorer

    # 首次调用时加载模型
    if _bert_scorer is None:
        # [修改] 指向本地 roberta-large
        local_roberta = f"{MODEL_ROOT}/AI-ModelScope/roberta-large"
        print(f"[BERTScore] Loading local Roberta from: {local_roberta}")

        # 注意：BERTScorer 的 model_type 参数如果传入路径，它就会读本地
        _bert_scorer = BERTScorer(lang="en", model_type=local_roberta, num_layers=17, rescale_with_baseline=False, device=None)


    hyp = (hyp or "").strip()
    ref = (ref or "").strip()
    if not hyp or not ref:
        return 0.0

    try:
        # BERTScorer.score() 需要一个假设列表和一个参考列表
        P, R, F1 = _bert_scorer.score([hyp], [ref])

        # 我们使用 F1 分数作为最终的相似度得分
        score = float(F1.mean().item())
        # BERTScore有时会返回略小于0或略大于1的值，做一下裁剪
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"\n[BERTScore ERROR] Failed to compute score. Error: {e}")
        return 0.0


# [新增] 召回率检查辅助函数
def check_id_recall(target_id: int, retrieved_ids: List[int], top_k_list: List[int] = [1, 2, 3]) -> Dict[str, int]:
    """
    检查目标文档ID是否出现在检索到的ID列表中。

    Args:
        target_id: 目标文档的行号 (int)
        retrieved_ids: RAG检索到的文档ID列表 (List[int])
        top_k_list: 需要统计的Top-K指标 (List[int])

    Returns:
        Dict[str, int]: 例如 {'recall@1': 0, 'recall@3': 1}
    """
    stats = {}
    for k in top_k_list:
        # 截取前 k 个检索到的 ID
        # 注意：如果 retrieved_ids 长度不足 k，切片会自动处理
        current_k_ids = retrieved_ids[:k]

        # 判断是否存在 (1为命中，0为未命中)
        stats[f"recall@{k}"] = 1 if target_id in current_k_ids else 0
    return stats