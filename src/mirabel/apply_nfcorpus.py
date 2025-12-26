# src/mirabel/apply_nfcorpus.py
from __future__ import annotations

import os
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

from mirabel.threshold import mirabel_threshold, mirabel_decision_with_gap
from rag.generate_llamaindex import generate_answer_via_api

# ----------------------------
# Helpers
# ----------------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 归一化，使内积近似余弦相似度。"""
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def load_index_and_mapping(index_dir: str = "index") -> Tuple[faiss.IndexFlatIP, Dict[str, Dict[str, Any]]]:
    """
    自适应加载索引与映射：
    - 若 index_dir 下存在成员索引文件（nf_member.index / nf_member_docs.json），则优先加载成员索引；
    - 否则加载全量索引（nf.index / nf_docs.json）。
    """
    # 成员索引文件名
    member_index_path = os.path.join(index_dir, "nf_member.index")
    member_map_path = os.path.join(index_dir, "nf_member_docs.json")
    # 全量索引文件名
    full_index_path = os.path.join(index_dir, "nf.index")
    full_map_path = os.path.join(index_dir, "nf_docs.json")

    if os.path.exists(member_index_path) and os.path.exists(member_map_path):
        index_path = member_index_path
        map_path = member_map_path
    elif os.path.exists(full_index_path) and os.path.exists(full_map_path):
        index_path = full_index_path
        map_path = full_map_path
    else:
        # 提示具体缺失的文件，便于排查
        raise FileNotFoundError(
            f"Could not find index or mapping in '{index_dir}'. "
            f"Checked:\n  {member_index_path}\n  {member_map_path}\n  {full_index_path}\n  {full_map_path}"
        )

    # 读取索引与映射
    index = faiss.read_index(index_path)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # 日志打印绝对路径，方便确认加载的是成员索引还是全量索引
    print(f"[load_index_and_mapping] Using index: {os.path.abspath(index_path)}")
    print(f"[load_index_and_mapping] Using mapping: {os.path.abspath(map_path)}")

    return index, mapping

# ----------------------------
# Core: Detect-and-Hide
# ----------------------------
def detect_and_hide_for_query(
    query: str,
    rho: float = 0.005,
    margin: Optional[float] = 0.02,
    gap_min: Optional[float] = 0.03,
    use_gap: bool = True,
    use_full_corpus: bool = True,
    topM_if_not_full: int = 200,
    top_k: int = 3,
    use_precise_mu: bool = True,
    index_dir: str = "index",
    snippet_clip: int = 120,
) -> Dict[str, Any]:
    """
    对单条查询执行：检索 → Mirabel 判定（可选 margin/gap）→ detect-and-hide。

    Parameters
    - query: 查询文本
    - rho: Gumbel 显著性水平；越小越保守
    - margin: 安全边距；None 表示不使用 margin
    - gap_min: top-1 主导差门槛；None 或 use_gap=False 表示不使用 gap
    - use_gap: 是否启用“Gumbel + margin + gap”的组合策略
    - use_full_corpus: True 用全库计算阈值（更稳），False 用限定候选池
    - topM_if_not_full: 非全库时的候选池大小（默认 200）
    - top_k: 最终返回给生成模型的文档数
    - use_precise_mu: True 使用 μ_n 精确式；False 使用近似式
    - index_dir: 索引与映射目录
    - snippet_clip: 返回的片段裁剪长度（字符数），用于日志/提示构造前的展示

    Returns
    - 一个包含判定细节与最终 top-k 的字典，便于上游生成与日志记录。
    """
    # 1) 加载索引与映射 + 嵌入模型
    index, mapping = load_index_and_mapping(index_dir=index_dir)
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # 2) 编码查询
    q_out = embedder.encode([query], batch_size=1)
    q_vec = l2_normalize(q_out["dense_vecs"])  # (1, 1024)

    # 3) 候选池规模
    topM_full = index.ntotal
    topM = topM_full if use_full_corpus else min(topM_if_not_full, topM_full)

    # 4) 检索候选池
    D, I = index.search(q_vec, topM)
    scores = D[0].astype("float32")
    cand_ids = I[0].astype("int64")

    # 5) Mirabel 判定（两种策略：组合 或 仅 Gumbel / Gumbel+margin）
    if use_gap and (gap_min is not None):
        is_member, tau, s_max, target_pos, s2, gap, res = mirabel_decision_with_gap(
            scores,
            rho=rho,
            margin=(0.0 if margin is None else margin),
            gap_min=gap_min,
            use_precise_mu=use_precise_mu,
        )
    else:
        res = mirabel_threshold(scores, rho=rho, use_precise_mu=use_precise_mu)
        s_max, tau = res.s_max, res.tau
        target_pos = res.target_idx
        s2 = float(np.max(scores[np.arange(scores.size) != target_pos])) if scores.size > 1 else s_max
        gap = s_max - s2
        if margin is None:
            is_member = bool(s_max > tau)                 # 仅 Gumbel
        else:
            is_member = bool(s_max > (tau + margin))      # Gumbel + margin

    target_doc_id = int(cand_ids[target_pos])

    # 6) detect-and-hide：在最终 top-k 中移除目标文档并补位
    final_ids: List[int] = list(map(int, cand_ids[:top_k]))
    if is_member and target_doc_id in final_ids:
        final_ids.remove(target_doc_id)
        for did in cand_ids[top_k:]:
            did_i = int(did)
            if did_i not in final_ids:
                final_ids.append(did_i)
                if len(final_ids) >= top_k:
                    break

    final_snippets = [
        (mapping.get(str(did), {}).get("text", "") or "").replace("\n", " ")[:snippet_clip]
        for did in final_ids
    ]

    # 7) 构造返回
    result = {
        "query": query,
        "rho": rho,
        "margin": margin,
        "gap_min": gap_min,
        "use_gap": use_gap,
        "use_full_corpus": use_full_corpus,
        "topM": int(topM),
        "top_k": int(top_k),
        "use_precise_mu": use_precise_mu,
        "s_max": float(s_max),
        "tau": float(tau),
        "s2": float(s2),
        "gap": float(gap),
        "is_member": bool(is_member),
        "target_pos": int(target_pos),
        "target_doc_id": target_doc_id,
        "n": int(res.n),
        "mu_q": float(res.mu_q),
        "sigma_q": float(res.sigma_q),
        "mu_n": float(res.mu_n),
        "beta_n": float(res.beta_n),
        "final_topk_ids": final_ids,
        "final_topk_snippets": final_snippets,
    }

    # 8) 打印关键信息（便于即时观察）
    print(f"\nQuery: {query}")
    print(
        f"s_max={result['s_max']:.4f}, tau={result['tau']:.4f}, "
        f"s_max-tau={result['s_max'] - result['tau']:.4f}, s2={result['s2']:.4f}, gap={result['gap']:.4f}, "
        f"is_member={result['is_member']}, target_doc_id={result['target_doc_id']}"
    )
    print("Final top-k after detect-and-hide:")
    for did, snip in zip(result["final_topk_ids"], result["final_topk_snippets"]):
        print(f"  id={did}\t{snip}")
    print(
        f"n={result['n']}, mu_q={result['mu_q']:.4f}, sigma_q={result['sigma_q']:.4f}, "
        f"mu_n={result['mu_n']:.4f}, beta_n={result['beta_n']:.4f}"
    )

    return result

# ----------------------------
# Generation via LlamaIndex OpenAI-compatible API
# ----------------------------
def run_with_llamaindex_answer(
    query_text: str,
    rho: float = 0.005, margin: Optional[float] = 0.02, gap_min: Optional[float] = 0.03,
    use_gap: bool = True, use_full_corpus: bool = True,
    topM_if_not_full: int = 200, top_k: int = 3, use_precise_mu: bool = True,
    # 下面三个参数不需要显式传，默认从 .env 读取；保留以防需要覆盖
    model_id: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None,
    per_ctx_clip: int = 600, max_tokens: int = 256,
) -> Dict[str, Any]:
    # 1) Mirabel 检测 + detect-and-hide
    result = detect_and_hide_for_query(
        query=query_text,
        rho=rho, margin=margin, gap_min=gap_min, use_gap=use_gap,
        use_full_corpus=use_full_corpus, topM_if_not_full=topM_if_not_full,
        top_k=top_k, use_precise_mu=use_precise_mu
    )
    contexts = result["final_topk_snippets"]

    # 2) 代理（如需要）
    proxies = {
        "http": os.getenv("HTTP_PROXY"),
        "https": os.getenv("HTTPS_PROXY")
    }

    # 3) 调用请求版生成器
    answer = generate_answer_via_api(
        contexts=contexts,
        question=query_text,
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        timeout=90,
        retries=5,
        proxies=proxies,
        per_ctx_clip=per_ctx_clip,
    )

    print("\nGenerated Answer (via OpenAI-compatible API):\n", answer)
    return {"mirabel": result, "answer": answer}

# ----------------------------
# Demo / CLI entry
# ----------------------------
def main():
    print("CWD:", os.getcwd())

    # 示例：仅做 detect-and-hide（打印统计与最终 top-k）
    query = "What are common symptoms related to medical conditions?"
    _ = detect_and_hide_for_query(
        query=query,
        rho=0.005,
        margin=0.02,        # 设置为 None 可关闭 margin
        gap_min=0.03,       # 设置为 None 或 use_gap=False 可关闭 gap
        use_gap=True,
        use_full_corpus=True,
        topM_if_not_full=200,
        top_k=3,
        use_precise_mu=True,
        index_dir="index",
    )

    # 你也可以测试“仅 Gumbel”或“Gumbel + margin”的版本：
    # 仅 Gumbel：
    # _ = detect_and_hide_for_query(query, rho=0.005, margin=None, gap_min=None, use_gap=False, use_full_corpus=True, top_k=3)
    # Gumbel + margin（无 gap）：
    # _ = detect_and_hide_for_query(query, rho=0.005, margin=0.02, gap_min=None, use_gap=False, use_full_corpus=True, top_k=3)

if __name__ == "__main__":
    print("CWD:", os.getcwd())
    out = run_with_llamaindex_answer(
        query_text="What are common symptoms related to medical conditions?",
        rho=0.005, margin=0.02, gap_min=0.03, use_gap=True,
        use_full_corpus=True, topM_if_not_full=200, top_k=3, use_precise_mu=True,
        # 通常不显式传，直接从 .env 读取：
        # model_id="llama-3.1-8b", api_base="https://jeniya.top/v1", api_key="sk-xxxxx"
    )