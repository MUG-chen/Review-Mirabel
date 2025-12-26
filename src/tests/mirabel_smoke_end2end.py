# src/tests/mirabel_smoke_end2end.py
from __future__ import annotations

import numpy as np
import faiss

from FlagEmbedding import BGEM3FlagModel
from mirabel.threshold import mirabel_threshold

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """对向量做 L2 归一化，内积即可近似余弦相似度。"""
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_index_dense(dense_vecs: np.ndarray) -> faiss.IndexFlatIP:
    """使用 FAISS 内积索引（需向量先 L2 归一化）。"""
    dim = dense_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense_vecs)
    return index

def detect_and_hide(cand_ids: np.ndarray, scores: np.ndarray, top_k: int, rho: float = 0.05):
    """
    在候选池上计算 Mirabel 阈值；若判为成员攻击，则从最终 top-k 中移除目标文档并补位。
    返回：(is_member, tau, s_max, target_doc_id, final_topk_ids)
    """
    # 计算阈值（每查询自适应）
    res = mirabel_threshold(scores, rho=rho, use_precise_mu=True)
    is_member = res.is_member
    tau = res.tau
    s_max = res.s_max
    target_pos = res.target_idx
    target_doc_id = int(cand_ids[target_pos])

    # 最终 top-k 列表
    final_ids = list(map(int, cand_ids[:top_k]))
    if is_member and target_doc_id in final_ids:
        # 从最终 top-k 移除目标文档
        final_ids.remove(target_doc_id)
        # 用第 k+1 名及之后的候选补位（避免重复）
        for did in cand_ids[top_k:]:
            did = int(did)
            if did not in final_ids:
                final_ids.append(did)
                if len(final_ids) >= top_k:
                    break

    return is_member, tau, s_max, target_doc_id, final_ids

def main():
    # 1) 准备少量“玩具文档”（真实项目里替换为 NFCorpus 文档分块）
    docs = [
        "Deep learning enables end-to-end learning.",
        "Transformers are powerful models for NLP.",
        "COVID-19 is a respiratory disease caused by SARS-CoV-2.",
        "The Eiffel Tower is located in Paris, France.",
        "Retrieval-augmented generation reduces hallucinations in LLMs.",
        "Membership inference attacks target privacy leakage in RAG systems.",
        "Graph-based retrieval can improve specialized question answering.",
        "Differential privacy adds noise to protect sensitive information.",
    ]

    # 2) 加载 bge-m3 并编码文档与查询（只用 dense_vecs）
    # 如果你的环境是 CPU，也能运行；GPU 更快。首次加载会下载权重，需耐心等待。
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    doc_out = model.encode(docs, batch_size=len(docs))
    doc_dense = l2_normalize(doc_out["dense_vecs"])  # (N_doc, 1024)
    index = build_index_dense(doc_dense)

    # 3) 构造一个与某一文档高度相关的查询（模拟“成员攻击查询”的相似度特征）
    query = "How can retrieval augmentation reduce hallucinations in large language models?"
    q_out = model.encode([query], batch_size=1)
    q_dense = l2_normalize(q_out["dense_vecs"])  # (1, 1024)

    # 4) 先取较大的候选池（真实使用建议 topM=100；此处用文档总数）
    topM = min(len(docs), 20)
    D, I = index.search(q_dense, topM)  # D: 相似度分数；I: 文档索引
    scores = D[0].astype("float32")
    cand_ids = I[0]

    print("Candidate ranking (similarity, doc_id, snippet):")
    for sim, did in zip(scores, cand_ids):
        print(f"  {sim:.4f}\t{int(did)}\t{docs[int(did)][:70]}")

    # 5) 计算 Mirabel 阈值并执行 detect-and-hide
    top_k = 3
    rho = 0.05
    is_member, tau, s_max, target_doc_id, final_topk_ids = detect_and_hide(
        cand_ids=cand_ids, scores=scores, top_k=top_k, rho=rho
    )

    print("\nMirabel decision:")
    print(f"  s_max={s_max:.4f}, tau={tau:.4f}, is_member={is_member}, target_doc_id={target_doc_id}")

    print("\nFinal top-k after detect-and-hide:")
    for did in final_topk_ids:
        print(f"  id={did}\t{docs[did]}")

    # 6) 额外打印中间量（便于确认统计是否合理）
    from mirabel.threshold import mirabel_threshold
    res = mirabel_threshold(scores, rho=rho, use_precise_mu=True)
    print("\nDebug stats:")
    print(f"  n={res.n}, mu_q={res.mu_q:.4f}, sigma_q={res.sigma_q:.4f}, mu_n={res.mu_n:.4f}, beta_n={res.beta_n:.4f}, c={res.c:.4f}")

if __name__ == "__main__":
    main()