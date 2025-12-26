# src/rag/search_nfcorpus.py
from __future__ import annotations

import os
import json
from typing import Tuple, List

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """对向量做 L2 归一化，保证内积≈余弦相似度。"""
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def load_index_and_mapping(index_dir: str = "index") -> Tuple[faiss.IndexFlatIP, dict]:
    """加载 FAISS 索引与文档映射（index/nf.index 与 index/nf_docs.json）。"""
    index_path = os.path.join(index_dir, "nf.index")
    map_path = os.path.join(index_dir, "nf_docs.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Doc mapping not found: {map_path}")

    index = faiss.read_index(index_path)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)  # 注意：json 的键是字符串形式的文档行号
    return index, mapping

def search_query(query: str, topM: int = 100) -> Tuple[np.ndarray, List[int], dict]:
    """
    对单条查询做检索，返回候选池分数、候选文档索引、映射。
    - query: 查询文本
    - topM: 候选池规模（建议 100 或更大，后续 Mirabel 阈值更稳）
    """
    # 加载索引与映射
    index, mapping = load_index_and_mapping(index_dir="index")

    # 加载嵌入模型并编码查询（只用 dense_vecs）
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    q_out = model.encode([query], batch_size=1)
    q_vec = l2_normalize(q_out["dense_vecs"])  # (1, 1024)

    # 在索引上检索候选池
    # 如果索引文档数少于 topM，FAISS 会返回上限数量
    D, I = index.search(q_vec, topM)
    scores = D[0].astype("float32")              # 相似度分数
    cand_ids = [int(x) for x in I[0].tolist()]   # 文档行号（整数）

    # 打印前若干候选（便于观察）
    print(f"\nQuery: {query}")
    print("Top candidates:")
    for sim, rid in zip(scores[:10], cand_ids[:10]):
        # 映射的键是字符串形式的行号
        entry = mapping.get(str(rid), {})
        snippet = entry.get("text", "")[:120].replace("\n", " ")
        print(f"{sim:.4f}\tidx={rid}\t{snippet}")

    return scores, cand_ids, mapping

def main():
    # 示例查询；你可以替换成更贴近医疗/科学领域的真实问题
    query = "What are common symptoms related to medical conditions?"
    # 建议候选池用 100；如果想更稳，可以设为 200
    scores, cand_ids, mapping = search_query(query, topM=100)

    # 如果你想在此脚本里直接接 Mirabel 检测，可在这里调用：
    # from mirabel.threshold import mirabel_threshold
    # res = mirabel_threshold(scores, rho=0.05, use_precise_mu=True)
    # print("\nMirabel quick check:", res)

    # 本脚本仅用于检索校验；Mirabel 的完整应用请运行 src/mirabel/apply_nfcorpus.py
    print("\nDone: retrieved", len(cand_ids), "candidates.")

if __name__ == "__main__":
    # 可选：打印当前工作目录，便于确认配置正确
    print("CWD:", os.getcwd())
    main()