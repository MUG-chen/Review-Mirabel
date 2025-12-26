# src/rag/build_index_nfcorpus.py
from __future__ import annotations
import os, json
import numpy as np
import faiss
from beir.datasets.data_loader import GenericDataLoader
from FlagEmbedding import BGEM3FlagModel

def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def main():
    data_root = os.path.join("data", "nfcorpus")
    index_dir = "index"
    os.makedirs(index_dir, exist_ok=True)

    corpus, _, _ = GenericDataLoader(data_folder=data_root).load(split="test")
    doc_ids = list(corpus.keys())

    # 为了先跑通，用子集（例如前 5000 篇）；跑通后再改为全量 len(doc_ids)
    max_docs = min(5000, len(doc_ids))
    doc_ids = doc_ids[:max_docs]

    texts = []
    for did in doc_ids:
        entry = corpus[did]
        title = entry.get("title", "")
        text = entry.get("text", "")
        full = (title + " " + text).strip()
        texts.append(full)

    print(f"Encoding {len(texts)} docs with bge-m3...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    out = model.encode(texts, batch_size=64)  # 根据内存/显存调整 batch_size
    dense = l2_normalize(out["dense_vecs"])   # (N, 1024)

    dim = dense.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense)
    index_path = os.path.join(index_dir, "nf.index")
    faiss.write_index(index, index_path)
    print("FAISS index saved to:", index_path)

    # 保存 doc 映射（id -> 原文）
    mapping = {i: {"doc_id": doc_ids[i], "text": texts[i]} for i in range(len(doc_ids))}
    map_path = os.path.join(index_dir, "nf_docs.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    print("Doc mapping saved to:", map_path)

if __name__ == "__main__":
    main()