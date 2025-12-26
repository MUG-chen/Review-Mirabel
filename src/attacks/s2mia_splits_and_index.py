# src/attacks/s2mia_splits_and_index.py
from __future__ import annotations
import os
import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

@dataclass
class Splits:
    member_all: List[int]
    nonmember_all: List[int]
    member_ref: List[int]
    member_eval: List[int]
    non_ref: List[int]
    non_eval: List[int]

def load_full_mapping(index_dir: str = "index") -> Dict[str, Dict]:
    """加载全量索引映射（由 build_index_nfcorpus.py 生成）。"""
    map_path = os.path.join(index_dir, "nf_docs.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Mapping not found: {map_path}. Run build_index_nfcorpus.py first.")
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not mapping:
        raise ValueError("Mapping file is empty.")
    return mapping  # keys are str row indices: "0", "1", ...

def make_member_nonmember_splits(
    mapping: Dict[str, Dict],
    member_ratio: float = 0.8,
    seed: int = 42
) -> Splits:
    """对全量行号进行成员/非成员划分，并在各自内部分为 reference/evaluation (5:5)。"""
    rng = random.Random(seed)
    all_rows = sorted([int(k) for k in mapping.keys()])
    n_total = len(all_rows)
    if n_total < 10:
        raise ValueError(f"Too few docs in mapping: {n_total}")
    n_member = int(n_total * member_ratio)
    if n_member < 2 or n_total - n_member < 2:
        raise ValueError("Member/nonmember sizes are too small. Adjust member_ratio.")

    rows_shuffled = all_rows[:]
    rng.shuffle(rows_shuffled)
    member_rows = rows_shuffled[:n_member]
    nonmember_rows = rows_shuffled[n_member:]

    def split_half(rows: List[int]) -> Tuple[List[int], List[int]]:
        half = len(rows) // 2
        return rows[:half], rows[half:]

    m_ref, m_eval = split_half(member_rows)
    n_ref, n_eval = split_half(nonmember_rows)

    return Splits(
        member_all=member_rows,
        nonmember_all=nonmember_rows,
        member_ref=m_ref,
        member_eval=m_eval,
        non_ref=n_ref,
        non_eval=n_eval,
    )

def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_member_index(
    mapping: Dict[str, Dict],
    member_rows: List[int],
    out_dir: str = "index_member",
    batch_size: int = 64
) -> None:
    """仅对成员集合构建新的索引与映射文件（index_member/nf_member.index & nf_member_docs.json）。"""
    os.makedirs(out_dir, exist_ok=True)
    if not member_rows:
        raise ValueError("Empty member_rows.")

    texts: List[str] = []
    for orig_row in member_rows:
        rec = mapping.get(str(orig_row), {})
        txt = (rec.get("text", "") or "").strip()
        texts.append(txt)

    # 嵌入
    local_bge_path = "/root/autodl-tmp/pycharm_Mirabel/models/BAAI/bge-m3"
    embedder = BGEM3FlagModel(local_bge_path, use_fp16=True)
    dense_vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        out = embedder.encode(batch, batch_size=len(batch))
        dense_vecs.append(out["dense_vecs"])
    dense = np.vstack(dense_vecs)
    dense = l2_normalize(dense)

    # 构建索引
    dim = dense.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense)
    index_path = os.path.join(out_dir, "nf_member.index")
    faiss.write_index(index, index_path)

    # 新映射：新行号 -> {orig_row, doc_id, text}
    new_map = {
        str(i): {
            "orig_row": member_rows[i],
            "doc_id": mapping[str(member_rows[i])].get("doc_id", ""),
            "text": mapping[str(member_rows[i])].get("text", ""),
        }
        for i in range(len(member_rows))
    }
    map_path = os.path.join(out_dir, "nf_member_docs.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(new_map, f, ensure_ascii=False)

    print(f"[Member Index] Saved index to {index_path}")
    print(f"[Member Index] Saved mapping to {map_path}")
    print(f"[Member Index] Size: {len(member_rows)} docs")

def save_splits(splits: Splits, out_path: str = "configs/s2mia_splits.json") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    to_dump = {
        "member_all": splits.member_all,
        "nonmember_all": splits.nonmember_all,
        "member_ref": splits.member_ref,
        "member_eval": splits.member_eval,
        "non_ref": splits.non_ref,
        "non_eval": splits.non_eval,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_dump, f, ensure_ascii=False)
    print(f"[Splits] Saved to {out_path}")
    print(
        f"[Splits] member_all={len(splits.member_all)}, nonmember_all={len(splits.nonmember_all)}, "
        f"member_ref={len(splits.member_ref)}, member_eval={len(splits.member_eval)}, "
        f"non_ref={len(splits.non_ref)}, non_eval={len(splits.non_eval)}"
    )

def main():
    # 1) 读取全量映射
    mapping = load_full_mapping(index_dir="index")

    # 2) 划分成员/非成员 + 参考/评估
    splits = make_member_nonmember_splits(mapping, member_ratio=0.8, seed=42)

    # 3) 保存划分
    save_splits(splits, out_path="configs/s2mia_splits.json")

    # 4) 基于成员集合重建索引
    build_member_index(mapping, splits.member_all, out_dir="index_member", batch_size=64)

    print("[Done] Splits + Member index prepared.")

if __name__ == "__main__":
    main()