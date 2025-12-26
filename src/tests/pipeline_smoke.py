# src/tests/pipeline_smoke.py
from __future__ import annotations
import os
from dotenv import load_dotenv, find_dotenv

from attacks.s2mia_pipeline import (
    generate_answer_no_defense,
    generate_answer_with_defense,
)

# 加载 .env
load_dotenv(find_dotenv(), override=False)

def main():
    # 使用成员索引（index_member），确保你已跑完第1步构建成员索引
    index_dir_member = "index_member"
    query = "What are common symptoms related to medical conditions?"

    # 无防御路径
    print("\n=== No-Defense Path ===")
    res_no_def = generate_answer_no_defense(
        query=query,
        index_dir=index_dir_member,
        topM=None,            # 用全库作为候选池
        top_k=3,
        per_ctx_clip=400,
        max_tokens=128,
    )
    print("Top-k IDs:", res_no_def["final_topk_ids"])
    print("Answer (no defense):", (res_no_def["answer"] or "")[:200])

    # 有防御路径（Mirabel detect-and-hide）
    print("\n=== Defense Path (Mirabel) ===")
    res_def = generate_answer_with_defense(
        query=query,
        index_dir=index_dir_member,
        rho=0.005,
        margin=0.02,
        gap_min=0.03,
        use_gap=True,
        use_full_corpus=True,
        topM_if_not_full=200,
        top_k=3,
        use_precise_mu=True,
        per_ctx_clip=400,
        max_tokens=128,
    )
    # 防御路径会在控制台打印 Mirabel 判定统计（来自 detect_and_hide_for_query）
    print("Answer (defense):", (res_def["answer"] or "")[:200])

if __name__ == "__main__":
    main()