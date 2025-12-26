# src/defense/rewrite.py
from __future__ import annotations
import os
from rag.generate_llamaindex import paraphrase_via_api


def paraphrase_query(query: str) -> str:
    """
    使用一个强大的LLM来改写（意译）给定的查询。
    Args:
        query: 原始查询字符串。
    Returns:
        经过改写的新查询字符串。
    """
    print(f"[Defense/Rewrite] Original query: '{query[:100]}...'")

    # 从环境变量中读取用于改写的模型ID，如果未设置，则使用一个强大的默认模型
    rewrite_model_id = os.getenv("REWRITE_LLM_MODEL_ID", "gpt-4-turbo-preview")

    try:
        # ✅ 关键改动：调用我们即将创建的 paraphrase_via_api 函数
        rewritten_query = paraphrase_via_api(
            text_to_paraphrase=query,
            model_id=rewrite_model_id,
            temperature=0.7,  # 增加一点创造性以获得更好的改写效果
        )

        # 清理可能存在的多余引号或前缀
        rewritten_query = rewritten_query.strip().strip('\"')

        print(f"[Defense/Rewrite] Rewritten query: '{rewritten_query[:100]}...'")

        # 如果改写失败或返回空，则为了安全返回原始查询
        if not rewritten_query:
            print("[Defense/Rewrite] WARNING: Rewriting returned empty string. Falling back to original query.")
            return query

        return rewritten_query

    except Exception as e:
        print(f"[Defense/Rewrite] ERROR: Failed to paraphrase query. Error: {e}. Falling back to original query.")
        return query
