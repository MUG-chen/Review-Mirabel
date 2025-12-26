# src/tests/generator_smoke.py
from __future__ import annotations
import os
from dotenv import load_dotenv, find_dotenv
from rag.generate_llamaindex import generate_answer_via_api

# 加载 .env
load_dotenv(find_dotenv(), override=False)

def main():
    # 构造少量上下文与问题
    contexts = [
        "RAG injects retrieved knowledge into prompts to reduce hallucinations.",
        "It typically includes indexing, retrieval, reranking, and generation."
    ]
    question = "Summarize the core idea of RAG in one sentence."

    # 可选：读取代理（若设置了 HTTP_PROXY/HTTPS_PROXY）
    proxies = {
        "http": os.getenv("HTTP_PROXY"),
        "https": os.getenv("HTTPS_PROXY")
    }

    # 调用生成器（requests 版，固定参数保证稳定）
    ans = generate_answer_via_api(
        contexts=contexts,
        question=question,
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        timeout=60,
        retries=3,
        proxies=proxies,
        per_ctx_clip=400
    )
    print("Answer:", ans)

if __name__ == "__main__":
    main()