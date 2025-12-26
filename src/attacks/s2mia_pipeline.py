# src/attacks/s2mia_pipeline.py
from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

from mirabel.apply_nfcorpus import detect_and_hide_for_query, l2_normalize
from rag.generate_llamaindex import (
    generate_answer_via_api,        # 通用问答
    generate_s2mia_via_api,         # 续写专用
)
from defense.prompt_guard import is_attack_query
from defense.rewrite import paraphrase_query
from defense.pad_config import PADConfig
from rag.pad_local_generate import generate_with_pad_core

def load_index_and_mapping(index_dir: str) -> Tuple[faiss.IndexFlatIP, Dict[str, Dict[str, Any]]]:
    member_index_path = os.path.join(index_dir, "nf_member.index")
    member_map_path = os.path.join(index_dir, "nf_member_docs.json")
    if os.path.exists(member_index_path) and os.path.exists(member_map_path):
        index_path = member_index_path
        map_path = member_map_path
    else:
        index_path = os.path.join(index_dir, "nf.index")
        map_path = os.path.join(index_dir, "nf_docs.json")
    index = faiss.read_index(index_path)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return index, mapping

def retrieve_topk_no_defense(excerpt_or_query: str, index_dir: str, topM: Optional[int] = None, top_k: int = 3) -> Dict[str, Any]:
    index, mapping = load_index_and_mapping(index_dir=index_dir)
    # 指向你下载好的 bge-m3 绝对路径
    local_bge_path = "/root/autodl-tmp/pycharm_Mirabel/models/BAAI/bge-m3"
    embedder = BGEM3FlagModel(local_bge_path, use_fp16=True)
    q_out = embedder.encode([excerpt_or_query], batch_size=1)
    q_vec = l2_normalize(q_out["dense_vecs"])

    M = index.ntotal if (topM is None) else min(topM, index.ntotal)
    D, I = index.search(q_vec, M)
    cand_ids = I[0].astype("int64")

    # 这些是 Faiss 的内部 ID (0, 1, 2...)
    raw_local_ids = list(map(int, cand_ids[:top_k]))
    final_global_ids = []
    contexts = []
    for local_id in raw_local_ids:
        # 从 mapping 中获取对应的记录
        # mapping 的 key 是字符串类型的 local_id
        record = mapping.get(str(local_id))

        if record:
            # 获取原始行号 (int)
            orig_row = record.get("orig_row")
            final_global_ids.append(int(orig_row))

            # 获取文本上下文
            txt = (record.get("text", "") or "").replace("\n", " ")
            contexts.append(txt)
        else:
            # 理论上不应该发生，除非索引和mapping文件不匹配
            print(f"[Warn] Local ID {local_id} not found in mapping!")
            final_global_ids.append(-1)
            contexts.append("")
        # 返回转换后的 Global IDs
    return {"final_topk_ids": final_global_ids, "final_topk_contexts": contexts}

def _read_proxies_from_env() -> Optional[dict]:
    http_proxy = os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY")
    if http_proxy or https_proxy:
        return {"http": http_proxy, "https": https_proxy}
    return None

# ---------- 通用问答版（供 apply_nfcorpus / pipeline_smoke 使用） ----------
def generate_answer_no_defense(
    query: str,
    index_dir: str = "index_member",
    topM: Optional[int] = None, top_k: int = 3,
    per_ctx_clip: int = 600, max_tokens: int = 256
) -> Dict[str, Any]:
    pack = retrieve_topk_no_defense(query, index_dir=index_dir, topM=topM, top_k=top_k)
    contexts = pack["final_topk_contexts"]
    proxies = _read_proxies_from_env()
    answer = generate_answer_via_api(
        contexts=contexts, question=query,
        max_tokens=max_tokens, temperature=0.0, top_p=1.0,
        timeout=90, retries=5, proxies=proxies, per_ctx_clip=per_ctx_clip
    )
    return {"contexts": contexts, "answer": answer, "final_topk_ids": pack["final_topk_ids"]}

def generate_answer_with_defense(
    query: str,
    index_dir: str = "index_member",
    rho: float = 0.005, margin: Optional[float] = 0.02, gap_min: Optional[float] = 0.03,
    use_gap: bool = True, use_full_corpus: bool = True, topM_if_not_full: int = 200,
    top_k: int = 3, use_precise_mu: bool = True,
    per_ctx_clip: int = 600, max_tokens: int = 256
) -> Dict[str, Any]:
    res = detect_and_hide_for_query(
        query=query, rho=rho, margin=margin, gap_min=gap_min, use_gap=use_gap,
        use_full_corpus=use_full_corpus, topM_if_not_full=topM_if_not_full,
        top_k=top_k, use_precise_mu=use_precise_mu, index_dir=index_dir,
        snippet_clip=1000
    )
    contexts = res["final_topk_snippets"]
    proxies = _read_proxies_from_env()
    answer = generate_answer_via_api(
        contexts=contexts, question=query,
        max_tokens=max_tokens, temperature=0.0, top_p=1.0,
        timeout=90, retries=5, proxies=proxies, per_ctx_clip=per_ctx_clip
    )
    return {"mirabel": res, "contexts": contexts, "answer": answer}

# ---------- 续写 S2MIA 版（供 s2mia.py 使用） ----------
def generate_s2mia_no_defense(
    excerpt: str,
    index_dir: str = "index_member",
    topM: Optional[int] = None, top_k: int = 3,
    per_ctx_clip: int = 600, max_tokens: int = 256, cont_words: int = 120
) -> Dict[str, Any]:
    pack = retrieve_topk_no_defense(excerpt, index_dir=index_dir, topM=topM, top_k=top_k)
    contexts = pack["final_topk_contexts"]
    proxies = _read_proxies_from_env()
    answer = generate_s2mia_via_api(
        contexts=contexts, excerpt=excerpt,
        max_tokens=max_tokens, temperature=0.0, top_p=1.0,
        timeout=90, retries=5, proxies=proxies,
        per_ctx_clip=per_ctx_clip, cont_words=cont_words
    )
    return {"contexts": contexts, "answer": answer, "final_topk_ids": pack["final_topk_ids"]}

def generate_s2mia_with_defense(
    excerpt: str,
    index_dir: str = "index_member",
    rho: float = 0.005, margin: Optional[float] = 0.02, gap_min: Optional[float] = 0.03,
    use_gap: bool = True, use_full_corpus: bool = True, topM_if_not_full: int = 200,
    top_k: int = 3, use_precise_mu: bool = True,
    per_ctx_clip: int = 600, max_tokens: int = 256, cont_words: int = 120
) -> Dict[str, Any]:
    res = detect_and_hide_for_query(
        query=excerpt, rho=rho, margin=margin, gap_min=gap_min, use_gap=use_gap,
        use_full_corpus=use_full_corpus, topM_if_not_full=topM_if_not_full,
        top_k=top_k, use_precise_mu=use_precise_mu, index_dir=index_dir,
        snippet_clip=1000
    )
    contexts = res["final_topk_snippets"]
    proxies = _read_proxies_from_env()
    answer = generate_s2mia_via_api(
        contexts=contexts, excerpt=excerpt,
        max_tokens=max_tokens, temperature=0.0, top_p=1.0,
        timeout=90, retries=5, proxies=proxies,
        per_ctx_clip=per_ctx_clip, cont_words=cont_words
    )
    retrieved_ids = res.get("final_topk_ids", [])

    return {
        "mirabel": res,
        "contexts": contexts,
        "answer": answer,
        "final_topk_ids": retrieved_ids
    }

def generate_s2mia_with_prompt_guard_defense(
    excerpt: str,
    index_dir: str = "index_member",
    topM: Optional[int] = None, top_k: int = 3,
    per_ctx_clip: int = 600, max_tokens: int = 256, cont_words: int = 120
) -> Dict[str, Any]:
    """
    使用 Prompt Guard 防御的 S2MIA RAG 流水线。
    1. 首先检查查询是否为攻击。
    2. 如果是攻击，则直接拦截并返回空结果。
    3. 如果不是攻击，则按无防御流程继续。
    """
    # 步骤 1: 使用 Prompt Guard 进行前置检查
    if is_attack_query(excerpt):
        # 如果检测到攻击，返回一个模拟的、空的RAG输出
        # s2mia.py 中的 eval_one_doc 会将空 answer 视为攻击失败
        return {"contexts": [], "answer": "", "final_topk_ids": []}

    # 步骤 2: 如果查询被认为是安全的，则正常执行无防御的 RAG 流程
    # 这是一种“放行”策略
    print("[PromptGuard] Query deemed safe, proceeding with standard RAG.")
    return generate_s2mia_no_defense(
        excerpt=excerpt,
        index_dir=index_dir,
        topM=topM,
        top_k=top_k,
        per_ctx_clip=per_ctx_clip,
        max_tokens=max_tokens,
        cont_words=cont_words,
    )


def generate_s2mia_with_rewrite_defense(
        excerpt: str,
        index_dir: str = "index_member",
        topM: Optional[int] = None, top_k: int = 3,
        per_ctx_clip: int = 600, max_tokens: int = 256, cont_words: int = 120
) -> Dict[str, Any]:
    """
    ✅ 新增函数：使用“基于改写的防御方式”的S2MIA RAG流水线。
    1. 首先调用LLM改写原始查询。
    2. 然后使用改写后的查询执行标准的、无防御的RAG流程。
    """
    print("[Defense] Activating Rewrite-Based Defense...")

    # 步骤 1: 改写查询
    rewritten_excerpt = paraphrase_query(excerpt)

    # 步骤 2: 使用改写后的查询执行无防御的RAG流程
    return generate_s2mia_no_defense(
        excerpt=rewritten_excerpt,
        index_dir=index_dir,
        topM=topM,
        top_k=top_k,
        per_ctx_clip=per_ctx_clip,
        max_tokens=max_tokens,
        cont_words=cont_words,
    )


def generate_s2mia_with_pad_defense(
        excerpt: str,
        index_dir: str,
        model,  # 必须传入已加载的本地模型
        tokenizer,  # 必须传入已加载的本地tokenizer
        top_k: int = 3,
        per_ctx_clip: int = 600,
        max_tokens: int = 96,
        cont_words: int = 120  # 保留接口兼容，实际由 max_tokens 控制生成长度
) -> Dict[str, Any]:
    """
    使用 PAD 防御的 S2MIA RAG 流程。
    """
    # 1. 正常检索
    pack = retrieve_topk_no_defense(excerpt, index_dir=index_dir, top_k=top_k)
    contexts = pack["final_topk_contexts"]
    final_ids = pack["final_topk_ids"]
    # 2. 构建 Prompt (专为续写任务设计)
    clipped_ctx = [(c or "").replace("\n", " ")[:per_ctx_clip] for c in contexts]
    context_block = "\n".join([f"- {c}" for c in clipped_ctx])

    # 构造 Prompt：引导模型根据 Context 续写 Excerpt
    full_prompt = (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{context_block}\n"
        f"---------------------\n"
        f"Given the context information and no prior knowledge, "
        f"continue the following text exactly as it appears in the context:\n\n"
        f"{excerpt}"
    )
    # 3. 配置 PAD 参数
    # 根据 S2MIA 的需求调整配置，例如 max_new_tokens
    pad_cfg = PADConfig(
        max_new_tokens=max_tokens,  # 使用传入的 max_tokens (s2mia 默认96)
        epsilon_base=0.2,  # 默认隐私预算，可视情况从args传入
        lambda_amp=3.0,
        temperature=0.0  # S2MIA 偏好确定性生成
    )
    # 4. 调用本地 PAD 生成核心
    # generate_with_pad_core 来自 rag.pad_local_generate
    generated_text, pad_report = generate_with_pad_core(
        full_prompt=full_prompt,
        pad_cfg=pad_cfg,
        model=model,
        tokenizer=tokenizer
    )

    # 5. 返回兼容格式
    return {
        "contexts": contexts,
        "answer": generated_text,  # 续写内容
        "final_topk_ids": final_ids,
        "pad_report": pad_report
    }