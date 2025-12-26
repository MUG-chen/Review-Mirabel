# 文件: src/attacks/ia_pipeline.py

from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import json
from typing import Dict, Any, List, Optional, Tuple

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from utils.vector_utils import l2_normalize
from mirabel.apply_nfcorpus import detect_and_hide_for_query
from rag.generate_llamaindex import generate_answer_via_api, generate_answer_via_pad_local, PADConfig, build_rag_prompt
from attacks.ia_utils import parse_yes_no
from attacks.ia_utils import rewrite_query_for_retrieval
from defense.prompt_guard import is_attack_query
from defense.rewrite import paraphrase_query


def _read_proxies_from_env() -> Optional[dict]:
    http_proxy, https_proxy = os.getenv("HTTP_PROXY"), os.getenv("HTTPS_PROXY")
    return {"http": http_proxy, "https": https_proxy} if http_proxy or https_proxy else None


def retrieve_topk_for_ia(query_text: str, index: faiss.Index, mapping: Dict[str, Dict], embedder: BGEM3FlagModel,
                         top_k: int = 3) -> Tuple[List[str], List[int]]:
    """
    执行检索，返回 (文本列表, 全局ID列表)
    """
    q_out = embedder.encode([query_text], batch_size=1)
    q_vec = l2_normalize(q_out["dense_vecs"])
    D, I = index.search(q_vec, top_k)
    cand_ids = I[0].astype("int64")
    raw_local_ids = list(map(int, cand_ids))

    final_global_ids = []
    contexts = []
    for local_id in raw_local_ids:
        record = mapping.get(str(local_id))
        if record:
            final_global_ids.append(int(record.get("orig_row")))
            txt = (record.get("text", "") or "").replace("\n", " ")
            contexts.append(txt)
        else:
            # 这种情况极少发生，兜底
            final_global_ids.append(-1)
            contexts.append("")
    return contexts, final_global_ids


def get_rag_responses(
        queries: List[str],
        defense: bool,
        defense_mode: str,
        oracle_context: Optional[str],
        index: Optional[faiss.Index],
        mapping: Optional[Dict[str, Dict]],
        embedder: Optional[BGEM3FlagModel],
        index_dir_member: Optional[str],
        mirabel_cfg: Optional[Dict[str, Any]],
        pad_cfg: Optional[PADConfig] = None,
        pad_model_id_or_path: Optional[str] = None,
        top_k: int = 3,
        per_ctx_clip: int = 600,
        max_tokens: int = 32
) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    获取 RAG 回答，并根据是否开启防御以及防御模式，选择不同的策略。

    Returns:
        raw_responses: LLM 原始回答列表
        parsed_responses: 解析后的回答列表
        all_retrieved_ids_per_query: 每个查询检索到的全局ID列表 [ [id1, id2], [id3, id4], ... ]
    """
    raw_responses, parsed_responses = [], []
    all_retrieved_ids_per_query = [] # 收集每个问题的检索ID
    debug_logs = []  # <--- [新增] 用于存储 Prompt 和 Response 的日志列表
    proxies = _read_proxies_from_env()

    for query in queries:
        contexts = []
        current_retrieved_ids = [] # 当前问题的检索ID
        current_query = query

        if defense:
            if defense_mode == 'prompt_guard':
                if is_attack_query(query):
                    print(f"[PromptGuard] Attack query detected and blocked: '{query[:70]}...'")
                    raw_responses.append("[Blocked by Prompt Guard]")
                    parsed_responses.append("unknown")
                    # 'continue' 会立即跳过本次循环的剩余部分，
                    # 直接开始处理下一个查询。因此，后面的 RAG 调用不会被执行。
                    all_retrieved_ids_per_query.append([]) # 被拦截，无检索ID
                    continue
            elif defense_mode == 'rewrite':
                print(f"[RewriteDefense] Original IA query: '{query[:70]}...'")
                # 对原始查询进行改写，并更新 current_query
                current_query = paraphrase_query(query)
                print(f"[RewriteDefense] Rewritten IA query: '{current_query[:70]}...'")

        # 第二步：上下文获取
        # 如果代码执行到这里，说明查询是安全的（或未开启Prompt Guard）
        # 现在根据情况获取上下文
        if oracle_context:
            contexts = [oracle_context]
            current_retrieved_ids = []
            if top_k > 1 and index and mapping and embedder:
                distractors = retrieve_topk_for_ia(query, index, mapping, embedder, top_k=top_k)
                contexts.extend([d for d in distractors if d != oracle_context][:top_k - 1])

        elif defense and defense_mode == 'mirabel':
            # Mirabel 防御逻辑
            if not (index_dir_member and mirabel_cfg): raise ValueError(
                "Mirabel defense mode requires 'index_dir_member' and 'mirabel_cfg'.")
            mirabel_result = detect_and_hide_for_query(query=query, index_dir=index_dir_member, **mirabel_cfg,
                                                       top_k=top_k)
            contexts = mirabel_result["final_topk_snippets"]
            current_retrieved_ids = mirabel_result.get("final_topk_ids", [])

        else:
            # 无防御 / Prompt Guard 安全模式 / Rewrite 模式
            # 无防御或Prompt Guard安全时, current_query == query
            # Rewrite模式时, current_query 是改写后的查询
            if not (index and mapping and embedder): raise ValueError(
                "Standard RAG process requires 'index', 'mapping', and 'embedder'.")  # 修改了报错信息

            if defense and defense_mode == 'rewrite':
                retrieval_query = current_query
            else:
                retrieval_query = rewrite_query_for_retrieval(query)
            contexts, current_retrieved_ids = retrieve_topk_for_ia(retrieval_query, index, mapping, embedder, top_k=top_k)

        all_retrieved_ids_per_query.append(current_retrieved_ids)

        # 第三步：RAG 生成回答 (此部分逻辑不变)
        clipped = [(c or "")[:per_ctx_clip] for c in contexts]
        ctx_block = "\n".join([f"- {c}" for c in clipped])

        full_prompt_log = build_rag_prompt(contexts, current_query, per_ctx_clip=per_ctx_clip)

        if defense and defense_mode == "pad":
            if pad_cfg is None:
                # 兜底：避免忘记传 cfg
                pad_cfg = PADConfig(max_new_tokens=max_tokens, temperature=0.0, top_p=1.0)
            else:
                pad_cfg.max_new_tokens = max_tokens  # 对齐 pipeline 的 max_tokens
            raw_answer, _pad_report = generate_answer_via_pad_local(
                contexts=contexts,
                question=current_query,
                pad_cfg=pad_cfg,
                model_id_or_path=pad_model_id_or_path,  # 可为 None，则读 PAD_MODEL_ID
                per_ctx_clip=per_ctx_clip,
            )
        else:
            raw_answer = generate_answer_via_api(
                contexts=contexts,
                question=current_query,
                per_ctx_clip=per_ctx_clip,
                max_tokens=max_tokens,
                temperature=0.0,
                proxies=proxies
            )

        raw_responses.append(raw_answer)
        parsed_responses.append(parse_yes_no(raw_answer))

        # 3. [新增] 记录日志
        debug_logs.append({
            "question": current_query,
            "full_prompt": full_prompt_log,  # <--- 学长要的完整 Prompt
            "model_response": raw_answer  # <--- 学长要的原始回复
        })

    return raw_responses, parsed_responses, all_retrieved_ids_per_query, debug_logs
