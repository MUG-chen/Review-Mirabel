# src/rag/generate_llamaindex.py
from __future__ import annotations
import os
import time
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import requests
from dotenv import load_dotenv, find_dotenv

# 加载项目根的 .env
load_dotenv(find_dotenv(), override=False)

def read_llm_env(default_model: Optional[str] = None) -> tuple[str, str, str]:
    api_base = os.getenv("LLM_API_BASE", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model_id = os.getenv("LLM_MODEL_ID", default_model or "")
    if not api_base or not api_key or not model_id:
        raise RuntimeError("Missing LLM_API_BASE / LLM_API_KEY / LLM_MODEL_ID in .env or env.")
    return api_base, api_key, model_id


def build_s2mia_prompt(
        contexts: List[str],
        excerpt: str,
        per_ctx_clip: int = 600,
        cont_words: int = 120,
) -> str:
    """
    S2MIA 续写式提示（Few-shot强化版）：
    通过示例，强制LLM进行精确的、无元描述的续写。
    """
    clipped = [(c or "").replace("\n", " ")[:per_ctx_clip] for c in contexts]
    ctx_block = "\n".join([f"- {c}" for c in clipped])

    # 包含了强硬指令和示例的系统说明
    system_note = (
        "You are an expert continuation extractor. Your ONLY task is to find the exact excerpt in the provided contexts and then output the verbatim text that immediately follows it. "
        f"The continuation should be no more than {cont_words} words. "
        "Do not paraphrase, do not summarize, and do not add any explanation or meta comments like 'The continuation is:'. "
        "If no exact match is found, output nothing."
    )

    # Few-shot 示例
    example1_context = "- The quick brown fox jumps over the lazy dog."
    example1_excerpt = "The quick brown fox"
    example1_continuation = "jumps over the lazy dog."

    example2_context = "- Retrieval-augmented generation (RAG) is a powerful technique. It combines retrieval with a generator model."
    example2_excerpt = "RAG) is a powerful technique."
    example2_continuation = "It combines retrieval with a generator model."

    # 组合成最终的Prompt
    prompt = (
        f"{system_note}\n\n"
        f"--- Example 1 ---\n"
        f"Contexts:\n{example1_context}\n\n"
        f"Excerpt:\n{example1_excerpt}\n\n"
        f"Continuation:\n{example1_continuation}\n\n"
        f"--- Example 2 ---\n"
        f"Contexts:\n{example2_context}\n\n"
        f"Excerpt:\n{example2_excerpt}\n\n"
        f"Continuation:\n{example2_continuation}\n\n"
        f"--- Current Task ---\n"
        f"Contexts:\n{ctx_block}\n\n"
        f"Excerpt:\n{excerpt}\n\n"
        f"Continuation:"
    )
    return prompt

# def build_s2mia_prompt(
#     contexts: List[str],
#     excerpt: str,
#     per_ctx_clip: int = 600,
#     cont_words: int = 120,
# ) -> str:
#     """
#     S2MIA 续写式提示：给定 excerpt（来自文档前半段）与上下文，如果 excerpt 出现在 contexts 中，
#     输出 excerpt 在原文中紧随其后的续写（限定 cont_words 词以内）。允许逐字复制，不要元描述。
#     """
#     clipped = [(c or "").replace("\n", " ")[:per_ctx_clip] for c in contexts]
#     ctx_block = "\n".join([f"- {c}" for c in clipped])
#     system_note = (
#         "You are given knowledge contexts and an excerpt. "
#         f"If the excerpt exactly appears in the contexts, output the exact continuation text that follows this excerpt in the source context (up to {cont_words} words). "
#         "It is allowed to copy text verbatim from the contexts. "
#         "Do not add any explanation or meta comments. Return only the continuation text."
#     )
#     prompt = (
#         f"{system_note}\n\n"
#         f"Contexts:\n{ctx_block}\n\n"
#         f"Excerpt:\n{excerpt}\n\n"
#         "Continuation:"
#     )
#     return prompt

def _normalize_continuation(text: str) -> str:
    ans = (text or "").strip()
    # 移除常见的前缀与拒答/无匹配提示
    lowers = ans.lower()
    bad_prefixes = [
        "the continuation text is:", "continuation:", "no exact match found", "no exact match",
        "i don't know", "the continuation text", "we did not find the excerpt",
        "the excerpt is not found", "no match", "not provided in the given contexts",
    ]
    for bp in bad_prefixes:
        if lowers.startswith(bp):
            ans = ans[len(bp):].strip()
            lowers = ans.lower()
    # 去除首尾引号
    if ans.startswith(("'", '"', "“", "‘")):
        ans = ans[1:].lstrip()
    if ans.endswith(("'", '"', "”", "’", ".")):
        # 保留句点的可选逻辑，这里仅简单处理
        ans = ans.rstrip("'\"”’").strip()
    return ans

def generate_s2mia_via_api_from_prompt(
    prompt: str,
    model_id: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: int = 90,
    retries: int = 5,
    proxies: Optional[dict] = None,
    sleep_backoff: float = 1.0,
) -> str:
    """
    S2MIA 专用 API 调用：使用专用系统消息，禁止 I don't know/No match/元描述。
    """
    if api_base is None or api_key is None or model_id is None:
        env_base, env_key, env_model = read_llm_env(model_id)
        api_base = api_base or env_base
        api_key = api_key or env_key
        model_id = model_id or env_model

    system_msg = (
        "You are a continuation extractor. "
        "Given knowledge contexts and an excerpt, "
        "output ONLY the exact continuation text that follows the excerpt in the source context. "
        "It is allowed to copy text verbatim from the contexts. "
        "Do not say 'I don't know'. Do not say 'no match'. Do not add any meta commentary."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout, proxies=proxies)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return _normalize_continuation(text)
        except Exception as e:
            last_error = e
            print(f"[S2MIA API FAIL] attempt={attempt+1}/{retries}, err={repr(e)}")
            time.sleep(sleep_backoff * (2 ** attempt))
            continue
    # 兜底：返回空字符串（上游不缓存失败）
    return ""

def generate_s2mia_via_api(
    contexts: List[str],
    excerpt: str,
    model_id: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: int = 90,
    retries: int = 5,
    proxies: Optional[dict] = None,
    per_ctx_clip: int = 600,
    cont_words: int = 120,
) -> str:
    prompt = build_s2mia_prompt(contexts, excerpt, per_ctx_clip=per_ctx_clip, cont_words=cont_words)
    return generate_s2mia_via_api_from_prompt(
        prompt=prompt,
        model_id=model_id, api_base=api_base, api_key=api_key,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        timeout=timeout, retries=retries, proxies=proxies
    )


def build_rag_prompt(contexts: List[str], question: str, per_ctx_clip: int = 600) -> str:
    """通用 RAG 问答提示：可在 pipeline_smoke/apply_nfcorpus 中使用。"""
    clipped = [(c or "").replace("\n", " ")[:per_ctx_clip] for c in contexts]
    ctx_block = "\n".join([f"- {c}" for c in clipped])
    prompt = (
        "You are a helpful assistant. Below is a query from a user and some relevant contexts.\n"
        "Answer the question given the information in those contexts. Your answer should be short and concise.\n"
        "If you cannot find the answer to the question, just say \"I don't know\".\n\n"
        f"Contexts:\n{ctx_block}\n\n"
        f"Query: {question}\n"
        "Answer:"
    )
    return prompt

def generate_answer_via_api_from_prompt(
        prompt: str,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 90,
        retries: int = 5,
        proxies: Optional[dict] = None,
        sleep_backoff: float = 1.0,
) -> str:
    """通用问答版 API 调用（/v1/chat/completions）。"""
    if api_base is None or api_key is None or model_id is None:
        env_base, env_key, env_model = read_llm_env(model_id)
        api_base = api_base or env_base
        api_key = api_key or env_key
        model_id = model_id or env_model

    system_msg = (
        "You are a factual, context-bound Question-Answering engine. "
        "Your task is to answer the user's question based *SOLELY* on the provided text contexts. "
        "DO NOT use any external knowledge. DO NOT guess. "
        "If the answer is present in the contexts, provide it concisely. "
        "If the contexts do not contain the answer, you MUST respond with the exact phrase: \"I don't know\"."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout, proxies=proxies)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return (text or "").strip()
        except Exception as e:
            last_error = e
            print(f"[QA API FAIL] attempt={attempt + 1}/{retries}, err={repr(e)}")
            time.sleep(sleep_backoff * (2 ** attempt))
            continue
    return "I don't know"


def generate_answer_via_api(
        contexts: List[str],
        question: str,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 90,
        retries: int = 5,
        proxies: Optional[dict] = None,
        per_ctx_clip: int = 600,
) -> str:
    """通用问答版封装，供 apply_nfcorpus/generator_smoke/pipeline_smoke 使用。"""
    prompt = build_rag_prompt(contexts, question, per_ctx_clip=per_ctx_clip)
    return generate_answer_via_api_from_prompt(
        prompt=prompt,
        model_id=model_id, api_base=api_base, api_key=api_key,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        timeout=timeout, retries=retries, proxies=proxies
    )


def generate_via_api_generic(
        user_prompt: str,
        system_prompt: str,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 90,
        retries: int = 5,
        proxies: Optional[dict] = None,
        sleep_backoff: float = 1.0,
) -> str:
    """一个完全通用的 API 调用函数，接受用户和系统 prompt。"""

    # 动态加载环境变量，确保总能获取到最新的配置
    if api_base is None or api_key is None or model_id is None:
        env_base, env_key, env_model = read_llm_env(model_id)
        api_base = api_base or env_base
        api_key = api_key or env_key
        model_id = model_id or env_model

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout, proxies=proxies)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            # 增加对空响应的检查
            if text is None:
                return ""
            return text.strip()
        except Exception as e:
            last_error = e
            print(f"\n[Generic API FAIL] attempt={attempt + 1}/{retries}, err={repr(e)}")
            time.sleep(sleep_backoff * (2 ** attempt))
            continue
    return ""  # 所有重试失败后返回空字符串


def paraphrase_via_api(
        text_to_paraphrase: str,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 90,
        retries: int = 3,
        proxies: Optional[dict] = None,
) -> str:
    """
    ✅ 新增函数：使用通用的API调用函数来实现文本改写（Paraphrasing）。
    """
    # 系统提示，指导LLM进行高质量的改写
    system_prompt = (
        "You are an expert paraphraser. Your task is to rewrite the given text while preserving its core meaning, intent, and key entities. "
        "The rewritten text should be grammatically correct and stylistically different from the original. "
        "Your output must ONLY be the paraphrased text, without any additional explanations, comments, or quotation marks."
    )

    # 用户提示，简单地包含待改写的文本
    user_prompt = text_to_paraphrase
    # 复用您项目中已有的通用API调用函数，这非常完美！
    rewritten_text = generate_via_api_generic(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        proxies=proxies,
    )
    return rewritten_text

# ============================================================
# PAD 本地生成：仅新增“本地 PAD 调用函数”，不接管 API，不透传 defense_mode
# ============================================================

@dataclass
class PADConfig:
    # DP/RDP
    epsilon_base: float = 0.2
    delta: float = 1e-5
    alpha: float = 10.0
    lambda_amp: float = 3.0
    # screening thresholds
    tau_conf: float = 0.90
    tau_margin: float = 1.00
    # lower bounds
    delta_min: float = 0.4
    sigma_min: float = 0.01
    # calibration weights
    w_entropy: float = 0.3
    w_pos: float = 0.2
    w_conf: float = 0.5
    pos_k: float = 0.1
    # generation
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None

class RDPAccountant:
    def __init__(self, alpha: float, delta: float):
        if alpha <= 1:
            raise ValueError("alpha must be > 1.")
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.rdp_total = 0.0
        self.steps_total = 0
        self.steps_protected = 0
    def update(self, sigma_t: float, delta_t: float, protected: bool):
        eps_t = (self.alpha * (delta_t ** 2)) / (2.0 * (sigma_t ** 2))
        self.rdp_total += float(eps_t)
        self.steps_total += 1
        if protected:
            self.steps_protected += 1
    def epsilon(self) -> float:
        return self.rdp_total + (math.log(1.0 / self.delta) / (self.alpha - 1.0))
    def report(self) -> Dict[str, Any]:
        gamma = (self.steps_protected / self.steps_total) if self.steps_total > 0 else 0.0
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "rdp_total": self.rdp_total,
            "epsilon_total": self.epsilon(),
            "steps_total": self.steps_total,
            "steps_protected": self.steps_protected,
            "gamma": gamma,
        }

class PADMechanism:
    def __init__(self, cfg: PADConfig):
        self.cfg = cfg
        self.acc = RDPAccountant(alpha=cfg.alpha, delta=cfg.delta)
        self.sigma_base = 1.0
    @staticmethod
    def _logit_margin(logits):
        top2 = logits.topk(k=2).values
        return float((top2[0] - top2[1]).item())
    @staticmethod
    def _normalized_entropy(probs):
        eps = 1e-12
        p = probs.clamp_min(eps)
        ent = -(p * p.log()).sum().item()
        return float(ent / math.log(probs.numel()))
    @staticmethod
    def _f_margin(m: float) -> float:
        return 1.0 / (1.0 + math.log(1.0 + max(m, 0.0)))
    def _estimate_delta_t(self, margin_t: float) -> float:
        raw = self._f_margin(margin_t)
        return float(max(self.cfg.delta_min, min(1.0, raw)))
    def _calibrate(self, probs, t: int) -> float:
        H = self._normalized_entropy(probs)
        f_pos = 1.0 / (1.0 + self.cfg.pos_k * float(t))
        f_conf = 1.0 - float(probs.max().item())
        return float((1.0 - self.cfg.w_entropy) + self.cfg.w_entropy * H
                     + self.cfg.w_pos * f_pos + self.cfg.w_conf * f_conf)
    def perturb_logits(self, logits, t: int):
        import torch
        cfg = self.cfg
        probs = torch.softmax(logits, dim=-1)
        max_p = float(probs.max().item())
        margin_t = self._logit_margin(logits)
        # screening: minimal noise
        if (max_p > cfg.tau_conf) and (margin_t > cfg.tau_margin):
            sigma_t = float(cfg.sigma_min)
            # 记账：这里你可改成 delta_t=0.0 来严格贴论文伪代码
            delta_t = float(cfg.delta_min)
            noise = torch.normal(0.0, sigma_t, size=logits.shape, device=logits.device, dtype=logits.dtype)
            self.acc.update(sigma_t=sigma_t, delta_t=delta_t, protected=False)
            return logits + noise
        # protected step
        delta_t = self._estimate_delta_t(margin_t)
        calib = self._calibrate(probs, t=t)
        sigma_t = self.sigma_base * calib * (delta_t / max(cfg.epsilon_base, 1e-12)) * cfg.lambda_amp
        sigma_t = float(max(cfg.sigma_min, sigma_t))
        noise = torch.normal(0.0, sigma_t, size=logits.shape, device=logits.device, dtype=logits.dtype)
        self.acc.update(sigma_t=sigma_t, delta_t=delta_t, protected=True)
        return logits + noise

_LOCAL_LLM_CACHE: Dict[str, Tuple[Any, Any]] = {}

def _load_local_llm_cached(model_id_or_path: str):
    """
    PAD 本地模型加载（带缓存）
    - 若环境支持 bitsandbytes：启用 4bit
    - 否则：普通 fp16/fp32 加载（Windows 更稳）
    """
    if model_id_or_path in _LOCAL_LLM_CACHE:
        return _LOCAL_LLM_CACHE[model_id_or_path]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检测 bitsandbytes 是否可用
    use_4bit = False
    try:
        import importlib.metadata as importlib_metadata
        _ = importlib_metadata.version("bitsandbytes")
        use_4bit = True
    except Exception:
        use_4bit = False

    kwargs = dict(
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if use_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
    model.eval()

    _LOCAL_LLM_CACHE[model_id_or_path] = (tokenizer, model)
    return tokenizer, model

def generate_answer_via_pad_local(
    contexts: List[str],
    question: str,
    pad_cfg: PADConfig,
    model_id_or_path: Optional[str] = None,
    per_ctx_clip: int = 600,
) -> Tuple[str, Dict[str, Any]]:
    """
    ✅ 仅负责 PAD 本地生成，不接管 API 调用、不接收 defense_mode。
    Returns:
        (answer_text, pad_report)
    """
    import torch
    model_id_or_path = model_id_or_path or os.getenv("PAD_MODEL_ID", "")
    if not model_id_or_path:
        raise RuntimeError("PAD local generation requires model_id_or_path or env PAD_MODEL_ID.")
    tokenizer, model = _load_local_llm_cached(model_id_or_path)
    device = next(model.parameters()).device
    # prompt 复用你已有的 RAG prompt（保证与 API 版本尽量一致）
    prompt = build_rag_prompt(contexts, question, per_ctx_clip=per_ctx_clip)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    pad = PADMechanism(cfg=pad_cfg)
    if pad_cfg.seed is not None:
        torch.manual_seed(pad_cfg.seed)
    generated_ids: List[int] = []
    max_new = int(pad_cfg.max_new_tokens)
    with torch.no_grad():
        for t in range(1, max_new + 1):
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1, :]  # [V]
            noisy_logits = pad.perturb_logits(logits, t=t)
            # decoding
            if pad_cfg.temperature <= 0.0:
                next_id = int(torch.argmax(noisy_logits).item())
            else:
                scaled = noisy_logits / float(pad_cfg.temperature)
                probs = torch.softmax(scaled, dim=-1)
                if pad_cfg.top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > pad_cfg.top_p
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    pick = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = int(sorted_idx[pick].item())
                else:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated_ids.append(next_id)
            next_tok = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if next_id == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, pad.acc.report()