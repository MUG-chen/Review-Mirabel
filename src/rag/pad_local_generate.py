from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from defense.pad_config import PADConfig
from defense.pad import PADMechanism

@torch.no_grad()
def generate_with_pad_core(
        full_prompt: str,
        pad_cfg: PADConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer
) -> Tuple[str, Dict[str, Any]]:
    """
    核心生成逻辑。
    直接接收 prompt 字符串和已加载的模型实例，避免重复加载。
    """
    device = next(model.parameters()).device  # 获取模型所在的设备
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    # 初始化 PAD 机制
    pad = PADMechanism(cfg=pad_cfg, vocab_size=tokenizer.vocab_size, device=device)
    if pad_cfg.seed is not None:
        torch.manual_seed(pad_cfg.seed)
    generated = []

    # 生成循环
    for t in range(1, pad_cfg.max_new_tokens + 1):
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :]  # [V]
        # === PAD 核心防御 ===
        noisy_logits, step_info = pad.perturb_logits(logits, t=t)
        # ===================
        # 简单的 Greedy 或采样策略
        if pad_cfg.temperature <= 1e-5:
            next_id = int(torch.argmax(noisy_logits).item())
        else:
            scaled = noisy_logits / pad_cfg.temperature
            probs = torch.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)

        # 更新 input_ids
        next_tok = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if next_id == tokenizer.eos_token_id:
            break
    output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return output_text, pad.acc.report()

def build_rag_prompt(contexts: List[str], question: str, per_ctx_clip: int = 600) -> str:
    clipped = [(c or "").replace("\n", " ")[:per_ctx_clip] for c in contexts]
    ctx_block = "\n".join([f"- {c}" for c in clipped])
    prompt = (
        "You are a factual, context-bound Question-Answering engine. "
        "Your task is to answer the user's question based *SOLELY* on the provided text contexts. "
        "DO NOT use any external knowledge. DO NOT guess. "
        "If the contexts do not contain the answer, you MUST respond with the exact phrase: \"I don't know\".\n\n"
        f"Contexts:\n{ctx_block}\n\n"
        f"Query: {question}\n"
        "Answer:"
    )
    return prompt


def load_local_llm(model_id_or_path: str):
    """
    8GB 显存建议：4bit 量化 + device_map=auto。
    你可以按需加环境变量开关（只在 defense_mode=pad 才加载，不影响其它实验）。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bitsandbytes 4bit（需要你 pip 安装 bitsandbytes）
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_answer_via_pad_local(
    contexts: List[str],
    question: str,
    pad_cfg: PADConfig,
    model_id_or_path: str,
    per_ctx_clip: int = 600,
) -> Tuple[str, Dict[str, Any]]:
    """
    返回: (answer_text, pad_report)
    """
    tokenizer, model = load_local_llm(model_id_or_path)
    device = next(model.parameters()).device

    prompt = build_rag_prompt(contexts, question, per_ctx_clip=per_ctx_clip)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # 初始化 PAD
    pad = PADMechanism(cfg=pad_cfg, vocab_size=tokenizer.vocab_size, device=device)

    # 可选：设随机种子（确保可复现）
    if pad_cfg.seed is not None:
        torch.manual_seed(pad_cfg.seed)

    generated = []
    for t in range(1, pad_cfg.max_new_tokens + 1):
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :]  # [V]

        noisy_logits, step_info = pad.perturb_logits(logits, t=t)

        # 解码策略：论文主要是“在 logits 上扰动”，你可继续 greedy 或采样
        if pad_cfg.temperature <= 0.0:
            next_id = int(torch.argmax(noisy_logits).item())
        else:
            # temperature + top_p 采样
            scaled = noisy_logits / float(pad_cfg.temperature)
            probs = torch.softmax(scaled, dim=-1)

            if pad_cfg.top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > pad_cfg.top_p
                # 保留至少一个
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_id = int(sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item())
            else:
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_id)
        # 拼回 input_ids
        next_tok = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

        # 终止条件
        if next_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, pad.acc.report()
    # 如果想精简，可以改成：
    # return generate_with_pad_core(prompt, pad_cfg, model, tokenizer)
