# 文件: src/defense/prompt_guard.py

import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM  # 引入用于生成模型的类
)

# --- 全局配置，请在这里修改要使用的防御模型 ---
# ✅ 核心修改 1: 将模型ID更改为Qwen Guard
_GUARD_MODEL_ID = "Qwen/Qwen3Guard-Gen-8B"

# --- 全局变量，用于缓存模型，避免重复加载 ---
_GUARD_MODEL = None
_GUARD_TOKENIZER = None
_MODEL_TYPE = None  # 用于区分模型类型: 'classifier' 或 'generator'


def _extract_qwen_guard_label(content: str) -> str | None:
    """
    使用正则表达式从 Qwen-Guard 的输出中提取安全标签。
    例如从 "Safety: Unsafe\nCategories: Violent" 中提取 "Unsafe"。
    """
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    safe_label_match = re.search(safe_pattern, content)
    return safe_label_match.group(1) if safe_label_match else None


def _load_guard_model():
    """
    ✅ 核心修改 2: 延迟加载函数，现在能自动识别并加载不同类型的模型。
    """
    global _GUARD_MODEL, _GUARD_TOKENIZER, _MODEL_TYPE

    if _GUARD_MODEL is not None:
        return  # 如果已加载，直接返回

    print(f"[Defense] Initializing Guard model: {_GUARD_MODEL_ID}")

    # --- 根据模型名称判断加载方式 ---
    if "qwen" in _GUARD_MODEL_ID.lower() and "guard" in _GUARD_MODEL_ID.lower():
        # --- 加载 Qwen Guard (生成式模型) ---
        _MODEL_TYPE = "generator"
        print("[Defense] Detected Qwen-Guard model. Loading as a large language model (generator).")
        print("[Defense] WARNING: This is a large model and may require significant VRAM and time to load.")

        _GUARD_TOKENIZER = AutoTokenizer.from_pretrained(_GUARD_MODEL_ID)
        _GUARD_MODEL = AutoModelForCausalLM.from_pretrained(
            _GUARD_MODEL_ID,
            torch_dtype="auto",
            device_map="auto"  # 自动处理多GPU或GPU/CPU分配，对大模型至关重要
        )
        print(f"[Defense] Qwen-Guard model loaded successfully on device(s): {_GUARD_MODEL.device}")

    elif "llama-prompt-guard" in _GUARD_MODEL_ID.lower():
        # --- 加载 Llama Guard (分类模型) ---
        _MODEL_TYPE = "classifier"
        print("[Defense] Detected Llama-Prompt-Guard model. Loading as a sequence classification model.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _GUARD_TOKENIZER = AutoTokenizer.from_pretrained(_GUARD_MODEL_ID)
        _GUARD_MODEL = AutoModelForSequenceClassification.from_pretrained(_GUARD_MODEL_ID).to(device)
        print(f"[Defense] Llama-Prompt-Guard model loaded successfully on device: {device}")

    else:
        raise ValueError(
            f"Unsupported guard model ID: {_GUARD_MODEL_ID}. This script only supports 'qwen-guard' and 'llama-prompt-guard' types.")

    _GUARD_MODEL.eval()


def is_attack_query(query: str, attack_threshold: float = 0.9) -> bool:
    """
    ✅ 核心修改 3: 使用防御模型判断查询是否为攻击。
    函数内部逻辑会根据加载的模型类型自动切换。
    """
    try:
        _load_guard_model()  # 确保模型已加载

        # --- 分支逻辑：根据模型类型执行不同的推理 ---
        if _MODEL_TYPE == "generator":
            # --- Qwen-Guard 的生成式推理逻辑 ---
            messages = [{"role": "user", "content": query}]
            # 注意: Qwen Guard官方示例没有用add_generation_prompt=True，我们遵循它
            text = _GUARD_TOKENIZER.apply_chat_template(messages, tokenize=False)
            model_inputs = _GUARD_TOKENIZER([text], return_tensors="pt").to(_GUARD_MODEL.device)

            with torch.no_grad():
                generated_ids = _GUARD_MODEL.generate(**model_inputs, max_new_tokens=128, do_sample=False)
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                response_content = _GUARD_TOKENIZER.decode(output_ids, skip_special_tokens=True).strip()

            safe_label = _extract_qwen_guard_label(response_content)

            print(
                f"[QwenGuard] Query: '{query[:50]}...' -> Raw Response: '{response_content}' -> Parsed Label: {safe_label}")

            # 如果标签是 "Unsafe" 或 "Controversial"，则视为攻击
            if safe_label in ["Unsafe", "Controversial"]:
                print(f"[QwenGuard] ATTACK DETECTED! (Label: {safe_label})")
                return True
            return False

        elif _MODEL_TYPE == "classifier":
            # --- Llama-Guard 的分类式推理逻辑 (保持不变) ---
            inputs = _GUARD_TOKENIZER(query, return_tensors="pt", truncation=True, max_length=512).to(
                _GUARD_MODEL.device)
            with torch.no_grad():
                outputs = _GUARD_MODEL(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=-1)
            # Llama Guard 2 的 "attack" 标签通常是 LABEL_1 (索引为1)
            attack_prob = probabilities[0, 1].item()

            print(f"[LlamaGuard] Query: '{query[:50]}...' -> Attack Probability: {attack_prob:.4f}")
            # 注意: attack_threshold 参数主要用于分类模型
            if attack_prob > attack_threshold:
                print(f"[LlamaGuard] ATTACK DETECTED! (Prob: {attack_prob:.4f} > Threshold: {attack_threshold})")
                return True
            return False

    except Exception as e:
        print(f"[PromptGuard ERROR] An error occurred during guard model inference: {e}")
        # 安全默认：如果防御模块出错，为了不中断服务，默认判定为"非攻击"
        return False

    return False
