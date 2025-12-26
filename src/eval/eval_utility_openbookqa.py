import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
import faiss
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 将 src 目录添加到路径中
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 引入项目中的生成函数
from rag.generate_llamaindex import (
    generate_answer_via_pad_local,
    generate_answer_via_api,
    generate_via_api_generic,
    PADConfig
)
from utils.vector_utils import l2_normalize


def load_openbookqa_data():
    """加载 OpenBookQA 数据集和事实库"""
    print("Loading OpenBookQA dataset...")
    dataset = load_dataset("allenai/openbookqa", "main", trust_remote_code=False)

    test_questions = []
    for item in dataset["test"]:
        choices = [label + ") " + text for label, text in zip(item["choices"]["label"], item["choices"]["text"])]
        test_questions.append({
            "id": item["id"],
            "question": item["question_stem"],
            "choices": choices,
            "answerKey": item["answerKey"]
        })

    facts = set()
    print("Extracting facts for Knowledge Base...")
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            if "fact1" in item and item["fact1"]:
                facts.add(item["fact1"])
            elif split in ["train", "validation"]:
                try:
                    ans_key = item["answerKey"]
                    if "label" in item["choices"] and "text" in item["choices"]:
                        labels = item["choices"]["label"]
                        texts = item["choices"]["text"]
                        if ans_key in labels:
                            idx = labels.index(ans_key)
                            correct_text = texts[idx]
                            synthetic_fact = f"{item['question_stem']} {correct_text}"
                            facts.add(synthetic_fact)
                except Exception:
                    pass

    fact_list = list(facts)
    if not fact_list:
        raise ValueError("CRITICAL ERROR: No facts extracted. Check dataset schema.")

    print(f"Loaded {len(test_questions)} test questions and {len(fact_list)} facts for retrieval.")
    return test_questions, fact_list


def build_index(fact_list, embedder):
    """构建 Faiss 索引"""
    print("Encoding facts and building index...")
    batch_size = 64
    all_vecs = []
    for i in tqdm(range(0, len(fact_list), batch_size), desc="Embedding"):
        batch = fact_list[i: i + batch_size]
        emb = embedder.encode(batch)["dense_vecs"]
        emb = l2_normalize(emb)
        all_vecs.append(emb)
    all_vecs = np.concatenate(all_vecs, axis=0)
    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)
    return index


def extract_answer_letter(response):
    """解析答案字母"""
    clean_resp = response.strip().upper()
    import re
    # 1. 简单匹配单字母
    if clean_resp in ["A", "B", "C", "D"]:
        return clean_resp
    # 2. 匹配 "Answer: A"
    match = re.search(r'(?:Answer:|Option)\s*([A-D])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 3. 匹配行首 "A)"
    match = re.search(r'^\s*([A-D])[\)\.]', response, re.MULTILINE)
    if match:
        return match.group(1).upper()
    return "Unknown"


def generate_answer_via_local_baseline(contexts, question, model, tokenizer, max_new_tokens=10):
    """
    本地 Baseline 生成 (非 API, 非 PAD)
    用于与 PAD 进行公平对比
    """
    context_text = "\n".join(contexts)
    input_text = f"Context:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"

    if tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
            model.device)
    else:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
    generated_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Defense Utility on OpenBookQA")

    parser.add_argument("--defense_mode", type=str, default="none", choices=["none", "pad"], help="Defense mode")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="results/openbookqa_eval")

    # 路径参数
    parser.add_argument("--model_path", type=str, default="", help="Path to local LLM (for Local Baseline / PAD)")

    # API 参数
    parser.add_argument("--use_api", action="store_true", help="Use API for baseline (requires .env setup)")
    parser.add_argument("--api_model_name", type=str, default="", help="Optional: Override model_id in .env")

    parser.add_argument("--max_tokens", type=int, default=20)

    args = parser.parse_args()

    # 1. 准备 Embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bge_path = "/root/autodl-tmp/pycharm_Mirabel/models/BAAI/bge-m3"
    embedder = BGEM3FlagModel(bge_path, use_fp16=True, device=device)

    # 2. 准备数据 & 索引
    questions, facts = load_openbookqa_data()
    index = build_index(facts, embedder)

    # 3. 准备模型 (Local Baseline 用)
    local_baseline_model = None
    local_baseline_tokenizer = None

    if args.defense_mode == "none" and not args.use_api:
        if not args.model_path:
            raise ValueError("Local Baseline requires --model_path. Or add --use_api to use remote API.")
        print(f"Loading local model for Baseline: {args.model_path}")
        local_baseline_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        local_baseline_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        local_baseline_model.eval()

    # PAD 模式参数
    if args.defense_mode == "pad":
        if not args.model_path:
            raise ValueError("PAD Defense requires --model_path (local weights).")
        print(f"PAD Defense Enabled. Model Path: {args.model_path}")
        pad_cfg = PADConfig(max_new_tokens=args.max_tokens, temperature=0.0)
    else:
        pad_cfg = None

    # 4. 执行评估
    correct_count = 0
    total_count = 0
    results = []

    if args.defense_mode == "none" and args.use_api:
        mode_str = "API-Baseline"
    elif args.defense_mode == "none":
        mode_str = "Local-Baseline"
    else:
        mode_str = "Local-PAD"

    print(f"\nStarting Evaluation (Mode: {mode_str})...")

    for item in tqdm(questions, desc="Answering"):
        q_text = item["question"]

        # 检索
        q_vec = embedder.encode([q_text])["dense_vecs"]
        q_vec = l2_normalize(q_vec)
        D, I = index.search(q_vec, args.top_k)
        retrieved_facts = [facts[idx] for idx in I[0]]

        # 构造增强问题 (包含选项)
        augmented_question = f"{q_text}\nChoices:\n" + "\n".join(
            item["choices"]) + "\nPlease analyze the context and answer with the option letter only (e.g. A)."

        response = ""

        # ================= 分支逻辑 =================
        if args.defense_mode == "pad":
            # PAD (Local)
            response, _ = generate_answer_via_pad_local(
                contexts=retrieved_facts,
                question=augmented_question,
                pad_cfg=pad_cfg,
                model_id_or_path=args.model_path
            )

        elif args.defense_mode == "none" and args.use_api:
            # API (Baseline) - 使用 Generic 接口以完全对齐 Local Prompt
            # 1. 构造和 Local 完全一致的 Prompt 格式
            context_text = "\n".join(retrieved_facts)
            # 这里把 Context, Question, Choices 拼在一起，模拟本地的 input_text
            user_input = f"Context:\n{context_text}\n\nQuestion:\n{augmented_question}\n\nAnswer:"

            # 2. System Prompt 设置宽松一点，或者干脆只设为 Helpful Assistant
            # 关键是去掉 "If not found say I don't know" 这种指令
            system_prompt = "You are a helpful assistant. Analyze the context and question to choose the best option."
            model_arg = args.api_model_name if args.api_model_name else None

            # 3. 调用通用接口
            response = generate_via_api_generic(
                user_prompt=user_input,
                system_prompt=system_prompt,
                model_id=model_arg,
                temperature=0.0,
                max_tokens=args.max_tokens
                # api_base / api_key 自动读取
            )

        else:
            # Local (Baseline)
            response = generate_answer_via_local_baseline(
                contexts=retrieved_facts,
                question=augmented_question,
                model=local_baseline_model,
                tokenizer=local_baseline_tokenizer,
                max_new_tokens=args.max_tokens
            )
        # ============================================

        # 统计
        pred_label = extract_answer_letter(response)
        is_correct = (pred_label == item["answerKey"])
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "id": item["id"],
            "pred": pred_label,
            "gold": item["answerKey"],
            "correct": is_correct,
            "response": response
        })

    # 5. 保存结果
    accuracy = correct_count / total_count if total_count > 0 else 0
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    filename = f"utility_{mode_str}.json"
    out_file = os.path.join(args.out_dir, filename)

    print(f"\n" + "=" * 40)
    print(f"Evaluation Finished!")
    print(f"Mode: {mode_str}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    print(f"Saving to: {out_file}")
    print(f"=" * 40)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "config": vars(args), "details": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
