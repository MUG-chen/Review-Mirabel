# 文件: src/attacks/ia_utils.py
"""
IA (Interrogation Attack) 工具函数
- 生成文档摘要和问题
- 使用交叉编码器进行问题筛选（reranking）
- 生成标准答案
"""

from __future__ import annotations
import os
import re
import requests
import time
from typing import List, Tuple, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv(), override=False)

# 全局变量：缓存交叉编码器模型
_CROSS_ENCODER_MODEL = None
_CROSS_ENCODER_TOKENIZER = None
_CROSS_ENCODER_DEVICE = None


# ==================== LLM 调用函数 ====================

def _read_llm_env(env_prefix: str = "IA_QGEN") -> Tuple[str, str, str]:
    """
    读取 LLM 环境变量

    Args:
        env_prefix: 环境变量前缀，如 "IA_QGEN" 或 "IA_SHADOW"

    Returns:
        (api_base, api_key, model_id)
    """
    api_base = os.getenv("LLM_API_BASE", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")

    # 根据前缀读取不同的模型ID
    if env_prefix == "IA_QGEN":
        model_id = os.getenv("IA_QGEN_MODEL", os.getenv("LLM_MODEL_ID", "gpt-4o"))
    elif env_prefix == "IA_SHADOW":
        model_id = os.getenv("IA_SHADOW_MODEL", os.getenv("LLM_MODEL_ID", "gpt-4o-mini"))
    else:
        model_id = os.getenv("LLM_MODEL_ID", "gpt-4o")

    if not api_base or not api_key:
        raise RuntimeError(f"Missing LLM_API_BASE or LLM_API_KEY in .env")

    return api_base, api_key, model_id


def _call_llm(
        prompt: str,
        model_id: str,
        api_base: str,
        api_key: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,  # ✅ 新增 top_p 参数，默认值为 1.0
        max_retries: int = 5
) -> str:
    """
    调用 OpenAI 兼容的 LLM API

    Args:
        prompt: 提示词
        model_id: 模型ID
        api_base: API 基础URL
        api_key: API 密钥
        max_tokens: 最大生成token数
        temperature: 温度参数
        max_retries: 最大重试次数

    Returns:
        生成的文本
    """
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top-p": top_p
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM API Error] Attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                raise RuntimeError(f"LLM API call failed after {max_retries} attempts: {e}")


# ==================== 交叉编码器函数 ====================

def _load_cross_encoder():
    """延迟加载交叉编码器模型（crystina-z/monoELECTRA_LCE_nneg31）"""
    global _CROSS_ENCODER_MODEL, _CROSS_ENCODER_TOKENIZER, _CROSS_ENCODER_DEVICE

    if _CROSS_ENCODER_MODEL is None:
        model_name = "/root/autodl-tmp/pycharm_Mirabel/models/BAAI/bge-reranker-large"
        print(f"[IA] Loading cross-encoder: {model_name}")

        _CROSS_ENCODER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _CROSS_ENCODER_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        _CROSS_ENCODER_MODEL.eval()

        # 自动选择设备
        if torch.cuda.is_available():
            _CROSS_ENCODER_DEVICE = torch.device("cuda")
            _CROSS_ENCODER_MODEL = _CROSS_ENCODER_MODEL.cuda()
            print(f"[IA] Cross-encoder loaded on GPU")
        else:
            _CROSS_ENCODER_DEVICE = torch.device("cpu")
            print(f"[IA] Cross-encoder loaded on CPU")

    return _CROSS_ENCODER_MODEL, _CROSS_ENCODER_TOKENIZER, _CROSS_ENCODER_DEVICE


def rerank_questions_with_cross_encoder(
        doc_text: str,
        questions: List[str],
        top_k: int = 30,
        batch_size: int = 16
) -> List[Tuple[str, float]]:
    """
    使用交叉编码器对生成的问题进行重排序

    Args:
        doc_text: 目标文档文本
        questions: 生成的问题列表
        top_k: 返回前 k 个最相关的问题
        batch_size: 批处理大小

    Returns:
        List of (question, score) tuples, sorted by relevance score (descending)
    """
    if not questions:
        return []

    model, tokenizer, device = _load_cross_encoder()

    scores = []

    # 批处理计算相关性分数
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]

        # 构造输入对：[CLS] question [SEP] document [SEP]
        inputs = tokenizer(
            batch_questions,
            [doc_text] * len(batch_questions),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits[:, 0].cpu().numpy().tolist()
            scores.extend(batch_scores)

    # 按分数降序排序
    question_score_pairs = list(zip(questions, scores))
    question_score_pairs.sort(key=lambda x: x[1], reverse=True)

    return question_score_pairs[:top_k]


# ==================== 问题和答案生成 ====================


def generate_interrogation_materials(
        doc_id: str,
        doc_text: str,
        num_questions_generate: int = 30,  # 默认值调整为更实际的 30
        num_questions_select: int = 15,  # 默认值调整为更实际的 15
        use_cross_encoder: bool = True,
        cache: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str]]:
    """
    【升级版】使用小样本提示生成高质量的摘要和探究性问题。
    集成了摘要和问题生成，以提高效率和上下文一致性。
    """
    # 1. ✅ 改进的缓存键：加入版本号，以便在修改 prompt 后强制重新生成
    prompt_version = "v1.1_fewshot"
    cache_key = f"materials_{prompt_version}_{doc_id}_gen{num_questions_generate}_sel{num_questions_select}_ce{use_cross_encoder}"

    if cache is not None and cache_key in cache:
        cached_data = cache[cache_key]
        return cached_data["summary"], cached_data["questions"]

    api_base, api_key, model_id = _read_llm_env("IA_QGEN")

    # 2. ✅ 核心改进：采用Few-Shot Prompt，将摘要和问题生成合并
    #    这个 prompt 直接指导 LLM 学习如何提取关键细节并转化为 Yes/No 问题
    questions_prompt = f"""You are a security analyst tasked with creating highly specific questions to determine if a document is present in a database. Your goal is to generate a brief summary and a list of yes/no questions based *only* on the provided document. The questions must probe for unique details, facts, or figures that would be impossible to guess correctly without having read this exact text.

### Requirements:
1.  First, provide a "Summary:" of 2-3 sentences.
2.  Then, provide "Questions:" generating exactly {num_questions_generate} questions.
3.  Each question must be answerable with "Yes," "No," or "I don't know."
4.  Questions must be diverse and cover different, specific aspects of the document.
5.  Format the questions as a numbered list (e.g., "1. Question text?").

---

### Example Document:
The Hubble Space Telescope's primary mirror, spanning 2.4 meters, was discovered to have a spherical aberration after its launch in 1990. The error was minuscule, just 1/50th the thickness of a human hair (approximately 2.2 micrometers), but it was enough to blur the telescope's vision. A corrective optics package, COSTAR, was installed during a servicing mission in 1993 to fix the issue.

### Example Output:
Summary: The Hubble Space Telescope was launched with a flawed 2.4-meter primary mirror, suffering from a 2.2-micrometer spherical aberration. This issue was later corrected in 1993 by installing the COSTAR optics package.

Questions:
1. Was the Hubble Telescope's primary mirror exactly 2.4 meters in diameter?
2. Did the flaw in the mirror measure 1/50th the thickness of a human hair?
3. Was the corrective optics package named COSTAR installed in 1992?
4. Is the mirror's aberration described as chromatic aberration?
5. Did the servicing mission to install COSTAR occur in December 1993?

---

### Document to Process:
{doc_text[:3500]} 

### Your Output:
"""

    # 3. ✅ 优化的采样参数：低 temperature 以保证事实性
    response_text = _call_llm(
        questions_prompt,
        model_id,
        api_base,
        api_key,
        max_tokens=2500,  # 留足空间
        temperature=0.1,  # 降低随机性，强制模型关注事实
        top_p=0.95
    )

    # 4. ✅ 更鲁棒的解析逻辑
    summary_match = re.search(r"Summary:\s*(.*?)\s*Questions:", response_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "Summary could not be generated."

    questions_match = re.search(r"Questions:\s*(.*)", response_text, re.DOTALL)
    questions_text = questions_match.group(1).strip() if questions_match else ""

    # 使用 _clean_questions 处理，该函数也应被优化
    raw_questions = [line.strip() for line in questions_text.split('\n') if line.strip()]
    questions = _clean_questions(raw_questions)

    # 后续的 reranking 逻辑保持不变
    if len(questions) < num_questions_select:
        print(f"[WARNING] Only generated {len(questions)} valid questions for doc {doc_id}")
        selected_questions = questions
    else:
        # 确保不多于请求生成的数量
        questions = questions[:num_questions_generate]

        if use_cross_encoder and len(questions) > num_questions_select:
            ranked_questions = rerank_questions_with_cross_encoder(
                doc_text=doc_text,
                questions=questions,
                top_k=num_questions_select
            )
            selected_questions = [q for q, score in ranked_questions]

            if len(ranked_questions) > 0:
                max_score = ranked_questions[0][1]
                min_score = ranked_questions[-1][1]
                print(
                    f"[IA] Doc {doc_id}: Selected {len(selected_questions)} questions (scores: {max_score:.3f} to {min_score:.3f})")
        else:
            selected_questions = questions[:num_questions_select]

    if cache is not None:
        cache[cache_key] = {
            "summary": summary,
            "questions": selected_questions
        }

    return summary, selected_questions


def _clean_questions(raw_questions: List[str]) -> List[str]:
    """【升级版】更智能地清理问题列表"""
    cleaned = []
    for q in raw_questions:
        # 匹配 "1. Question text?" 或 "1) Question text?" 等格式
        match = re.match(r"^\s*\d+[\.\)]\s*(.+)", q)
        if match:
            question_text = match.group(1).strip()
            # 确保以问号结尾，增加规范性
            if not question_text.endswith('?'):
                question_text += '?'
            cleaned.append(question_text)
    return cleaned

def generate_ground_truths(
        doc_id: str,
        doc_text: str,
        summary: str,
        questions: List[str],
        cache: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    使用影子模型生成问题的标准答案

    Args:
        doc_id: 文档ID
        doc_text: 文档文本
        summary: 文档摘要
        questions: 问题列表
        cache: 缓存字典

    Returns:
        标准答案列表
    """
    cache_key = f"ground_truths_{doc_id}_{len(questions)}"

    if cache is not None and cache_key in cache:
        return cache[cache_key]

    api_base, api_key, model_id = _read_llm_env("IA_SHADOW")

    ground_truths = []

    for question in questions:
        # ✅ 修正：使用论文中的提示词格式
        prompt = f"""Based on the following document, answer the yes/no question. Answer with "Yes," "No," or "I don't know" only.

Document:
{doc_text[:3000]}

Question: {question}

Answer:"""

        answer = _call_llm(prompt, model_id, api_base, api_key, max_tokens=10, temperature=0.0)
        ground_truths.append(answer.strip())

    if cache is not None:
        cache[cache_key] = ground_truths

    return ground_truths


def compute_ia_score(
        rag_answers: List[str],
        ground_truths: List[str],
        lambda_penalty: float = 1.0
) -> float:
    """
    计算 IA 分数（基于原论文方法，并进行归一化）

    公式: IA_score = (正确回答数 - λ × IDK 回答数) / 问题总数

    Args:
        rag_answers: RAG 系统的回答列表（已解析为 yes/no/unknown）
        ground_truths: 标准答案列表（原始文本，需要解析）
        lambda_penalty: "I don't know" 惩罚系数（默认 1.0）

    Returns:
        归一化后的 IA 分数（浮点数）
    """
    if not rag_answers or not ground_truths:
        return 0.0

    # 确保答案列表和 ground truth 列表长度一致
    if len(rag_answers) != len(ground_truths):
        print(f"[WARNING] compute_ia_score: Length mismatch between RAG answers ({len(rag_answers)}) and ground truths ({len(ground_truths)}). Truncating to minimum length.")
        min_len = min(len(rag_answers), len(ground_truths))
        rag_answers = rag_answers[:min_len]
        ground_truths = ground_truths[:min_len]

    correct_count = 0
    idk_count = 0

    for rag_ans, gt_ans in zip(rag_answers, ground_truths):
        # 解析 ground truth (确保是 'yes', 'no', 'unknown' 之一)
        gt_parsed = parse_yes_no(gt_ans)

        # 统计 "I don't know" 回答
        if rag_ans == "unknown":
            idk_count += 1
            continue # 不计入正确回答

        # 统计正确匹配 (仅当两者都不是 unknown 时才计数)
        # Note: rag_ans 已经是 parse_yes_no 的结果，所以可以直接比较
        if rag_ans == gt_parsed and rag_ans != "unknown":
            correct_count += 1

    # 计算原始分数
    raw_score = float(correct_count) - lambda_penalty * float(idk_count)

    # ✅ **关键改动：进行归一化**
    num_questions = len(ground_truths) # 即 N_q
    if num_questions == 0:
        return 0.0 # 避免除以零

    normalized_score = raw_score / num_questions

    return normalized_score

def parse_yes_no(text: str) -> str:
    """
    解析 LLM 回答中的 yes/no 答案（严格模式）

    Args:
        text: LLM 返回的文本

    Returns:
        "yes", "no", 或 "unknown"
    """
    if not text:
        return "unknown"

    text_clean = text.lower().strip().rstrip('.')

    # ✅ 严格匹配（优先）
    if text_clean in ["yes", "no", "i don't know", "i do not know"]:
        if text_clean == "yes":
            return "yes"
        elif text_clean == "no":
            return "no"
        else:
            return "unknown"

    # ✅ 宽松匹配（仅当严格匹配失败时）
    if text_clean.startswith("yes"):
        return "yes"
    if text_clean.startswith("no") and "i don't know" not in text_clean:
        return "no"

    # ✅ 检测 IDK 模式
    idk_patterns = [
        "i don't know",
        "i do not know",
        "cannot",
        "can't",
        "unable",
        "not sure",
        "uncertain",
        "no information"
    ]

    for pattern in idk_patterns:
        if pattern in text_clean:
            return "unknown"

    return "unknown"


def detect_idk(text: str) -> bool:
    """
    检测回答是否为 "I don't know" 类型（已弃用，使用 parse_yes_no 替代）

    Args:
        text: LLM 返回的文本

    Returns:
        True 如果是 IDK 回答
    """
    return parse_yes_no(text) == "unknown"


def rewrite_query_for_retrieval(question: str) -> str:
    """
    使用 LLM 将 Yes/No 问题重写为适合检索的关键词式查询。

    Args:
        question: 原始的 Yes/No 问题。

    Returns:
        重写后的、更适合检索的查询字符串。
    """
    api_base, api_key, model_id = _read_llm_env("IA_QGEN") # 复用 IA 提问模型

    # 构建 Prompt：
    # 目标是让 LLM 从 Yes/No 问题中提取核心实体和事实，形成一个简洁的搜索查询。
    # 引用了论文中的一些例子（Table 4）来指导 LLM。
    prompt = f"""You are an expert search query generator. Your task is to convert a specific yes/no question into a concise, keyword-focused search query. The query should aim to retrieve documents that can directly answer the question. Avoid general terms and focus on specific entities, facts, or claims mentioned in the question.

### Examples inspired by research papers:

1.  Question: Was the Hubble Telescope's primary mirror exactly 2.4 meters in diameter?
    Search Query: Hubble Telescope primary mirror diameter 2.4 meters

2.  Question: Did the settlement occur in fiscal year Q3 2021?
    Search Query: fiscal year settlement Q3 2021

3.  Question: Was the corrective optics package named COSTAR installed in 1992?
    Search Query: COSTAR corrective optics package installation year

4.  Question: Was the legal firm involved named Wexler & Associates?
    Search Query: Wexler & Associates legal firm

### Current Task:

Question: {question}
Search Query:"""

    # 调用LLM进行查询重写，使用低 temperature 保证确定性
    rewritten_query = _call_llm(
        prompt,
        model_id,
        api_base,
        api_key,
        max_tokens=50,      # 查询通常很短
        temperature=0.0,    # 保持高确定性
        top_p=1.0
    )
    return rewritten_query.strip()