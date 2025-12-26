# 文件: src/attacks/mba_utils.py

from __future__ import annotations
import re
import random
import pickle
from typing import List, Dict, Tuple

# 全局变量，用于单例加载 TF-IDF 模型
_tfidf_vectorizer = None


def _get_tfidf_vectorizer():
    """使用单例模式加载预计算的 TF-IDF 模型。"""
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        path = "cache/tfidf_vectorizer.pkl"
        print(f"\n[MBA Utils] Loading precomputed TF-IDF vectorizer from {path}...")
        try:
            with open(path, 'rb') as f:
                _tfidf_vectorizer = pickle.load(f)
            print("[MBA Utils] TF-IDF vectorizer loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"TF-IDF model not found at {path}. "
                "Please run 'python src/attacks/precompute_tfidf.py' first."
            )
    return _tfidf_vectorizer


def generate_masks_tfidf(
        text: str,
        num_masks: int = 15,
        min_word_len: int = 4
) -> Tuple[str, Dict[str, str]]:
    """
    【TF-IDF 智能版】使用 TF-IDF 分数选择单词进行掩码。
    """
    vectorizer = _get_tfidf_vectorizer()
    vocabulary = vectorizer.vocabulary_

    # 1. 找到所有合格的单词及其位置
    words_with_indices = [
        (match.group(0).lower(), match.start(), match.end())  # 转小写以匹配词汇表
        for match in re.finditer(r'\b\w+\b', text)
    ]

    # 2. 计算每个词的 TF-IDF 分数
    word_scores = []
    doc_words = [word_info[0] for word_info in words_with_indices]

    for word, start, end in words_with_indices:
        if len(word) >= min_word_len and word in vocabulary:
            # TF-IDF score = TF * IDF
            tf = doc_words.count(word) / len(doc_words)
            idf = vectorizer.idf_[vocabulary[word]]
            score = tf * idf
            # 原始文本中的词（保留大小写）和位置信息
            original_word = text[start:end]
            word_scores.append(((original_word, start, end), score))

    # 3. 按分数从高到低排序，选出 top-N
    word_scores.sort(key=lambda x: x[1], reverse=True)

    # 4. 选择最终要掩码的词 (避免重复选择同一个位置)
    masks_to_apply, added_positions = [], set()
    for (word_info, score) in word_scores:
        if len(masks_to_apply) >= num_masks: break
        start, end = word_info[1], word_info[2]
        if (start, end) not in added_positions:
            masks_to_apply.append(word_info)
            added_positions.add((start, end))

    # 按在原文中的出现顺序排序
    masks_to_apply.sort(key=lambda x: x[1])

    # 5. 构建掩码文本和答案
    masked_text = text
    mask_answers = {}
    for i, (word, start, end) in enumerate(reversed(masks_to_apply)):
        mask_id = len(masks_to_apply) - i
        mask_placeholder = f"[MASK_{mask_id}]"
        mask_answers[mask_placeholder] = word
        masked_text = masked_text[:start] + mask_placeholder + masked_text[end:]

    return masked_text, mask_answers


# 随机掩码

# def generate_masks(
#         text: str,
#         num_masks: int = 15,
#         min_word_len: int = 4,
#         avoid_stopwords: bool = True,
#         seed: int = 42  # 加入随机种子以保证可复现性
# ) -> Tuple[str, Dict[str, str]]:
#     """
#     【简化版】随机选择单词进行掩码。
#
#     Args:
#         text (str): 原始文档文本。
#         num_masks (int): 希望生成的掩码数量。
#         min_word_len (int): 单词最小长度。
#         avoid_stopwords (bool): 是否过滤停用词。
#         seed (int): 随机种子。
#
#     Returns:
#         Tuple[str, Dict[str, str]]: 掩码文本和答案。
#     """
#     if avoid_stopwords:
#         stopwords = set(
#             ["a", "an", "the", "in", "on", "at", "for", "to", "of", "by", "with", "is", "am", "are", "was", "were",
#              "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", "it", "we",
#              "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "and", "but", "or",
#              "as", "if", "that", "which", "who", "what", "where", "when", "why", "how"])
#     else:
#         stopwords = set()
#
#     # 1. 找到所有合格的单词及其位置
#     words_with_indices = [
#         (match.group(0), match.start(), match.end())
#         for match in re.finditer(r'\b\w+\b', text)
#     ]
#     eligible_words = [
#         word_info for word_info in words_with_indices
#         if len(word_info[0]) >= min_word_len and word_info[0].lower() not in stopwords
#     ]
#
#     if not eligible_words:
#         return text, {}
#
#     # 2. 随机选择要掩码的词
#     # 确保选择的数量不超过合格词的总数
#     k = min(num_masks, len(eligible_words))
#
#     # 使用随机种子以保证每次运行选择的词都一样
#     rng = random.Random(seed)
#     masks_to_apply = rng.sample(eligible_words, k)
#
#     # 3. 按在原文中的出现顺序排序
#     masks_to_apply.sort(key=lambda x: x[1])
#
#     # 4. 构建掩码文本和答案字典
#     masked_text = text
#     mask_answers = {}
#     for i, (word, start, end) in enumerate(reversed(masks_to_apply)):
#         mask_id = len(masks_to_apply) - i
#         mask_placeholder = f"[MASK_{mask_id}]"
#         mask_answers[mask_placeholder] = word
#         masked_text = masked_text[:start] + mask_placeholder + masked_text[end:]
#
#     return masked_text, mask_answers