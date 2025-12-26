# 文件: src/attacks/precompute_tfidf.py

import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from attacks.s2mia_splits_and_index import load_full_mapping


def main():
    # --- 新增这两行用于调试 ---
    print(f"DEBUG: Current Working Directory is: {os.getcwd()}")
    print(f"DEBUG: Files in current dir: {os.listdir('.')}")
    # ------------------------

    print("Loading full corpus...")
    # 我们用全量文档来计算 IDF，这样更准
    mapping = load_full_mapping(index_dir="index")
    corpus = [doc['text'] for doc in mapping.values()]

    print(f"Building TF-IDF vectorizer for {len(corpus)} documents...")
    # min_df=5 表示忽略在少于5个文档中出现的词，可以过滤掉一些噪音和拼写错误
    vectorizer = TfidfVectorizer(stop_words='english', min_df=5)
    vectorizer.fit(corpus)

    # 我们只需要 vectorizer 本身（它内部存储了IDF值和词汇表）
    output_path = "cache/tfidf_vectorizer.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"TF-IDF vectorizer saved successfully to: {output_path}")


if __name__ == "__main__":
    main()