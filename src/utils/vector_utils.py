# 文件: src/utils/vector_utils.py

import numpy as np

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """对向量做 L2 归一化，保证内积≈余弦相似度。"""
    x = x.astype("float32")
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm