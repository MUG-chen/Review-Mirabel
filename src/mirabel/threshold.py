# src/mirabel/threshold.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

@dataclass
class MirabelThresholdResult:
    """封装一次阈值计算的完整结果，便于调试与日志记录。"""
    tau: float                 # Gumbel 上侧阈值
    s_max: float               # 观测到的最大相似度
    target_idx: int            # s_max 在候选池中的索引位置
    is_member: bool            # 判定是否“成员攻击”
    # 下面是中间量（可选：记录以便分析/可视化）
    n: int                     # S' 的样本量（去掉最大值后）
    mu_q: float                # S' 的均值
    sigma_q: float             # S' 的标准差（ddof=0）
    mu_n: float                # Gumbel 位置参数
    beta_n: float              # Gumbel 尺度参数
    c: float                   # 显著性水平 rho 对应的 Gumbel 临界常数

def _compute_gumbel_params(mu_q: float, sigma_q: float, n: int, use_precise_mu: bool) -> Tuple[float, float]:
    """
    根据极值理论，给定基础正态参数估计（mu_q, sigma_q）与样本量 n，
    计算 Gumbel(μ_n, β_n) 的参数。提供精确式与近似式两种。
    """
    if n <= 1 or sigma_q <= 0.0:
        # 边界保护：样本太少或方差为0时，返回退化参数
        return mu_q, 0.0

    ln_n = math.log(n)
    if use_precise_mu:
        # 精确式：mu_n = mu_q + sigma_q * sqrt(2 ln n - ln ln n - ln(4π))
        # 当 n 很小时，该项可能数值不稳定，请结合候选池大小使用。
        mu_n = mu_q + sigma_q * math.sqrt(2.0 * ln_n - math.log(ln_n) - math.log(4.0 * math.pi))
    else:
        # 近似式：mu_n ≈ mu_q + sigma_q * sqrt(2 ln n)
        mu_n = mu_q + sigma_q * math.sqrt(2.0 * ln_n)

    beta_n = sigma_q / math.sqrt(2.0 * ln_n)
    return mu_n, beta_n

def mirabel_threshold(
    scores: Iterable[float],
    rho: float = 0.05,
    use_precise_mu: bool = True,
    eps: float = 1e-12,
) -> MirabelThresholdResult:
    """
    基于每查询的候选池相似度集合 S_q，自适应计算 Gumbel 阈值 τ，并给出是否“成员攻击”的判定。

    参数
    - scores: 候选池的相似度分数序列（余弦/内积），长度需>=2；建议先对向量做 L2 归一化后用内积。
    - rho: 显著性水平（默认 0.05），越大越容易判定为成员（降低漏报、提高误报）。
    - use_precise_mu: 是否使用 μ_n 的精确式（包含 −ln ln n 与 −ln(4π) 修正）。候选池较小时建议 True。
    - eps: 数值稳定项。

    返回
    - MirabelThresholdResult，包含 τ、s_max、target_idx、is_member 以及中间量（便于调试/记录）。

    参考公式
    - β_n = σ_q / sqrt(2 ln n)
    - μ_n = μ_q + σ_q * sqrt(2 ln n − ln ln n − ln(4π))（或近似 μ_q + σ_q * sqrt(2 ln n)）
    - c = −ln(−ln(1 − ρ))
    - τ = μ_n + c * β_n
    """
    scores = np.asarray(list(scores), dtype=np.float64)
    if scores.size < 2:
        raise ValueError("scores must contain at least 2 values.")

    # 找最大值及其位置（疑似目标文档）
    target_idx = int(np.argmax(scores))
    s_max = float(scores[target_idx])

    # 去掉最大值后的集合 S'
    mask = np.ones_like(scores, dtype=bool)
    mask[target_idx] = False
    S_prime = scores[mask]

    n = int(S_prime.size)
    # 均值/方差（ddof=0）——与论文一致
    mu_q = float(S_prime.mean()) if n > 0 else float(s_max)  # 极端情况下退化
    sigma_q = float(S_prime.std(ddof=0)) if n > 0 else 0.0

    # 避免极端退化
    if sigma_q < eps:
        sigma_q = 0.0  # 标明退化；参数函数中会处理

    # 计算 Gumbel 参数
    mu_n, beta_n = _compute_gumbel_params(mu_q, sigma_q, n, use_precise_mu=use_precise_mu)

    # Gumbel 分位数常数 c
    # 注意：rho ∈ (0,1)，rho 越大阈值越高（更“宽松”），越容易判成员。
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1)")
    c = -math.log(-math.log(1.0 - rho))

    # 阈值 τ
    tau = float(mu_n + c * beta_n)

    # 判定
    is_member = bool(s_max > tau)

    return MirabelThresholdResult(
        tau=tau,
        s_max=s_max,
        target_idx=target_idx,
        is_member=is_member,
        n=n,
        mu_q=mu_q,
        sigma_q=sigma_q,
        mu_n=mu_n,
        beta_n=beta_n,
        c=c,
    )

def mirabel_detect(scores: Iterable[float], rho: float = 0.05, use_precise_mu: bool = True) -> Tuple[bool, int, float]:
    """
    精简版接口：仅返回 (is_member, target_idx, tau)。
    便于在检索后处理的在线判定中快速使用。
    """
    res = mirabel_threshold(scores, rho=rho, use_precise_mu=use_precise_mu)
    return res.is_member, res.target_idx, res.tau

def mirabel_decision_with_gap(
    scores,
    rho: float = 0.005,
    margin: float = 0.02,
    gap_min: float = 0.03,
    use_precise_mu: bool = True,
):
    """
    组合判定：
      1) s_max > tau + margin  （Gumbel 阈值 + 安全边距）
      2) (s_max - s_2) >= gap_min （top-1 主导差：最大值与次大值的差）
    返回：is_member, tau, s_max, target_idx, s_2, gap, res
      - res：为调试保留原始 MirabelThresholdResult（含 mu_q、sigma_q、mu_n、beta_n、c 等）
    """
    import numpy as np
    res = mirabel_threshold(scores, rho=rho, use_precise_mu=use_precise_mu)
    s_max = res.s_max
    tau = res.tau
    scores = np.asarray(list(scores), dtype=np.float32)

    # 次大值 s_2
    mask = np.ones_like(scores, dtype=bool)
    mask[res.target_idx] = False
    s_2 = float(np.max(scores[mask])) if mask.sum() > 0 else s_max
    gap = s_max - s_2

    # 组合判定
    cond_margin = (s_max > tau + margin)
    cond_gap = (gap >= gap_min)
    is_member = bool(cond_margin and cond_gap)

    return is_member, tau, s_max, res.target_idx, s_2, gap, res