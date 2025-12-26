from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PADConfig:
    # === DP / RDP 参数（论文默认示例见附录E）===
    epsilon_base: float = 0.2
    delta: float = 1e-5
    alpha: float = 10.0
    lambda_amp: float = 3.0

    # === screening 阈值 ===
    tau_conf: float = 0.90
    tau_margin: float = 1.00

    # === 下界（论文强调数值稳定） ===
    delta_min: float = 0.4     # ∆_min
    sigma_min: float = 0.01    # σ_min（你可按实际再调）

    # === calibration 权重（论文 w_entropy=0.3, w_pos=0.2，w_conf 可设 0.5 或 0.5-0.2-0.3 自己配）===
    w_entropy: float = 0.3
    w_pos: float = 0.2
    w_conf: float = 0.5

    # === 位置因子 f_pos(t)=1/(1+0.1*t) 的系数 ===
    pos_k: float = 0.1

    # 生成参数（pad 本地生成时用）
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None

    # 选择性 DP：γ-relaxation 统计用（论文会 report γ；实现上我们统计“触发增强保护”的比例）
    # 这里只做统计，不强制“只保护γ比例”，因为论文是“对被保护步集合P提供DP保证”
