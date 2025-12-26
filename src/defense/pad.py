from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import math
import torch

from defense.pad_config import PADConfig


class RDPAccountant:
    """论文 §3.5：逐步累积 RDP cost，并可转成 (ε,δ)."""

    def __init__(self, alpha: float, delta: float):
        if alpha <= 1:
            raise ValueError("alpha must be > 1 for RDP.")
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.rdp_total = 0.0
        self.steps_total = 0
        self.steps_protected = 0  # 触发增强保护的步数（用于统计 γ）

    def update(self, sigma_t: float, delta_t: float, protected: bool):
        # ε^RDP_t = α ∆^2 / (2 σ^2)
        eps_t = (self.alpha * (delta_t ** 2)) / (2.0 * (sigma_t ** 2))
        self.rdp_total += float(eps_t)
        self.steps_total += 1
        if protected:
            self.steps_protected += 1

    def epsilon(self) -> float:
        # ε = RDP_total(α) + log(1/δ)/(α-1)
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
    """PAD：给定 logits s_t，输出 noisy logits，并更新 accountant。"""

    def __init__(self, cfg: PADConfig, vocab_size: int, device: torch.device):
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.device = device
        self.acc = RDPAccountant(alpha=cfg.alpha, delta=cfg.delta)

        # σ_base：论文里有一个“base noise scale”，你贴的版本里表述略混杂。
        # 为了严格可控，我们用最直接的：后续 σ_t 会乘 (∆_t/ε_base)*λ_amp，所以这里设 1.0 即可。
        self.sigma_base = 1.0

    @staticmethod
    def _logit_margin(logits: torch.Tensor) -> float:
        # logits: [V]
        top2 = torch.topk(logits, k=2).values
        return float((top2[0] - top2[1]).item())

    @staticmethod
    def _normalized_entropy(probs: torch.Tensor) -> float:
        # probs: [V]
        # H(p) = -1/log|V| * Σ p log p
        eps = 1e-12
        p = probs.clamp_min(eps)
        ent = -(p * p.log()).sum().item()
        return float(ent / math.log(probs.numel()))

    @staticmethod
    def _f_margin(m: float) -> float:
        # f(m) = 1/(1 + log(1+m))
        return 1.0 / (1.0 + math.log(1.0 + max(m, 0.0)))

    def _estimate_delta_t(self, margin_t: float) -> float:
        # ∆_t = clamp(f(margin_t), [∆_min, 1])
        raw = self._f_margin(margin_t)
        return float(max(self.cfg.delta_min, min(1.0, raw)))

    def _calibrate(self, probs: torch.Tensor, t: int) -> float:
        H = self._normalized_entropy(probs)
        f_pos = 1.0 / (1.0 + self.cfg.pos_k * float(t))
        f_conf = 1.0 - float(probs.max().item())
        # calibrate = (1-w_entropy) + w_entropy*H + w_pos*f_pos + w_conf*f_conf
        return float((1.0 - self.cfg.w_entropy) + self.cfg.w_entropy * H
                     + self.cfg.w_pos * f_pos + self.cfg.w_conf * f_conf)

    def perturb_logits(self, logits: torch.Tensor, t: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        输入 logits s_t (shape [V])，返回 noisy logits s~_t。
        返回额外 step_info 方便调试/记录。
        """
        cfg = self.cfg
        # 计算 probs、margin、maxprob
        probs = torch.softmax(logits, dim=-1)
        max_p = float(probs.max().item())
        margin_t = self._logit_margin(logits)

        # screening：高置信就只加 σ_min（论文：minimal noise）
        if (max_p > cfg.tau_conf) and (margin_t > cfg.tau_margin):
            sigma_t = float(cfg.sigma_min)
            delta_t = float(cfg.delta_min)  # 这里可记 0 或 delta_min；论文伪代码 Update(σ_min,0)；
                                           # 但若记 0 会让 ε 不增长。为了“保守一致”，建议记 delta_min。
            noise = torch.normal(
                mean=0.0, std=sigma_t,
                size=logits.shape, device=logits.device, dtype=logits.dtype
            )
            self.acc.update(sigma_t=sigma_t, delta_t=delta_t, protected=False)
            return logits + noise, {
                "protected": False,
                "sigma_t": sigma_t,
                "delta_t": delta_t,
                "max_p": max_p,
                "margin_t": margin_t,
            }

        # 否则：增强保护（论文：sensitivity + calibration）
        delta_t = self._estimate_delta_t(margin_t)
        calib = self._calibrate(probs, t=t)

        # σ_t = σ_base * calib * (∆_t/ε_base) * λ_amp
        sigma_t = self.sigma_base * calib * (delta_t / max(cfg.epsilon_base, 1e-12)) * cfg.lambda_amp
        sigma_t = float(max(cfg.sigma_min, sigma_t))

        noise = torch.normal(
            mean=0.0, std=sigma_t,
            size=logits.shape, device=logits.device, dtype=logits.dtype
        )
        self.acc.update(sigma_t=sigma_t, delta_t=delta_t, protected=True)

        return logits + noise, {
            "protected": True,
            "sigma_t": sigma_t,
            "delta_t": delta_t,
            "max_p": max_p,
            "margin_t": margin_t,
            "calib": calib,
        }
