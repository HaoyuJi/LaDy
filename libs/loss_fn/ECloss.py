import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyConsistencyLoss(nn.Module):
    def __init__(self, warmup_steps=2000):
        """
        smooth_weight: 控制能量残差时间平滑强度
        velocity_threshold: 屏蔽低速帧（避免噪声放大）
        warmup_steps: 能量损失逐渐引入（避免早期干扰）
        """
        super().__init__()
        self.warmup_steps = warmup_steps
        self.register_buffer("step", torch.zeros(1))

    def forward(self, M, G, F_forces, tau_hat, q_dot):
        """
        M: (N, T, d, d)
        G, F_forces, tau_hat, q_dot: (N, T, d)
        """
        self.step += 1

        # -------------------
        # 1. 动能 K
        # -------------------
        K = 0.5 * torch.matmul(q_dot.unsqueeze(-2), torch.matmul(M, q_dot.unsqueeze(-1)))
        K = K.squeeze(-1).squeeze(-1)   # (N, T)

        # ΔK (N, T-1)
        delta_K = torch.diff(K, dim=1)

        # -------------------
        # 2. 净力与功率
        # -------------------
        tau_net = tau_hat - F_forces - G              # (N, T, d)
        P_net = torch.sum(tau_net * q_dot, dim=-1)    # (N, T)
        Work_net = 0.5 * (P_net[:, :-1] + P_net[:, 1:])  # (N, T-1)

        # -------------------
        # 3. 能量残差 r_E
        # -------------------
        denom = torch.abs(delta_K) + torch.abs(Work_net)
        mask = (denom > 1e-3).float()
        r_E = (delta_K - Work_net) / (denom + 0.1)
        r_E = r_E * mask

        # r_E = (delta_K - Work_net)/(torch.abs(delta_K) + torch.abs(Work_net) + 0.1)   # (N, T-1)

        # -------------------
        # 5. Huber Loss（抗噪声）
        # -------------------
        loss_main = F.smooth_l1_loss(r_E, torch.zeros_like(r_E))

        # -------------------
        # 7. Warm-Up 机制
        # -------------------
        weight = min(1.0, float(self.step.item() / self.warmup_steps))
        return weight * loss_main
