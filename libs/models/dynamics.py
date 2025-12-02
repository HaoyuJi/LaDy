import torch
import torch.nn as nn

import torch.nn.functional as Func

class Net_M(nn.Module):
    """ M(q) = L(q)L(q)^T """

    def __init__(self, dof, hidden_size=128):
        super().__init__()
        self.dof = dof
        self.mlp = nn.Sequential(
            nn.Linear(dof, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dof * (dof + 1) // 2)
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q shape: (..., d)
        # 1. Lower triangular matrix
        l_elements = self.mlp(q)  # (..., d*(d+1)/2)

        L = torch.zeros(*q.shape[:-1], self.dof, self.dof, device=q.device)
        tril_indices = torch.tril_indices(row=self.dof, col=self.dof, offset=0)
        L[..., tril_indices[0], tril_indices[1]] = l_elements

        # Using softplus to ensure diagonal elements > 0
        diag_indices = torch.arange(self.dof)
        L[..., diag_indices, diag_indices] = Func.softplus(L[..., diag_indices, diag_indices]) + 1e-5

        # 4. M = LL^T for SPD
        M = torch.matmul(L, L.transpose(-1, -2))
        return M


class Net_C(nn.Module):
    """ C(q, q_dot) """

    def __init__(self, dof, hidden_size=128):
        super().__init__()
        self.dof = dof
        self.mlp = nn.Sequential(
            nn.Linear(dof * 2, hidden_size),  # q and q_dot
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dof * (dof - 1) // 2)
        )

    def forward(self, q: torch.Tensor, q_dot: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # q, q_dot shape: (..., d)

        M_dot = torch.cat([M[:, 0:1, :, :], M[:, 1:, :, :] - M[:, :-1, :, :]], dim=1)

        q_q_dot = torch.cat([q, q_dot], dim=-1)
        n_elements = self.mlp(q_q_dot)

        N = torch.zeros(*q.shape[:-1], self.dof, self.dof, device=q.device)

        triu_indices = torch.triu_indices(row=self.dof, col=self.dof, offset=1) #strictly upper triangular matrix

        N[..., triu_indices[0], triu_indices[1]] = n_elements

        N = N - N.transpose(-1,-2)

        C = 0.5 * (M_dot - N)

        return C


class Net_F(nn.Module):
    """ F(q) """

    def __init__(self, dof, hidden_size=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dof * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dof)
        )

    def forward(self, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        # q shape: (..., d)
        q_q_dot = torch.cat([q, q_dot], dim=-1)
        F = self.mlp(q_q_dot)
        return F

class Net_G(nn.Module):
    """ G(q) """
    def __init__(self, dof, hidden_size=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dof, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dof)
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q shape: (N, T, d) or (N*T, d)
        # G shape: (N, T, d) or (N*T, d)
        G = self.mlp(q)
        return G


class NeuralDynamicsModule(nn.Module):
    def __init__(self, dof, use_F=True, hidden_size=128):
        super().__init__()
        self.dof = dof
        self.use_F = use_F
        self.net_M = Net_M(dof, hidden_size)
        self.net_C = Net_C(dof, hidden_size)
        self.net_G = Net_G(dof, hidden_size)

        if use_F:
            self.net_F = Net_F(dof, hidden_size)

    def forward(self, q: torch.Tensor, q_dot: torch.Tensor, q_ddot: torch.Tensor) -> dict:
        """
        Input generalized coordinates and their derivatives, output decomposed dynamic terms and total pseudo torque.
        q, q_dot, q_ddot shapes: (N_m, T, d)
        """
        M = self.net_M(q)  # (N_m, T, d, d)
        C = self.net_C(q, q_dot, M)  # (N_m, T, d, d)
        G = self.net_G(q)  # (N_m, T, d)

        if self.use_F:
            F = self.net_F(q, q_dot)
        else:
            F = 0

        # Lagrange dynamic equation
        term_inertial = torch.matmul(M, q_ddot.unsqueeze(-1)).squeeze(-1)
        term_coriolis = torch.matmul(C, q_dot.unsqueeze(-1)).squeeze(-1)
        term_gravity = G
        term_otherforce = F

        tau_hat = term_inertial + term_coriolis + term_gravity + term_otherforce  # (N_m, T, d)

        return {
            "tau_hat": tau_hat,
            "M": M,
            "G": G,
            "F": F,
            "q_dot": q_dot
        }


class DynamicsFeatureProjector(nn.Module):
    """ Project and extend global dynamic features into joint by joint features """

    def __init__(self, dof, out_channels, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.projection = nn.Linear(dof, out_channels)

    def forward(self, tau_hat: torch.Tensor) -> torch.Tensor:
        # tau_hat shape: (N_m, T, d)

        # (N_m, T, d) -> (N_m, T, out_channels)
        projected_feature = self.projection(tau_hat)

        # (N_m, T, out_channels) -> (N_m, T, out_channels, V)
        expanded_feature = projected_feature.unsqueeze(-1).expand(-1, -1, -1, self.num_joints)

        # (N_m, T, out_channels, V) -> (N_m, out_channels, T, V)
        final_feature = expanded_feature.permute(0, 2, 1, 3)

        return final_feature


class DynamicsGateProjector(nn.Module):
    """ to get the dynamic gating signals  """

    def __init__(self, dof, out_channels, num_joints):
        super().__init__()
        self.num_joints = num_joints
        # self.projection = nn.Linear(dof, out_channels)

    def forward(self, tau_hat: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        # tau_hat shape: (N_m, T, d)

        power_per_dof = tau_hat * q_dot

        # (N_m, T, d) -> (N_m, T, out_channels)
        # projected_feature = self.projection(tau_hat)

        power = torch.sum(torch.abs(power_per_dof), dim=-1, keepdim=True)
        torque = torch.norm(tau_hat, p=2, dim=-1, keepdim=True)
        tau_dot = torch.cat([tau_hat[:, 0:1, :], tau_hat[:, 1:, :] - tau_hat[:, :-1, :]], dim=1)
        torque_jerk = torch.norm(tau_dot, p=2, dim=-1, keepdim=True)

        dyn_gate = torch.cat([power, torque, torque_jerk], dim=-1)

        # (N_m, T, out_channels) -> (N_m, out_channels, T)
        final_power = dyn_gate.permute(0, 2, 1)

        return final_power