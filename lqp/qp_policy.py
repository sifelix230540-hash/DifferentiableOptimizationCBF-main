from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .pdhg import pdhg_unroll, residual_loss


@dataclass
class LearnedQPConfig:
    state_dim: int = 4
    ref_dim: int = 2
    action_dim: int = 2
    horizon: int = 4
    n_constraints: int = 48
    n_iter: int = 10
    alpha: float = 0.8
    u_max: float = 8.0
    damping: float = 1e-5

    @property
    def nqp(self) -> int:
        return self.horizon * self.action_dim


class LearnedQPPolicy(nn.Module):
    """Paper-inspired learned QP nominal controller.

    `P` and `H` are state-independent.
    `q` depends affinely on `[x, ref]`.
    `b` depends affinely on `x`.
    """

    def __init__(self, config: LearnedQPConfig):
        super().__init__()
        self.config = config

        nqp = config.nqp
        mqp = config.n_constraints
        q_in = config.state_dim + config.ref_dim

        self.P_factor_raw = nn.Parameter(0.05 * torch.randn(nqp, nqp))
        self.H = nn.Parameter(0.05 * torch.randn(mqp, nqp))
        self.Wq = nn.Linear(q_in, nqp, bias=False)
        self.Wb = nn.Linear(config.state_dim, mqp, bias=True)
        self.softplus = nn.Softplus()

        self._init_bias_constraints()

    def _init_bias_constraints(self) -> None:
        with torch.no_grad():
            self.Wb.weight.zero_()
            self.Wb.bias.fill_(1.0)

    def build_problem(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if ref.ndim == 1:
            ref = ref.unsqueeze(0)

        batch = state.shape[0]
        nqp = self.config.nqp
        mqp = self.config.n_constraints

        tril = torch.tril(self.P_factor_raw)
        diag = torch.diagonal(tril, dim1=-2, dim2=-1)
        tril = tril - torch.diag_embed(diag) + torch.diag_embed(self.softplus(diag) + 1e-3)
        p_single = tril @ tril.transpose(-1, -2) + 1e-3 * torch.eye(
            nqp, device=state.device, dtype=state.dtype
        )
        P = p_single.unsqueeze(0).expand(batch, -1, -1)
        H = self.H.unsqueeze(0).expand(batch, -1, -1)

        qr_input = torch.cat([state, ref], dim=-1)
        q = self.Wq(qr_input)
        b = self.Wb(state)

        return P, q, H, b

    def forward(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        P, q, H, b = self.build_problem(state, ref)
        y, solver_state = pdhg_unroll(
            P=P,
            q=q,
            H=H,
            b=b,
            n_iter=self.config.n_iter,
            alpha=self.config.alpha,
            damping=self.config.damping,
        )
        u = y[..., : self.config.action_dim]
        u = self.config.u_max * torch.tanh(u / max(self.config.u_max, 1e-6))

        if not return_aux:
            return u

        aux = {
            "P": P,
            "q": q,
            "H": H,
            "b": b,
            "y": y,
            "z": solver_state.z,
            "lam": solver_state.lam,
            "residual_loss": residual_loss(P, q, H, b, y, solver_state),
        }
        return u, aux

    @torch.no_grad()
    def act_numpy(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        self.eval()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device)
        action = self.forward(state_t, ref_t)
        return action.squeeze(0).detach().cpu().numpy()

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "LearnedQPPolicy":
        payload: dict[str, Any] = torch.load(path, map_location=map_location)
        model = cls(LearnedQPConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
