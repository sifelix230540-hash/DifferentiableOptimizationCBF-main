from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PDHGState:
    z: torch.Tensor
    lam: torch.Tensor


def project_nonnegative(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def recover_primal(
    P: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    damping: float = 1e-5,
) -> torch.Tensor:
    """Recover the primal variable y from the paper-style z variable.

    Shapes:
    - P:   [B, n, n]
    - q:   [B, n]
    - H:   [B, m, n]
    - b:   [B, m]
    - z:   [B, m]
    """
    batch, n, _ = P.shape
    _, m, _ = H.shape
    eye_n = torch.eye(n, device=P.device, dtype=P.dtype).expand(batch, -1, -1)
    eye_m = torch.eye(m, device=P.device, dtype=P.dtype).expand(batch, -1, -1)

    P_inv = torch.linalg.inv(P + damping * eye_n)
    hpht = H @ P_inv @ H.transpose(-1, -2)
    hpht_pinv = torch.linalg.pinv(hpht + damping * eye_m)
    rhs = (z - b + (H @ P_inv @ q.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
    y = -torch.bmm(P_inv, q.unsqueeze(-1)) + torch.bmm(
        torch.bmm(P_inv, H.transpose(-1, -2)),
        torch.bmm(hpht_pinv, rhs),
    )
    return y.squeeze(-1)


def pdhg_unroll(
    P: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    b: torch.Tensor,
    n_iter: int,
    alpha: float = 0.8,
    damping: float = 1e-5,
) -> tuple[torch.Tensor, PDHGState]:
    """Unroll the PDHG-style QP solver used by the learned controller.

    The implementation follows the paper's high-level structure:
    - compute problem-dependent affine transform terms
    - alternate affine updates and projection to the positive orthant
    - recover the primal variable after a fixed number of iterations
    """
    batch, n, _ = P.shape
    _, m, _ = H.shape
    eye_n = torch.eye(n, device=P.device, dtype=P.dtype).expand(batch, -1, -1)
    eye_m = torch.eye(m, device=P.device, dtype=P.dtype).expand(batch, -1, -1)

    P_inv = torch.linalg.inv(P + damping * eye_n)
    hpht = H @ P_inv @ H.transpose(-1, -2)
    F = torch.linalg.inv(eye_m + hpht + damping * eye_m)
    mu = torch.bmm(
        F,
        (torch.bmm(H, torch.bmm(P_inv, q.unsqueeze(-1))) - b.unsqueeze(-1)),
    ).squeeze(-1)

    z = torch.zeros(batch, m, device=P.device, dtype=P.dtype)
    lam = torch.zeros(batch, m, device=P.device, dtype=P.dtype)

    for _ in range(n_iter):
        z = project_nonnegative(
            torch.bmm(eye_m - 2.0 * alpha * F, z.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(alpha * (eye_m - 2.0 * F), lam.unsqueeze(-1)).squeeze(-1)
            - 2.0 * alpha * mu
        )
        lam = torch.bmm(F, (z + lam).unsqueeze(-1)).squeeze(-1) + mu

    y = recover_primal(P=P, q=q, H=H, b=b, z=z, damping=damping)
    return y, PDHGState(z=z, lam=lam)


def residual_loss(
    P: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    state: PDHGState,
) -> torch.Tensor:
    prim = torch.bmm(H, y.unsqueeze(-1)).squeeze(-1) + b - state.z
    dual = torch.bmm(P, y.unsqueeze(-1)).squeeze(-1) + q + torch.bmm(
        H.transpose(-1, -2),
        state.lam.unsqueeze(-1),
    ).squeeze(-1)
    return (prim.square().sum(dim=-1) + dual.square().sum(dim=-1)).mean()
