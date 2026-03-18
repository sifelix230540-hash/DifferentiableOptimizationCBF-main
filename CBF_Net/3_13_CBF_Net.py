"""Neural SDF (Signed Distance Field) 学习演示。

用神经网络从点云数据中学习障碍物的有符号距离场，
训练后的网络可直接作为 CBF 的 h(x) 使用。

训练目标：
  1. 表面点处 b(x) ≈ 0
  2. 空间中任意点处 ||∇b(x)|| ≈ 1  (Eikonal 约束，保证是合法 SDF)
  3. 表面外侧 b > 0，内侧 b < 0     (符号监督)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
#  1. 神经 SDF 网络
# ============================================================================

class NeuralSDF(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
#  2. 生成训练数据
# ============================================================================

def make_circle_data(
    center=(0.0, 0.0), radius=1.0,
    n_surface=500, n_space=2000, noise_std=0.01,
):
    """生成圆形障碍物的表面点云 + 带符号标签的空间采样。

    返回:
      surface_pts : (n_surface, 2) 表面点（加微量噪声模拟传感器误差）
      space_pts   : (n_space, 2)   空间均匀采样
      space_sdf   : (n_space, 1)   对应的真实 SDF 值
    """
    cx, cy = center

    # 表面点：沿圆周均匀采样 + 径向高斯噪声
    theta = np.random.uniform(0, 2 * np.pi, n_surface)
    r_noise = np.random.randn(n_surface) * noise_std
    surface_pts = np.stack([
        cx + (radius + r_noise) * np.cos(theta),
        cy + (radius + r_noise) * np.sin(theta),
    ], axis=1)

    # 空间采样：在 [-3, 3]^2 内均匀撒点
    space_pts = np.random.uniform(-3, 3, (n_space, 2))
    dist = np.sqrt((space_pts[:, 0] - cx) ** 2 + (space_pts[:, 1] - cy) ** 2)
    space_sdf = (dist - radius).reshape(-1, 1)

    return (
        torch.tensor(surface_pts, dtype=torch.float32),
        torch.tensor(space_pts, dtype=torch.float32),
        torch.tensor(space_sdf, dtype=torch.float32),
    )


# ============================================================================
#  3. 训练
# ============================================================================

def train(epochs=3000, lr=1e-3):
    model = NeuralSDF()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    surface_pts, space_pts, space_sdf = make_circle_data()

    # 损失权重
    w_surface = 30.0   # 表面点 b→0
    w_eikonal = 1.0    # ||∇b||→1
    w_sdf = 1.0        # 空间中 b 与真实 SDF 的回归

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- A. 表面损失：表面点处 |b| → 0 ---
        b_surface = model(surface_pts)
        loss_surface = torch.mean(b_surface ** 2)

        # --- B. SDF 回归损失：空间中 b 接近真实 SDF ---
        b_space = model(space_pts)
        loss_sdf = torch.mean((b_space - space_sdf) ** 2)

        # --- C. Eikonal 损失：||∇b|| → 1 ---
        coords = torch.randn(1000, 2, requires_grad=True)
        b_grid = model(coords)
        grad = torch.autograd.grad(
            b_grid, coords,
            grad_outputs=torch.ones_like(b_grid),
            create_graph=True,
        )[0]
        grad_norm = torch.linalg.norm(grad, dim=1)
        loss_eikonal = torch.mean((grad_norm - 1.0) ** 2)

        total = w_surface * loss_surface + w_sdf * loss_sdf + w_eikonal * loss_eikonal
        total.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"[Epoch {epoch:4d}]  "
                f"surface={loss_surface.item():.4f}  "
                f"sdf={loss_sdf.item():.4f}  "
                f"eikonal={loss_eikonal.item():.4f}  "
                f"total={total.item():.4f}"
            )

    return model, surface_pts


# ============================================================================
#  4. 可视化
# ============================================================================

def visualize(model: NeuralSDF, surface_pts: torch.Tensor):
    model.eval()
    res = 300
    x = np.linspace(-3, 3, res)
    y = np.linspace(-3, 3, res)
    xx, yy = np.meshgrid(x, y)
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        zz = model(grid).numpy().reshape(res, res)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：SDF 色图 + 零等值线（学到的障碍边界）
    ax = axes[0]
    c = ax.contourf(xx, yy, zz, levels=50, cmap="RdBu")
    fig.colorbar(c, ax=ax, label="b(x)")
    ax.contour(xx, yy, zz, levels=[0.0], colors="lime", linewidths=2)
    sp = surface_pts.numpy()
    ax.scatter(sp[:, 0], sp[:, 1], s=1, c="black", alpha=0.3, label="surface pts")
    ax.set_title("Learned SDF  (green = b=0 contour)")
    ax.set_aspect("equal")
    ax.legend()

    # 右图：梯度模长分布（理想为全 1）
    ax = axes[1]
    grid.requires_grad_(True)
    b = model(grid)
    grad = torch.autograd.grad(b, grid, torch.ones_like(b), create_graph=False)[0]
    gn = torch.linalg.norm(grad, dim=1).detach().numpy().reshape(res, res)
    c2 = ax.contourf(xx, yy, gn, levels=50, cmap="viridis")
    fig.colorbar(c2, ax=ax, label="||grad b||")
    ax.contour(xx, yy, gn, levels=[1.0], colors="red", linewidths=1.5)
    ax.set_title("Gradient norm  (red = ||grad||=1)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("neural_sdf_result.png", dpi=150)
    print("可视化已保存: neural_sdf_result.png")
    plt.show()


# ============================================================================
#  主入口
# ============================================================================

if __name__ == "__main__":
    model, surface_pts = train()
    visualize(model, surface_pts)
