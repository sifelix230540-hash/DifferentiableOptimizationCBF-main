"""Learn an implicit self-collision boundary h(q) via MLP.

Pipeline:
  1. generate_dataset  — Monte-Carlo sample C-space, label via coal backend
  2. train_boundary_mlp — train a small MLP to regress signed distance
  3. extract_tangent_planes — sample h(q)≈0 boundary, compute grad → local hyperplanes
  4. visualize_boundary_gui — PyBullet GUI to verify boundary quality
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import (  # noqa: E402
    Robot, SimulationScene, load_config, _resolve,
)
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (  # noqa: E402
    build_monitored_link_pairs,
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
    sample_revolute_configurations,
)
from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (  # noqa: E402
    build_coal_link_models,
    classify_self_collision_sample,
)

ARTIFACTS = Path(_resolve("artifacts/sdf_exp"))


# ── dataclass configs ──────────────────────────────────────────────
@dataclass
class DatasetConfig:
    num_samples: int = 200_000
    seed: int = 42
    min_index_gap: int = 2
    penetration_thresh: float = -0.001
    output_npz: str = str(ARTIFACTS / "boundary_dataset.npz")
    progress_every: int = 2000


@dataclass
class TrainConfig:
    dataset_npz: str = str(ARTIFACTS / "boundary_dataset.npz")
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    lr: float = 3e-4
    epochs: int = 200
    batch_size: int = 2048
    val_ratio: float = 0.1
    distance_clamp: float = 0.10
    model_pt: str = str(ARTIFACTS / "boundary_mlp.pt")
    meta_json: str = str(ARTIFACTS / "boundary_mlp_meta.json")


@dataclass
class TangentConfig:
    model_pt: str = str(ARTIFACTS / "boundary_mlp.pt")
    meta_json: str = str(ARTIFACTS / "boundary_mlp_meta.json")
    num_boundary_points: int = 2000
    h_threshold: float = 0.005
    output_json: str = str(ARTIFACTS / "boundary_tangent_planes.json")


@dataclass
class VisConfig:
    model_pt: str = str(ARTIFACTS / "boundary_mlp.pt")
    meta_json: str = str(ARTIFACTS / "boundary_mlp_meta.json")
    tangent_json: str = str(ARTIFACTS / "boundary_tangent_planes.json")
    num_test: int = 200
    seed: int = 99
    camera_distance: float = 1.45
    camera_yaw: float = -220.0
    camera_pitch: float = -28.0


# ── MLP ────────────────────────────────────────────────────────────
class BoundaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q).squeeze(-1)


# ── 1. dataset generation ─────────────────────────────────────────
def generate_dataset(cfg: DatasetConfig | None = None) -> Path:
    cfg = cfg or DatasetConfig()
    rng = np.random.default_rng(cfg.seed)
    created = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created = True
    try:
        robot = Robot(load_config())
        q_base, _ = robot.get_joint_state()
        rev_ids, _, joint_limits, q_indices = extract_revolute_metadata(robot)
        monitored_link_ids, _monitored_link_names = extract_self_collision_monitor_metadata(robot)
        pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=cfg.min_index_gap)
        link_models = build_coal_link_models(robot, monitored_link_ids)
        sampled = sample_revolute_configurations(
            q_base, q_indices, joint_limits,
            num_samples=cfg.num_samples, rng=rng,
        )
        dim = len(rev_ids)
        q_arr = np.empty((cfg.num_samples, dim), dtype=np.float32)
        dist_arr = np.empty(cfg.num_samples, dtype=np.float32)
        label_arr = np.empty(cfg.num_samples, dtype=np.int8)

        for i, q_full in enumerate(sampled):
            robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
            m = classify_self_collision_sample(
                robot, link_models=link_models,
                monitored_pairs=pairs,
                penetration_thresh=cfg.penetration_thresh,
            )
            q_arr[i] = q_full[q_indices].astype(np.float32)
            d = float(m["min_distance"])
            depth = m.get("contact_penetration_depth")
            if m["is_collision"] and depth is not None:
                d = min(d, float(depth))
            if m["is_collision"] and d >= 0.0:
                d = float(depth) if depth is not None else -1e-4
            dist_arr[i] = np.float32(d)
            label_arr[i] = np.int8(1 if m["is_collision"] else 0)
            if (i + 1) % cfg.progress_every == 0 or i == cfg.num_samples - 1:
                col = int(np.sum(label_arr[: i + 1]))
                pct = 100.0 * (i + 1) / cfg.num_samples
                print(f"\r[dataset] {pct:5.1f}%  [{i+1}/{cfg.num_samples}]  collision={col}", end="", flush=True)
        print()

        out = Path(cfg.output_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        lo = np.array([l for l, _ in joint_limits], dtype=np.float32)
        hi = np.array([h for _, h in joint_limits], dtype=np.float32)
        np.savez_compressed(
            out, q=q_arr, signed_distance=dist_arr, label=label_arr,
            joint_lower=lo, joint_upper=hi,
        )
        print(f"[dataset] saved -> {out}  ({cfg.num_samples} samples, collision={int(np.sum(label_arr))})")
        return out
    finally:
        if created and p.isConnected():
            p.disconnect()


# ── 2. train MLP ──────────────────────────────────────────────────
def train_boundary_mlp(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or TrainConfig()
    data = np.load(cfg.dataset_npz)
    q_all = torch.from_numpy(data["q"].astype(np.float32))
    sd_all = torch.from_numpy(data["signed_distance"].astype(np.float32))
    lo = torch.from_numpy(data["joint_lower"].astype(np.float32))
    hi = torch.from_numpy(data["joint_upper"].astype(np.float32))

    sd_all = sd_all.clamp(-cfg.distance_clamp, cfg.distance_clamp)
    span = (hi - lo).clamp(min=1e-6)
    q_norm = (q_all - lo) / span

    n = q_norm.shape[0]
    n_val = max(int(n * cfg.val_ratio), 1)
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = BoundaryMLP(q_norm.shape[1], cfg.hidden_dims)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm_t = train_idx[torch.randperm(train_idx.shape[0])]
        epoch_loss = 0.0
        count = 0
        for start in range(0, perm_t.shape[0], cfg.batch_size):
            idx = perm_t[start: start + cfg.batch_size]
            pred = model(q_norm[idx])
            loss = nn.functional.mse_loss(pred, sd_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * idx.shape[0]
            count += idx.shape[0]
        scheduler.step()
        train_mse = epoch_loss / max(count, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(q_norm[val_idx])
            val_mse = float(nn.functional.mse_loss(val_pred, sd_all[val_idx]))
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"[train] epoch {epoch:4d}/{cfg.epochs}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  best_val={best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(cfg.model_pt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    meta = {
        "input_dim": int(q_norm.shape[1]),
        "hidden_dims": cfg.hidden_dims,
        "joint_lower": lo.tolist(),
        "joint_upper": hi.tolist(),
        "distance_clamp": cfg.distance_clamp,
        "best_val_mse": best_val,
        "epochs": cfg.epochs,
    }
    Path(cfg.meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] saved model -> {out}")
    print(f"[train] saved meta  -> {cfg.meta_json}")
    return out


# ── helpers: load model ───────────────────────────────────────────
def _load_model(model_pt: str, meta_json: str) -> tuple[BoundaryMLP, dict]:
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    model = BoundaryMLP(int(meta["input_dim"]), meta["hidden_dims"])
    model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
    model.eval()
    return model, meta


def _normalize_q(q: np.ndarray, meta: dict) -> torch.Tensor:
    lo = np.array(meta["joint_lower"], dtype=np.float32)
    hi = np.array(meta["joint_upper"], dtype=np.float32)
    span = np.maximum(hi - lo, 1e-6)
    return torch.from_numpy(((q.astype(np.float32) - lo) / span))


# ── 3. extract tangent planes ─────────────────────────────────────
def extract_tangent_planes(cfg: TangentConfig | None = None) -> Path:
    cfg = cfg or TangentConfig()
    model, meta = _load_model(cfg.model_pt, cfg.meta_json)
    lo = np.array(meta["joint_lower"], dtype=np.float32)
    hi = np.array(meta["joint_upper"], dtype=np.float32)
    dim = len(lo)

    rng = np.random.default_rng(0)
    candidates = rng.uniform(lo, hi, size=(cfg.num_boundary_points * 50, dim)).astype(np.float32)
    q_t = _normalize_q(candidates, meta)
    with torch.no_grad():
        h_vals = model(q_t).numpy()
    near_mask = np.abs(h_vals) < cfg.h_threshold
    near_idx = np.where(near_mask)[0]
    if near_idx.size == 0:
        sorted_idx = np.argsort(np.abs(h_vals))
        near_idx = sorted_idx[: min(cfg.num_boundary_points, len(sorted_idx))]
    if near_idx.size > cfg.num_boundary_points:
        near_idx = rng.choice(near_idx, size=cfg.num_boundary_points, replace=False)

    boundary_q = candidates[near_idx]
    q_t_bd = torch.from_numpy(((boundary_q - lo) / np.maximum(hi - lo, 1e-6)))
    q_t_bd.requires_grad_(True)
    h = model(q_t_bd)
    grads = torch.autograd.grad(h.sum(), q_t_bd)[0].detach().numpy()
    h_vals_bd = h.detach().numpy()

    span = np.maximum(hi - lo, 1e-6).astype(np.float64)
    planes = []
    for i in range(boundary_q.shape[0]):
        grad_norm = grads[i] / span
        g_len = float(np.linalg.norm(grad_norm))
        if g_len < 1e-12:
            continue
        normal = (grad_norm / g_len).tolist()
        q0 = boundary_q[i].astype(np.float64)
        offset = float(-np.dot(grad_norm / g_len, q0))
        planes.append({
            "q0": q0.tolist(),
            "normal": normal,
            "offset": offset,
            "h_value": float(h_vals_bd[i]),
        })

    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_planes": len(planes),
        "h_threshold": cfg.h_threshold,
        "joint_lower": lo.tolist(),
        "joint_upper": hi.tolist(),
        "planes": planes,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tangent] extracted {len(planes)} tangent planes -> {out}")
    return out


# ── 4. GUI verification ──────────────────────────────────────────
def visualize_boundary_gui(cfg: VisConfig | None = None) -> None:
    cfg = cfg or VisConfig()
    model, meta = _load_model(cfg.model_pt, cfg.meta_json)
    lo = np.array(meta["joint_lower"], dtype=np.float32)
    hi = np.array(meta["joint_upper"], dtype=np.float32)
    dim = len(lo)

    tangent_data = json.loads(Path(cfg.tangent_json).read_text(encoding="utf-8"))
    planes = tangent_data.get("planes", [])

    rng = np.random.default_rng(cfg.seed)
    test_q = rng.uniform(lo, hi, size=(cfg.num_test, dim)).astype(np.float32)
    q_t = _normalize_q(test_q, meta)
    with torch.no_grad():
        h_vals = model(q_t).numpy()
    order = np.argsort(np.abs(h_vals))
    test_q = test_q[order]
    h_vals = h_vals[order]

    robot_cfg = load_config()
    scene = SimulationScene(robot_cfg)
    scene.enable_rendering()
    robot = Robot(robot_cfg)
    q_base, dq_base = robot.get_joint_state()
    rev_ids, rev_names, _, q_indices = extract_revolute_metadata(robot)
    monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(robot)
    pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=2)
    link_models = build_coal_link_models(robot, monitored_link_ids)
    monitored_names_by_id = {
        int(link_id): str(name)
        for link_id, name in zip(monitored_link_ids, monitored_link_names)
    }

    status_ids = [-1] * 8
    idx = 0
    last = -1

    def _compose(rq):
        q = q_base.copy()
        q[q_indices] = rq
        return q

    def _render(i):
        nonlocal last
        rq = test_q[i]
        q = _compose(rq)
        robot.set_joint_state(q, dq=np.zeros_like(q))
        m = classify_self_collision_sample(
            robot, link_models=link_models, monitored_pairs=pairs,
        )
        h_pred = float(h_vals[i])
        coal_d = float(m["min_distance"])
        coal_col = bool(m["is_collision"])
        nn_col = h_pred < 0

        violated = 0
        if planes:
            rq64 = rq.astype(np.float64)
            for pl in planes:
                val = float(np.dot(pl["normal"], rq64) + pl["offset"])
                if val < 0:
                    violated += 1

        ee_pos, _ = robot.get_ee_pose()
        base_pos, _ = robot.get_robobase_pose()
        tgt = 0.65 * np.array(ee_pos) + 0.35 * np.array(base_pos)
        tgt[2] = max(tgt[2], 0.35)
        p.resetDebugVisualizerCamera(cfg.camera_distance, cfg.camera_yaw, cfg.camera_pitch, tgt.tolist())

        anchor = np.array(base_pos) + np.array([0, -0.38, 0.92])
        lines = [
            f"[{i+1}/{len(test_q)}]  h(q)={h_pred:+.5f}  coal_dist={coal_d:+.5f}",
            f"NN collision={nn_col}  coal collision={coal_col}  agree={nn_col == coal_col}",
            f"tangent planes violated: {violated}/{len(planes)}",
            f"contact depth: {m.get('contact_penetration_depth', 'N/A')}",
            "N/Right=next  P/Left=prev  Q=quit",
        ]
        if len(status_ids) < len(lines):
            status_ids.extend([-1] * (len(lines) - len(status_ids)))
        for ti, line in enumerate(lines):
            pos = (anchor + np.array([0, 0, -0.07 * ti])).tolist()
            status_ids[ti] = p.addUserDebugText(
                line, pos, textColorRGB=[0.08, 0.08, 0.08], textSize=1.15,
                replaceItemUniqueId=status_ids[ti],
            )

        for lid in monitored_link_ids:
            color = [0.82, 0.82, 0.82, 1.0]
            ap = m.get("active_pair") or []
            if int(lid) in [int(x) for x in ap]:
                color = [0.95, 0.2, 0.1, 1.0] if coal_col else [0.1, 0.8, 0.3, 1.0]
            p.changeVisualShape(int(robot.body_id), int(lid), rgbaColor=color)

        print(f"[vis] {i+1}/{len(test_q)} h={h_pred:+.5f} coal={coal_d:+.5f} agree={nn_col==coal_col}")
        last = i

    print("[vis] GUI ready.")
    try:
        while p.isConnected():
            if last != idx:
                _render(idx)
            keys = p.getKeyboardEvents()
            triggered = lambda k: bool(keys.get(k, 0) & p.KEY_WAS_TRIGGERED)
            if triggered(ord("q")) or triggered(ord("Q")):
                break
            if triggered(ord("n")) or triggered(ord("N")) or triggered(p.B3G_RIGHT_ARROW):
                idx = (idx + 1) % len(test_q)
            elif triggered(ord("p")) or triggered(ord("P")) or triggered(p.B3G_LEFT_ARROW):
                idx = (idx - 1) % len(test_q)
            time.sleep(1 / 30)
    finally:
        robot.set_joint_state(q_base, dq_base)
        if p.isConnected():
            p.disconnect()


# ── CLI entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dataset", "train", "tangent", "vis", "all"])
    ap.add_argument("--num-samples", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--num-boundary", type=int, default=2000)
    args = ap.parse_args()

    if args.stage in ("dataset", "all"):
        generate_dataset(DatasetConfig(num_samples=args.num_samples))
    if args.stage in ("train", "all"):
        train_boundary_mlp(TrainConfig(epochs=args.epochs))
    if args.stage in ("tangent", "all"):
        extract_tangent_planes(TangentConfig(num_boundary_points=args.num_boundary))
    if args.stage in ("vis", "all"):
        visualize_boundary_gui()
