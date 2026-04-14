import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


URDF_PATH = Path(
    r"assets/cad_exports/model_CAD/Zu 5.SLDASM/urdf/Zu 5.SLDASM.urdf"
)
OUTPUT_PATH = Path("joint4_envelope.png")
TCP_SAMPLE_COUNT = 120_000


def parse_xyz(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()], dtype=float)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def joint_transform(origin_xyz: np.ndarray, origin_rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_matrix(origin_rpy)
    transform[:3, 3] = origin_xyz
    return transform


def rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def load_joint_data(urdf_path: Path) -> dict[str, dict[str, np.ndarray]]:
    root = ET.parse(urdf_path).getroot()
    joint_map: dict[str, dict[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        joint_map[joint.attrib["name"]] = {
            "xyz": parse_xyz(origin.attrib["xyz"]),
            "rpy": parse_xyz(origin.attrib["rpy"]),
            "axis": parse_xyz(axis.attrib["xyz"]),
        }
    return joint_map


def orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / np.linalg.norm(axis)
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(axis, trial)) > 0.95:
        trial = np.array([0.0, 0.0, 1.0], dtype=float)
    u = trial - np.dot(trial, axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)
    return u, v


def rotate_z_batch(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)
    x = cos_t * vectors[:, 0] - sin_t * vectors[:, 1]
    y = sin_t * vectors[:, 0] + cos_t * vectors[:, 1]
    z = vectors[:, 2]
    return np.column_stack((x, y, z))


def main() -> None:
    joints = load_joint_data(URDF_PATH)

    j1 = joints["Joint01"]
    j2 = joints["Joint02"]
    j3 = joints["Joint03"]
    j4 = joints["Joint04"]
    j5 = joints["Joint05"]
    j6 = joints["Joint06"]

    t_base_to_j1 = joint_transform(j1["xyz"], j1["rpy"])
    t_j1_to_j2 = joint_transform(j2["xyz"], j2["rpy"])
    t_base_to_j2 = t_base_to_j1 @ t_j1_to_j2

    axis_base = t_base_to_j2[:3, :3] @ (j2["axis"] / np.linalg.norm(j2["axis"]))
    axis_base = axis_base / np.linalg.norm(axis_base)
    origin_j2_base = t_base_to_j2[:3, 3]

    t23 = j3["xyz"]
    t34 = j4["xyz"]
    t45 = j5["xyz"]
    t56 = j6["xyz"]
    axis_local = j2["axis"] / np.linalg.norm(j2["axis"])
    r45 = rpy_to_matrix(j5["rpy"])

    t23_parallel_local = np.dot(t23, axis_local) * axis_local
    t34_parallel_local = np.dot(t34, axis_local) * axis_local
    t23_perp_local = t23 - t23_parallel_local
    t34_perp_local = t34 - t34_parallel_local

    l23 = np.linalg.norm(t23_perp_local)
    l34 = np.linalg.norm(t34_perp_local)
    r_outer = l23 + l34
    r_inner = abs(l23 - l34)

    center_base = origin_j2_base + t_base_to_j2[:3, :3] @ (t23_parallel_local + t34_parallel_local)
    u_base = t_base_to_j2[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=float)
    u_base = u_base - np.dot(u_base, axis_base) * axis_base
    u_base = u_base / np.linalg.norm(u_base)
    v_base = np.cross(axis_base, u_base)
    v_base = v_base / np.linalg.norm(v_base)
    plot_u, plot_v = orthonormal_basis_from_axis(axis_base)

    q2_vals = np.linspace(-math.pi, math.pi, 220)
    q3_vals = np.linspace(-math.pi, math.pi, 220)
    joint4_samples = []
    for q2 in q2_vals:
        r2 = rot_about_axis(axis_local, q2)
        for q3 in q3_vals:
            r3 = rot_about_axis(axis_local, q3)
            p_local = r2 @ t23 + r2 @ r3 @ t34
            p_base = origin_j2_base + t_base_to_j2[:3, :3] @ p_local
            joint4_samples.append(p_base)
    joint4_samples = np.array(joint4_samples)

    rng = np.random.default_rng(0)
    q2_tcp = rng.uniform(-math.pi, math.pi, TCP_SAMPLE_COUNT)
    q3_tcp = rng.uniform(-math.pi, math.pi, TCP_SAMPLE_COUNT)
    q4_tcp = rng.uniform(-math.pi, math.pi, TCP_SAMPLE_COUNT)
    q5_tcp = rng.uniform(-math.pi, math.pi, TCP_SAMPLE_COUNT)

    theta2 = q2_tcp
    theta23 = q2_tcp + q3_tcp
    theta234 = q2_tcp + q3_tcp - q4_tcp

    t23_batch = np.broadcast_to(t23, (TCP_SAMPLE_COUNT, 3))
    t34_batch = np.broadcast_to(t34, (TCP_SAMPLE_COUNT, 3))
    t56_batch = np.broadcast_to(t56, (TCP_SAMPLE_COUNT, 3))

    joint4_local_tcp = rotate_z_batch(t23_batch, theta2) + rotate_z_batch(t34_batch, theta23)
    t56_after_q5 = rotate_z_batch(t56_batch, q5_tcp)
    tcp_offset_local = t45 + t56_after_q5 @ r45.T
    tcp_local = joint4_local_tcp + rotate_z_batch(tcp_offset_local, theta234)
    tcp_samples = origin_j2_base + tcp_local @ t_base_to_j2[:3, :3].T

    tcp_offset_norm = np.linalg.norm(tcp_offset_local, axis=1)
    tcp_offset_radius = float(np.mean(tcp_offset_norm))
    tcp_projected_offset_max = float(np.max(np.linalg.norm(tcp_offset_local[:, :2], axis=1)))
    tcp_projected_offset_min = float(np.min(np.linalg.norm(tcp_offset_local[:, :2], axis=1)))
    tcp_projected_outer = r_outer + tcp_projected_offset_max
    if r_inner <= tcp_projected_offset_max and r_outer >= tcp_projected_offset_min:
        tcp_projected_inner = 0.0
    else:
        tcp_projected_inner = max(
            0.0,
            min(abs(r_inner - tcp_projected_offset_max), abs(tcp_projected_offset_min - r_outer)),
        )

    theta = np.linspace(0.0, 2.0 * math.pi, 500)
    outer_circle = center_base + r_outer * (
        np.outer(np.cos(theta), u_base) + np.outer(np.sin(theta), v_base)
    )
    inner_circle = center_base + r_inner * (
        np.outer(np.cos(theta), u_base) + np.outer(np.sin(theta), v_base)
    )
    tcp_outer_circle = center_base + tcp_projected_outer * (
        np.outer(np.cos(theta), u_base) + np.outer(np.sin(theta), v_base)
    )
    tcp_inner_circle = center_base + tcp_projected_inner * (
        np.outer(np.cos(theta), u_base) + np.outer(np.sin(theta), v_base)
    )

    def project(points: np.ndarray) -> np.ndarray:
        rel = points - center_base
        return np.column_stack((rel @ plot_u, rel @ plot_v))

    joint4_samples_2d = project(joint4_samples)
    tcp_samples_2d = project(tcp_samples)
    outer_2d = project(outer_circle)
    inner_2d = project(inner_circle)
    tcp_outer_2d = project(tcp_outer_circle)
    tcp_inner_2d = project(tcp_inner_circle)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(
        joint4_samples_2d[:, 0],
        joint4_samples_2d[:, 1],
        s=1,
        alpha=0.05,
        label="Joint04 origin sample",
    )
    ax1.scatter(
        tcp_samples_2d[:, 0],
        tcp_samples_2d[:, 1],
        s=1,
        alpha=0.03,
        color="tab:purple",
        label="TCP sample (Joint06 origin)",
    )
    ax1.plot(outer_2d[:, 0], outer_2d[:, 1], color="tab:red", lw=2.5, label="Joint04 outer envelope")
    ax1.plot(inner_2d[:, 0], inner_2d[:, 1], color="tab:green", lw=2.5, label="Joint04 inner envelope")
    ax1.plot(
        tcp_outer_2d[:, 0],
        tcp_outer_2d[:, 1],
        color="tab:purple",
        lw=2.2,
        ls="--",
        label="TCP projected outer bound",
    )
    if tcp_projected_inner > 1e-9:
        ax1.plot(
            tcp_inner_2d[:, 0],
            tcp_inner_2d[:, 1],
            color="tab:purple",
            lw=2.2,
            ls=":",
            label="TCP projected inner bound",
        )
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Projected envelope on plane normal to shared axis\n(q1 fixed, TCP := Joint06 origin)")
    ax1.set_xlabel("Plane axis u (m)")
    ax1.set_ylabel("Plane axis v (m)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(joint4_samples[:, 0], joint4_samples[:, 1], joint4_samples[:, 2], s=1, alpha=0.02)
    ax2.scatter(
        tcp_samples[:, 0],
        tcp_samples[:, 1],
        tcp_samples[:, 2],
        s=1,
        alpha=0.03,
        color="tab:purple",
        label="TCP sample",
    )
    ax2.plot(outer_circle[:, 0], outer_circle[:, 1], outer_circle[:, 2], color="tab:red", lw=2.0)
    ax2.plot(inner_circle[:, 0], inner_circle[:, 1], inner_circle[:, 2], color="tab:green", lw=2.0)
    ax2.plot(
        tcp_outer_circle[:, 0],
        tcp_outer_circle[:, 1],
        tcp_outer_circle[:, 2],
        color="tab:purple",
        lw=2.0,
        ls="--",
    )
    ax2.scatter(*origin_j2_base, color="black", s=40, label="Joint02 origin")
    ax2.scatter(*center_base, color="tab:orange", s=40, label="Annulus center")
    ax2.set_title("Base frame view")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_zlabel("z (m)")
    ax2.legend(loc="upper left")

    fig.suptitle(
        f"Joint04 + TCP envelope from URDF | "
        f"Joint04: [{r_inner:.4f}, {r_outer:.4f}] m, "
        f"TCP offset={tcp_offset_radius:.4f} m"
    )
    fig.savefig(OUTPUT_PATH, dpi=220)

    print(f"Saved plot to: {OUTPUT_PATH.resolve()}")
    print(f"Joint02 origin: {origin_j2_base}")
    print(f"Envelope center: {center_base}")
    print(f"Axis direction: {axis_base}")
    print(f"Inner radius: {r_inner:.6f} m")
    print(f"Outer radius: {r_outer:.6f} m")
    print(f"TCP offset magnitude (Joint04 -> Joint06 origin): {tcp_offset_radius:.6f} m")
    print(f"TCP projected radial offset range: [{tcp_projected_offset_min:.6f}, {tcp_projected_offset_max:.6f}] m")
    print(f"TCP projected radial envelope: [{tcp_projected_inner:.6f}, {tcp_projected_outer:.6f}] m")


if __name__ == "__main__":
    main()
