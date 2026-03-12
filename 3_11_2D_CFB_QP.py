import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp
from matplotlib.patches import Circle, Polygon

# =====================================================================
# CBF-QP Safety Filter Benchmark (with Animation)
# Scene: static corridor + multiple dynamic crossing obstacles
# Compare: Nominal Controller (no safety) vs CBF-QP Safety Filter
# =====================================================================

# --- 参数 ---
dt = 0.01
alpha_cbf = 4
u_max = 3
kp = 1.2
max_steps = 1500
goal_tol = 0.15
safety_margin = 0.1
slack_weight = 800.0

start = np.array([0.0, 0.0])
goal = np.array([5.0, 0.0])

# --- 静态障碍: 形成宽度约 1.1 的水平通道 ---
STATIC_OBS = [
    {"center": np.array([2.5, 1.1]),  "radius": 0.55, "velocity": np.zeros(2)},
    {"center": np.array([2.5, -1.1]), "radius": 0.55, "velocity": np.zeros(2)},
]

# =====================================================================
# Dynamic obstacles
# =====================================================================

def dyn1_center(t):
    return np.array([2.5, max(1.3 - 1.2 * t, -1.5)])

def dyn1_velocity(t):
    return np.array([0.0, -1.2]) if 1.3 - 1.2 * t > -1.5 else np.zeros(2)

def dyn2_center(t):
    if t < 0.5:
        return np.array([3.5, 1.3])
    y = 1.3 - 0.8 * (t - 0.5)
    return np.array([3.5, max(y, -1.5)])

def dyn2_velocity(t):
    if t < 0.5:
        return np.zeros(2)
    return np.array([0.0, -0.8]) if 1.3 - 0.8 * (t - 0.5) > -1.5 else np.zeros(2)

def dyn3_center(t):
    if t < 2.0:
        return np.array([4.2, -1.3])
    y = -1.3 + 0.9 * (t - 2.0)
    return np.array([4.2, min(y, 1.5)])

def dyn3_velocity(t):
    if t < 2.0:
        return np.zeros(2)
    return np.array([0.0, 0.9]) if -1.3 + 0.9 * (t - 2.0) < 1.5 else np.zeros(2)

# 倾斜轨迹障碍: 从左上斜穿至右下
_dir_norm = np.sqrt(1.0 + 0.36)
_dx4, _dy4 = 1.0 / _dir_norm, -0.6 / _dir_norm

def dyn4_center(t):
    if t < 0.5:
        return np.array([1.6, 1.0])
    prog = min((t - 0.5) * 1.1, 3.5)
    return np.array([1.6 + prog * _dx4, 1.0 + prog * _dy4])

def dyn4_velocity(t):
    if t < 0.5:
        return np.zeros(2)
    if (t - 0.5) * 1.1 >= 3.5:
        return np.zeros(2)
    return np.array([1.1 * _dx4, 1.1 * _dy4])

DYN_OBS_DEFS = [
    {"center_fn": dyn1_center, "velocity_fn": dyn1_velocity,
     "radius": 0.22, "color": "orange",     "name": "DynObs1 (x=2.5 ↓)"},
    {"center_fn": dyn2_center, "velocity_fn": dyn2_velocity,
     "radius": 0.20, "color": "gold",       "name": "DynObs2 (x=3.5 ↓)"},
    {"center_fn": dyn3_center, "velocity_fn": dyn3_velocity,
     "radius": 0.20, "color": "darkorange", "name": "DynObs3 (x=4.2 ↑)"},
    {"center_fn": dyn4_center, "velocity_fn": dyn4_velocity,
     "radius": 0.20, "color": "purple",     "name": "DynObs4 (inclined ↘)"},
]


def get_obstacles(t):
    obs = [dict(o) for o in STATIC_OBS]
    for d in DYN_OBS_DEFS:
        obs.append({
            "center": d["center_fn"](t),
            "radius": d["radius"],
            "velocity": d["velocity_fn"](t),
        })
    return obs


def barrier_fn(p, c, r):
    d = p - c
    nd = np.linalg.norm(d)
    h = nd - (r + safety_margin)
    g = d / nd if nd > 1e-9 else np.array([1.0, 0.0])
    return h, g


def barrier_at_next_step(x, c_next, r):
    """
    离散时间一阶近似:
      h(x + dt*u, t+dt) ≈ h(x, t+dt) + dt * grad_h · u
    其中 h(x, t+dt) = ||x - c(t+dt)|| - (r + margin)
    """
    d = x - c_next
    nd = np.linalg.norm(d)
    h_next = nd - (r + safety_margin)
    if nd < 1e-9:
        return 0.0, np.array([1.0, 0.0])
    grad_h = d / nd
    return h_next, grad_h


def nominal_ctrl(x):
    return np.clip(kp * (goal - x), -u_max, u_max)


# =====================================================================
# Simulation
# =====================================================================
def simulate(use_cbf):
    x = start.copy()
    path = [x.copy()]
    h_hist = []
    fails = 0
    reached = False
    slack_hist = []

    for step in range(max_steps):
        t = step * dt
        obs = get_obstacles(t)
        bp = [barrier_fn(x, o["center"], o["radius"]) for o in obs]
        h_hist.append(min(p[0] for p in bp))

        u_des = nominal_ctrl(x)
        u_cmd = u_des.copy()
        step_max_slack = 0.0

        if use_cbf:
            u_var = cp.Variable(2)
            slk = cp.Variable(len(obs), nonneg=True)
            cons = [u_var >= -u_max, u_var <= u_max]
            for i, (hv, gh) in enumerate(bp):
                v_obs = obs[i]["velocity"]
                # 标准 CBF-QP: ∇h·(u - v_obs) + α·h + s >= 0
                cons.append(gh @ (u_var - v_obs) + alpha_cbf * hv + slk[i] >= 0)
            obj = cp.Minimize(
                cp.sum_squares(u_var - u_des)
                + slack_weight * cp.sum(slk)
                + 1e-2 * cp.sum_squares(slk)
            )
            prob = cp.Problem(obj, cons)
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                           eps_abs=1e-7, eps_rel=1e-7)
            except cp.SolverError:
                fails += 1
            else:
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u_var.value is not None:
                    u_cmd = np.asarray(u_var.value).reshape(2)
                    if slk.value is not None:
                        step_max_slack = float(np.max(np.asarray(slk.value)))
                else:
                    fails += 1

        x = x + dt * u_cmd
        path.append(x.copy())
        slack_hist.append(step_max_slack)
        if np.linalg.norm(x - goal) < goal_tol:
            reached = True
            break

    return {
        "path": np.array(path),
        "h": np.array(h_hist),
        "reached": reached,
        "fails": fails,
        "slack": np.array(slack_hist),
    }


# =====================================================================
# Run
# =====================================================================
print("Simulating Nominal vs CBF-QP ...")
res_nom = simulate(False)
res_cbf = simulate(True)

nom_col = int(np.sum(np.array(res_nom["h"]) < 0))
cbf_col = int(np.sum(np.array(res_cbf["h"]) < 0))
cbf_max_slack = float(np.max(res_cbf["slack"])) if len(res_cbf["slack"]) > 0 else 0.0
print(f"Nominal: reached={res_nom['reached']}, collision_steps={nom_col}, "
      f"min_h={min(res_nom['h']):.4f}, steps={len(res_nom['h'])}")
print(f"CBF-QP:  reached={res_cbf['reached']}, collision_steps={cbf_col}, "
      f"min_h={min(res_cbf['h']):.4f}, steps={len(res_cbf['h'])}, "
      f"max_slack={cbf_max_slack:.4e}")

# =====================================================================
# Static comparison figure
# =====================================================================
fig_s, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
th = np.linspace(0, 2 * np.pi, 200)

for i, o in enumerate(STATIC_OBS):
    ax1.add_patch(Circle(o["center"], o["radius"], color="red", alpha=0.22,
                         label="Static Obstacles" if i == 0 else None))
    ax1.plot(o["center"][0] + o["radius"] * np.cos(th),
             o["center"][1] + o["radius"] * np.sin(th), "r-", lw=1.4)

for ts in [0.5, 3.0, 5.0]:
    for d in DYN_OBS_DEFS:
        c = d["center_fn"](ts)
        r = d["radius"]
        ax1.add_patch(Circle(c, r, color=d["color"], alpha=0.10))
        ax1.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                 "--", color=d["color"], lw=0.7)

for d in DYN_OBS_DEFS:
    ax1.plot([], [], "-", color=d["color"], lw=2, label=d["name"])

ax1.plot(res_nom["path"][:, 0], res_nom["path"][:, 1],
         "--", color="gray", lw=2.2, label=f"Nominal (col={nom_col})")
ax1.plot(res_cbf["path"][:, 0], res_cbf["path"][:, 1],
         "-o", color="blue", ms=1.5, lw=1.6, label=f"CBF-QP (col={cbf_col})")
ax1.scatter(*start, color="green", s=100, zorder=5, label="Start")
ax1.scatter(*goal, color="blue", marker="*", s=200, zorder=5, label="Goal")
ax1.set_xlim(-0.5, 5.8)
ax1.set_ylim(-1.8, 1.8)
ax1.set_aspect("equal")
ax1.grid(True)
ax1.legend(loc="upper left", fontsize=6.5)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title("Trajectory Comparison")

t_n = np.arange(len(res_nom["h"])) * dt
t_c = np.arange(len(res_cbf["h"])) * dt
ax2.plot(t_n, res_nom["h"], "--", color="gray", lw=2, label="Nominal min h")
ax2.plot(t_c, res_cbf["h"], "-", color="blue", lw=2, label="CBF-QP min h")
ax2.axhline(0, color="red", ls=":", lw=1.2, label="Safety boundary")
ax2.fill_between(t_n, res_nom["h"], 0, where=np.array(res_nom["h"]) < 0,
                 color="red", alpha=0.15, label="Collision region")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("min h(x)")
ax2.set_title(f"Safety (nom={min(res_nom['h']):.3f}, cbf={min(res_cbf['h']):.3f})")
ax2.grid(True)
ax2.legend(fontsize=8)
fig_s.suptitle(
    f"CBF-QP Safety Filter | nom reached={res_nom['reached']}, "
    f"cbf reached={res_cbf['reached']}, qp fails={res_cbf['fails']}, "
    f"max slack={cbf_max_slack:.2e}", fontsize=11)
plt.tight_layout()

# =====================================================================
# Animation
# =====================================================================
n_nom, n_cbf = len(res_nom["path"]), len(res_cbf["path"])
n_max = max(n_nom, n_cbf)
skip = 4
frame_list = list(range(0, n_max, skip))

fig_a, ax_a = plt.subplots(figsize=(10, 5.5))

for o in STATIC_OBS:
    ax_a.add_patch(Circle(o["center"], o["radius"], color="red", alpha=0.22))
    ax_a.plot(o["center"][0] + o["radius"] * np.cos(th),
              o["center"][1] + o["radius"] * np.sin(th), "r-", lw=1.4)

dyn_polys = []
dyn_lines = []
for d in DYN_OBS_DEFS:
    c0 = d["center_fn"](0)
    r = d["radius"]
    pts = np.column_stack([c0[0] + r * np.cos(th), c0[1] + r * np.sin(th)])
    poly = Polygon(pts, color=d["color"], alpha=0.35)
    ax_a.add_patch(poly)
    line, = ax_a.plot([], [], "-", color=d["color"], lw=1.5)
    dyn_polys.append(poly)
    dyn_lines.append(line)

nom_trail, = ax_a.plot([], [], "--", color="gray", lw=1.5, alpha=0.8, label="Nominal")
cbf_trail, = ax_a.plot([], [], "-", color="blue", lw=1.8, label="CBF-QP")
nom_dot, = ax_a.plot([], [], "o", color="dimgray", ms=11, zorder=10)
cbf_dot, = ax_a.plot([], [], "o", color="royalblue", ms=11, zorder=10)

ax_a.scatter(*start, color="green", s=100, zorder=5, label="Start")
ax_a.scatter(*goal, color="blue", marker="*", s=200, zorder=5, label="Goal")

info = ax_a.text(0.02, 0.95, "", transform=ax_a.transAxes, fontsize=9, va="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75))

ax_a.set_xlim(-0.5, 5.8)
ax_a.set_ylim(-1.8, 1.8)
ax_a.set_aspect("equal")
ax_a.grid(True)
ax_a.legend(loc="upper right", fontsize=8)
ax_a.set_xlabel("x1")
ax_a.set_ylabel("x2")
ax_a.set_title("CBF-QP Safety Filter: Real-time Comparison")


def anim_update(fi):
    idx = frame_list[fi]
    t = idx * dt

    for k, d in enumerate(DYN_OBS_DEFS):
        c = d["center_fn"](t)
        r = d["radius"]
        cx = c[0] + r * np.cos(th)
        cy = c[1] + r * np.sin(th)
        dyn_polys[k].set_xy(np.column_stack([cx, cy]))
        dyn_lines[k].set_data(cx, cy)

    ni = min(idx, n_nom - 1)
    nom_trail.set_data(res_nom["path"][:ni + 1, 0], res_nom["path"][:ni + 1, 1])
    nom_dot.set_data([res_nom["path"][ni, 0]], [res_nom["path"][ni, 1]])

    ci = min(idx, n_cbf - 1)
    cbf_trail.set_data(res_cbf["path"][:ci + 1, 0], res_cbf["path"][:ci + 1, 1])
    cbf_dot.set_data([res_cbf["path"][ci, 0]], [res_cbf["path"][ci, 1]])

    nh = res_nom["h"][min(ni, len(res_nom["h"]) - 1)]
    ch = res_cbf["h"][min(ci, len(res_cbf["h"]) - 1)]

    nom_dot.set_color("red" if nh < 0 else ("goldenrod" if nh < 0.1 else "dimgray"))
    cbf_dot.set_color("red" if ch < 0 else ("cyan" if ch < 0.1 else "royalblue"))

    sn = "!! COLLISION !!" if nh < 0 else "safe"
    sc = "!! COLLISION !!" if ch < 0 else "safe"
    info.set_text(
        f"t = {t:.2f}s\n"
        f"Nominal  h = {nh:+.3f}  [{sn}]\n"
        f"CBF-QP   h = {ch:+.3f}  [{sc}]"
    )
    return [nom_trail, cbf_trail, nom_dot, cbf_dot, info] + dyn_polys + dyn_lines


ani = animation.FuncAnimation(fig_a, anim_update, frames=len(frame_list),
                              interval=50, blit=False)

save_dir = (r"c:\Users\12049\OneDrive\Desktop\科研相关\博一春季"
            r"\免示教焊接轨迹规划\相关资料\CBF_grad_optim_on_trajPlanning"
            r"\DifferentiableOptimizationCBF-main")
gif_path = save_dir + r"\cbf_demo.gif"
print(f"Saving animation to {gif_path} ...")
try:
    ani.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"GIF saved: {gif_path}")
except Exception as e:
    print(f"GIF save failed: {e}. Will show animation in window instead.")

plt.show()
