import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp
from matplotlib.patches import Circle, Polygon
import time
from pathlib import Path

# =====================================================================
# Nominal vs CBF-QP vs MPC-DCBF Benchmark
# =====================================================================

# --- 基础参数 ---
dt = 0.01           # 增大步长: 0.01 -> 0.05 (减少5倍迭代)
max_steps = 1500      # 相应减少 (0.05*400=20s, 足够到达目标)
goal_tol = 0.15
safety_margin = 0.2

start = np.array([0.0, 0.0])
goal = np.array([5.0, 0.0])

# --- 单步 CBF-QP 参数 ---
alpha_cbf = 4
u_max = 3
kp = 1.2
slack_weight = 800.0

# --- MPC-DCBF 专有参数 ---
N_mpc = 15           # 预测时域 (dt=0.05时, 往后看0.5s)
gamma_dcbf = 0.2     # 离散CBF的衰减率 (0 < gamma <= 1)
mpc_Q = 1.5          # 目标跟踪权重
mpc_R = 0.2          # 控制输入平滑权重
mpc_slack_W = 5000 # 避障松弛变量惩罚权重

# =====================================================================
# 环境与障碍物定义
# =====================================================================

STATIC_OBS = [
    {"center": np.array([2.5, 1.1]),  "radius": 0.55, "velocity": np.zeros(2)},
    {"center": np.array([2.5, -1.1]), "radius": 0.55, "velocity": np.zeros(2)},
]

def dyn1_center(t): return np.array([2.5, max(1.3 - 1.2 * t, -1.5)])
def dyn1_velocity(t): return np.array([0.0, -1.2]) if 1.3 - 1.2 * t > -1.5 else np.zeros(2)

def dyn2_center(t):
    if t < 0.5: return np.array([3.5, 1.3])
    y = 1.3 - 0.8 * (t - 0.5)
    return np.array([3.5, max(y, -1.5)])
def dyn2_velocity(t):
    if t < 0.5: return np.zeros(2)
    return np.array([0.0, -0.8]) if 1.3 - 0.8 * (t - 0.5) > -1.5 else np.zeros(2)

def dyn3_center(t):
    if t < 1: return np.array([4.2, -1.3])
    y = -1.3 + 0.9 * (t - 1)
    return np.array([4.2, min(y, 1.5)])
def dyn3_velocity(t):
    if t < 2.0: return np.zeros(2)
    return np.array([0.0, 0.9]) if -1.3 + 0.9 * (t - 2.0) < 1.5 else np.zeros(2)

_dir_norm = np.sqrt(1.0 + 0.36)
_dx4, _dy4 = 1.0 / _dir_norm, -0.6 / _dir_norm

def dyn4_center(t):
    if t < 0.5: return np.array([1.6, 1.0])
    prog = min((t - 0.5) * 1.1, 3.5)
    return np.array([1.6 + prog * _dx4, 1.0 + prog * _dy4])
def dyn4_velocity(t):
    if t < 0.5: return np.zeros(2)
    if (t - 0.5) * 1.1 >= 3.5: return np.zeros(2)
    return np.array([1.1 * _dx4, 1.1 * _dy4])

DYN_OBS_DEFS = [
    {"center_fn": dyn1_center, "velocity_fn": dyn1_velocity, "radius": 0.22, "color": "orange",     "name": "DynObs1"},
    {"center_fn": dyn2_center, "velocity_fn": dyn2_velocity, "radius": 0.20, "color": "gold",       "name": "DynObs2"},
    {"center_fn": dyn3_center, "velocity_fn": dyn3_velocity, "radius": 0.20, "color": "darkorange", "name": "DynObs3"},
    {"center_fn": dyn4_center, "velocity_fn": dyn4_velocity, "radius": 0.20, "color": "purple",     "name": "DynObs4"},
]

def get_obstacles_at(t):
    obs = [{"center": o["center"].copy(), "radius": o["radius"], "velocity": o["velocity"].copy()} for o in STATIC_OBS]
    for d in DYN_OBS_DEFS:
        obs.append({"center": d["center_fn"](t), "radius": d["radius"], "velocity": d["velocity_fn"](t)})
    return obs

def nominal_ctrl(x):
    return np.clip(kp * (goal - x), -u_max, u_max)

def get_min_h(x, obs_list):
    min_h = float('inf')
    for o in obs_list:
        min_h = min(min_h, np.linalg.norm(x - o["center"]) - (o["radius"] + safety_margin))
    return min_h

# =====================================================================
# 预构建参数化 QP（避免每步重建 cvxpy 问题）
# =====================================================================
n_obs = len(STATIC_OBS) + len(DYN_OBS_DEFS)

def build_cbf_problem():
    """DPP-compliant CBF-QP.
    rhs_p[i] = alpha_cbf*h(x) - gh[i]@vel[i]  在外部预计算为标量，避免 param*param 违规。
    约束: gh[i] @ u_var + rhs_p[i] + slk[i] >= 0
    """
    u_var = cp.Variable(2)
    slk = cp.Variable(n_obs, nonneg=True)
    u_des_p = cp.Parameter(2)
    gh_p = cp.Parameter((n_obs, 2))        # 梯度方向 (2D, DPP ok)
    rhs_p = cp.Parameter(n_obs)             # 预计算标量 rhs

    cons = [u_var >= -u_max, u_var <= u_max]
    for i in range(n_obs):
        cons.append(gh_p[i] @ u_var + rhs_p[i] + slk[i] >= 0)

    obj = cp.Minimize(cp.sum_squares(u_var - u_des_p) + slack_weight * cp.sum(slk))
    prob = cp.Problem(obj, cons)
    return prob, u_var, u_des_p, gh_p, rhs_p


def build_mpc_problem():
    """DPP-compliant MPC-DCBF QP.

    关键设计：只用 2D Parameter，且不出现 param*param 乘积。
    线性化 DCBF 约束:
        h_k(x_k)   ≈ c_k[i]   + N_k[i]   @ X[k]
        h_k+1(x_k+1) ≈ c_kp1[i] + N_kp1[i] @ X[k+1]
    其中:
        N_k[i]    = 单位法向量 (外部计算, 2D param [N_mpc*n_obs, 2])
        c_k[i]    = dist_k - n_k @ xref_k - rsafe  (外部计算, 1D param [N_mpc*n_obs])
    这样约束里只有 param @ var + param，完全 DPP。
    """
    M = N_mpc * n_obs   # 展平的约束总数

    X = cp.Variable((N_mpc + 1, 2))
    U = cp.Variable((N_mpc, 2))
    slk = cp.Variable((n_obs, N_mpc), nonneg=True)

    x0_p    = cp.Parameter(2)
    N_k_p   = cp.Parameter((M, 2))   # 法向量 k,   行 = k*n_obs+j
    N_kp1_p = cp.Parameter((M, 2))   # 法向量 k+1
    c_k_p   = cp.Parameter(M)        # h_k 的常数项
    c_kp1_p = cp.Parameter(M)        # h_k+1 的常数项

    cons = [X[0] == x0_p]
    cost = 0.0

    for k in range(N_mpc):
        cons.append(X[k + 1] == X[k] + dt * U[k])
        cons.append(U[k] >= -u_max)
        cons.append(U[k] <= u_max)
        cost += mpc_Q * cp.sum_squares(X[k + 1] - goal) + mpc_R * cp.sum_squares(U[k])

    for k in range(N_mpc):
        for j in range(n_obs):
            idx = k * n_obs + j
            hk_expr   = c_k_p[idx]   + N_k_p[idx]   @ X[k]
            hkp1_expr = c_kp1_p[idx] + N_kp1_p[idx] @ X[k + 1]
            cons.append(hkp1_expr >= (1 - gamma_dcbf) * hk_expr - slk[j, k])
            cons.append(hkp1_expr >= -slk[j, k])

    cost += mpc_slack_W * cp.sum_squares(slk)
    prob = cp.Problem(cp.Minimize(cost), cons)

    params = {"x0": x0_p, "N_k": N_k_p, "N_kp1": N_kp1_p,
              "c_k": c_k_p, "c_kp1": c_kp1_p}
    return prob, X, U, params

# =====================================================================
# Simulation Core
# =====================================================================
def simulate(mode="nominal"):
    x = start.copy()
    path = [x.copy()]
    h_hist = []
    fails = 0
    reached = False
    U_prev = np.zeros((N_mpc, 2))

    # 预构建参数化问题（只构建一次！）
    if mode == "cbf":
        cbf_prob, cbf_u, cbf_u_des, cbf_gh, cbf_rhs = build_cbf_problem()
    elif mode == "mpc":
        mpc_prob, mpc_X, mpc_U, mpc_p = build_mpc_problem()

    t_start = time.time()

    for step in range(max_steps):
        t = step * dt
        obs_current = get_obstacles_at(t)
        h_hist.append(get_min_h(x, obs_current))

        u_des = nominal_ctrl(x)
        u_cmd = u_des.copy()

        # ---------------------------------------------------------
        # 模式 1: 单步 CBF-QP (参数化, 不重建)
        # ---------------------------------------------------------
        if mode == "cbf":
            cbf_u_des.value = u_des
            gh_vals = np.zeros((n_obs, 2))
            rhs_vals = np.zeros(n_obs)
            for i, o in enumerate(obs_current):
                d_vec = x - o["center"]
                nd = np.linalg.norm(d_vec)
                hv = nd - (o["radius"] + safety_margin)
                gh = d_vec / nd if nd > 1e-9 else np.array([1.0, 0.0])
                gh_vals[i] = gh
                rhs_vals[i] = alpha_cbf * hv - gh @ o["velocity"]
            cbf_gh.value = gh_vals
            cbf_rhs.value = rhs_vals

            try:
                cbf_prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
                if cbf_prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and cbf_u.value is not None:
                    u_cmd = np.asarray(cbf_u.value).reshape(2)
                else:
                    fails += 1
            except:
                fails += 1

        # ---------------------------------------------------------
        # 模式 2: 多步 MPC-DCBF (参数化, 不重建)
        # ---------------------------------------------------------
        elif mode == "mpc":
            X_ref = np.zeros((N_mpc + 1, 2))
            X_ref[0] = x
            for k in range(N_mpc):
                X_ref[k + 1] = X_ref[k] + dt * U_prev[k]

            M = N_mpc * n_obs
            N_k_val = np.zeros((M, 2))
            N_kp1_val = np.zeros((M, 2))
            c_k_val = np.zeros(M)
            c_kp1_val = np.zeros(M)

            obs_cache = [get_obstacles_at(t + k * dt) for k in range(N_mpc + 1)]

            for k in range(N_mpc):
                obs_k = obs_cache[k]
                obs_kp1 = obs_cache[k + 1]
                for j in range(n_obs):
                    idx = k * n_obs + j
                    rsafe = obs_k[j]["radius"] + safety_margin

                    vec_k = X_ref[k] - obs_k[j]["center"]
                    dk = np.linalg.norm(vec_k)
                    n_k = vec_k / dk if dk > 1e-5 else np.array([1.0, 0.0])
                    N_k_val[idx] = n_k
                    c_k_val[idx] = dk - n_k @ X_ref[k] - rsafe

                    vec_kp1 = X_ref[k + 1] - obs_kp1[j]["center"]
                    dkp1 = np.linalg.norm(vec_kp1)
                    n_kp1 = vec_kp1 / dkp1 if dkp1 > 1e-5 else np.array([1.0, 0.0])
                    N_kp1_val[idx] = n_kp1
                    c_kp1_val[idx] = dkp1 - n_kp1 @ X_ref[k + 1] - rsafe

            mpc_p["x0"].value = x
            mpc_p["N_k"].value = N_k_val
            mpc_p["N_kp1"].value = N_kp1_val
            mpc_p["c_k"].value = c_k_val
            mpc_p["c_kp1"].value = c_kp1_val

            try:
                mpc_prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
                if mpc_prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and mpc_U.value is not None:
                    u_cmd = np.asarray(mpc_U.value[0]).reshape(2)
                    U_prev[:-1] = mpc_U.value[1:]
                    U_prev[-1] = mpc_U.value[-1]
                else:
                    fails += 1
            except:
                fails += 1

        x = x + dt * u_cmd
        path.append(x.copy())
        if np.linalg.norm(x - goal) < goal_tol:
            reached = True
            break

    elapsed = time.time() - t_start
    print(f"  [{mode}] 用时 {elapsed:.2f}s, 步数 {step+1}, 到达={reached}, 失败={fails}")
    return {"path": np.array(path), "h": np.array(h_hist), "reached": reached, "fails": fails}

# =====================================================================
# Run and Compare
# =====================================================================
print("Simulating Nominal...")
res_nom = simulate("nominal")
print("Simulating CBF-QP...")
res_cbf = simulate("cbf")
print("Simulating MPC-DCBF...")
res_mpc = simulate("mpc")

# =====================================================================
# Visualization (Static & Animation)
# =====================================================================
# (略微保留了你的原代码画图结构，新增了MPC曲线)
fig_s, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
th = np.linspace(0, 2 * np.pi, 200)

for o in STATIC_OBS:
    ax1.add_patch(Circle(o["center"], o["radius"], color="red", alpha=0.22))

ax1.plot(res_nom["path"][:, 0], res_nom["path"][:, 1], "--", color="gray", lw=2.2, label="Nominal")
ax1.plot(res_cbf["path"][:, 0], res_cbf["path"][:, 1], "-o", color="blue", ms=1.5, lw=1.6, label="CBF-QP")
ax1.plot(res_mpc["path"][:, 0], res_mpc["path"][:, 1], "-s", color="forestgreen", ms=1.5, lw=2.0, label="MPC-DCBF")

ax1.scatter(*start, color="green", s=100, zorder=5)
ax1.scatter(*goal, color="blue", marker="*", s=200, zorder=5)
ax1.set_aspect("equal"); ax1.legend(); ax1.grid(True); ax1.set_title("Trajectory Comparison")

t_n = np.arange(len(res_nom["h"])) * dt
t_c = np.arange(len(res_cbf["h"])) * dt
t_m = np.arange(len(res_mpc["h"])) * dt
ax2.plot(t_n, res_nom["h"], "--", color="gray", lw=2, label="Nominal h")
ax2.plot(t_c, res_cbf["h"], "-", color="blue", lw=2, label="CBF-QP h")
ax2.plot(t_m, res_mpc["h"], "-", color="forestgreen", lw=2, label="MPC-DCBF h")
ax2.axhline(0, color="red", ls=":", lw=1.2)
ax2.grid(True); ax2.legend(); ax2.set_title("Safety Distance (h(x) >= 0 is safe)")

plt.tight_layout()

# --- 动画 ---
n_max = max(len(res_nom["path"]), len(res_cbf["path"]), len(res_mpc["path"]))
skip = 4
frame_list = list(range(0, n_max, skip))
fig_a, ax_a = plt.subplots(figsize=(10, 5.5))

for o in STATIC_OBS:
    ax_a.add_patch(Circle(o["center"], o["radius"], color="red", alpha=0.22))

dyn_polys, dyn_lines = [], []
for d in DYN_OBS_DEFS:
    poly = Polygon(np.zeros((200, 2)), color=d["color"], alpha=0.35)
    ax_a.add_patch(poly)
    line, = ax_a.plot([], [], "-", color=d["color"], lw=1.5)
    dyn_polys.append(poly)
    dyn_lines.append(line)

nom_trail, = ax_a.plot([], [], "--", color="gray", alpha=0.8, label="Nominal")
cbf_trail, = ax_a.plot([], [], "-", color="blue", lw=1.5, label="CBF-QP")
mpc_trail, = ax_a.plot([], [], "-", color="forestgreen", lw=2.0, label="MPC-DCBF")

nom_dot, = ax_a.plot([], [], "o", color="dimgray", ms=11, zorder=10)
cbf_dot, = ax_a.plot([], [], "o", color="royalblue", ms=11, zorder=10)
mpc_dot, = ax_a.plot([], [], "X", color="darkgreen", ms=11, zorder=10)

ax_a.scatter(*start, color="green", s=100, zorder=5)
ax_a.scatter(*goal, color="blue", marker="*", s=200, zorder=5)
ax_a.set_xlim(-0.5, 5.8); ax_a.set_ylim(-1.8, 1.8); ax_a.set_aspect("equal"); ax_a.legend()

def anim_update(fi):
    idx = frame_list[fi]
    t = idx * dt

    for k, d in enumerate(DYN_OBS_DEFS):
        c, r = d["center_fn"](t), d["radius"]
        cx, cy = c[0] + r * np.cos(th), c[1] + r * np.sin(th)
        dyn_polys[k].set_xy(np.column_stack([cx, cy]))
        dyn_lines[k].set_data(cx, cy)

    ni = min(idx, len(res_nom["path"]) - 1)
    nom_trail.set_data(res_nom["path"][:ni+1, 0], res_nom["path"][:ni+1, 1])
    nom_dot.set_data([res_nom["path"][ni, 0]], [res_nom["path"][ni, 1]])

    ci = min(idx, len(res_cbf["path"]) - 1)
    cbf_trail.set_data(res_cbf["path"][:ci+1, 0], res_cbf["path"][:ci+1, 1])
    cbf_dot.set_data([res_cbf["path"][ci, 0]], [res_cbf["path"][ci, 1]])

    mi = min(idx, len(res_mpc["path"]) - 1)
    mpc_trail.set_data(res_mpc["path"][:mi+1, 0], res_mpc["path"][:mi+1, 1])
    mpc_dot.set_data([res_mpc["path"][mi, 0]], [res_mpc["path"][mi, 1]])

    return [nom_trail, cbf_trail, mpc_trail, nom_dot, cbf_dot, mpc_dot] + dyn_polys + dyn_lines

ani = animation.FuncAnimation(fig_a, anim_update, frames=len(frame_list), interval=50, blit=False)
mp4_path = Path(__file__).with_name("cbf_demo.mp4")
print(f"Saving animation to {mp4_path} ...")
try:
    writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    ani.save(mp4_path, writer=writer)
    print(f"Saved mp4: {mp4_path}")
except Exception as e:
    print(f"保存 mp4 失败: {e}")
plt.show()