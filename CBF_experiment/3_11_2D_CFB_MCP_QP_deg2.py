import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp
from matplotlib.patches import Circle, Polygon
import time
from pathlib import Path

from lqp.qp_policy import LearnedQPPolicy

# =====================================================================
# Nominal vs CBF-QP vs MPC-DCBF Benchmark
# =====================================================================

# --- 基础参数 ---
dt = 0.02           # 增大步长: 0.01 -> 0.05 (减少5倍迭代)
max_steps = 1500      # 相应减少 (0.05*400=20s, 足够到达目标)
goal_tol = 0.15
safety_margin = 0.2

mass = 2.0
start = np.array([0.0, 0.0, 0.0, 0.0])  # [px, py, vx, vy]
goal = np.array([5.0, 0.0])

# --- 单步 CBF-QP 参数 ---
alpha_cbf = 4
u_max = 8          # 力输入上限
kp = 1.2
kd = 1.5
slack_weight = 800.0

# --- MPC-DCBF 专有参数 ---
N_mpc = 30           # 预测时域 (dt=0.05时, 往后看0.5s)
gamma_dcbf = 0.01    # 离散CBF的衰减率 (越小越保守, 需匹配dt)
mpc_Q = 2          # 目标跟踪权重
mpc_track_W = 0   # 参考轨迹跟踪权重
mpc_V = 0.2          # 速度衰减权重
mpc_R = 0.08          # 控制输入平滑权重
mpc_slack_W = 20000 # 避障松弛变量惩罚权重

# --- 停滞脱困参数 ---
stagnation_window = 20
stagnation_progress_eps = 0.03
stagnation_speed_eps = 0.10
stagnation_min_dist = 0.8
orth_bias_mag = 1.8
orth_bias_hold_steps = 45
orth_bias_decay = 0.92

# --- 静态路径规划参数（RRT*） ---
use_rrt_nominal = False   # 先弃用 RRT 名义路径，但保留相关代码
plan_bounds = (-0.5, 5.8, -1.8, 1.8)
rrt_step_len = 0.35
rrt_goal_sample_rate = 0.18
rrt_search_radius = 0.7
rrt_max_iter = 1200
rrt_clearance = 0.08
path_point_spacing = 0.10
path_lookahead = 10

# --- Learned QP nominal controller ---
nominal_backend = "pd"    # "pd" / "rrt" / "learned_qp"
learned_qp_checkpoint = Path(__file__).resolve().parents[1] / "lqp" / "checkpoints" / "learned_qp_nominal.pt"
_learned_qp_policy = None

# =====================================================================
# 环境与障碍物定义
# =====================================================================

STATIC_OBS = [
    {"center": np.array([2.5, 0.0]), "radius": 0.55, "velocity": np.zeros(2)},
]

def dyn1_center(t): return np.array([2.5, max(1.3 - 1.2 * t, -1.5)])
def dyn1_velocity(t): return np.array([0.0, -1.2]) if 1.3 - 1.2 * t > -1.5 else np.zeros(2)

def dyn2_center(t):
    if t < 1: return np.array([3.5, 1.3])
    y = 1.3 - 0.8 * (t - 1)
    return np.array([3.5, max(y, -1.5)])
def dyn2_velocity(t):
    if t < 1: return np.zeros(2)
    return np.array([0.0, -0.8]) if 1.3 - 0.8 * (t - 1) > -1.5 else np.zeros(2)

def dyn3_center(t):
    if t < 2: return np.array([4.2, -1.3])
    y = -1.3 + 0.9 * (t - 2)
    return np.array([4.2, min(y, 1.5)])
def dyn3_velocity(t):
    if t < 3: return np.zeros(2)
    return np.array([0.0, 0.9]) if -1.3 + 0.9 * (t - 3) < 1.5 else np.zeros(2)

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

_static_path_cache = None

def _in_bounds(p):
    xmin, xmax, ymin, ymax = plan_bounds
    return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

def _point_collision_free(p):
    for o in STATIC_OBS:
        if np.linalg.norm(p - o["center"]) <= o["radius"] + safety_margin + rrt_clearance:
            return False
    return _in_bounds(p)

def _segment_collision_free(p0, p1):
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    n_chk = max(2, int(np.ceil(seg_len / 0.05)))
    for i in range(n_chk + 1):
        pt = p0 + (i / n_chk) * seg
        if not _point_collision_free(pt):
            return False
    return True

def _extract_path(nodes, parents, goal_idx):
    path = []
    idx = goal_idx
    while idx >= 0:
        path.append(nodes[idx])
        idx = parents[idx]
    path.reverse()
    return np.array(path)

def _shortcut_path(path):
    if len(path) <= 2:
        return path
    short = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _segment_collision_free(path[i], path[j]):
                break
            j -= 1
        short.append(path[j])
        i = j
    return np.array(short)

def _densify_path(path, spacing):
    pts = [path[0]]
    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            continue
        n_seg = max(1, int(np.ceil(seg_len / spacing)))
        for j in range(1, n_seg + 1):
            pts.append(path[i] + (j / n_seg) * seg)
    return np.array(pts)

def _fallback_detour_path(start_pos, goal_pos):
    obs = STATIC_OBS[0]
    offset_y = obs["radius"] + safety_margin + 0.45
    waypoint = np.array([obs["center"][0], offset_y])
    return np.array([start_pos, waypoint, goal_pos])

def plan_static_rrt_star(start_pos, goal_pos):
    rng = np.random.default_rng(7)
    start_pos = np.asarray(start_pos, dtype=float)
    goal_pos = np.asarray(goal_pos, dtype=float)

    if _segment_collision_free(start_pos, goal_pos):
        return _densify_path(np.array([start_pos, goal_pos]), path_point_spacing)

    nodes = [start_pos]
    parents = [-1]
    costs = [0.0]
    goal_indices = []

    xmin, xmax, ymin, ymax = plan_bounds

    for _ in range(rrt_max_iter):
        if rng.random() < rrt_goal_sample_rate:
            sample = goal_pos
        else:
            sample = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])

        dists = np.array([np.linalg.norm(n - sample) for n in nodes])
        nearest_idx = int(np.argmin(dists))
        nearest = nodes[nearest_idx]
        direction = sample - nearest
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            continue

        new_pos = nearest + direction / dist * min(rrt_step_len, dist)
        if not _point_collision_free(new_pos):
            continue
        if not _segment_collision_free(nearest, new_pos):
            continue

        near_indices = [i for i, n in enumerate(nodes) if np.linalg.norm(n - new_pos) <= rrt_search_radius]
        best_parent = nearest_idx
        best_cost = costs[nearest_idx] + np.linalg.norm(new_pos - nearest)

        for i in near_indices:
            cand_cost = costs[i] + np.linalg.norm(nodes[i] - new_pos)
            if cand_cost < best_cost and _segment_collision_free(nodes[i], new_pos):
                best_parent = i
                best_cost = cand_cost

        nodes.append(new_pos)
        parents.append(best_parent)
        costs.append(best_cost)
        new_idx = len(nodes) - 1

        for i in near_indices:
            rewired_cost = best_cost + np.linalg.norm(nodes[i] - new_pos)
            if rewired_cost + 1e-9 < costs[i] and _segment_collision_free(new_pos, nodes[i]):
                parents[i] = new_idx
                costs[i] = rewired_cost

        if np.linalg.norm(new_pos - goal_pos) <= rrt_step_len and _segment_collision_free(new_pos, goal_pos):
            nodes.append(goal_pos)
            parents.append(new_idx)
            costs.append(best_cost + np.linalg.norm(goal_pos - new_pos))
            goal_indices.append(len(nodes) - 1)

    if goal_indices:
        best_goal_idx = min(goal_indices, key=lambda i: costs[i])
        raw_path = _extract_path(nodes, parents, best_goal_idx)
    else:
        raw_path = _fallback_detour_path(start_pos, goal_pos)

    return _densify_path(_shortcut_path(raw_path), path_point_spacing)

def get_static_nominal_path():
    global _static_path_cache
    if _static_path_cache is None:
        _static_path_cache = plan_static_rrt_star(start[:2], goal)
    return _static_path_cache

def get_path_target(pos):
    path = get_static_nominal_path()
    dists = np.linalg.norm(path - pos, axis=1)
    nearest_idx = int(np.argmin(dists))
    target_idx = min(nearest_idx + path_lookahead, len(path) - 1)
    return path[target_idx]


def get_learned_qp_policy():
    global _learned_qp_policy
    if _learned_qp_policy is None:
        if not learned_qp_checkpoint.exists():
            raise FileNotFoundError(f"learned QP checkpoint not found: {learned_qp_checkpoint}")
        _learned_qp_policy = LearnedQPPolicy.load_checkpoint(learned_qp_checkpoint)
    return _learned_qp_policy

def nominal_ctrl(x, u_bias=None):
    pos = x[:2]
    vel = x[2:]
    if nominal_backend == "learned_qp":
        u_nom = get_learned_qp_policy().act_numpy(x, goal)
    else:
        use_rrt_target = nominal_backend == "rrt" or use_rrt_nominal
        target = get_path_target(pos) if use_rrt_target else goal
        u_nom = kp * (target - pos) - kd * vel
    if u_bias is not None:
        u_nom = u_nom + u_bias
    return np.clip(u_nom, -u_max, u_max)

def step_dynamics(x, u):
    pos = x[:2]
    vel = x[2:]
    acc = u / mass
    x_next = np.empty_like(x)
    x_next[:2] = pos + dt * vel + 0.5 * dt**2 * acc
    x_next[2:] = vel + dt * acc
    return x_next

def build_track_reference(x0):
    """沿当前名义控制滚出参考轨迹，供 MPC 跟踪。"""
    X_track = np.zeros((N_mpc + 1, 4))
    X_track[0] = x0
    for k in range(N_mpc):
        u_ref = nominal_ctrl(X_track[k])
        X_track[k + 1] = step_dynamics(X_track[k], u_ref)
    return X_track

def choose_orthogonal_bias_sign(x, default_sign=1.0):
    pos = x[:2]
    goal_dir = goal - pos
    if use_rrt_nominal:
        ref_dir = get_path_target(pos) - pos
        goal_norm = np.linalg.norm(goal_dir)
        ref_norm = np.linalg.norm(ref_dir)
        if goal_norm > 1e-9 and ref_norm > 1e-9:
            cross = goal_dir[0] * ref_dir[1] - goal_dir[1] * ref_dir[0]
            if abs(cross) > 1e-6:
                return np.sign(cross)
    if abs(pos[1]) > 1e-3:
        return np.sign(pos[1])
    return 1.0 if default_sign >= 0 else -1.0

def build_orthogonal_bias(pos, sign):
    goal_dir = goal - pos
    goal_norm = np.linalg.norm(goal_dir)
    if goal_norm < 1e-9:
        return np.zeros(2)
    ortho = sign * np.array([-goal_dir[1], goal_dir[0]]) / goal_norm
    return orth_bias_mag * ortho

def is_stagnating(dist_hist, speed_hist):
    if len(dist_hist) < stagnation_window:
        return False
    progress = dist_hist[-stagnation_window] - dist_hist[-1]
    avg_speed = np.mean(speed_hist[-stagnation_window:])
    return (
        dist_hist[-1] > stagnation_min_dist
        and progress < stagnation_progress_eps
        and avg_speed < stagnation_speed_eps
    )

def get_min_h(x, obs_list):
    pos = x[:2]
    min_h = float('inf')
    for o in obs_list:
        min_h = min(min_h, np.linalg.norm(pos - o["center"]) - (o["radius"] + safety_margin))
    return min_h

# =====================================================================
# 预构建参数化 QP（避免每步重建 cvxpy 问题）
# =====================================================================
n_obs = len(STATIC_OBS) + len(DYN_OBS_DEFS)

def build_cbf_problem():
    """DPP-compliant CBF-QP.
    对二阶质量块采用近似 HOCBF:
        gh @ (u / mass) + 2 * alpha_cbf * h_dot + alpha_cbf^2 * h + slack >= 0
    外部预计算 rhs_p[i] = mass * (2 * alpha_cbf * h_dot + alpha_cbf^2 * h)。
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

    A = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.5 * dt**2 / mass, 0.0],
        [0.0, 0.5 * dt**2 / mass],
        [dt / mass, 0.0],
        [0.0, dt / mass],
    ])

    X = cp.Variable((N_mpc + 1, 4))
    U = cp.Variable((N_mpc, 2))
    slk = cp.Variable((n_obs, N_mpc), nonneg=True)

    x0_p    = cp.Parameter(4)
    X_tr_p  = cp.Parameter((N_mpc + 1, 4))
    N_k_p   = cp.Parameter((M, 2))   # 法向量 k,   行 = k*n_obs+j
    N_kp1_p = cp.Parameter((M, 2))   # 法向量 k+1
    c_k_p   = cp.Parameter(M)        # h_k 的常数项
    c_kp1_p = cp.Parameter(M)        # h_k+1 的常数项

    cons = [X[0] == x0_p]
    cost = 0.0

    for k in range(N_mpc):
        cons.append(X[k + 1] == A @ X[k] + B @ U[k])
        cons.append(U[k] >= -u_max)
        cons.append(U[k] <= u_max)
        cost += (
            mpc_Q * cp.sum_squares(X[k + 1, :2] - goal)
            + mpc_track_W * cp.sum_squares(X[k + 1] - X_tr_p[k + 1])
            + mpc_V * cp.sum_squares(X[k + 1, 2:])
            + mpc_R * cp.sum_squares(U[k])
        )

    for k in range(N_mpc):
        for j in range(n_obs):
            idx = k * n_obs + j
            hk_expr   = c_k_p[idx]   + N_k_p[idx]   @ X[k, :2]
            hkp1_expr = c_kp1_p[idx] + N_kp1_p[idx] @ X[k + 1, :2]
            cons.append(hkp1_expr >= (1 - gamma_dcbf) * hk_expr - slk[j, k])
            cons.append(hkp1_expr >= 0)

    cost += mpc_slack_W * cp.sum_squares(slk)
    prob = cp.Problem(cp.Minimize(cost), cons)

    params = {"x0": x0_p, "X_tr": X_tr_p, "N_k": N_k_p, "N_kp1": N_kp1_p,
              "c_k": c_k_p, "c_kp1": c_kp1_p}
    return prob, X, U, params

# =====================================================================
# Simulation Core
# =====================================================================
def simulate(mode="nominal"):
    x = start.copy()
    path = [x[:2].copy()]
    h_hist = []
    fails = 0
    reached = False
    U_prev = np.zeros((N_mpc, 2))
    dist_hist = []
    speed_hist = []
    bias_sign = 1.0
    bias_hold = 0

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

        dist_hist.append(np.linalg.norm(x[:2] - goal))
        speed_hist.append(np.linalg.norm(x[2:]))
        if is_stagnating(dist_hist, speed_hist) and bias_hold == 0:
            next_sign = choose_orthogonal_bias_sign(x, bias_sign)
            if abs(x[1]) < 0.03 and next_sign == bias_sign and len(dist_hist) > 2 * stagnation_window:
                next_sign = -bias_sign
            bias_sign = next_sign
            bias_hold = orth_bias_hold_steps

        u_orth = np.zeros(2)
        if bias_hold > 0:
            u_orth = build_orthogonal_bias(x[:2], bias_sign)
            bias_hold -= 1

        u_des = nominal_ctrl(x, u_bias=u_orth)
        u_cmd = u_des.copy()

        # ---------------------------------------------------------
        # 模式 1: 单步 CBF-QP (参数化, 不重建)
        # ---------------------------------------------------------
        if mode == "cbf":
            cbf_u_des.value = u_des
            gh_vals = np.zeros((n_obs, 2))
            rhs_vals = np.zeros(n_obs)
            pos = x[:2]
            vel = x[2:]
            for i, o in enumerate(obs_current):
                d_vec = pos - o["center"]
                nd = np.linalg.norm(d_vec)
                h = nd - (o["radius"] + safety_margin)
                gh = d_vec / nd if nd > 1e-9 else np.array([1.0, 0.0])
                h_dot = gh @ (vel - o["velocity"])
                gh_vals[i] = gh
                rhs_vals[i] = mass * (2 * alpha_cbf * h_dot + alpha_cbf**2 * h)
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
            X_ref = np.zeros((N_mpc + 1, 4))
            X_ref[0] = x
            for k in range(N_mpc):
                X_ref[k + 1] = step_dynamics(X_ref[k], U_prev[k])
            X_track = build_track_reference(x)

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

                    vec_k = X_ref[k, :2] - obs_k[j]["center"]
                    dk = np.linalg.norm(vec_k)
                    if dk < rsafe + 0.01:
                        n_k = vec_k / dk if dk > 1e-9 else np.array([1.0, 0.0])
                        ref_k = obs_k[j]["center"] + n_k * (rsafe + 0.02)
                        dk = rsafe + 0.02
                    else:
                        n_k = vec_k / dk
                        ref_k = X_ref[k, :2]
                    N_k_val[idx] = n_k
                    c_k_val[idx] = dk - n_k @ ref_k - rsafe

                    vec_kp1 = X_ref[k + 1, :2] - obs_kp1[j]["center"]
                    dkp1 = np.linalg.norm(vec_kp1)
                    if dkp1 < rsafe + 0.01:
                        n_kp1 = vec_kp1 / dkp1 if dkp1 > 1e-9 else np.array([1.0, 0.0])
                        ref_kp1 = obs_kp1[j]["center"] + n_kp1 * (rsafe + 0.02)
                        dkp1 = rsafe + 0.02
                    else:
                        n_kp1 = vec_kp1 / dkp1
                        ref_kp1 = X_ref[k + 1, :2]
                    N_kp1_val[idx] = n_kp1
                    c_kp1_val[idx] = dkp1 - n_kp1 @ ref_kp1 - rsafe

            mpc_p["x0"].value = x
            mpc_p["X_tr"].value = X_track
            mpc_p["N_k"].value = N_k_val
            mpc_p["N_kp1"].value = N_kp1_val
            mpc_p["c_k"].value = c_k_val
            mpc_p["c_kp1"].value = c_kp1_val

            try:
                mpc_prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                               eps_abs=1e-6, eps_rel=1e-6, max_iter=8000, polish=True)
                if mpc_prob.status == cp.OPTIMAL and mpc_U.value is not None:
                    u_cmd = np.asarray(mpc_U.value[0]).reshape(2)
                    U_prev[:-1] = mpc_U.value[1:]
                    U_prev[-1] = mpc_U.value[-1]
                else:
                    u_cmd = np.clip(-kd * x[2:], -u_max, u_max)
                    U_prev[:] = 0
                    fails += 1
            except:
                u_cmd = np.clip(-kd * x[2:], -u_max, u_max)
                U_prev[:] = 0
                fails += 1

        x = step_dynamics(x, u_cmd)
        path.append(x[:2].copy())
        if np.linalg.norm(x[:2] - goal) < goal_tol:
            reached = True
            break

    elapsed = time.time() - t_start
    print(f"  [{mode}] 用时 {elapsed:.2f}s, 步数 {step+1}, 到达={reached}, 失败={fails}")
    return {"path": np.array(path), "h": np.array(h_hist), "reached": reached, "fails": fails}

# =====================================================================
# Run and Compare
# =====================================================================
print(f"Nominal backend: {nominal_backend}")
if nominal_backend == "learned_qp":
    print(f"Learned QP checkpoint: {learned_qp_checkpoint}")
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

ax1.scatter(*start[:2], color="green", s=100, zorder=5)
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

ax_a.scatter(*start[:2], color="green", s=100, zorder=5)
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