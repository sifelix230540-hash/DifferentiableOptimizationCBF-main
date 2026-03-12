import argparse
import copy
import json
import time
from sys import platform

import numpy as np
import proxsuite

from DifferentiableOptimizationCBF.envs.unicycle_env import UnicycleEnv
from DifferentiableOptimizationCBF.unicycle_plot_utils import plot_unicycle


def debug_log(hypothesis_id, location, message, data, run_id="pre-fix"):
    # region agent log
    with open(
        r"c:\Users\12049\OneDrive\Desktop\科研相关\博一春季\免示教焊接轨迹规划\相关资料\CBF_grad_optim_on_trajPlanning\DifferentiableOptimizationCBF-main\debug-1450bb.log",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "1450bb",
                    "runId": run_id,
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    # endregion


def get_Q_mat(q):
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -q[3]],
            [0.0, 0.0, q[2]],
            [0.0, 0.0, -q[1]],
            [0.0, 0.0, q[0]],
        ]
    )


def probe_proxsuite_api():
    H = np.eye(2, dtype=np.float64)
    g_vec = np.array([1.0, 2.0], dtype=np.float64)
    g_col = np.array([[1.0], [2.0]], dtype=np.float64)
    C = np.eye(2, dtype=np.float64)
    l_vec = np.array([0.0, 0.0], dtype=np.float64)
    l_col = np.array([[0.0], [0.0]], dtype=np.float64)
    u_vec = np.array([1.0e20, 1.0e20], dtype=np.float64)

    tests = [
        (
            "H6",
            "kwargs_vec_l_only",
            lambda qp: qp.init(H=H, g=g_vec, C=C, l=l_vec),
        ),
        (
            "H6,H9",
            "positional_vec_l_only",
            lambda qp: qp.init(H, g_vec, None, None, C, l_vec),
        ),
        (
            "H7",
            "kwargs_col_l_only",
            lambda qp: qp.init(H=H, g=g_col, C=C, l=l_col),
        ),
        (
            "H8",
            "kwargs_vec_with_u",
            lambda qp: qp.init(H=H, g=g_vec, C=C, l=l_vec, u=u_vec),
        ),
    ]

    for hypothesis_id, name, test in tests:
        qp = proxsuite.proxqp.dense.QP(2, 0, 2)
        try:
            test(qp)
            debug_log(
                hypothesis_id,
                "unicycle_exp.py:80",
                "probe_result",
                {"test": name, "passed": True},
            )
        except Exception as exc:
            debug_log(
                hypothesis_id,
                "unicycle_exp.py:80",
                "probe_result",
                {
                    "test": name,
                    "passed": False,
                    "exception_type": type(exc).__name__,
                    "exception_text": str(exc),
                },
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_plot", action="store_true")
    args = parser.parse_args()

    env = UnicycleEnv()
    env.reset(set_init_state=[-1.0, -3.0, np.pi / 4])
    unicycle_env_setup()
    initialized = False

    # define CBFQP solver
    qp = proxsuite.proxqp.dense.QP(2, 0, 2)
    # region agent log
    debug_log(
        "H5",
        "unicycle_exp.py:69",
        "qp_backend",
        {
            "qp_type": str(type(qp)),
            "proxsuite_module": getattr(proxsuite, "__file__", "unknown"),
            "proxsuite_version": getattr(proxsuite, "__version__", "unknown"),
        },
    )
    # endregion
    # region agent log
    probe_proxsuite_api()
    # endregion

    # task parameters
    target_x = 5.0
    target_y = 3.0
    kv = 0.5
    kω = 2.0
    β = 1.05
    γ = 1.0

    # store data
    history = []
    comp_times = []

    for i in range(5000):
        # compute performance controller
        v = kv * np.sqrt(
            (target_x - env.state[0]) ** 2 + (target_y - env.state[1]) ** 2
        )

        target_θ = np.arctan2(target_y - env.state[1], target_x - env.state[0])

        ω = kω * (target_θ - env.state[2])
        control = np.array([v, ω])

        # get CBF
        tic = time.time()
        αs, Js = get_cbf_unicycle_env(env.robot_r, env.robot_q)
        if i == 0:
            # region agent log
            debug_log(
                "H1",
                "unicycle_exp.py:97",
                "cbf_raw_outputs",
                {
                    "alpha_types": [type(αs[0]).__name__, type(αs[1]).__name__],
                    "js0_array_dtype": str(np.asarray(Js[0]).dtype),
                    "js1_array_dtype": str(np.asarray(Js[1]).dtype),
                    "js0_shape": list(np.asarray(Js[0]).shape),
                    "js1_shape": list(np.asarray(Js[1]).shape),
                },
            )
            # endregion

        if i >= 10:
            # account for JIT run
            comp_times.append(time.time() - tic)

        Q_mat = get_Q_mat(env.robot_q)
        QF_mat = Q_mat @ env.F_mat

        J1 = np.array(Js[0])[-1, 7:][[0, 1, 3, 4, 5, 6]][np.newaxis, :]
        J2 = np.array(Js[1])[-1, 7:][[0, 1, 3, 4, 5, 6]][np.newaxis, :]

        C1 = J1 @ QF_mat
        C2 = J2 @ QF_mat

        C = np.vstack((C1, C2))
        lb = -γ * np.array([[αs[0] - β], [αs[1] - β]])

        # define CBFQP
        H = np.eye(2)
        g = -control[:, np.newaxis]
        if i == 0:
            # region agent log
            debug_log(
                "H2,H7,H8,H9",
                "unicycle_exp.py:125",
                "qp_inputs_pre_init",
                {
                    "H": {
                        "dtype": str(H.dtype),
                        "shape": list(H.shape),
                        "f_contig": bool(H.flags.f_contiguous),
                        "c_contig": bool(H.flags.c_contiguous),
                        "finite": bool(np.isfinite(H).all()),
                    },
                    "g": {
                        "dtype": str(g.dtype),
                        "shape": list(g.shape),
                        "ndim": int(g.ndim),
                        "finite": bool(np.isfinite(g).all()),
                    },
                    "C": {
                        "dtype": str(C.dtype),
                        "shape": list(C.shape),
                        "f_contig": bool(C.flags.f_contiguous),
                        "c_contig": bool(C.flags.c_contiguous),
                        "finite": bool(np.isfinite(C).all()),
                    },
                    "lb": {
                        "dtype": str(lb.dtype),
                        "shape": list(lb.shape),
                        "ndim": int(lb.ndim),
                        "finite": bool(np.isfinite(lb).all()),
                    },
                },
            )
            # endregion

        # solve CBFQP
        try:
            if initialized:
                qp.update(H=H, g=g, C=C, l=lb)
            else:
                qp.init(H=H, g=g, C=C, l=lb)
                initialized = True
        except Exception as exc:
            # region agent log
            debug_log(
                "H2,H5",
                "unicycle_exp.py:160",
                "qp_call_failed",
                {
                    "initialized": initialized,
                    "exception_type": type(exc).__name__,
                    "exception_text": str(exc),
                },
            )
            # endregion
            raise

        qp.solve()

        # Get safe action
        safe_control = qp.results.x
        safe_control = np.clip(safe_control, -20.0, 20.0)

        # apply safe action
        env.step(safe_control)

        # store data
        history.append(copy.deepcopy(env.state))

    print("Average computation time: ", np.mean(np.array(comp_times)))

    if args.show_plot:
        plot_unicycle(history)


if __name__ == "__main__":
    if platform == "darwin":
        from julia.api import Julia

        jl = Julia(compiled_modules=False)

    import julia

    j = julia.Julia()
    unicycle_env_setup = j.include("dc_utils/unicycle_env_setup.jl")
    get_cbf_unicycle_env = j.include("dc_utils/get_cbf_unicycle_env.jl")

    main()
