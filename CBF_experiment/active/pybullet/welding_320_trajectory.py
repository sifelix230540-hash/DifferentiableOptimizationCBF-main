import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from CBF_experiment.active.pybullet.welding_320_common import _normalize


class LineSlerpTrajectory:
    """单段直线位置插值加四元数球面插值轨迹。"""

    def __init__(self, start_pos, start_quat, goal_pos, goal_quat, duration, dt):
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.start_quat = np.array(start_quat, dtype=float)
        self.goal_quat = np.array(goal_quat, dtype=float)
        self.duration = float(duration)
        self.dt = float(dt)
        self.linear_velocity = (self.goal_pos - self.start_pos) / max(self.duration, 1e-6)
        self.slerp = Slerp(
            np.array([0.0, self.duration]),
            Rotation.from_quat(np.vstack([self.start_quat, self.goal_quat])),
        )

    def sample(self, t):
        tau = float(np.clip(t, 0.0, self.duration))
        blend = tau / max(self.duration, 1e-6)
        pos = (1.0 - blend) * self.start_pos + blend * self.goal_pos
        quat = self.slerp([tau]).as_quat()[0]
        if tau >= self.duration:
            return pos, quat, np.zeros(3), np.zeros(3)
        next_t = min(self.duration, tau + self.dt)
        next_quat = self.slerp([next_t]).as_quat()[0]
        ang_vel = (
            Rotation.from_quat(next_quat) * Rotation.from_quat(quat).inv()
        ).as_rotvec() / max(next_t - tau, 1e-6)
        return pos, quat, self.linear_velocity.copy(), ang_vel

    def reference_points(self, n):
        return [self.sample(t)[0] for t in np.linspace(0.0, self.duration, n, endpoint=True)]


class PiecewiseLineSlerpTrajectory:
    """按时间拼接多段参考轨迹。"""

    def __init__(self, segments: list[LineSlerpTrajectory]):
        self.segments = segments
        self.durations = [seg.duration for seg in segments]
        self.cumulative = np.cumsum(self.durations)
        self.duration = float(self.cumulative[-1]) if len(self.cumulative) else 0.0
        self.dt = segments[0].dt if segments else 0.0

    def current_segment_index(self, t: float) -> int:
        if not self.segments:
            return 0
        tau = float(np.clip(t, 0.0, self.duration))
        return int(np.searchsorted(self.cumulative, tau, side="right"))

    def sample(self, t):
        if not self.segments:
            raise RuntimeError("空轨迹无法采样")
        tau = float(np.clip(t, 0.0, self.duration))
        segment_index = min(self.current_segment_index(tau), len(self.segments) - 1)
        prev_end = 0.0 if segment_index == 0 else self.cumulative[segment_index - 1]
        local_t = tau - prev_end
        return self.segments[segment_index].sample(local_t)

    def reference_points(self, n):
        return [self.sample(t)[0] for t in np.linspace(0.0, self.duration, n, endpoint=True)]


class JointWaypointTrajectory:
    """将 RRT 路点序列包装成可采样的末端参考轨迹。"""

    def __init__(self, waypoints_pos, waypoints_quat, duration, dt, planner_status="unknown"):
        self.waypoints_pos = [np.array(pnt, dtype=float) for pnt in waypoints_pos]
        self.waypoints_quat = [np.array(qt, dtype=float) for qt in waypoints_quat]
        if len(self.waypoints_pos) == 1:
            self.waypoints_pos.append(self.waypoints_pos[0].copy())
            self.waypoints_quat.append(self.waypoints_quat[0].copy())
        self.duration = float(duration)
        self.dt = float(dt)
        self.goal_pos = self.waypoints_pos[-1].copy()
        self.goal_quat = self.waypoints_quat[-1].copy()
        self.planner_status = planner_status

        seg_len = []
        for idx in range(len(self.waypoints_pos) - 1):
            seg_len.append(float(np.linalg.norm(self.waypoints_pos[idx + 1] - self.waypoints_pos[idx])))
        self._seg_len = np.array(seg_len, dtype=float)
        total_len = float(np.sum(self._seg_len))
        if total_len < 1e-9:
            self._seg_time = np.full(len(self._seg_len), self.duration / max(len(self._seg_len), 1), dtype=float)
        else:
            self._seg_time = self.duration * self._seg_len / total_len
        self._cum_time = np.cumsum(self._seg_time) if len(self._seg_time) > 0 else np.array([self.duration])

    def _sample_pose(self, tau):
        if len(self._seg_time) == 0:
            return self.waypoints_pos[0].copy(), self.waypoints_quat[0].copy(), 0, 1e-6

        tau = float(np.clip(tau, 0.0, self.duration))
        seg_idx = int(np.searchsorted(self._cum_time, tau, side="right"))
        seg_idx = min(seg_idx, len(self._seg_time) - 1)
        t0 = 0.0 if seg_idx == 0 else float(self._cum_time[seg_idx - 1])
        t1 = float(self._cum_time[seg_idx])
        dt_seg = max(t1 - t0, 1e-6)
        alpha = float(np.clip((tau - t0) / dt_seg, 0.0, 1.0))

        p0, p1 = self.waypoints_pos[seg_idx], self.waypoints_pos[seg_idx + 1]
        q0, q1 = self.waypoints_quat[seg_idx], self.waypoints_quat[seg_idx + 1]
        pos = (1.0 - alpha) * p0 + alpha * p1
        quat = Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([q0, q1])))([alpha]).as_quat()[0]
        return pos, quat, seg_idx, dt_seg

    def sample(self, t):
        tau = float(np.clip(t, 0.0, self.duration))
        pos, quat, seg_idx, dt_seg = self._sample_pose(tau)
        if tau >= self.duration:
            return pos, quat, np.zeros(3), np.zeros(3)

        p0 = self.waypoints_pos[seg_idx]
        p1 = self.waypoints_pos[seg_idx + 1]
        lin_vel = (p1 - p0) / dt_seg

        next_tau = min(self.duration, tau + self.dt)
        _, next_quat, _, _ = self._sample_pose(next_tau)
        ang_vel = (
            Rotation.from_quat(next_quat) * Rotation.from_quat(quat).inv()
        ).as_rotvec() / max(next_tau - tau, 1e-6)
        return pos, quat, lin_vel, ang_vel

    def reference_points(self, n):
        return [self.sample(t)[0] for t in np.linspace(0.0, self.duration, n, endpoint=True)]


class PathProgressTrajectory:
    """把分段轨迹重参数化为按弧长进度采样的连续路径。"""

    def __init__(self, base_trajectory: PiecewiseLineSlerpTrajectory):
        self.base_trajectory = base_trajectory
        self.segments = list(base_trajectory.segments)
        self.duration = float(base_trajectory.duration)
        self.dt = float(base_trajectory.dt)

        positions: list[np.ndarray] = []
        quats: list[np.ndarray] = []
        times: list[float] = []
        self.segment_end_progress: list[float] = []

        time_offset = 0.0
        for seg in self.segments:
            seg_positions, seg_quats, seg_times = self._segment_control_points(seg)
            if positions:
                seg_positions = seg_positions[1:]
                seg_quats = seg_quats[1:]
                seg_times = seg_times[1:]
            positions.extend(seg_positions)
            quats.extend(seg_quats)
            times.extend([time_offset + float(tau) for tau in seg_times])
            time_offset += float(seg.duration)

        self.positions = [np.array(pnt, dtype=float) for pnt in positions]
        self.quats = [np.array(qt, dtype=float) for qt in quats]
        self.times = np.array(times, dtype=float) if times else np.zeros(1)

        progress = [0.0]
        for idx in range(len(self.positions) - 1):
            ds = float(np.linalg.norm(self.positions[idx + 1] - self.positions[idx]))
            progress.append(progress[-1] + ds)
        self.progress = np.array(progress, dtype=float) if progress else np.zeros(1)
        self.progress_end = float(self.progress[-1]) if len(self.progress) else 0.0

        point_offset = 0
        for seg in self.segments:
            seg_points, _, _ = self._segment_control_points(seg)
            point_offset += len(seg_points) - 1
            point_offset = min(point_offset, len(self.progress) - 1)
            self.segment_end_progress.append(float(self.progress[point_offset]))

    def _segment_control_points(self, seg):
        if hasattr(seg, "waypoints_pos"):
            seg_positions = [np.array(pnt, dtype=float) for pnt in seg.waypoints_pos]
            seg_quats = [np.array(qt, dtype=float) for qt in seg.waypoints_quat]
            local_times = [0.0]
            if hasattr(seg, "_seg_time"):
                for dt_seg in seg._seg_time:
                    local_times.append(local_times[-1] + float(dt_seg))
            else:
                local_times = list(np.linspace(0.0, seg.duration, len(seg_positions), endpoint=True))
            return seg_positions, seg_quats, local_times

        seg_positions = [np.array(seg.start_pos, dtype=float), np.array(seg.goal_pos, dtype=float)]
        seg_quats = [np.array(seg.start_quat, dtype=float), np.array(seg.goal_quat, dtype=float)]
        return seg_positions, seg_quats, [0.0, float(seg.duration)]

    def _interval_index_from_progress(self, s: float) -> int:
        if len(self.progress) <= 1:
            return 0
        tau = float(np.clip(s, 0.0, self.progress_end))
        idx = int(np.searchsorted(self.progress, tau, side="right") - 1)
        return min(max(idx, 0), len(self.progress) - 2)

    def current_segment_index(self, s: float) -> int:
        if not self.segment_end_progress:
            return 0
        tau = float(np.clip(s, 0.0, self.progress_end))
        return int(np.searchsorted(np.array(self.segment_end_progress), tau, side="right"))

    def sample_by_progress(self, s: float):
        if len(self.positions) == 0:
            raise RuntimeError("空轨迹无法采样")
        if len(self.positions) == 1:
            return self.positions[0].copy(), self.quats[0].copy(), np.zeros(3), np.zeros(3)

        tau = float(np.clip(s, 0.0, self.progress_end))
        idx = self._interval_index_from_progress(tau)
        s0 = float(self.progress[idx])
        s1 = float(self.progress[idx + 1])
        ds = max(s1 - s0, 1e-9)
        alpha = float(np.clip((tau - s0) / ds, 0.0, 1.0))

        p0, p1 = self.positions[idx], self.positions[idx + 1]
        q0, q1 = self.quats[idx], self.quats[idx + 1]
        pos = (1.0 - alpha) * p0 + alpha * p1
        quat = Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([q0, q1])))([alpha]).as_quat()[0]

        t0 = float(self.times[idx])
        t1 = float(self.times[idx + 1]) if idx + 1 < len(self.times) else t0
        dt_seg = max(t1 - t0, 1e-6)
        lin_vel = (p1 - p0) / dt_seg

        if tau >= self.progress_end:
            return pos, quat, np.zeros(3), np.zeros(3)
        ang_vel = (
            Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()
        ).as_rotvec() / dt_seg
        return pos, quat, lin_vel, ang_vel

    def sample(self, t: float):
        if len(self.times) == 0:
            raise RuntimeError("空轨迹无法采样")
        tau = float(np.clip(t, 0.0, self.duration))
        progress = float(np.interp(tau, self.times, self.progress))
        return self.sample_by_progress(progress)

    def advance_progress(self, current_progress: float, projected_progress: float) -> float:
        return float(np.clip(max(current_progress, projected_progress), 0.0, self.progress_end))

    def project_progress(self, pos: np.ndarray, hint_progress: float | None = None, search_radius: float | None = None) -> float:
        query = np.array(pos, dtype=float)
        if len(self.positions) <= 1:
            return 0.0

        if hint_progress is None:
            search_lo = 0.0
            search_hi = self.progress_end
            hint = None
        else:
            radius = self.progress_end if search_radius is None else float(search_radius)
            hint = float(np.clip(hint_progress, 0.0, self.progress_end))
            search_lo = max(0.0, hint - radius)
            search_hi = min(self.progress_end, hint + radius)

        best_progress = 0.0
        best_dist_sq = float("inf")
        best_hint_dist = float("inf")
        for idx in range(len(self.positions) - 1):
            seg_start = float(self.progress[idx])
            seg_end = float(self.progress[idx + 1])
            if seg_end < search_lo or seg_start > search_hi:
                continue

            p0, p1 = self.positions[idx], self.positions[idx + 1]
            seg = p1 - p0
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-12:
                alpha = 0.0
                proj = p0
            else:
                alpha = float(np.clip(np.dot(query - p0, seg) / seg_len_sq, 0.0, 1.0))
                proj = p0 + alpha * seg
                if seg_end > seg_start:
                    alpha_lo = float(np.clip((search_lo - seg_start) / (seg_end - seg_start), 0.0, 1.0))
                    alpha_hi = float(np.clip((search_hi - seg_start) / (seg_end - seg_start), 0.0, 1.0))
                    alpha = float(np.clip(alpha, alpha_lo, alpha_hi))
                    proj = p0 + alpha * seg

            dist_sq = float(np.dot(query - proj, query - proj))
            cand_progress = float(self.progress[idx] + alpha * np.sqrt(seg_len_sq))
            cand_hint_dist = abs(cand_progress - hint) if hint is not None else 0.0
            if (
                dist_sq < best_dist_sq - 1e-12
                or (abs(dist_sq - best_dist_sq) <= 1e-12 and cand_hint_dist < best_hint_dist)
            ):
                best_dist_sq = dist_sq
                best_progress = cand_progress
                best_hint_dist = cand_hint_dist

        if best_dist_sq == float("inf"):
            return self.project_progress(query, hint_progress=None)
        return best_progress

    def compute_path_errors(self, pos: np.ndarray, progress_value: float) -> tuple[float, float]:
        ref_pos, _, _, _ = self.sample_by_progress(progress_value)
        idx = self._interval_index_from_progress(progress_value)
        if len(self.positions) <= 1:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = _normalize(self.positions[idx + 1] - self.positions[idx])
        delta = np.array(pos, dtype=float) - ref_pos
        lag_error = float(np.dot(delta, tangent))
        contour_vec = delta - lag_error * tangent
        contour_error = float(np.linalg.norm(contour_vec))
        return lag_error, contour_error

    def reference_points(self, n):
        return [self.sample_by_progress(s)[0] for s in np.linspace(0.0, self.progress_end, n, endpoint=True)]
