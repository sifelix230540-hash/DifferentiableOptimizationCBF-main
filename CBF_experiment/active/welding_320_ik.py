from dataclasses import dataclass

import numpy as np

from CBF_experiment.active.welding_320_common import ExperimentConfig


@dataclass
class IKCandidate:
    q: np.ndarray
    position_error: float
    orientation_error: float
    distance_to_reference: float
    is_collision_free: bool
    seed_tag: str


class MultiSeedIKSolver:
    def __init__(self, robot, config: ExperimentConfig):
        self.robot = robot
        self.config = config
        self._rng = np.random.default_rng(config.planner_seed + 101)

    def _build_seed_pool(self, reference_q, extra_seed_qs=None) -> list[tuple[str, np.ndarray]]:
        reference_q = np.zeros(self.robot.dof) if reference_q is None else np.array(reference_q, dtype=float)
        current_q, _ = self.robot.get_joint_state()
        lower, upper = self.robot.get_active_joint_limits()
        seeds: list[tuple[str, np.ndarray]] = [
            ("current", np.array(current_q, dtype=float)),
            ("reference", reference_q.copy()),
            ("midpoint", 0.5 * (lower + upper)),
            ("zeros", np.zeros_like(reference_q)),
        ]
        if extra_seed_qs is not None:
            for idx, q_seed in enumerate(extra_seed_qs):
                seeds.append((f"extra_{idx}", np.array(q_seed, dtype=float)))
        for idx in range(self.config.ik_random_seeds):
            noise = self._rng.normal(scale=self.config.ik_jitter_scale, size=self.robot.dof)
            q_seed = np.clip(reference_q + noise, lower, upper)
            seeds.append((f"jitter_{idx}", q_seed))
        return seeds

    def solve(
        self,
        target_pos,
        target_quat,
        reference_q=None,
        obstacle_body_ids=None,
        extra_seed_qs=None,
    ) -> list[IKCandidate]:
        reference_q = np.zeros(self.robot.dof) if reference_q is None else np.array(reference_q, dtype=float)
        candidates: list[IKCandidate] = []
        dedup: set[tuple[float, ...]] = set()
        for seed_tag, seed_q in self._build_seed_pool(reference_q, extra_seed_qs=extra_seed_qs):
            q_candidate = np.array(self.robot.calculate_ik(target_pos, target_quat, rest_poses=seed_q), dtype=float)
            key = tuple(np.round(q_candidate, 6).tolist())
            if key in dedup:
                continue
            dedup.add(key)
            errors = self.robot.evaluate_pose_candidate(q_candidate, target_pos, target_quat)
            if errors["position_error"] > self.config.ik_position_tolerance:
                continue
            is_free = self.robot.is_state_collision_free(
                q_candidate,
                obstacle_body_ids=[] if obstacle_body_ids is None else list(obstacle_body_ids),
                safety_margin=self.config.safety_margin,
            )
            candidates.append(IKCandidate(
                q=q_candidate,
                position_error=float(errors["position_error"]),
                orientation_error=float(errors["orientation_error"]),
                distance_to_reference=float(np.linalg.norm(q_candidate - reference_q)),
                is_collision_free=bool(is_free),
                seed_tag=seed_tag,
            ))

        candidates.sort(
            key=lambda cand: (
                not cand.is_collision_free,
                cand.position_error,
                cand.orientation_error,
                cand.distance_to_reference,
            )
        )
        return candidates[: self.config.ik_max_candidates]
