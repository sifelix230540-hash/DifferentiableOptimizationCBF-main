import unittest

import numpy as np
import pybullet as p

from CBF_experiment.active.pybullet.welding_320_common import (
    ExperimentConfig,
    SimulationScene,
    build_cbf_contact_visualization_specs,
    build_surface_cloud_visualization_specs,
)


class CBFContactVisualizationTests(unittest.TestCase):
    def test_build_specs_generates_bidirectional_normals_for_each_contact(self):
        specs = build_cbf_contact_visualization_specs([
            {
                "link_name": "welding_gun_base",
                "obs_link_name": "l2",
                "h_val": -0.002,
                "point_on_link": [1.0, 2.0, 3.0],
                "point_on_obstacle": [1.0, 2.03, 3.0],
                "normal_on_link": [1.0, 0.0, 0.0],
                "normal_on_obstacle": [0.0, 1.0, 0.0],
            }
        ], normal_length=0.05)

        self.assertEqual(len(specs), 1)
        self.assertTrue(np.allclose(specs[0]["point_on_link"], [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(specs[0]["point_on_obstacle"], [1.0, 2.03, 3.0]))
        self.assertTrue(np.allclose(specs[0]["normal_on_link"], [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(specs[0]["normal_on_obstacle"], [0.0, 1.0, 0.0]))
        self.assertAlmostEqual(specs[0]["normal_length"], 0.05)
        self.assertEqual(specs[0]["label"], "welding_gun_base -> l2 | h=-2.0mm")

    def test_build_specs_skips_contacts_missing_required_points(self):
        specs = build_cbf_contact_visualization_specs([
            {
                "link_name": "ok",
                "h_val": 0.001,
                "point_on_link": [0.0, 0.0, 0.0],
                "point_on_obstacle": [0.0, 0.01, 0.0],
                "normal": [0.0, 1.0, 0.0],
            },
            {
                "link_name": "missing_obstacle_point",
                "h_val": 0.002,
                "point_on_link": [1.0, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
            },
        ])

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["label"], "ok | h=1.0mm")

    def test_build_surface_cloud_visualization_specs_skips_empty_clouds(self):
        specs = build_surface_cloud_visualization_specs(
            [
                {
                    "link_name": "robot_link",
                    "points": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                    "color": [1.0, 0.0, 0.0],
                },
                {
                    "link_name": "empty",
                    "points": [],
                    "color": [0.0, 1.0, 0.0],
                },
            ],
            point_size=6,
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["link_name"], "robot_link")
        self.assertEqual(specs[0]["point_size"], 6)
        self.assertTrue(np.allclose(specs[0]["points"], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))


class CollisionVisualFrameTests(unittest.TestCase):
    def test_link_collision_visual_uses_center_of_mass_frame(self):
        cid = p.connect(p.DIRECT)
        try:
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                basePosition=[0.0, 0.0, 0.0],
                linkMasses=[1.0],
                linkCollisionShapeIndices=[shape],
                linkVisualShapeIndices=[-1],
                linkPositions=[[0.2, 0.0, 0.0]],
                linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
                linkInertialFramePositions=[[0.3, 0.0, 0.0]],
                linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
                linkParentIndices=[0],
                linkJointTypes=[p.JOINT_FIXED],
                linkJointAxis=[[0.0, 0.0, 1.0]],
            )
            scene = SimulationScene(ExperimentConfig())
            try:
                pose_pos, pose_orn = scene._get_body_link_pose(body_id, 0)
                link_state = p.getLinkState(body_id, 0, computeForwardKinematics=True)
                self.assertTrue(np.allclose(pose_pos, link_state[0]))
                self.assertTrue(np.allclose(pose_orn, link_state[1]))
                self.assertFalse(np.allclose(link_state[0], link_state[4]))
            finally:
                if p.isConnected(scene.client_id):
                    p.disconnect(scene.client_id)
        finally:
            if p.isConnected(cid):
                p.disconnect(cid)


if __name__ == "__main__":
    unittest.main()
