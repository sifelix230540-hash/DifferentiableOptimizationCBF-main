import unittest
from unittest import mock

from CBF_experiment.active.pybullet import welding_320_common as common


class SimulationSceneModeTests(unittest.TestCase):
    def test_uses_direct_connection_when_config_requests_direct(self):
        cfg = common.ExperimentConfig()
        cfg.pybullet_connection_mode = "DIRECT"

        with mock.patch.object(common.p, "connect", return_value=7) as mock_connect, \
             mock.patch.object(common.p, "setAdditionalSearchPath"), \
             mock.patch.object(common.p, "setGravity"), \
             mock.patch.object(common.p, "setTimeStep"), \
             mock.patch.object(common.p, "configureDebugVisualizer"), \
             mock.patch.object(common.p, "resetDebugVisualizerCamera"), \
             mock.patch.object(common.SimulationScene, "_build_environment", return_value=0.0), \
             mock.patch.object(common.SimulationScene, "_draw_axes"):
            scene = common.SimulationScene(cfg)

        mock_connect.assert_called_once_with(common.p.DIRECT)
        self.assertFalse(scene.gui_enabled)


if __name__ == "__main__":
    unittest.main()
