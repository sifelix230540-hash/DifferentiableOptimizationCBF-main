import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import CBF_Net.train_3d_link_neural_sdf as neural_sdf


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))
        self.config = {
            "hidden": 8,
            "n_layers": 1,
            "n_frequencies": 6,
            "use_fourier": False,
            "quadratic": True,
            "activation": "softplus",
            "omega_0": 30.0,
            "init_type": "mfgi",
            "sphere_init_params": [1.6, 1.0],
        }

    def forward(self, x):
        return torch.zeros(x.shape[0], device=x.device)


class Train3DLinkNeuralSDFTests(unittest.TestCase):
    def test_quadratic_network_uses_quadratic_output_layer(self):
        model = neural_sdf.NeuralSDF3D(hidden=8, n_layers=2, quadratic=True)

        layer_modules = model._get_layer_modules()

        self.assertIsInstance(layer_modules[-1], neural_sdf.QuadraticLayer)

    def test_main_uses_unsupervised_pipeline_by_default(self):
        dataset = neural_sdf.SDFDataset(
            surface_points=np.zeros((8, 3), dtype=np.float32),
            surface_normals=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (8, 1)),
            aabb_min=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            aabb_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            center=np.zeros(3, dtype=np.float32),
            scale=1.0,
        )
        result = neural_sdf.TrainResult(
            model=_DummyModel(),
            loss_history=[{
                "epoch": 0,
                "loss_total": 1.0,
                "loss_sdf": 0.5,
                "loss_eikonal": 0.2,
                "loss_dd": 0.1,
                "loss_normal": 0.1,
                "loss_inter": 0.1,
            }],
            final_loss=1.0,
            elapsed_seconds=0.01,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            mesh_path = Path(tmpdir) / "dummy.stl"
            mesh_path.write_text("solid dummy\nendsolid dummy\n", encoding="utf-8")
            link_info = neural_sdf.LinkMeshInfo("dummy_link", mesh_path)
            with mock.patch.object(neural_sdf, "parse_urdf_links", return_value=[link_info]), \
                 mock.patch.object(neural_sdf, "generate_pointcloud_data", return_value=(dataset, {"is_watertight": True, "n_train_points": 8, "n_vertices": 8, "n_faces": 12})) as pointcloud_mock, \
                 mock.patch.object(neural_sdf, "train_link_sdf_unsupervised", return_value=result) as unsup_mock, \
                 mock.patch.object(neural_sdf, "_save_loss_curve"), \
                 mock.patch.object(neural_sdf, "_save_sdf_slices"), \
                 mock.patch.object(neural_sdf, "_save_surface_eval"), \
                 mock.patch.object(neural_sdf, "_save_isosurface"):
                neural_sdf.main([
                    "--urdf", "dummy.urdf",
                    "--output-dir", str(output_dir),
                    "--n-iterations", "1",
                    "--n-points", "4",
                    "--device", "cpu",
                    "--link-names", "dummy_link",
                ])

        pointcloud_mock.assert_called_once()
        unsup_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
