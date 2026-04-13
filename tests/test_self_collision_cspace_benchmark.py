import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "self_collision_cspace_benchmark.py"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SelfCollisionCSpaceBenchmarkTests(unittest.TestCase):
    def test_run_benchmark_writes_expected_outputs(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_benchmark_outputs")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)

            original_hull = module.run_hull_builder
            original_samples = module.generate_evaluation_samples
            original_eval = module.run_hull_evaluation
            try:
                def _fake_hull_builder(params):
                    hull_json = pathlib.Path(params.HULL_OUTPUT_JSON)
                    hull_png = pathlib.Path(params.HULL_OUTPUT_PNG)
                    hull_json.parent.mkdir(parents=True, exist_ok=True)
                    hull_png.parent.mkdir(parents=True, exist_ok=True)
                    hull_json.write_text(json.dumps({
                        "dimension": 2,
                        "clusters": [{
                            "cluster_id": 0,
                            "equations_joint": [
                                [1.0, 0.0, -1.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 1.0, -1.0],
                                [0.0, -1.0, 0.0],
                            ],
                            "equations_normalized": [
                                [1.0, 0.0, -1.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 1.0, -1.0],
                                [0.0, -1.0, 0.0],
                            ],
                        }],
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    hull_png.write_bytes(b"png")
                    return {"clusters": [{"cluster_id": 0}]}

                def _fake_generate_samples(params):
                    sample_json = pathlib.Path(params.SAMPLE_OUTPUT_JSON)
                    sample_json.parent.mkdir(parents=True, exist_ok=True)
                    sample_json.write_text(json.dumps({
                        "collision_samples": [[0.2, 0.2], [1.2, 1.2]],
                        "free_samples": [[1.5, 1.5], [0.5, 0.5]],
                        "collision_distances": [-0.01, -0.03],
                        "free_distances": [0.02, 0.10],
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    return {"num_collision_samples": 2, "num_free_samples": 2}

                def _fake_eval(params):
                    eval_json = pathlib.Path(params.EVAL_OUTPUT_JSON)
                    eval_png = pathlib.Path(params.EVAL_SUMMARY_PNG)
                    eval_json.parent.mkdir(parents=True, exist_ok=True)
                    eval_png.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "coverage": 0.5,
                        "miss_rate": 0.5,
                        "false_positive_rate": 0.5,
                        "boundary_collision_coverage": 1.0,
                        "boundary_free_false_positive_rate": 0.0,
                        "num_collision_samples": 2,
                        "num_free_samples": 2,
                    }
                    eval_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    eval_png.write_bytes(b"png")
                    return payload

                module.run_hull_builder = _fake_hull_builder
                module.generate_evaluation_samples = _fake_generate_samples
                module.run_hull_evaluation = _fake_eval

                class _Params:
                    WORK_DIR = tmpdir_path
                    HULL_OUTPUT_JSON = str(tmpdir_path / "self_collision_cspace_hulls.json")
                    HULL_OUTPUT_PNG = str(tmpdir_path / "self_collision_cspace_hulls.png")
                    SAMPLE_OUTPUT_JSON = str(tmpdir_path / "self_collision_eval_samples.json")
                    EVAL_OUTPUT_JSON = str(tmpdir_path / "self_collision_eval_report.json")
                    EVAL_SUMMARY_PNG = str(tmpdir_path / "self_collision_eval_summary.png")

                summary = module.run_benchmark(_Params)
            finally:
                module.run_hull_builder = original_hull
                module.generate_evaluation_samples = original_samples
                module.run_hull_evaluation = original_eval

            self.assertTrue((tmpdir_path / "self_collision_cspace_hulls.json").exists())
            self.assertTrue((tmpdir_path / "self_collision_cspace_hulls.png").exists())
            self.assertTrue((tmpdir_path / "self_collision_eval_samples.json").exists())
            self.assertTrue((tmpdir_path / "self_collision_eval_report.json").exists())
            self.assertTrue((tmpdir_path / "self_collision_eval_summary.png").exists())
            self.assertIn("evaluation", summary)
            self.assertAlmostEqual(summary["evaluation"]["coverage"], 0.5)

    def test_write_evaluation_summary_plot_creates_png(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_benchmark_plot")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = pathlib.Path(tmpdir) / "summary.png"
            module.write_evaluation_summary_plot({
                "coverage": 0.8,
                "miss_rate": 0.2,
                "false_positive_rate": 0.1,
                "boundary_collision_coverage": 0.7,
                "boundary_free_false_positive_rate": 0.3,
            }, out_png)

            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
