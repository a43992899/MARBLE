from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


MARBLE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = MARBLE_ROOT / "scripts" / "run_muq_probe.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_muq_probe", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MuQProbeConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def build(self, task: str, smoke: bool = False):
        args = SimpleNamespace(
            task=task,
            model_dir="/tmp/muq-large-msd-iter",
            smoke=smoke,
        )
        return self.runner.build_config(args, MARBLE_ROOT, Path("/tmp/output") / task)

    def test_all_tasks_use_twelve_transformer_layers_and_mlp_reduce(self):
        for task in self.runner.TASK_CONFIGS:
            with self.subTest(task=task):
                config = self.build(task)
                model_args = config["model"]["init_args"]
                encoder = model_args["encoder"]
                self.assertEqual(encoder["init_args"]["train_mode"], "freeze")
                transforms = model_args["emb_transforms"]
                self.assertEqual(
                    transforms[0]["init_args"]["layers"],
                    ["1..12"],
                )
                self.assertEqual(
                    transforms[1]["init_args"],
                    {"num_layers": 12, "hidden_size": 1024},
                )
                self.assertNotIn("ckpt_path", config)

    def test_beat_is_upsampled_to_one_hundred_hz(self):
        config = self.build("GTZANBeatTracking")
        transforms = config["model"]["init_args"]["emb_transforms"]
        self.assertEqual(
            transforms[-1],
            {
                "class_path": "marble.modules.transforms.LinearInterpolation",
                "init_args": {"target_frames": 1000},
            },
        )
        self.assertEqual(config["model"]["init_args"]["fps"], 100)

    def test_smoke_disables_logger_and_checkpoint_callbacks(self):
        config = self.build("GS", smoke=True)
        self.assertFalse(config["trainer"]["logger"])
        self.assertEqual(config["trainer"]["fast_dev_run"], 1)
        callbacks = [item["class_path"] for item in config["trainer"]["callbacks"]]
        self.assertFalse(any(path.endswith("ModelCheckpoint") for path in callbacks))
        self.assertFalse(any(path.endswith("LearningRateMonitor") for path in callbacks))


if __name__ == "__main__":
    unittest.main()
