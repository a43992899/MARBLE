from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


MARBLE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = MARBLE_ROOT / "scripts" / "run_mert_probe.py"
MODEL_DIR = Path("/tmp/MERT-v1-95M").resolve()


def load_runner():
    spec = importlib.util.spec_from_file_location("run_mert_probe", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frequency_values(value, result=None):
    result = result or {"fps": [], "label_fps": [], "label_freq": []}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in result:
                result[key].append(item)
            frequency_values(item, result)
    elif isinstance(value, list):
        for item in value:
            frequency_values(item, result)
    return result


class MERTProbeConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def build(self, task: str, smoke: bool = False):
        args = SimpleNamespace(
            task=task,
            model_dir=str(MODEL_DIR),
            smoke=smoke,
        )
        return self.runner.build_config(args, MARBLE_ROOT, Path("/tmp/output") / task)

    def test_all_tasks_use_local_frozen_mert_and_mlp_reduce(self):
        for task in self.runner.TASK_CONFIGS:
            with self.subTest(task=task):
                config = self.build(task)
                model_args = config["model"]["init_args"]
                encoder = model_args["encoder"]
                self.assertEqual(encoder["init_args"]["train_mode"], "freeze")
                self.assertEqual(
                    encoder["init_args"]["pre_trained_folder"],
                    str(MODEL_DIR),
                )
                transforms = model_args["emb_transforms"]
                self.assertEqual(transforms[0]["init_args"]["layers"], ["1..12"])
                self.assertEqual(
                    transforms[1]["init_args"],
                    {"num_layers": 12, "hidden_size": 768},
                )
                audio_transforms = config["data"]["init_args"]["audio_transforms"]
                for split in ("train", "val", "test"):
                    matched = [
                        item
                        for item in audio_transforms[split]
                        if item["class_path"] == self.runner.FEATURE_EXTRACTOR
                    ]
                    self.assertEqual(len(matched), 1)
                    self.assertEqual(
                        matched[0]["init_args"]["pre_trained_folder"],
                        str(MODEL_DIR),
                    )
                self.assertNotIn("ckpt_path", config)

    def test_beat_is_fully_converted_to_one_hundred_hz(self):
        config = self.build("GTZANBeatTracking")
        transforms = config["model"]["init_args"]["emb_transforms"]
        self.assertEqual(
            transforms[-1],
            {
                "class_path": "marble.modules.transforms.LinearInterpolation",
                "init_args": {"target_frames": 1000},
            },
        )
        values = frequency_values(config)
        for key, items in values.items():
            self.assertTrue(items, key)
            self.assertEqual(set(items), {100}, key)

    def test_smoke_disables_logger_and_checkpoint_callbacks(self):
        config = self.build("GS", smoke=True)
        self.assertFalse(config["trainer"]["logger"])
        self.assertEqual(config["trainer"]["fast_dev_run"], 1)
        callbacks = [item["class_path"] for item in config["trainer"]["callbacks"]]
        self.assertFalse(any(path.endswith("ModelCheckpoint") for path in callbacks))
        self.assertFalse(any(path.endswith("LearningRateMonitor") for path in callbacks))


if __name__ == "__main__":
    unittest.main()
