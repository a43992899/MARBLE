from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch


MARBLE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = MARBLE_ROOT / "scripts" / "run_spear_probe.py"
MODEL_DIR = Path("/tmp/spear-xlarge-v2").resolve()


def load_runner():
    spec = importlib.util.spec_from_file_location("run_spear_probe", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_rates(value, result=None):
    result = [] if result is None else result
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "sample_rate":
                result.append(item)
            sample_rates(item, result)
    elif isinstance(value, list):
        for item in value:
            sample_rates(item, result)
    return result


def decoder_in_dims(value, result=None):
    result = [] if result is None else result
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "in_dim":
                result.append(item)
            decoder_in_dims(item, result)
    elif isinstance(value, list):
        for item in value:
            decoder_in_dims(item, result)
    return result


class SPEARProbeConfigTest(unittest.TestCase):
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

    def test_all_tasks_use_local_frozen_top_layer_and_16khz(self):
        for task in self.runner.TASK_CONFIGS:
            with self.subTest(task=task):
                config = self.build(task)
                model_args = config["model"]["init_args"]
                encoder = model_args["encoder"]
                self.assertEqual(
                    encoder["class_path"],
                    "marble.encoders.SPEAR.model.SPEAR_Encoder",
                )
                self.assertEqual(encoder["init_args"]["train_mode"], "freeze")
                self.assertEqual(
                    encoder["init_args"]["pre_trained_folder"],
                    str(MODEL_DIR),
                )
                transforms = model_args["emb_transforms"]
                self.assertEqual(transforms[0]["init_args"]["layers"], [0])
                self.assertEqual(decoder_in_dims(model_args["decoders"]), [1280])
                self.assertEqual(sample_rates(config), [16000] * 4)
                self.assertEqual(config["trainer"]["precision"], "32-true")
                self.assertNotIn("ckpt_path", config)

    def test_frame_tasks_use_explicit_target_rates(self):
        beat = self.build("GTZANBeatTracking")["model"]["init_args"]
        self.assertEqual(beat["emb_transforms"][-1]["init_args"], {"target_frames": 1000})
        self.assertEqual(beat["fps"], 100)
        self.assertNotIn("in_dim", beat["decoders"][0]["init_args"])
        self.assertEqual(
            beat["decoders"][0]["init_args"]["joint_decoder"]["init_args"]["in_dim"],
            1280,
        )
        chords = self.build("Chords1217")["model"]["init_args"]
        self.assertEqual(chords["emb_transforms"][-1]["init_args"], {"target_frames": 375})

    def test_smoke_disables_logger_and_checkpoint_callbacks(self):
        config = self.build("GS", smoke=True)
        self.assertFalse(config["trainer"]["logger"])
        self.assertEqual(config["trainer"]["fast_dev_run"], 1)
        callbacks = [item["class_path"] for item in config["trainer"]["callbacks"]]
        self.assertFalse(any(path.endswith("ModelCheckpoint") for path in callbacks))
        self.assertFalse(any(path.endswith("LearningRateMonitor") for path in callbacks))

    def test_waveform_datamodule_import_does_not_require_music_tokenizer(self):
        self.assertNotIn("music_tokenizer", sys.modules)
        module = importlib.import_module("marble.core.base_datamodule")
        self.assertTrue(hasattr(module, "BaseAudioDataset"))
        self.assertNotIn("music_tokenizer", sys.modules)


class SPEAREncoderTest(unittest.TestCase):
    def test_adapter_returns_documented_top_layer_and_stays_frozen(self):
        from marble.encoders.SPEAR.model import SPEAR_Encoder

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, audio, audio_lens, return_middle_layers=True):
                self.last_lens = audio_lens
                self.last_return_middle = return_middle_layers
                return {
                    "encoder_out": torch.ones(audio.shape[0], 7, 1280, device=audio.device),
                    "hidden_states": [],
                }

        fake = FakeModel()
        with TemporaryDirectory() as directory, patch(
            "marble.encoders.SPEAR.model._load_official_model",
            return_value=fake,
        ) as loader:
            encoder = SPEAR_Encoder(directory, train_mode="freeze")
            output = encoder(torch.zeros(2, 1, 32000))
            loader.assert_called_once_with(Path(directory).resolve())
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0].shape, (2, 7, 1280))
            self.assertEqual(fake.last_lens.tolist(), [32000, 32000])
            self.assertTrue(fake.last_return_middle)
            encoder.train()
            self.assertFalse(fake.training)
            self.assertFalse(any(parameter.requires_grad for parameter in fake.parameters()))


if __name__ == "__main__":
    unittest.main()
