import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from marble.encoders.BestRQ.model import BestRQCheckpointEncoder
from marble.modules.transforms import MLPReduce


ROOT = Path(__file__).resolve().parents[1]


class _FakeFrontend(nn.Module):
    def get_mel(self, waveform, normalize=True):
        assert normalize
        return waveform.unsqueeze(-1).expand(-1, -1, 1024)


class _FakeLayer(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = float(delta)

    def forward(self, hidden, relative_position_embeddings=None):
        del relative_position_embeddings
        return hidden + self.delta


class _FakeBackbone(nn.Module):
    hidden_size = 1024

    def __init__(self):
        super().__init__()
        self.feature_extractor = _FakeFrontend()
        self.layers = nn.ModuleList([_FakeLayer(1), _FakeLayer(2)])
        self.output_norm = nn.LayerNorm(1024)

    def forward(self, wav=None, mel=None):
        if (wav is None) == (mel is None):
            raise ValueError
        hidden = mel
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output_norm(hidden), None, None


class BestRQCheckpointEncoderTest(unittest.TestCase):
    def test_frozen_forward_exposes_layers_and_preserves_rng(self):
        backbone = _FakeBackbone()
        config = {"run": {"method": "bestrq", "stage": "pretrain"}}

        def build():
            torch.rand(7)
            return backbone, config

        torch.manual_seed(123)
        rng_before = torch.random.get_rng_state().clone()
        with (
            mock.patch.object(
                BestRQCheckpointEncoder,
                "_build_backbone",
                side_effect=build,
            ),
            mock.patch.object(
                BestRQCheckpointEncoder,
                "_load_backbone",
                return_value=SimpleNamespace(ok=True),
            ),
        ):
            encoder = BestRQCheckpointEncoder(
                checkpoint="/tmp/fake.ckpt",
                recipe="/tmp/fake.yaml",
                source_root="/tmp",
                expected_num_layers=2,
            )

        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        encoder.train(True)
        self.assertFalse(encoder.training)
        self.assertFalse(encoder.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

        layers = encoder(torch.randn(2, 1, 8))
        self.assertEqual(len(layers), 2)
        self.assertEqual(tuple(layers[0].shape), (2, 8, 1024))
        expected, _, _ = backbone(mel=torch.randn(2, 8, 1024))
        self.assertEqual(tuple(layers[-1].shape), tuple(expected.shape))

        reduced = MLPReduce(num_layers=2, hidden_size=1024)(layers)
        self.assertEqual(tuple(reduced.shape), (2, 1, 8, 1024))

    def test_generated_configs_use_mlp24_and_beat_100hz(self):
        spec = importlib.util.spec_from_file_location(
            "run_bestrq_probe",
            ROOT / "scripts" / "run_bestrq_probe.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            for task in module.TASK_CONFIGS:
                args = SimpleNamespace(
                    task=task,
                    lineage="exp1.1",
                    checkpoint="/tmp/epoch=100-step=200000.ckpt",
                    recipe="/tmp/source.yaml",
                    source_root="/tmp/mert2",
                    smoke=task == "GTZANGenre",
                )
                output = Path(directory) / task
                config = module.build_config(args, ROOT, output)
                self.assertNotIn("ckpt_path", config)
                if not args.smoke:
                    logger = config["trainer"]["logger"]["init_args"]
                    self.assertEqual(
                        logger["name"], f"probe.{task}.exp1.1.s200000.mlp24"
                    )
                    self.assertIn("-s200000-", logger["id"])
                    self.assertEqual(logger["group"], "bestrq-s200000-mlp24")
                transforms = config["model"]["init_args"]["emb_transforms"]
                self.assertTrue(transforms[0]["class_path"].endswith("MLPReduce"))
                self.assertEqual(transforms[0]["init_args"]["num_layers"], 24)
                self.assertEqual(transforms[0]["init_args"]["hidden_size"], 1024)
                callbacks = config["trainer"]["callbacks"]
                self.assertTrue(
                    any(
                        item["class_path"].endswith("MetricsJSONCallback")
                        for item in callbacks
                    )
                )
                self.assertFalse(
                    any(
                        item["class_path"].endswith(
                            ("ModelCheckpoint", "LoadLatestCheckpointCallback")
                        )
                        for item in callbacks
                    )
                )
                if args.smoke:
                    self.assertFalse(config["trainer"]["logger"])
                    self.assertFalse(
                        any(
                            item["class_path"].endswith("LearningRateMonitor")
                            for item in callbacks
                        )
                    )
                if task == "GTZANBeatTracking":
                    self.assertEqual(len(transforms), 2)
                    self.assertTrue(
                        transforms[1]["class_path"].endswith("LinearInterpolation")
                    )
                    self.assertEqual(transforms[1]["init_args"]["target_frames"], 1000)
                    model_args = config["model"]["init_args"]
                    self.assertEqual(model_args["fps"], 100)
                    for split in ("train", "val", "test"):
                        self.assertEqual(
                            config["data"]["init_args"][split]["init_args"]["label_freq"],
                            100,
                        )
                else:
                    self.assertEqual(len(transforms), 1)

        self.assertEqual(module.checkpoint_step("epoch=050-step=100000.ckpt"), 100000)
        with self.assertRaisesRegex(ValueError, "step=<integer>"):
            module.checkpoint_step("checkpoint.ckpt")



if __name__ == "__main__":
    unittest.main()
