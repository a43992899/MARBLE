import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from marble.encoders.BestRQ.model import BestRQCheckpointEncoder
from marble.modules.spectrogram_cache import SpectrogramCacheDataset


ROOT = Path(__file__).resolve().parents[1]


class PrecomputedMelCQTFrontend(nn.Module):
    sample_rate = 48_000
    n_mels = 128
    hop_length = 4

    def get_mel(self, value, normalize=True):
        assert normalize
        return value.mean(dim=-1, keepdim=True).expand(-1, -1, 1024)


class MelCQTPSNRFrontend(PrecomputedMelCQTFrontend):
    pass


class _OnlineMel(nn.Module):
    def forward(self, waveform):
        frames = waveform[:, ::4].abs().add(0.1)
        return frames[:, None, :].expand(-1, 128, -1)


class _Layer(nn.Module):
    def forward(self, hidden, *args, **kwargs):
        return hidden + 1


class _Backbone(nn.Module):
    hidden_size = 1024
    output_frame_rate = 25.0

    def __init__(self, frontend=None):
        super().__init__()
        self.feature_extractor = frontend or PrecomputedMelCQTFrontend()
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(self, mel=None, **kwargs):
        del kwargs
        value = mel
        for layer in self.layers:
            value = layer(value)
        return value, None, None


class _LabelOnlyDataset:
    clip_seconds = 10.0
    index_map = [(0, 1, 22_050, 220_500)]
    meta = [{"audio_path": "data/example.wav", "label": "rock"}]
    LABEL2IDX = {"rock": 7}

    def __len__(self):
        return 1

    def __getitem__(self, index):  # pragma: no cover - must never be called
        raise AssertionError(f"audio dataset __getitem__ was called: {index}")


class _Track:
    total_frames = 800

    def __init__(self, root, relative_path, *, modalities):
        self.root = root
        self.relative_path = relative_path
        assert modalities == ("mel",)

    def load_slice(self, modality, start, stop):
        assert modality == "mel"
        return torch.full((stop - start, 128), 3.0)


class BestRQSpectrogramFrontendTest(unittest.TestCase):
    def test_sz_encoder_accepts_raw_mel_and_rejects_waveform(self):
        backbone = _Backbone()
        with (
            mock.patch.object(
                BestRQCheckpointEncoder,
                "_build_backbone",
                return_value=(backbone, {"run": {"method": "bestrq", "stage": "pretrain"}}),
            ),
            mock.patch.object(BestRQCheckpointEncoder, "_load_backbone"),
        ):
            encoder = BestRQCheckpointEncoder(
                checkpoint="/tmp/fake.ckpt",
                recipe="/tmp/fake.yaml",
                source_root="/tmp",
                expected_num_layers=2,
                frontend_mode="sz",
            )
        layers = encoder(torch.randn(2, 20, 128))
        self.assertEqual(len(layers), 2)
        self.assertEqual(tuple(layers[-1].shape), (2, 20, 1024))
        self.assertEqual(encoder.sample_rate, 48_000)
        with self.assertRaisesRegex(ValueError, "raw-dB Mel"):
            encoder(torch.randn(2, 48_000))

    def test_psnr_encoder_detects_subclass_and_round_trips_frequency_time(self):
        backbone = _Backbone(MelCQTPSNRFrontend())
        encoded = []

        def compress(array, kind):
            encoded.append((array.copy(), kind))
            return array

        with (
            mock.patch.object(
                BestRQCheckpointEncoder,
                "_build_backbone",
                return_value=(backbone, {"run": {"method": "bestrq", "stage": "pretrain"}}),
            ),
            mock.patch.object(BestRQCheckpointEncoder, "_load_backbone"),
            mock.patch.object(
                BestRQCheckpointEncoder, "_build_online_mel", return_value=_OnlineMel()
            ),
            mock.patch(
                "music_tokenizer.data.psnr_spectrogram_codec.compress_spectrogram",
                side_effect=compress,
            ),
            mock.patch(
                "music_tokenizer.data.psnr_spectrogram_codec.decompress_spectrogram",
                side_effect=lambda payload: payload,
            ),
        ):
            encoder = BestRQCheckpointEncoder(
                checkpoint="/tmp/fake.ckpt",
                recipe="/tmp/psnr.yaml",
                source_root="/tmp",
                expected_num_layers=2,
                frontend_mode="psnr",
            )
            layers = encoder(torch.randn(2, 40))
        self.assertTrue(encoder.is_precomputed_frontend)
        self.assertTrue(encoder.is_psnr_frontend)
        self.assertEqual(tuple(layers[-1].shape), (2, 10, 1024))
        self.assertEqual(len(encoded), 2)
        self.assertTrue(all(array.shape == (128, 10) for array, _ in encoded))
        self.assertTrue(all(kind == "mel" for _, kind in encoded))

    def test_cache_dataset_reads_mel_slice_without_audio_getitem(self):
        with (
            mock.patch(
                "marble.modules.spectrogram_cache.validate_spectrogram_contract",
                return_value={"sha256": "abc"},
            ),
            mock.patch(
                "marble.modules.spectrogram_cache.SpectrogramTrack", _Track
            ),
        ):
            dataset = SpectrogramCacheDataset(
                _LabelOnlyDataset(),
                root="/tmp/cache",
                expected_contract_sha256="abc",
            )
            mel, target, identifier = dataset[0]
        self.assertEqual(tuple(mel.shape), (500, 128))
        self.assertTrue(torch.equal(mel[:300], torch.full_like(mel[:300], 3.0)))
        self.assertTrue(torch.equal(mel[300:], torch.full_like(mel[300:], -120.0)))
        self.assertEqual(target, 7)
        self.assertEqual(identifier, "data/example.wav")

    def test_cache_contract_root_can_be_explicitly_remapped_to_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            contract_root = root / "live-marble"
            snapshot_root = root / "snapshot-marble"
            (cache_root / "_meta").mkdir(parents=True)
            contract_root.mkdir()
            snapshot_root.mkdir()
            (cache_root / "_meta" / "contract.json").write_text(
                json.dumps({"source_root": str(contract_root)}) + "\n",
                encoding="utf-8",
            )
            with mock.patch(
                "marble.modules.spectrogram_cache.validate_spectrogram_contract",
                return_value={"sha256": "abc"},
            ):
                dataset = SpectrogramCacheDataset(
                    _LabelOnlyDataset(),
                    root=str(cache_root),
                    expected_contract_sha256="abc",
                    source_root=str(snapshot_root),
                    contract_source_root=str(contract_root),
                )
                self.assertEqual(dataset.source_root, snapshot_root.resolve())
                self.assertEqual(
                    dataset.contract_source_root, contract_root.resolve()
                )
                with self.assertRaisesRegex(ValueError, "source_root mismatch"):
                    SpectrogramCacheDataset(
                        _LabelOnlyDataset(),
                        root=str(cache_root),
                        expected_contract_sha256="abc",
                        source_root=str(snapshot_root),
                        contract_source_root=str(root / "wrong-origin"),
                    )

    def test_exp2_runner_sets_frontend_cache_rate_and_probe_seed(self):
        spec = importlib.util.spec_from_file_location(
            "run_bestrq_probe_exp2", ROOT / "scripts" / "run_bestrq_probe.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                task="GTZANBeatTracking",
                lineage="exp2.2",
                checkpoint="/tmp/epoch=001-step=060000.ckpt",
                recipe="/tmp/exp2.2.yaml",
                source_root="/tmp/mert2",
                frontend_mode="sz",
                spectrogram_root="/tmp/marble-sz",
                expected_contract_sha256="abc",
                spectrogram_contract_source_root="/tmp/live-marble",
                probe_seed=2,
                smoke=False,
            )
            config = module.build_config(args, ROOT, Path(directory))
        self.assertEqual(config["seed_everything"], 2)
        self.assertEqual(config["model"]["init_args"]["sample_rate"], 48_000)
        encoder = config["model"]["init_args"]["encoder"]["init_args"]
        self.assertEqual(encoder["frontend_mode"], "sz")
        data = config["data"]["init_args"]
        self.assertEqual(data["spectrogram_cache"]["root"], "/tmp/marble-sz")
        self.assertEqual(data["spectrogram_cache"]["source_root"], str(ROOT))
        self.assertEqual(
            data["spectrogram_cache"]["contract_source_root"],
            "/tmp/live-marble",
        )
        self.assertEqual(data["train"]["init_args"]["sample_rate"], 48_000)
        for split in ("train", "val", "test"):
            self.assertEqual(data[split]["init_args"]["channel_mode"], "mix")
        self.assertEqual(data["train"]["init_args"]["label_freq"], 100)
        self.assertEqual(
            config["model"]["init_args"]["emb_transforms"][1]["init_args"][
                "target_frames"
            ],
            1000,
        )

    def test_psnr_runner_uses_waveform_data_without_sz_cache(self):
        spec = importlib.util.spec_from_file_location(
            "run_bestrq_probe_psnr", ROOT / "scripts" / "run_bestrq_probe.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                task="GTZANGenre",
                lineage="exp2.8.1",
                checkpoint="/tmp/epoch=030-step=060000.ckpt",
                recipe="/tmp/exp2.8.1.yaml",
                source_root="/tmp/mert2",
                frontend_mode="psnr",
                spectrogram_root=None,
                expected_contract_sha256=None,
                spectrogram_contract_source_root=None,
                probe_seed=1234,
                smoke=False,
            )
            config = module.build_config(args, ROOT, Path(directory))
        self.assertEqual(config["model"]["init_args"]["sample_rate"], 48_000)
        encoder = config["model"]["init_args"]["encoder"]["init_args"]
        self.assertEqual(encoder["frontend_mode"], "psnr")
        data = config["data"]["init_args"]
        self.assertNotIn("spectrogram_cache", data)
        for split in ("train", "val", "test"):
            self.assertEqual(data[split]["init_args"]["sample_rate"], 48_000)
            self.assertEqual(data[split]["init_args"]["channel_mode"], "mix")


if __name__ == "__main__":
    unittest.main()
