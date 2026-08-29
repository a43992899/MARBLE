from __future__ import annotations

import unittest
from pathlib import Path

import mir_eval
import numpy as np
import torch
import torch.nn as nn

from marble.tasks.GTZANBeatTracking.probe import (
    BeatDownbeatPostProcessor,
    BeatDownbeatTempoMultitaskDecoder,
    tempo_from_beat_times,
)


def event_logits(times, duration=10.0, fps=100):
    logits = torch.full((int(duration * fps),), -8.0)
    frames = np.round(np.asarray(times) * fps).astype(int)
    logits[torch.as_tensor(frames)] = 8.0
    return logits


class _JointDecoder(nn.Module):
    def forward(self, x):
        batch, _, frames, _ = x.shape
        logits = torch.zeros(batch, frames, 3)
        logits[:, :, 0] = 2.0
        logits[:, :, 2] = 9.0
        return logits


class _TempoDecoder(nn.Module):
    def forward(self, logits):
        return logits.mean(dim=1)


class GTZANBeatTrackingRegressionTest(unittest.TestCase):
    def test_joint_downbeat_dbn_preserves_oracle_meter(self):
        beat_times = np.arange(0.5, 10.0, 0.5)
        downbeat_times = np.arange(0.5, 10.0, 2.0)
        processor = BeatDownbeatPostProcessor(fps=100, beats_per_bar=(4,))
        estimated_beats, estimated_downbeats = processor(
            event_logits(beat_times), event_logits(downbeat_times)
        )
        self.assertGreater(mir_eval.beat.f_measure(beat_times, estimated_beats), 0.95)
        self.assertGreater(
            mir_eval.beat.f_measure(downbeat_times, estimated_downbeats), 0.95
        )
        self.assertLessEqual(len(estimated_downbeats), len(downbeat_times) + 1)

    def test_tempo_is_derived_from_decoded_inter_beat_intervals(self):
        beat_times = np.arange(0.5, 10.0, 0.5)
        self.assertAlmostEqual(tempo_from_beat_times(beat_times), 120.0)
        self.assertEqual(tempo_from_beat_times(np.array([0.5])), 0.0)

    def test_multitask_decoder_feeds_trained_beat_head_to_tempo_decoder(self):
        decoder = object.__new__(BeatDownbeatTempoMultitaskDecoder)
        nn.Module.__init__(decoder)
        decoder.fps = 100
        decoder.joint_decoder = _JointDecoder()
        decoder.tempo_decoder = _TempoDecoder()
        decoder.use_ssl_for_tempo = False
        output = decoder(torch.zeros(2, 1, 20, 4))
        self.assertTrue(torch.equal(output["tempo"], torch.full((2,), 2.0)))

    def test_configs_use_annotated_tempo_instead_of_clip_beat_count(self):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        configs = sorted(config_root.glob("probe.*.GTZANBeatTracking*.yaml"))
        self.assertEqual(len(configs), 5)
        for config in configs:
            source = config.read_text(encoding="utf-8")
            self.assertEqual(source.count("use_local_bpm: false"), 3, config.name)


if __name__ == "__main__":
    unittest.main()
