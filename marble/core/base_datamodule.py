# marble/core/base_datamodule.py

import os
import json
from abc import ABCMeta, abstractmethod, ABC
from typing import Sequence, Union, Dict, List, Tuple, Optional

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.utils import instantiate_from_config
from marble.modules.transforms import AudioTransformDataset
from marble.modules.spectrogram_cache import SpectrogramCacheDataset


class BaseDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train: dict,
        val: dict,
        test: dict,
        audio_transforms: dict | None = None,  # 改成 dict
        spectrogram_cache: dict | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # audio_transforms 是 dict，包含 train/val/test keys
        self.audio_transforms = audio_transforms or {"train": [], "val": [], "test": []}
        self.spectrogram_cache = dict(spectrogram_cache or {})

        self.train_config = train
        self.val_config   = val
        self.test_config  = test

    def _wrap(self, dataset: Dataset, stage: str) -> Dataset:
        """根据 stage 选对应的 transforms 列表来 wrap Dataset"""
        transforms = [
            instantiate_from_config(cfg) 
            for cfg in self.audio_transforms.get(stage, [])
        ]
        if self.spectrogram_cache:
            if transforms:
                raise ValueError(
                    "Waveform audio_transforms are incompatible with spectrogram_cache"
                )
            dataset = SpectrogramCacheDataset(dataset, **self.spectrogram_cache)
        if transforms:
            return AudioTransformDataset(dataset, transforms)
        print(f"No transforms for stage '{stage}', using original dataset.")
        return dataset

    def setup(self, stage: str | None = None):
        # 原始 train/val dataset
        train_ds = instantiate_from_config(self.train_config)
        val_ds   = instantiate_from_config(self.val_config)
        # 分别 wrap
        self.train_dataset = self._wrap(train_ds, "train")
        self.val_dataset   = self._wrap(val_ds,   "val")
        test_ds = instantiate_from_config(self.test_config)
        self.test_dataset = self._wrap(test_ds, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class BaseAudioDataset(Dataset, ABC):
    """
    A generic base class for audio-based datasets that:
      1. Loads metadata from a JSONL file (one JSON object per line).
      2. Builds a list of clip‐slices (index_map) based on `clip_seconds` and `min_clip_ratio`.
      3. Prepares resamplers if the original sample rate differs from the target sample_rate.
      4. Provides a common `_load_and_preprocess` method that:
         - Loads a specific slice of audio from disk.
         - Handles channel selection/mixing.
         - Resamples to `self.sample_rate` if needed.
         - Pads or truncates to exactly `self.clip_len_target`.
      5. Leaves label/target generation to subclasses via the abstract `get_targets` method.
    """

    def __init__(
        self,
        jsonl: str,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        label_freq: int = -1,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
        backend: Optional[str] = None
    ):
        """
        Args:
            jsonl (str): Path to a JSONL file, where each line is a JSON object 
                          containing at least:
                            - "audio_path": path to the audio file on disk
                            - "sample_rate": original sample rate (int)
                            - "num_samples": total number of samples (int)
                          Subclasses may expect additional fields (e.g., labels).
            sample_rate (int): Target sample rate to which all audio will be resampled.
            channels (int): Desired number of output channels (e.g., 1 for mono, 2 for stereo).
            clip_seconds (float): Duration (in seconds) of each clip to slice from the audio.
            label_freq (int): Frame rate at which labels will be produced (e.g., 10 → 10 labels per second).
            channel_mode (str): How to downmix when `channels == 1` but the audio is multi-channel.
                                Options: "first", "mix", "random".
            min_clip_ratio (float): Minimum fraction of a final (possibly partial) clip to keep.
                                    E.g., if clip_seconds = 10, orig_sr = 44100, num_samples = 450000,
                                    then there are 10 full clips (10·44100 = 441000), plus 9000 samples left.
                                    If 9000/44100 = 0.204 > min_clip_ratio, a partial clip is kept.
        """
        super().__init__()
        assert os.path.isfile(jsonl), f"JSONL file not found: {jsonl}"
        assert isinstance(sample_rate, (int, float)), "sample_rate must be an integer or float"
        self.jsonl = jsonl
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.channel_mode = channel_mode
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.label_freq = label_freq
        self.min_clip_ratio = min_clip_ratio
        self.backend = backend

        # Validate channel_mode if expecting mono output
        if self.channels == 1 and self.channel_mode not in ("first", "mix", "random"):
            raise ValueError(f"channel_mode must be one of 'first', 'mix', 'random' when channels=1, got: {self.channel_mode}")

        # 1. Load metadata entries from the JSONL file
        with open(self.jsonl, "r") as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # 2. Pre-create resamplers for any original sample rate != target sample_rate
        self.resamplers: dict = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

        # 3. Build index_map: a list of tuples (file_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: List[Tuple[int, int, int, int]] = self._build_index_map(
            metas=self.meta,
            clip_seconds=self.clip_seconds,
            min_clip_ratio=self.min_clip_ratio
        )

    def __len__(self) -> int:
        """
        Total number of clips (slices) across all audio files.
        """
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns:
            waveform (torch.Tensor): Tensor of shape (channels, clip_len_target).
            targets (Any): Whatever the subclass defines as target (e.g., frame-level labels).
            audio_path (str): Path to the original audio file from which the clip was taken.
        """
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info = self.meta[file_idx]
        audio_path = info["audio_path"]

        # 1. Load & preprocess the audio slice
        waveform = self._load_and_preprocess(
            path=audio_path,
            slice_idx=slice_idx,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames
        )
        # waveform.shape == (self.channels, self.clip_len_target)

        # 2. Delegate target/label creation to subclass
        targets = self.get_targets(
            file_idx=file_idx,
            slice_idx=slice_idx,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames
        )

        return waveform, targets, audio_path

    @abstractmethod
    def get_targets(
        self,
        file_idx: int,
        slice_idx: int,
        orig_sr: int,
        orig_clip_frames: int
    ):
        """
        Abstract method: subclasses must override this to produce whatever labels/targets
        are needed for a given clip.
        
        Args:
            file_idx (int): Index of the audio file in self.meta.
            slice_idx (int): Which slice number (0-based) of length `orig_clip_frames` within the file.
            orig_sr (int): Original sample rate of the audio file.
            orig_clip_frames (int): Number of samples per clip at original sample rate 
                                    (i.e., floor(clip_seconds * orig_sr)).
        Returns:
            Any data structure representing the target(s) for this clip.
        """
        raise NotImplementedError("Subclasses must implement get_targets().")

    def _load_and_preprocess(
        self,
        path: str,
        slice_idx: int,
        orig_sr: int,
        orig_clip_frames: int
    ) -> torch.Tensor:
        """
        Internal helper to:
          1. Load a slice of audio via torchaudio.load( frame_offset & num_frames ).
          2. Handle channel selection/mixing or replication.
          3. Resample to self.sample_rate if needed.
          4. Pad with zeros or truncate to exactly self.clip_len_target.

        Returns:
            waveform (torch.Tensor): Tensor of shape (channels, clip_len_target).
        """
        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip_frames,
            backend=self.backend
        )  # waveform shape: (orig_channels, actual_frames)

        # 1. Channel handling / downmixing / replication
        waveform = self._select_and_mix_channels(waveform)

        # 2. Resample if original SR != target
        waveform = self._resample_if_needed(waveform, orig_sr)

        # 3. Pad or truncate to exact length
        waveform = self._pad_or_truncate(waveform)

        return waveform

    def _select_and_mix_channels(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        If the original number of channels >= desired channels:
          - If target is mono (channels == 1):
            * "first": take the first channel
            * "mix": average over all original channels
            * "random": 50% chance to average; otherwise pick a random original channel
          - If target > 1: keep the first `self.channels` channels
        If original < desired:
          - Replicate the last channel until there are exactly `self.channels`.
        """
        orig_ch = waveform.size(0)
        if orig_ch >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    return waveform[0:1, :]
                elif self.channel_mode == "mix":
                    return waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    if torch.rand(()) < 0.5:
                        return waveform.mean(dim=0, keepdim=True)
                    else:
                        idx = torch.randint(0, orig_ch, ())
                        return waveform[idx : idx + 1, :]
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                # Keep the first self.channels channels
                return waveform[: self.channels, :]
        else:
            # orig_ch < self.channels → replicate last channel
            deficit = self.channels - orig_ch
            tail = waveform[-1:, :].repeat(deficit, 1)
            return torch.cat([waveform, tail], dim=0)

    def _resample_if_needed(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        If orig_sr != self.sample_rate, run the waveform through the corresponding resampler.
        """
        if orig_sr != self.sample_rate:
            return self.resamplers[orig_sr](waveform)
        return waveform

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Pads with zeros on the right or truncates to make waveform.size(1) == self.clip_len_target.
        """
        cur_len = waveform.size(1)
        if cur_len < self.clip_len_target:
            pad_amt = self.clip_len_target - cur_len
            return F.pad(waveform, (0, pad_amt), mode="constant", value=0.0)
        elif cur_len > self.clip_len_target:
            return waveform[:, : self.clip_len_target]
        return waveform

    def _build_index_map(
        self,
        metas: List[dict],
        clip_seconds: float,
        min_clip_ratio: float
    ) -> List[Tuple[int, int, int, int]]:
        """
        Construct a list of tuples indicating how to slice each audio file into fixed-length clips.

        Each entry is (file_idx, slice_idx, orig_sr, orig_clip_frames), where:
          - file_idx: index into self.meta
          - slice_idx: which clip number within that file (0-based)
          - orig_sr: the original sample rate of that file
          - orig_clip_frames: number of samples per clip at the original sample rate
                              (i.e., floor(clip_seconds * orig_sr))

        If the final partial segment has length >= min_clip_ratio * orig_clip_frames, it is included as well.
        """
        index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(metas):
            orig_sr = int(info["sample_rate"])
            total_samples = int(info["num_samples"])
            orig_clip_frames = int(clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            if rem / orig_clip_frames >= min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                index_map.append((file_idx, slice_idx, orig_sr, orig_clip_frames))

        return index_map
