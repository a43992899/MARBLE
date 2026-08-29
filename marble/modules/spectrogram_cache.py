"""Dataset adapter that replaces waveform I/O with immutable SZ3 Mel clips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrogramCacheDataset(Dataset):
    """Wrap a MARBLE audio dataset while never invoking its audio decoder."""

    def __init__(
        self,
        dataset: Dataset,
        root: str,
        expected_contract_sha256: str,
        frame_rate: int | None = None,
        source_root: str | None = None,
        contract_source_root: str | None = None,
    ) -> None:
        # Ordinary waveform MARBLE tasks import this module through
        # BaseDataModule but do not use the MERT2 cache adapter.  Keep the
        # source-side dependency lazy so unrelated encoders remain standalone.
        from music_tokenizer.data import spectrograms

        expected_frame_rate = spectrograms.FRAME_RATE
        frame_rate = expected_frame_rate if frame_rate is None else int(frame_rate)
        if frame_rate != expected_frame_rate:
            raise ValueError(
                f"SZ3 Mel cache requires frame_rate={expected_frame_rate}"
            )
        for attribute in ("meta", "index_map", "clip_seconds"):
            if not hasattr(dataset, attribute):
                raise TypeError(
                    f"SpectrogramCacheDataset requires source.{attribute}: "
                    f"{type(dataset).__name__}"
                )
        self.dataset = dataset
        self.root = str(Path(root).expanduser().resolve())
        self.source_root = (
            Path(source_root).expanduser().resolve() if source_root is not None else None
        )
        self.contract_source_root = (
            Path(contract_source_root).expanduser().resolve()
            if contract_source_root is not None
            else self.source_root
        )
        self.frame_rate = frame_rate
        self.db_floor = spectrograms.DB_FLOOR
        self.mel_bins = spectrograms.MEL_BINS
        self.track_class = spectrograms.SpectrogramTrack
        self.contract = spectrograms.validate_spectrogram_contract(
            self.root, expected_sha256=expected_contract_sha256
        )
        if self.contract_source_root is not None:
            contract_path = Path(self.root) / "_meta" / "contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract_source = Path(str(contract.get("source_root", ""))).expanduser()
            if (
                not contract_source.is_absolute()
                or contract_source.resolve() != self.contract_source_root
            ):
                raise ValueError(
                    "Spectrogram cache source_root mismatch: "
                    f"expected={self.contract_source_root}, "
                    f"actual={contract.get('source_root')!r}, "
                    f"contract={contract_path}"
                )

    def __len__(self) -> int:
        return len(self.dataset)

    def _index(self, index: int) -> tuple[int, int, int, int]:
        entry = self.dataset.index_map[int(index)]
        if len(entry) < 4:
            raise ValueError(f"Invalid MARBLE index_map entry: {entry!r}")
        return tuple(int(value) for value in entry[:4])

    def _targets_and_id(
        self,
        index: int,
        *,
        file_idx: int,
        slice_idx: int,
        orig_sr: int,
        orig_clip_frames: int,
    ) -> tuple[Any, str]:
        info = self.dataset.meta[file_idx]
        audio_path = str(info["audio_path"])
        if hasattr(self.dataset, "get_targets"):
            target = self.dataset.get_targets(
                file_idx=file_idx,
                slice_idx=slice_idx,
                orig_sr=orig_sr,
                orig_clip_frames=orig_clip_frames,
            )
            return target, audio_path
        if hasattr(self.dataset, "beat_times_meta"):
            # Beat labels are already factored from waveform loading in the
            # dataset. Temporarily replace only that instance method; no audio
            # decoder is called, and the original label implementation remains
            # the single owner of event rounding/widening semantics.
            had_override = "_load_and_preprocess" in self.dataset.__dict__
            previous = self.dataset.__dict__.get("_load_and_preprocess")
            self.dataset._load_and_preprocess = lambda **_kwargs: torch.empty(0)
            try:
                _, target, identifier = self.dataset[int(index)]
            finally:
                if had_override:
                    self.dataset._load_and_preprocess = previous
                else:
                    del self.dataset._load_and_preprocess
            return target, str(identifier)

        label = info.get("label")
        label_map = getattr(self.dataset, "LABEL2IDX", None)
        if label_map is not None and isinstance(label, str):
            target: Any = int(label_map[label])
        elif isinstance(label, list):
            target = torch.from_numpy(np.asarray(label, dtype=np.float32))
        else:
            target = label
        identifier = str(info.get("ori_uid", audio_path))
        return target, identifier

    def _cache_key(self, audio_path: str) -> str:
        path = Path(audio_path)
        if path.is_absolute():
            if self.source_root is None:
                raise ValueError(
                    "Absolute MARBLE audio_path requires spectrogram_cache.source_root: "
                    f"{audio_path}"
                )
            try:
                path = path.resolve().relative_to(self.source_root)
            except ValueError as exc:
                raise ValueError(
                    f"MARBLE audio_path is outside cache source_root: {audio_path}"
                ) from exc
        if ".." in path.parts:
            raise ValueError(f"MARBLE SZ cache key escapes source_root: {audio_path}")
        return path.as_posix()

    def _mel_clip(self, relative_path: str, slice_idx: int) -> torch.Tensor:
        clip_frames = int(round(float(self.dataset.clip_seconds) * self.frame_rate))
        start = int(round(slice_idx * float(self.dataset.clip_seconds) * self.frame_rate))
        track = self.track_class(
            self.root, relative_path, modalities=("mel",)
        )
        output = torch.full(
            (clip_frames, self.mel_bins), self.db_floor, dtype=torch.float32
        )
        if start >= track.total_frames:
            return output
        stop = min(start + clip_frames, track.total_frames)
        output[: stop - start] = track.load_slice("mel", start, stop)
        return output

    def __getitem__(self, index: int):
        file_idx, slice_idx, orig_sr, orig_clip_frames = self._index(index)
        info = self.dataset.meta[file_idx]
        audio_path = str(info["audio_path"])
        relative_path = self._cache_key(audio_path)
        target, identifier = self._targets_and_id(
            int(index),
            file_idx=file_idx,
            slice_idx=slice_idx,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames,
        )
        return self._mel_clip(relative_path, slice_idx), target, identifier
