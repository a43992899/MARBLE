"""Frozen music-tokenizer BestRQ checkpoint adapter for MARBLE probes."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from jsonargparse import ArgumentParser

from marble.core.base_encoder import BaseEncoder
from music_tokenizer.engine.config import load_recipe, resolved_container
from music_tokenizer.legacy.checkpoint import (
    extract_state_dict,
    load_checkpoint,
    load_state_dict_with_policy,
)


class BestRQCheckpointEncoder(BaseEncoder):
    """Load only the frozen BestRQ backbone and expose every block output.

    MARBLE's MLPReduce consumes a sequence of [B, T, H] tensors. The training
    checkpoint contains the complete Lightning module and optimizer, but a
    downstream probe needs only state below model.model.
    """

    NAME = "BestRQ"
    SAMPLING_RATE = 24_000
    TOKEN_RATE = 25
    NUM_FEATURES = 1024

    def __init__(
        self,
        checkpoint: str,
        recipe: str,
        source_root: str,
        expected_num_layers: int = 24,
        frontend_mode: str = "online",
        train_mode: str = "freeze",
    ) -> None:
        super().__init__()
        if train_mode != "freeze":
            raise ValueError("BestRQCheckpointEncoder only supports train_mode='freeze'")
        if frontend_mode not in {"online", "sz", "psnr"}:
            raise ValueError("frontend_mode must be 'online', 'sz', or 'psnr'")
        self.frontend_mode = str(frontend_mode)

        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.recipe = str(Path(recipe).expanduser().resolve())
        self.source_root = str(Path(source_root).expanduser().resolve())
        self.expected_num_layers = int(expected_num_layers)
        self.sample_rate = self.SAMPLING_RATE

        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            backbone, config = self._build_backbone()
            self.load_report = self._load_backbone(backbone)
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        actual_layers = len(backbone.layers)
        if actual_layers != self.expected_num_layers:
            raise ValueError(
                f"Expected {self.expected_num_layers} backbone layers, got {actual_layers}"
            )
        hidden_size = int(getattr(backbone, "hidden_size", -1))
        if hidden_size != self.NUM_FEATURES:
            raise ValueError(
                f"Expected hidden_size={self.NUM_FEATURES}, got {hidden_size}"
            )
        self.sample_rate = int(
            getattr(backbone.feature_extractor, "sample_rate", self.SAMPLING_RATE)
        )
        self.token_rate = int(
            round(float(getattr(backbone, "output_frame_rate", self.TOKEN_RATE)))
        )
        frontend_type = type(backbone.feature_extractor)
        self.is_precomputed_frontend = any(
            candidate.__name__ == "PrecomputedMelCQTFrontend"
            for candidate in frontend_type.__mro__
        )
        self.is_psnr_frontend = frontend_type.__name__ == "MelCQTPSNRFrontend"
        if self.frontend_mode == "sz" and not self.is_precomputed_frontend:
            raise ValueError(
                "frontend_mode='sz' requires a PrecomputedMelCQTFrontend recipe"
            )
        if self.frontend_mode == "psnr" and not self.is_psnr_frontend:
            raise ValueError(
                "frontend_mode='psnr' requires a MelCQTPSNRFrontend recipe"
            )
        if self.is_precomputed_frontend and self.frontend_mode in {"online", "psnr"}:
            self.online_mel = self._build_online_mel(backbone.feature_extractor)
        else:
            self.online_mel = None

        self.model = backbone
        self.resolved_run = dict(config["run"])
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.train(False)

    def _build_backbone(self) -> tuple[nn.Module, dict[str, Any]]:
        config = resolved_container(load_recipe(self.recipe))
        run = config["run"]
        if run["method"] != "bestrq" or run["stage"] != "pretrain":
            raise ValueError(
                "BestRQCheckpointEncoder requires a bestrq/pretrain recipe, "
                f"got {run['method']}/{run['stage']}"
            )

        node = config["model"]["init_args"]["encoder"]
        frontend = node["init_args"]["feature_extractor"]["init_args"]
        cmvn = Path(frontend["cmvn_path"])
        if not cmvn.is_absolute():
            frontend["cmvn_path"] = str((Path(self.source_root) / cmvn).resolve())

        parser = ArgumentParser()
        parser.add_subclass_arguments(
            nn.Module,
            "encoder",
            required=True,
            fail_untyped=False,
        )
        parsed = parser.parse_object({"encoder": node})
        return parser.instantiate_classes(parsed).encoder, config

    def _load_backbone(self, backbone: nn.Module):
        payload = load_checkpoint(self.checkpoint)
        state = extract_state_dict(payload)
        prefix = "model.model."
        encoder_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise RuntimeError(
                f"No checkpoint keys found below required prefix {prefix!r}"
            )
        report = load_state_dict_with_policy(
            backbone,
            encoder_state,
            path=self.checkpoint,
            strict=True,
        )
        del encoder_state, state, payload
        gc.collect()
        return report

    @staticmethod
    def _build_online_mel(frontend: nn.Module) -> nn.Module:
        try:
            from nnAudio.features import MelSpectrogram
        except ImportError as exc:
            raise RuntimeError(
                "Online guard for cached checkpoints requires nnAudio==0.3.4"
            ) from exc
        return MelSpectrogram(
            sr=int(frontend.sample_rate),
            n_fft=int(frontend.spec.n_fft),
            hop_length=int(frontend.hop_length),
            n_mels=int(frontend.n_mels),
            fmax=float(frontend.spec.mel_fmax),
        )

    def _online_cached_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        amplitude = self.online_mel(waveform.to(torch.float32))
        # nnAudio's centered STFT emits one endpoint frame for an exact-hop
        # clip. Cache crops are half-open [start, stop), so trim that endpoint
        # to keep online and SZ probes on the same 50 Hz timebase.
        target_frames = waveform.shape[-1] // int(self.model.feature_extractor.hop_length)
        amplitude = amplitude[..., :target_frames]
        raw_mel = (20.0 * torch.log10(amplitude.clamp_min(1e-6))).transpose(1, 2)
        if self.frontend_mode == "psnr":
            return self._psnr_round_trip(raw_mel)
        return raw_mel

    @staticmethod
    def _psnr_round_trip(raw_mel: torch.Tensor) -> torch.Tensor:
        """Apply the schema-4 Mel PSNR40/-60dB SZ3 contract in memory."""

        import numpy as np

        from music_tokenizer.data.psnr_spectrogram_codec import (
            compress_spectrogram,
            decompress_spectrogram,
        )

        restored = []
        for sample in raw_mel.detach().cpu().numpy():
            frequency_time = np.ascontiguousarray(sample.T, dtype=np.float32)
            compressed = compress_spectrogram(frequency_time, "mel")
            decoded = decompress_spectrogram(compressed)
            restored.append(torch.from_numpy(np.ascontiguousarray(decoded.T)))
        return torch.stack(restored).to(device=raw_mel.device, dtype=torch.float32)

    def train(self, mode: bool = True):
        """Keep the frozen encoder deterministic when its parent enters train mode."""

        super().train(False)
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, ...]:
        del args, kwargs
        if self.is_precomputed_frontend and self.frontend_mode == "sz":
            if x.ndim != 3 or x.shape[-1] != self.model.feature_extractor.n_mels:
                raise ValueError(
                    "SZ frontend expects raw-dB Mel [B,T,128], got "
                    f"{tuple(x.shape)}"
                )
            raw_mel = x
        else:
            if x.ndim == 3:
                x = x.mean(dim=1)
            if x.ndim != 2:
                raise ValueError(
                    f"Expected waveform [B,T] or [B,C,T], got {tuple(x.shape)}"
                )
            raw_mel = None

        captured: list[torch.Tensor] = []
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output)
            )
            for layer in self.model.layers
        ]
        try:
            self.model.eval()
            with torch.no_grad():
                input_device = raw_mel.device if raw_mel is not None else x.device
                with torch.autocast(device_type=input_device.type, enabled=False):
                    if self.is_precomputed_frontend:
                        if raw_mel is None:
                            raw_mel = self._online_cached_mel(x)
                        mel = self.model.feature_extractor.get_mel(raw_mel, normalize=True)
                    else:
                        mel = self.model.feature_extractor.get_mel(
                            x.to(torch.float32), normalize=True
                        )
                final, _, _ = self.model(mel=mel)
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) != self.expected_num_layers:
            raise RuntimeError(
                f"Captured {len(captured)} layers, expected {self.expected_num_layers}"
            )
        # FusedTransformerEncoder applies a stack-level output norm after its
        # last block. Replace the raw hook output so layer 24 matches the real
        # encoder result; Conformer already returns the same tensor here.
        captured[-1] = final
        return tuple(captured)
