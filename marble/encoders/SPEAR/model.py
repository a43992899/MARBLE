"""Frozen SPEAR XLarge v2 encoder adapter."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import types

import torch

from marble.core.base_encoder import BaseEncoder


def _load_official_model(model_dir: Path):
    """Load the complete local custom-code snapshot as one offline package.

    Transformers' dynamic-module loader does not copy SPEAR's recursively
    imported ``zipformer.py`` when ``from_pretrained`` receives a local path.
    Importing the locked snapshot as a package preserves the official relative
    imports, after which the official PreTrainedModel class can load the local
    safetensors without network access.
    """

    identity = hashlib.sha256(str(model_dir).encode("utf-8")).hexdigest()[:16]
    package_name = f"_marble_spear_snapshot_{identity}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__file__ = str(model_dir / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(model_dir)]
        sys.modules[package_name] = package
    configuration = importlib.import_module(f"{package_name}.configuration_spear")
    modeling = importlib.import_module(f"{package_name}.modeling_spear")
    config = configuration.SpearConfig.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    return modeling.SpearModel.from_pretrained(
        str(model_dir),
        config=config,
        local_files_only=True,
    )


class SPEAR_Encoder(BaseEncoder):
    """Expose the official SPEAR top-layer representation to MARBLE."""

    NAME = "SPEAR-xlarge-speech-audio-v2"
    HUGGINGFACE_MODEL_NAME = "marcoyang/spear-xlarge-speech-audio-v2"
    MODEL_REVISION = "c7cdaa4a95de55e3739edd2a14ce4f48c8a5d2cf"
    SAMPLING_RATE = 16000
    TOKEN_RATE = 50
    NUM_FEATURES = 1280
    N_ZIPFORMER_LAYERS = 13

    def __init__(
        self,
        pre_trained_folder: str,
        train_mode: str = "freeze",
    ) -> None:
        super().__init__()
        if train_mode != "freeze":
            raise ValueError("The SPEAR MARBLE adapter supports frozen probing only")
        if not pre_trained_folder:
            raise ValueError("pre_trained_folder must name an offline SPEAR snapshot")
        model_dir = Path(pre_trained_folder).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"SPEAR model directory does not exist: {model_dir}")

        self.sample_rate = self.SAMPLING_RATE
        self.model_dir = model_dir
        self.model = _load_official_model(model_dir)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        """Keep the frozen upstream encoder in eval mode during probe fitting."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor]:
        del args, kwargs
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 2:
            raise ValueError(f"SPEAR expects [batch, samples] mono audio, got {tuple(x.shape)}")

        parameter = next(self.model.parameters())
        audio = x.to(device=parameter.device, dtype=parameter.dtype)
        audio_lens = torch.full(
            (audio.shape[0],),
            audio.shape[1],
            dtype=torch.long,
            device=audio.device,
        )
        # The official XLarge v2 snapshot is FP32 and its model-card example
        # performs eval/no-grad inference.  Keep that contract even if a probe
        # trainer enables autocast for its trainable head.
        with torch.no_grad(), torch.autocast(
            device_type=audio.device.type,
            enabled=False,
        ):
            outputs = self.model(
                audio.float(),
                audio_lens,
                return_middle_layers=True,
            )
        if not isinstance(outputs, dict) or "encoder_out" not in outputs:
            raise RuntimeError("SPEAR did not return the documented encoder_out dictionary")
        encoder_out = outputs["encoder_out"]
        if encoder_out.ndim != 3 or encoder_out.shape[-1] != self.NUM_FEATURES:
            raise RuntimeError(
                "Unexpected SPEAR encoder_out shape: "
                f"{tuple(encoder_out.shape)}; expected [batch, frames, {self.NUM_FEATURES}]"
            )
        return (encoder_out,)
