"""Callbacks for machine-readable MARBLE evaluation artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


class MetricsJSONCallback(Callback):
    """Persist scalar fit/test callback metrics as auditable JSON artifacts."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)

    @staticmethod
    def _scalars(metrics):
        result = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()
            if isinstance(value, (bool, int, float, str)) or value is None:
                result[str(key)] = value
        return result

    def _write(self, trainer, stage: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": stage,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "global_step": int(trainer.global_step),
            "current_epoch": int(trainer.current_epoch),
            "metrics": self._scalars(trainer.callback_metrics),
        }
        target = self.output_dir / f"{stage}_metrics.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def on_fit_end(self, trainer, pl_module) -> None:
        del pl_module
        self._write(trainer, "fit")

    def on_test_end(self, trainer, pl_module) -> None:
        del pl_module
        self._write(trainer, "test")
