#!/usr/bin/env python3
"""Run one frozen MuQ MARBLE probe from fit through test."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


TASK_CONFIGS = {
    "GTZANGenre": "probe.MuQ.GTZANGenre.yaml",
    "GTZANBeatTracking": "probe.MuQ.GTZANBeatTracking.100hz.yaml",
    "GS": "probe.MuQ.GS.yaml",
    "EMO": "probe.MuQ.EMO.yaml",
    "Chords1217": "probe.MuQ.Chords1217.yaml",
}
MODEL_NAME = "OpenMuQ/MuQ-large-msd-iter"
TRANSFORMER_LAYERS = "1..12"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=tuple(TASK_CONFIGS))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_config(args, marble_root: Path, task_output: Path) -> dict:
    source = marble_root / "configs" / TASK_CONFIGS[args.task]
    config = yaml.safe_load(source.read_text(encoding="utf-8"))

    trainer = config["trainer"]
    trainer["devices"] = 1
    trainer["num_nodes"] = 1
    trainer["default_root_dir"] = str(task_output)
    trainer["enable_checkpointing"] = False
    if getattr(args, "smoke", False):
        trainer["fast_dev_run"] = 1
        trainer["num_sanity_val_steps"] = 0
    excluded_callbacks = ["ModelCheckpoint", "LoadLatestCheckpointCallback"]
    if getattr(args, "smoke", False):
        excluded_callbacks.append("LearningRateMonitor")
    trainer["callbacks"] = [
        callback
        for callback in trainer["callbacks"]
        if not callback["class_path"].endswith(tuple(excluded_callbacks))
    ]
    trainer["callbacks"].append(
        {
            "class_path": "marble.modules.metrics_callback.MetricsJSONCallback",
            "init_args": {"output_dir": str(task_output)},
        }
    )

    logger = trainer["logger"]["init_args"]
    logger.update(
        {
            "project": "marble",
            "name": f"probe.{args.task}.MuQ-large-msd-iter.mlp12",
            "save_dir": str(task_output),
            "id": f"muq-large-msd-iter-mlp12-{args.task.lower()}",
            "group": "muq-large-msd-iter-mlp12",
            "resume": "allow",
            "log_model": False,
        }
    )
    if getattr(args, "smoke", False):
        trainer["logger"] = False

    model_args = config["model"]["init_args"]
    model_args["encoder"] = {
        "class_path": "marble.encoders.MuQ.model.MuQ_Encoder",
        "init_args": {
            "pre_trained_folder": str(Path(args.model_dir).resolve()),
            "train_mode": "freeze",
        },
    }
    transforms = [
        {
            "class_path": "marble.modules.transforms.LayerSelector",
            "init_args": {"layers": [TRANSFORMER_LAYERS]},
        },
        {
            "class_path": "marble.modules.transforms.MLPReduce",
            "init_args": {"num_layers": 12, "hidden_size": 1024},
        },
    ]
    if args.task == "GTZANBeatTracking":
        transforms.append(
            {
                "class_path": "marble.modules.transforms.LinearInterpolation",
                "init_args": {"target_frames": 1000},
            }
        )
    model_args["emb_transforms"] = transforms
    config.pop("ckpt_path", None)
    return config


def save_compact_probe_state(model, target: Path, provenance: dict) -> None:
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"state_dict": state, "provenance": provenance}, temporary)
    os.replace(temporary, target)


def model_artifacts(model_dir: Path) -> dict:
    artifacts = {}
    for name in ("config.json", "model.safetensors"):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"MuQ model artifact is missing: {path}")
        stat = path.stat()
        artifacts[name] = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return artifacts


def main() -> None:
    from lightning.pytorch.cli import LightningCLI

    args = parse_args()
    marble_root = Path(__file__).resolve().parents[1]
    task_output = Path(args.output_root).resolve() / args.task
    task_output.mkdir(parents=True, exist_ok=True)
    complete = task_output / "complete.json"
    if complete.is_file():
        print(f"Already complete: {complete}", flush=True)
        return

    model_dir = Path(args.model_dir).resolve()
    artifacts = model_artifacts(model_dir)
    config = build_config(args, marble_root, task_output)
    config_path = task_output / "resolved-config.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"Refusing to overwrite changed config: {config_path}")
    config_path.write_text(rendered, encoding="utf-8")

    provenance = {
        "model": MODEL_NAME,
        "model_dir": str(model_dir),
        "model_artifacts": artifacts,
        "task": args.task,
        "layer_reduce": {
            "type": "LayerSelector+MLPReduce",
            "selected_hidden_states": TRANSFORMER_LAYERS,
            "num_layers": 12,
            "hidden_size": 1024,
        },
        "beat_fps": 100 if args.task == "GTZANBeatTracking" else None,
        "smoke": bool(args.smoke),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(task_output / "source.json", provenance)

    os.chdir(marble_root)
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        args=["--config", str(config_path)],
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    if not args.smoke:
        save_compact_probe_state(
            cli.model,
            task_output / "probe-trainable.ckpt",
            provenance,
        )
    cli.trainer.test(cli.model, datamodule=cli.datamodule)

    test_metrics = task_output / "test_metrics.json"
    if not test_metrics.is_file():
        raise RuntimeError(f"Test metrics were not written: {test_metrics}")
    result = json.loads(test_metrics.read_text(encoding="utf-8"))
    if not result.get("metrics"):
        raise RuntimeError(f"Test metrics are empty: {test_metrics}")
    atomic_json(
        complete,
        {
            **provenance,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "test_metrics": result["metrics"],
        },
    )
    print(f"Completed MuQ {args.task}: {complete}", flush=True)


if __name__ == "__main__":
    main()
