#!/usr/bin/env python3
"""Run one frozen MERT-v1-95M MARBLE probe from fit through test."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


TASK_CONFIGS = {
    "GTZANGenre": "probe.MERT-v1-95M.GTZANGenre.yaml",
    "GTZANBeatTracking": "probe.MERT-v1-95M.GTZANBeatTracking.yaml",
    "GS": "probe.MERT-v1-95M.GS.yaml",
    "EMO": "probe.MERT-v1-95M.EMO.yaml",
    "Chords1217": "probe.MERT-v1-95M.Chords1217.yaml",
}
MODEL_NAME = "m-a-p/MERT-v1-95M"
MODEL_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
TRANSFORMER_LAYERS = "1..12"
FEATURE_EXTRACTOR = "marble.encoders.MERT.model.MERT_v1_95M_FeatureExtractor"


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


def inject_feature_extractor_path(config: dict, model_dir: Path) -> None:
    audio_transforms = config["data"]["init_args"]["audio_transforms"]
    updated = set()
    for split in ("train", "val", "test"):
        for transform in audio_transforms[split]:
            if transform["class_path"] == FEATURE_EXTRACTOR:
                transform.setdefault("init_args", {})["pre_trained_folder"] = str(model_dir)
                updated.add(split)
    if updated != {"train", "val", "test"}:
        raise RuntimeError(f"Missing MERT feature extractor splits: {updated}")


def replace_beat_frequency(value, counts: dict[str, int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in counts and item == 75:
                value[key] = 100
                counts[key] += 1
            else:
                replace_beat_frequency(item, counts)
    elif isinstance(value, list):
        for item in value:
            replace_beat_frequency(item, counts)


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
            "name": f"probe.{args.task}.MERT-v1-95M.mlp12",
            "save_dir": str(task_output),
            "id": f"mert-v1-95m-mlp12-{args.task.lower()}",
            "group": "mert-v1-95m-mlp12",
            "resume": "allow",
            "log_model": False,
        }
    )
    if getattr(args, "smoke", False):
        trainer["logger"] = False

    model_dir = Path(args.model_dir).resolve()
    model_args = config["model"]["init_args"]
    model_args["encoder"] = {
        "class_path": "marble.encoders.MERT.model.MERT_v1_95M_Encoder",
        "init_args": {
            "pre_trained_folder": str(model_dir),
            "train_mode": "freeze",
            "force_half": False,
            "preprocess_in_forward": False,
        },
    }
    transforms = [
        {
            "class_path": "marble.modules.transforms.LayerSelector",
            "init_args": {"layers": [TRANSFORMER_LAYERS]},
        },
        {
            "class_path": "marble.modules.transforms.MLPReduce",
            "init_args": {"num_layers": 12, "hidden_size": 768},
        },
    ]
    if args.task == "GTZANBeatTracking":
        transforms.append(
            {
                "class_path": "marble.modules.transforms.LinearInterpolation",
                "init_args": {"target_frames": 1000},
            }
        )
        counts = {"fps": 0, "label_fps": 0, "label_freq": 0}
        replace_beat_frequency(config, counts)
        if not all(counts.values()):
            raise RuntimeError(f"Incomplete Beat 100 Hz conversion: {counts}")
    model_args["emb_transforms"] = transforms
    inject_feature_extractor_path(config, model_dir)
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
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"MERT model artifact is missing: {path}")
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
        "model_revision": MODEL_REVISION,
        "model_dir": str(model_dir),
        "model_artifacts": artifacts,
        "task": args.task,
        "layer_reduce": {
            "type": "LayerSelector+MLPReduce",
            "selected_hidden_states": TRANSFORMER_LAYERS,
            "num_layers": 12,
            "hidden_size": 768,
        },
        "native_fps": 75,
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
    print(f"Completed MERT-v1-95M {args.task}: {complete}", flush=True)


if __name__ == "__main__":
    main()
