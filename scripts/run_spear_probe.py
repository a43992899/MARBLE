#!/usr/bin/env python3
"""Run one offline frozen SPEAR XLarge v2 MARBLE probe from fit through test."""

from __future__ import annotations

import argparse
import hashlib
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
MODEL_NAME = "marcoyang/spear-xlarge-speech-audio-v2"
MODEL_REVISION = "c7cdaa4a95de55e3739edd2a14ce4f48c8a5d2cf"
MODEL_FILES = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "README.md": "cf559314226620e642e7ff4c00b6ebe5bbff6816247b273dfc401107042590f8",
    "config.json": "2db3f5e02610635a9bc5f47b8c215dd8b218e0bc05cfad40d113240bb5f1e76f",
    "configuration_spear.py": "756fd33eb7a6e247484b07a9101c11511a9e3d1af1ae6db48a69468ba92755bf",
    "model.safetensors": "2430597be74dee2271e44f71dbb48f7942423c20313147a5fb85e2af3402e266",
    "modeling_spear.py": "6aa3c3885d3aef09cd01b1e971346f6bd00f41af10432800fc52fad78fd7a698",
    "spear_model.py": "67fb98cece56404a3b1addfd5e1c4567f60705d86a8966cb2081e77171f8a23e",
    "spear_modules.py": "0aa0c542b0e4139c3861c821348a13f56179c5b6ac64c8645f776becf49a7288",
    "zipformer.py": "c5535c4cd2a6acd54fc385af6a1448e7e11d4dd6ffb0dad639e4d26b87976021",
}
SAMPLE_RATE = 16000
HIDDEN_SIZE = 1280


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifacts(model_dir: Path) -> dict:
    artifacts = {}
    for name, expected_sha256 in MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"SPEAR model artifact is missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SPEAR artifact SHA256 mismatch for {name}: "
                f"{actual_sha256}; expected {expected_sha256}"
            )
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    return artifacts


def replace_sample_rate(value, counts: list[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "sample_rate":
                if item != 24000:
                    raise RuntimeError(f"Unexpected source sample_rate: {item}")
                value[key] = SAMPLE_RATE
                counts[0] += 1
            else:
                replace_sample_rate(item, counts)
    elif isinstance(value, list):
        for item in value:
            replace_sample_rate(item, counts)


def replace_decoder_in_dim(value, counts: list[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "in_dim":
                if item != 1024:
                    raise RuntimeError(f"Unexpected source decoder in_dim: {item}")
                value[key] = HIDDEN_SIZE
                counts[0] += 1
            else:
                replace_decoder_in_dim(item, counts)
    elif isinstance(value, list):
        for item in value:
            replace_decoder_in_dim(item, counts)


def build_config(args, marble_root: Path, task_output: Path) -> dict:
    source = marble_root / "configs" / TASK_CONFIGS[args.task]
    config = yaml.safe_load(source.read_text(encoding="utf-8"))

    trainer = config["trainer"]
    trainer["devices"] = 1
    trainer["num_nodes"] = 1
    trainer["precision"] = "32-true"
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
            "name": f"probe.{args.task}.SPEAR-xlarge-v2.top",
            "save_dir": str(task_output),
            "id": f"spear-xlarge-v2-top-{args.task.lower()}",
            "group": "spear-xlarge-v2-top",
            "resume": "allow",
            "log_model": False,
        }
    )
    if getattr(args, "smoke", False):
        trainer["logger"] = False

    model_args = config["model"]["init_args"]
    model_args["encoder"] = {
        "class_path": "marble.encoders.SPEAR.model.SPEAR_Encoder",
        "init_args": {
            "pre_trained_folder": str(Path(args.model_dir).resolve()),
            "train_mode": "freeze",
        },
    }
    transforms = [
        {
            "class_path": "marble.modules.transforms.LayerSelector",
            "init_args": {"layers": [0]},
        }
    ]
    if args.task == "GTZANBeatTracking":
        transforms.append(
            {
                "class_path": "marble.modules.transforms.LinearInterpolation",
                "init_args": {"target_frames": 1000},
            }
        )
    elif args.task == "Chords1217":
        transforms.append(
            {
                "class_path": "marble.modules.transforms.LinearInterpolation",
                "init_args": {"target_frames": 375},
            }
        )
    model_args["emb_transforms"] = transforms
    decoder_counts = [0]
    replace_decoder_in_dim(model_args["decoders"], decoder_counts)
    if decoder_counts[0] != 1:
        raise RuntimeError(
            f"Expected exactly one decoder in_dim, got {decoder_counts[0]}"
        )

    sample_rate_counts = [0]
    replace_sample_rate(config, sample_rate_counts)
    if sample_rate_counts[0] != 4:
        raise RuntimeError(
            f"Expected model plus three dataset sample rates, got {sample_rate_counts[0]}"
        )
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
        "representation": {
            "type": "official encoder_out top layer",
            "source_hidden_states": 1,
            "hidden_size": HIDDEN_SIZE,
        },
        "sample_rate": SAMPLE_RATE,
        "model_dtype": "float32",
        "native_fps": 50,
        "beat_fps": 100 if args.task == "GTZANBeatTracking" else None,
        "chords_fps": 25 if args.task == "Chords1217" else None,
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
    print(f"Completed SPEAR XLarge v2 {args.task}: {complete}", flush=True)


if __name__ == "__main__":
    main()
