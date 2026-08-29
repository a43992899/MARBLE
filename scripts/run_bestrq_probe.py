#!/usr/bin/env python3
"""Run one frozen BestRQ MARBLE probe from fit through test."""

from __future__ import annotations

import argparse
import json
import os
import re
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=tuple(TASK_CONFIGS))
    parser.add_argument(
        "--lineage", required=True, choices=("exp1.1", "exp1.3", "exp2.1", "exp2.2", "exp2.3", "exp2.4", "exp2.5", "exp2.6", "exp2.7", "exp2.8.1", "exp2.8.2", "exp2.9.1", "exp2.9.2", "exp2.9.3")
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--frontend-mode", choices=("online", "sz", "psnr"), default="online"
    )
    parser.add_argument("--spectrogram-root")
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--spectrogram-contract-source-root")
    parser.add_argument("--probe-seed", type=int, default=1234)
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_step(checkpoint: str) -> int:
    match = re.search(r"(?:^|-)step=(\d+)\.ckpt$", Path(checkpoint).name)
    if match is None:
        raise ValueError(
            f"Checkpoint filename must end in '-step=<integer>.ckpt': {checkpoint}"
        )
    return int(match.group(1))


def build_config(args, marble_root: Path, task_output: Path) -> dict:
    source = marble_root / "configs" / TASK_CONFIGS[args.task]
    config = yaml.safe_load(source.read_text(encoding="utf-8"))

    trainer = config["trainer"]
    trainer["devices"] = 1
    trainer["num_nodes"] = 1
    frontend_mode = getattr(args, "frontend_mode", "online")
    probe_seed = int(getattr(args, "probe_seed", 1234))
    cached_checkpoint = args.lineage.startswith("exp2.")
    if frontend_mode == "sz" and not cached_checkpoint:
        raise ValueError("SZ evaluation requires an exp2 cached-frontend checkpoint")
    if frontend_mode == "sz" and (
        not getattr(args, "spectrogram_root", None)
        or not getattr(args, "expected_contract_sha256", None)
        or not getattr(args, "spectrogram_contract_source_root", None)
    ):
        raise ValueError(
            "SZ evaluation requires --spectrogram-root and "
            "--expected-contract-sha256 and "
            "--spectrogram-contract-source-root"
        )
    model_sample_rate = 48_000 if cached_checkpoint else 24_000
    config["seed_everything"] = probe_seed
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
        if not callback["class_path"].endswith(
            tuple(excluded_callbacks)
        )
    ]
    trainer["callbacks"].append(
        {
            "class_path": "marble.modules.metrics_callback.MetricsJSONCallback",
            "init_args": {"output_dir": str(task_output)},
        }
    )

    safe_lineage = args.lineage.replace(".", "")
    step_tag = f"s{checkpoint_step(args.checkpoint):06d}"
    legacy_identity = (
        not cached_checkpoint and frontend_mode == "online" and probe_seed == 1234
    )
    identity_suffix = (
        ""
        if legacy_identity
        else f"-{frontend_mode}-seed{probe_seed}"
    )
    run_name = f"probe.{args.task}.{args.lineage}.{step_tag}.mlp24{identity_suffix}"
    logger = trainer["logger"]["init_args"]
    logger.update(
        {
            "project": "marble",
            "name": run_name,
            "save_dir": str(task_output),
            "id": f"bestrq-{safe_lineage}-{step_tag}-mlp24{identity_suffix}-{args.task.lower()}",
            "group": (
                f"bestrq-{step_tag}-mlp24"
                if legacy_identity
                else f"bestrq-{safe_lineage}-{step_tag}-mlp24{identity_suffix}"
            ),
            "resume": "allow",
            "log_model": False,
        }
    )
    if getattr(args, "smoke", False):
        trainer["logger"] = False

    model_args = config["model"]["init_args"]
    model_args["sample_rate"] = model_sample_rate
    model_args["encoder"] = {
        "class_path": "marble.encoders.BestRQ.model.BestRQCheckpointEncoder",
        "init_args": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "recipe": str(Path(args.recipe).resolve()),
            "source_root": str(Path(args.source_root).resolve()),
            "expected_num_layers": 24,
            "train_mode": "freeze",
            "frontend_mode": frontend_mode,
        },
    }
    data_args = config["data"]["init_args"]
    for split in ("train", "val", "test"):
        split_args = data_args[split]["init_args"]
        split_args["sample_rate"] = model_sample_rate
        if cached_checkpoint:
            # Both guards must use the cache contract's mean-to-mono policy;
            # otherwise online and SZ probes differ in channel augmentation too.
            split_args["channel_mode"] = "mix"
    if frontend_mode == "sz":
        data_args["spectrogram_cache"] = {
            "root": str(Path(args.spectrogram_root).resolve()),
            "source_root": str(marble_root.resolve()),
            "contract_source_root": str(
                Path(args.spectrogram_contract_source_root).resolve()
            ),
            "expected_contract_sha256": args.expected_contract_sha256,
            "frame_rate": 50,
        }
    transforms = [
        {
            "class_path": "marble.modules.transforms.MLPReduce",
            "init_args": {"num_layers": 24, "hidden_size": 1024},
        }
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

    config = build_config(args, marble_root, task_output)
    config_path = task_output / "resolved-config.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"Refusing to overwrite changed config: {config_path}")
    config_path.write_text(rendered, encoding="utf-8")

    checkpoint = Path(args.checkpoint).resolve()
    recipe = Path(args.recipe).resolve()
    provenance = {
        "lineage": args.lineage,
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "checkpoint_step": checkpoint_step(str(checkpoint)),
        "recipe": str(recipe),
        "recipe_bytes": recipe.stat().st_size,
        "recipe_mtime_ns": recipe.stat().st_mtime_ns,
        "layer_reduce": {"type": "MLPReduce", "num_layers": 24, "hidden_size": 1024},
        "beat_fps": 100 if args.task == "GTZANBeatTracking" else None,
        "smoke": bool(args.smoke),
        "frontend_mode": args.frontend_mode,
        "probe_seed": int(args.probe_seed),
        "spectrogram_root": args.spectrogram_root,
        "spectrogram_contract_sha256": args.expected_contract_sha256,
        "spectrogram_contract_source_root": (
            args.spectrogram_contract_source_root
        ),
        "audio_channel_policy": "mean_to_mono" if args.lineage.startswith("exp2.") else None,
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
    print(f"Completed {args.lineage} {args.task}: {complete}", flush=True)


if __name__ == "__main__":
    main()
