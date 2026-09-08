# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
"""Minimal, fail-closed pretraining driver used by the KDN release recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.multiprocessing as mp
from lightning.fabric.strategies import FSDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import get_stateful_stream_tok_dataset  # noqa: E402
from lit_gpt import FusedCrossEntropyLoss  # noqa: E402
from lit_gpt.model import Block, Config, GPT  # noqa: E402
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset  # noqa: E402


_TRAIN_CONFIG = re.compile(
    r"^tsz(?P<batch>[1-9][0-9]*)x(?P<length>[1-9][0-9]*)(?P<length_unit>[kK]?)_"
    r"(?P<tokens>[1-9][0-9]*)(?P<token_unit>[BbMm])$"
)
_CHECKPOINT_MARKER_SCHEMA = "kdn-training-checkpoint-commit-v1"
_FINAL_MARKER_SCHEMA = "kdn-training-final-commit-v1"
_CHECKPOINT_METADATA_SCHEMA = "kdn-training-checkpoint-generation-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SNAPSHOT_NAME = "source_snapshot"
_SOURCE_SNAPSHOT_TEMP_NAME = ".source_snapshot.tmp"


@dataclass(frozen=True)
class TrainingRecipe:
    global_batch_size: int
    sequence_length: int
    max_tokens: int
    world_size: int
    per_device_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    tokens_per_step: int
    max_steps: int
    warmup_steps: int


def _scaled_integer(value: str, unit: str, *, kind: str) -> int:
    scale = {"": 1, "k": 1024, "b": 1_000_000_000, "m": 1_000_000}
    key = unit.lower()
    if key not in scale:
        raise ValueError(f"unsupported {kind} unit: {unit!r}")
    return int(value) * scale[key]


def resolve_training_recipe(
    train_config: str,
    *,
    nodes: int,
    devices_per_node: int,
    micro_batch_size: int,
    max_steps: int | None,
) -> TrainingRecipe:
    """Parse ``tsz<B>x<L>_<TOKENS>`` and derive exact optimizer-step math."""

    match = _TRAIN_CONFIG.fullmatch(train_config)
    if match is None:
        raise ValueError(
            f"unsupported train_config {train_config!r}; expected e.g. tsz128x4k_100B"
        )
    if nodes <= 0 or devices_per_node <= 0 or micro_batch_size <= 0:
        raise ValueError("nodes, devices_per_node, and micro_batch_size must be positive")
    global_batch_size = int(match.group("batch"))
    sequence_length = _scaled_integer(
        match.group("length"), match.group("length_unit"), kind="sequence length"
    )
    max_tokens_value = _scaled_integer(
        match.group("tokens"), match.group("token_unit"), kind="token budget"
    )
    world_size = nodes * devices_per_node
    if global_batch_size % world_size:
        raise ValueError(
            f"global batch {global_batch_size} must be divisible by world size {world_size}"
        )
    per_device_batch_size = global_batch_size // world_size
    if per_device_batch_size % micro_batch_size:
        raise ValueError(
            f"per-device batch {per_device_batch_size} must be divisible by micro batch "
            f"{micro_batch_size}"
        )
    accumulation = per_device_batch_size // micro_batch_size
    tokens_per_step = global_batch_size * sequence_length
    token_limited_steps = max_tokens_value // tokens_per_step
    if token_limited_steps <= 0:
        raise ValueError("token budget must cover at least one optimizer step")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    resolved_steps = token_limited_steps if max_steps is None else int(max_steps)
    warmup_steps = max(1, 1_000_000_000 // tokens_per_step)
    return TrainingRecipe(
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        max_tokens=max_tokens_value,
        world_size=world_size,
        per_device_batch_size=per_device_batch_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation,
        tokens_per_step=tokens_per_step,
        max_steps=resolved_steps,
        warmup_steps=warmup_steps,
    )


def resolve_run_paths(
    output_root: str | os.PathLike, train_config: str, exp_name: str
) -> tuple[Path, Path]:
    if not exp_name or Path(exp_name).name != exp_name:
        raise ValueError("exp_name must be one non-empty path component")
    run_name = f"{train_config}_{exp_name}"
    root = Path(output_root).expanduser().resolve()
    return root / "outputs" / run_name, root / "wandb" / run_name


def validate_resume_request(
    out_dir: str | os.PathLike,
    *,
    resume: bool,
    run_manifest: str | os.PathLike | None = None,
) -> Path | None:
    """Validate either a committed resume or an exact pre-checkpoint restart."""

    out_dir = Path(out_dir)
    checkpoint = out_dir / "latest-checkpoint.json"
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"explicit resume requires committed checkpoint marker: {checkpoint}"
            )
        return checkpoint
    if out_dir.exists():
        if run_manifest is not None:
            manifest = Path(os.path.abspath(Path(run_manifest).expanduser()))
            resolved_out = out_dir.resolve()
            if (
                manifest.parent == resolved_out
                and manifest.is_file()
                and not manifest.is_symlink()
            ):
                entries = {path.name: path for path in out_dir.iterdir()}
                names = set(entries)
                if names == {manifest.name}:
                    return None
                snapshot = entries.get(_SOURCE_SNAPSHOT_NAME)
                if names == {manifest.name, _SOURCE_SNAPSHOT_NAME}:
                    _validate_source_snapshot(snapshot)
                    return None
                temporary = entries.get(_SOURCE_SNAPSHOT_TEMP_NAME)
                if (
                    names == {manifest.name, _SOURCE_SNAPSHOT_TEMP_NAME}
                    and temporary is not None
                    and _safe_snapshot_temporary(temporary)
                ):
                    # Only the private atomic-publication temporary is
                    # recoverable. _snapshot_repository validates or rebuilds it.
                    return None
        raise FileExistsError(
            f"output directory already exists; refusing implicit resume: {out_dir}"
        )
    return None


def mark_no_weight_decay(model: torch.nn.Module) -> None:
    """Freeze decay membership before FSDP turns parameters into sharded views."""

    for parameter in model.parameters():
        parameter._kdn_no_weight_decay = bool(
            getattr(parameter, "_no_weight_decay", False) or parameter.ndim < 2
        )


def build_adamw_param_groups(
    model: torch.nn.Module, *, weight_decay: float
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    names = {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        bucket = "no_decay" if getattr(parameter, "_kdn_no_weight_decay", False) else "decay"
        (no_decay if bucket == "no_decay" else decay).append(parameter)
        names[bucket].append(name)
    if not decay or not no_decay:
        raise RuntimeError("AdamW requires non-empty decay and no-decay parameter groups")
    return (
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        names,
    )


def training_complete(*, step_count: int, max_steps: int) -> bool:
    if step_count > max_steps:
        raise RuntimeError(
            f"optimizer step count {step_count} exceeded exact max_steps {max_steps}"
        )
    return step_count == max_steps


def checkpoint_action(
    *,
    step_count: int,
    max_steps: int,
    save_step_interval: int,
    time_limit_reached: bool,
    smoke_stop_after_step: int | None = None,
) -> str | None:
    """Choose one checkpoint action, giving natural completion precedence."""

    if training_complete(step_count=step_count, max_steps=max_steps):
        return "final"
    if smoke_stop_after_step is not None and step_count == smoke_stop_after_step:
        return "latest-smoke-stop"
    if time_limit_reached:
        return "latest-stop"
    if step_count > 0 and step_count % save_step_interval == 0:
        return "latest"
    return None


def step_metrics(
    *,
    iter_num: int,
    step_count: int,
    loss: float,
    learning_rate: float,
    grad_norm: float,
    tokens_per_step: int,
) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {
        "event": "optimizer_step",
        "iter_num": int(iter_num),
        "optimizer_step": int(step_count),
        "global_tokens": int(step_count * tokens_per_step),
        "loss": float(loss),
        "learning_rate": float(learning_rate),
        "grad_norm": float(grad_norm),
    }
    json.dumps(metrics, allow_nan=False)
    return metrics


def get_lr(
    *, step_count: int, warmup_steps: int, max_steps: int, learning_rate: float, min_lr: float
) -> float:
    if step_count < warmup_steps:
        return learning_rate * step_count / warmup_steps
    if step_count >= max_steps:
        return min_lr
    ratio = (step_count - warmup_steps) / max(1, max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (learning_rate - min_lr)


def _source_snapshot_files() -> dict[Path, Path]:
    sources: dict[Path, Path] = {}
    package = REPO_ROOT / "lit_gpt"
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(REPO_ROOT)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"source snapshot input must not be a symlink: {path}")
        if path.is_file():
            sources[relative] = path
    for relative in (
        Path("pretrain.py"),
        Path("data.py"),
        Path("scripts/train_kdn_1.3B_100B.sh"),
    ):
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source snapshot input is missing or unsafe: {path}")
        sources[relative] = path
    return sources


def _validate_source_snapshot(snapshot: Path | None) -> None:
    if snapshot is None or snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError(f"source snapshot is missing or unsafe: {snapshot}")
    expected = _source_snapshot_files()
    expected_files = set(expected)
    expected_directories = {
        parent
        for relative in expected_files
        for parent in relative.parents
        if parent != Path(".")
    }
    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if path.is_symlink():
            raise RuntimeError(f"source snapshot contains a symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(f"source snapshot contains a non-file entry: {relative}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError("source snapshot file set does not match the release source")
    for relative, source in expected.items():
        copied = snapshot / relative
        if (
            copied.stat().st_size != source.stat().st_size
            or _sha256_file(copied) != _sha256_file(source)
        ):
            raise RuntimeError(f"source snapshot differs from release source: {relative}")


def _safe_snapshot_temporary(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir() or path.is_mount():
        return False
    return all(
        not child.is_symlink()
        and not child.is_mount()
        and (child.is_file() or child.is_dir())
        for child in path.rglob("*")
    )


def _snapshot_repository(out_dir: Path) -> None:
    snapshot = out_dir / _SOURCE_SNAPSHOT_NAME
    temporary = out_dir / _SOURCE_SNAPSHOT_TEMP_NAME
    if snapshot.exists() or snapshot.is_symlink():
        _validate_source_snapshot(snapshot)
        return
    if temporary.exists() or temporary.is_symlink():
        if not _safe_snapshot_temporary(temporary):
            raise RuntimeError(f"unsafe source snapshot temporary: {temporary}")
        try:
            _validate_source_snapshot(temporary)
        except RuntimeError:
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, snapshot)
            _fsync_directory(out_dir)
            return

    temporary.mkdir()
    for relative, source in _source_snapshot_files().items():
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
    _validate_source_snapshot(temporary)
    os.replace(temporary, snapshot)
    _fsync_directory(out_dir)


def _checkpoint_data_path(out_dir: Path, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return out_dir / "latest-data-state-ckpt.pth"
    return out_dir / f"latest-data-states-rank-{rank}-ckpt.pth"


def _generation_data_path(out_dir: Path, generation: str, rank: int) -> Path:
    return out_dir / f"data-{generation}-rank-{rank:05d}.pth"


def _generation_model_path(out_dir: Path, generation: str) -> Path:
    return out_dir / f"model-{generation}.pth"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _file_record(path: Path) -> dict[str, int | str]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"checkpoint artifact is not a regular file: {path}")
    return {"name": path.name, "size": value.st_size, "sha256": _sha256_file(path)}


def _checkpoint_metadata(
    *,
    step_count: int,
    iter_num: int,
    world_size: int,
    accumulation_steps: int,
    hparams_sha256: str,
    run_manifest_sha256: str | None,
) -> dict[str, object]:
    if step_count <= 0 or iter_num != step_count * accumulation_steps:
        raise ValueError("checkpoint generation must be an optimizer-step boundary")
    return {
        "schema": _CHECKPOINT_METADATA_SCHEMA,
        "generation": f"step-{step_count:09d}",
        "optimizer_step": step_count,
        "iter_num": iter_num,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation_steps,
        "hparams_sha256": hparams_sha256,
        "run_manifest_sha256": run_manifest_sha256,
    }


def _record_path(out_dir: Path, record: object, expected_name: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"name", "size", "sha256"}:
        raise RuntimeError(f"invalid checkpoint artifact record for {expected_name}")
    if record["name"] != expected_name:
        raise RuntimeError(
            f"checkpoint artifact name mismatch: expected {expected_name}, got {record['name']!r}"
        )
    if type(record["size"]) is not int or record["size"] <= 0:
        raise RuntimeError(f"invalid checkpoint artifact size for {expected_name}")
    if not isinstance(record["sha256"], str) or _SHA256.fullmatch(record["sha256"]) is None:
        raise RuntimeError(f"invalid checkpoint artifact SHA256 for {expected_name}")
    path = out_dir / expected_name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"checkpoint artifact is missing or not regular: {path}")
    actual = _file_record(path)
    if actual != record:
        raise RuntimeError(f"checkpoint artifact hash/size mismatch: {path}")
    return path


def _load_model_checkpoint_metadata(path: Path) -> dict[str, object]:
    """Read checkpoint structure with meta tensors, without materializing weights."""

    checkpoint = torch.load(
        path,
        map_location="meta",
        weights_only=False,
        mmap=True,
    )
    expected = {
        "model",
        "optimizer",
        "hparams",
        "iter_num",
        "step_count",
        "checkpoint_meta",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected:
        raise RuntimeError(f"model checkpoint root schema changed: {path}")
    if not isinstance(checkpoint["model"], Mapping) or not checkpoint["model"]:
        raise RuntimeError(f"model checkpoint has no model state: {path}")
    if not isinstance(checkpoint["hparams"], dict):
        raise RuntimeError(f"model checkpoint hparams changed: {path}")
    return checkpoint


def _validate_data_checkpoint_payload(
    path: Path, *, expected_metadata: dict[str, object], expected_rank: int
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "metadata",
        "rank",
        "kdn_data_manifest",
        "state_dict",
        "rng_state",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError(f"rank {expected_rank} data checkpoint schema changed")
    if payload["metadata"] != expected_metadata or payload["rank"] != expected_rank:
        raise RuntimeError(f"rank {expected_rank} data checkpoint generation changed")
    if not isinstance(payload["kdn_data_manifest"], dict):
        raise RuntimeError(f"rank {expected_rank} data manifest is missing")
    rng = payload["rng_state"]
    if not isinstance(rng, dict) or set(rng) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise RuntimeError(f"rank {expected_rank} RNG checkpoint is incomplete")


def validate_checkpoint_commit(
    out_dir: str | os.PathLike,
    *,
    expected_world_size: int,
    expected_max_steps: int,
    expected_hparams_sha256: str,
    expected_run_manifest_sha256: str | None,
) -> dict[str, object]:
    """Validate the latest transaction marker and every artifact it commits."""

    out_dir = Path(out_dir).resolve()
    marker_path = out_dir / "latest-checkpoint.json"
    try:
        if marker_path.is_symlink():
            raise RuntimeError(f"checkpoint marker must not be a symlink: {marker_path}")
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid committed checkpoint marker: {marker_path}") from exc
    if not isinstance(marker, dict) or set(marker) != {"schema", "metadata", "model", "data"}:
        raise RuntimeError("checkpoint marker schema fields changed")
    if marker["schema"] != _CHECKPOINT_MARKER_SCHEMA:
        raise RuntimeError("checkpoint marker schema changed")
    metadata = marker["metadata"]
    expected_metadata_keys = {
        "schema",
        "generation",
        "optimizer_step",
        "iter_num",
        "world_size",
        "gradient_accumulation_steps",
        "hparams_sha256",
        "run_manifest_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
        raise RuntimeError("checkpoint generation metadata changed")
    step_count = metadata["optimizer_step"]
    accumulation = metadata["gradient_accumulation_steps"]
    if (
        metadata["schema"] != _CHECKPOINT_METADATA_SCHEMA
        or type(step_count) is not int
        or not 0 < step_count <= expected_max_steps
        or type(accumulation) is not int
        or accumulation <= 0
        or type(metadata["iter_num"]) is not int
        or type(metadata["world_size"]) is not int
        or metadata["generation"] != f"step-{step_count:09d}"
        or metadata["iter_num"] != step_count * accumulation
        or metadata["world_size"] != expected_world_size
        or metadata["hparams_sha256"] != expected_hparams_sha256
        or metadata["run_manifest_sha256"] != expected_run_manifest_sha256
    ):
        raise RuntimeError("checkpoint generation metadata does not match this run")
    generation = metadata["generation"]
    model_path = _record_path(
        out_dir, marker["model"], _generation_model_path(out_dir, generation).name
    )
    model_checkpoint = _load_model_checkpoint_metadata(model_path)
    if (
        model_checkpoint["checkpoint_meta"] != metadata
        or model_checkpoint["step_count"] != step_count
        or model_checkpoint["iter_num"] != metadata["iter_num"]
        or _json_sha256(model_checkpoint["hparams"]) != expected_hparams_sha256
        or model_checkpoint["optimizer"] is None
    ):
        raise RuntimeError("model checkpoint payload does not match commit marker")
    data = marker["data"]
    if not isinstance(data, list) or len(data) != expected_world_size:
        raise RuntimeError("checkpoint marker has incomplete per-rank data state")
    for rank, record in enumerate(data):
        data_path = _record_path(
            out_dir, record, _generation_data_path(out_dir, generation, rank).name
        )
        _validate_data_checkpoint_payload(
            data_path, expected_metadata=metadata, expected_rank=rank
        )
    return marker


def validate_final_checkpoint(
    out_dir: str | os.PathLike,
    *,
    expected_world_size: int,
    expected_max_steps: int,
    expected_accumulation_steps: int,
    expected_run_manifest_sha256: str | None,
) -> dict[str, object]:
    """Validate a naturally completed, atomically committed final checkpoint."""

    out_dir = Path(out_dir).resolve()
    marker_path = out_dir / "final-checkpoint.json"
    try:
        if marker_path.is_symlink():
            raise RuntimeError(f"final checkpoint marker must not be a symlink: {marker_path}")
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid final checkpoint marker: {marker_path}") from exc
    if not isinstance(marker, dict) or set(marker) != {"schema", "metadata", "model"}:
        raise RuntimeError("final checkpoint marker fields changed")
    if marker["schema"] != _FINAL_MARKER_SCHEMA:
        raise RuntimeError("final checkpoint marker schema changed")
    metadata = marker["metadata"]
    expected_keys = {
        "schema",
        "generation",
        "optimizer_step",
        "iter_num",
        "world_size",
        "gradient_accumulation_steps",
        "hparams_sha256",
        "run_manifest_sha256",
        "kind",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise RuntimeError("final checkpoint metadata changed")
    if (
        metadata["schema"] != _CHECKPOINT_METADATA_SCHEMA
        or metadata["kind"] != "final"
        or metadata["optimizer_step"] != expected_max_steps
        or type(metadata["optimizer_step"]) is not int
        or metadata["generation"] != f"step-{expected_max_steps:09d}"
        or metadata["iter_num"] != expected_max_steps * expected_accumulation_steps
        or type(metadata["iter_num"]) is not int
        or metadata["world_size"] != expected_world_size
        or type(metadata["world_size"]) is not int
        or metadata["gradient_accumulation_steps"] != expected_accumulation_steps
        or type(metadata["gradient_accumulation_steps"]) is not int
        or not isinstance(metadata["hparams_sha256"], str)
        or _SHA256.fullmatch(metadata["hparams_sha256"]) is None
        or metadata["run_manifest_sha256"] != expected_run_manifest_sha256
    ):
        raise RuntimeError("final checkpoint does not match natural run completion")
    model_path = _record_path(out_dir, marker["model"], "final-model-ckpt.pth")
    model_checkpoint = _load_model_checkpoint_metadata(model_path)
    if (
        model_checkpoint["checkpoint_meta"] != metadata
        or model_checkpoint["step_count"] != expected_max_steps
        or model_checkpoint["iter_num"]
        != expected_max_steps * expected_accumulation_steps
        or _json_sha256(model_checkpoint["hparams"])
        != metadata["hparams_sha256"]
        or model_checkpoint["optimizer"] is not None
    ):
        raise RuntimeError("final model payload does not match commit marker")
    return marker


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng_state(value: object, device: torch.device) -> None:
    if not isinstance(value, dict) or set(value) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise RuntimeError("checkpoint RNG state is incomplete")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if device.type == "cuda":
        if not isinstance(value["torch_cuda"], torch.Tensor):
            raise RuntimeError("checkpoint CUDA RNG state is missing")
        torch.cuda.set_rng_state(value["torch_cuda"], device)
    elif value["torch_cuda"] is not None:
        raise RuntimeError("CPU checkpoint restoration received CUDA RNG state")


def _atomic_hardlink(target: Path, alias: Path) -> None:
    temporary = alias.with_name(f".{alias.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    os.link(target, temporary)
    os.replace(temporary, alias)
    _fsync_directory(alias.parent)


def _remove_previous_generation(
    out_dir: Path, marker: object, *, keep_names: set[str]
) -> None:
    if not isinstance(marker, dict):
        return
    records = [marker.get("model")]
    data = marker.get("data")
    if isinstance(data, list):
        records.extend(data)
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            continue
        path = out_dir / record["name"]
        if path.parent == out_dir and path.name not in keep_names:
            path.unlink(missing_ok=True)


def _run_manifest_unchanged(args: argparse.Namespace) -> bool:
    if args.run_manifest is None:
        return args.run_manifest_sha256 is None
    try:
        return _sha256_file(args.run_manifest) == args.run_manifest_sha256
    except OSError:
        return False


def _save_final_checkpoint(
    *, fabric: L.Fabric, state: dict[str, object], args: argparse.Namespace
) -> None:
    if not _all_ranks_true(_run_manifest_unchanged(args), device=fabric.device):
        raise RuntimeError("run manifest changed before final checkpoint publication")
    metadata = _checkpoint_metadata(
        step_count=state["step_count"],
        iter_num=state["iter_num"],
        world_size=fabric.world_size,
        accumulation_steps=args.gradient_accumulation_steps,
        hparams_sha256=args.hparams_sha256,
        run_manifest_sha256=args.run_manifest_sha256,
    )
    metadata = {**metadata, "kind": "final"}
    payload = {**state, "optimizer": None, "checkpoint_meta": metadata}
    destination = args.out_dir / "final-model-ckpt.pth"
    temporary = args.out_dir / ".final-model-ckpt.pth.tmp"
    if fabric.global_rank == 0:
        temporary.unlink(missing_ok=True)
    fabric.barrier()
    fabric.save(temporary, payload)
    publication_error = None
    if fabric.global_rank == 0:
        try:
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            os.replace(temporary, destination)
            marker = {
                "schema": _FINAL_MARKER_SCHEMA,
                "metadata": metadata,
                "model": _file_record(destination),
            }
            _atomic_json_save(marker, args.out_dir / "final-checkpoint.json")
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
    publication_error = fabric.broadcast(publication_error, src=0)
    if publication_error is not None:
        raise RuntimeError(f"final checkpoint publication failed: {publication_error}")


def _save_latest_checkpoint(
    *,
    fabric: L.Fabric,
    state: dict[str, object],
    args: argparse.Namespace,
    train_dataloader,
) -> None:
    """Commit a resumable model/data/RNG generation with the marker last."""

    if not args.use_stream_tok:
        raise RuntimeError(
            "resumable checkpoints require the stateful streaming-tokenizer loader"
        )
    metadata = _checkpoint_metadata(
        step_count=state["step_count"],
        iter_num=state["iter_num"],
        world_size=fabric.world_size,
        accumulation_steps=args.gradient_accumulation_steps,
        hparams_sha256=args.hparams_sha256,
        run_manifest_sha256=args.run_manifest_sha256,
    )
    generation = metadata["generation"]
    already_committed = False
    existing_error = None
    if fabric.global_rank == 0:
        marker_path = args.out_dir / "latest-checkpoint.json"
        if marker_path.is_file():
            try:
                existing = json.loads(marker_path.read_text())
                if existing.get("metadata", {}).get("generation") == generation:
                    validate_checkpoint_commit(
                        args.out_dir,
                        expected_world_size=fabric.world_size,
                        expected_max_steps=args.max_steps,
                        expected_hparams_sha256=args.hparams_sha256,
                        expected_run_manifest_sha256=args.run_manifest_sha256,
                    )
                    already_committed = True
            except Exception as exc:
                existing_error = f"{type(exc).__name__}: {exc}"
    existing_error = fabric.broadcast(existing_error, src=0)
    already_committed = fabric.broadcast(already_committed, src=0)
    if existing_error is not None:
        raise RuntimeError(f"existing checkpoint marker is invalid: {existing_error}")
    if already_committed:
        return

    data_path = _generation_data_path(
        args.out_dir, generation, fabric.global_rank
    )
    local_error = None
    try:
        loader_state = train_dataloader.state_dict()
        data_payload = {
            "metadata": metadata,
            "rank": fabric.global_rank,
            "kdn_data_manifest": getattr(train_dataloader, "kdn_data_manifest", None),
            "state_dict": loader_state,
            "rng_state": _capture_rng_state(fabric.device),
        }
        _atomic_torch_save(data_payload, data_path)
        _validate_data_checkpoint_payload(
            data_path,
            expected_metadata=metadata,
            expected_rank=fabric.global_rank,
        )
        _file_record(data_path)
    except Exception as exc:
        local_error = f"rank {fabric.global_rank}: {type(exc).__name__}: {exc}"
    if not _all_ranks_true(local_error is None, device=fabric.device):
        raise RuntimeError(local_error or "another rank failed data checkpoint validation")

    if not _all_ranks_true(_run_manifest_unchanged(args), device=fabric.device):
        raise RuntimeError("run manifest changed before checkpoint publication")

    model_path = _generation_model_path(args.out_dir, generation)
    model_temporary = args.out_dir / f".{model_path.name}.tmp"
    previous_marker = None
    if fabric.global_rank == 0:
        model_temporary.unlink(missing_ok=True)
        marker_path = args.out_dir / "latest-checkpoint.json"
        if marker_path.is_file():
            previous_marker = json.loads(marker_path.read_text())
    fabric.barrier()

    state["checkpoint_meta"] = metadata
    fabric.save(model_temporary, state)
    publication_error = None
    if fabric.global_rank == 0:
        try:
            with model_temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.chmod(model_temporary, 0o444)
            os.replace(model_temporary, model_path)
            model_record = _file_record(model_path)
            data_records = [
                _file_record(_generation_data_path(args.out_dir, generation, rank))
                for rank in range(fabric.world_size)
            ]

            _atomic_hardlink(model_path, args.out_dir / "latest-model-ckpt.pth")
            for rank in range(fabric.world_size):
                _atomic_hardlink(
                    _generation_data_path(args.out_dir, generation, rank),
                    _checkpoint_data_path(args.out_dir, rank, fabric.world_size),
                )
            marker = {
                "schema": _CHECKPOINT_MARKER_SCHEMA,
                "metadata": metadata,
                "model": model_record,
                "data": data_records,
            }
            _atomic_json_save(marker, args.out_dir / "latest-checkpoint.json")
            try:
                _remove_previous_generation(
                    args.out_dir,
                    previous_marker,
                    keep_names={
                        model_record["name"],
                        *(record["name"] for record in data_records),
                    },
                )
            except OSError as exc:
                print(f"warning: could not remove prior checkpoint generation: {exc}")
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
    publication_error = fabric.broadcast(publication_error, src=0)
    if publication_error is not None:
        raise RuntimeError(f"checkpoint transaction failed: {publication_error}")


def _all_ranks_true(value: bool, *, device: torch.device) -> bool:
    result = torch.tensor(int(value), device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MIN)
    return bool(result.item())


def _any_rank_true(value: bool, *, device: torch.device) -> bool:
    result = torch.tensor(int(value), device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
    return bool(result.item())


def _next_batch_or_raise(
    iterator, *, device: torch.device, step_count: int, max_steps: int
):
    """Synchronize rank-local exhaustion before any rank enters model collectives."""

    batch = None
    try:
        batch = next(iterator)
    except StopIteration:
        pass
    if _any_rank_true(batch is None, device=device):
        raise RuntimeError(
            f"training data exhausted on at least one rank at optimizer step "
            f"{step_count} before max_steps={max_steps}"
        )
    return batch


def _create_packed_dataloader(
    *, data_dir: str, batch_size: int, block_size: int, fabric: L.Fabric, seed: int
) -> DataLoader:
    files = sorted(Path(data_dir).glob("train_slim*"))
    if not files:
        raise FileNotFoundError(f"no packed train_slim shards under {Path(data_dir).resolve()}")
    rng = random.Random(seed)
    rng.shuffle(files)
    dataset = PackedDataset(
        [str(path) for path in files],
        n_chunks=8,
        block_size=block_size + 1,
        shuffle=True,
        seed=seed + fabric.global_rank,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
    )
    combined = CombinedDataset(datasets=[dataset], seed=seed, weights=[1.0])
    return DataLoader(combined, batch_size=batch_size, shuffle=False, pin_memory=True)


def _build_logger(args: argparse.Namespace) -> WandbLogger:
    kwargs = {
        "project": args.wandb_project,
        "name": args.exp_name,
        "id": args.exp_name,
        "save_dir": str(args.wandb_dir),
        "dir": str(args.wandb_dir),
        "version": args.exp_name,
        "group": args.exp_group,
    }
    if args.debug:
        kwargs["mode"] = "disabled"
    return WandbLogger(**kwargs)


def main(args: argparse.Namespace) -> bool:
    strategy_args = {
        "auto_wrap_policy": {Block},
        "state_dict_type": "full",
        "use_orig_params": True,
        "sync_module_states": True,
    }
    if args.nodes > 1:
        strategy_args["sharding_strategy"] = "HYBRID_SHARD"
    fabric = L.Fabric(
        # Slurm creates one process per GPU. Lightning still needs the complete
        # per-node topology so its SLURMEnvironment validation and LOCAL_RANK
        # device selection agree with --ntasks-per-node.
        devices=args.devices_per_node,
        num_nodes=args.nodes,
        strategy=FSDPStrategy(**strategy_args),
        precision="bf16-mixed",
        loggers=[_build_logger(args)],
    )
    fabric.launch()
    if fabric.world_size != args.recipe.world_size:
        raise RuntimeError(
            f"launched world size {fabric.world_size} does not match recipe "
            f"world size {args.recipe.world_size}"
        )
    fabric.seed_everything(args.seed)

    resume_record = None
    error = None
    if fabric.global_rank == 0:
        try:
            validate_resume_request(
                args.out_dir, resume=args.resume, run_manifest=args.run_manifest
            )
            if args.resume:
                resume_record = validate_checkpoint_commit(
                    args.out_dir,
                    expected_world_size=args.recipe.world_size,
                    expected_max_steps=args.max_steps,
                    expected_hparams_sha256=args.hparams_sha256,
                    expected_run_manifest_sha256=args.run_manifest_sha256,
                )
            if not args.resume:
                # validate_resume_request has already proved that an existing
                # directory contains only the immutable bootstrap artifacts.
                args.out_dir.mkdir(parents=True, exist_ok=True)
                args.wandb_dir.mkdir(parents=True, exist_ok=True)
                _snapshot_repository(args.out_dir)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = fabric.broadcast(error, src=0)
    if error is not None:
        raise RuntimeError(error)
    resume_record = fabric.broadcast(resume_record, src=0)
    fabric.barrier()

    config = Config.from_name(args.model_name)
    if config.block_size != args.recipe.sequence_length:
        raise ValueError(
            f"train_config sequence length {args.recipe.sequence_length} does not match "
            f"model block_size {config.block_size}"
        )
    if args.use_stream_tok:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = sys.maxsize
        train_dataloader = get_stateful_stream_tok_dataset(
            corpus_name=args.corpus_name,
            path=args.train_data_dir_raw,
            split="train",
            tokenizer=tokenizer,
            block_size=config.block_size + 1,
            rank=fabric.global_rank,
            world_size=fabric.world_size,
            batch_size=args.micro_batch_size,
            num_workers=args.train_num_workers,
        )
    else:
        train_dataloader = _create_packed_dataloader(
            data_dir=args.train_data_dir,
            batch_size=args.micro_batch_size,
            block_size=config.block_size,
            fabric=fabric,
            seed=args.seed,
        )
        train_dataloader = fabric.setup_dataloaders(train_dataloader)

    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
    mark_no_weight_decay(model)
    groups, group_manifest = build_adamw_param_groups(
        model, weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        fused=True,
    )
    model, optimizer = fabric.setup(model, optimizer)
    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": args.hparams,
        "iter_num": 0,
        "step_count": 0,
        "checkpoint_meta": {},
    }
    if resume_record is not None:
        resume_checkpoint = args.out_dir / resume_record["model"]["name"]
        fabric.load(resume_checkpoint, state)
        post_load_error = None
        if fabric.global_rank == 0:
            try:
                if _file_record(resume_checkpoint) != resume_record["model"]:
                    raise RuntimeError("model checkpoint changed during collective load")
            except Exception as exc:
                post_load_error = f"{type(exc).__name__}: {exc}"
        post_load_error = fabric.broadcast(post_load_error, src=0)
        if post_load_error is not None:
            raise RuntimeError(post_load_error)
        if state["hparams"] != args.hparams:
            raise RuntimeError("checkpoint training recipe differs from current arguments")
        if state["checkpoint_meta"] != resume_record["metadata"]:
            raise RuntimeError("model checkpoint generation does not match commit marker")
        if type(state["iter_num"]) is not int or type(state["step_count"]) is not int:
            raise TypeError("checkpoint iter_num and step_count must be integers")

    fabric.logger.log_hyperparams(args.hparams)
    fabric.print(
        f"Training {args.model_name}: world={fabric.world_size}, "
        f"micro_batch={args.micro_batch_size}, accumulation={args.gradient_accumulation_steps}, "
        f"max_steps={args.max_steps}, optimizer_groups="
        f"{len(group_manifest['decay'])}/{len(group_manifest['no_decay'])}"
    )
    return train(
        args=args,
        fabric=fabric,
        state=state,
        train_dataloader=train_dataloader,
        resume=args.resume,
        resume_record=resume_record,
    )


def train(
    *,
    args: argparse.Namespace,
    fabric: L.Fabric,
    state: dict[str, object],
    train_dataloader,
    resume: bool,
    resume_record: dict[str, object] | None,
) -> bool:
    model = state["model"]
    optimizer = state["optimizer"]
    if state["iter_num"] != state["step_count"] * args.gradient_accumulation_steps:
        raise RuntimeError(
            "checkpoint was saved between optimizer steps: "
            f"iter_num={state['iter_num']} step_count={state['step_count']} "
            f"accumulation={args.gradient_accumulation_steps}"
        )
    training_complete(step_count=state["step_count"], max_steps=args.max_steps)
    if (
        args.smoke_stop_after_step is not None
        and state["step_count"] >= args.smoke_stop_after_step
    ):
        raise RuntimeError(
            "smoke_stop_after_step must be later than the resumed optimizer step"
        )
    loss_fn = FusedCrossEntropyLoss()

    if resume and args.use_stream_tok:
        if resume_record is None:
            raise RuntimeError("resume checkpoint marker was not loaded")
        data_record = resume_record["data"][fabric.global_rank]
        data_path = args.out_dir / data_record["name"]
        local_data_valid = False
        try:
            local_data_valid = _file_record(data_path) == data_record
        except (OSError, RuntimeError):
            pass
        if not _all_ranks_true(local_data_valid, device=fabric.device):
            raise RuntimeError("per-rank data checkpoint changed before load")
        saved = torch.load(data_path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or set(saved) != {
            "metadata",
            "rank",
            "kdn_data_manifest",
            "state_dict",
            "rng_state",
        }:
            raise RuntimeError(f"rank {fabric.global_rank} data checkpoint schema changed")
        if saved["metadata"] != resume_record["metadata"]:
            raise RuntimeError(
                f"rank {fabric.global_rank} data generation does not match commit marker"
            )
        if saved["rank"] != fabric.global_rank:
            raise RuntimeError(f"rank {fabric.global_rank} loaded another rank's data state")
        expected_manifest = getattr(train_dataloader, "kdn_data_manifest", None)
        if saved.get("kdn_data_manifest") != expected_manifest:
            raise RuntimeError("streaming data manifest changed since checkpoint")
        train_dataloader.load_state_dict(saved["state_dict"])
        _restore_rng_state(saved["rng_state"], fabric.device)
    elif resume:
        raise RuntimeError("fail-closed resume is supported only for streaming-tokenizer data")
    if not _all_ranks_true(_run_manifest_unchanged(args), device=fabric.device):
        raise RuntimeError("run manifest changed during checkpoint restoration")

    def save_checkpoint(*, final: bool) -> None:
        if final:
            # Commit a resumable optimizer/data/RNG generation at the exact
            # terminal step before publishing the optimizer-free eval artifact.
            _save_latest_checkpoint(
                fabric=fabric,
                state=state,
                args=args,
                train_dataloader=train_dataloader,
            )
            _save_final_checkpoint(fabric=fabric, state=state, args=args)
        else:
            _save_latest_checkpoint(
                fabric=fabric,
                state=state,
                args=args,
                train_dataloader=train_dataloader,
            )

    if training_complete(step_count=state["step_count"], max_steps=args.max_steps):
        save_checkpoint(final=True)
        return True

    iterator = iter(train_dataloader)
    step_loss = torch.zeros((), device=fabric.device, dtype=torch.float32)
    started = time.monotonic()
    while not training_complete(step_count=state["step_count"], max_steps=args.max_steps):
        batch = _next_batch_or_raise(
            iterator,
            device=fabric.device,
            step_count=state["step_count"],
            max_steps=args.max_steps,
        )

        if args.use_stream_tok:
            input_ids = batch["input_ids"][:, : model.config.block_size].contiguous().to(fabric.device)
            targets = batch["labels"][:, 1 : model.config.block_size + 1].contiguous().to(fabric.device)
        else:
            input_ids = batch[:, : model.config.block_size].contiguous()
            targets = batch[:, 1 : model.config.block_size + 1].contiguous()

        lr = get_lr(
            step_count=state["step_count"],
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        next_iter = state["iter_num"] + 1
        accumulating = next_iter % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=accumulating):
            loss = loss_fn(model(input_ids), targets)
            finite = _all_ranks_true(bool(torch.isfinite(loss).item()), device=fabric.device)
            if not finite:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite loss at iter {state['iter_num']} step {state['step_count']}"
                )
            step_loss += loss.detach().float()
            fabric.backward(loss / args.gradient_accumulation_steps)
        state["iter_num"] = next_iter

        if accumulating:
            continue

        grad_norm = fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
        grad_norm_value = (
            float(grad_norm.detach().float().item())
            if torch.is_tensor(grad_norm)
            else float(grad_norm)
        )
        finite_grad = _all_ranks_true(math.isfinite(grad_norm_value), device=fabric.device)
        if not finite_grad:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite gradient norm at iter {state['iter_num']} "
                f"step {state['step_count']}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state["step_count"] += 1

        mean_loss = step_loss / args.gradient_accumulation_steps
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.SUM)
            mean_loss /= fabric.world_size
        metrics = step_metrics(
            iter_num=state["iter_num"],
            step_count=state["step_count"],
            loss=float(mean_loss.item()),
            learning_rate=lr,
            grad_norm=grad_norm_value,
            tokens_per_step=args.tokens_per_step,
        )
        if fabric.global_rank == 0:
            print("TRAIN_STEP_JSON=" + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
        fabric.log_dict(
            {
                "metric/train_loss": metrics["loss"],
                "metric/learning_rate": metrics["learning_rate"],
                "metric/grad_norm": metrics["grad_norm"],
                "metric/global_tokens": metrics["global_tokens"],
            },
            step=state["step_count"],
        )
        step_loss.zero_()

        time_limit_reached = _any_rank_true(
            bool(
                args.actual_train_seconds
                and time.monotonic() - started >= args.actual_train_seconds
            ),
            device=fabric.device,
        )
        action = checkpoint_action(
            step_count=state["step_count"],
            max_steps=args.max_steps,
            save_step_interval=args.save_step_interval,
            time_limit_reached=time_limit_reached,
            smoke_stop_after_step=args.smoke_stop_after_step,
        )
        if action == "latest":
            save_checkpoint(final=False)
        elif action == "final":
            save_checkpoint(final=True)
            return True
        elif action == "latest-stop":
            save_checkpoint(final=False)
            fabric.print("training time limit reached; latest checkpoint saved")
            return False
        elif action == "latest-smoke-stop":
            save_checkpoint(final=False)
            fabric.print(
                f"smoke stop reached at optimizer step {state['step_count']}; "
                "latest checkpoint saved"
            )
            return False

    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KDN language-model pretraining")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--wandb_dir", default=None)
    parser.add_argument("--train_data_dir", default="")
    parser.add_argument("--corpus_name", default="fineweb-edu-sample")
    parser.add_argument("--train_data_dir_raw", default="")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--exp_group", default="kdn_1.3B_100B")
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_manifest", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_stream_tok", action="store_true")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--min_lr", type=float, default=4e-5)
    parser.add_argument("--warmup_tokens", type=int, default=1_000_000_000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--smoke_stop_after_step", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--save_step_interval", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_num_workers", type=int, default=4)
    parser.add_argument("--actual_train_time", type=int, default=9_900)
    parser.add_argument("--micro_batch_size", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=None)
    parser.add_argument("--devices_per_node", type=int, default=None)
    parser.add_argument("--wandb_project", default="llm_next_gen")
    return parser


def configure_args(args: argparse.Namespace) -> argparse.Namespace:
    args.nodes = args.nnodes or int(os.environ.get("SLURM_NNODES", "1"))
    args.devices_per_node = args.devices_per_node or torch.cuda.device_count()
    if args.devices_per_node <= 0:
        raise RuntimeError("KDN training requires at least one visible CUDA device")
    args.recipe = resolve_training_recipe(
        args.train_config,
        nodes=args.nodes,
        devices_per_node=args.devices_per_node,
        micro_batch_size=args.micro_batch_size,
        max_steps=args.max_steps,
    )
    args.gradient_accumulation_steps = args.recipe.gradient_accumulation_steps
    args.tokens_per_step = args.recipe.tokens_per_step
    args.max_tokens = args.recipe.max_tokens
    args.max_steps = args.recipe.max_steps
    if args.smoke_stop_after_step is not None and not (
        0 < args.smoke_stop_after_step < args.max_steps
    ):
        raise ValueError(
            "smoke_stop_after_step must be positive and strictly less than max_steps"
        )
    args.warmup_steps = max(1, int(args.warmup_tokens) // args.recipe.tokens_per_step)
    args.actual_train_seconds = max(0, int(args.actual_train_time)) * 60
    args.out_dir, default_wandb = resolve_run_paths(
        args.output_root, args.train_config, args.exp_name
    )
    args.wandb_dir = Path(args.wandb_dir).resolve() if args.wandb_dir else default_wandb
    if args.run_manifest is not None:
        args.run_manifest = args.run_manifest.expanduser().resolve()
        if not args.run_manifest.is_file():
            raise FileNotFoundError(f"run manifest does not exist: {args.run_manifest}")
        args.run_manifest_sha256 = _sha256_file(args.run_manifest)
    else:
        args.run_manifest_sha256 = None
    if args.smoke_stop_after_step is not None:
        try:
            manifest = json.loads(args.run_manifest.read_text())
        except (AttributeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "smoke_stop_after_step requires a readable smoke run manifest"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "kdn-1.3b-training-v1"
            or not isinstance(manifest.get("run"), dict)
            or manifest["run"].get("smoke") is not True
        ):
            raise ValueError(
                "smoke_stop_after_step requires a run manifest declaring run.smoke=true"
            )
    if args.use_stream_tok:
        if not args.train_data_dir_raw:
            raise ValueError("--use_stream_tok requires --train_data_dir_raw")
    elif not args.train_data_dir:
        raise ValueError("packed training requires --train_data_dir")
    if args.save_step_interval <= 0:
        raise ValueError("save_step_interval must be positive")
    if args.warmup_tokens <= 0:
        raise ValueError("warmup_tokens must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.min_lr < 0 or args.min_lr > args.learning_rate:
        raise ValueError("min_lr must be between zero and learning_rate")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        raise ValueError("AdamW betas must be in [0, 1)")
    if args.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    args.hparams = {
        "model_name": args.model_name,
        "train_config": args.train_config,
        "exp_name": args.exp_name,
        "exp_group": args.exp_group,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "warmup_tokens": args.warmup_tokens,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "use_stream_tok": args.use_stream_tok,
        "corpus_name": args.corpus_name,
        "train_data_dir_raw": str(Path(args.train_data_dir_raw).resolve())
        if args.use_stream_tok
        else None,
        "train_data_dir": str(Path(args.train_data_dir).resolve())
        if not args.use_stream_tok
        else None,
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "train_num_workers": args.train_num_workers,
        "run_manifest_sha256": args.run_manifest_sha256,
        **asdict(args.recipe),
    }
    args.hparams["warmup_steps"] = args.warmup_steps
    args.hparams_sha256 = _json_sha256(args.hparams)
    return args


def cli(argv: list[str] | None = None) -> int:
    args = configure_args(build_parser().parse_args(argv))
    main(args)
    return 0


def validate_final_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate a committed final KDN checkpoint")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--run_manifest", type=Path)
    args = parser.parse_args(argv)
    marker = validate_final_checkpoint(
        args.out_dir,
        expected_world_size=args.world_size,
        expected_max_steps=args.max_steps,
        expected_accumulation_steps=args.gradient_accumulation_steps,
        expected_run_manifest_sha256=(
            _sha256_file(args.run_manifest) if args.run_manifest is not None else None
        ),
    )
    print(json.dumps(marker, sort_keys=True, allow_nan=False))
    return 0


def validate_bootstrap_restart_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an exact KDN pre-checkpoint restart directory"
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--run_manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_resume_request(
        args.out_dir,
        resume=False,
        run_manifest=args.run_manifest,
    )
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "bootstrap_restart": True}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-final-checkpoint":
        raise SystemExit(validate_final_cli(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--validate-bootstrap-restart":
        raise SystemExit(validate_bootstrap_restart_cli(sys.argv[2:]))
    else:
        mp.set_start_method("spawn")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        raise SystemExit(cli())
