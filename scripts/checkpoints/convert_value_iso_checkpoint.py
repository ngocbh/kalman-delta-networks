#!/usr/bin/env python3
"""Convert a research value-isotropic KDN checkpoint to the released schema.

The released layer is the research layer with two parameters renamed; nothing
is fused, reshaped, or reordered, so the conversion is a pure key rename:

    *.qn_proj.*  ->  *.omega_proj.*
    *.mu_param   ->  *.initial_precision_param

Optimizer state is keyed by parameter position, which a rename does not
disturb, so it is carried across untouched.

Scope: this produces weights loadable by the released model for evaluation,
inference, or export. It does NOT produce a checkpoint that `pretrain.py
--resume` will accept -- research checkpoints predate this repo's root schema
(they carry no `checkpoint_meta`) and resuming additionally needs the per-rank
data states and the run-manifest/marker protocol. The script reports any root
key delta it finds.

The script is read-only by default; it reports what it would do. Pass --out to
write a converted copy. It never writes in place and never overwrites.

    python scripts/checkpoints/convert_value_iso_checkpoint.py \\
      --checkpoint /path/to/final-model-ckpt.pth

    python scripts/checkpoints/convert_value_iso_checkpoint.py \\
      --checkpoint /path/to/final-model-ckpt.pth \\
      --out /path/to/final-model-ckpt-released.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.qn_proj\."), ".omega_proj."),
    (re.compile(r"\.mu_param$"), ".initial_precision_param"),
)

# Research arm name -> released config name.
MODEL_NAMES = {
    "value_iso_kdn_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit": "value_iso_kdn_1.3B",
    "swa_value_iso_kdn_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit": "swa_value_iso_kdn_1.3B",
}

# Root keys this repo's pretrain.py requires of a resumable checkpoint.
RELEASE_ROOT_KEYS = frozenset(
    {"model", "optimizer", "hparams", "iter_num", "step_count", "checkpoint_meta"}
)


def rename_key(key: str) -> str:
    for pattern, replacement in RENAMES:
        key = pattern.sub(replacement, key)
    return key


def rename_state(state: dict) -> tuple[dict, int]:
    renamed: dict = {}
    changed = 0
    for key, value in state.items():
        new_key = rename_key(key)
        if new_key != key:
            changed += 1
        if new_key in renamed:
            raise RuntimeError(f"rename collision on {new_key!r}")
        renamed[new_key] = value
    if len(renamed) != len(state):
        raise RuntimeError("rename changed the parameter count")
    return renamed, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the converted checkpoint here; omit for a read-only report",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after conversion, load the state dict into the released config "
        "with strict=True (requires CUDA)",
    )
    args = parser.parse_args()

    source = args.checkpoint
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"checkpoint is missing or not a regular file: {source}")
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite existing path: {args.out}")
        if args.out.resolve() == source.resolve():
            raise SystemExit("refusing to convert in place; choose a different --out")

    print(f"reading {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise SystemExit("checkpoint root is not a dict containing 'model'")

    model_state, changed = rename_state(dict(checkpoint["model"]))
    print(f"  model tensors: {len(model_state)} ({changed} renamed)")
    if changed == 0:
        print("  nothing to rename -- this checkpoint already uses the released schema")

    missing_root = sorted(RELEASE_ROOT_KEYS - set(checkpoint))
    if missing_root:
        print(
            f"  note: root keys absent for pretrain.py --resume: {missing_root}. "
            "The converted weights load into the released model, but this file "
            "is not a resumable release checkpoint."
        )

    hparams = checkpoint.get("hparams")
    if isinstance(hparams, dict):
        old_name = hparams.get("model_name")
        new_name = MODEL_NAMES.get(old_name)
        if new_name is not None:
            print(f"  model_name: {old_name} -> {new_name}")
            hparams = {**hparams, "model_name": new_name}
        elif old_name is not None:
            print(f"  model_name left as-is: {old_name}")

    if args.out is None:
        print("\nread-only report; pass --out PATH to write the converted checkpoint")
        for key in sorted(model_state)[:6]:
            print(f"    {key}")
        return 0

    converted = {**checkpoint, "model": model_state}
    if isinstance(hparams, dict):
        converted["hparams"] = hparams

    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    if temporary.exists():
        raise SystemExit(f"refusing to reuse existing temporary: {temporary}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, temporary)
    temporary.replace(args.out)
    print(f"wrote {args.out}")

    if args.verify:
        from lit_gpt.config import Config
        from lit_gpt.model import GPT

        name = (converted.get("hparams") or {}).get("model_name", "value_iso_kdn_1.3B")
        print(f"verifying strict load into {name}")
        model = GPT(Config.from_name(name))
        model.load_state_dict(model_state, strict=True)
        print("  strict load OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
