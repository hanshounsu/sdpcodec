#!/usr/bin/env python3
"""Import the RVQ6561 baseline checkpoints into this project.

The script copies the three selected checkpoint files and replaces the saved
Lightning hyperparameter config with the current project config. State-dict keys
are preserved because the SDPCodec module attribute names are intentionally the
same as the source run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sdpcodec_hubert_rvq6561.yaml"
DEFAULT_DEST = Path("/data/hounsu/voice/sdpcodec/outputs/imported_checkpoints/rvq6561_baseline")
SOURCES = (
    (
        "last.ckpt",
        Path("/data/hounsu/voice/bigcodec/outputs/_home_migrated_2026-06-01/2026-05-24/mislabeled_fsq_hydra_rvq6561-resume-18-03-54/pl_log/last.ckpt"),
    ),
    (
        "step=460000.0-stoi=0.8556.ckpt",
        Path("/data/hounsu/voice/bigcodec/outputs/_home_migrated_2026-06-01/2026-05-24/mislabeled_fsq_hydra_rvq6561-resume-18-03-54/pl_log/step=460000.0-stoi=0.8556.ckpt"),
    ),
    (
        "step=470000.0-stoi=0.8553.ckpt",
        Path("/data/hounsu/voice/bigcodec/outputs/_home_migrated_2026-06-01/2026-05-24/mislabeled_fsq_hydra_rvq6561-resume-18-03-54/pl_log/step=470000.0-stoi=0.8553.ckpt"),
    ),
)


def import_checkpoint(src: Path, dst: Path, cfg) -> None:
    import torch

    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    hyper_parameters = ckpt.setdefault("hyper_parameters", {})
    if isinstance(hyper_parameters, dict):
        hyper_parameters["cfg"] = cfg
    torch.save(ckpt, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--raw-copy",
        action="store_true",
        help="copy bytes without rewriting Lightning hyper_parameters.cfg",
    )
    args = parser.parse_args()

    cfg = None
    if not args.raw_copy:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
    args.dest.mkdir(parents=True, exist_ok=True)

    for name, src in SOURCES:
        dst = args.dest / name
        if args.raw_copy:
            shutil.copy2(src, dst)
        else:
            import_checkpoint(src, dst, cfg)
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
