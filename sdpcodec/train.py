"""Train the SDPCodec HuBERT RVQ6561 baseline."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from sdpcodec.config_compat import expand_legacy_config
from sdpcodec.data import SdpCodecDataModule
from sdpcodec.system import SdpCodecLightningModule


class SamplerStateCheckpointCallback(pl.Callback):
    """Save DataModule sampler state so mid-epoch resumes stay aligned."""

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        sampler = getattr(datamodule, "_train_sampler", None)
        if sampler is None:
            return
        try:
            checkpoint["datamodule_sampler_state"] = sampler.state_dict()
        except Exception:
            return


class RankZeroProgressBar(TQDMProgressBar):
    """Only show the progress bar on rank 0."""

    @property
    def _is_enabled(self) -> bool:
        return self.trainer.global_rank == 0


def _is_main_process() -> bool:
    for key in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return int(value) == 0
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def _device_count(devices: Any) -> int:
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str):
        if devices.strip().isdigit():
            return int(devices)
        if "," in devices:
            return len([item for item in devices.split(",") if item.strip()])
    return 1


def _configure_reproducibility(cfg: Any) -> None:
    seed_everything(int(getattr(cfg, "seed", 1024)), workers=True)

    deterministic = bool(getattr(cfg, "deterministic", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    train_cfg = getattr(cfg, "train", None)
    if train_cfg is None:
        return

    matmul_precision = getattr(train_cfg, "float32_matmul_precision", None)
    if torch.cuda.is_available() and matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    allow_tf32 = getattr(train_cfg, "allow_tf32", None)
    if allow_tf32 is not None and not deterministic:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)


def _hydra_run_dir() -> tuple[Path, str]:
    try:
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
        config_name = str(hydra_cfg.job.config_name)
    except Exception:
        run_dir = Path.cwd()
        config_name = "sdpcodec_hubert_rvq6561"
    return run_dir.resolve(), config_name


def _timestamp_slug(run_dir: Path) -> str:
    return f"{run_dir.parent.name}-{run_dir.name}"


def _prepare_log_dir(cfg: Any, run_dir: Path) -> Path:
    log_dir = Path(str(cfg.log_dir))
    if not log_dir.is_absolute():
        log_dir = run_dir / log_dir
    with open_dict(cfg):
        cfg.log_dir = str(log_dir)
    return log_dir


def _save_resolved_config(cfg: Any, run_dir: Path, config_name: str) -> None:
    if not _is_main_process():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / f"{config_name}.yaml")
    hydra_dir = run_dir / "hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / f"{config_name}.yaml")


def _wandb_name(cfg: Any, ts_slug: str) -> str:
    base_name = str(getattr(cfg.train, "wandb_name", "sdpcodec/hubert/rvq6561")).strip("/")
    if getattr(cfg.train, "wandb_append_timestamp", True):
        return f"{base_name}/{ts_slug}"
    return base_name


def _build_logger(cfg: Any, run_dir: Path, ts_slug: str):
    enabled = bool(getattr(cfg.train, "wandb_enabled", True))
    enabled = enabled and os.environ.get("WANDB_MODE", "").lower() != "disabled"
    if not enabled:
        return False

    run_id = str(getattr(cfg, "id", None) or ts_slug)
    return WandbLogger(
        save_dir=str(run_dir / "logs"),
        name=_wandb_name(cfg, ts_slug),
        project=str(getattr(cfg.train, "wandb_project", "SDPCodec")),
        offline=bool(getattr(cfg.train, "wandb_offline", False) or os.environ.get("WANDB_MODE", "").lower() == "offline"),
        id=run_id,
        resume="allow" if getattr(cfg, "ckpt", None) else None,
    )


def _build_callbacks(cfg: Any, wandb_enabled: bool) -> list[pl.Callback]:
    callbacks: list[pl.Callback] = []
    if wandb_enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    checkpointing = bool(getattr(cfg.train.trainer, "enable_checkpointing", True))
    if checkpointing:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=str(cfg.log_dir),
                save_top_k=3,
                save_last=True,
                monitor="val_stats/stoi",
                mode="max",
                filename="step={val_stats/total_step:07}-stoi={val_stats/stoi:.4f}",
                auto_insert_metric_name=False,
            ),
        )
        callbacks.append(SamplerStateCheckpointCallback())
    return callbacks


def _trainer_kwargs(cfg: Any, callbacks: list[pl.Callback]) -> dict[str, Any]:
    kwargs = dict(OmegaConf.to_container(cfg.train.trainer, resolve=True))
    kwargs.setdefault("deterministic", bool(getattr(cfg, "deterministic", False)))

    if _device_count(kwargs.get("devices", 1)) > 1:
        kwargs["strategy"] = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=bool(getattr(cfg.train, "ddp_find_unused_parameters", True)),
            static_graph=bool(getattr(cfg.train, "ddp_static_graph", False)),
            gradient_as_bucket_view=bool(getattr(cfg.train, "ddp_gradient_as_bucket_view", False)),
        )
        if bool(kwargs.get("enable_progress_bar", True)):
            callbacks.insert(0, RankZeroProgressBar())
    return kwargs


@hydra.main(version_base=None, config_path="../configs", config_name="sdpcodec_hubert_rvq6561")
def train(cfg: Any) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via",
        category=UserWarning,
        module=r"torch\.distributed\.distributed_c10d",
    )

    _configure_reproducibility(cfg)
    run_dir, config_name = _hydra_run_dir()
    ts_slug = _timestamp_slug(run_dir)
    _prepare_log_dir(cfg, run_dir)
    _save_resolved_config(cfg, run_dir, config_name)

    logger = _build_logger(cfg, run_dir, ts_slug)
    runtime_cfg = expand_legacy_config(cfg)
    callbacks = _build_callbacks(runtime_cfg, wandb_enabled=bool(logger))
    kwargs = _trainer_kwargs(runtime_cfg, callbacks)

    datamodule = SdpCodecDataModule(runtime_cfg)
    model = SdpCodecLightningModule(runtime_cfg)
    trainer = pl.Trainer(**kwargs, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=datamodule, ckpt_path=getattr(runtime_cfg, "ckpt", None))


if __name__ == "__main__":
    train()
