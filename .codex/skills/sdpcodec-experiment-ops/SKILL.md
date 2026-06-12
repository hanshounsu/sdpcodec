---
name: sdpcodec-experiment-ops
description: Use only inside the SDPCodec repo for experiment lookup, SLURM launcher creation or editing, W&B/Hydra run-name checks, metric/log diagnosis, and launch/resume triage.
---

# SDPCodec Experiment Ops

This is a repo-local SDPCodec skill. Do not install it globally.

Use this whenever the user asks about an SDPCodec experiment name, W&B run,
Hydra run dir, checkpoint, launcher, SLURM run, resume, or metric/log diagnosis.

## Hard Rule: W&B Names Must Match Config

- W&B names must always reflect the actual live config. A wrong W&B name is a
  blocking bug, not a cosmetic issue.
- For every SLURM launcher you create or edit, derive the batch slug from the
  launcher variables:
  `b${PER_DEVICE_BATCH}ddp${GPUS}acc${GRAD_ACCUM}`.
- Pass `train.wandb_name` explicitly from the computed name. Do not rely on a
  copied config default when the launcher overrides GPU count, per-device batch,
  or gradient accumulation.
- If `WANDB_RUN_NAME` is supplied from the environment, reject it unless it
  starts with the launcher's computed semantic prefix and batch slug. Never
  silently launch with a stale tag such as `b4ddp4acc1` when the runtime config
  is `b4ddp2acc2`.
- Before reporting a run, cross-check `train.wandb_name` against
  `hydra/overrides.yaml`, `hydra/config.yaml`, launcher stdout lines such as
  `global_batch=...`, and W&B metadata. If they disagree, call out the mismatch
  and use the Hydra/runtime values as source of truth.

## Launch / Resume Checks

- Do not launch, resume, cancel, or replace an experiment unless the user asks
  for that exact action in the current conversation.
- For fresh SLURM training, run mandatory debug before main training.
- Keep SDPCodec manual optimization semantics straight: real accumulation is
  `train.gradient_accumulation_steps`; keep
  `train.trainer.accumulate_grad_batches=1` unless the user explicitly asks for
  a different Lightning behavior.
- Effective training batch is
  `GPU count x dataset.train.batch_size x train.gradient_accumulation_steps`.
- Before resume, verify the exact `run_dir`, `pl_log/last.ckpt`,
  `hydra/overrides.yaml`, and runtime logs. Do not relaunch from scratch as a
  substitute for a missing checkpoint without explicit approval.

## Bounded Lookup

- W&B display names are labels, not filesystem paths.
- Use the final timestamp/hash slug to locate run dirs under
  `/data/hounsu/voice/sdpcodec/outputs`.
- If a run or checkpoint is not found after checking the likely output dir,
  Hydra files, W&B local files, and relevant SLURM logs, say where you checked
  and stop. Do not start broad unbounded filesystem searches.
