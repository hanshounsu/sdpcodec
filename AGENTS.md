# SDPCodec Agent Notes

Repo-specific procedural knowledge lives in `.codex/skills/`. Use
`sdpcodec-experiment-ops` for experiment lookup, SLURM launcher creation or
editing, W&B/Hydra run-name checks, launch/resume triage, and metric/log
diagnosis. Do not install SDPCodec-specific knowledge as a global Codex skill.

## Artifact Placement

- Keep source code, configs, docs, and small manifests in `/home/hounsu/voice/sdpcodec`.
- Do not store large generated artifacts in the repo tree.
- Checkpoints, training outputs, logs, W&B runs, generated audio, and bulk evaluation artifacts belong under:

  `/data/hounsu/voice/sdpcodec/outputs`

- Imported reference checkpoints should go under:

  `/data/hounsu/voice/sdpcodec/outputs/imported_checkpoints`

## Permissions

- Prefer running Docker jobs with the host uid/gid, for example `scripts/train_docker.sh`, so files do not become `root` or `nobody` on the host.
- If copied or migrated files become unreadable, reapply ACLs with:

  `./scripts/fix_output_permissions.sh`

## SLURM Resources

- For SLURM-managed GPU jobs, request CPU and memory by GPU count:
  CPU = `16 x GPU count`, RAM = `26G x GPU count`.
- For example, a 2-GPU job should request `--cpus-per-task=32` and
  `--mem=52G`.
- SDPCodec uses manual optimization; its real gradient accumulation is
  `train.gradient_accumulation_steps`, not Lightning's trainer accumulation.
  Keep SDPCodec effective train batch size at least 16:
  `GPU count x dataset.train.batch_size x train.gradient_accumulation_steps >= 16`.
  For the standard 2-GPU, per-device batch 4 launch, set
  `train.gradient_accumulation_steps=2` and keep
  `train.trainer.accumulate_grad_batches=1`.
- W&B names must always match the live experiment config. This is a hard
  requirement, not a cosmetic label. For every new or edited SLURM launcher,
  derive the batch slug from the actual runtime variables:
  `b${PER_DEVICE_BATCH}ddp${GPUS}acc${GRAD_ACCUM}`. Do not hard-code an old
  `b*ddp*acc*` tag, and do not trust a copied W&B name until it is checked
  against `dataset.train.batch_size`, `train.trainer.devices`, and
  `train.gradient_accumulation_steps`.
- If `WANDB_RUN_NAME` is user-provided or inherited from the environment, the
  launcher must reject it unless it matches the computed semantic prefix and
  batch slug for that launcher. Never launch a run with a stale W&B name such
  as `b4ddp4acc1` when the actual runtime is `b4ddp2acc2`.
- Before reporting or comparing an experiment, treat `train.wandb_name` as a
  label that must be cross-checked against `hydra/overrides.yaml`,
  `hydra/config.yaml`, launcher stdout lines such as `global_batch=...`, and
  W&B metadata. If the label and config disagree, report the mismatch clearly
  and use the config/logged runtime values as the source of truth.
- Never launch, submit, resume, restart, or otherwise start any experiment
  unless the user explicitly asks to run that exact action in the current
  conversation. Editing configs/scripts, answering questions, checking status,
  or discussing settings is not permission to run.
- Never cancel, stop, replace, requeue, preempt, or otherwise interrupt an
  existing experiment unless the user explicitly asks to stop that exact job.
  This is stricter than ordinary debugging convenience: do not cancel a run in
  order to apply a config change, even if the current run is wrong, unless the
  user says to cancel it.
- If a requested resume has no valid checkpoint, state that exact resume is not
  possible and list the checked paths. Do not relaunch from scratch, reuse a
  debug checkpoint, or submit a "closest" replacement unless the user explicitly
  approves that fallback.
- Every fresh SLURM training experiment must run mandatory debug before main
  training. This is not optional agent discretion. Debug must execute the
  smallest practical train+val path: a few train steps, one or a few
  non-sanity validation batches, normal validation metric stack enabled, W&B
  disabled. For MLS+LS/all experiments, debug should use `dataset.name=librispeech`
  unless the user specifically asks to debug the full mixed dataset.
  Main training may start only after debug succeeds.
- Do not bypass debug with `RUN_DEBUG=False` or equivalent unless the user
  explicitly asks for that specific run to skip it.
- SLURM launchers must use writable cache paths for validation metrics. Avoid
  shared root-owned torch lock/trust files by setting `TORCH_HOME` and
  `S3PRL_CACHE_ROOT` under a repo/output cache such as
  `/data/hounsu/voice/sdpcodec/cache/torch`.
- When tracking a launch, do not report success after process start alone.
  Verify that debug reached non-sanity validation and that main training
  started cleanly.

## Baseline

- The RVQ6561 baseline is HuBERT content encoder + WavLM speaker encoder + Perceiver sampler.
- `token_mixer` is not part of this baseline unless a future experiment explicitly adds a speaker-token stack.
