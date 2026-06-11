# SDPCodec Agent Notes

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
