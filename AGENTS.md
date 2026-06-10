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

## Baseline

- The RVQ6561 baseline is HuBERT content encoder + WavLM speaker encoder + Perceiver sampler.
- `token_mixer` is not part of this baseline unless a future experiment explicitly adds a speaker-token stack.
