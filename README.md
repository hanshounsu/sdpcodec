# SDPCodec

Clean training entrypoint for the HuBERT RVQ6561 baseline:

`sdpcodec/hubert/MLS+LS/25Hz/rvq6561/nearbyrefclip/b4ddp4acc1/train6s/val6s`

The 2026-05-23/2026-05-24 output directories contain mixed stale Hydra files.
For this repo, the RVQ source of truth is the W&B/launch metadata from the
RVQ continuation, not the stale saved Hydra config.

## Run

```bash
cd /home/hounsu/voice/sdpcodec
./scripts/train_default.sh
```

Docker run with the current host uid/gid, so output files do not become
`root`/`nobody` on the host:

```bash
./scripts/train_docker.sh
```

Useful local overrides:

```bash
./scripts/train_default.sh train.wandb_enabled=false train.trainer.devices=1 train.trainer.max_steps=10
```

## Public Names

- `SdpCodecDataModule`: LibriSpeech + MLS dataloader wrapper.
- `SdpCodecLightningModule`: joint HuBERT/content/F0 codec training module.
- `SdpCodecSystem`: alias for `SdpCodecLightningModule`.
- `sdpcodec_hubert_rvq6561.yaml`: the single baseline config.

## Config Surface

The default YAML now uses the implementation keys directly. There is no runtime
translation layer from selector/list aliases to old BigCodec keys.

- `model.speaker_encoder.speaker_encoder_type=wavlm`
- `model.speaker_encoder.use_perceiver_encoder=true`
- `model.speaker_encoder.use_memory_cattn=false`
- `model.codec_encoder.use_hubert=true`
- `model.codec_decoder.quantizer_type=rvq`
- `model.codec_decoder.spk_cond.type=[mhca]`
- `model.f0_codec.spk_cond.type=[mhca, concat]`

The original RVQ6561 source run used the WavLM speaker encoder with the
Perceiver sampler only. `token_mixer` is a later speaker-token stack stage and is
not part of this baseline config.

## Checkpoints

Checkpoint provenance and the imported ckpt files live in:

`/data/hounsu/voice/sdpcodec/outputs/imported_checkpoints/rvq6561_baseline`

The import helper rewrites Lightning `hyper_parameters.cfg` to the current
project config:

```bash
./scripts/import_rvq6561_checkpoints.py
```

Default ACLs have been applied to this project and the BigCodec output tree.
To reapply them after a migration or copied directory:

```bash
./scripts/fix_output_permissions.sh
```
