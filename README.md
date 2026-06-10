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

Useful local overrides:

```bash
./scripts/train_default.sh train.wandb_enabled=false train.trainer.devices=1 train.trainer.max_steps=10
```

## Public Names

- `SdpCodecDataModule`: LibriSpeech + MLS dataloader wrapper.
- `SdpCodecLightningModule`: joint HuBERT/content/F0 codec training module.
- `SdpCodecSystem`: alias for `SdpCodecLightningModule`.
- `sdpcodec_hubert_rvq6561.yaml`: the single baseline config.

The copied `vq`, `module`, `criterions`, `common`, `metrics`, and `ptl`
directories are compatibility implementation code from BigCodec. The public API
above is the new surface.
