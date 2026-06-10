# RVQ6561 Source Note

Baseline target:

`trixbase/hubert/MLS+LS/25Hz/cb6561/nearbyrefclip/b4ddp4acc1/train6s/val6s/2026-05-24-18-03-54`

The old output directories mix stale Hydra files and RVQ continuation logs.
Do not treat these saved Hydra configs as the source of truth:

- `/data/hounsu/voice/bigcodec/outputs/_home_migrated_2026-06-01/2026-05-23/18-03-54/hydra/config.yaml`
- `/data/hounsu/voice/bigcodec/outputs/_home_migrated_2026-06-01/2026-05-24/mislabeled_fsq_hydra_rvq6561-resume-18-03-54/hydra/config.yaml`

The RVQ continuation metadata records the intended overrides:

- `model.codec_decoder.quantizer_type=rvq`
- `model.codec_decoder.codebook_size=6561`
- `dataset.train.batch_size=4`
- `train.trainer.devices=4`
- `train.gradient_accumulation_steps=1`
- `dataset.train_ref_clip_mode=nearby`

Corresponding metadata path:

`/data/hounsu/voice/bigcodec/outputs/_home_migrated_2026-06-01/2026-05-24/mislabeled_fsq_hydra_rvq6561-resume-18-03-54/logs/wandb/run-20260524_020053-rvq-resume-3074/files/wandb-metadata.json`

The clean config in this repo is:

`configs/sdpcodec_hubert_rvq6561.yaml`
