#!/usr/bin/env bash
# Run SDPCodec HuBERT RVQ6561 with compact 2-component discretized FCPE mixture loss.

#SBATCH --job-name=sdpcodec_mix2
#SBATCH --partition=all
#SBATCH --account=hounsu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=52G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/hounsu/voice/sdpcodec/slurm_logs/%x-%j.out
#SBATCH --error=/data/hounsu/voice/sdpcodec/slurm_logs/%x-%j.err

set -euo pipefail

export CONFIG_NAME="${CONFIG_NAME:-sdpcodec_hubert_rvq6561_fcpe_dist_mix2}"
export FCPE_RUN_SUFFIX="${FCPE_RUN_SUFFIX:-mix2_asym_cdf01}"
export REPO_ROOT="${REPO_ROOT:-/home/hounsu/voice/sdpcodec}"

exec bash "${REPO_ROOT}/scripts/slurm_sdpcodec_hubert_rvq6561_fcpe_dist_4090.sbatch"
