#!/usr/bin/env bash
# Run the current SDPCodec HuBERT RVQ6561 baseline config.

#SBATCH --job-name=sdpcodec_hu_rvq6561
#SBATCH --partition=all
#SBATCH --account=hounsu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/hounsu/voice/sdpcodec/slurm_logs/%x-%j.out
#SBATCH --error=/data/hounsu/voice/sdpcodec/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hounsu/voice/sdpcodec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/hounsu/voice/sdpcodec/outputs}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-/data/hounsu/voice/sdpcodec/slurm_logs}"
DOCKER_IMAGE="${DOCKER_IMAGE:-hounsu_sdpcodec:260416}"
CONDA_ENV="${CONDA_ENV:-bigcodec}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/${CONDA_ENV}/bin/python}"
REF_ENV_SCRIPT="${REF_ENV_SCRIPT:-/home/hounsu/run_hubert_enc_vocosformer_24kHz_trixbase.sh}"

GPUS="${GPUS:-2}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
VAL_BATCH="${VAL_BATCH:-4}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-9}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-sdpcodec-hubert-rvq6561-current-b4ddp4acc1}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${OUTPUT_ROOT}/$(date +%F)/$(date +%H-%M-%S)-${RUN_NAME_SUFFIX}}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

export HF_HOME="${HF_HOME:-/data/hounsu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/hounsu/.cache}"
export TORCH_HOME="${TORCH_HOME:-/data/hounsu/.cache/torch}"
export S3PRL_CACHE_ROOT="${S3PRL_CACHE_ROOT:-${TORCH_HOME}/s3prl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/hounsu/.cache/matplotlib}"
export HF_HUB_ENABLE_PROGRESS_BARS="${HF_HUB_ENABLE_PROGRESS_BARS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONUNBUFFERED=1

extract_ref_env() {
  local name="$1"
  sed -nE "s/.*-e ${name}=([^[:space:]\\\\]+).*/\\1/p" "${REF_ENV_SCRIPT}" | head -1
}

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(extract_ref_env WANDB_API_KEY)}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$(extract_ref_env HUGGINGFACE_HUB_TOKEN)}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN}}"

if [[ -z "${WANDB_API_KEY}" || -z "${HUGGINGFACE_HUB_TOKEN}" ]]; then
  echo "[run] missing WANDB_API_KEY or HUGGINGFACE_HUB_TOKEN from ${REF_ENV_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${SLURM_LOG_ROOT}" "${OUTPUT_ROOT}" "${HF_HOME}" "${TORCH_HOME}" "${S3PRL_CACHE_ROOT}" "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

GPU_LIST="${SLURM_JOB_GPUS:-}"
DOCKER_GPU_ARGS=(--gpus all)
if [[ -n "${GPU_LIST}" ]]; then
  DOCKER_GPU_ARGS=(--gpus "\"device=${GPU_LIST}\"")
fi

UID_GID="$(id -u):$(id -g)"

ARGS=(
  hydra.run.dir="${HYDRA_RUN_DIR}"
  hydra.job.chdir=True
  log_dir=pl_log
  train.trainer.devices="${GPUS}"
  train.trainer.accumulate_grad_batches="${GRAD_ACCUM}"
  train.trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
  train.trainer.num_sanity_val_steps="${NUM_SANITY_VAL_STEPS}"
  dataset.train.batch_size="${PER_DEVICE_BATCH}"
  dataset.val.batch_size="${VAL_BATCH}"
  train.wandb_name="sdpcodec/hubert/MLS+LS/25Hz/rvq6561/nearbyrefclip/b4ddp4acc1/train6s/val6s/current"
)

if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_OVERRIDES}"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "[run] host=$(hostname)"
echo "[run] repo=${REPO_ROOT}"
echo "[run] git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[run] docker_image=${DOCKER_IMAGE}"
echo "[run] hydra_run_dir=${HYDRA_RUN_DIR}"
echo "[run] slurm_log_root=${SLURM_LOG_ROOT}"
echo "[run] gpus=${GPUS}"
echo "[run] global_batch=$((GPUS * PER_DEVICE_BATCH * GRAD_ACCUM)) (${GPUS} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM})"
echo "[run] docker_gpus=${DOCKER_GPU_ARGS[*]}"
echo "[run] extra_overrides=${EXTRA_OVERRIDES:-none}"

docker image inspect "${DOCKER_IMAGE}" >/dev/null
docker run --rm --user "${UID_GID}" -e HOME=/home/hounsu \
  -v /home/hounsu:/home/hounsu -v /data:/data -w "${REPO_ROOT}" \
  "${DOCKER_IMAGE}" "${PYTHON_BIN}" -c \
  "import torch, pytorch_lightning, hydra, datasets; print('env ok', torch.__version__)"

srun --ntasks=1 --unbuffered --cpu-bind=cores,verbose docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  --init \
  --user "${UID_GID}" \
  -e HOME=/home/hounsu \
  -e HF_HOME="${HF_HOME}" \
  -e HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
  -e HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  -e TORCH_HOME="${TORCH_HOME}" \
  -e S3PRL_CACHE_ROOT="${S3PRL_CACHE_ROOT}" \
  -e MPLCONFIGDIR="${MPLCONFIGDIR}" \
  -e WANDB_MODE="${WANDB_MODE}" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
  -e TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL}" \
  -e PYTHONUNBUFFERED=1 \
  -v /home/hounsu:/home/hounsu \
  -v /data:/data \
  -w "${REPO_ROOT}" \
  "${DOCKER_IMAGE}" \
  "${PYTHON_BIN}" -u -m sdpcodec.train "${ARGS[@]}"
