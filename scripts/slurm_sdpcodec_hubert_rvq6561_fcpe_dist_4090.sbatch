#!/usr/bin/env bash
# Run SDPCodec HuBERT RVQ6561 with raw FCPE distribution target loss.

#SBATCH --job-name=sdpcodec_fcpe_dist
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

REPO_ROOT="${REPO_ROOT:-/home/hounsu/voice/sdpcodec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/hounsu/voice/sdpcodec/outputs}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-/data/hounsu/voice/sdpcodec/slurm_logs}"
CACHE_ROOT="${CACHE_ROOT:-/data/hounsu/voice/sdpcodec/cache}"
DOCKER_IMAGE="${DOCKER_IMAGE:-hounsu_sdpcodec:260416}"
CONDA_ENV="${CONDA_ENV:-bigcodec}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/${CONDA_ENV}/bin/python}"
REF_ENV_SCRIPT="${REF_ENV_SCRIPT:-/home/hounsu/run_hubert_enc_vocosformer_24kHz_trixbase.sh}"
CONFIG_NAME="${CONFIG_NAME:-sdpcodec_hubert_rvq6561_fcpe_dist}"
FCPE_RUN_SUFFIX="${FCPE_RUN_SUFFIX:-wbce_a2}"

GPUS="${GPUS:-2}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
VAL_BATCH="${VAL_BATCH:-4}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-9}"
RUN_DEBUG="${RUN_DEBUG:-True}"
DEBUG_TRAIN_STEPS="${DEBUG_TRAIN_STEPS:-2}"
DEBUG_MAX_STEPS="${DEBUG_MAX_STEPS:-${DEBUG_TRAIN_STEPS}}"
DEBUG_VAL_CHECK_INTERVAL="${DEBUG_VAL_CHECK_INTERVAL:-1}"
DEBUG_VAL_BATCHES="${DEBUG_VAL_BATCHES:-1}"
DEBUG_SANITY_VAL_STEPS="${DEBUG_SANITY_VAL_STEPS:-0}"
DEBUG_DATASET_NAME="${DEBUG_DATASET_NAME:-librispeech}"
RUN_HASH="${RUN_HASH:-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')}"
RUN_STAMP="${RUN_STAMP:-$(date +%F-%H-%M-%S)-${RUN_HASH}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${OUTPUT_ROOT}/${RUN_STAMP}}"
BATCH_TAG="b${PER_DEVICE_BATCH}ddp${GPUS}acc${GRAD_ACCUM}"
EXPECTED_WANDB_PREFIX="sdpcodec/hubert/MLS+LS/25Hz/rvq6561/nearbyrefclip/${BATCH_TAG}/train6s/val6s/normf0_in/fcpe_dist_out/raw/${FCPE_RUN_SUFFIX}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPECTED_WANDB_PREFIX}/${RUN_STAMP}}"
if [[ "${WANDB_RUN_NAME}" != "${EXPECTED_WANDB_PREFIX}/"* ]]; then
  echo "[run] WANDB_RUN_NAME must start with computed prefix ${EXPECTED_WANDB_PREFIX}/: ${WANDB_RUN_NAME}" >&2
  exit 2
fi
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

export HF_HOME="${HF_HOME:-/data/hounsu/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export S3PRL_CACHE_ROOT="${S3PRL_CACHE_ROOT:-${TORCH_HOME}/s3prl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export HF_HUB_ENABLE_PROGRESS_BARS="${HF_HUB_ENABLE_PROGRESS_BARS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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

mkdir -p "${SLURM_LOG_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${HF_HOME}" "${TORCH_HOME}" "${S3PRL_CACHE_ROOT}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cd "${REPO_ROOT}"

GPU_LIST="${SLURM_JOB_GPUS:-}"
DOCKER_GPU_ARGS=(--gpus all)
if [[ -n "${GPU_LIST}" ]]; then
  DOCKER_GPU_ARGS=(--gpus "\"device=${GPU_LIST}\"")
fi

UID_GID="$(id -u):$(id -g)"
CONTAINER_PREFIX="sdpcodec_fcpedist_${SLURM_JOB_ID:-manual}"

cleanup_containers() {
  docker rm -f "${CONTAINER_PREFIX}_debug" "${CONTAINER_PREFIX}_main" >/dev/null 2>&1 || true
}
trap cleanup_containers EXIT INT TERM

ARGS=(
  --config-name "${CONFIG_NAME}"
  hydra.run.dir="${HYDRA_RUN_DIR}"
  hydra.job.chdir=True
  log_dir=pl_log
  train.trainer.devices="${GPUS}"
  train.trainer.accumulate_grad_batches=1
  train.gradient_accumulation_steps="${GRAD_ACCUM}"
  train.trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
  train.trainer.num_sanity_val_steps="${NUM_SANITY_VAL_STEPS}"
  dataset.train.batch_size="${PER_DEVICE_BATCH}"
  dataset.val.batch_size="${VAL_BATCH}"
  train.wandb_project="BigCodec"
  train.wandb_name="${WANDB_RUN_NAME}"
)

if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_OVERRIDES}"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "[run] host=$(hostname)"
echo "[run] repo=${REPO_ROOT}"
echo "[run] git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[run] docker_image=${DOCKER_IMAGE}"
echo "[run] config=${CONFIG_NAME}"
echo "[run] hydra_run_dir=${HYDRA_RUN_DIR}"
echo "[run] wandb_project=BigCodec"
echo "[run] wandb_name=${WANDB_RUN_NAME}"
echo "[run] batch_tag=${BATCH_TAG}"
echo "[run] expected_wandb_prefix=${EXPECTED_WANDB_PREFIX}"
echo "[run] slurm_log_root=${SLURM_LOG_ROOT}"
echo "[run] torch_home=${TORCH_HOME}"
echo "[run] s3prl_cache_root=${S3PRL_CACHE_ROOT}"
echo "[run] gpus=${GPUS}"
echo "[run] global_batch=$((GPUS * PER_DEVICE_BATCH * GRAD_ACCUM)) (${GPUS} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM})"
echo "[run] docker_gpus=${DOCKER_GPU_ARGS[*]}"
echo "[run] run_debug=${RUN_DEBUG}"
echo "[run] debug_dataset_name=${DEBUG_DATASET_NAME}"
echo "[run] debug_train_steps=${DEBUG_TRAIN_STEPS}"
echo "[run] debug_max_steps=${DEBUG_MAX_STEPS}"
echo "[run] debug_val_check_interval=${DEBUG_VAL_CHECK_INTERVAL}"
echo "[run] debug_val_batches=${DEBUG_VAL_BATCHES}"
echo "[run] extra_overrides=${EXTRA_OVERRIDES:-none}"

docker image inspect "${DOCKER_IMAGE}" >/dev/null
docker run --rm --user "${UID_GID}" -e HOME=/home/hounsu \
  -v /home/hounsu:/home/hounsu -v /data:/data -w "${REPO_ROOT}" \
  "${DOCKER_IMAGE}" "${PYTHON_BIN}" -c \
  "import torch, pytorch_lightning, hydra, datasets; print('env ok', torch.__version__)"

run_train() {
  local phase="$1"
  shift
  srun --ntasks=1 --unbuffered --cpu-bind=cores,verbose docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  --init \
  --shm-size 64G \
  --name "${CONTAINER_PREFIX}_${phase}" \
  --user "${UID_GID}" \
  -e HOME=/home/hounsu \
  -e HF_HOME="${HF_HOME}" \
  -e HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
  -e HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
  -e HUGGINGFACE_HUB_TOKEN \
  -e HF_TOKEN \
  -e XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  -e TORCH_HOME="${TORCH_HOME}" \
  -e S3PRL_CACHE_ROOT="${S3PRL_CACHE_ROOT}" \
  -e MPLCONFIGDIR="${MPLCONFIGDIR}" \
  -e WANDB_MODE="${WANDB_MODE}" \
  -e WANDB_API_KEY \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  -e TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL}" \
  -e PYTHONUNBUFFERED=1 \
  -v /home/hounsu:/home/hounsu \
  -v /data:/data \
  -w "${REPO_ROOT}" \
  "${DOCKER_IMAGE}" \
  "${PYTHON_BIN}" -u -m sdpcodec.train "$@"
}

verify_debug_log() {
  local log_path="${SLURM_LOG_ROOT}/${SLURM_JOB_NAME:-sdpcodec_fcpe_dist}-${SLURM_JOB_ID}.out"
  if [[ ! -f "${log_path}" ]]; then
    echo "[debug] cannot verify log; missing ${log_path}" >&2
    exit 2
  fi
  if grep -Eq 'Traceback|RuntimeError|PermissionError|Error executing job|CUDA out of memory|Bus error|SIGBUS|Warning: disabling UTMOS' "${log_path}"; then
    echo "[debug] fatal/error pattern found in ${log_path}; refusing main training" >&2
    exit 2
  fi
  if ! grep -Eq 'Validation:' "${log_path}"; then
    echo "[debug] non-sanity validation did not run; refusing main training" >&2
    exit 2
  fi
  if ! grep -Eq 'UTMOS torch.hub cache|Loading UTMOS predictor|val_stats/utmos' "${log_path}"; then
    echo "[debug] UTMOS validation metric was not exercised; refusing main training" >&2
    exit 2
  fi
  if ! grep -Eq 'Loading WavLM_large model|val_stats/wavlm|WavLM validation' "${log_path}"; then
    echo "[debug] WavLM validation metric was not exercised; refusing main training" >&2
    exit 2
  fi
}

if [[ "${RUN_DEBUG}" == "True" ]]; then
  DEBUG_RUN_DIR="${HYDRA_RUN_DIR}-debug${DEBUG_TRAIN_STEPS}train-${DEBUG_VAL_BATCHES}val"
  echo "[debug] starting ${DEBUG_TRAIN_STEPS} train steps then ${DEBUG_VAL_BATCHES} val batch on dataset=${DEBUG_DATASET_NAME}"
  echo "[debug] hydra_run_dir=${DEBUG_RUN_DIR}"
  DEBUG_ARGS=(
    "${ARGS[@]}"
    hydra.run.dir="${DEBUG_RUN_DIR}"
    dataset.name="${DEBUG_DATASET_NAME}"
    train.trainer.max_steps="${DEBUG_MAX_STEPS}"
    train.trainer.val_check_interval="${DEBUG_VAL_CHECK_INTERVAL}"
    train.trainer.limit_val_batches="${DEBUG_VAL_BATCHES}"
    train.trainer.num_sanity_val_steps="${DEBUG_SANITY_VAL_STEPS}"
    train.wandb_enabled=False
  )
  run_train debug "${DEBUG_ARGS[@]}"
  verify_debug_log
  echo "[debug] passed; starting main training"
fi

run_train main "${ARGS[@]}"
