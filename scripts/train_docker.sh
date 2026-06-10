#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-hounsu_sdpcodec:260416}"
CONDA_ENV="${CONDA_ENV:-bigcodec}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/${CONDA_ENV}/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/hounsu/voice/sdpcodec}"
UID_GID="$(id -u):$(id -g)"

mkdir -p "${HOME}/.cache/matplotlib" "${HOME}/.cache/fontconfig"

exec docker run --rm --init \
  --user "${UID_GID}" \
  -e HOME="${HOME}" \
  -e MPLCONFIGDIR="${HOME}/.cache/matplotlib" \
  -e XDG_CACHE_HOME="${HOME}/.cache" \
  -v /home/hounsu:/home/hounsu \
  -v /data:/data \
  -w "${REPO_ROOT}" \
  "${IMAGE}" \
  "${PYTHON_BIN}" -u -m sdpcodec.train "$@"
