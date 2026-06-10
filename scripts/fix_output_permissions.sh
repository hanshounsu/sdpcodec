#!/usr/bin/env bash
set -euo pipefail

TARGETS=(
  /home/hounsu/voice/sdpcodec
  /data/hounsu/voice/bigcodec/outputs
)

USER_ID="$(id -u)"
GROUP_ID="$(id -g)"

sudo setfacl -R \
  -m "u:${USER_ID}:rwX,d:u:${USER_ID}:rwX,g:${GROUP_ID}:rwX,d:g:${GROUP_ID}:rwX,m:rwX" \
  "${TARGETS[@]}"

echo "Applied recursive and default ACLs for uid=${USER_ID}, gid=${GROUP_ID}."
