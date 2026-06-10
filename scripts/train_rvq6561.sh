#!/usr/bin/env bash
set -euo pipefail

cd /home/hounsu/voice/sdpcodec
python3 -m sdpcodec.train "$@"
