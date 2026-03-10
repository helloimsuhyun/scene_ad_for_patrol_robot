#!/usr/bin/env bash

set -e

# ─── 작업 디렉토리를 sentrynexcontrol 로 고정 (vision_server 모듈 경로) ───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/sentrynexcontrol"

CONDA_UV="$HOME/miniconda3/envs/dl/bin/uvicorn"

# ─── 시작 전 이미 점유된 포트 정리 ───
lsof -t -i:8000 | xargs -r kill -9 2>/dev/null || true
lsof -t -i:8001 | xargs -r kill -9 2>/dev/null || true

"$CONDA_UV" vision_server.http_server:app --host 0.0.0.0 --port 8000 &
PID1=$!

"$CONDA_UV" stream_server.signaling_server:app --host 0.0.0.0 --port 8001 &
PID2=$!

trap "kill $PID1 $PID2" INT TERM

wait