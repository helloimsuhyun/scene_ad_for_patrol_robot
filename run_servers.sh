#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/sentrynexcontrol"

CONDA_PY="$HOME/miniconda3/envs/dl/bin/python"

lsof -t -i:8000 | xargs -r kill -9 2>/dev/null || true
lsof -t -i:8001 | xargs -r kill -9 2>/dev/null || true

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset LD_LIBRARY_PATH

"$CONDA_PY" -m uvicorn vision_server.http_server:app --host 0.0.0.0 --port 8000 &
PID1=$!

"$CONDA_PY" -m uvicorn stream_server.signaling_server:app --host 0.0.0.0 --port 8001 &
PID2=$!

trap "kill $PID1 $PID2" INT TERM
wait