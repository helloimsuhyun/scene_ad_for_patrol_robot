#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/sentrynexcontrol"

CONDA_PY="$HOME/miniconda3/envs/dl/bin/python"

PORT1=8000
PORT2=8001

graceful_kill_port() {
    local PORT="$1"

    local PIDS
    PIDS=$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)

    if [ -z "$PIDS" ]; then
        return 0
    fi

    echo "[INFO] Gracefully stopping process on port $PORT: $PIDS"

    kill -TERM $PIDS 2>/dev/null || true

    for i in {1..8}; do
        sleep 1

        local STILL_ALIVE=""
        for PID in $PIDS; do
            if ps -p "$PID" > /dev/null 2>&1; then
                STILL_ALIVE="$STILL_ALIVE $PID"
            fi
        done

        if [ -z "$STILL_ALIVE" ]; then
            echo "[INFO] Port $PORT stopped cleanly"
            return 0
        fi
    done

    echo "[WARN] Force killing process on port $PORT:$STILL_ALIVE"
    kill -KILL $STILL_ALIVE 2>/dev/null || true
}

echo "[INFO] Stop old servers cleanly..."

graceful_kill_port "$PORT1"
graceful_kill_port "$PORT2"

sleep 1

echo "[INFO] Reset environment..."

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset CUDA_VISIBLE_DEVICES
unset LD_LIBRARY_PATH

echo "[INFO] Check CUDA..."

"$CONDA_PY" - <<'PY'
import os
import sys
import torch

print("[PYTHON]", sys.executable)
print("[CUDA_VISIBLE_DEVICES]", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[TORCH]", torch.__version__)
print("[TORCH CUDA BUILD]", torch.version.cuda)
print("[CUDA AVAILABLE]", torch.cuda.is_available())
print("[CUDA DEVICE COUNT]", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Server startup stopped.")

print("[GPU]", torch.cuda.get_device_name(0))
PY

echo "[INFO] Start vision server :8000"
"$CONDA_PY" -m uvicorn vision_server.http_server:app \
    --host 0.0.0.0 \
    --port "$PORT1" \
    --timeout-graceful-shutdown 10 &
PID1=$!

echo "[INFO] Start stream server :8001"
"$CONDA_PY" -m uvicorn stream_server.signaling_server:app \
    --host 0.0.0.0 \
    --port "$PORT2" \
    --timeout-graceful-shutdown 10 &
PID2=$!

cleanup() {
    echo "[INFO] Stopping servers cleanly..."

    trap - INT TERM EXIT

    kill -TERM "$PID1" "$PID2" 2>/dev/null || true

    for i in {1..10}; do
        sleep 1

        ALIVE=0

        if ps -p "$PID1" > /dev/null 2>&1; then
            ALIVE=1
        fi

        if ps -p "$PID2" > /dev/null 2>&1; then
            ALIVE=1
        fi

        if [ "$ALIVE" -eq 0 ]; then
            echo "[INFO] Servers stopped cleanly"
            return 0
        fi
    done

    echo "[WARN] Servers did not stop in time. Force killing..."
    kill -KILL "$PID1" "$PID2" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

wait "$PID1" "$PID2"