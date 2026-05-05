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
        echo "[INFO] No process listening on port $PORT"
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

check_cuda() {
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

# 실제 CUDA runtime allocation 테스트
x = torch.randn(1, device="cuda")
torch.cuda.synchronize()
print("[CUDA ALLOC TEST] OK", x)
PY
}

echo "[INFO] Stop old servers cleanly..."

graceful_kill_port "$PORT1"
graceful_kill_port "$PORT2"

sleep 1

echo "[INFO] Reset environment..."

export PYTHONNOUSERSITE=1
unset PYTHONPATH

# GPU 명시. unset 하지 말 것.
export CUDA_VISIBLE_DEVICES=0

# LD_LIBRARY_PATH는 일부 torch/opencv extension에 영향 줄 수 있으므로 건드리지 않음.
# unset LD_LIBRARY_PATH

echo "[INFO] GPU process check before CUDA test..."
nvidia-smi || true

echo "[INFO] Check CUDA..."

if ! check_cuda; then
    echo ""
    echo "[ERROR] CUDA check failed."
    echo "[HINT] Check remaining GPU users:"
    echo "  sudo fuser -v /dev/nvidia*"
    echo ""
    echo "[HINT] If only Xorg remains and torch CUDA still fails, reload nvidia_uvm:"
    echo "  sudo rmmod nvidia_uvm"
    echo "  sudo modprobe nvidia_uvm"
    echo ""
    echo "[HINT] Then test again:"
    echo "  ./run_servers.sh"
    echo ""
    echo "[HINT] If it still fails, reboot:"
    echo "  sudo reboot"
    exit 1
fi

echo "[INFO] Start vision server :8000"
"$CONDA_PY" -m uvicorn vision_server.http_server:app \
    --host 0.0.0.0 \
    --port "$PORT1" \
    --workers 1 \
    --timeout-graceful-shutdown 10 &
PID1=$!

echo "[INFO] Start stream server :8001"
"$CONDA_PY" -m uvicorn stream_server.signaling_server:app \
    --host 0.0.0.0 \
    --port "$PORT2" \
    --workers 1 \
    --timeout-graceful-shutdown 10 &
PID2=$!

cleanup() {
    echo "[INFO] Stopping servers cleanly..."

    trap - INT TERM EXIT

    if [ -n "${PID1:-}" ] || [ -n "${PID2:-}" ]; then
        kill -TERM "${PID1:-}" "${PID2:-}" 2>/dev/null || true
    fi

    for i in {1..10}; do
        sleep 1

        ALIVE=0

        if [ -n "${PID1:-}" ] && ps -p "$PID1" > /dev/null 2>&1; then
            ALIVE=1
        fi

        if [ -n "${PID2:-}" ] && ps -p "$PID2" > /dev/null 2>&1; then
            ALIVE=1
        fi

        if [ "$ALIVE" -eq 0 ]; then
            echo "[INFO] Servers stopped cleanly"
            return 0
        fi
    done

    echo "[WARN] Servers did not stop in time. Force killing..."
    kill -KILL "${PID1:-}" "${PID2:-}" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

wait "$PID1" "$PID2"