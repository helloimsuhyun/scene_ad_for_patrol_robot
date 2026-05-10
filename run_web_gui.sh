#!/usr/bin/env bash
set -e

# ============================================================
# Flutter Web GUI runner
# - Can be executed from anywhere
# - Uses script location as root directory
# - Kills old process on selected port
# - Serves sentrynexcontrol/build/web
# - Adds no-cache HTTP headers to reduce browser cache issues
# - Auto-detects server PC LAN IP
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PORT="${1:-8095}"

PROJECT_DIR="$SCRIPT_DIR/sentrynexcontrol"
WEB_DIR="$PROJECT_DIR/build/web"

get_server_ip() {
    local IP=""

    # Try to find the IP used for the default route.
    IP=$(ip route get 8.8.8.8 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}' || true)

    # Fallback: first IP from hostname -I.
    if [ -z "$IP" ]; then
        IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
    fi

    if [ -z "$IP" ]; then
        IP="<SERVER_PC_IP>"
    fi

    echo "$IP"
}

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

    for i in {1..5}; do
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

cleanup() {
    echo ""
    echo "[INFO] Stopping Flutter Web GUI server..."

    trap - INT TERM EXIT

    if [ -n "${SERVER_PID:-}" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true

        for i in {1..5}; do
            sleep 1

            if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
                echo "[INFO] Web GUI server stopped cleanly"
                return 0
            fi
        done

        echo "[WARN] Web GUI server did not stop in time. Force killing..."
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}

SERVER_IP="$(get_server_ip)"

echo "[INFO] Root directory     : $SCRIPT_DIR"
echo "[INFO] Flutter project   : $PROJECT_DIR"
echo "[INFO] Web directory     : $WEB_DIR"
echo "[INFO] Web GUI port      : $PORT"
echo "[INFO] Server PC IP      : $SERVER_IP"

# ------------------------------------------------------------
# Check Flutter project directory
# ------------------------------------------------------------
if [ ! -d "$PROJECT_DIR" ]; then
    echo "[ERROR] Flutter project directory does not exist:"
    echo "        $PROJECT_DIR"
    exit 1
fi

# ------------------------------------------------------------
# Check build output
# ------------------------------------------------------------
if [ ! -d "$WEB_DIR" ]; then
    echo "[ERROR] build/web directory does not exist."
    echo ""
    echo "[HINT] Build first:"
    echo "  cd $PROJECT_DIR"
    echo "  flutter clean"
    echo "  rm -rf build .dart_tool"
    echo "  flutter pub get"
    echo "  flutter build web --release --no-web-resources-cdn --pwa-strategy=none"
    exit 1
fi

if [ ! -f "$WEB_DIR/index.html" ]; then
    echo "[ERROR] index.html not found:"
    echo "        $WEB_DIR/index.html"
    exit 1
fi

# ------------------------------------------------------------
# Stop old server on same port
# ------------------------------------------------------------
echo "[INFO] Stop old web GUI server cleanly..."
graceful_kill_port "$PORT"

sleep 1

# ------------------------------------------------------------
# Print build timestamp
# ------------------------------------------------------------
echo "[INFO] Build file timestamp:"
stat "$WEB_DIR/index.html" | grep Modify || true

if [ -f "$WEB_DIR/main.dart.js" ]; then
    stat "$WEB_DIR/main.dart.js" | grep Modify || true
fi

# ------------------------------------------------------------
# Start no-cache static server
# ------------------------------------------------------------
echo "[INFO] Start Flutter Web GUI no-cache server"
echo "[INFO] Same PC  : http://localhost:$PORT"
echo "[INFO] Other PC : http://$SERVER_IP:$PORT"
echo ""

cd "$WEB_DIR"

python3 - "$PORT" <<'PY' &
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

port = int(sys.argv[1])


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


server = HTTPServer(("0.0.0.0", port), NoCacheHandler)

print(f"[INFO] Serving no-cache HTTP on 0.0.0.0:{port}", flush=True)
server.serve_forever()
PY

SERVER_PID=$!

trap cleanup INT TERM EXIT

wait "$SERVER_PID"