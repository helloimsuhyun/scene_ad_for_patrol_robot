#!/usr/bin/env bash
set -e

# ============================================================
# Flutter Web GUI builder
# - Can be executed from anywhere
# - Uses script location as root directory
# - Builds sentrynexcontrol Flutter web release
# - Disables CDN resources and PWA service worker
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_DIR="$SCRIPT_DIR/sentrynexcontrol"
WEB_DIR="$PROJECT_DIR/build/web"

echo "[INFO] Root directory   : $SCRIPT_DIR"
echo "[INFO] Flutter project : $PROJECT_DIR"
echo "[INFO] Web output      : $WEB_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "[ERROR] Flutter project directory does not exist:"
    echo "        $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

echo "[INFO] Current directory:"
pwd

echo "[INFO] Flutter version:"
flutter --version

echo "[INFO] Clean old Flutter build files..."
flutter clean
rm -rf build .dart_tool

echo "[INFO] Get Flutter packages..."
flutter pub get

echo "[INFO] Build Flutter Web release..."
flutter build web \
    --release \
    --no-web-resources-cdn \
    --pwa-strategy=none

echo "[INFO] Check build output..."

if [ ! -d "$WEB_DIR" ]; then
    echo "[ERROR] build/web was not created."
    exit 1
fi

if [ ! -f "$WEB_DIR/index.html" ]; then
    echo "[ERROR] index.html not found in build/web."
    exit 1
fi

echo "[INFO] Build completed successfully."
echo "[INFO] Build output:"
echo "       $WEB_DIR"

echo "[INFO] Build file timestamp:"
stat "$WEB_DIR/index.html" | grep Modify || true

if [ -f "$WEB_DIR/main.dart.js" ]; then
    stat "$WEB_DIR/main.dart.js" | grep Modify || true
fi

echo ""
echo "[INFO] Next step:"
echo "  cd $SCRIPT_DIR"
echo "  ./run_web_gui.sh 8095"
