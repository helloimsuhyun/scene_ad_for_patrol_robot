#!/usr/bin/env bash

set -e

uvicorn vision_server.http_server:app --host 0.0.0.0 --port 8000 &
PID1=$!

uvicorn stream_server.signaling_server:app --host 0.0.0.0 --port 8001 &
PID2=$!

trap "kill $PID1 $PID2" INT TERM

wait