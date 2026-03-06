#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs
import requests

from cap_and_send import FrameBuffer, capture_and_send

# =============================
# 사용자 설정
# =============================
SERVER_URL = "http://127.0.0.1:8000"

START_PLACE = 0
PLACE_WIDTH = 2

N_FRAMES = 5
SAMPLE_DT = 0.2
CAPTURE_TIMEOUT_S = 5.0
POST_TIMEOUT_S = 10.0
GET_TIMEOUT_S = 3.0

W, H, FPS = 640, 480, 30

KEY_SEND       = ord('s')
KEY_NEXT_PLACE = ord('p')
KEY_PREV_PLACE = ord('o')

# query GT만 로컬에서 토글
KEY_QUERY_NORMAL   = ord('n')
KEY_QUERY_ABNORMAL = ord('a')

KEY_QUIT = 27
# =============================


def z(i: int, w: int) -> str:
    return str(i).zfill(w)


def plc_str(plc_idx: int) -> str:
    return z(plc_idx, PLACE_WIDTH)


def get_place_status(server_url: str, place_id: str) -> dict:
    resp = requests.get(f"{server_url}/places/{place_id}", timeout=GET_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def get_server_mode(server_url: str, place_id: str) -> str:
    """
    서버 DB에 저장된 현재 mode 읽기
    """
    data = get_place_status(server_url, place_id)
    place = data.get("place", {})
    mode = place.get("mode", "bank")
    if mode not in ("bank", "th_calib", "query"):
        mode = "bank"
    return mode


def overlay_status(
    img_bgr: np.ndarray,
    place_id: str,
    mode_from_server: str,
    sending: bool,
    query_gt: str,
) -> np.ndarray:
    view = img_bgr.copy()
    line1 = f"place={place_id}  server_mode={mode_from_server}  query_gt={query_gt}  sending={int(sending)}"
    line2 = "Keys: s=send | p/o=place+/- | n/a=query gt | ESC=quit"
    cv2.putText(view, line1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(view, line2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return view


def main():
    buffer = FrameBuffer()

    plc_idx = START_PLACE
    query_gt: Literal["normal", "abnormal"] = "normal"

    sending_lock = threading.Lock()
    sending_flag = {"on": False}

    def set_sending(v: bool):
        with sending_lock:
            sending_flag["on"] = v

    def is_sending() -> bool:
        with sending_lock:
            return sending_flag["on"]

    print("Controls:")
    print(" s   : send batch to server")
    print(" p/o : next/prev place")
    print(" n/a : (query only) set GT label normal/abnormal")
    print(" ESC : quit")
    print("Mode is controlled by SERVER /places/{place_id}/config")
    print("Server:", SERVER_URL)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    pipeline.start(config)

    last_mode = "bank"

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            color_rgb = color_bgr[:, :, ::-1]
            buffer.update(color_rgb)

            place_id = plc_str(plc_idx)

            # 서버에서 현재 mode 조회
            try:
                last_mode = get_server_mode(SERVER_URL, place_id)
            except Exception as e:
                print(f"[WARN] get_server_mode failed for place={place_id}: {e}")

            view = overlay_status(color_bgr, place_id, last_mode, is_sending(), query_gt)
            cv2.imshow("CaptureSend", view)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_QUIT:
                break

            elif key == KEY_NEXT_PLACE:
                plc_idx += 1
                print("Move to place:", plc_str(plc_idx))

            elif key == KEY_PREV_PLACE:
                plc_idx = max(0, plc_idx - 1)
                print("Move to place:", plc_str(plc_idx))

            elif key == KEY_QUERY_NORMAL:
                query_gt = "normal"
                print("Query GT -> normal")

            elif key == KEY_QUERY_ABNORMAL:
                query_gt = "abnormal"
                print("Query GT -> abnormal")

            elif key == KEY_SEND:
                if is_sending():
                    print("[INFO] already sending... skip")
                    continue

                set_sending(True)
                place_id_now = plc_str(plc_idx)

                try:
                    mode_now = get_server_mode(SERVER_URL, place_id_now)
                except Exception as e:
                    print("[SEND FAIL] cannot get mode from server:", e)
                    set_sending(False)
                    continue

                if mode_now in ("bank", "th_calib"):
                    label = "normal"
                else:
                    label = query_gt

                def worker():
                    try:
                        print(f"[SEND] place={place_id_now} mode={mode_now} gt={label} n={N_FRAMES} dt={SAMPLE_DT}")
                        out = capture_and_send(
                            buffer=buffer,
                            server_url=SERVER_URL,
                            place_id=place_id_now,
                            mode=mode_now,
                            n_frames=N_FRAMES,
                            sample_dt=SAMPLE_DT,
                            capture_timeout_s=CAPTURE_TIMEOUT_S,
                            post_timeout_s=POST_TIMEOUT_S,
                            gt=label,
                        )
                        print("[SEND OK]", out)
                    except Exception as e:
                        print("[SEND FAIL]", e)
                    finally:
                        set_sending(False)

                threading.Thread(target=worker, daemon=True).start()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()