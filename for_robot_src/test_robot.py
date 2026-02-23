#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import threading
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs

# 네 cap_and_send.py에 있는 것들을 그대로 가져왔다고 가정
# (같은 폴더에 cap_and_send.py가 있으면 import로 사용)
from cap_and_send import FrameBuffer, capture_and_send

# =============================
# 사용자 설정
# =============================
SERVER_URL = "http://127.0.0.1:8000"

START_PLACE = 0
PLACE_WIDTH = 2

# 캡처/전송 옵션
N_FRAMES = 10

SAMPLE_DT = 0.4
CAPTURE_TIMEOUT_S = 5.0
POST_TIMEOUT_S = 10.0

# RealSense
W, H, FPS = 640, 480, 30

# 키
KEY_SEND       = ord('s')
KEY_NEXT_PLACE = ord('p')
KEY_PREV_PLACE = ord('o')

KEY_MODE_BANK     = ord('b')
KEY_MODE_TH_CALIB = ord('t')
KEY_MODE_QUERY    = ord('q')

KEY_QUIT = 27  # ESC
# =============================


def z(i: int, w: int) -> str:
    return str(i).zfill(w)


def plc_str(plc_idx: int) -> str:
    return z(plc_idx, PLACE_WIDTH)


def overlay_status(img_bgr: np.ndarray, place_id: str, mode: str, sending: bool) -> np.ndarray:
    view = img_bgr.copy()
    line1 = f"place={place_id}  mode={mode}  sending={int(sending)}"
    line2 = "Keys: s=send | p/o=place+/- | b=bank t=th_calib q=query | ESC=quit"
    cv2.putText(view, line1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(view, line2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return view


def main():
    buffer = FrameBuffer()

    plc_idx = START_PLACE
    mode: Literal["bank", "th_calib", "query"] = "bank"

    sending_lock = threading.Lock()
    sending_flag = {"on": False}  # mutable

    def set_sending(v: bool):
        with sending_lock:
            sending_flag["on"] = v

    def is_sending() -> bool:
        with sending_lock:
            return sending_flag["on"]

    print("Controls:")
    print(" s : send batch to server")
    print(" p/o : next/prev place")
    print(" b/t/q : mode bank / th_calib / query")
    print(" ESC : quit")
    print("Server:", SERVER_URL)

    # RealSense start
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())  # BGR uint8
            # cap_and_send는 RGB uint8을 기대하니 변환
            color_rgb = color_bgr[:, :, ::-1]

            # "ROS 콜백"처럼 계속 업데이트
            buffer.update(color_rgb)

            place_id = plc_str(plc_idx)
            view = overlay_status(color_bgr, place_id, mode, is_sending())
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

            elif key == KEY_MODE_BANK:
                mode = "bank"
                print("Mode -> bank")

            elif key == KEY_MODE_TH_CALIB:
                mode = "th_calib"
                print("Mode -> th_calib")

            elif key == KEY_MODE_QUERY:
                mode = "query"
                print("Mode -> query")

            elif key == KEY_SEND:
                # 전송 중이면 중복 전송 방지
                if is_sending():
                    print("[INFO] already sending... skip")
                    continue

                set_sending(True)
                place_id_now = plc_str(plc_idx)
                mode_now = mode
                if mode_now == "bank" or mode_now == "th_calib":
                    label = "normal"
                else :
                    label = "abnormal" 
                    
                def worker():
                    try:
                        print(f"[SEND] place={place_id_now} mode={mode_now} n={N_FRAMES} dt={SAMPLE_DT}")
                        out = capture_and_send(
                            buffer=buffer,
                            server_url=SERVER_URL,
                            place_id=place_id_now,
                            mode=mode_now,              # bank/th_calib/query
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