#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

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
N_FRAMES = 5
SAMPLE_DT = 0.2
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

# query 라벨 토글 키
KEY_QUERY_NORMAL   = ord('n')  # query에서 GT=normal
KEY_QUERY_ABNORMAL = ord('a')  # query에서 GT=abnormal

KEY_QUIT = 27  # ESC
# =============================


def z(i: int, w: int) -> str:
    return str(i).zfill(w)


def plc_str(plc_idx: int) -> str:
    return z(plc_idx, PLACE_WIDTH)


def overlay_status(
    img_bgr: np.ndarray,
    place_id: str,
    mode: str,
    sending: bool,
    query_gt: str,
) -> np.ndarray:
    view = img_bgr.copy()
    line1 = f"place={place_id}  mode={mode}  query_gt={query_gt}  sending={int(sending)}"
    line2 = "Keys: s=send | p/o=place+/- | b=bank t=th_calib q=query | n/a=query gt | ESC=quit"
    cv2.putText(view, line1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(view, line2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return view


def main():
    buffer = FrameBuffer()

    plc_idx = START_PLACE
    mode: Literal["bank", "th_calib", "query"] = "bank"

    # ✅ query 모드에서만 쓰는 GT 라벨 (키로 바꿀 수 있음)
    query_gt: Literal["normal", "abnormal"] = "normal"

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
    print(" n/a : (query only) set GT label normal/abnormal")
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
            view = overlay_status(color_bgr, place_id, mode, is_sending(), query_gt)
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

            # ✅ query GT 라벨 변경 (query 모드에서만 의미 있음)
            elif key == KEY_QUERY_NORMAL:
                query_gt = "normal"
                print("Query GT -> normal")

            elif key == KEY_QUERY_ABNORMAL:
                query_gt = "abnormal"
                print("Query GT -> abnormal")

            elif key == KEY_SEND:
                # 전송 중이면 중복 전송 방지
                if is_sending():
                    print("[INFO] already sending... skip")
                    continue

                set_sending(True)
                place_id_now = plc_str(plc_idx)
                mode_now = mode

                # ✅ 여기서 라벨 결정
                if mode_now in ("bank", "th_calib"):
                    label = "normal"
                else:
                    # query에서는 키로 설정한 GT 사용
                    label = query_gt

                def worker():
                    try:
                        print(f"[SEND] place={place_id_now} mode={mode_now} gt={label} n={N_FRAMES} dt={SAMPLE_DT}")
                        out = capture_and_send(
                            buffer=buffer,
                            server_url=SERVER_URL,
                            place_id=place_id_now,
                            mode=mode_now,              # bank/th_calib/query
                            n_frames=N_FRAMES,
                            sample_dt=SAMPLE_DT,
                            capture_timeout_s=CAPTURE_TIMEOUT_S,
                            post_timeout_s=POST_TIMEOUT_S,
                            gt=label,                   # ✅ GT 라벨 전송
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