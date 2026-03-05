#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs

# =============================
# 사용자 설정
# =============================
ROOT_DIR = Path("./dataset")

START_PLACE = 0
PLACE_WIDTH = 2

IDX_WIDTH = 6
AUTO_RESUME = True

# 캡처 옵션 (s 누르면 N_FRAMES장 저장)
N_FRAMES = 10
SAMPLE_DT = 0.2  # seconds

# RealSense
W, H, FPS = 640, 480, 30

# 키
KEY_SAVE       = ord('s')
KEY_NEXT_PLACE = ord('p')
KEY_PREV_PLACE = ord('o')

KEY_MODE_BANK     = ord('b')
KEY_MODE_TH_CALIB = ord('t')
KEY_MODE_QUERY    = ord('q')

KEY_QUERY_NORMAL   = ord('n')  # query에서 GT=normal
KEY_QUERY_ABNORMAL = ord('a')  # query에서 GT=abnormal

KEY_QUIT = 27  # ESC
# =============================

IMG_EXT = ".png"


def z(i: int, w: int) -> str:
    return str(i).zfill(w)


def plc_str(plc_idx: int) -> str:
    return z(plc_idx, PLACE_WIDTH)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def get_cls(mode: Literal["bank", "th_calib", "query"],
            query_gt: Literal["normal", "abnormal"]) -> str:
    # ✅ 하위 폴더를 만들지 않기 위해 mode+gt를 cls로 합침
    if mode == "query":
        return f"query_{query_gt}"   # query_normal / query_abnormal
    return mode                      # bank / th_calib


def get_save_dir(root: Path, plc_idx: int, cls: str) -> Path:
    return root / plc_str(plc_idx) / cls


def scan_last_index(folder: Path, cls_prefix: str) -> int:
    """
    {cls}_{000123}.png 형태에서 다음 인덱스를 찾음
    """
    if not folder.exists():
        return 0
    pat = re.compile(rf"^{re.escape(cls_prefix)}_(\d+){re.escape(IMG_EXT)}$")
    max_idx = -1
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def get_next_idx(root: Path, plc_idx: int, cls: str) -> int:
    if not AUTO_RESUME:
        return 0
    folder = get_save_dir(root, plc_idx, cls)
    return scan_last_index(folder, cls_prefix=cls)


def overlay_status(img_bgr: np.ndarray, plc_idx: int, mode: str, query_gt: str, cls: str, next_idx: int) -> np.ndarray:
    view = img_bgr.copy()
    line1 = f"place={plc_str(plc_idx)}  mode={mode}  query_gt={query_gt}  cls={cls}  next={z(next_idx, IDX_WIDTH)}"
    line2 = "Keys: s=save batch | p/o=place+/- | b=bank t=th_calib q=query | n/a=query gt | ESC=quit"
    cv2.putText(view, line1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(view, line2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return view


def save_one(folder: Path, cls: str, idx: int, bgr: np.ndarray) -> bool:
    ensure_dir(folder)
    p = folder / f"{cls}_{z(idx, IDX_WIDTH)}{IMG_EXT}"
    ok = cv2.imwrite(str(p), bgr)
    if ok:
        print("[SAVE]", p)
    else:
        print("[WARN] save failed:", p)
    return ok


def save_batch(pipeline: rs.pipeline, root: Path, plc_idx: int, cls: str, first_bgr: np.ndarray):
    folder = get_save_dir(root, plc_idx, cls)
    idx = get_next_idx(root, plc_idx, cls)

    # 첫 프레임 저장
    if save_one(folder, cls, idx, first_bgr):
        idx += 1
        saved = 1
    else:
        return

    t_next = time.time() + SAMPLE_DT

    while saved < N_FRAMES:
        now = time.time()
        if now < t_next:
            time.sleep(max(0.0, t_next - now))
        t_next += SAMPLE_DT

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        bgr = np.asanyarray(color_frame.get_data())

        if save_one(folder, cls, idx, bgr):
            idx += 1
            saved += 1


def main():
    ensure_dir(ROOT_DIR)

    plc_idx = START_PLACE
    mode: Literal["bank", "th_calib", "query"] = "bank"
    query_gt: Literal["normal", "abnormal"] = "normal"

    print("Controls:")
    print(" s : save batch to disk")
    print(" p/o : next/prev place")
    print(" b/t/q : mode bank / th_calib / query")
    print(" n/a : (query only) set GT label normal/abnormal")
    print(" ESC : quit")
    print("Save root:", str(ROOT_DIR.resolve()))

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

            color_bgr = np.asanyarray(color_frame.get_data())

            cls = get_cls(mode, query_gt)
            next_idx = get_next_idx(ROOT_DIR, plc_idx, cls)

            cv2.imshow("DatasetCapture", overlay_status(color_bgr, plc_idx, mode, query_gt, cls, next_idx))
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

            elif key == KEY_QUERY_NORMAL:
                query_gt = "normal"
                print("Query GT -> normal")

            elif key == KEY_QUERY_ABNORMAL:
                query_gt = "abnormal"
                print("Query GT -> abnormal")

            elif key == KEY_SAVE:
                cls_now = get_cls(mode, query_gt)
                print(f"[BATCH SAVE] place={plc_str(plc_idx)} cls={cls_now} n={N_FRAMES} dt={SAMPLE_DT}")
                save_batch(pipeline, ROOT_DIR, plc_idx, cls_now, color_bgr)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()