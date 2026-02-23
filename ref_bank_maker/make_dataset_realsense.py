#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from typing import Optional

# =============================
# 사용자 설정
# =============================
ROOT_DIR = Path("./dataset")
START_PLACE = 0
PLACE_WIDTH = 2

IDX_WIDTH = 4
AUTO_RESUME = True

SAVE_DEPTH = False

# 키
KEY_SAVE       = ord('s')
KEY_NEXT_PLACE = ord('p')
KEY_PREV_PLACE = ord('o')
KEY_QUIT       = ord('q')

KEY_CLASS_NOMAL          = ord('n')  # nomal
KEY_CLASS_NOMAL_FOR_Q    = ord('m')  # ✅ nomal_for_query
KEY_CLASS_UNNORMAL       = ord('u')  # unnormal

KEY_RESET_IDX  = ord('r')
# =============================

CLASS_NOMAL = "nomal"
CLASS_NOMAL_FOR_QUERY = "nomal_for_query"
CLASS_UNNORMAL = "unnormal"

IMG_EXT = ".png"

def z(i: int, w: int) -> str:
    return str(i).zfill(w)

def plc_str(plc_idx: int) -> str:
    return z(plc_idx, PLACE_WIDTH)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_save_dir(root: Path, plc_idx: int, cls: str) -> Path:
    return root / plc_str(plc_idx) / cls

def scan_last_index(folder: Path, cls_prefix: str) -> int:
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

def make_paths(root: Path, plc_idx: int, cls: str, idx: int) -> tuple[Path, Path, Optional[Path]]:
    folder = get_save_dir(root, plc_idx, cls)
    ensure_dir(folder)

    base = f"{cls}_{z(idx, IDX_WIDTH)}"
    color_path = folder / f"{base}{IMG_EXT}"
    depth_path = folder / f"{base}_depth{IMG_EXT}" if SAVE_DEPTH else None
    return folder, color_path, depth_path

def overlay_status(img_bgr, plc_idx: int, cls: str, idx: int):
    view = img_bgr.copy()
    line1 = f"class={cls}  place={plc_str(plc_idx)}  idx={z(idx, IDX_WIDTH)}"
    line2 = "Keys: s=save  p/o=place+/-  n=nomal  m=nomal_for_query  u=unnormal  r=reset  q=quit"
    cv2.putText(view, line1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(view, line2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    return view

def get_next_idx(root: Path, plc_idx: int, cls: str) -> int:
    if not AUTO_RESUME:
        return 0
    folder = get_save_dir(root, plc_idx, cls)
    return scan_last_index(folder, cls_prefix=cls)

def main():
    ensure_dir(ROOT_DIR)

    plc_idx = START_PLACE
    cls = CLASS_NOMAL
    idx = get_next_idx(ROOT_DIR, plc_idx, cls)

    print("Controls:")
    print(" s : save (color [+depth])")
    print(" p : next place | o : prev place")
    print(" n : class=nomal | m : class=nomal_for_query | u : class=unnormal")
    print(" r : reset index (DANGEROUS)")
    print(" q : quit")
    print("Start:", "place", plc_str(plc_idx), "class", cls, "idx", idx)
    print("Save root:", str(ROOT_DIR.resolve()))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            cv2.imshow("Capture", overlay_status(color, plc_idx, cls, idx))
            key = cv2.waitKey(1) & 0xFF

            if key == KEY_QUIT:
                break

            elif key == KEY_NEXT_PLACE:
                plc_idx += 1
                idx = get_next_idx(ROOT_DIR, plc_idx, cls)
                print("Move to place", plc_str(plc_idx), "| class", cls, "| idx", idx)

            elif key == KEY_PREV_PLACE:
                plc_idx = max(0, plc_idx - 1)
                idx = get_next_idx(ROOT_DIR, plc_idx, cls)
                print("Move to place", plc_str(plc_idx), "| class", cls, "| idx", idx)

            elif key == KEY_CLASS_NOMAL:
                cls = CLASS_NOMAL
                idx = get_next_idx(ROOT_DIR, plc_idx, cls)
                print("Class -> nomal | place", plc_str(plc_idx), "| idx", idx)

            elif key == KEY_CLASS_NOMAL_FOR_Q:
                cls = CLASS_NOMAL_FOR_QUERY
                idx = get_next_idx(ROOT_DIR, plc_idx, cls)
                print("Class -> nomal_for_query | place", plc_str(plc_idx), "| idx", idx)

            elif key == KEY_CLASS_UNNORMAL:
                cls = CLASS_UNNORMAL
                idx = get_next_idx(ROOT_DIR, plc_idx, cls)
                print("Class -> unnormal | place", plc_str(plc_idx), "| idx", idx)

            elif key == KEY_RESET_IDX:
                idx = 0
                print("[WARN] Reset idx to 0 for place", plc_str(plc_idx), "class", cls)

            elif key == KEY_SAVE:
                _, color_path, depth_path = make_paths(ROOT_DIR, plc_idx, cls, idx)

                ok1 = cv2.imwrite(str(color_path), color)
                ok2 = True
                if SAVE_DEPTH and depth_path is not None:
                    ok2 = cv2.imwrite(str(depth_path), depth)

                if ok1 and ok2:
                    print("Saved:", color_path)
                    idx += 1
                else:
                    print("[WARN] save failed:", color_path)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()