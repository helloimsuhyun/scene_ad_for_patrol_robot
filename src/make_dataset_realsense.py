# 테스트 데이터 형성용 리얼센스
# - gpt


import os
import re
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from datetime import datetime

# =============================
# 사용자 설정
# =============================
ROOT_DIR = "./ref_bank"        # ✅ 너 구조에 맞춤 (plc_idx 아래에 bank/th_calib/query)
START_PLACE = 0
PLACE_WIDTH = 2
PHOTO_WIDTH = 4                # 시간 저장이면 필요 없지만, 숫자 저장 모드도 지원

# 키
KEY_SAVE       = ord('s')
KEY_NEXT_PLACE = ord('p')
KEY_PREV_PLACE = ord('o')      # ✅ 이전 장소
KEY_RESET_IDX  = ord('r')
KEY_QUIT       = ord('q')

KEY_MODE_BANK  = ord('b')
KEY_MODE_TH    = ord('t')
KEY_MODE_QUERY = ord('i')      # i=infer/query(저장용)

AUTO_RESUME = True
USE_TIMESTAMP_NAME = True      # ✅ True면 파일명을 시간으로 저장 (충돌/인덱스관리 스트레스↓)
# =============================

def z(i, w):
    return str(i).zfill(w)

def plc_str(plc_idx: int):
    return z(plc_idx, PLACE_WIDTH)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def scan_last_index(folder: Path):
    """
    folder 안의 파일명 패턴:
      - USE_TIMESTAMP_NAME=True면: YYYYmmdd_HHMMSS_mmm.png -> 인덱스 스캔 의미 없음 (0부터 시작)
      - False면: 0000.png 같은 숫자 -> 마지막 번호 + 1 리턴
    """
    if not folder.exists():
        return 0
    if USE_TIMESTAMP_NAME:
        return 0

    pat = re.compile(r"^(\d+)\.png$")
    max_idx = -1
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

def make_save_paths(root_dir: Path, plc_idx: int, mode: str, idx: int):
    """
    저장 폴더: root_dir/plc_idx/mode/
    파일명: timestamp or numeric
    """
    folder = root_dir / plc_str(plc_idx) / mode
    ensure_dir(folder)

    if USE_TIMESTAMP_NAME:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base = ts
    else:
        base = z(idx, PHOTO_WIDTH)

    color_path = folder / f"{base}.png"
    depth_path = folder / f"{base}_depth.png"
    return folder, color_path, depth_path

def overlay_status(img_bgr, plc_idx, mode, idx):
    view = img_bgr.copy()
    txt = f"mode={mode}  place={plc_str(plc_idx)}  idx={z(idx, PHOTO_WIDTH)}"
    cv2.putText(view, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(view, "Keys: s=save  p/o=place+/-  b/t/i=mode  r=reset  q=quit",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    return view

def main():
    root_dir = Path(ROOT_DIR)
    ensure_dir(root_dir)

    mode = "bank"     # bank / th_calib / query
    plc_idx = START_PLACE

    # AUTO_RESUME: mode+place 폴더 기준으로 인덱스 이어찍기
    if AUTO_RESUME and not USE_TIMESTAMP_NAME:
        folder = root_dir / plc_str(plc_idx) / mode
        idx = scan_last_index(folder)
    else:
        idx = 0

    print("Controls:")
    print(" s : save (color+depth)")
    print(" p : next place | o : prev place")
    print(" b : mode=bank | t : mode=th_calib | i : mode=query")
    print(" r : reset index (only numeric mode)")
    print(" q : quit")
    print("Start:", "place", plc_str(plc_idx), "mode", mode, "idx", idx)

    # =============================
    # RealSense 설정
    # =============================
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())  # BGR uint8
            depth = np.asanyarray(depth_frame.get_data())  # uint16

            view = overlay_status(color, plc_idx, mode, idx)
            cv2.imshow("Capture", view)

            key = cv2.waitKey(1) & 0xFF

            if key == KEY_QUIT:
                break

            elif key == KEY_NEXT_PLACE:
                plc_idx += 1
                if AUTO_RESUME and not USE_TIMESTAMP_NAME:
                    idx = scan_last_index(root_dir / plc_str(plc_idx) / mode)
                else:
                    idx = 0
                print("Move to place", plc_str(plc_idx))

            elif key == KEY_PREV_PLACE:
                plc_idx = max(0, plc_idx - 1)
                if AUTO_RESUME and not USE_TIMESTAMP_NAME:
                    idx = scan_last_index(root_dir / plc_str(plc_idx) / mode)
                else:
                    idx = 0
                print("Move to place", plc_str(plc_idx))

            elif key == KEY_MODE_BANK:
                mode = "bank"
                if AUTO_RESUME and not USE_TIMESTAMP_NAME:
                    idx = scan_last_index(root_dir / plc_str(plc_idx) / mode)
                else:
                    idx = 0
                print("Mode -> bank")

            elif key == KEY_MODE_TH:
                mode = "th_calib"
                if AUTO_RESUME and not USE_TIMESTAMP_NAME:
                    idx = scan_last_index(root_dir / plc_str(plc_idx) / mode)
                else:
                    idx = 0
                print("Mode -> th_calib")

            elif key == KEY_MODE_QUERY:
                mode = "query"
                if AUTO_RESUME and not USE_TIMESTAMP_NAME:
                    idx = scan_last_index(root_dir / plc_str(plc_idx) / mode)
                else:
                    idx = 0
                print("Mode -> query")

            elif key == KEY_RESET_IDX:
                if not USE_TIMESTAMP_NAME:
                    idx = 0
                    print("Reset index to 0")

            elif key == KEY_SAVE:
                folder, color_path, depth_path = make_save_paths(root_dir, plc_idx, mode, idx)

                # 저장
                ok1 = cv2.imwrite(str(color_path), color)
                #ok2 = cv2.imwrite(str(depth_path), depth)  # 16bit png로 저장됨
                ok2 = True
                if ok1 and ok2:
                    print("Saved:", color_path)
                    if not USE_TIMESTAMP_NAME:
                        idx += 1
                else:
                    print("[WARN] save failed:", color_path)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
