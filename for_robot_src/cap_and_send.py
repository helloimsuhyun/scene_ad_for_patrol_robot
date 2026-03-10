#cap_and_send.py
#ros2 node에서 장소의 사진을 sampling dt 간격으로 n장 찍고, 서버로 전송하는 역할

import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Literal
from datetime import datetime

import numpy as np
import requests

#rgb numpy를 jpeg 바이너리 byte로 변환
def _encode_jpeg_from_rgb(rgb: np.ndarray, quality: int = 90) -> bytes:
    import cv2
    bgr = rgb[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()

#img와 meta batch를 post하는 함수 ( http )
def post_batch(server_url: str, images_rgb: List[np.ndarray], meta: Dict[str, Any], timeout_s: float = 5.0) -> Dict[str, Any]:
    url = server_url.rstrip("/") + "/place_imgs"
    files = []
    for i, img in enumerate(images_rgb):
        jpg = _encode_jpeg_from_rgb(img, quality=90) #jpg로 변환
        files.append(("images", (f"frame_{i:02d}.jpg", jpg, "image/jpeg")))
    files.append(("meta", (None, json.dumps(meta), "application/json")))
    r = requests.post(url, files=files, timeout=timeout_s)
    r.raise_for_status()
    return r.json() #서버로부터 받은 json


class FrameBuffer:
    def __init__(self):
        self.latest_rgb: Optional[np.ndarray] = None
        self._cond = threading.Condition() # 새 프레임이 들어오면 깨어남
        self._frame_counter: int = 0 # frame cnt
        self._last_served_counter: int = 0

    def update(self, rgb_uint8: np.ndarray) -> None: # ros 호출함수
        with self._cond:
            self.latest_rgb = rgb_uint8
            self._frame_counter += 1        
            self._cond.notify_all() # wait중인 쓰레드 꺠움

    def capture_n(self, n_frames: int = 10, sample_dt: float = 0.2, timeout_s: float = 5.0) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        last_take = 0.0
        last_counter = -1                 
        t_end = time.monotonic() + float(timeout_s) #마지막 종료시간 (timeout 까지 못 모으면 실패하도록)

        with self._cond:
            while len(frames) < int(n_frames):
                remaining = t_end - time.monotonic()
                if remaining <= 0:
                    break

                # 새 프레임 올 때까지 대기
                if self._frame_counter == last_counter:
                    self._cond.wait(timeout=remaining)
                    continue

                if self.latest_rgb is None:
                    self._cond.wait(timeout=remaining)
                    continue

                now = time.monotonic()
                if last_take != 0.0 and (now - last_take) < float(sample_dt): # 마지막 저장 이후 sample_dt 보다 덜 지낫으면 저장하지 않고 기다림
                    wait_t = min(float(sample_dt) - (now - last_take), remaining)
                    self._cond.wait(timeout=max(0.0, wait_t))
                    continue

                # 이제 남은 case는 새 frame이고 , sampling dt 이후 시간 > framse에 저장
                frames.append(self.latest_rgb.copy())
                last_take = now
                last_counter = self._frame_counter

        if len(frames) < int(n_frames):
            raise RuntimeError(f"timeout: captured {len(frames)}/{n_frames}")
        return frames
    
    #마지막으로 return한 이후에 들어온 새 프레임 1개를 반환
    def wait_new(self, timeout_s: float = 1.0) -> np.ndarray:

        t_end = time.monotonic() + float(timeout_s)

        with self._cond:
            while True:
                if self.latest_rgb is not None and self._frame_counter != self._last_served_counter:
                    self._last_served_counter = self._frame_counter
                    return self.latest_rgb.copy()

                remaining = t_end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("wait_new timeout")

                self._cond.wait(timeout=remaining)


def capture_and_send(
    buffer: "FrameBuffer",          # ros callback buffer
    server_url: str,                # http url
    place_id: str,
    mode: Literal["bank", "th_calib", "query"], # "normal", "query"
    n_frames: int = 10,
    sample_dt: float = 0.2,
    capture_timeout_s: float = 5.0, # capture timeout
    post_timeout_s: float = 5.0,    # http timeout
    gt = None,
) -> Dict[str, Any]:
    """
    ros node에서 update호출로 채워지는 buffer에서 [n_frames] 을 sample_dt간격으로 뽑아 http로 보냄
    """

    if mode not in ("bank", "th_calib", "query"):
        raise ValueError("mode must be 'bank' or 'query' or 'th_calib'")

    frames = buffer.capture_n(
        n_frames=n_frames,
        sample_dt=sample_dt,
        timeout_s=capture_timeout_s,
    )

    # meta
    meta = {
        "place_id": place_id,
        "timestamp": datetime.now().isoformat(),
        "n_frames": len(frames),
        "mode" : mode,
        "label" : gt # 평상시 None, 정답이 주어진 경우에는 normal, unnomal
    }

    return post_batch(
        server_url=server_url,
        images_rgb=frames,
        meta=meta,
        timeout_s=post_timeout_s,
    )


"""
use case -------------------------------------------

[ROS2 노드에서 cap_and_send.py 사용 흐름]

1) 이미지 토픽 구독 콜백에서 항상 buffer.update(rgb)를 호출
2) 필요할 때(서비스/버튼/GUI 요청 등) capture_and_send(...)를 호출한다.
   - capture_and_send는 n_frames를 모을 때까지 기다리고 + HTTP 전송까지 하므로 블로킹이다.
   - 그래서 ROS 콜백/서비스 콜백에서는 보통 별도 thread로 실행한다.
3) (선택) 실시간 스트리밍/실시간 처리(추론/송출)가 필요하면 wait_new()를 별도 thread 루프에서 사용한다.

----------------------------------------------------
(예) ROS2 노드 pseudo code

from cap_and_send import FrameBuffer, capture_and_send
import threading

buffer = FrameBuffer()

def image_callback(msg):
    rgb = msg_to_rgb_numpy(msg)   # cv_bridge 등으로 변환
    buffer.update(rgb)

# 스트리밍/실시간 처리용
def stream_worker():
    while True:
        try:
            rgb = buffer.wait_new(timeout_s=1.0)
            process(rgb)  # GUI 송출 
        except TimeoutError:
            # 프레임이 안 오는 상황: 카메라 상태 로그 정도만
            continue

threading.Thread(target=stream_worker, daemon=True).start()

# [B] 장소 사진 캡처+전송(필요할 때 호출)
def trigger_capture_send(place_id: str):
    def worker():
        try:
            out = capture_and_send(
                buffer=buffer,
                server_url="http://127.0.0.1:8000",
                place_id=place_id,
                n_frames=10,
                sample_dt=0.2,
                capture_timeout_s=5.0,
                post_timeout_s=5.0,
            )
            print("capture_and_send ok:", out)
        except Exception as e:
            print("capture_and_send failed:", e)

    threading.Thread(target=worker, daemon=True).start()
"""