import cv2
import threading
import time
import uvicorn
import requests
import json
import random
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from webrtc_sender import WebRTCSender

# ==============================================================
# 설정
# ==============================================================
DESKTOP_IP = "192.168.0.24"

VISION_SERVER_URL = f"http://{DESKTOP_IP}:8000"
SIGNALING_SERVER_URL = f"http://{DESKTOP_IP}:8001"

POSE_UPDATE_PERIOD = 1.0
GOAL_UPDATE_PERIOD = 15.0
POSE_POST_PERIOD = 0.5
GOAL_POST_PERIOD = 0.5
COMMAND_POLL_PERIOD = 0.3

PLACE_IDS = ["00", "01", "02", "03", "04"]

# ==============================================================
# FrameBuffer
# ==============================================================
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def update(self, frame):
        with self.condition:
            # OpenCV 캡처는 BGR, WebRTC용으로 RGB 저장
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.condition.notify_all()

    def wait_new(self, timeout=1.0):
        with self.condition:
            self.condition.wait(timeout)
            return self.frame


frame_buffer = FrameBuffer()

# ==============================================================
# 로봇 상태 (mock)
# status는 command polling 결과로만 바뀌도록 유지
# x, y, yaw는 1초마다 랜덤 변경
# goal_x, goal_y, goal_yaw, next_place_id는 15초마다 랜덤 변경
# ==============================================================
robot_state = {
    "x": 0.0,
    "y": 0.0,
    "yaw": 0.0,
    "status": "idle",
    "goal_x": 0.0,
    "goal_y": 0.0,
    "goal_yaw": 0.0,
    "next_place_id": "00",
}
state_lock = threading.Lock()

last_command = None

# ==============================================================
# 카메라
# ==============================================================
def camera_thread():
    print("[MOCK] local webcam start")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ camera open failed")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            frame_buffer.update(frame)
        time.sleep(1 / 30)


# ==============================================================
# mock pose 랜덤 업데이트 (1초)
# ==============================================================
def random_pose_update():
    while True:
        with state_lock:
            robot_state["x"] += random.uniform(-0.3, 0.3)
            robot_state["y"] += random.uniform(-0.3, 0.3)
            robot_state["yaw"] += random.uniform(-0.2, 0.2)

            # yaw 범위 정리
            if robot_state["yaw"] > 3.141592:
                robot_state["yaw"] -= 2 * 3.141592
            elif robot_state["yaw"] < -3.141592:
                robot_state["yaw"] += 2 * 3.141592

        time.sleep(POSE_UPDATE_PERIOD)


# ==============================================================
# mock goal 랜덤 업데이트 (15초)
# ==============================================================
def random_goal_update():
    while True:
        with state_lock:
            robot_state["goal_x"] = random.uniform(-5.0, 5.0)
            robot_state["goal_y"] = random.uniform(-5.0, 5.0)
            robot_state["goal_yaw"] = random.uniform(-3.141592, 3.141592)
            robot_state["next_place_id"] = random.choice(PLACE_IDS)

            goal_snapshot = {
                "goal_x": robot_state["goal_x"],
                "goal_y": robot_state["goal_y"],
                "goal_yaw": robot_state["goal_yaw"],
                "next_place_id": robot_state["next_place_id"],
            }

        print(f"[MOCK GOAL UPDATE] {goal_snapshot}")
        time.sleep(GOAL_UPDATE_PERIOD)


# ==============================================================
# Vision Server로 pose 전송
# ==============================================================
def post_robot_pose():
    while True:
        try:
            with state_lock:
                payload = {
                    "x": robot_state["x"],
                    "y": robot_state["y"],
                    "yaw": robot_state["yaw"],
                    "status": robot_state["status"],
                    "timestamp": datetime.now().isoformat(),
                }

            requests.post(
                f"{VISION_SERVER_URL}/robot/pose",
                json=payload,
                timeout=2,
            )

        except Exception as e:
            print(f"[POSE ERROR] {e}")

        time.sleep(POSE_POST_PERIOD)


# ==============================================================
# Vision Server로 goal 전송
# ==============================================================
def post_robot_goal():
    last_sent = None

    while True:
        try:
            with state_lock:
                payload = {
                    "x": robot_state["goal_x"],
                    "y": robot_state["goal_y"],
                    "yaw": robot_state["goal_yaw"],
                    "next_place_id": robot_state["next_place_id"],
                    "timestamp": datetime.now().isoformat(),
                }

            if payload != last_sent:
                requests.post(
                    f"{VISION_SERVER_URL}/robot/goal",
                    json=payload,
                    timeout=2,
                )
                last_sent = payload.copy()

        except Exception as e:
            print(f"[GOAL ERROR] {e}")

        time.sleep(GOAL_POST_PERIOD)


# ==============================================================
# 서버 -> 로봇 command polling
# status만 여기서 반영
# ==============================================================
def poll_robot_command():
    global last_command

    while True:
        try:
            resp = requests.get(f"{VISION_SERVER_URL}/robot/command", timeout=2)
            data = resp.json()
            cmd = data.get("command")

            if not cmd or cmd == last_command:
                time.sleep(COMMAND_POLL_PERIOD)
                continue

            print(f"[COMMAND RECEIVED] {cmd}")
            last_command = cmd

            with state_lock:
                if cmd == "start":
                    robot_state["status"] = "patrol"
                elif cmd == "pause":
                    robot_state["status"] = "pause"
                elif cmd == "resume":
                    robot_state["status"] = "patrol"
                elif cmd == "stop":
                    robot_state["status"] = "idle"
                elif cmd == "idle":
                    robot_state["status"] = "idle"
                elif cmd == "teach":
                    robot_state["status"] = "teach"
                else:
                    # 알 수 없는 command는 로그만 남기고 상태 유지
                    print(f"[COMMAND WARNING] unknown command: {cmd}")

        except Exception as e:
            print(f"[COMMAND POLL ERROR] {e}")

        time.sleep(COMMAND_POLL_PERIOD)


# ==============================================================
# FastAPI
# ==============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

webrtc_sender = None

# ==============================================================
# Startup / Shutdown
# ==============================================================
@app.on_event("startup")
def startup_event():
    global webrtc_sender

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=random_pose_update, daemon=True).start()
    threading.Thread(target=random_goal_update, daemon=True).start()
    threading.Thread(target=post_robot_pose, daemon=True).start()
    threading.Thread(target=post_robot_goal, daemon=True).start()
    threading.Thread(target=poll_robot_command, daemon=True).start()

    print(f"[MOCK] WebRTC -> {SIGNALING_SERVER_URL}")
    webrtc_sender = WebRTCSender(
        buffer=frame_buffer,
        signaling_base_url=SIGNALING_SERVER_URL,
    )
    webrtc_sender.start()


@app.on_event("shutdown")
def shutdown_event():
    global webrtc_sender
    if webrtc_sender:
        webrtc_sender.stop()


# ==============================================================
# Capture / Query label
# ==============================================================
current_label = "normal"
current_place_id = "00"


class QueryModel(BaseModel):
    label: str


@app.post("/patrol/capture")
async def patrol_capture():
    print("📸 [MOCK] capture command received")

    img_rgb = frame_buffer.wait_new(timeout=1.0)
    if img_rgb is None:
        return {"ok": False, "message": "no frame"}

    # frame_buffer에는 RGB로 저장되어 있으므로 저장/인코딩용으로 BGR 재변환
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return {"ok": False, "message": "jpeg encode failed"}

    meta = {
        "place_id": current_place_id,
        "timestamp": datetime.now().isoformat(),
        "n_frames": 1,
        "mode": "query",
        "label": current_label,
    }

    try:
        resp = requests.post(
            f"{VISION_SERVER_URL}/place_imgs",
            data={"meta": json.dumps(meta)},
            files=[("images", ("capture.jpg", buffer.tobytes(), "image/jpeg"))],
            timeout=5.0,
        )
        print(f"[CAPTURE -> VISION] status={resp.status_code}")
    except Exception as e:
        print(f"[CAPTURE ERROR] {e}")
        return {"ok": False, "message": str(e)}

    return {"ok": True}


@app.post("/patrol/place_and_capture")
async def patrol_place_and_capture():
    print("📍📸 [MOCK] place_and_capture command received")
    return await patrol_capture()


@app.post("/patrol/query_gt")
async def patrol_query_gt(req: QueryModel):
    global current_label
    current_label = req.label
    print(f"[MOCK] current query label -> {current_label}")
    return {"ok": True, "label": current_label}


# ==============================================================
# Mock 상태 수동 수정 API
# 랜덤 동작 중에도 테스트용으로 강제로 덮어쓸 수 있음
# ==============================================================
class MockPose(BaseModel):
    x: float
    y: float
    yaw: float
    status: Optional[str] = None


@app.post("/mock/pose")
async def set_pose(req: MockPose):
    with state_lock:
        robot_state["x"] = req.x
        robot_state["y"] = req.y
        robot_state["yaw"] = req.yaw
        if req.status is not None:
            robot_state["status"] = req.status

        snapshot = dict(robot_state)

    return {"ok": True, "state": snapshot}


class MockGoal(BaseModel):
    x: float
    y: float
    yaw: float
    next_place_id: Optional[str] = "00"


@app.post("/mock/goal")
async def set_goal(req: MockGoal):
    with state_lock:
        robot_state["goal_x"] = req.x
        robot_state["goal_y"] = req.y
        robot_state["goal_yaw"] = req.yaw
        robot_state["next_place_id"] = req.next_place_id or "00"

        snapshot = dict(robot_state)

    return {"ok": True, "state": snapshot}


@app.get("/mock/state")
async def get_mock_state():
    with state_lock:
        snapshot = dict(robot_state)
    return {"ok": True, "state": snapshot}


# ==============================================================
# Main
# ==============================================================
if __name__ == "__main__":
    print("============================================================")
    print("🤖 laptop_robot_mock start")
    print(f"VISION_SERVER_URL    = {VISION_SERVER_URL}")
    print(f"SIGNALING_SERVER_URL = {SIGNALING_SERVER_URL}")
    print("pose(x,y,yaw): 1초마다 랜덤 변경")
    print("goal: 15초마다 랜덤 변경")
    print("status: /robot/command polling 결과만 반영")
    print("============================================================")
    uvicorn.run(app, host="0.0.0.0", port=8090)