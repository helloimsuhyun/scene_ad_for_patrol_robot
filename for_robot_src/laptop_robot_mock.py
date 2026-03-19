import cv2
import threading
import time
import uvicorn
import requests
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from webrtc_sender import WebRTCSender

# ==============================================================
# 🚀 1. 핵심 통신 설정 (반드시 데스크탑 PC의 IP 주소로 수정하세요!)
# ==============================================================
DESKTOP_IP = "192.168.0.24" # <-- 여기에 데스크탑(관제서버)의 내부 IP를 넣으세요!

VISION_SERVER_URL = f"http://{DESKTOP_IP}:8000"
SIGNALING_SERVER_URL = f"http://{DESKTOP_IP}:8001"

# ==============================================================
# 🎥 2. 웹캠 캡처용 프레임 버퍼 생성 (카메라에서 사진을 퍼나르는 역할)
# ==============================================================
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def update(self, frame):
        with self.condition:
            # OpenCV 기본 BGR 포맷을 WebRTC 호환을 위해 RGB로 변환
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.condition.notify_all()

    def wait_new(self, timeout=1.0):
        with self.condition:
            self.condition.wait(timeout)
            return self.frame

frame_buffer = FrameBuffer()

def camera_thread():
    print("[MOCK] 로컬 웹캠을 시작합니다...")
    cap = cv2.VideoCapture(0) # 0번 메인웹캠 연결. 외부 캠이면 1을 시도해보세요.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # 웹캠 프레임을 버퍼에 밀어넣음 -> webrtc_sender가 낚아채서 데스크탑으로 전송
        frame_buffer.update(frame)
        time.sleep(1/30) # 30 FPS 제한

# ==============================================================
# 📡 3. FastAPI 로컬 수신 서버 (플러터 GUI의 버튼 명령을 받는 역할)
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

class QueryModel(BaseModel):
    label: str

@app.on_event("startup")
def startup_event():
    global webrtc_sender
    # 카메라 쓰레드 가동
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    # WebRTC 영상 송출 가동 (목적지: 데스크탑 Signaling 서버)
    print(f"[MOCK] 🌐 WebRTC 스트리밍 시작 (목적지: {SIGNALING_SERVER_URL})")
    webrtc_sender = WebRTCSender(buffer=frame_buffer, signaling_base_url=SIGNALING_SERVER_URL)
    webrtc_sender.start()

@app.on_event("shutdown")
def shutdown_event():
    global webrtc_sender
    if webrtc_sender:
        webrtc_sender.stop()

current_label = "normal"
current_place_id = "00"

@app.post("/patrol/capture")
async def patrol_capture():
    print("📸 [MOCK] 플러터에서 캡처 명령(c) 수신!")
    img_bgr = frame_buffer.wait_new(1.0)
    if img_bgr is None:
        return {"status": "error", "message": "no camera frame"}
    
    # BGR -> RGB (frame_buffer.update가 RGB로 변환하므로 현재 img_bgr은 RGB임)
    # 그러나 보통 cv2로 다시 저장할 때는 BGR이어야 하므로 색상 반전 처리
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    
    meta_obj = {
        "place_id": current_place_id,
        "timestamp": datetime.now().isoformat(),
        "n_frames": 1,
        "mode": "query",
        "label": current_label  # 'normal' 또는 'abnormal'
    }

    try:
        resp = requests.post(
            f"{VISION_SERVER_URL}/place_imgs",
            data={"meta": json.dumps(meta_obj)},
            files=[("images", ("capture.jpg", buffer.tobytes(), "image/jpeg"))],
            timeout=5.0
        )
        print(f"✅ 비전 서버 전송 완료: {resp.status_code}")
    except Exception as e:
        print(f"❌ 비전 서버 전송 실패: {e}")

    import base64
    b64_img = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return {"status": "Mock capture triggered", "image_b64": b64_img}

@app.post("/patrol/place_and_capture")
async def patrol_place_and_capture():
    print("📍📸 [MOCK] 플러터에서 이동+캡처 명령(v) 수신!")
    result = await patrol_capture()
    return result

@app.post("/patrol/query_gt")
async def patrol_query_gt(req: QueryModel):
    global current_label
    current_label = req.label
    print(f"📝 [MOCK] 플러터에서 라벨 토글 (z) 수신: {req.label}")
    return {"status": "Mock label toggled", "label": req.label}

# ==============================================================
# 실행 메인
# ==============================================================
if __name__ == "__main__":
    print(f"============================================================")
    print(f"🤖 패트롤 로봇 테스트 환경이 준비되었습니다! (웹캠 연동)")
    print(f"1) 데스크탑에서 AI서버(8000) & 시그널링서버(8001)를 켭니다.")
    print(f"2) Flutter 프론트를 켜고 `SENTRYNEX Control > 제어` 로 이동합니다.")
    print(f"3) 연결이 안 될 경우 데스크탑의 내부망 IP({DESKTOP_IP})를 꼭 확인하세요.")
    print(f"============================================================")
    
    # 8090 포트로 켜면 Flutter GUI가 버튼 클릭 시 이곳으로 명령(c, v, z)을 보냄
    uvicorn.run(app, host="0.0.0.0", port=8090)
