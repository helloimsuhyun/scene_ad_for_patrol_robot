import requests
import json
import time
import os
from pathlib import Path
from datetime import datetime

#---------- 서버 및 데이터 설정 ----------
SERVER_URL = "http://localhost:8000/place_imgs"
PLACE_ID = "00"
QUERY_DIR = Path("/home/choisuhyun/scene_ad_for_patrol_robot/data/dataset/00/query")


def send_simulation_request(image_path: Path):
    """
    로봇이 이미지를 찍어서 보낸 것처럼 HTTP POST 요청을 시뮬레이션합니다.
    """
    # 파일 이름에서 라벨 추출 (normal_... vs abnormal_...)
    if "abnormal" in image_path.name:
        label = "unnormal" # 서버에서 "unnormal"로 인식하므로 변환
    else:
        label = "normal"
    
    # meta 데이터 구성
    meta = {
        "place_id": PLACE_ID,
        "timestamp": datetime.now().isoformat(),
        "n_frames": 1,
        "mode": "query",
        "label": label
    }
    
    # multipart/form-data 요청 구성
    ext = image_path.suffix.lower()
    mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    
    files = [
        ('images', (image_path.name, open(image_path, 'rb'), mime_type))
    ]
    data = {
        'meta': json.dumps(meta)
    }
    
    try:
        print(f" >>> Sending [{label}] image: {image_path.name}")
        response = requests.post(SERVER_URL, files=files, data=data)
        
        if response.status_code == 200:
            res_json = response.json()
            print(f" <<< Success: Status={res_json.get('status')}")
        else:
            print(f" !!! Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f" !!! Connection Failed: {e}")
    finally:
        # 파일 핸들 닫기
        for _, (_, f, _) in files:
            f.close()

def run_simulation():
    if not QUERY_DIR.exists():
        print(f"Error: Query directory not found at {QUERY_DIR}")
        return

    # query 폴더 내 이미지 리스트 확보 (png, jpg, jpeg 모두 포함)
    all_imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        all_imgs.extend(list(QUERY_DIR.glob(ext)))
    all_imgs = sorted(all_imgs)
    
    if not all_imgs:
        print("No images found in query directory.")
        return

    print(f"--- Starting Robot Simulation with {len(all_imgs)} images ---")
    print(f"Target Server: {SERVER_URL}")
    print("---------------------------------------------------------")

    try:
        for img_path in all_imgs:
            send_simulation_request(img_path)
            # 2초 간격으로 전송 (플러터 인터페이스 변화 관찰용)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    run_simulation()
