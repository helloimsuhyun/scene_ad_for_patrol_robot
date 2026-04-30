import subprocess
import time
import sys
import os

def launch():
    print("🚀 SENTRYNEX Master Launcher 실행 중...")
    
    # 각 서버 파일의 절대 경로 설정
    base_dir = os.getcwd()
    vision_server = os.path.join(base_dir, "sentrynexcontrol", "vision_server", "http_server.py")
    stream_server = os.path.join(base_dir, "sentrynexcontrol", "stream_server", "signaling_server.py")
    
    print(f"1. HTTP 서버 시작 (Port: 8000)...")
    p1 = subprocess.Popen([sys.executable, vision_server], creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    print(f"2. 시그널링 서버 시작 (Port: 8001)...")
    p2 = subprocess.Popen([sys.executable, stream_server], creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    time.sleep(2) # 서버가 뜰 때까지 잠시 대기
    
    print(f"3. ngrok 터널 활성화...")
    try:
        p3 = subprocess.Popen(["ngrok", "http", "8000"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    except FileNotFoundError:
        print("❌ ngrok을 찾을 수 없습니다. 환경변수(PATH)에 등록되어 있는지 확인해주세요.")
    
    print("\n✅ 모든 서버가 별도 창에서 실행되었습니다.")
    print("통계, 로그, 맵 데이터를 보려면 ngrok 화면에 뜬 https 주소를 앱 설정에 넣으세요.")
    print("발표 시에는 이 창을 띄워두고 웹사이트에 접속만 하면 됩니다.")
    
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n종료 중...")
        p1.terminate()
        p2.terminate()

if __name__ == "__main__":
    launch()
