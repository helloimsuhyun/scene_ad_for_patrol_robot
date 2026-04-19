# Conda 환경 설정 및 필요 패키지 정리 가이드

의존 패키지 설치 및 환경설정 가이드

### [Component Name] Conda 설치 및 환경 구축

Conda가 설치되어 있지 않은 사용자를 위한 가이드와 통합 환경 설정 파일입니다.

#### 1. Miniconda 설치 (Linux)
```bash
# 다운로드 및 자동 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### [NEW] [environment.yml](file:///home/choisuhyun/scene_ad_for_patrol_robot/environment.yml)
Conda에서 즉시 설치 가능한 환경 설정 파일입니다.

```bash
# 1. 환경 생성 및 모든 패키지 동시 설치
conda env create -f environment.yml

# 2. 환경 활성화
conda activate capston_server

```

#### 파이썬 패키지 상세 목록 (참고용)
위의 [environment.yml](file:///home/choisuhyun/scene_ad_for_patrol_robot/environment.yml)을 사용하면 아래 패키지들이 한 번에 설치됩니다:
- **DL/CV**: `torch`, `torchvision`, `opencv-python`, `matplotlib`, `numpy`, `Pillow`
- **Backend**: `fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `requests`
- **Streaming/Robot**: `aiortc`, [av](file:///home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/vision_server/vis.py#578-681)
- **Utils**: `pyyaml`, `scipy`

### [Component Name] Flutter 프런트엔드

`sentrynexcontrol` 폴더의 GUI 앱을 위한 설정입니다.

- **Flutter SDK**: `^3.8.1` (버전 확인 필수: `flutter --version`)
- **패키지 설치**: `sentrynexcontrol` 폴더에서 아래 명령 실행
  ```bash
  flutter pub get
  ```
- **주요 의존성**:
  - `flutter_riverpod`: 상태 관리
  - `flutter_webrtc`: 실시간 영상 스트리밍
  - `audioplayers`: 오디오 이벤트 재생
  - `http`: API 통신

## Verification Plan


### 서버 실행 방법
1. [run_servers.sh](file:///home/choisuhyun/scene_ad_for_patrol_robot/run_servers.sh) 실행 시 포트 8000, 8001이 정상적으로 열리는지 확인.
2. `sentrynexcontrol`에서 `flutter run` 실행 시 빌드 오류가 없는지 확인.

# 오프라인 실행 가이드

플러터 사전 빌드
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter build web


cd ~/scene_ad_for_patrol_robot/sentrynexcontrol/build/web
python3 -m http.server 8080

같은 pc
http://localhost:8080 


다른 pc
http://서버PC_IP:8080