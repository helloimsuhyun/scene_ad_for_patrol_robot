# Capstone Server & GUI 실행 가이드

`scene_ad_for_patrol_robot` 프로젝트를 다른 PC에서 실행하기 위한 최소 가이드입니다.

## 1. Miniconda 설치

Conda가 없다면 먼저 설치합니다.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 2. Conda 환경 생성

```bash
cd ~/scene_ad_for_patrol_robot
conda env create -f environment.yml
conda activate capston_server
```

GPU 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 3. 서버 실행

```bash
cd ~/scene_ad_for_patrol_robot
conda activate capston_server
./run_servers.sh
```

## 4. Flutter GUI 개발 실행

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter run -d chrome
```

## 5. Flutter Web 빌드 후 실행

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter build web

cd build/web
python3 -m http.server 8080
```

접속 주소:

```text
http://localhost:8080
```

다른 PC에서 접속:

```text
http://서버PC_IP:8080
```

서버 PC IP 확인:

```bash
hostname -I
```

## 참고

`environment.yml`은 Conda 환경과 Python 패키지를 복원합니다.  
단, NVIDIA 드라이버와 GPU 자체는 설치하지 않습니다.

기본 포트:

| Component | Port |
|---|---:|
| Vision Server | 8000 |
| Stream Server | 8001 |
| Flutter Web GUI | 8080 |