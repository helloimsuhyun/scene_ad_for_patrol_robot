# 환경세팅 및 실행 가이드

> Git repository: `https://github.com/H3cRobotics/capston_patrol_server.git`  
> 로컬 프로젝트 폴더 이름: `scene_ad_for_patrol_robot`

---

## 구성 포트

| Component | Port |
|---|---:|
| Vision Server | 8000 |
| Stream Server | 8001 |
| Flutter Web GUI | 8095 |

---

# 1. 환경설정

## 1-1. NVIDIA 드라이버 확인 및 설치

GPU 추론을 사용하려면 NVIDIA 드라이버가 설치되어 있어야 합니다.

설치 확인:

```bash
nvidia-smi
```

정상이라면 GPU 정보가 출력됩니다.

드라이버가 없거나 오류가 나면 Ubuntu 기준으로 아래 명령을 실행합니다.

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

재부팅 후 다시 확인합니다.

```bash
nvidia-smi
```

> CUDA Toolkit은 일반적으로 별도 설치하지 않아도 됩니다.  
> PyTorch CUDA 런타임은 Conda 환경 생성 후 별도 pip 명령으로 설치합니다.

---

## 1-2. Miniconda 설치

Conda가 없다면 먼저 Miniconda를 설치합니다.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

설치 확인:

```bash
conda --version
```

---

## 1-3. 프로젝트 다운로드

프로젝트를 Git에서 다운로드합니다.

```bash
cd ~
git clone https://github.com/H3cRobotics/capston_patrol_server.git scene_ad_for_patrol_robot
cd ~/scene_ad_for_patrol_robot
```

---

## 1-4. Conda 환경 생성

프로젝트 폴더로 이동합니다.

```bash
cd ~/scene_ad_for_patrol_robot
```

`environment_min.yml`을 사용해 `dl` 환경을 생성합니다.

```bash
conda env create -f environment_min.yml
conda activate dl
```

Python 버전을 확인합니다.

```bash
python -V
which python
```

정상 예시:

```text
Python 3.10.x
/home/<USER>/miniconda3/envs/dl/bin/python
```

> `run_servers.sh`는 내부적으로 `$HOME/miniconda3/envs/dl/bin/python`을 사용합니다.  
> 따라서 Conda 환경 이름은 `dl`로 맞춰야 합니다.

---

## 1-5. PyTorch CUDA 설치

PyTorch CUDA 패키지를 별도로 설치합니다.

```bash
conda activate dl
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

PyTorch / CUDA 사용 가능 여부를 확인합니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

정상 예시:

```text
2.5.1+cu121
12.1
True
```

마지막 값이 `True`이면 GPU 사용 가능 상태입니다.

---

## 1-6. Flutter SDK 설치

Flutter Web GUI 빌드를 위해 Flutter SDK가 필요합니다.

설치 확인:

```bash
flutter --version
```

`flutter: command not found`가 나오면 아래 명령으로 설치합니다.

```bash
cd ~
git clone https://github.com/flutter/flutter.git -b stable
echo 'export PATH="$HOME/flutter/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
flutter --version
```

---

## 1-7. Flutter Web 최초 빌드

Flutter GUI를 Web으로 빌드합니다.

```bash
cd ~/scene_ad_for_patrol_robot
./build_web_gui.sh
```

빌드 결과는 아래 폴더에 생성됩니다.

```text
~/scene_ad_for_patrol_robot/sentrynexcontrol/build/web
```

> 최초 빌드 또는 패키지 변경 후 빌드는 인터넷이 필요합니다.

---

# 2. 실행

## 2-1. 백엔드 서버 실행

터미널 1에서 실행합니다.

```bash
cd ~/scene_ad_for_patrol_robot
conda activate dl
./run_servers.sh
```

---

## 2-2. Flutter Web GUI 실행

터미널 2에서 실행합니다.

```bash
cd ~/scene_ad_for_patrol_robot
./run_web_gui.sh 8095
```

접속 주소:

```text
같은 PC  : http://localhost:8095
다른 PC : run_web_gui.sh 실행 시 출력되는 http://서버PC_IP:8095
```

---

# 3. 전체 실행 순서 요약

## 3-1. 서버 실행

터미널 1:

```bash
cd ~/scene_ad_for_patrol_robot
conda activate dl
./run_servers.sh
```

## 3-2. GUI 실행

터미널 2:

```bash
cd ~/scene_ad_for_patrol_robot
./run_web_gui.sh 8095
```

## 3-3. 접속

```text
같은 PC  : http://localhost:8095
다른 PC : run_web_gui.sh 실행 시 출력되는 http://서버PC_IP:8095
```

---

# 4. 관리자 기능: 이벤트 초기화

> 백엔드 서버(`./run_servers.sh`)가 실행 중인 상태에서 다른 터미널에서 실행합니다.

```bash
cd ~/scene_ad_for_patrol_robot
conda activate dl
```

## 전체 초기화

```bash
python3 reset_events_admin.py --all
```

## 비전, 오디오, 인증만 초기화

```bash
python3 reset_events_admin.py --core
```

## 특정 이벤트만 초기화

```bash
python3 reset_events_admin.py --vision
python3 reset_events_admin.py --audio
python3 reset_events_admin.py --auth
python3 reset_events_admin.py --yolo
```

## DB만 초기화하고 파일은 유지

```bash
python3 reset_events_admin.py --all --db_only
```