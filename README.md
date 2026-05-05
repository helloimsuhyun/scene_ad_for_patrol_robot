# Capstone Server & GUI 실행 가이드

이 문서는 `capston_patrol_server` 프로젝트를 다른 PC에서 실행하기 위한 최소 실행 가이드입니다.

> Git repository: `https://github.com/H3cRobotics/capston_patrol_server.git`  
> 로컬 프로젝트 폴더 이름: `scene_ad_for_patrol_robot`

---

## 구성 포트

| Component | Port |
|---|---:|
| Vision Server | 8000 |
| Stream Server | 8001 |
| Flutter Web GUI | 8080 |

---

## 1. NVIDIA 드라이버 확인 및 설치

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

## 2. Miniconda 설치

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

## 3. 프로젝트 다운로드

프로젝트를 Git에서 다운로드합니다.

```bash
cd ~
git clone https://github.com/H3cRobotics/capston_patrol_server.git scene_ad_for_patrol_robot
```

다운로드 후 프로젝트 폴더로 이동합니다.

```bash
cd ~/scene_ad_for_patrol_robot
```

> Git repository 이름은 `capston_patrol_server`이지만, 로컬 폴더 이름은 `scene_ad_for_patrol_robot`로 사용합니다.  
> Git repository 이름과 로컬 폴더 이름은 달라도 문제 없습니다.

---

## 4. Conda 환경 생성

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

> Python은 3.10 사용을 권장합니다.  
> `environment_min.yml`에서 `python=3.10`으로 고정되어 있어야 합니다.

PyTorch CUDA 패키지를 별도로 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

PyTorch / CUDA 사용 가능 여부를 확인합니다.
> 본 프로젝트에서 검증한 기본 PyTorch 버전은 `torch==2.5.1+cu121`입니다.  
> 단, 다른 PC에서 `torch.cuda.is_available()`이 `False`로 나오는 경우 NVIDIA 드라이버 버전을 확인하고, 해당 PC 환경에 맞는 PyTorch CUDA wheel을 설치해야 합니다.

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

## 5. 서버 실행

프로젝트 폴더에서 서버 실행 스크립트를 실행합니다.

```bash
cd ~/scene_ad_for_patrol_robot
conda activate dl
./run_servers.sh
```

서버는 기본적으로 다음 포트를 사용합니다.

```text
Vision Server : 8000
Stream Server : 8001
```

> 현재 `run_servers.sh`는 내부적으로 `$HOME/miniconda3/envs/dl/bin/python`을 사용합니다.  
> 따라서 Conda 환경 이름은 `dl`로 맞춰야 합니다.

서버 실행 여부는 다른 터미널에서 아래 명령으로 확인할 수 있습니다.

```bash
lsof -i :8000
lsof -i :8001
```

---

## 6. Flutter GUI 개발 실행

Flutter GUI 폴더로 이동합니다.

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
```

Flutter 패키지를 설치합니다.

```bash
flutter pub get
```

Chrome으로 실행합니다.

```bash
flutter run -d chrome
```

---

## 7. Flutter Web 빌드 후 실행

GUI를 Web으로 빌드합니다.

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter build web
```

빌드 결과 폴더로 이동합니다.

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol/build/web
python3 -m http.server 8080
```

같은 PC에서 접속:

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

접속 예시:

```text
http://192.168.0.221:8080
```

---

## 8. 전체 실행 순서 요약

### 서버 실행

```bash
cd ~/scene_ad_for_patrol_robot
conda activate dl
./run_servers.sh
```

### GUI 개발 실행

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter run -d chrome
```

### GUI Web 빌드 실행

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter build web
cd build/web
python3 -m http.server 8080
```

---

## 9. Conda 환경 재생성

이미 `dl` 환경이 존재해서 생성이 실패하면 아래 명령으로 기존 환경을 제거한 뒤 다시 생성합니다.

```bash
conda deactivate
conda env remove -n dl
cd ~/scene_ad_for_patrol_robot
conda env create -f environment_min.yml
conda activate dl
```

그 후 PyTorch CUDA 패키지를 다시 설치합니다.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip cache purge
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

PyTorch / CUDA 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## 10. 참고 사항

`environment_min.yml`은 Conda 환경과 기본 Python 패키지를 복원합니다.

PyTorch CUDA 패키지는 별도 pip 명령으로 설치합니다.

NVIDIA 드라이버는 OS에 별도로 설치되어 있어야 하며, `nvidia-smi`가 정상 동작해야 GPU 추론을 사용할 수 있습니다.

`run_servers.sh`가 `$HOME/miniconda3/envs/dl/bin/python`을 직접 사용하므로 Conda 환경 이름은 `dl`로 유지하는 것을 권장합니다.

Git repository 이름과 로컬 폴더 이름은 달라도 됩니다.

이 문서에서는 다음 구조를 기준으로 설명합니다.

```text
Git repository : https://github.com/H3cRobotics/capston_patrol_server.git
로컬 폴더 이름 : scene_ad_for_patrol_robot
```