# Capstone Server & GUI 실행 가이드

`scene_ad_for_patrol_robot` 프로젝트를 다른 PC에서 실행하기 위한 최소 가이드입니다.

구성 포트:

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
> PyTorch CUDA 런타임은 별도 pip 설치 명령으로 설치합니다.

---

## 2. Miniconda 설치

Conda가 없다면 먼저 설치합니다.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

확인:

```bash
conda --version
```

---

## 3. Conda 환경 생성

프로젝트 폴더로 이동합니다.

```bash
cd ~/scene_ad_for_patrol_robot
```

`environment.yml`을 사용해 `dl` 환경을 생성합니다.

```bash
conda env create -f environment.yml
conda activate dl
```

PyTorch CUDA 패키지를 별도로 설치합니다.

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

PyTorch / CUDA 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

마지막 값이 `True`이면 GPU 사용 가능 상태입니다.

---

## 4. 서버 실행

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

---

## 5. Flutter GUI 개발 실행

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter run -d chrome
```

---

## 6. Flutter Web 빌드 후 실행

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter build web

cd build/web
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

---

## 참고

`environment.yml`은 Conda 환경과 기본 Python 패키지를 복원합니다.  
PyTorch CUDA 패키지는 `pip install torch... --index-url https://download.pytorch.org/whl/cu121` 명령으로 별도 설치합니다.

NVIDIA 드라이버는 OS에 별도로 설치되어 있어야 하며, `nvidia-smi`가 정상 동작해야 GPU 추론을 사용할 수 있습니다.