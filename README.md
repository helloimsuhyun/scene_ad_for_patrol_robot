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

Python 버전을 확인합니다.

```bash
python -V
which python
```

> Python은 3.10 또는 3.11 사용을 권장합니다.  
> PyTorch `2.5.1+cu121` 설치가 실패하면 Python 버전이 맞지 않은 경우가 많습니다.

PyTorch CUDA 패키지를 별도로 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

PyTorch / CUDA 확인:

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