## 서버 실행

```bash
cd ~/scene_ad_for_patrol_robot
./run_servers.sh
```

서버 포트:

```text
Vision Server : 8000
Stream Server : 8001
```

> `run_servers.sh`는 `$HOME/miniconda3/envs/dl/bin/python`을 사용하므로 Conda 환경 이름은 `dl`이어야 합니다.

---

## Flutter GUI 실행

### 1. Web 빌드

인터넷이 되는 환경에서 한 번만 실행합니다.

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol
flutter pub get
flutter build web
```

빌드 결과는 아래 폴더에 생성됩니다.

```text
~/scene_ad_for_patrol_robot/sentrynexcontrol/build/web
```

---

### 2. Web GUI 실행

빌드가 이미 완료되어 있다면 오프라인에서도 실행할 수 있습니다.

```bash
cd ~/scene_ad_for_patrol_robot/sentrynexcontrol/build/web
python3 -m http.server 8080
```

접속 주소:

```text
같은 PC : http://localhost:8080
다른 PC : http://서버PC_IP:8080
```

서버 PC IP 확인:

```bash
hostname -I
```