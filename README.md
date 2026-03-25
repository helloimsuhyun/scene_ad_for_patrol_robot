# 🔎 Scene Anomaly Detection (k-NN Embedding Based)

본 프로젝트는 **DINO feature embedding + k-NN 거리 기반 이상 감지 시스템**이다.  
순찰 로봇의 장소별 상태 변화를 자동 감지하는 것을 목표로 설계되었다.

---

## ⚙️ readme - ant_branch
원활한 gui 개발을 위해 gpu의존 .py와 추론을 더미화 하여 랜덤한 값을 추론값으로 내보내도록 하였다.

# 사용방법


# 서버실행
./run_servers.sh

# gui실헹

cd sentrynexcontrol
flutter run -d linux # 네트워크 있는 경우
flutter run -d web-server # 없는 경우 

#OFFLine 추론
python -m sentrynexcontrol.vision_server.Offline_eval