## 🔎 k-NN 기반 이상 감지 (Scene Anomaly Detection)

본 프로젝트는 **k-NN 임베딩 거리 기반 이상 감지(Anomaly Detection)** 시스템을 구현한 코드이다.

이미지로부터 **DINOv3 feature embedding**을 추출하고, 정상 데이터로 구성된 **reference bank**와의 k-NN 거리 비교를 통해 이상 여부를 판단한다.

또한 데이터 및 환경 변화에 대응하기 위해 **Adaptive Threshold 기반 자동 임계값 설정**을 적용하여 보다 안정적인 이상 감지를 수행한다.

본 파이프라인은 **순찰 로봇의 장소(place)별 상태 감지**를 목표로 설계되었으며, 모델 개발 및 검증을 위해 **MVTec AD 데이터셋**과 실제 순찰 환경에서 수집한 데이터를 사용하였다.

---

## ⚙️ Pipeline Overview

Image  
→ DINOv3 Embedding Extraction  
→ Place-specific Reference Bank  
→ k-NN Distance Computation  
→ Adaptive Thresholding  
→ Anomaly Detection

---


## 📈 MVTec AD 실험 결과 요약

MVTec AD 데이터셋 일부 클래스에 대해 실험을 수행하였으며, 클래스별 난이도 차이가 크게 나타났다.

### ✅ Easy Classes (높은 성능)
- `grid`, `hazelnut`
- F1 Score: **94–97**
- 낮은 FPR, 안정적인 검출

### ⚖️ Medium Classes (튜닝 시 개선 가능)
- `cable`, `bottle`, `toothbrush`
- F1 Score: **84–85**
- 일부 정상 이미지 오탐 발생

### ⚠️ Hard Classes (Global Embedding 한계)
- `screw`, `transistor`
- 작은 결함에서 Recall 감소
- Patch-level scoring 등 추가 개선 필요

> Global embedding 기반 방식의 한계가 잘 나타난 결과이다.

---

## 📷 Demo Examples (MVTec)

<p align="center">
  <img src="./example_chg.png" width="745"/>
</p>

<p align="center">
  <img src="./example_leather_cls_query_dist_curve.png" width="500"/>
</p>

---

## 🏭 실제 순찰 환경 데이터 실험

실제 순찰 환경에서 수집된 데이터(`place=00`)에 대해 동일한 파이프라인을 적용하여 성능을 평가하였다.

### 실험 설정
- k = 3
- Percentile = 97
- Threshold = 0.0264

### Confusion Matrix

|               | Pred Normal | Pred Anomaly |
|--------------|------------|--------------|
| **Normal**   | 62         | 5            |
| **Anomaly**  | 6          | 47           |

### Evaluation Metrics

- Accuracy: **90.83%**
- Recall: **88.68%**
- Precision: **90.38%**
- F1 Score: **89.52%**
- FPR: **7.46%**
- FNR: **11.32%**

총 120장의 이미지(정상 67장, 이상 53장)에 대해 평가하였다.

본 실험을 통해 실제 순찰 환경에서 해당 이상 감지 시스템 적용 가능성을 확인하였다.

---

## 📷 Demo Examples (Real Patrol Environment)

<p align="center">
  <img src="./exam_chg.png" width="745"/>
</p>

<p align="center">
  <img src="./exam_chg2.png" width="745"/>
</p>

<p align="center">
  <img src="./exam_query_dist_curve.png" width="500"/>
</p>

---

## 🚀 Future Improvements

- Patch-level anomaly localization
- SAM 기반 object-level scoring
- RGB-D 정합 기반 구조적 변화 감지
- 실시간 로봇 시스템 통합

---

## 📚 References

- Deep Nearest Neighbor Anomaly Detection — NeurIPS 2022  
- An Anomaly Detection System via Moving Surveillance Robots with Human — IROS 2021  
- Semantic Scene Difference Detection in Daily Life Patroling by Mobile Robots using Pre-Trained Large-Scale Vision-Language Model — ICRA 2024