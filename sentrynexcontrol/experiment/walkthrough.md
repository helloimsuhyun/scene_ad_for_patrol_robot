# Patrol Robot Anomaly Detection Pipeline Walkthrough

## 1. 개요 (Overview)
본 프로젝트는 순찰 로봇의 이상 감지를 위해 **DINOv2 기반의 시각적 유사도**와 **SAM(Segment Anything) 기반의 기하학적 정밀 검증**을 결합한 하이브리드 파이프라인을 사용합니다. 최근 발생한 코드 유실 사고 이후, 가장 안정적이고 성능이 뛰어난 'DINO+SAM 하이브리드' 버전으로 복구되었습니다.

## 2. 주요 복구 기능 (Restored Features)

### A. DINOv2 Top-M 사전 선택 (Global Pre-selection)
- 수많은 Reference Bank 이미지 중 쿼리와 가장 유사한 상위 $M$개(기본값 3)를 DINOv2 `[CLS]` 토큰 유사도로 먼저 뽑습니다.
- **효과**: 모든 뱅크에 대해 SuperGlue를 돌릴 필요가 없어 속도가 획기적으로 개선되었습니다.

### B. 하이브리드 검증 모드 (Hybrid Refinement)
- `refine_mode`: `sam`, `dino`, `hybrid` 중 선택 가능.
- **Hybrid 로직**:
  - DINO 유사도 > 0.85: "동일 물체(조명 노이즈)"로 판단하여 즉시 기각 (**False Positive 방지**).
  - DINO 유사도 < 0.65: "확실한 이상"으로 판단하여 SAM IoU 검증을 우회하고 확정 (**Bypass**).
  - 그 사이: SAM IoU (임계치 0.60)를 통해 형태적 변화를 최종 확인.

### C. 가독성 높은 시각화 (Visualization)
- `dino_overlay.png`: 전체 Query 이미지 위에 탐지된 후보들의 DINO 유사도 점수와 상태(Confirmed/Rejected)를 오버레이합니다.
- `sam_compare.png`: 정합된 Reference와 Query의 차이를 직관적으로 비교합니다.

## 3. 실행 방법 (Execution)

### 이상 감지 수행 (Inference)
```bash
python3 ex.py --place 08 --mode infer \
    --dino_model dinov2_vits14 \
    --dino_top_m 3 \
    --refine_mode hybrid
```

### 주요 파라미터 임계치
| 파라미터 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `dino_sim_reject_thresh` | 0.85 | 이 값보다 유사하면 동일한 질감(노이즈)으로 간주하고 무시함 |
| `dino_sim_bypass_thresh` | 0.65 | 이 값보다 유사도가 낮으면 확연한 다른 물체로 간주하고 SAM 없이 확정 |
| `top_p` | 0.10 | Diff Map에서 상위 10% 픽셀을 후보로 추출 |

## 4. 복구 결과 확인
- `ex.py`: 뱅크 사전 선택 및 매칭 로직 복구 완료.
- `sam_refine.py`: DINO semantic check 및 SAM XOR 로직 복구 완료.
- `matcher.py`, `dino_emb.py`: 기존 모듈과의 API 정합성 확인 완료.
