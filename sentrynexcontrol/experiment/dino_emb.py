# dino_emb.py
# DINOv2 기반 패치 feature 추출기
# - ResNet과 동일한 인터페이스 (feat: [C, gh, gw])로 맞춰서
#   ex.py의 compute_dist_map_local_search 를 그대로 재사용 가능하게 함

import torch
import torch.nn.functional as F
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_dino_model(model_name: str = "dinov2_vits14", device=None):
    """
    DINOv2 모델 로드 (torch.hub 사용).
    model_name: "dinov2_vits14" | "dinov2_vitb14" | "dinov2_vitl14"
      - vits14: 경량 (추천 첫 실험용)
      - vitb14: 중간, 성능 ↑
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.eval().to(device)
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# 전처리 transform
# ─────────────────────────────────────────────────────────────────────────────

def make_dino_transform(img_size: int = 560):
    """
    DINOv2 입력 전처리.
    img_size는 patch_size(14)의 배수여야 함.
    560 = 14 * 40  →  40x40 패치 그리드
    """
    assert img_size % 14 == 0, f"img_size({img_size})는 14의 배수여야 합니다"
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 핵심: 패치 토큰을 spatial grid 형태로 추출
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_dino_grid(
    model,
    device,
    x,                      # PIL Tensor [3, H, W]
    patch_size: int = 14,
    n_last_blocks: int = 1, # 마지막 몇 개 블록을 평균낼지
):
    """
    DINOv2 중간 레이어 패치 토큰을 [C, gh, gw] spatial grid로 반환.

    왜 중간 레이어를 쓰나:
    - 마지막(최종) 레이어는 CLS token 쪽으로 정보가 집약돼 전역적
    - n_last_blocks >= 4 이면 중간~후기 레이어를 평균내어
      local texture + semantic 정보를 동시에 가짐 → 국소적 변화에 강함
    - 기존에 DINOv2가 국소 변화를 못 잡았던 이유:
      CLS token 또는 최종 레이어만 쓴 경우가 많음

    반환: feat [C, gh, gw], (gh, gw)
    """
    x = x.unsqueeze(0).to(device)   # [1, 3, H, W]
    _, _, H, W = x.shape
    gh = H // patch_size             # grid height (560/14 = 40)
    gw = W // patch_size             # grid width

    # ---------- 중간 레이어 patch token 추출 ----------
    # return_class_token=False → patch token만 반환
    # get_intermediate_layers: list of [1, N_patches, C]
    features = model.get_intermediate_layers(
        x, n=n_last_blocks, return_class_token=False
    )

    if n_last_blocks == 1:
        # 단일 레이어
        feat = features[0].squeeze(0)           # [N_patches, C]
    else:
        # 여러 레이어 평균 → 중간 레이어 정보 포함
        feat = torch.stack(features, dim=0).mean(dim=0).squeeze(0)  # [N_patches, C]

    C = feat.shape[-1]
    feat = feat.reshape(gh, gw, C)   # [gh, gw, C]
    feat = feat.permute(2, 0, 1)     # [C, gh, gw]

    # ---------- Instance Normalization (조명 불변성) ----------
    # ResNet과 동일한 처리: 채널별 공간 평균/분산 제거
    mean = feat.mean(dim=(1, 2), keepdim=True)
    std  = feat.std(dim=(1, 2), keepdim=True) + 1e-6
    feat = (feat - mean) / std
    # ---------------------------------------------------------

    # 각 위치의 C-dim 벡터를 단위벡터로 → cosine sim 계산용
    feat = F.normalize(feat, dim=0)

    return feat, (gh, gw)
