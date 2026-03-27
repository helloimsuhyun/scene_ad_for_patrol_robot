# vis.py
from __future__ import annotations

from typing import List
import numpy as np
import cv2
import numpy as np

def patch_idx_to_box(patch_idx: int, grid_w: int, patch_size: int):
    row = patch_idx // grid_w
    col = patch_idx % grid_w
    x1 = col * patch_size
    y1 = row * patch_size
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    return x1, y1, x2, y2


def draw_top_p_heatmap(
    img_bgr: np.ndarray,
    top_patch_idx: List[int],
    top_patch_vals: List[float],
    img_size: int = 560,
    patch_size: int = 14,
    alpha: float = 0.45,
    blur_ksize: int = 0,
    normalize_each: bool = True,
    abs_min: float | None = None,
    abs_max: float | None = None,
) -> np.ndarray:
    """
    top-p patch index/value를 sparse heatmap으로 만든 뒤
    원본 이미지 위에 overlay 한다.

    Args:
        img_bgr: 원본 이미지 (H,W,3)
        top_patch_idx: 상위 patch 인덱스 리스트
        top_patch_vals: 각 patch score/dist 리스트
        img_size: 임베딩 추출 시 사용한 입력 크기
        patch_size: ViT patch size
        alpha: overlay 강도
        blur_ksize: heatmap smoothing용 blur kernel size (0이면 안 함)
        normalize_each: top_patch_vals 범위를 [0,1]로 정규화할지 여부
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    if len(top_patch_idx) == 0 or len(top_patch_vals) == 0:
        return img_bgr.copy()

    h, w = img_bgr.shape[:2]

    gh = img_size // patch_size
    gw = img_size // patch_size
    num_patches = gh * gw

    heat = np.zeros((num_patches,), dtype=np.float32)

    vals = np.asarray(top_patch_vals, dtype=np.float32)

    if normalize_each:
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax > vmin:
            vals = (vals - vmin) / (vmax - vmin)
        else:
            vals = np.ones_like(vals, dtype=np.float32)
    else:
        if abs_min is None or abs_max is None:
            raise ValueError("normalize_each=False 일 때 abs_min, abs_max를 지정해야 합니다.")
        if abs_max <= abs_min:
            raise ValueError("abs_max must be greater than abs_min")

        vals = (vals - abs_min) / (abs_max - abs_min)
        vals = np.clip(vals, 0.0, 1.0)

    for idx, val in zip(top_patch_idx, vals):
        idx = int(idx)
        if 0 <= idx < num_patches:
            heat[idx] = max(float(heat[idx]), float(val))

    heat = heat.reshape(gh, gw)

    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        heat = cv2.GaussianBlur(heat, (blur_ksize, blur_ksize), 0)

    heat_up = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    heat_up = np.clip(heat_up, 0.0, 1.0)

    heat_u8 = (heat_up * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    out = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0)

    # colorbar 비슷한 간단한 legend
    bar_h = 18
    bar_w = min(220, w - 20)
    x0 = 10
    y0 = max(10, h - bar_h - 10)

    grad = np.linspace(0, 255, bar_w, dtype=np.uint8)[None, :]
    grad = np.repeat(grad, bar_h, axis=0)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

    out[y0:y0 + bar_h, x0:x0 + bar_w] = grad_color
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(out, "low", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, "high", (x0 + bar_w - 35, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    return out