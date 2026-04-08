#warp_util.py

import cv2
import numpy as np

# query -> bank
def warp_query_to_bank(query_bgr, H, bank_hw):
    h, w = bank_hw

    # query 원본 영역을 마스킹
    mask = np.ones(query_bgr.shape[:2], dtype=np.uint8) * 255

    # warp -- 
    warped = cv2.warpPerspective(query_bgr, H, (w, h))
    warped_mask = cv2.warpPerspective(mask, H, (w, h))

    #warp 과정에서 보간처리된 영역을 다시 binary로 변경
    warped_mask = np.where(warped_mask > 127, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    warped_mask = cv2.erode(warped_mask, kernel, iterations=1)

    return warped, warped_mask

# bank -> query
def warp_bank_to_query(bank_bgr, H_inv, query_hw):
    h, w = query_hw

    mask = np.ones(bank_bgr.shape[:2], dtype=np.uint8) * 255

    warped = cv2.warpPerspective(bank_bgr, H_inv, (w, h))
    warped_mask = cv2.warpPerspective(mask, H_inv, (w, h))

    warped_mask = np.where(warped_mask > 127, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    warped_mask = cv2.erode(warped_mask, kernel, iterations=1)

    return warped, warped_mask


# warped_mask와 dino patch gird 사이즈를 받아서 각 patch의 사용여부를 thr기준으로 계산
def make_patch_valid_mask(mask_hw, grid_h, grid_w, thr=0.8):
    mask = cv2.resize(mask_hw, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    mask = mask.astype(np.float32) / 255.0
    return mask >= thr


def crop_common_valid_region(warped_q_bgr, ref_bgr, warped_mask, margin=16, min_size=64):
    """
    warped_mask 기준으로 공통 valid bbox ROI 를 잡아서
    warped query / ref / mask를 동일하게 crop.

    margin: bbox 안쪽으로 추가로 깎을 픽셀 수
    min_size: 너무 작은 crop 방지
    """
    ys, xs = np.where(warped_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return None, None, None, None

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    # 안쪽으로 조금 더 깎아서 border 영향 감소
    y0 = min(max(y0 + margin, 0), warped_q_bgr.shape[0] - 1)
    x0 = min(max(x0 + margin, 0), warped_q_bgr.shape[1] - 1)
    y1 = max(min(y1 - margin, warped_q_bgr.shape[0]), y0 + 1)
    x1 = max(min(x1 - margin, warped_q_bgr.shape[1]), x0 + 1)

    crop_h = y1 - y0
    crop_w = x1 - x0
    if crop_h < min_size or crop_w < min_size:
        return None, None, None, None

    warped_q_crop = warped_q_bgr[y0:y1, x0:x1]
    ref_crop = ref_bgr[y0:y1, x0:x1]
    mask_crop = warped_mask[y0:y1, x0:x1]

    bbox = (y0, y1, x0, x1)
    return warped_q_crop, ref_crop, mask_crop, bbox

def crop_common_safe_region(
    warped_q_bgr,
    ref_bgr,
    warped_mask,
    erode_kernel=9,
    erode_iter=2,
    margin=8,
    min_size=64,
):
    """
    warped_mask에서 border를 충분히 깎은 safe_mask를 만든 뒤,
    그 안전영역 bbox만 crop해서 반환.
    """
    safe_mask = warped_mask.copy()

    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    safe_mask = cv2.erode(safe_mask, kernel, iterations=erode_iter)

    ys, xs = np.where(safe_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return None, None, None, None

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    y0 = min(max(y0 + margin, 0), warped_q_bgr.shape[0] - 1)
    x0 = min(max(x0 + margin, 0), warped_q_bgr.shape[1] - 1)
    y1 = max(min(y1 - margin, warped_q_bgr.shape[0]), y0 + 1)
    x1 = max(min(x1 - margin, warped_q_bgr.shape[1]), x0 + 1)

    crop_h = y1 - y0
    crop_w = x1 - x0
    if crop_h < min_size or crop_w < min_size:
        return None, None, None, None

    warped_q_crop = warped_q_bgr[y0:y1, x0:x1]
    ref_crop = ref_bgr[y0:y1, x0:x1]
    mask_crop = safe_mask[y0:y1, x0:x1]

    warped_q_crop[mask_crop == 0] = 0
    ref_crop[mask_crop == 0] = 0

    bbox = (y0, y1, x0, x1)
    return warped_q_crop, ref_crop, mask_crop, bbox