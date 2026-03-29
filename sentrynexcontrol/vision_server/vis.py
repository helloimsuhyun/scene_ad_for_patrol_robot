# vis.py
from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

from .warp_utils import warp_query_to_bank, crop_common_valid_region


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

    bar_h = 18
    bar_w = min(220, w - 20)
    x0 = 10
    y0 = max(10, h - bar_h - 10)

    grad = np.linspace(0, 255, bar_w, dtype=np.uint8)[None, :]
    grad = np.repeat(grad, bar_h, axis=0)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

    out[y0:y0 + bar_h, x0:x0 + bar_w] = grad_color
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(out, "low", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "high", (x0 + bar_w - 35, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def make_overlay(a_bgr: np.ndarray, b_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if a_bgr is None or b_bgr is None:
        raise ValueError("input image is None")

    h = min(a_bgr.shape[0], b_bgr.shape[0])
    w = min(a_bgr.shape[1], b_bgr.shape[1])

    a_rs = cv2.resize(a_bgr, (w, h))
    b_rs = cv2.resize(b_bgr, (w, h))
    return cv2.addWeighted(a_rs, alpha, b_rs, 1.0 - alpha, 0.0)


def to_model_view_bgr_from_bgr(img_bgr: np.ndarray, img_size: int = 560) -> np.ndarray:
    """
    make_transform의 공간계와 최대한 맞추기 위한 간단한 view 변환.
    Resize(img_size) -> CenterCrop(img_size)
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = transforms.Resize(img_size)(img_pil)
    img_pil = transforms.CenterCrop(img_size)(img_pil)

    out_rgb = np.array(img_pil)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr

def draw_patch_match_lines_aligned(
    query_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    query_patch_idx,
    ref_patch_idx,
    query_patch_vals,
    query_grid_hw,
    ref_grid_hw,
    max_lines: int = 20,
) -> np.ndarray:
    q = query_bgr.copy()
    r = ref_bgr.copy()

    h = max(q.shape[0], r.shape[0])
    wq = q.shape[1]
    wr = r.shape[1]

    if q.shape[0] != h:
        q = cv2.resize(q, (wq, h))
    if r.shape[0] != h:
        r = cv2.resize(r, (wr, h))

    canvas = np.concatenate([q, r], axis=1).copy()

    if hasattr(query_patch_idx, "detach"):
        query_patch_idx = query_patch_idx.detach().cpu().tolist()
    else:
        query_patch_idx = list(query_patch_idx)

    if hasattr(ref_patch_idx, "detach"):
        ref_patch_idx = ref_patch_idx.detach().cpu().tolist()
    else:
        ref_patch_idx = list(ref_patch_idx)

    if query_patch_vals is None:
        query_patch_vals = [1.0] * len(query_patch_idx)
    elif hasattr(query_patch_vals, "detach"):
        query_patch_vals = query_patch_vals.detach().cpu().tolist()
    else:
        query_patch_vals = list(query_patch_vals)

    n = min(len(query_patch_idx), len(ref_patch_idx), len(query_patch_vals), max_lines)
    if n == 0:
        return canvas

    order = np.argsort(-np.asarray(query_patch_vals[:n], dtype=np.float32))

    q_crop_shape = q.shape[:2]
    r_crop_shape = r.shape[:2]

    for oi in order:
        q_idx = int(query_patch_idx[oi])
        r_idx = int(ref_patch_idx[oi])

        qx1, qy1, qx2, qy2 = patch_idx_to_box_aligned(q_idx, query_grid_hw, q_crop_shape)
        rx1, ry1, rx2, ry2 = patch_idx_to_box_aligned(r_idx, ref_grid_hw, r_crop_shape)

        qcx, qcy = patch_idx_to_center_aligned(q_idx, query_grid_hw, q_crop_shape)
        rcx, rcy = patch_idx_to_center_aligned(r_idx, ref_grid_hw, r_crop_shape)

        rx1 += q.shape[1]
        rx2 += q.shape[1]
        rcx += q.shape[1]

        cv2.rectangle(canvas, (qx1, qy1), (qx2, qy2), (0, 255, 255), 2)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.line(canvas, (qcx, qcy), (rcx, rcy), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "query_crop", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "ref_crop", (q.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas

def save_aligned_debug_vis(
    query_path: str | Path,
    align_vis: Dict[str, Any],
    out_prefix: str | Path,
    img_size: int = 560,
    patch_size: int = 14,
):
    """
    저장물:
      - *_00_query_raw.jpg
      - *_01_query_aligned.jpg
      - *_02_query_ref_overlay.jpg
      - *_03_query_aligned_crop_heatmap.jpg
    """
    if not align_vis:
        return

    data = align_vis.get("data")
    if not isinstance(data, dict):
        return

    ref_path_raw = data.get("best_ref_img_path")
    H = data.get("H")
    crop_bbox = data.get("crop_bbox")
    top_patch_idx = data.get("top_patch_idx", [])
    top_patch_vals = data.get("top_patch_vals", [])
    grid_hw = data.get("grid_hw")
    crop_shape = data.get("crop_shape")

    if not ref_path_raw or H is None or crop_bbox is None or grid_hw is None or crop_shape is None:
        return

    query_path = Path(query_path)
    out_prefix = Path(out_prefix)

    query_raw = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    ref_img = cv2.imread(str(ref_path_raw), cv2.IMREAD_COLOR)

    if query_raw is None or ref_img is None:
        return

    H = np.asarray(H, dtype=np.float32)

    warped_q_bgr, warped_mask = warp_query_to_bank(
        query_raw,
        H,
        bank_hw=ref_img.shape[:2],
    )

    warped_q_crop, ref_crop, mask_crop, _ = crop_common_valid_region(
        warped_q_bgr,
        ref_img,
        warped_mask,
        margin=8,
        min_size=64,
    )

    if warped_q_crop is None or ref_crop is None:
        return

    overlay_img = make_overlay(warped_q_bgr, ref_img, alpha=0.5)

    if hasattr(top_patch_idx, "detach"):
        top_patch_idx = top_patch_idx.detach().cpu().tolist()
    else:
        top_patch_idx = list(top_patch_idx)

    if hasattr(top_patch_vals, "detach"):
        top_patch_vals = top_patch_vals.detach().cpu().tolist()
    else:
        top_patch_vals = list(top_patch_vals)

    # 중요:
    # aligned patch idx는 crop 기준이므로 crop 원본 위에 바로 그려야 함.
    vis_img = draw_top_p_heatmap_aligned(
        img_bgr=warped_q_crop,
        top_patch_idx=top_patch_idx,
        top_patch_vals=top_patch_vals,
        grid_hw=grid_hw,
        alpha=0.45,
        blur_ksize=0,
        normalize_each=False,
        abs_min=0.2,
        abs_max=0.80,
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_prefix.parent / f"{out_prefix.name}_00_query_raw.jpg"), query_raw)
    cv2.imwrite(str(out_prefix.parent / f"{out_prefix.name}_01_query_aligned.jpg"), warped_q_bgr)
    cv2.imwrite(str(out_prefix.parent / f"{out_prefix.name}_02_query_ref_overlay.jpg"), overlay_img)
    cv2.imwrite(str(out_prefix.parent / f"{out_prefix.name}_03_query_aligned_crop_heatmap.jpg"), vis_img)


def patch_idx_to_center(patch_idx: int, grid_w: int, patch_size: int):
    row = patch_idx // grid_w
    col = patch_idx % grid_w
    cx = int(col * patch_size + patch_size // 2)
    cy = int(row * patch_size + patch_size // 2)
    return cx, cy


def draw_patch_match_lines(
    query_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    query_patch_idx,
    ref_patch_idx,
    query_patch_vals=None,
    img_size: int = 560,
    patch_size: int = 14,
    max_lines: int = 20,
) -> np.ndarray:
    q = cv2.resize(query_bgr, (img_size, img_size))
    r = cv2.resize(ref_bgr, (img_size, img_size))

    canvas = np.concatenate([q, r], axis=1).copy()
    grid_w = img_size // patch_size

    if hasattr(query_patch_idx, "detach"):
        query_patch_idx = query_patch_idx.detach().cpu().tolist()
    else:
        query_patch_idx = list(query_patch_idx)

    if hasattr(ref_patch_idx, "detach"):
        ref_patch_idx = ref_patch_idx.detach().cpu().tolist()
    else:
        ref_patch_idx = list(ref_patch_idx)

    if query_patch_vals is None:
        query_patch_vals = [1.0] * len(query_patch_idx)
    elif hasattr(query_patch_vals, "detach"):
        query_patch_vals = query_patch_vals.detach().cpu().tolist()
    else:
        query_patch_vals = list(query_patch_vals)

    n = min(len(query_patch_idx), len(ref_patch_idx), len(query_patch_vals), max_lines)
    if n == 0:
        return canvas

    order = np.argsort(-np.asarray(query_patch_vals[:n], dtype=np.float32))

    for oi in order:
        q_idx = int(query_patch_idx[oi])
        r_idx = int(ref_patch_idx[oi])

        qx1, qy1, qx2, qy2 = patch_idx_to_box(q_idx, grid_w, patch_size)
        rx1, ry1, rx2, ry2 = patch_idx_to_box(r_idx, grid_w, patch_size)

        qcx, qcy = patch_idx_to_center(q_idx, grid_w, patch_size)
        rcx, rcy = patch_idx_to_center(r_idx, grid_w, patch_size)

        # ref는 오른쪽 panel이라 x offset 추가
        rx1 += img_size
        rx2 += img_size
        rcx += img_size

        cv2.rectangle(canvas, (qx1, qy1), (qx2, qy2), (0, 255, 255), 2)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.line(canvas, (qcx, qcy), (rcx, rcy), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "query", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "ref", (img_size + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def save_patch_match_vis(
    query_path,
    ref_path,
    patch_vis: Dict[str, Any],
    out_path,
):
    if not patch_vis:
        return

    top_patch_idx = patch_vis.get("top_patch_idx", [])
    top_patch_vals = patch_vis.get("top_patch_vals", [])
    top_patch_match_idx = patch_vis.get("top_patch_match_idx", [])
    img_size = int(patch_vis.get("img_size", 560))
    repr_mode = patch_vis.get("repr_mode", "global_patch")

    if len(top_patch_idx) == 0 or len(top_patch_match_idx) == 0:
        return

    query_bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if query_bgr is None or ref_bgr is None:
        return

    if repr_mode == "global_patch_with_aligned":
        H = patch_vis.get("H", None)
        crop_bbox = patch_vis.get("crop_bbox", None)
        grid_hw = patch_vis.get("grid_hw", None)
        crop_shape = patch_vis.get("crop_shape", None)

        if H is None or crop_bbox is None or grid_hw is None or crop_shape is None:
            return

        H = np.asarray(H, dtype=np.float32)

        warped_q_bgr, warped_mask = warp_query_to_bank(
            query_bgr,
            H,
            bank_hw=ref_bgr.shape[:2],
        )

        warped_q_crop, ref_crop, mask_crop, _ = crop_common_valid_region(
            warped_q_bgr,
            ref_bgr,
            warped_mask,
            margin=8,
            min_size=64,
        )

        if warped_q_crop is None or ref_crop is None:
            return

        # 중요:
        # aligned patch idx는 crop 좌표계 기준이므로
        # 원본 전체 이미지가 아니라 crop 이미지 위에 바로 그려야 좌표가 맞는다.
        vis_img = draw_patch_match_lines_aligned(
            query_bgr=warped_q_crop,
            ref_bgr=ref_crop,
            query_patch_idx=top_patch_idx,
            ref_patch_idx=top_patch_match_idx,
            query_patch_vals=top_patch_vals,
            query_grid_hw=grid_hw,
            ref_grid_hw=grid_hw,
            max_lines=len(top_patch_idx),
        )
    else:
        patch_size = int(patch_vis.get("patch_size", 14))
        query_mv = to_model_view_bgr_from_bgr(query_bgr, img_size=img_size)
        ref_mv = to_model_view_bgr_from_bgr(ref_bgr, img_size=img_size)

        vis_img = draw_patch_match_lines(
            query_bgr=query_mv,
            ref_bgr=ref_mv,
            query_patch_idx=top_patch_idx,
            ref_patch_idx=top_patch_match_idx,
            query_patch_vals=top_patch_vals,
            img_size=img_size,
            patch_size=patch_size,
            max_lines=len(top_patch_idx),
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis_img)

def patch_idx_to_box_aligned(
    patch_idx: int,
    grid_hw,
    crop_shape,
):
    Hp, Wp = grid_hw
    crop_h, crop_w = crop_shape

    row = patch_idx // Wp
    col = patch_idx % Wp

    cell_w = crop_w / float(Wp)
    cell_h = crop_h / float(Hp)

    x1 = int(round(col * cell_w))
    y1 = int(round(row * cell_h))
    x2 = int(round((col + 1) * cell_w))
    y2 = int(round((row + 1) * cell_h))
    return x1, y1, x2, y2


def patch_idx_to_center_aligned(
    patch_idx: int,
    grid_hw,
    crop_shape,
):
    x1, y1, x2, y2 = patch_idx_to_box_aligned(patch_idx, grid_hw, crop_shape)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


def patch_idx_to_box_aligned(
    patch_idx: int,
    grid_hw,
    crop_shape,
):
    Hp, Wp = grid_hw
    crop_h, crop_w = crop_shape

    row = patch_idx // Wp
    col = patch_idx % Wp

    cell_w = crop_w / float(Wp)
    cell_h = crop_h / float(Hp)

    x1 = int(round(col * cell_w))
    y1 = int(round(row * cell_h))
    x2 = int(round((col + 1) * cell_w))
    y2 = int(round((row + 1) * cell_h))
    return x1, y1, x2, y2


def patch_idx_to_center_aligned(
    patch_idx: int,
    grid_hw,
    crop_shape,
):
    x1, y1, x2, y2 = patch_idx_to_box_aligned(patch_idx, grid_hw, crop_shape)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


def draw_top_p_heatmap_aligned(
    img_bgr: np.ndarray,
    top_patch_idx,
    top_patch_vals,
    grid_hw,
    alpha: float = 0.45,
    blur_ksize: int = 0,
    normalize_each: bool = True,
    abs_min: float | None = None,
    abs_max: float | None = None,
) -> np.ndarray:
    """
    aligned CNN grid 기준 sparse heatmap overlay
    grid_hw = (Hp, Wp)
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    if len(top_patch_idx) == 0 or len(top_patch_vals) == 0:
        return img_bgr.copy()

    h, w = img_bgr.shape[:2]
    gh, gw = grid_hw
    num_patches = gh * gw

    heat = np.zeros((num_patches,), dtype=np.float32)

    if hasattr(top_patch_idx, "detach"):
        top_patch_idx = top_patch_idx.detach().cpu().tolist()
    else:
        top_patch_idx = list(top_patch_idx)

    if hasattr(top_patch_vals, "detach"):
        top_patch_vals = top_patch_vals.detach().cpu().tolist()
    else:
        top_patch_vals = list(top_patch_vals)

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

    bar_h = 18
    bar_w = min(220, w - 20)
    x0 = 10
    y0 = max(10, h - bar_h - 10)

    grad = np.linspace(0, 255, bar_w, dtype=np.uint8)[None, :]
    grad = np.repeat(grad, bar_h, axis=0)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

    out[y0:y0 + bar_h, x0:x0 + bar_w] = grad_color
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(out, "low", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "high", (x0 + bar_w - 35, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out
