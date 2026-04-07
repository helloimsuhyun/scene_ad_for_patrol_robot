# distnace_util.py

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .warp_utils import (
    warp_query_to_bank,
    make_patch_valid_mask,
    crop_common_safe_region,
)

def compute_dist_map_local_search(q_feat, r_feat, valid_mask=None, radius=1):
    assert q_feat.ndim == 3 and r_feat.ndim == 3
    C, H, W = q_feat.shape

    q = q_feat.unsqueeze(0)
    r = r_feat.unsqueeze(0)

    k = 2 * radius + 1
    r_unfold = F.unfold(r, kernel_size=k, padding=radius)
    r_unfold = r_unfold.squeeze(0).view(C, k * k, H * W)
    q_flat = q.squeeze(0).view(C, 1, H * W)

    sims = (q_flat * r_unfold).sum(dim=0)
    best_sim, _ = sims.max(dim=0)

    dist_map = 1.0 - best_sim.view(H, W)
    dist = dist_map.detach().cpu().numpy()

    if valid_mask is not None:
        dist = np.where(valid_mask, dist, 0.0)

    return dist


def compute_compound_score(
    dist_map: np.ndarray,
    valid_mask: np.ndarray,
    top_p: float = 0.05,
    alpha: float = 0.6,
    min_cut: float = 0.20,
    singleton_weight: float = 0.25,
    component_min_area: int = 2,
) -> dict:
    Hp, Wp = dist_map.shape
    empty_bin = np.zeros((Hp, Wp), dtype=np.uint8)
    empty_best = np.zeros((Hp, Wp), dtype=bool)

    masked = dist_map * valid_mask.astype(np.float32)
    flat_valid = masked[valid_mask > 0]

    if flat_valid.size == 0:
        return {
            "score": 0.0, "area": 0, "peak": 0.0, "mean": 0.0, "cut": min_cut,
            "n_top": 0, "valid_count": 0, "bin_map": empty_bin,
            "best_comp_mask": empty_best, "all_comp_scores": []
        }

    n_top = max(1, int(np.ceil(flat_valid.size * top_p)))
    top_threshold = np.sort(flat_valid)[-n_top]
    peak = float(flat_valid.max())
    cut = float(max(alpha * peak, min_cut, top_threshold))

    bin_map = (masked >= cut).astype(np.uint8)
    if bin_map.sum() == 0:
        return {
            "score": 0.0, "area": 0, "peak": peak, "mean": 0.0, "cut": cut,
            "n_top": n_top, "valid_count": int(flat_valid.size), "bin_map": bin_map,
            "best_comp_mask": empty_best, "all_comp_scores": []
        }

    struct = np.ones((3, 3), dtype=np.int32)
    labeled, n_labels = ndimage.label(bin_map, structure=struct)

    best_score, best_area, best_peak, best_mean, best_lbl = 0.0, 0, 0.0, 0.0, -1
    valid_count = int(flat_valid.size)
    all_comp_scores = []

    for lbl in range(1, n_labels + 1):
        comp_mask = labeled == lbl
        vals = dist_map[comp_mask]
        excess = np.clip(vals - cut, a_min=0.0, a_max=None)
        area = int(comp_mask.sum())

        if area >= component_min_area:
            raw_score = float(excess.sum())
        else:
            raw_score = float(singleton_weight * excess.max()) if excess.size > 0 else 0.0

        norm_score = raw_score / np.sqrt(max(valid_count, 1))

        all_comp_scores.append({
            "score": norm_score,
            "mask": comp_mask,
            "area": area,
            "peak": float(vals.max()),
            "mean": float(vals.mean()),
        })

        if norm_score > best_score:
            best_score = norm_score
            best_area = area
            best_peak = float(vals.max())
            best_mean = float(vals.mean())
            best_lbl = lbl

    best_comp_mask = (labeled == best_lbl) if best_lbl > 0 else empty_best

    return {
        "score": best_score,
        "area": best_area,
        "peak": best_peak,
        "mean": best_mean,
        "cut": cut,
        "n_top": n_top,
        "valid_count": valid_count,
        "bin_map": bin_map,
        "best_comp_mask": best_comp_mask,
        "all_comp_scores": all_comp_scores,
    }

def score_one_pair(
    q_bgr: np.ndarray,
    r_bgr: np.ndarray,
    sg,
    backbone,
    device: str,
    radius: int = 1,
    top_p: float = 0.05,
    alpha: float = 0.6,
    min_cut: float = 0.20,
    singleton_weight: float = 0.25,
    component_min_area: int = 2,
):
    match_res = sg.match_and_estimate(q_bgr, r_bgr)
    if not match_res.get("ok", False):
        return None, {"reason": "align_fail", "detail": match_res.get("reason", "")}

    H = match_res["H"]
    if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3):
        return None, {"reason": "invalid_H"}

    warped_q, warped_mask = warp_query_to_bank(q_bgr, H.astype(np.float64), r_bgr.shape[:2])
    q_crop, r_crop, mask_crop, bbox = crop_common_safe_region(warped_q, r_bgr, warped_mask)
    if q_crop is None:
        return None, {"reason": "crop_fail"}

    q_feat, _ = backbone.extract_grid(q_crop, device)
    r_feat, _ = backbone.extract_grid(r_crop, device)

    grid_h, grid_w = q_feat.shape[1], q_feat.shape[2]
    valid_mask = make_patch_valid_mask(mask_crop, grid_h, grid_w)

    if int(valid_mask.sum()) < 10:
        return None, {"reason": "too_few_valid_patches", "valid_count": int(valid_mask.sum())}

    dist_map = compute_dist_map_local_search(q_feat, r_feat, valid_mask=valid_mask, radius=radius)

    result = compute_compound_score(
        dist_map, valid_mask,
        top_p=top_p, alpha=alpha, min_cut=min_cut,
        singleton_weight=singleton_weight, component_min_area=component_min_area,
    )

    debug = {
        "reason": "ok",
        "score": result["score"],
        "area": result["area"],
        "peak": result["peak"],
        "mean": result["mean"],
        "cut": result["cut"],
        "inliers": int(match_res.get("inliers", 0)),
        "inlier_ratio": float(match_res.get("inlier_ratio", 0.0)),
        "bbox": [int(v) for v in bbox] if bbox is not None else None,
        "q_crop": q_crop,
        "r_crop": r_crop,
        "dist_map": dist_map,
        "valid_mask": valid_mask,
        "bin_map": result["bin_map"],
        "best_comp_mask": result["best_comp_mask"],
        "all_comp_scores": result["all_comp_scores"],
    }
    return result["score"], debug




def patch_mask_to_bbox(comp_mask: np.ndarray):
    ys, xs = np.where(comp_mask)
    if len(ys) == 0:
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)


def expand_patch_bbox(bbox, Hp, Wp, patch_margin=1):
    y0, x0, y1, x1 = bbox
    return (
        max(0, y0 - patch_margin),
        max(0, x0 - patch_margin),
        min(Hp, y1 + patch_margin),
        min(Wp, x1 + patch_margin),
    )


def patch_bbox_to_image_bbox(
    patch_bbox,
    img_h: int,
    img_w: int,
    grid_h: int,
    grid_w: int,
    crop_margin_ratio: float = 0.20,
    min_crop_size: int = 96,
):
    py0, px0, py1, px1 = patch_bbox
    sy = img_h / float(grid_h)
    sx = img_w / float(grid_w)

    y0 = int(np.floor(py0 * sy))
    x0 = int(np.floor(px0 * sx))
    y1 = int(np.ceil(py1 * sy))
    x1 = int(np.ceil(px1 * sx))

    h, w = y1 - y0, x1 - x0
    my, mx = int(round(h * crop_margin_ratio)), int(round(w * crop_margin_ratio))

    y0 = max(0, y0 - my)
    x0 = max(0, x0 - mx)
    y1 = min(img_h, y1 + my)
    x1 = min(img_w, x1 + mx)

    h, w = y1 - y0, x1 - x0
    if h < min_crop_size:
        pad = (min_crop_size - h) // 2 + 1
        y0 = max(0, y0 - pad)
        y1 = min(img_h, y1 + pad)
    if w < min_crop_size:
        pad = (min_crop_size - w) // 2 + 1
        x0 = max(0, x0 - pad)
        x1 = min(img_w, x1 + pad)

    return (y0, x0, y1, x1)


def build_flagged_component_regions(
    flagged_comps: list,
    q_crop: np.ndarray,
    r_crop: np.ndarray,
    valid_mask: np.ndarray,
    patch_margin: int = 1,
    crop_margin_ratio: float = 0.20,
    min_patch_area: int = 2,
    min_crop_size: int = 96,
):
    if len(flagged_comps) == 0:
        return []

    img_h, img_w = q_crop.shape[:2]
    grid_h, grid_w = valid_mask.shape[:2]
    regions = []

    for comp in flagged_comps:
        comp_mask = comp.get("mask", None)
        if comp_mask is None:
            continue

        area = int(comp.get("area", 0))
        if area < min_patch_area:
            continue

        pbbox = patch_mask_to_bbox(comp_mask)
        if pbbox is None:
            continue

        pbbox = expand_patch_bbox(pbbox, grid_h, grid_w, patch_margin=patch_margin)
        ibbox = patch_bbox_to_image_bbox(
            pbbox, img_h=img_h, img_w=img_w, grid_h=grid_h, grid_w=grid_w,
            crop_margin_ratio=crop_margin_ratio, min_crop_size=min_crop_size,
        )

        y0, x0, y1, x1 = ibbox
        if (y1 - y0) <= 0 or (x1 - x0) <= 0:
            continue

        q_region = q_crop[y0:y1, x0:x1].copy()
        r_region = r_crop[y0:y1, x0:x1].copy()

        valid_rs = cv2.resize(valid_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_region = valid_rs[y0:y1, x0:x1].copy()

        regions.append({
            "score": float(comp.get("score", 0.0)),
            "area": area,
            "peak": float(comp.get("peak", 0.0)),
            "mean": float(comp.get("mean", 0.0)),
            "patch_bbox": tuple(map(int, pbbox)),
            "img_bbox": tuple(map(int, ibbox)),
            "q_region": q_region,
            "r_region": r_region,
            "mask_region": mask_region,
        })

    return sorted(regions, key=lambda x: x["score"], reverse=True)



@torch.no_grad()
def compute_local_search_dist_map_from_feats(q_feat: torch.Tensor, r_feat: torch.Tensor, radius: int = 1):
    C, H, W = q_feat.shape
    dist_map = torch.zeros((H, W), device=q_feat.device, dtype=torch.float32)

    for y in range(H):
        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)
        for x in range(W):
            x0 = max(0, x - radius)
            x1 = min(W, x + radius + 1)
            qv = q_feat[:, y, x]
            rv = r_feat[:, y0:y1, x0:x1]
            sims = (rv * qv[:, None, None]).sum(dim=0)
            best_sim = sims.max()
            dist_map[y, x] = 1.0 - best_sim

    return dist_map


def topk_mean_score(dist_map: np.ndarray, top_p: float = 0.10):
    flat = dist_map.reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(np.ceil(flat.size * top_p)))
    vals = np.sort(flat)[-k:]
    return float(vals.mean())


def make_top_p_mask(dist_map: np.ndarray, top_p: float = 0.10):
    flat = dist_map.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(dist_map, dtype=bool), 0.0, 0
    k = max(1, int(np.ceil(flat.size * top_p)))
    thr_top = float(np.sort(flat)[-k])
    return dist_map >= thr_top, thr_top, k


@torch.no_grad()
def verify_bbox_with_local_search(q_region: np.ndarray, r_region: np.ndarray, backbone, device, radius: int = 1, top_p: float = 0.10):
    q_feat, (Hf, Wf) = backbone.extract_grid(q_region, device)
    r_feat, _ = backbone.extract_grid(r_region, device)

    dist_map_t = compute_local_search_dist_map_from_feats(q_feat=q_feat, r_feat=r_feat, radius=radius)
    dist_map = dist_map_t.detach().cpu().numpy()

    score = topk_mean_score(dist_map, top_p=top_p)
    top_p_mask, top_p_thr, top_k = make_top_p_mask(dist_map, top_p=top_p)

    return {
        "score": score,
        "dist_map": dist_map,
        "feat_hw": (Hf, Wf),
        "top_p_mask": top_p_mask,
        "top_p_thr": top_p_thr,
        "top_k": top_k,
        "top_p": top_p,
    }




