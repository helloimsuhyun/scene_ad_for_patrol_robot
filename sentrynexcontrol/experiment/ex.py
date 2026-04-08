#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from config import load_cfg
from dino_emb import load_dino_model, make_dino_transform
from matcher import SuperGlueMatcher, SuperGlueMatchConfig
from warp_utils import warp_query_to_bank, make_patch_valid_mask, crop_common_safe_region


ROOT = "/home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/experiment/recv"
OUT_ROOT = "/home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/experiment/recv/out"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def BGR_to_RGB(img_bgr_uint8):
    img_rgb = img_bgr_uint8[:, :, ::-1]
    return Image.fromarray(img_rgb).convert("RGB")


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def plot_compound_scores(out_dir: Path):
    path = out_dir / "infer_all_results.json"
    if not path.exists():
        print("[PLOT] infer_all_results.json 없음")
        return

    data = json.loads(path.read_text())
    normal_scores, abnormal_scores = [], []

    for item in data:
        is_abnormal = Path(item["query"]).name.startswith("abnormal_")
        s = float(item.get("score", 0.0))
        (abnormal_scores if is_abnormal else normal_scores).append(s)

    plt.figure(figsize=(10, 4))
    x_n = np.random.normal(0, 0.04, size=len(normal_scores))
    x_a = np.random.normal(1, 0.04, size=len(abnormal_scores))
    plt.scatter(x_n, normal_scores, alpha=0.6, s=15, label="normal")
    plt.scatter(x_a, abnormal_scores, alpha=0.6, s=15, label="abnormal")
    plt.xticks([0, 1], ["normal", "abnormal"])

    thr_path = out_dir / "threshold.json"
    if thr_path.exists():
        meta = json.loads(thr_path.read_text())
        thr = meta.get("threshold", None)
        if thr is not None:
            plt.axhline(thr, linestyle="--")

    all_scores = normal_scores + abnormal_scores
    if all_scores:
        low = np.percentile(all_scores, 1)
        high = np.percentile(all_scores, 99)
        plt.ylim(low, high)

    plt.legend()
    plt.savefig(out_dir / "compound_score_scatter.png")
    plt.close()
    print(f"[PLOT] saved: {out_dir / 'compound_score_scatter.png'}")


def plot_bbox_scores(out_dir: Path):
    path = out_dir / "infer_all_results.json"
    if not path.exists():
        print("[PLOT] infer_all_results.json 없음")
        return

    data = json.loads(path.read_text())
    normal_scores, abnormal_scores = [], []

    for item in data:
        is_abnormal = Path(item["query"]).name.startswith("abnormal_")
        for r in item.get("flagged_regions", []):
            s = float(r.get("final_bbox_score", 0.0))
            (abnormal_scores if is_abnormal else normal_scores).append(s)

    plt.figure(figsize=(10, 4))
    x_n = np.random.normal(0, 0.04, size=len(normal_scores))
    x_a = np.random.normal(1, 0.04, size=len(abnormal_scores))
    plt.scatter(x_n, normal_scores, alpha=0.6, s=15, label="normal")
    plt.scatter(x_a, abnormal_scores, alpha=0.6, s=15, label="abnormal")
    plt.xticks([0, 1], ["normal", "abnormal"])

    thr_path = out_dir / "threshold.json"
    if thr_path.exists():
        meta = json.loads(thr_path.read_text())
        thr = meta.get("verifier_threshold", None)
        if thr is not None:
            plt.axhline(thr, linestyle="--")

    all_scores = normal_scores + abnormal_scores
    if all_scores:
        low = np.percentile(all_scores, 1)
        high = np.percentile(all_scores, 99)
        plt.ylim(low, high)

    plt.legend()
    plt.savefig(out_dir / "bbox_score_scatter.png")
    plt.close()
    print(f"[PLOT] saved: {out_dir / 'bbox_score_scatter.png'}")


def save_cc_heatmap(case_dir, q_crop, dist_map, valid_mask):
    h, w = q_crop.shape[:2]
    d = np.clip(dist_map, 0.0, 1.0)
    heat = (d * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    heat_rs = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_rs = cv2.resize(
        valid_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    overlay = q_crop.copy()
    blended = cv2.addWeighted(q_crop, 0.5, heat_rs, 0.5, 0)
    overlay[valid_rs] = blended[valid_rs]
    overlay[~valid_rs] = (128, 128, 128)

    cv2.imwrite(str(case_dir / "cc_heat.png"), heat_rs)
    cv2.imwrite(str(case_dir / "cc_overlay.png"), overlay)


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


def calibrate_threshold(
    scores: np.ndarray,
    method: str = "robust",
    k: float = 3.0,
    percentile: float = 99.0,
    trim_outliers: bool = True,
    trim_k: float = 3.5,
) -> dict:
    if len(scores) == 0:
        raise ValueError("calibrate_threshold: scores가 비어있습니다.")

    scores = np.array(scores, dtype=np.float32)
    scores_raw = scores.copy()

    raw_median = float(np.median(scores_raw))
    raw_mad = float(np.median(np.abs(scores_raw - raw_median)))
    raw_sigma = float(raw_mad * 1.4826)

    removed_scores = []
    upper_bound = None

    if trim_outliers and len(scores_raw) >= 10 and raw_mad > 1e-12:
        upper_bound = float(raw_median + trim_k * raw_sigma)
        keep_mask = scores_raw <= upper_bound
        removed_scores = scores_raw[~keep_mask].tolist()
        if int(keep_mask.sum()) >= max(5, int(0.8 * len(scores_raw))):
            scores = scores_raw[keep_mask]
        else:
            scores = scores_raw.copy()
    else:
        scores = scores_raw.copy()

    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    mean = float(scores.mean())
    std = float(scores.std())

    if method == "robust":
        thr = median + k * mad * 1.4826
    elif method == "gaussian":
        thr = mean + k * std
    elif method == "percentile":
        thr = float(np.percentile(scores, percentile))
    else:
        raise ValueError(f"알 수 없는 method: {method}")

    thr = float(thr)
    top5 = sorted(scores.tolist())[-5:]
    raw_top5 = sorted(scores_raw.tolist())[-5:]

    print(f"\n[CALIB] raw_n={len(scores_raw)}, used_n={len(scores)}, method={method}, k={k}")
    print(f"[CALIB] raw_median={raw_median:.4f}, raw_mad={raw_mad:.4f}, raw_sigma={raw_sigma:.4f}")
    if upper_bound is not None:
        print(f"[CALIB] trim_upper_bound={upper_bound:.4f}, removed={len(removed_scores)}")
        print(f"[CALIB] removed_top={ [round(v,4) for v in sorted(removed_scores)[-5:]] }")

    print(f"[CALIB] median={median:.4f}, mad={mad:.4f}")
    print(f"[CALIB] mean={mean:.4f}, std={std:.4f}")
    print(f"[CALIB] min={scores.min():.4f}, max={scores.max():.4f}")
    print(f"[CALIB] raw_top5={[round(v,4) for v in raw_top5]}")
    print(f"[CALIB] top5={[round(v,4) for v in top5]}")
    print(f"[CALIB] threshold={thr:.4f}")

    return {
        "threshold": thr,
        "method": method,
        "k": float(k),
        "n": int(len(scores)),
        "raw_n": int(len(scores_raw)),
        "median": median,
        "mad": mad,
        "mean": mean,
        "std": std,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "top5": [round(v, 5) for v in top5],
        "percentile": float(percentile),
        "trim_used": bool(trim_outliers),
        "trim_k": float(trim_k),
        "trim_upper_bound": None if upper_bound is None else round(upper_bound, 6),
        "trim_removed_n": int(len(removed_scores)),
        "trim_removed_scores": [round(float(v), 6) for v in sorted(removed_scores)],
        "raw_top5": [round(v, 5) for v in raw_top5],
    }


@torch.no_grad()
def build_dino_bank(bank_dir: Path, dino_model, dino_tfm, device: str, cache_name: str = "dino_bank.npz") -> dict:
    cache_path = bank_dir / cache_name
    bank_paths = sorted([p for p in bank_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    path_strs = [str(p) for p in bank_paths]

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = data["paths"].tolist()
        if cached_paths == path_strs:
            print(f"[DINO Bank] 캐시 로드: {cache_path} ({len(cached_paths)}장)")
            return {"embs": data["embs"], "paths": path_strs}

    print(f"[DINO Bank] 임베딩 계산 중... ({len(bank_paths)}장)")
    embs, valid_paths = [], []

    for p in bank_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_pil = BGR_to_RGB(img_bgr)
        x = dino_tfm(img_pil).unsqueeze(0).to(device)

        feats = dino_model.get_intermediate_layers(x, n=1, return_class_token=False)
        feat = feats[0].squeeze(0).mean(dim=0)
        feat = F.normalize(feat, dim=0)
        embs.append(feat.cpu().numpy().astype(np.float32))
        valid_paths.append(str(p))

    embs_np = np.stack(embs, axis=0)
    np.savez_compressed(cache_path, embs=embs_np, paths=np.array(valid_paths, dtype=object))
    print(f"[DINO Bank] 캐시 저장 완료: {cache_path}")
    return {"embs": embs_np, "paths": valid_paths}


@torch.no_grad()
def dino_preselect(q_bgr: np.ndarray, bank_dino: dict, dino_model, dino_tfm, device: str, top_m: int = 5) -> list:
    if bank_dino is None:
        return []

    img_pil = BGR_to_RGB(q_bgr)
    x = dino_tfm(img_pil).unsqueeze(0).to(device)
    feats = dino_model.get_intermediate_layers(x, n=1, return_class_token=False)
    feat = feats[0].squeeze(0).mean(dim=0)
    feat = F.normalize(feat, dim=0)
    q_emb = feat.cpu().numpy().astype(np.float32)

    bank_embs = bank_dino["embs"]
    sims = bank_embs @ q_emb
    top_m = min(top_m, len(sims))
    top_idx = np.argsort(sims)[::-1][:top_m]
    return [Path(bank_dino["paths"][i]) for i in top_idx]


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


def draw_text_box(img, lines, org=(10, 20), line_h=18):
    out = img.copy()
    x, y = org
    for i, line in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(out, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def colorize_patch_mask(mask_patch, out_h, out_w, color=(0, 0, 255), alpha=0.55, base=None):
    mask_u8 = mask_patch.astype(np.uint8)
    mask_rs = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    if base is None:
        base = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out = base.copy()

    color_img = np.zeros_like(out)
    color_img[:] = color
    out[mask_rs] = cv2.addWeighted(out, 1 - alpha, color_img, alpha, 0)[mask_rs]
    return out, mask_rs


def save_alignment_summary(case_dir, q_crop, r_crop, valid_mask, meta_lines=None):
    h, w = q_crop.shape[:2]
    valid_vis = np.zeros((h, w, 3), dtype=np.uint8)
    valid_rs = cv2.resize(valid_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid_vis[valid_rs] = (0, 255, 0)
    valid_vis[~valid_rs] = (100, 100, 100)

    panel = np.hstack([q_crop, r_crop, valid_vis])
    if meta_lines:
        panel = draw_text_box(panel, meta_lines, org=(10, 20))
    cv2.imwrite(str(case_dir / "stage1_alignment_summary.png"), panel)


def save_component_summary(case_dir, q_crop, bin_map, best_comp_mask, flagged_regions):
    h, w = q_crop.shape[:2]
    hot_all, _ = colorize_patch_mask(bin_map, h, w, color=(0, 165, 255), alpha=0.55, base=q_crop)
    best_only, _ = colorize_patch_mask(best_comp_mask, h, w, color=(0, 0, 255), alpha=0.60, base=q_crop)

    bbox_vis = q_crop.copy()
    for i, reg in enumerate(flagged_regions):
        y0, x0, y1, x1 = reg["img_bbox"]
        cv2.rectangle(bbox_vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        txt = f"#{i} s={reg['score']:.3f} a={reg['area']}"
        cv2.putText(bbox_vis, txt, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    panel = np.vstack([np.hstack([q_crop, hot_all]), np.hstack([best_only, bbox_vis])])
    panel = draw_text_box(
        panel,
        ["TL: q_crop", "TR: compound all hot-zone", "BL: best component", "BR: flagged component bbox"],
        org=(10, 20),
    )
    cv2.imwrite(str(case_dir / "stage3_component_summary.png"), panel)


def make_dist_overlay(base_bgr, dist_map, color_map=cv2.COLORMAP_JET, alpha=0.45, abs_min=0.0, abs_max=1.0):
    h, w = base_bgr.shape[:2]
    d = dist_map.astype(np.float32)
    if d.size == 0:
        return base_bgr.copy()

    d_vis = np.clip((d - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)
    heat = (d_vis * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, color_map)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)

    return cv2.addWeighted(base_bgr, 1 - alpha, heat, alpha, 0)


def save_flagged_region_visuals(case_dir, verifier_results):
    for i, reg in enumerate(verifier_results):
        sub_dir = case_dir / f"bbox_{i:02d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        q_region = reg["q_region"]
        r_region = reg["r_region"]
        mask_region = reg["mask_region"]
        vscore = reg["verifier_score"]
        dist_map = reg["verifier_dist_map"]
        top_p_mask = reg.get("verifier_top_p_mask", None)
        top_p_thr = reg.get("verifier_top_p_thr", None)
        top_k = reg.get("verifier_top_k", None)
        top_p = reg.get("verifier_top_p", None)

        cv2.imwrite(str(sub_dir / "q_region.png"), q_region)
        cv2.imwrite(str(sub_dir / "r_region.png"), r_region)
        cv2.imwrite(str(sub_dir / "mask_region.png"), (mask_region.astype(np.uint8) * 255))

        q_overlay = make_dist_overlay(q_region, dist_map)
        r_overlay = make_dist_overlay(r_region, dist_map)

        q_overlay = draw_text_box(
            q_overlay,
            [f"bbox verifier score = {vscore:.4f}", f"img_bbox = {reg['img_bbox']}", f"patch_bbox = {reg['patch_bbox']}"],
            org=(8, 18),
        )
        r_overlay = draw_text_box(
            r_overlay,
            [f"component score = {reg['score']:.4f}", f"area = {reg['area']}", f"peak = {reg['peak']:.4f}"],
            org=(8, 18),
        )

        cv2.imwrite(str(sub_dir / "verifier_pair.png"), np.hstack([q_overlay, r_overlay]))

        d = dist_map.astype(np.float32)
        d_vis = np.clip(d, 0.0, 1.0)
        heat = (d_vis * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(str(sub_dir / "verifier_heat.png"), heat)
        np.save(sub_dir / "verifier_dist_map.npy", dist_map)

        if top_p_mask is not None:
            h, w = q_region.shape[:2]
            top_mask_rs = cv2.resize(top_p_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            cv2.imwrite(str(sub_dir / "verifier_top_p_mask.png"), (top_mask_rs.astype(np.uint8) * 255))

            cyan = np.zeros_like(q_region)
            cyan[:] = (255, 255, 0)
            q_top = q_region.copy()
            r_top = r_region.copy()
            q_blend = cv2.addWeighted(q_region, 0.35, cyan, 0.65, 0)
            r_blend = cv2.addWeighted(r_region, 0.35, cyan, 0.65, 0)
            q_top[top_mask_rs] = q_blend[top_mask_rs]
            r_top[top_mask_rs] = r_blend[top_mask_rs]
            q_top = draw_text_box(
                q_top,
                [f"top_p = {top_p}", f"top_k = {top_k}", f"top_thr = {top_p_thr:.4f}" if top_p_thr is not None else "top_thr = NA"],
                org=(8, 18),
            )
            cv2.imwrite(str(sub_dir / "verifier_top_p_pair.png"), np.hstack([q_top, r_top]))


def save_verifier_summary(case_dir, q_crop, verifier_results):
    vis = q_crop.copy()
    for i, reg in enumerate(verifier_results):
        y0, x0, y1, x1 = reg["img_bbox"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
        txt = f"#{i} comp={reg['score']:.3f} ver={reg['verifier_score']:.3f}"
        cv2.putText(vis, txt, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(case_dir / "stage5_verifier_summary.png"), vis)


def save_verified_bbox_overlay(
    case_dir,
    q_crop,
    verified_regions,
    alpha=0.45,
    abs_min=0.0,
    abs_max=1.0,
    vis_thr_abs=0.30,
):
    vis = q_crop.copy()

    for reg in verified_regions:
        y0, x0, y1, x1 = reg["img_bbox"]
        dist_map = reg["verifier_dist_map"].astype(np.float32)

        if dist_map.size == 0 or (y1 - y0) <= 0 or (x1 - x0) <= 0:
            continue

        d_vis = np.clip((dist_map - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)
        thr_norm = np.clip((vis_thr_abs - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)
        mask = d_vis >= thr_norm
        if mask.sum() == 0:
            continue

        heat = (d_vis * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        mask_rs = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)

        roi = vis[y0:y1, x0:x1].copy()
        blended = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
        roi[mask_rs] = blended[mask_rs]
        vis[y0:y1, x0:x1] = roi

        cv2.putText(vis, f"{reg['verifier_score']:.3f}", (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(case_dir / "stage6_verified_bbox_overlay_abs.png"), vis)


def save_verified_bbox_top_p_overlay(case_dir, q_crop, verified_regions, alpha=0.65):
    vis = q_crop.copy()

    for i, reg in enumerate(verified_regions):
        y0, x0, y1, x1 = reg["img_bbox"]
        top_p_mask = reg.get("verifier_top_p_mask", None)

        if top_p_mask is None or (y1 - y0) <= 0 or (x1 - x0) <= 0:
            continue

        mask_rs = cv2.resize(top_p_mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
        roi = vis[y0:y1, x0:x1].copy()

        color = np.zeros_like(roi)
        color[:] = (255, 255, 0)
        blended = cv2.addWeighted(roi, 1 - alpha, color, alpha, 0)
        roi[mask_rs] = blended[mask_rs]
        vis[y0:y1, x0:x1] = roi

        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
        cv2.putText(vis, f"#{i} ver={reg['verifier_score']:.3f}", (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(case_dir / "stage6_verified_bbox_top_p_overlay_rel.png"), vis)


def run_calibration(
    bank_dir: Path,
    th_calib_dir: Path,
    out_dir: Path,
    sg,
    cc_backbone,
    verifier_backbone,
    device: str,
    cfg: dict,
    plc_idx: str = "",
    radius: int = 1,
    calib_method: str = "robust",
    cc_k: float = 1.5,
    final_k: float = 2.5,
    calib_max_imgs: int = 30,
    calib_n_ref: int = 5,
    seed: int = 0,
    dino_model=None,
    dino_tfm=None,
    bank_dino: dict = None,
    dino_top_m: int = 5,
    proposal_top_k: int = 3,
    final_threshold_floor: float = 0.0,
) -> dict:
    random.seed(seed)

    pcfg = cfg.get("patchcore", {})
    top_p = float(pcfg.get("top_p", 0.05))
    alpha = float(pcfg.get("alpha", 0.6))
    min_cut = float(pcfg.get("min_cut", 0.20))
    sw = float(pcfg.get("singleton_weight", 0.25))
    cma = int(pcfg.get("component_min_area", 2))

    th_paths = list_images(th_calib_dir)
    bank_paths = list_images(bank_dir)

    if len(th_paths) == 0:
        raise RuntimeError(f"th_calib 이미지가 없습니다: {th_calib_dir}")
    if len(bank_paths) == 0:
        raise RuntimeError(f"bank 이미지가 없습니다: {bank_dir}")

    random.shuffle(th_paths)
    use_th = th_paths[:calib_max_imgs]
    out_dir.mkdir(parents=True, exist_ok=True)

    cc_scores = []
    cc_pair_log = []

    for th_idx, q_path in enumerate(use_th):
        q_bgr = cv2.imread(str(q_path))
        if q_bgr is None:
            print(f"  [CALIB-CC-SKIP] 읽기 실패: {q_path.name}")
            continue

        if dino_model is not None and dino_tfm is not None:
            bank_candidates = dino_preselect(q_bgr, bank_dino, dino_model, dino_tfm, device, top_m=dino_top_m)
            if len(bank_candidates) == 0:
                shuffled = bank_paths.copy()
                random.shuffle(shuffled)
                bank_candidates = shuffled[:calib_n_ref]
            else:
                bank_candidates = bank_candidates[:calib_n_ref]
        else:
            shuffled = bank_paths.copy()
            random.shuffle(shuffled)
            bank_candidates = shuffled[:calib_n_ref]

        cand_scores, cand_infos = [], []
        for r_path in bank_candidates:
            r_bgr = cv2.imread(str(r_path))
            if r_bgr is None:
                continue

            score, debug = score_one_pair(
                q_bgr, r_bgr, sg, cc_backbone, device,
                radius=radius, top_p=top_p, alpha=alpha, min_cut=min_cut,
                singleton_weight=sw, component_min_area=cma,
            )

            if score is None:
                print(f"  [CALIB-CC-SKIP] {q_path.name} ↔ {r_path.name}: {debug.get('reason')}")
                continue

            cand_scores.append(float(score))
            cand_infos.append({"r": r_path.name, "score": float(score), "all_comp": debug.get("all_comp_scores", []), "debug": debug})

        if len(cand_scores) == 0:
            print(f"  [CALIB-CC-SKIP] {q_path.name}: 모든 bank 후보 실패")
            continue

        best_idx = int(np.argmin(cand_scores))
        best_info = cand_infos[best_idx]
        best_score = float(cand_scores[best_idx])

        comp_scores = [float(c["score"]) for c in best_info.get("all_comp", [])]
        if len(comp_scores) == 0:
            print(f"  [CALIB-CC-SKIP] {q_path.name}: best-match에 component가 없습니다.")
            continue

        cc_scores.extend(comp_scores)
        cc_pair_log.append({
            "q": q_path.name,
            "best_r": best_info["r"],
            "score": round(best_score, 6),
            "n_comps": len(comp_scores),
            "comp_scores": [round(v, 6) for v in comp_scores],
            "all_cands": [{"r": ci["r"], "score": round(ci["score"], 6)} for ci in cand_infos],
        })

        print(
            f"  [CALIB-CC] {th_idx+1}/{len(use_th)} {q_path.name} "
            f"→ best={best_info['r']} score={best_score:.4f} "
            f"comps={len(comp_scores)} comp_max={max(comp_scores):.4f}"
        )

    if len(cc_scores) == 0:
        raise RuntimeError("캘리브레이션 실패: compound score를 하나도 수집하지 못했습니다.")

    cc_scores_arr = np.array(cc_scores, dtype=np.float32)
    cc_percentile = float(cc_k) if calib_method == "percentile" else 99.0
    cc_calib = calibrate_threshold(
        cc_scores_arr,
        method=calib_method,
        k=cc_k,
        percentile=cc_percentile,
        trim_outliers=True,
        trim_k=3.5,
    )
    compound_thr = float(cc_calib["threshold"])

    calib_final_scores = []
    calib_final_log = []

    for th_idx, q_path in enumerate(use_th):
        q_bgr = cv2.imread(str(q_path))
        if q_bgr is None:
            print(f"  [CALIB-FINAL-SKIP] 읽기 실패: {q_path.name}")
            continue

        if dino_model is not None and dino_tfm is not None:
            bank_candidates = dino_preselect(q_bgr, bank_dino, dino_model, dino_tfm, device, top_m=dino_top_m)
            if len(bank_candidates) == 0:
                shuffled = bank_paths.copy()
                random.shuffle(shuffled)
                bank_candidates = shuffled[:calib_n_ref]
            else:
                bank_candidates = bank_candidates[:calib_n_ref]
        else:
            shuffled = bank_paths.copy()
            random.shuffle(shuffled)
            bank_candidates = shuffled[:calib_n_ref]

        cand_scores, cand_infos = [], []
        for r_path in bank_candidates:
            r_bgr = cv2.imread(str(r_path))
            if r_bgr is None:
                continue

            score, debug = score_one_pair(
                q_bgr, r_bgr, sg, cc_backbone, device,
                radius=radius, top_p=top_p, alpha=alpha, min_cut=min_cut,
                singleton_weight=sw, component_min_area=cma,
            )
            if score is None:
                continue

            cand_scores.append(float(score))
            cand_infos.append({"r_path": r_path, "score": float(score), "debug": debug})

        if len(cand_scores) == 0:
            print(f"  [CALIB-FINAL-SKIP] {q_path.name}: 모든 bank 후보 실패")
            continue

        best_idx = int(np.argmin(cand_scores))
        best_score = float(cand_scores[best_idx])
        best_info = cand_infos[best_idx]
        best_debug = best_info["debug"]

        all_comps = best_debug.get("all_comp_scores", [])
        if len(all_comps) == 0:
            print(f"  [CALIB-FINAL-SKIP] {q_path.name}: component 없음")
            continue

        topk_comps = sorted(all_comps, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:proposal_top_k]

        flagged_regions = build_flagged_component_regions(
            flagged_comps=topk_comps,
            q_crop=best_debug["q_crop"],
            r_crop=best_debug["r_crop"],
            valid_mask=best_debug["valid_mask"],
            patch_margin=1,
            crop_margin_ratio=0.20,
            min_patch_area=2,
            min_crop_size=96,
        )

        if len(flagged_regions) == 0:
            print(f"  [CALIB-FINAL-SKIP] {q_path.name}: region 없음")
            continue

        bbox_scores_this_img = []
        for reg in flagged_regions:
            out = verify_bbox_with_local_search(
                q_region=reg["q_region"],
                r_region=reg["r_region"],
                backbone=verifier_backbone,
                device=device,
                radius=1,
                top_p=0.10,
            )
            bbox_scores_this_img.append(float(out["score"]))

        if len(bbox_scores_this_img) == 0:
            print(f"  [CALIB-FINAL-SKIP] {q_path.name}: verifier score 없음")
            continue

        bbox_scores_sorted = sorted(bbox_scores_this_img, reverse=True)
        final_score = float(max(bbox_scores_sorted)) if bbox_scores_sorted else 0.0
        calib_final_scores.append(final_score)

        calib_final_log.append({
            "q": q_path.name,
            "best_r": best_info["r_path"].name,
            "compound_score": round(best_score, 6),
            "bbox_scores": [round(v, 6) for v in bbox_scores_this_img],
            "bbox_scores_sorted": [round(v, 6) for v in bbox_scores_sorted],
            "final_score": round(final_score, 6),
            "n_regions": len(flagged_regions),
        })

        print(
            f"  [CALIB-FINAL] {th_idx+1}/{len(use_th)} {q_path.name} "
            f"→ best={best_info['r_path'].name} compound={best_score:.4f} "
            f"bbox_n={len(bbox_scores_this_img)} final={final_score:.4f}"
        )

    if len(calib_final_scores) >= 5:
        final_arr = np.array(calib_final_scores, dtype=np.float32)
        final_percentile = float(final_k) if calib_method == "percentile" else 99.0
        final_calib = calibrate_threshold(
            final_arr,
            method=calib_method,
            k=final_k,
            percentile=final_percentile,
            trim_outliers=False,
            trim_k=3.5,
        )
        final_threshold = float(final_calib["threshold"])
    else:
        final_threshold = 0.49
        final_calib = None
        print("[CALIB WARN] final score 샘플 부족, 폴백 0.49 사용")

    if final_threshold_floor > 0 and final_threshold < final_threshold_floor:
        print(f"[CALIB FIX] final_thr {final_threshold:.4f} → {final_threshold_floor:.4f} (floor)")
        final_threshold = float(final_threshold_floor)

    from datetime import datetime
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    thr_json = {
        "plc_idx": plc_idx,
        "repr_mode": "compound_cc_max_verifier",
        "method": calib_method,
        "cc_k": float(cc_k),
        "threshold": float(compound_thr),
        "num_th": int(len(cc_scores)),
        "final_k": float(final_k),
        "final_threshold": float(final_threshold),
        "verifier_threshold": float(final_threshold),
        "num_final": int(len(calib_final_scores)),
        "proposal_top_k": int(proposal_top_k),
        "created_at": created_at,
        "radius": radius,
        "top_p": top_p,
        "alpha": alpha,
        "min_cut": min_cut,
        "final_threshold_floor": float(final_threshold_floor),
    }

    if cc_calib is not None:
        thr_json["cc_stats"] = {
            "median": round(cc_calib["median"], 5),
            "mad": round(cc_calib["mad"], 5),
            "mean": round(cc_calib["mean"], 5),
            "std": round(cc_calib["std"], 5),
            "min": round(cc_calib["min"], 5),
            "max": round(cc_calib["max"], 5),
            "top5": cc_calib["top5"],
        }

    if final_calib is not None:
        thr_json["final_stats"] = {
            "median": round(final_calib["median"], 5),
            "mad": round(final_calib["mad"], 5),
            "mean": round(final_calib["mean"], 5),
            "std": round(final_calib["std"], 5),
            "min": round(final_calib["min"], 5),
            "max": round(final_calib["max"], 5),
            "top5": final_calib["top5"],
        }

    (out_dir / "threshold.json").write_text(json.dumps(thr_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "calib_pair_log.json").write_text(json.dumps(cc_pair_log, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "calib_final_log.json").write_text(json.dumps(calib_final_log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[CALIB] threshold.json 저장 완료: {out_dir / 'threshold.json'}")
    print(f"[CALIB] compound_thr={compound_thr:.4f}, num_th={len(cc_scores)}")
    print(f"[CALIB] final_thr={final_threshold:.4f}, num_final={len(calib_final_scores)}")

    return thr_json


def run_inference(
    bank_dir: Path,
    query_dir: Path,
    out_dir: Path,
    sg,
    cc_backbone,
    verifier_backbone,
    device: str,
    cfg: dict,
    radius: int = 1,
    n_ref_candidates: int = 5,
    seed: int = 0,
    dino_model=None,
    dino_tfm=None,
    bank_dino: dict = None,
    dino_top_m: int = 5,
    proposal_top_k: int = 3,
):
    random.seed(seed)
    all_infer_results = []

    thr_path = out_dir / "threshold.json"
    if not thr_path.exists():
        raise FileNotFoundError(f"threshold.json이 없습니다. --mode calib를 먼저 실행하세요: {thr_path}")

    meta = json.loads(thr_path.read_text(encoding="utf-8"))
    thr = float(meta["threshold"])
    final_thr = float(meta.get("verifier_threshold", meta.get("final_threshold", 0.49)))

    if meta.get("method") == "robust":
        k_str = meta.get("robust_k", meta.get("k", "NA"))
    elif meta.get("method") == "gaussian":
        k_str = meta.get("gaussian_k", meta.get("k", "NA"))
    else:
        k_str = meta.get("percentile", meta.get("k", "NA"))

    print(f"\n[INFER] compound_thr={thr:.4f} (method={meta.get('method')}, k={k_str})")
    print(f"[INFER] final_thr={final_thr:.4f} (max verifier 기준)")

    pcfg = cfg.get("patchcore", {})
    top_p = float(pcfg.get("top_p", 0.05))
    alpha = float(pcfg.get("alpha", 0.6))
    min_cut = float(pcfg.get("min_cut", 0.20))
    sw = float(pcfg.get("singleton_weight", 0.25))
    cma = int(pcfg.get("component_min_area", 2))

    ref_paths = list_images(bank_dir)
    query_paths = list_images(query_dir)

    if len(ref_paths) == 0:
        raise RuntimeError(f"bank 이미지 없음: {bank_dir}")
    if len(query_paths) == 0:
        print(f"[INFER] query 이미지 없음: {query_dir}")
        return

    eval_y_true, eval_y_pred = [], []

    for q_path in query_paths:
        print(f"\n[INFER] {q_path.name}")
        q_bgr = cv2.imread(str(q_path))
        if q_bgr is None:
            print(f"  [FAIL] 이미지 읽기 실패: {q_path}")
            continue

        case_dir = out_dir / q_path.stem
        case_dir.mkdir(parents=True, exist_ok=True)

        if dino_model is not None and bank_dino is not None:
            candidates_to_try = dino_preselect(q_bgr, bank_dino, dino_model, dino_tfm, device, top_m=dino_top_m)
            if len(candidates_to_try) == 0:
                shuffled_refs = ref_paths.copy()
                random.shuffle(shuffled_refs)
                candidates_to_try = shuffled_refs[:n_ref_candidates]
            else:
                candidates_to_try = candidates_to_try[:n_ref_candidates]
        else:
            shuffled_refs = ref_paths.copy()
            random.shuffle(shuffled_refs)
            candidates_to_try = shuffled_refs[:n_ref_candidates]

        cand_scores, cand_debugs = [], []

        for r_path in candidates_to_try:
            r_bgr = cv2.imread(str(r_path))
            if r_bgr is None:
                continue

            score, debug = score_one_pair(
                q_bgr, r_bgr, sg, cc_backbone, device,
                radius=radius, top_p=top_p, alpha=alpha, min_cut=min_cut,
                singleton_weight=sw, component_min_area=cma,
            )

            if score is None:
                print(f"  [SKIP] {r_path.name}: {debug.get('reason')}")
                continue

            cand_scores.append(float(score))
            cand_debugs.append({"r_path": r_path, "score": float(score), "debug": debug})

        if len(cand_scores) == 0:
            print("  [FAIL] 모든 ref 후보 매칭 실패")
            continue

        best_idx = int(np.argmin(cand_scores))
        best_score = float(cand_scores[best_idx])
        best_info = cand_debugs[best_idx]
        best_debug = best_info["debug"]

        cv2.imwrite(str(case_dir / "q_crop.png"), best_debug["q_crop"])
        cv2.imwrite(str(case_dir / "r_crop.png"), best_debug["r_crop"])

        save_cc_heatmap(case_dir, best_debug["q_crop"], best_debug["dist_map"], best_debug["valid_mask"])

        all_comps = best_debug.get("all_comp_scores", [])
        topk_comps = sorted(all_comps, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:proposal_top_k]

        flagged_regions = build_flagged_component_regions(
            flagged_comps=topk_comps,
            q_crop=best_debug["q_crop"],
            r_crop=best_debug["r_crop"],
            valid_mask=best_debug["valid_mask"],
            patch_margin=1,
            crop_margin_ratio=0.20,
            min_patch_area=2,
            min_crop_size=96,
        )

        verifier_results = []
        for reg in flagged_regions:
            out = verify_bbox_with_local_search(
                q_region=reg["q_region"],
                r_region=reg["r_region"],
                backbone=verifier_backbone,
                device=device,
                radius=1,
                top_p=0.10,
            )
            verifier_results.append({
                **reg,
                "verifier_score": float(out["score"]),
                "verifier_dist_map": out["dist_map"],
                "verifier_feat_hw": out["feat_hw"],
                "verifier_top_p_mask": out["top_p_mask"],
                "verifier_top_p_thr": out["top_p_thr"],
                "verifier_top_k": out["top_k"],
                "verifier_top_p": out["top_p"],
            })

        verifier_scores_sorted = sorted([r["verifier_score"] for r in verifier_results], reverse=True)
        final_score = float(max(verifier_scores_sorted)) if verifier_scores_sorted else 0.0
        is_anomaly = bool(final_score > final_thr)
        best_verifier_score = max(verifier_scores_sorted, default=0.0)
        verified_regions = [r for r in verifier_results if r["verifier_score"] > final_thr]

        save_alignment_summary(
            case_dir,
            best_debug["q_crop"],
            best_debug["r_crop"],
            best_debug["valid_mask"],
            meta_lines=[
                f"best_ref = {best_info['r_path'].name}",
                f"compound_best = {best_score:.5f}",
                f"compound_thr = {thr:.5f}",
                f"final_score = {final_score:.5f}",
                f"final_thr = {final_thr:.5f}",
                f"inliers = {best_debug['inliers']}",
                f"peak = {best_debug['peak']:.4f}, area = {best_debug['area']}",
            ],
        )

        save_component_summary(case_dir, best_debug["q_crop"], best_debug["bin_map"], best_debug["best_comp_mask"], flagged_regions)
        save_flagged_region_visuals(case_dir, verifier_results)
        save_verifier_summary(case_dir, best_debug["q_crop"], verifier_results)
        save_verified_bbox_top_p_overlay(case_dir, best_debug["q_crop"], verifier_results, alpha=0.65)
        save_verified_bbox_overlay(case_dir, best_debug["q_crop"], verified_regions, alpha=0.45)

        label_str = "ANOMALY" if is_anomaly else "NORMAL"
        n_flagged = len(topk_comps)
        n_regions = len(flagged_regions)
        n_verified = len(verified_regions)

        print(
            f"  [RESULT] {label_str} | "
            f"compound={best_score:.4f} (thr={thr:.4f}) | "
            f"final={final_score:.4f} (thr={final_thr:.4f}) | "
            f"props={n_flagged}, regions={n_regions}, verified={n_verified}"
        )

        result_meta = {
            "query": str(q_path),
            "query_name": q_path.name,
            "best_ref": str(best_info["r_path"]),
            "best_ref_name": best_info["r_path"].name,
            "label": label_str,
            "is_anomaly": bool(is_anomaly),
            "score": round(best_score, 6),
            "compound_score": round(best_score, 6),
            "threshold": round(thr, 6),
            "compound_threshold": round(thr, 6),
            "final_score": round(final_score, 6),
            "final_threshold": round(final_thr, 6),
            "best_verifier_score": round(best_verifier_score, 6),
            "proposal_top_k": int(proposal_top_k),
            "num_flagged_components": int(n_flagged),
            "num_flagged_regions": int(n_regions),
            "num_verified_regions": int(n_verified),
            "flagged_comp_scores": [round(float(c["score"]), 6) for c in topk_comps],
            "all_cand_scores": [round(float(s), 6) for s in cand_scores],
            "verifier_threshold": round(final_thr, 6),
            "flagged_regions": [
                {
                    "idx": i,
                    "component_score": round(float(r["score"]), 6),
                    "verifier_score": round(float(r["verifier_score"]), 6),
                    "final_bbox_score": round(float(r["verifier_score"]), 6),
                    "is_verified": bool(r["verifier_score"] > final_thr),
                    "area": int(r["area"]),
                    "peak": round(float(r["peak"]), 6),
                    "mean": round(float(r["mean"]), 6),
                    "patch_bbox": list(r["patch_bbox"]),
                    "img_bbox": list(r["img_bbox"]),
                    "verifier_top_p": r.get("verifier_top_p", None),
                    "verifier_top_k": int(r.get("verifier_top_k", 0)),
                    "verifier_top_p_thr": round(float(r.get("verifier_top_p_thr", 0.0)), 6),
                }
                for i, r in enumerate(verifier_results)
            ],
            "verified_regions": [
                {
                    "idx": i,
                    "final_bbox_score": round(float(r["verifier_score"]), 6),
                    "component_score": round(float(r["score"]), 6),
                    "verifier_score": round(float(r["verifier_score"]), 6),
                    "img_bbox": list(r["img_bbox"]),
                    "patch_bbox": list(r["patch_bbox"]),
                }
                for i, r in enumerate(verified_regions)
            ],
        }

        (case_dir / "result.json").write_text(json.dumps(result_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        all_infer_results.append(result_meta)

        eval_y_true.append(1 if q_path.name.startswith("abnormal_") else 0)
        eval_y_pred.append(1 if is_anomaly else 0)

    if len(eval_y_true) > 0:
        _print_eval_report(eval_y_true, eval_y_pred, out_dir)

    (out_dir / "infer_all_results.json").write_text(json.dumps(all_infer_results, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_summary_gallery(out_dir)
    plot_compound_scores(out_dir)
    plot_bbox_scores(out_dir)


def _save_summary_gallery(out_dir: Path):
    gallery_dir = out_dir / "모음"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for case_dir in sorted(out_dir.iterdir()):
        result_path = case_dir / "result.json"
        if not result_path.exists():
            continue

        meta = json.loads(result_path.read_text(encoding="utf-8"))
        label = meta.get("label", "ANOMALY" if meta["is_anomaly"] else "NORMAL")
        score = meta.get("final_score", meta.get("score", 0.0))
        threshold = meta.get("final_threshold", meta.get("threshold", 0.0))

        overlay_path = None
        candidates = ["stage6_verified_bbox_overlay_abs.png", "q_crop.png"]
        for c in candidates:
            p = case_dir / c
            if p.exists():
                overlay_path = p
                break

        if overlay_path is None:
            continue

        overlay_img = cv2.imread(str(overlay_path))
        if overlay_img is None:
            continue

        r_crop_path = case_dir / "r_crop.png"
        ref_img = cv2.imread(str(r_crop_path)) if r_crop_path.exists() else None

        h_o, w_o = overlay_img.shape[:2]
        if ref_img is not None:
            h_r, w_r = ref_img.shape[:2]
            if h_r != h_o:
                ref_img = cv2.resize(ref_img, (int(w_r * h_o / h_r), h_o))
            divider = np.ones((h_o, 3, 3), dtype=np.uint8) * 200
            side_by_side = np.hstack([ref_img, divider, overlay_img])
        else:
            side_by_side = overlay_img

        h_s, w_s = side_by_side.shape[:2]
        bar = np.zeros((36, w_s, 3), dtype=np.uint8)

        if "ANOMALY" in label:
            bar[:] = (0, 30, 200)
        else:
            bar[:] = (0, 140, 0)

        txt = f"{label}  [FINAL]  score={score:.5f} / thr={threshold:.5f}  ← 정상(ref) | 쿼리+mask →   [{case_dir.name[:20]}]"
        cv2.putText(bar, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        annotated = np.vstack([bar, side_by_side])
        prefix = "ANOMALY" if meta["is_anomaly"] else "NORMAL"
        out_name = f"{prefix}_{case_dir.name}.png"
        cv2.imwrite(str(gallery_dir / out_name), annotated)
        saved += 1

    print(f"\n[모음] 갤러리 {saved}장 저장 완료: {gallery_dir}")


def _print_eval_report(y_true, y_pred, out_dir: Path):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    acc = (tp + tn) / len(y_t)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 40)
    print(" [Evaluation Report]")
    print("=" * 40)
    print(f" Total: {len(y_t)}  Anomaly: {np.sum(y_t==1)}  Normal: {np.sum(y_t==0)}")
    print(f" TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f" Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
    print("=" * 40 + "\n")

    report = (
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}\n"
        f"Precision={precision:.4f}  Recall={recall:.4f}  "
        f"F1={f1:.4f}  Acc={acc:.4f}\n"
    )
    (out_dir / "evaluation_report.txt").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", required=True, help="실험 장소명 (recv/{place}/bank, query 구조)")
    parser.add_argument("--mode", required=True, choices=["calib", "infer"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=int, default=1)

    parser.add_argument("--calib_method", default=None, choices=["robust", "gaussian", "percentile"])
    parser.add_argument("--calib_k", type=float, default=None)
    parser.add_argument("--cc_k", type=float, default=None)
    parser.add_argument("--final_k", type=float, default=None)
    parser.add_argument("--final_threshold_floor", type=float, default=0.0)

    parser.add_argument("--calib_max_imgs", type=int, default=30)
    parser.add_argument("--calib_n_ref", type=int, default=5)
    parser.add_argument("--n_ref_candidates", type=int, default=5)

    parser.add_argument("--dino_model", type=str, default=None)
    parser.add_argument("--dino_top_m", type=int, default=5)

    args = parser.parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg(ROOT)
    calib_cfg = cfg.get("calib", {})

    calib_method = args.calib_method or calib_cfg.get("method", "robust")

    if args.calib_k is not None:
        calib_k = float(args.calib_k)
    else:
        if calib_method == "robust":
            calib_k = float(calib_cfg.get("robust_k", calib_cfg.get("k", 3.0)))
        elif calib_method == "gaussian":
            calib_k = float(calib_cfg.get("gaussian_k", calib_cfg.get("k", 3.0)))
        elif calib_method == "percentile":
            calib_k = float(calib_cfg.get("percentile", 99.0))
        else:
            raise ValueError(f"알 수 없는 calib_method: {calib_method}")

    cc_k = float(args.cc_k if args.cc_k is not None else calib_cfg.get("cc_k", calib_k))
    final_k = float(args.final_k if args.final_k is not None else calib_cfg.get("final_k", calib_k))

    img_size = int(cfg.get("embed", {}).get("img_size", 560))

    from backbone_wrapper import build_local_backbone

    cc_backbone = build_local_backbone(backbone_name="resnet18_layer3", img_size=img_size)
    verifier_backbone = build_local_backbone(backbone_name="resnet18_layer3", img_size=224)

    dino_model_inst, dino_tfm_inst = None, None
    if args.dino_model is not None:
        print(f"[DINO] 모델 로드 중: {args.dino_model}")
        dino_model_inst, _ = load_dino_model(model_name=args.dino_model, device=device)
        dino_tfm_inst = make_dino_transform(img_size=img_size)
        print("[DINO] 로드 완료")

    sg_cfg = SuperGlueMatchConfig(
        resize_long_side=640,
        weights="indoor",
        max_keypoints=1024,
        keypoint_threshold=0.003,
        match_threshold=0.2,
        sinkhorn_iterations=20,
    )
    sg = SuperGlueMatcher(sg_cfg, device=device)

    bank_dir = Path(ROOT) / args.place / "bank"
    th_calib_dir = Path(ROOT) / args.place / "th_calib"
    query_dir = Path(ROOT) / args.place / "query"
    out_dir = Path(OUT_ROOT) / args.place
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "calib":
        print(f"\n[MODE] CALIBRATION  place={args.place}")
        bank_dino_data = None
        if dino_model_inst is not None:
            bank_dino_data = build_dino_bank(bank_dir, dino_model_inst, dino_tfm_inst, device)

        run_calibration(
            bank_dir=bank_dir,
            th_calib_dir=th_calib_dir,
            out_dir=out_dir,
            sg=sg,
            cc_backbone=cc_backbone,
            verifier_backbone=verifier_backbone,
            device=device,
            cfg=cfg,
            plc_idx=args.place,
            radius=args.radius,
            calib_method=calib_method,
            cc_k=cc_k,
            final_k=final_k,
            calib_max_imgs=args.calib_max_imgs,
            calib_n_ref=args.calib_n_ref,
            seed=args.seed,
            dino_model=dino_model_inst,
            dino_tfm=dino_tfm_inst,
            bank_dino=bank_dino_data,
            dino_top_m=args.dino_top_m,
            final_threshold_floor=args.final_threshold_floor,
        )
        return

    if args.mode == "infer":
        print(f"\n[MODE] INFERENCE  place={args.place}")
        bank_dino_infer = None
        if dino_model_inst is not None:
            bank_dino_infer = build_dino_bank(bank_dir, dino_model_inst, dino_tfm_inst, device)

        run_inference(
            bank_dir=bank_dir,
            query_dir=query_dir,
            out_dir=out_dir,
            sg=sg,
            cc_backbone=cc_backbone,
            verifier_backbone=verifier_backbone,
            device=device,
            cfg=cfg,
            radius=args.radius,
            n_ref_candidates=args.n_ref_candidates,
            seed=args.seed,
            dino_model=dino_model_inst,
            dino_tfm=dino_tfm_inst,
            bank_dino=bank_dino_infer,
            dino_top_m=args.dino_top_m,
        )
        return


if __name__ == "__main__":
    main()