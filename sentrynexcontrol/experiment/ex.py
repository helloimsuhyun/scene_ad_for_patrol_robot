#!/usr/bin/env python3

# ============================================================
# ex.py  ─  실험용 이상감지 파이프라인
#
# [실행 모드]
#   --mode calib  : bank 이미지 전체를 추론해 "정상 분포" 모델링 → threshold.json 저장
#   --mode infer  : query 이미지를 추론해 threshold와 비교 → 이상/정상 판정
#   --mode vis    : 기존 시각화 디버그 모드 (SAM 포함, 기본)
#
# [캘리브레이션 아이디어]
#   정상(bank) 이미지들을 서로 쿼리 ↔ 레퍼런스로 추론하여
#   compound score(connected component excess score) 분포를 획득.
#   이를 MAD(Robust) 또는 Gaussian 기반으로 thresholding → 오탐 기각.
# ============================================================

import argparse
from pathlib import Path
import json
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage

from cnn_emb import load_model, extract_grid_layers, make_aligned_local_transform
from dino_emb import load_dino_model, extract_dino_grid, make_dino_transform
from config import load_cfg
from warp_utils import (
    warp_query_to_bank,
    make_patch_valid_mask,
    crop_common_safe_region,
)
from matcher import SuperGlueMatcher, SuperGlueMatchConfig
from PIL import Image
from skimage.exposure import match_histograms  # 조명 정규화용
from sam_refine import load_sam_model, refine_with_sam, save_sam_outputs  # SAM 정제

ROOT = "/home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/experiment/recv"
OUT_ROOT = "/home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/experiment/recv/out"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def BGR_to_RGB(img_bgr_uint8):
    img_rgb = img_bgr_uint8[:, :, ::-1]
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    return img_pil

def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


import matplotlib.pyplot as plt


def plot_compound_scores(out_dir: Path):
    path = out_dir / "infer_all_results.json"
    if not path.exists():
        print("[PLOT] infer_all_results.json 없음")
        return

    data = json.loads(path.read_text())

    normal_scores = []
    abnormal_scores = []

    for item in data:
        is_abnormal = Path(item["query"]).name.startswith("abnormal_")
        s = float(item.get("score", 0.0))

        if is_abnormal:
            abnormal_scores.append(s)
        else:
            normal_scores.append(s)

    plt.figure(figsize=(10, 4))

    # 🔥 jitter
    x_n = np.random.normal(0, 0.04, size=len(normal_scores))
    x_a = np.random.normal(1, 0.04, size=len(abnormal_scores))

    plt.scatter(x_n, normal_scores, alpha=0.6, s=15, label="normal")
    plt.scatter(x_a, abnormal_scores, alpha=0.6, s=15, label="abnormal")

    plt.xticks([0, 1], ["normal", "abnormal"])

    # 🔥 threshold line
    thr_path = out_dir / "threshold.json"
    if thr_path.exists():
        meta = json.loads(thr_path.read_text())
        thr = meta.get("threshold", None)
        if thr is not None:
            plt.axhline(thr, linestyle="--")

    # 🔥 ylim 개선 (핵심)
    all_scores = normal_scores + abnormal_scores
    if len(all_scores) > 0:
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

    normal_scores = []
    abnormal_scores = []

    for item in data:
        is_abnormal = Path(item["query"]).name.startswith("abnormal_")

        for r in item.get("flagged_regions", []):
            s = float(r.get("final_bbox_score", 0.0))

            if is_abnormal:
                abnormal_scores.append(s)
            else:
                normal_scores.append(s)

    plt.figure(figsize=(10, 4))

    # 🔥 jitter 적용
    x_n = np.random.normal(0, 0.04, size=len(normal_scores))
    x_a = np.random.normal(1, 0.04, size=len(abnormal_scores))

    plt.scatter(x_n, normal_scores, alpha=0.6, s=15, label="normal")
    plt.scatter(x_a, abnormal_scores, alpha=0.6, s=15, label="abnormal")

    plt.xticks([0, 1], ["normal", "abnormal"])

    # 🔥 verifier threshold
    thr_path = out_dir / "threshold.json"
    if thr_path.exists():
        meta = json.loads(thr_path.read_text())
        thr = meta.get("verifier_threshold", None)
        if thr is not None:
            plt.axhline(thr, linestyle="--")

    # 🔥 ylim 개선
    all_scores = normal_scores + abnormal_scores
    if len(all_scores) > 0:
        low = np.percentile(all_scores, 1)
        high = np.percentile(all_scores, 99)
        plt.ylim(low, high)

    plt.legend()
    plt.savefig(out_dir / "bbox_score_scatter.png")
    plt.close()

    print(f"[PLOT] saved: {out_dir / 'bbox_score_scatter.png'}")


def compute_ssim_map_feature(q_feat, r_feat, valid_mask=None, window_size=7,
                             C1=0.01**2, C2=0.03**2):
    """
    q_feat, r_feat: torch.Tensor, shape (C,H,W)
    valid_mask: np.ndarray or None, shape (H,W), True=valid
    return: np.ndarray, shape (H,W), higher = more different
    """
    assert q_feat.ndim == 3 and r_feat.ndim == 3
    C, H, W = q_feat.shape
    assert r_feat.shape == (C, H, W)

    q = q_feat.unsqueeze(0)   # [1,C,H,W]
    r = r_feat.unsqueeze(0)   # [1,C,H,W]

    window = torch.ones((C, 1, window_size, window_size),
                        device=q.device, dtype=q.dtype) / (window_size * window_size)

    mu_q = F.conv2d(q, window, padding=window_size // 2, groups=C)
    mu_r = F.conv2d(r, window, padding=window_size // 2, groups=C)

    mu_q2 = mu_q * mu_q
    mu_r2 = mu_r * mu_r
    mu_qr = mu_q * mu_r

    sigma_q2 = F.conv2d(q * q, window, padding=window_size // 2, groups=C) - mu_q2
    sigma_r2 = F.conv2d(r * r, window, padding=window_size // 2, groups=C) - mu_r2
    sigma_qr = F.conv2d(q * r, window, padding=window_size // 2, groups=C) - mu_qr

    ssim = ((2 * mu_qr + C1) * (2 * sigma_qr + C2)) / (
        (mu_q2 + mu_r2 + C1) * (sigma_q2 + sigma_r2 + C2) + 1e-8
    )  # [1,C,H,W]

    # 채널 평균 -> [H,W]
    ssim_map = ssim.mean(dim=1).squeeze(0)

    # distance map으로 변환
    dist_map = 1.0 - ssim_map
    dist_map = torch.clamp(dist_map, min=0.0, max=2.0)

    dist = dist_map.detach().cpu().numpy()

    if valid_mask is not None:
        fill_v = float(dist.max()) if dist.size > 0 else 1.0
        dist = np.where(valid_mask, dist, fill_v)

    return dist

def save_named_outputs(save_dir, q_crop, r_crop, dist_map, valid_mask, prefix="dist"):
    save_dir.mkdir(parents=True, exist_ok=True)

    dist_norm = dist_map.astype(np.float32)

    # 절대 cosine 거리값 (0~1) 그대로 시각화
    # → 조명처럼 전체가 올라가면 heatmap 전체가 빨개짐 (상대화 X)
    dist_vis = np.clip(dist_norm, 0.0, 1.0)

    heat = (np.clip(dist_vis, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    invalid = ~valid_mask.astype(bool)
    heat[invalid] = (128, 128, 128)

    # q_crop, r_crop이 완전히 동일한 shape라고 가정 (warp & crop 결과이므로)
    # resize할 목표 (Width, Height) 명시
    target_size = (q_crop.shape[1], q_crop.shape[0])

    heat_rs = cv2.resize(heat, target_size, interpolation=cv2.INTER_NEAREST)
    valid_rs = cv2.resize(valid_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay_q = q_crop.copy()
    overlay_r = r_crop.copy()
    alpha = 0.4

    # valid_rs와 r_crop의 크기가 완벽히 1대1 매칭되므로 에러 발생하지 않음
    overlay_q[valid_rs] = cv2.addWeighted(
        q_crop[valid_rs], 1 - alpha,
        heat_rs[valid_rs], alpha,
        0
    )
    overlay_r[valid_rs] = cv2.addWeighted(
        r_crop[valid_rs], 1 - alpha,
        heat_rs[valid_rs], alpha,
        0
    )

    overlay_q[~valid_rs] = (128, 128, 128)
    overlay_r[~valid_rs] = (128, 128, 128)

    np.save(save_dir / f"{prefix}_map.npy", dist_map)
    cv2.imwrite(str(save_dir / f"{prefix}_heat.png"), heat)
    cv2.imwrite(str(save_dir / f"{prefix}_overlay_q.png"), overlay_q)
    cv2.imwrite(str(save_dir / f"{prefix}_overlay_r.png"), overlay_r)

def compute_dist_map_local_search(q_feat, r_feat, valid_mask=None, radius=1):
    """
    고속화된 (Vectorized) local search dist map 계산 (for문 제거)
    q_feat, r_feat: [C, H, W] (정규화된 상태라고 가정)
    """
    assert q_feat.ndim == 3 and r_feat.ndim == 3
    C, H, W = q_feat.shape
    
    # [1, C, H, W]
    q = q_feat.unsqueeze(0)
    r = r_feat.unsqueeze(0)
    
    # 1) r_feat(Bank)의 각 위치에서 반경(radius) 만큼의 이웃 패치를 묶기 (Unfold)
    k = 2 * radius + 1
    r_unfold = F.unfold(r, kernel_size=k, padding=radius) 
    # [1, C*k*k, H*W] -> [C, k*k, H*W]
    r_unfold = r_unfold.squeeze(0).view(C, k*k, H*W)
    
    # 2) q_feat(Query)를 1D 공간 순서 텐서로 변환 -> [C, 1, H*W]
    q_flat = q.squeeze(0).view(C, 1, H*W)
    
    # 3) 내적 계산 (채널 합산) -> [k*k, H*W]
    sims = (q_flat * r_unfold).sum(dim=0)  
    
    # 4) 이웃 반경 내 최댓값 찾기 -> [H*W]
    best_sim, _ = sims.max(dim=0)  
    
    # 거리 맵 복원: 거리 = 1 - 유사도
    dist_map = 1.0 - best_sim.view(H, W)
    dist = dist_map.detach().cpu().numpy()

    if valid_mask is not None:
        # Invalid 영역은 이상 현상이 아니므로 거리(distance)를 0.0으로 줘서 제외
        dist = np.where(valid_mask, dist, 0.0)

    return dist


# =============================================================================
# ① compound score 계산 함수
#    connected component excess score 방식
#    (distance.py의 _score_connected_excess 로직을 이 실험 파일에서 독립 구현)
# =============================================================================

def compute_compound_score(
    dist_map: np.ndarray,   # (H, W)  ─ local search로 얻은 거리 heatmap
    valid_mask: np.ndarray, # (H, W)  ─ valid patch mask (bool)
    top_p: float = 0.05,    # 상위 top_p 비율만 hotspot으로
    alpha: float = 0.6,     # peak 대비 하한 비율 (min_cut 결정)
    min_cut: float = 0.20,  # 절대 최소 threshold
    singleton_weight: float = 0.25, # 단일 patch component 가중치
    component_min_area: int = 2,    # 1픽셀짜리 component 처리 구분
) -> dict:
    """
    dist_map에서 top-p 선택 → cut = max(alpha*peak, min_cut) 으로 필터
    → 8-connected component를 찾아 excess(dist - cut) 합계 중 최대를 score로 반환.

    [반환]
    {
        "score"         : float      ← 클수록 이상 (0 = 정상)
        "area"          : int        ← best component 크기
        "peak"          : float      ← best component 내 최대 거리
        "mean"          : float      ← best component 내 평균 거리
        "cut"           : float      ← 적용된 threshold
        "n_top"         : int        ← top-p 선택된 픽셀 수
        "bin_map"       : np.ndarray ← cut 이상 전체 hot-zone (0/1, uint8)
        "best_comp_mask": np.ndarray ← score 최고 1개 component 마스크 (bool)
    }
    """
    Hp, Wp = dist_map.shape
    empty_masks = (
        np.zeros((Hp, Wp), dtype=np.uint8),  # bin_map 빈값
        np.zeros((Hp, Wp), dtype=bool),       # best_comp_mask 빈값
    )

    # valid 영역만 보기
    masked = dist_map * valid_mask.astype(np.float32)  # invalid 영역은 0 처리

    # --- top-p 선택 ---
    flat_valid = masked[valid_mask > 0]  # valid 픽셀 값만 추출
    if flat_valid.size == 0:
        return {"score": 0.0, "area": 0, "peak": 0.0, "mean": 0.0,
                "cut": min_cut, "n_top": 0,
                "bin_map": empty_masks[0], "best_comp_mask": empty_masks[1],"valid_count": 0,"all_comp_scores": [],}

    n_top = max(1, int(np.ceil(flat_valid.size * top_p)))
    # 상위 n_top 값의 최솟값을 top_threshold로 삼아 binary map 생성
    top_threshold = np.sort(flat_valid)[-n_top]  # 오름차순 → 마지막 n_top 기준

    peak = float(flat_valid.max())
    # cut은 top-p 하한과 absolute lower bound 중 큰 값
    cut = float(max(alpha * peak, min_cut, top_threshold))

    # --- binary hot-zone map (cut 이상 전체 픽셀) ---
    bin_map = (masked >= cut).astype(np.uint8)  # (Hp, Wp)

    if bin_map.sum() == 0:
        return {
            "score": 0.0,
            "area": 0,
            "peak": peak,
            "mean": 0.0,
            "cut": cut,
            "n_top": n_top,
            "valid_count": int(flat_valid.size),
            "bin_map": bin_map,
            "best_comp_mask": empty_masks[1],
            "all_comp_scores": [],
        }

    # --- 8-connected component 라벨링 ---
    struct = np.ones((3, 3), dtype=np.int32)  # 8-방향 연결
    labeled, n_labels = ndimage.label(bin_map, structure=struct)

    best_score = 0.0
    best_area  = 0
    best_peak  = 0.0
    best_mean  = 0.0
    best_lbl   = -1   # 최고 score component의 label
    valid_count = int(flat_valid.size)  # 정규화용 valid 패치 수

    # [component-level] 모든 component 점수 수집 (캘리브레이션용)
    all_comp_scores = []

    for lbl in range(1, n_labels + 1):
        comp_mask = labeled == lbl
        vals = dist_map[comp_mask]  # dist_map 원본값 사용 (masked 아님)
        excess = np.clip(vals - cut, a_min=0.0, a_max=None)
        area = int(comp_mask.sum())

        # area에 따라 score 방식 다르게 적용
        if area >= component_min_area:
            raw_score = float(excess.sum())
        else:
            raw_score = float(singleton_weight * excess.max()) if excess.size > 0 else 0.0

        # [Fix2] valid 패치 수로 정규화: 단위 패치당 excess
        # 이미지에 따라 valid 팩치 수가 다르면 score 스케일이 달라지는 문제를 방지
        norm_score = raw_score / np.sqrt(max(valid_count, 1))

        # 모든 component 정보 수집 (component-level 캘리브레이션에서 활용)
        all_comp_scores.append({
            "score" : norm_score,
            "mask"  : comp_mask,   # bool, patch grid 크기
            "area"  : area,
            "peak"  : float(vals.max()),
            "mean"  : float(vals.mean()),
        })

        if norm_score > best_score:
            best_score = norm_score
            best_area  = area
            best_peak  = float(vals.max())
            best_mean  = float(vals.mean())
            best_lbl   = lbl  # 최고 score component label 기록

    # best component 마스크 생성 (score 최고 1개만)
    best_comp_mask = (labeled == best_lbl) if best_lbl > 0 \
                     else np.zeros((Hp, Wp), dtype=bool)

    return {
        "score"          : best_score,      # [Fix2] valid 패치 수 정규화된
        "area"           : best_area,
        "peak"           : best_peak,
        "mean"           : best_mean,
        "cut"            : cut,
        "n_top"          : n_top,
        "valid_count"    : valid_count,     # 디버그용
        "bin_map"        : bin_map,         # cut 이상 전체 hot-zone
        "best_comp_mask" : best_comp_mask,  # 최고 score 1개 component
        "valid_count": int(flat_valid.size),
        "all_comp_scores": all_comp_scores, # [component-level] 전체 component 목록
    }



# =============================================================================
# ② threshold 계산 함수 (MAD-Robust / Gaussian 선택 가능)
# =============================================================================

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

    # -----------------------------
    # 1) raw 기준 robust 통계
    # -----------------------------
    raw_median = float(np.median(scores_raw))
    raw_mad = float(np.median(np.abs(scores_raw - raw_median)))
    raw_sigma = float(raw_mad * 1.4826)

    removed_scores = []
    upper_bound = None

    # -----------------------------
    # 2) optional trimming
    # -----------------------------
    if trim_outliers and len(scores_raw) >= 10 and raw_mad > 1e-12:
        upper_bound = float(raw_median + trim_k * raw_sigma)
        keep_mask = scores_raw <= upper_bound
        removed_scores = scores_raw[~keep_mask].tolist()

        # 전부 날아가는 거 방지
        if int(keep_mask.sum()) >= max(5, int(0.8 * len(scores_raw))):
            scores = scores_raw[keep_mask]
        else:
            scores = scores_raw.copy()
    else:
        scores = scores_raw.copy()

    # -----------------------------
    # 3) trimmed(or raw) 기준 threshold 계산
    # -----------------------------
    median = float(np.median(scores))
    mad    = float(np.median(np.abs(scores - median)))
    mean   = float(scores.mean())
    std    = float(scores.std())

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


# =============================================================================
# ③ DINO global preselection 헬퍼
# =============================================================================

@torch.no_grad()
def build_dino_bank(
    bank_dir: Path,
    dino_model,          # load_dino_model()으로 로드한 모델
    dino_tfm,            # make_dino_transform()
    device: str,
    cache_name: str = "dino_bank.npz",
) -> dict:
    """
    bank 이미지들의 DINO global 임베딩을 NPZ로 캐싱.
    이미 같은 파일 목록으로 캐시가 있으면 그대로 로드.

    return: {"embs": np.ndarray (N, D), "paths": list[str]}
    """
    cache_path = bank_dir / cache_name
    bank_paths = sorted([p for p in bank_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMG_EXTS])

    path_strs = [str(p) for p in bank_paths]

    # 캐시가 있으면 paths 목록 확인 후 재사용
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = data["paths"].tolist()
        if cached_paths == path_strs:
            print(f"[DINO Bank] 캐시 로드: {cache_path} ({len(cached_paths)}장)")
            return {"embs": data["embs"], "paths": path_strs}

    # 새로 계산
    print(f"[DINO Bank] 임베딩 계산 중... ({len(bank_paths)}장)")
    embs = []
    valid_paths = []
    for p in bank_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_pil = BGR_to_RGB(img_bgr)
        x = dino_tfm(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

        # CLS 토큰 대신 patch_mean 사용 → 공간 정보 보존
        feats = dino_model.get_intermediate_layers(x, n=1, return_class_token=False)
        feat = feats[0].squeeze(0).mean(dim=0)  # [D] 패치 평균
        feat = F.normalize(feat, dim=0)
        embs.append(feat.cpu().numpy().astype(np.float32))
        valid_paths.append(str(p))

    embs_np = np.stack(embs, axis=0)  # (N, D)
    np.savez_compressed(cache_path, embs=embs_np,
                        paths=np.array(valid_paths, dtype=object))
    print(f"[DINO Bank] 캐시 저장 완료: {cache_path}")
    return {"embs": embs_np, "paths": valid_paths}


@torch.no_grad()
def dino_preselect(
    q_bgr: np.ndarray,
    bank_dino: dict,      # build_dino_bank() 결과
    dino_model,
    dino_tfm,
    device: str,
    top_m: int = 5,
) -> list:
    """
    query BGR → DINO patch_mean 임베딩 추출
    → bank_dino["embs"]와 코사인 유사도 계산
    → top-M 유사한 bank 이미지 Path 목록 반환

    bank_dino가 None이면 빈 리스트 반환 (폴백 처리는 호출 측에서)
    """
    if bank_dino is None:
        return []

    img_pil = BGR_to_RGB(q_bgr)
    x = dino_tfm(img_pil).unsqueeze(0).to(device)

    feats = dino_model.get_intermediate_layers(x, n=1, return_class_token=False)
    feat = feats[0].squeeze(0).mean(dim=0)       # [D]
    feat = F.normalize(feat, dim=0)
    q_emb = feat.cpu().numpy().astype(np.float32)  # (D,)

    bank_embs = bank_dino["embs"]                # (N, D)
    sims = bank_embs @ q_emb                     # (N,): 코사인 유사도

    top_m = min(top_m, len(sims))
    top_idx = np.argsort(sims)[::-1][:top_m]     # 유사도 높은 순
    return [Path(bank_dino["paths"][i]) for i in top_idx]


# =============================================================================
# ③ 쿼리-레퍼런스 1쌍에 대한 compound score 계산 (공통 파이프라인)
# =============================================================================

def score_one_pair(
    q_bgr: np.ndarray,   # 쿼리 이미지 (BGR)
    r_bgr: np.ndarray,   # 레퍼런스 이미지 (BGR)
    sg,                  # SuperGlueMatcher 인스턴스
    backbone,         # CNN (layer3)
    device: str,
    radius: int = 1,
    top_p: float = 0.05,
    alpha: float = 0.6,
    min_cut: float = 0.20,
    singleton_weight: float = 0.25,
    component_min_area: int = 2,
) -> tuple:
    """
    q_bgr, r_bgr 한 쌍에 대해:
      SuperGlue 정합 → warp/crop → CNN feature → dist map → compound score
    반환: (score: float | None, debug: dict)
      score=None이면 실패
    """
    # 1) SuperGlue 정합
    match_res = sg.match_and_estimate(q_bgr, r_bgr)
    if not match_res.get("ok", False):
        return None, {"reason": "align_fail", "detail": match_res.get("reason", "")}

    H = match_res["H"]
    if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3):
        return None, {"reason": "invalid_H"}

    # 2) warp + safe crop
    warped_q, warped_mask = warp_query_to_bank(q_bgr, H.astype(np.float64), r_bgr.shape[:2])
    q_crop, r_crop, mask_crop, bbox = crop_common_safe_region(
        warped_q, r_bgr, warped_mask
    )
    if q_crop is None:
        return None, {"reason": "crop_fail"}

    # 3) CNN feature 추출
    q_feat, _ = backbone.extract_grid(q_crop, device)
    r_feat, _ = backbone.extract_grid(r_crop, device)

    grid_h, grid_w = q_feat.shape[1], q_feat.shape[2]
    valid_mask = make_patch_valid_mask(mask_crop, grid_h, grid_w)  # (H, W) bool

    if int(valid_mask.sum()) < 10:
        return None, {"reason": "too_few_valid_patches",
                      "valid_count": int(valid_mask.sum())}

    # 4) local radius search dist map
    dist_map = compute_dist_map_local_search(
        q_feat, r_feat,
        valid_mask=valid_mask,
        radius=radius,
    )

    # 5) compound score (connected component excess)
    result = compute_compound_score(
        dist_map, valid_mask,
        top_p=top_p, alpha=alpha, min_cut=min_cut,
        singleton_weight=singleton_weight,
        component_min_area=component_min_area,
    )

    debug = {
        "reason"    : "ok",
        "score"     : result["score"],
        "area"      : result["area"],
        "peak"      : result["peak"],
        "mean"      : result["mean"],
        "cut"       : result["cut"],
        "inliers"   : int(match_res.get("inliers", 0)),
        "inlier_ratio": float(match_res.get("inlier_ratio", 0.0)),
        "bbox"      : [int(v) for v in bbox] if bbox is not None else None,
        "q_crop"    : q_crop,            # 시각화용
        "r_crop"    : r_crop,
        "dist_map"  : dist_map,          # raw 히트맵
        "valid_mask": valid_mask,
        "bin_map"        : result["bin_map"],         # cut 이상 전체 hot-zone
        "best_comp_mask" : result["best_comp_mask"],  # 최고 score 1개 component
        "all_comp_scores": result["all_comp_scores"], # [component-level] 전체 목록
    }
    return result["score"], debug


# =============================================================================
# ④ 캘리브레이션 모드: bank 이미지 전체를 정상으로 간주하여 score 모집
# =============================================================================

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
) -> dict:
    """
    1-pass:
      th_calib 각 이미지에 대해 best ref 1개 선택
      -> best ref의 component score 분포 수집
      -> compound threshold 계산

    2-pass:
      th_calib 각 이미지에 대해 다시 best ref 1개 선택
      -> top-K component proposal 생성
      -> verifier score 계산
      -> image-level final_score = mean(top2 verifier)
      -> final threshold 계산
    """
    random.seed(seed)

    pcfg = cfg.get("patchcore", {})
    top_p    = float(pcfg.get("top_p", 0.05))
    alpha    = float(pcfg.get("alpha", 0.6))
    min_cut  = float(pcfg.get("min_cut", 0.20))
    sw       = float(pcfg.get("singleton_weight", 0.25))
    cma      = int(pcfg.get("component_min_area", 2))

    th_paths   = list_images(th_calib_dir)
    bank_paths = list_images(bank_dir)

    if len(th_paths) == 0:
        raise RuntimeError(f"th_calib 이미지가 없습니다: {th_calib_dir}")
    if len(bank_paths) == 0:
        raise RuntimeError(f"bank 이미지가 없습니다: {bank_dir}")

    random.shuffle(th_paths)
    use_th = th_paths[:calib_max_imgs]

    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 1-pass: compound threshold calibration
    # =========================================================
    cc_scores = []
    cc_pair_log = []

    for th_idx, q_path in enumerate(use_th):
        q_bgr = cv2.imread(str(q_path))
        if q_bgr is None:
            print(f"  [CALIB-CC-SKIP] 읽기 실패: {q_path.name}")
            continue

        # preselect
        if dino_model is not None and dino_tfm is not None:
            bank_candidates = dino_preselect(
                q_bgr, bank_dino, dino_model, dino_tfm, device, top_m=dino_top_m
            )
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

        cand_scores = []
        cand_infos  = []

        for r_path in bank_candidates:
            r_bgr = cv2.imread(str(r_path))
            if r_bgr is None:
                continue

            score, debug = score_one_pair(
                q_bgr, r_bgr, sg, cc_backbone, device,
                radius=radius,
                top_p=top_p,
                alpha=alpha,
                min_cut=min_cut,
                singleton_weight=sw,
                component_min_area=cma,
            )

            if score is None:
                print(f"  [CALIB-CC-SKIP] {q_path.name} ↔ {r_path.name}: {debug.get('reason')}")
                continue

            cand_scores.append(float(score))
            cand_infos.append({
                "r": r_path.name,
                "score": float(score),
                "all_comp": debug.get("all_comp_scores", []),
                "debug": debug,
            })

        if len(cand_scores) == 0:
            print(f"  [CALIB-CC-SKIP] {q_path.name}: 모든 bank 후보 실패")
            continue

        # best ref 선택
        best_idx  = int(np.argmin(cand_scores))
        best_info = cand_infos[best_idx]
        best_score = float(cand_scores[best_idx])

        # best ref의 모든 component score 수집
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
    cc_calib = calibrate_threshold(
        cc_scores_arr,
        method=calib_method,
        k=cc_k,
        trim_outliers=True,
        trim_k=3.5,
    )
    compound_thr = float(cc_calib["threshold"])

    # =========================================================
    # 2-pass: final threshold calibration (top2-mean)
    # =========================================================
    calib_final_scores = []
    calib_final_log = []

    for th_idx, q_path in enumerate(use_th):
        q_bgr = cv2.imread(str(q_path))
        if q_bgr is None:
            print(f"  [CALIB-FINAL-SKIP] 읽기 실패: {q_path.name}")
            continue

        # preselect (1-pass와 동일)
        if dino_model is not None and dino_tfm is not None:
            bank_candidates = dino_preselect(
                q_bgr, bank_dino, dino_model, dino_tfm, device, top_m=dino_top_m
            )
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

        cand_scores = []
        cand_infos  = []

        for r_path in bank_candidates:
            r_bgr = cv2.imread(str(r_path))
            if r_bgr is None:
                continue

            score, debug = score_one_pair(
                q_bgr, r_bgr, sg, cc_backbone, device,
                radius=radius,
                top_p=top_p,
                alpha=alpha,
                min_cut=min_cut,
                singleton_weight=sw,
                component_min_area=cma,
            )

            if score is None:
                continue

            cand_scores.append(float(score))
            cand_infos.append({
                "r_path": r_path,
                "score": float(score),
                "debug": debug,
            })

        if len(cand_scores) == 0:
            print(f"  [CALIB-FINAL-SKIP] {q_path.name}: 모든 bank 후보 실패")
            continue

        # best ref 선택
        best_idx   = int(np.argmin(cand_scores))
        best_score = float(cand_scores[best_idx])
        best_info  = cand_infos[best_idx]
        best_debug = best_info["debug"]

        all_comps = best_debug.get("all_comp_scores", [])
        if len(all_comps) == 0:
            print(f"  [CALIB-FINAL-SKIP] {q_path.name}: component 없음")
            continue

        # proposal top-K
        topk_comps = sorted(
            all_comps,
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True
        )[:proposal_top_k]

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
        final_score = float(max(bbox_scores_sorted)) if len(bbox_scores_sorted) > 0 else 0.0

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
            f"→ best={best_info['r_path'].name} "
            f"compound={best_score:.4f} "
            f"bbox_n={len(bbox_scores_this_img)} "
            f"final={final_score:.4f}"
        )

    if len(calib_final_scores) >= 5:
        final_arr = np.array(calib_final_scores, dtype=np.float32)
        final_calib = calibrate_threshold(
            final_arr,
            method=calib_method,
            k=final_k,
            trim_outliers=False,
            trim_k=3.5,
        )
        final_threshold = float(final_calib["threshold"])
    else:
        final_threshold = 0.49
        final_calib = None
        print("[CALIB WARN] final score 샘플 부족, 폴백 0.49 사용")

    from datetime import datetime
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    thr_json = {
        "plc_idx": plc_idx,
        "repr_mode": "compound_cc_topkmean_verifier",
        "method": calib_method,

        # compound 단계
        "cc_k": float(cc_k),
        "threshold": float(compound_thr),
        "num_th": int(len(cc_scores)),

        # final 단계
        "final_k": float(final_k),
        "final_threshold": float(final_threshold),
        "verifier_threshold": float(final_threshold),  # 하위 호환용 유지
        "num_final": int(len(calib_final_scores)),

        "proposal_top_k": int(proposal_top_k),
        "created_at": created_at,

        # patchcore config snapshot
        "radius": radius,
        "top_p": top_p,
        "alpha": alpha,
        "min_cut": min_cut,
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

    thr_path = out_dir / "threshold.json"
    thr_path.write_text(
        json.dumps(thr_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / "calib_pair_log.json").write_text(
        json.dumps(cc_pair_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / "calib_final_log.json").write_text(
        json.dumps(calib_final_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[CALIB] threshold.json 저장 완료: {thr_path}")
    print(f"[CALIB] compound_thr={compound_thr:.4f}, num_th={len(cc_scores)}")
    print(f"[CALIB] final_thr={final_threshold:.4f}, num_final={len(calib_final_scores)}")

    return thr_json

# =========== bbox 추론

def patch_mask_to_bbox(comp_mask: np.ndarray):
    """
    comp_mask: (Hp, Wp) bool
    return: (y0, x0, y1, x1)  # y1, x1은 exclusive
    """
    ys, xs = np.where(comp_mask)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (y0, x0, y1, x1)

def expand_patch_bbox(bbox, Hp, Wp, patch_margin=1):
    """
    bbox: (y0, x0, y1, x1) on patch grid
    """
    y0, x0, y1, x1 = bbox
    y0 = max(0, y0 - patch_margin)
    x0 = max(0, x0 - patch_margin)
    y1 = min(Hp, y1 + patch_margin)
    x1 = min(Wp, x1 + patch_margin)
    return (y0, x0, y1, x1)


def patch_bbox_to_image_bbox(
    patch_bbox,
    img_h: int,
    img_w: int,
    grid_h: int,
    grid_w: int,
    crop_margin_ratio: float = 0.20,
    min_crop_size: int = 96,
):
    """
    patch grid bbox -> image pixel bbox
    return: (y0, x0, y1, x1)  # exclusive
    """
    py0, px0, py1, px1 = patch_bbox

    sy = img_h / float(grid_h)
    sx = img_w / float(grid_w)

    y0 = int(np.floor(py0 * sy))
    x0 = int(np.floor(px0 * sx))
    y1 = int(np.ceil(py1 * sy))
    x1 = int(np.ceil(px1 * sx))

    h = y1 - y0
    w = x1 - x0

    my = int(round(h * crop_margin_ratio))
    mx = int(round(w * crop_margin_ratio))

    y0 = max(0, y0 - my)
    x0 = max(0, x0 - mx)
    y1 = min(img_h, y1 + my)
    x1 = min(img_w, x1 + mx)

    # 최소 crop 크기 보장
    h = y1 - y0
    w = x1 - x0

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
    """
    threshold 초과 component들을 verifier용 bbox region으로 변환.
    아직 모델 inference는 하지 않고, crop 후보만 만든다.

    return: List[dict]
    [
      {
        "score": float,
        "area": int,
        "peak": float,
        "mean": float,
        "patch_bbox": (py0, px0, py1, px1),
        "img_bbox": (y0, x0, y1, x1),
        "q_region": np.ndarray,
        "r_region": np.ndarray,
        "mask_region": np.ndarray,
      },
      ...
    ]
    """
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
            pbbox,
            img_h=img_h,
            img_w=img_w,
            grid_h=grid_h,
            grid_w=grid_w,
            crop_margin_ratio=crop_margin_ratio,
            min_crop_size=min_crop_size,
        )

        y0, x0, y1, x1 = ibbox
        if (y1 - y0) <= 0 or (x1 - x0) <= 0:
            continue

        q_region = q_crop[y0:y1, x0:x1].copy()
        r_region = r_crop[y0:y1, x0:x1].copy()

        # valid mask도 image size로 맞춰서 같은 bbox 잘라두면 나중 verifier에서 유용함
        valid_rs = cv2.resize(
            valid_mask.astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
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

    # 높은 component score 순으로 정렬
    regions = sorted(regions, key=lambda x: x["score"], reverse=True)
    return regions

# =============bbox local resnet
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


def bgr_to_pil_rgb(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


@torch.no_grad()
def compute_local_search_dist_map_from_feats(
    q_feat: torch.Tensor,   # [C,H,W], normalized
    r_feat: torch.Tensor,   # [C,H,W], normalized
    radius: int = 1,
):
    C, H, W = q_feat.shape
    dist_map = torch.zeros((H, W), device=q_feat.device, dtype=torch.float32)

    for y in range(H):
        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)

        for x in range(W):
            x0 = max(0, x - radius)
            x1 = min(W, x + radius + 1)

            qv = q_feat[:, y, x]                     # [C]
            rv = r_feat[:, y0:y1, x0:x1]            # [C,hh,ww]
            sims = (rv * qv[:, None, None]).sum(dim=0)   # [hh,ww]
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
    """
    dist_map에서 상위 top_p 비율 위치만 True인 mask 반환
    """
    flat = dist_map.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(dist_map, dtype=bool), 0.0, 0

    k = max(1, int(np.ceil(flat.size * top_p)))
    thr_top = float(np.sort(flat)[-k])
    top_mask = dist_map >= thr_top
    return top_mask, thr_top, k


@torch.no_grad()
def verify_bbox_with_local_search(
    q_region: np.ndarray,
    r_region: np.ndarray,
    backbone,
    device,
    radius: int = 1,
    top_p: float = 0.10,
):
    """
    q_region, r_region: BGR uint8 crop
    return:
    {
        "score": float,
        "dist_map": np.ndarray,
        "feat_hw": (Hf, Wf),
        "top_p_mask": np.ndarray(bool),
        "top_p_thr": float,
        "top_k": int,
        "top_p": float,
    }
    """

    q_feat, (Hf, Wf) = backbone.extract_grid(q_region, device)
    r_feat, _ = backbone.extract_grid(r_region, device)

    dist_map_t = compute_local_search_dist_map_from_feats(
        q_feat=q_feat,
        r_feat=r_feat,
        radius=radius,
    )
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
        cv2.putText(out, str(line), (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(line), (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def colorize_patch_mask(mask_patch, out_h, out_w, color=(0, 0, 255), alpha=0.55, base=None):
    """
    patch-grid bool/uint8 mask -> image 크기로 nearest resize 후 overlay
    color: BGR
    """
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
    """
    q_crop / r_crop / valid overlay를 한 장에 저장
    """
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
    """
    compound 결과를 한 번에 보기:
    - 전체 hot-zone
    - best component
    - flagged bbox
    """
    h, w = q_crop.shape[:2]

    # 전체 hot-zone
    hot_all, _ = colorize_patch_mask(bin_map, h, w, color=(0, 165, 255), alpha=0.55, base=q_crop)

    # best component
    best_only, _ = colorize_patch_mask(best_comp_mask, h, w, color=(0, 0, 255), alpha=0.60, base=q_crop)

    # bbox overlay
    bbox_vis = q_crop.copy()
    for i, reg in enumerate(flagged_regions):
        y0, x0, y1, x1 = reg["img_bbox"]
        cv2.rectangle(bbox_vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        txt = f"#{i} s={reg['score']:.3f} a={reg['area']}"
        cv2.putText(bbox_vis, txt, (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    top = np.hstack([q_crop, hot_all])
    bot = np.hstack([best_only, bbox_vis])
    panel = np.vstack([top, bot])

    panel = draw_text_box(
        panel,
        [
            "TL: q_crop",
            "TR: compound all hot-zone",
            "BL: best component",
            "BR: flagged component bbox",
        ],
        org=(10, 20),
    )
    cv2.imwrite(str(case_dir / "stage3_component_summary.png"), panel)


def make_dist_overlay(base_bgr, dist_map, color_map=cv2.COLORMAP_JET, alpha=0.45,
                      abs_min=0.0, abs_max=1.0):
    """
    dist_map: float [Hf, Wf] -> image size overlay
    절대 기준 시각화: bbox별 min-max 정규화 X
    """
    h, w = base_bgr.shape[:2]
    d = dist_map.astype(np.float32)
    if d.size == 0:
        return base_bgr.copy()

    # 절대 기준 clip
    d_vis = np.clip((d - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)

    heat = (d_vis * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, color_map)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)

    out = cv2.addWeighted(base_bgr, 1 - alpha, heat, alpha, 0)
    return out


def save_flagged_region_visuals(case_dir, verifier_results):
    """
    bbox별 q/r/verifier heatmap + top-p mask 저장
    """
    for i, reg in enumerate(verifier_results):
        sub_dir = case_dir / f"bbox_{i:02d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        q_region = reg["q_region"]
        r_region = reg["r_region"]
        mask_region = reg["mask_region"]
        vscore = reg["verifier_score"]
        dist_map = reg["verifier_dist_map"]

        top_p_mask = reg.get("verifier_top_p_mask", None)
        top_p_thr  = reg.get("verifier_top_p_thr", None)
        top_k      = reg.get("verifier_top_k", None)
        top_p      = reg.get("verifier_top_p", None)

        cv2.imwrite(str(sub_dir / "q_region.png"), q_region)
        cv2.imwrite(str(sub_dir / "r_region.png"), r_region)
        cv2.imwrite(str(sub_dir / "mask_region.png"), (mask_region.astype(np.uint8) * 255))

        # 절대 기준 verifier heat overlay
        q_overlay = make_dist_overlay(q_region, dist_map)
        r_overlay = make_dist_overlay(r_region, dist_map)

        q_overlay = draw_text_box(
            q_overlay,
            [f"bbox verifier score = {vscore:.4f}",
             f"img_bbox = {reg['img_bbox']}",
             f"patch_bbox = {reg['patch_bbox']}"],
            org=(8, 18),
        )
        r_overlay = draw_text_box(
            r_overlay,
            [f"component score = {reg['score']:.4f}",
             f"area = {reg['area']}",
             f"peak = {reg['peak']:.4f}"],
            org=(8, 18),
        )

        pair = np.hstack([q_overlay, r_overlay])
        cv2.imwrite(str(sub_dir / "verifier_pair.png"), pair)

        # raw abs heat
        d = dist_map.astype(np.float32)
        d_vis = np.clip(d, 0.0, 1.0)
        heat = (d_vis * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(str(sub_dir / "verifier_heat.png"), heat)
        np.save(sub_dir / "verifier_dist_map.npy", dist_map)

        # -----------------------------
        # top-p mask 저장
        # -----------------------------
        if top_p_mask is not None:
            h, w = q_region.shape[:2]

            top_mask_rs = cv2.resize(
                top_p_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            cv2.imwrite(
                str(sub_dir / "verifier_top_p_mask.png"),
                (top_mask_rs.astype(np.uint8) * 255)
            )

            cyan = np.zeros_like(q_region)
            cyan[:] = (255, 255, 0)  # BGR cyan

            q_top = q_region.copy()
            r_top = r_region.copy()

            q_blend = cv2.addWeighted(q_region, 0.35, cyan, 0.65, 0)
            r_blend = cv2.addWeighted(r_region, 0.35, cyan, 0.65, 0)

            q_top[top_mask_rs] = q_blend[top_mask_rs]
            r_top[top_mask_rs] = r_blend[top_mask_rs]

            q_top = draw_text_box(
                q_top,
                [f"top_p = {top_p}",
                 f"top_k = {top_k}",
                 f"top_thr = {top_p_thr:.4f}" if top_p_thr is not None else "top_thr = NA"],
                org=(8, 18),
            )

            top_pair = np.hstack([q_top, r_top])
            cv2.imwrite(str(sub_dir / "verifier_top_p_pair.png"), top_pair)


def save_verifier_summary(case_dir, q_crop, verifier_results):
    """
    q_crop 위에 bbox별 verifier score를 한 장에 요약
    """
    vis = q_crop.copy()
    for i, reg in enumerate(verifier_results):
        y0, x0, y1, x1 = reg["img_bbox"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
        txt = f"#{i} comp={reg['score']:.3f} ver={reg['verifier_score']:.3f}"
        cv2.putText(vis, txt, (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(case_dir / "stage5_verifier_summary.png"), vis)

# local heatmap > 원본
def save_verified_bbox_overlay(
    case_dir,
    q_crop,
    verified_regions,
    alpha=0.45,
    abs_min=0.0,
    abs_max=1.0,
    vis_thr_abs=0.30,   # 절대 임계값
):
    """
    절대 기준 heatmap overlay
    - bbox별 min-max 정규화 안 함
    - verifier_dist_map의 절대값을 그대로 사용
    - vis_thr_abs 이상만 overlay
    """
    vis = q_crop.copy()

    for reg in verified_regions:
        y0, x0, y1, x1 = reg["img_bbox"]
        dist_map = reg["verifier_dist_map"].astype(np.float32)

        if dist_map.size == 0 or (y1 - y0) <= 0 or (x1 - x0) <= 0:
            continue

        # 1) 절대값 기준 clip + normalize
        d_vis = np.clip((dist_map - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)

        # 2) 절대 threshold 이상만 표시
        thr_norm = np.clip((vis_thr_abs - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)
        mask = d_vis >= thr_norm
        if mask.sum() == 0:
            continue

        # 3) heatmap 생성
        heat = (d_vis * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)

        mask_rs = cv2.resize(
            mask.astype(np.uint8),
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 4) overlay
        roi = vis[y0:y1, x0:x1].copy()
        blended = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
        roi[mask_rs] = blended[mask_rs]
        vis[y0:y1, x0:x1] = roi

        # 5) 점수 텍스트
        cv2.putText(
            vis,
            f"{reg['verifier_score']:.3f}",
            (x0, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(case_dir / "stage6_verified_bbox_overlay_abs.png"), vis)


def save_verified_bbox_top_p_overlay(case_dir, q_crop, verified_regions, alpha=0.65):
    """
    최종 anomaly로 승인된 bbox들의 top-p patch mask를
    원래 q_crop 위에 오버레이해서 저장
    """
    vis = q_crop.copy()

    for i, reg in enumerate(verified_regions):
        y0, x0, y1, x1 = reg["img_bbox"]
        top_p_mask = reg.get("verifier_top_p_mask", None)

        if top_p_mask is None:
            continue
        if (y1 - y0) <= 0 or (x1 - x0) <= 0:
            continue

        # patch mask -> bbox image 크기로 resize
        mask_rs = cv2.resize(
            top_p_mask.astype(np.uint8),
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        roi = vis[y0:y1, x0:x1].copy()

        # cyan overlay
        color = np.zeros_like(roi)
        color[:] = (255, 255, 0)  # BGR cyan

        blended = cv2.addWeighted(roi, 1 - alpha, color, alpha, 0)
        roi[mask_rs] = blended[mask_rs]
        vis[y0:y1, x0:x1] = roi

        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
        cv2.putText(
            vis,
            f"#{i} ver={reg['verifier_score']:.3f}",
            (x0, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(case_dir / "stage6_verified_bbox_top_p_overlay_rel.png"), vis)

# =============================================================================
# ⑤ 추론 모드: query 이미지를 bank와 비교, threshold와 비교하여 판정
# =============================================================================

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
    n_ref_candidates: int = 5,  # bank 중 몇 장과 매칭 시도할지
    seed: int = 0,
    # --- [3단계] SAM 파라미터 (None이면 SAM 스킵) ---
    sam_predictor=None,
    sam_dv_iou_thresh: float = 0.60,
    sam_top_p: float = 0.10,
    sam_abs_floor: float = 0.20,
    sam_peak_alpha: float = 0.50,
    sam_min_blob_peak: float = 0.30,
    sam_min_blob_mean: float = 0.25,
    # --- DINO preselection (None이면 랜덤 폴백) ---
    dino_model=None,
    dino_tfm=None,
    bank_dino: dict = None,
    dino_top_m: int = 5,
    proposal_top_k: int = 3,
) -> None:
    """
    query 이미지마다:
      bank 후보들과 매칭 -> best ref 선택(min compound score)
      -> top-K component proposal 생성
      -> verifier score 계산
      -> final_score = mean(max verifier)
      -> final_threshold와 비교하여 최종 판정
    """
    random.seed(seed)
    all_infer_results = []

    # threshold.json 로드
    thr_path = out_dir / "threshold.json"
    if not thr_path.exists():
        raise FileNotFoundError(
            f"threshold.json이 없습니다. --mode calib를 먼저 실행하세요: {thr_path}"
        )

    meta = json.loads(thr_path.read_text(encoding="utf-8"))

    # CC threshold
    thr = float(meta["threshold"])

    if meta.get("method") == "robust":
        k_str = meta.get("robust_k", meta.get("k", "NA"))
    elif meta.get("method") == "gaussian":
        k_str = meta.get("gaussian_k", meta.get("k", "NA"))
    else:
        k_str = meta.get("percentile", meta.get("k", "NA"))

    # final threshold: 이제 verifier_threshold를 "final_score threshold"로 사용
    final_thr = float(meta.get("verifier_threshold", meta.get("final_threshold", 0.49)))
    if ("verifier_threshold" not in meta) and ("final_threshold" not in meta):
        print("[INFER WARN] threshold.json에 final/verifier threshold 없음. 폴백 0.49 사용")

    print(f"\n[INFER] compound_thr={thr:.4f} (method={meta.get('method')}, k={k_str})")
    print(f"[INFER] final_thr={final_thr:.4f} (max verifier 기준)")

    pcfg = cfg.get("patchcore", {})
    top_p   = float(pcfg.get("top_p", 0.05))
    alpha   = float(pcfg.get("alpha", 0.6))
    min_cut = float(pcfg.get("min_cut", 0.20))
    sw      = float(pcfg.get("singleton_weight", 0.25))
    cma     = int(pcfg.get("component_min_area", 2))

    ref_paths   = list_images(bank_dir)
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

        # -------------------------------------------------
        # 1) bank 후보 선택
        # -------------------------------------------------
        if dino_model is not None and bank_dino is not None:
            candidates_to_try = dino_preselect(
                q_bgr, bank_dino, dino_model, dino_tfm, device, top_m=dino_top_m
            )
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

        cand_scores = []
        cand_debugs = []

        # -------------------------------------------------
        # 2) 후보 ref 각각과 score 계산
        # -------------------------------------------------
        for r_path in candidates_to_try:
            r_bgr = cv2.imread(str(r_path))
            if r_bgr is None:
                continue

            score, debug = score_one_pair(
                q_bgr, r_bgr, sg, cc_backbone, device,
                radius=radius,
                top_p=top_p,
                alpha=alpha,
                min_cut=min_cut,
                singleton_weight=sw,
                component_min_area=cma,
            )

            if score is None:
                print(f"  [SKIP] {r_path.name}: {debug.get('reason')}")
                continue

            cand_scores.append(float(score))
            cand_debugs.append({
                "r_path": r_path,
                "score": float(score),
                "debug": debug,
            })

        if len(cand_scores) == 0:
            print("  [FAIL] 모든 ref 후보 매칭 실패")
            continue

        # -------------------------------------------------
        # 3) best ref 선택 (compound score 최소)
        # -------------------------------------------------
        best_idx   = int(np.argmin(cand_scores))
        best_score = float(cand_scores[best_idx])
        best_info  = cand_debugs[best_idx]
        best_debug = best_info["debug"]

        # -------------------------------------------------
        # 4) component proposal top-K
        # -------------------------------------------------
        all_comps = best_debug.get("all_comp_scores", [])
        topk_comps = sorted(
            all_comps,
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True
        )[:proposal_top_k]

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

        # -------------------------------------------------
        # 5) bbox verifier
        # -------------------------------------------------
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

        # -------------------------------------------------
        # 6) 최종 점수 집계: mean(top2)
        # -------------------------------------------------
        verifier_scores_sorted = sorted(
            [r["verifier_score"] for r in verifier_results],
            reverse=True
        )

        final_score = float(max(verifier_scores_sorted)) if len(verifier_scores_sorted) > 0 else 0.0

        is_anomaly = bool(final_score > final_thr)

        # 참고용
        cc_is_anomaly = bool(best_score > thr)
        best_verifier_score = max(verifier_scores_sorted, default=0.0)

        # 시각화용 verified_regions
        verified_regions = [
            r for r in verifier_results
            if r["verifier_score"] > final_thr
        ]

        # -------------------------------------------------
        # 7) 단계별 시각화 저장
        # -------------------------------------------------
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

        save_component_summary(
            case_dir,
            best_debug["q_crop"],
            best_debug["bin_map"],
            best_debug["best_comp_mask"],
            flagged_regions,
        )

        save_flagged_region_visuals(case_dir, verifier_results)
        save_verifier_summary(case_dir, best_debug["q_crop"], verifier_results)
        save_verified_bbox_top_p_overlay(
            case_dir,
            best_debug["q_crop"],
            verifier_results,
            alpha=0.65,
        )
        save_verified_bbox_overlay(
            case_dir,
            best_debug["q_crop"],
            verified_regions,
            alpha=0.45,
        )

        # -------------------------------------------------
        # 8) SAM 후처리 (원하면 유지, 없으면 그대로)
        # -------------------------------------------------
        sam_meta = {
            "sam_used": False,
            "sam_ok": None,
            "sam_dv_iou": None,
        }

        # 여기서는 기존 구조 최대한 유지하려고,
        # SAM predictor 없으면 그대로 is_anomaly 사용
        # predictor 있으면 네 기존 로직 있던 곳에 붙이면 됨
        # (현재 답변에서는 mean(top2) 구조가 핵심이라 SAM은 보수적으로 둠)

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

        # -------------------------------------------------
        # 9) result.json 저장
        # -------------------------------------------------
        result_meta = {
            "query": str(q_path),
            "query_name": q_path.name,
            "best_ref": str(best_info["r_path"]),
            "best_ref_name": best_info["r_path"].name,

            "label": label_str,
            "is_anomaly": bool(is_anomaly),

            # compound 단계
            "score": round(best_score, 6),            # 기존 호환용
            "compound_score": round(best_score, 6),
            "threshold": round(thr, 6),               # 기존 호환용
            "compound_threshold": round(thr, 6),
            "cc_is_anomaly": bool(cc_is_anomaly),

            # final 단계
            "final_score": round(final_score, 6),
            "final_threshold": round(final_thr, 6),
            "best_verifier_score": round(best_verifier_score, 6),

            "proposal_top_k": int(proposal_top_k),
            "num_flagged_components": int(n_flagged),
            "num_flagged_regions": int(n_regions),
            "num_verified_regions": int(n_verified),

            "flagged_comp_scores": [round(float(c["score"]), 6) for c in topk_comps],
            "all_cand_scores": [round(float(s), 6) for s in cand_scores],
        }

        # 하위 호환용
        result_meta["verifier_threshold"] = round(final_thr, 6)

        result_meta["flagged_regions"] = [
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
        ]

        result_meta["verified_regions"] = [
            {
                "idx": i,
                "final_bbox_score": round(float(r["verifier_score"]), 6),
                "component_score": round(float(r["score"]), 6),
                "verifier_score": round(float(r["verifier_score"]), 6),
                "img_bbox": list(r["img_bbox"]),
                "patch_bbox": list(r["patch_bbox"]),
            }
            for i, r in enumerate(verified_regions)
        ]

        result_meta.update(sam_meta)

        (case_dir / "result.json").write_text(
            json.dumps(result_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_infer_results.append(result_meta)

        # -------------------------------------------------
        # 10) 평가 집계
        # -------------------------------------------------
        eval_y_true.append(1 if q_path.name.startswith("abnormal_") else 0)
        eval_y_pred.append(1 if is_anomaly else 0)

    # -------------------------------------------------
    # 11) 전체 결과 저장
    # -------------------------------------------------
    if len(eval_y_true) > 0:
        _print_eval_report(eval_y_true, eval_y_pred, out_dir)

    (out_dir / "infer_all_results.json").write_text(
        json.dumps(all_infer_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    _save_summary_gallery(out_dir)
    plot_compound_scores(out_dir)
    plot_bbox_scores(out_dir)



def _save_summary_gallery(out_dir: Path):
    """
    out_dir 내 각 query 케이스 폴더에서 result.json 읽기
    → 최종 오버레이 이미지 선택 (SAM > compound_best > q_crop)
    → 레이블 색상 바 + score 텍스트 삽입
    → out_dir / '모음' / {ANOMALY|NORMAL}_{stem}.png 저장
    """
    gallery_dir = out_dir / "모음"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for case_dir in sorted(out_dir.iterdir()):
        # 각 case_dir는 query 케이스 폴더 (result.json이 있는 것만)
        result_path = case_dir / "result.json"
        if not result_path.exists():
            continue

        meta      = json.loads(result_path.read_text(encoding="utf-8"))
        label     = meta.get("label", "ANOMALY" if meta["is_anomaly"] else "NORMAL")
        score     = meta.get("score", 0.0)
        threshold = meta.get("threshold", 0.0)
        sam_used  = meta.get("sam_used", False)

        # 최종 오버레이 이미지 선택 우선순위:
        # ① SAM 검증 결과 (sam_overlay_q.png)  ← SAM 사용한 경우
        # ② compound best component (compound_best_component.png)  ← 두 번째
        # ③ 원본 query crop (q_crop.png)  ← fallback
        overlay_path = None
        candidates = ["stage6_verified_bbox_overlay_rel.png", "compound_best_component.png", "q_crop.png"]
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

        # ref 이미지 로드 (왼쪽 패널 = 정상 기준)
        r_crop_path = case_dir / "r_crop.png"
        ref_img = cv2.imread(str(r_crop_path)) if r_crop_path.exists() else None

        # 동일 높이로 맞추기 (height 기준 리사이즈)
        h_o, w_o = overlay_img.shape[:2]
        if ref_img is not None:
            h_r, w_r = ref_img.shape[:2]
            # overlay와 ref 높이를 h_o로 통일
            if h_r != h_o:
                ref_img = cv2.resize(ref_img, (int(w_r * h_o / h_r), h_o))
            # 구분선 (3px 흰색 세로선)
            divider = np.ones((h_o, 3, 3), dtype=np.uint8) * 200
            side_by_side = np.hstack([ref_img, divider, overlay_img])
        else:
            side_by_side = overlay_img  # ref 없으면 overlay만

        # 상단 레이블 바 생성 (36px)
        h_s, w_s = side_by_side.shape[:2]
        bar = np.zeros((36, w_s, 3), dtype=np.uint8)

        # 레이블별 색상 (BGR)
        if "ANOMALY" in label:
            bar[:] = (0, 30, 200)    # 빨강 계열
        elif "SAM_REJ" in label:
            bar[:] = (180, 80, 0)    # 파랑 계열 (SAM이 기각)
        else:
            bar[:] = (0, 140, 0)     # 초록 계열 (NORMAL)

        # 소스 이미지명 (SAM 사용 여부 표시)
        src_tag = "[SAM]" if sam_used else "[CC]"
        txt = (f"{label}  {src_tag}  "
               f"score={score:.5f} / thr={threshold:.5f}  "
               f"← 정상(ref) | 쿼리+mask →   [{case_dir.name[:20]}]")
        cv2.putText(bar, txt, (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        annotated = np.vstack([bar, side_by_side])  # 바 + side-by-side

        # 파일명: ANOMALY_stem.png or NORMAL_stem.png
        prefix   = "ANOMALY" if meta["is_anomaly"] else "NORMAL"
        out_name = f"{prefix}_{case_dir.name}.png"
        cv2.imwrite(str(gallery_dir / out_name), annotated)
        saved += 1

        # sam_compare.png 도 모음 폴더에 복사 (넘겨가며 보기용)
        sam_cmp = case_dir / "sam_compare.png"
        if sam_cmp.exists():
            import shutil
            shutil.copy(str(sam_cmp),
                        str(gallery_dir / f"SAM_{prefix}_{case_dir.name}.png"))

    print(f"\n[모음] 갤러리 {saved}장 저장 완료: {gallery_dir}")


def _print_eval_report(y_true, y_pred, out_dir: Path):
    """Confusion matrix + 지표 출력 및 파일 저장"""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    acc       = (tp + tn) / len(y_t)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0

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


def save_outputs(save_dir, q_crop, r_crop, dist_map, valid_mask):
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) dist 기반 absolute heatmap
    dist_abs = np.clip(dist_map, 0.0, 1.0)
    heat = (dist_abs * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # 2) invalid -> gray on patch heatmap
    invalid = ~valid_mask.astype(bool)
    heat[invalid] = (128, 128, 128)

    # 3) resize to image size
    heat_rs = cv2.resize(
        heat,
        (q_crop.shape[1], q_crop.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    valid_rs = cv2.resize(
        valid_mask.astype(np.uint8),
        (q_crop.shape[1], q_crop.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # 4) overlay only on valid region
    overlay_q = q_crop.copy()
    overlay_r = r_crop.copy()

    alpha = 0.4

    overlay_q[valid_rs] = cv2.addWeighted(
        q_crop[valid_rs], 1 - alpha,
        heat_rs[valid_rs], alpha,
        0
    )

    overlay_r[valid_rs] = cv2.addWeighted(
        r_crop[valid_rs], 1 - alpha,
        heat_rs[valid_rs], alpha,
        0
    )

    # 5) invalid region gray
    overlay_q[~valid_rs] = (128, 128, 128)
    overlay_r[~valid_rs] = (128, 128, 128)

    # 6) save
    np.save(save_dir / "dist_map.npy", dist_map)
    np.save(save_dir / "valid_mask.npy", valid_mask.astype(np.uint8))

    cv2.imwrite(str(save_dir / "heat_abs.png"), heat)
    cv2.imwrite(str(save_dir / "overlay_q.png"), overlay_q)
    cv2.imwrite(str(save_dir / "r_crop.png"), r_crop)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--place",  required=True, help="실험 장소명 (recv/{place}/bank, query 구조)")
    parser.add_argument("--mode",   default="vis",
                        choices=["calib", "infer", "vis"],
                        help="calib: threshold 캘리브레이션 | infer: 이상감지 추론 | vis: 기존 시각화")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--radius", type=int, default=1)
    # 캘리브레이션 전용 인자
    parser.add_argument(
        "--calib_method",
        default=None,
        choices=["robust", "gaussian", "percentile"],
        help="threshold 계산 방식. 미지정 시 config.calib.method 사용"
    )

    parser.add_argument(
        "--calib_k",
        type=float,
        default=None,
        help="k 값. 미지정 시 config.calib.* 또는 config.calib.k 사용"
    )
    parser.add_argument("--cc_k", type=float, default=None,
                    help="1차 component gate calibration k")
    parser.add_argument("--final_k", type=float, default=None,
                        help="2차 verifier threshold calibration k")

    
    parser.add_argument("--calib_max_imgs", type=int, default=30,
                        help="캘리브레이션 시 th_calib 이미지 최대 수 (Fix1 best-match 방식)")
    parser.add_argument("--calib_n_ref",    type=int, default=5,
                        help="캘리브레이션 시 이미지당 bank 시도 수 (infer의 n_ref_candidates와 맞출 것)")
    # 추론 전용 인자
    parser.add_argument("--n_ref_candidates", type=int, default=5,
                        help="추론 시 bank에서 시도할 최대 ref 수")
    # SAM 관련 인자 (vis 모드에서만 사용)
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="SAM 체크포인트 경로 (예: sam_vit_b_01ec64.pth). 없으면 SAM 정제 스킵")
    parser.add_argument("--sam_model", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_top_p",     type=float, default=0.10,
                        help="dist map 상위 p%만 hot_zone으로 (예: 0.10 = 상위 10%)")
    parser.add_argument("--sam_abs_floor", type=float, default=0.20,
                        help="hot_zone 최소 절대 threshold")
    parser.add_argument("--sam_peak_alpha",type=float, default=0.50,
                        help="peak * alpha 하한으로 쓸 비율")
    parser.add_argument("--sam_min_blob_peak", type=float, default=0.30,
                        help="그림자/노이즈 기각용 최소 피크 거리")
    parser.add_argument("--sam_min_blob_mean", type=float, default=0.25,
                        help="그림자/노이즈 기각용 최소 평균 밀도 거리")
    parser.add_argument("--sam_dv_iou_thresh", type=float, default=0.60,
                        help="SAM BBox 교차 검증 IoU 임계값 (예: 0.60 넘으면 배경/조명 변화로 기각)")
    # DINO preselection 인자
    parser.add_argument("--dino_model", type=str, default=None,
                        help="DINO preselection 모델 (\"dinov2_vits14\" | \"dinov2_vitb14\"). "
                             "None이면 랜덤 bank 선택 (7기존 방식)")
    parser.add_argument("--dino_top_m", type=int, default=5,
                        help="DINO preselection 후보 수 (calib_n_ref 및 n_ref_candidates와 동일하게 위위 허용)")
    args = parser.parse_args()

    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg(ROOT)

    calib_cfg = cfg.get("calib", {})

    # method: CLI 우선, 없으면 config
    calib_method = args.calib_method or calib_cfg.get("method", "robust")

    # k: method별 우선값 사용, 없으면 공통 k fallback
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
    cc_k = float(
        args.cc_k if args.cc_k is not None
        else calib_cfg.get("cc_k", calib_k)
    )

    final_k = float(
        args.final_k if args.final_k is not None
        else calib_cfg.get("final_k", calib_k)
    )

    img_size  = int(cfg.get("embed", {}).get("img_size", 560))

    from backbone_wrapper import build_local_backbone

    cc_backbone = build_local_backbone(backbone_name="resnet18_layer3",img_size=img_size,)
    verifier_backbone = build_local_backbone(
    backbone_name="resnet18_layer3",   #
            img_size=224,
        )

    # DINO preselection 조건부 설정
    dino_model_inst = None
    dino_tfm_inst   = None
    if args.dino_model is not None:
        print(f"[DINO] 모델 로드 중: {args.dino_model}")
        dino_model_inst, _ = load_dino_model(model_name=args.dino_model, device=device)
        dino_tfm_inst = make_dino_transform(img_size=img_size)
        print(f"[DINO] 로드 완료")

    sg_cfg = SuperGlueMatchConfig(
        resize_long_side=640,
        weights="indoor",
        max_keypoints=1024,
        keypoint_threshold=0.003,
        match_threshold=0.2,
        sinkhorn_iterations=20,
    )
    sg = SuperGlueMatcher(sg_cfg, device=device)



    bank_dir     = Path(ROOT) / args.place / "bank"
    th_calib_dir = Path(ROOT) / args.place / "th_calib"  # 캘리브레이션용 정상 이미지
    query_dir    = Path(ROOT) / args.place / "query"
    out_dir      = Path(OUT_ROOT) / args.place
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # [모드 분기]
    # ----------------------------------------------------------------

    # --- calib 모드: threshold 캘리브레이션 ---
    if args.mode == "calib":
        print(f"\n[MODE] CALIBRATION  place={args.place}")

        # DINO bank 임베딩 사전 구성 (calib에서 preselection 사용 시)
        bank_dino_data = None
        if dino_model_inst is not None:
            bank_dino_data = build_dino_bank(
                bank_dir, dino_model_inst, dino_tfm_inst, device
            )

        run_calibration(
            bank_dir=bank_dir,
            th_calib_dir=th_calib_dir,
            out_dir=out_dir,
            sg=sg,
            cc_backbone = cc_backbone,
            verifier_backbone = verifier_backbone,
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
        )
        return

    # --- infer 모드: 이상감지 추론 (compound score + 선택적 SAM 검증) ---
    if args.mode == "infer":
        print(f"\n[MODE] INFERENCE  place={args.place}")

        # SAM 조건부 로드 (--sam_ckpt 있으면)
        infer_sam = None
        if args.sam_ckpt is not None:
            ckpt = Path(args.sam_ckpt)
            if not ckpt.exists():
                print(f"[WARN] SAM 체크포인트 없음: {ckpt}. SAM 스터파.")
            else:
                print(f"[INFO] SAM 로드: {ckpt}")
                infer_sam = load_sam_model(
                    checkpoint_path=str(ckpt),
                    model_type=args.sam_model,
                    device=device,
                )
                print("[INFO] SAM 로드 완료")

        # DINO bank 임베딩 (infer 전용)
        bank_dino_infer = None
        if dino_model_inst is not None:
            bank_dino_infer = build_dino_bank(
                bank_dir, dino_model_inst, dino_tfm_inst, device
            )

        run_inference(
            bank_dir=bank_dir,
            query_dir=query_dir,
            out_dir=out_dir,
            sg=sg,
            cc_backbone = cc_backbone,
            verifier_backbone = verifier_backbone,
            device=device,
            cfg=cfg,
            radius=args.radius,
            n_ref_candidates=args.n_ref_candidates,
            seed=args.seed,
            # SAM 파라미터 (None이면 SAM 스킵)
            sam_predictor=infer_sam,
            sam_dv_iou_thresh=args.sam_dv_iou_thresh,
            sam_top_p=args.sam_top_p,
            sam_abs_floor=args.sam_abs_floor,
            sam_peak_alpha=args.sam_peak_alpha,
            sam_min_blob_peak=args.sam_min_blob_peak,
            sam_min_blob_mean=args.sam_min_blob_mean,
            # DINO preselection (None이면 기존 랜덤 방식)
            dino_model=dino_model_inst,
            dino_tfm=dino_tfm_inst,
            bank_dino=bank_dino_infer,
            dino_top_m=args.dino_top_m,
        )
        return


if __name__ == "__main__":
    main()