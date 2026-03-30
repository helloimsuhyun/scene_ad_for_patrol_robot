# distance.py
# 1. threshold 캘리브레이션 : def calibrate_place(bank_root, plc_idx, model, device):
# 2. event단위 q 이미지에 대한 추론 : def infer_event(imgs_bgr: List[np.ndarray],plc_idx: str,bank_root,model,device)
# - suhyun
# 파라미터 설정은 config.py에서 여기 파일에 의해서만 의존함

# superpoint - superglue를 붙이기 위한 수정본

import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

from .banker import load_bank_by_place, BGR_to_RGB, rebuild_bank
from .dino_emb import make_embed, make_transform
from .cnn_emb import extract_grid_layers, make_aligned_local_transform
from .config import load_cfg

import cv2

from .warp_utils import (
    warp_query_to_bank,
    make_patch_valid_mask,
    crop_common_valid_region,
    warp_bank_to_query,
    crop_common_safe_region
)

# ----------------------------------------------------for debug

from scipy import ndimage
import math
import time
from contextlib import contextmanager


ENABLE_TIMING_LOG = True
DEBUG_VIS = True


# ----------------------------------------------------


# vis helper
def get_top_p_patch_info(repr_mode: str, debug: dict):
    if repr_mode == "global_patch_pool":
        pool_debug = debug["pool_debug"]
        return pool_debug["top_patch_idx"], pool_debug["top_vals"]

    elif repr_mode in {"global_patch", "global_patch_with_aligned"}:
        patch_topk = debug["patch_topk"]
        return patch_topk["top_patch_idx"], patch_topk["top_patch_vals"]

    else:
        return None, None


def _cuda_sync_if_needed(device):
    try:
        if device is not None and "cuda" in str(device) and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

@contextmanager
def _timer(stats: dict, key: str, device=None):
    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    yield
    _cuda_sync_if_needed(device)
    dt = time.perf_counter() - t0
    stats[key] = stats.get(key, 0.0) + dt


def _stats_to_float_dict(stats: dict) -> dict:
    return {str(k): float(v) for k, v in stats.items()}


def _print_timing(prefix: str, stats: dict):
    if not ENABLE_TIMING_LOG:
        return
    parts = [f"{k}={v:.4f}s" for k, v in stats.items()]
    print(f"[TIMING] {prefix} | " + " | ".join(parts))


def _merge_timing(dst: dict, src: Optional[dict]):
    if not isinstance(src, dict):
        return
    for k, v in src.items():
        try:
            dst[k] = dst.get(k, 0.0) + float(v)
        except Exception:
            pass
# ------------------------------------------------

def _to_vis_bgr_from_rgb_tensor(x: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W], normalized tensor
    return: uint8 BGR image
    """
    if x.dim() != 3 or x.shape[0] != 3:
        raise ValueError(f"expected [3,H,W], got {tuple(x.shape)}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)

    y = x.detach().float() * std + mean
    y = y.clamp(0.0, 1.0)
    y = (y * 255.0).byte().permute(1, 2, 0).cpu().numpy()  # HWC RGB
    y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    return y

# =========================================================================================== dist 계산 함수

def _infer_patch_grid(P: int):
    side = int(math.sqrt(P))
    if side * side != P:
        raise ValueError(f"Patch count P={P} is not a perfect square.")
    return side, side

# global dist 계산 함수 > 거리와 제일 유사한 k개 유사도, 인덱스 반환
def _dist_global(q: torch.Tensor, ref: torch.Tensor, k: int):
    """
    q: (D,) or (1,D)
    ref: (N,D)
    return dist(float), (topk_sim(1,k), topk_idx(1,k))
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)  # (1,D)

    q = F.normalize(q, dim=1)
    # ref = F.normalize(ref, dim=1) # bank쪽에서 한번에 정규화

    k = min(k, ref.shape[0])
    sim = q @ ref.T                       # (1,N)
    topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)
    dist = (1.0 - topk_sim).mean().item()
    return dist, (topk_sim, topk_idx)

def _dist_patchcore(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
    k: int = 3,
    alpha: float = 0.6,
):
    q = F.normalize(q_patch, dim=1)
    # ref = F.normalize(ref_patch, dim=2)  # bank쪽에서 한번에 정규화
    ref = ref_patch

    # (N, Pq, Pr)
    sim = torch.einsum("qd,npd->nqp", q, ref)
    max_sim, best_ref_patch_idx = sim.max(dim=2)   # (N, Pq)
    dist = 1.0 - max_sim

    Pq = dist.shape[1]
    m = max(1, int(Pq * top_p))

    # --------------------------------------------------
    # ref image별:
    # 1) top-p patch 선택
    # 2) cut = max(alpha * peak, 0.2) thresholding
    # 3) 남은 patch 평균을 score로 사용
    # --------------------------------------------------
    score_per_img = []
    per_img_top_vals = []
    per_img_top_idx = []
    per_img_keep_mask = []

    for n in range(dist.shape[0]):
        patch_dist = dist[n]  # (Pq,)

        # top-p
        top_vals, top_idx = torch.topk(patch_dist, k=m)   # (m,), (m,)

        # peak-relative + absolute lower bound threshold
        peak_val = top_vals.max()
        cut = max(alpha * float(peak_val.item()), 0.2)

        keep = top_vals >= cut

        # 혹시 전부 제거되는 상황 방지
        if keep.sum() == 0:
            keep = torch.zeros_like(top_vals, dtype=torch.bool)
            keep[0] = True

        kept_vals = top_vals[keep]   # (m',)
        kept_idx = top_idx[keep]     # (m',)

        score_one = kept_vals.mean()

        score_per_img.append(score_one)
        per_img_top_vals.append(kept_vals)
        per_img_top_idx.append(kept_idx)
        per_img_keep_mask.append(keep)

    score_per_img = torch.stack(score_per_img, dim=0)  # (N,)

    # score가 작을수록 정상에 가까움
    k2 = min(k, score_per_img.shape[0])
    best_val, best_idx = torch.topk(-score_per_img, k=k2)
    topk_score = -best_val                             # (k,)

    # 가장 좋은 ref image 1개 기준
    best_img_idx = best_idx[0]                         # scalar
    best_patch_dist = dist[best_img_idx]              # (Pq,)
    best_patch_match_idx = best_ref_patch_idx[best_img_idx]

    # best ref에서 thresholding 후 남은 patch들
    top_patch_vals = per_img_top_vals[int(best_img_idx.item())]
    top_patch_idx = per_img_top_idx[int(best_img_idx.item())]
    keep_mask = per_img_keep_mask[int(best_img_idx.item())]   # top-p 내부 mask

    score = topk_score.mean().item()

    debug = {
        "topk_score": topk_score,            # (k,)
        "topk_idx": best_idx,                # (k,)
        "best_img_idx": best_img_idx,        # scalar
        "best_patch_dist": best_patch_dist,  # (Pq,)
        "best_patch_match_idx": best_patch_match_idx,
        "top_patch_idx": top_patch_idx,      # (m') thresholding 후 살아남은 patch idx
        "top_patch_vals": top_patch_vals,    # (m') thresholding 후 살아남은 값
        "keep_mask": keep_mask,              # (m,) top-p 내부 keep mask
        "alpha": alpha,
    }
    return score, debug

import numpy as np
import torch
import torch.nn.functional as F


def _build_filtered_map_from_topk(
    patch_dist_valid: torch.Tensor,   # (Pv,)
    patch_match_valid: torch.Tensor,  # (Pv,)
    valid_orig_idx_t: torch.Tensor,   # (Pv,)
    P: int,
    Hp: int,
    Wp: int,
    top_p: float = 0.05,
    alpha: float = 0.6,
    min_cut = 0.2,
):
    """
    patch_dist_valid에서 top-p를 고른 뒤,
    cut = max(alpha * peak, 0.2) 이상만 남겨 2D filtered map으로 복원.
    """
    device = patch_dist_valid.device

    k_top = max(1, int(np.ceil(patch_dist_valid.numel() * top_p)))
    top_vals, top_local_idx = torch.topk(
        patch_dist_valid,
        k=min(k_top, patch_dist_valid.numel())
    )

    top_idx = valid_orig_idx_t[top_local_idx]
    top_match_idx = patch_match_valid[top_local_idx]

    full_patch_dist = torch.zeros(P, device=device, dtype=patch_dist_valid.dtype)
    full_patch_match = torch.full(
        (P,), fill_value=-1, device=device, dtype=torch.long
    )

    if top_vals.numel() == 0:
        filtered_map = full_patch_dist.view(Hp, Wp)
        return {
            "filtered_map": filtered_map,
            "kept_vals": top_vals,
            "kept_idx": top_idx,
            "kept_match_idx": top_match_idx,
            "cut": torch.tensor(min_cut, device=device, dtype=patch_dist_valid.dtype),
            "peak_val": 0.0,
            "full_patch_dist": full_patch_dist,
            "full_patch_match": full_patch_match,
        }

    peak = top_vals.max()
    cut = torch.maximum(
        alpha * peak,
        torch.tensor(min_cut, device=device, dtype=patch_dist_valid.dtype)
    )

    keep = top_vals >= cut
    kept_vals = top_vals[keep]
    kept_idx = top_idx[keep]
    kept_match_idx = top_match_idx[keep]

    # 아무 것도 안 남으면 최고값 1개는 남긴다
    if kept_vals.numel() == 0:
        kept_vals = top_vals[:1]
        kept_idx = top_idx[:1]
        kept_match_idx = top_match_idx[:1]

    full_patch_dist[kept_idx] = kept_vals
    full_patch_match[kept_idx] = kept_match_idx
    filtered_map = full_patch_dist.view(Hp, Wp)

    return {
        "filtered_map": filtered_map,
        "kept_vals": kept_vals,
        "kept_idx": kept_idx,
        "kept_match_idx": kept_match_idx,
        "cut": cut,
        "peak_val": float(peak.item()),
        "full_patch_dist": full_patch_dist,
        "full_patch_match": full_patch_match,
    }


def _score_connected_excess(
    filtered_map: torch.Tensor,       # (Hp, Wp)
    tau: float,
    singleton_weight: float = 0.25,
    component_min_area: int = 2,
):
    """
    filtered heatmap에서 connected component를 찾고
    area >= 2 : sum(H - tau)
    area == 1 : singleton_weight * max(H - tau)

    반환:
        {
            "score": best_score (tensor),
            "mask": best component mask (Hp, Wp) bool,
            "area": best area,
            "peak": best component peak,
            "mean": best component mean,
        }
    """
    H = filtered_map
    device = H.device
    Hp, Wp = H.shape

    bin_map = (H > tau)
    visited = torch.zeros((Hp, Wp), device=device, dtype=torch.bool)

    dirs = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    best_score = torch.tensor(0.0, device=device, dtype=H.dtype)
    best_mask = torch.zeros((Hp, Wp), device=device, dtype=torch.bool)
    best_area = 0
    best_peak = 0.0
    best_mean = 0.0

    for y in range(Hp):
        for x in range(Wp):
            if (not bin_map[y, x]) or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            coords = []

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))

                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < Hp and 0 <= nx < Wp:
                        if bin_map[ny, nx] and (not visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            ys = torch.tensor([p[0] for p in coords], device=device, dtype=torch.long)
            xs = torch.tensor([p[1] for p in coords], device=device, dtype=torch.long)
            vals = H[ys, xs]

            excess = torch.clamp(vals - tau, min=0.0)
            area = len(coords)

            if area >= component_min_area:
                score = excess.sum()
            else:
                score = singleton_weight * excess.max()

            if score > best_score:
                best_score = score
                best_area = area
                best_peak = float(vals.max().item())
                best_mean = float(vals.mean().item())
                best_mask = torch.zeros((Hp, Wp), device=device, dtype=torch.bool)
                best_mask[ys, xs] = True

    return {
        "score": best_score,
        "mask": best_mask,
        "area": best_area,
        "peak": best_peak,
        "mean": best_mean,
    }


def _dist_patchcore_masked_local(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    valid_mask_1d,              # (P,) bool
    top_p: float = 0.05,
    k: int = 1,
    alpha: float = 0.6,
    radius: int = 1,
    singleton_weight = 0.25,
    min_cut = 0.2,
    component_min_area = 2,
    grid_hw=None,
):
    q = F.normalize(q_patch, dim=1)
    ref = ref_patch

    valid_mask_1d = torch.as_tensor(valid_mask_1d, device=q.device, dtype=torch.bool)

    if valid_mask_1d.numel() != q.shape[0]:
        raise ValueError(f"valid_mask size mismatch: {valid_mask_1d.numel()} vs {q.shape[0]}")

    valid_count = int(valid_mask_1d.sum().item())
    if valid_count < 4:
        return None, {"ok": False, "reason": "too_few_valid_patches"}

    N, P, D = ref.shape

    if grid_hw is None:
        Hp, Wp = _infer_patch_grid(P)
    else:
        Hp, Wp = int(grid_hw[0]), int(grid_hw[1])
        if Hp * Wp != P:
            raise ValueError(f"grid_hw mismatch: {(Hp, Wp)} vs P={P}")

    q2 = q.view(Hp, Wp, D)
    ref2 = ref.view(N, Hp, Wp, D)
    valid2 = valid_mask_1d.view(Hp, Wp)

    best_dist_list = []
    best_match_idx_list = []
    valid_orig_idx = []

    # --------------------------------------------------
    # 1) valid patch마다 local window에서 best match 찾기
    # --------------------------------------------------

    edge_margin = 1 # 테두리 바로 옆은

    for y in range(Hp):
        for x in range(Wp):
            if not valid2[y, x]:
                continue

            qv = q2[y, x]

            y0 = max(0, y - radius)
            y1 = min(Hp, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(Wp, x + radius + 1)

            ref_win = ref2[:, y0:y1, x0:x1, :].reshape(N, -1, D)

            sim = torch.einsum("d,nld->nl", qv, ref_win)
            max_sim, best_local_idx = sim.max(dim=1)
            dist = 1.0 - max_sim

            win_w = x1 - x0
            dy = torch.div(best_local_idx, win_w, rounding_mode="floor")
            dx = best_local_idx % win_w
            best_global_idx = (y0 + dy) * Wp + (x0 + dx)

            best_dist_list.append(dist)
            best_match_idx_list.append(best_global_idx)
            valid_orig_idx.append(y * Wp + x)

    if len(best_dist_list) < 4:
        return None, {"ok": False, "reason": "too_few_valid_patches_after_local"}

    dist_map = torch.stack(best_dist_list, dim=1)              # (N, Pv)
    match_idx_map = torch.stack(best_match_idx_list, dim=1)    # (N, Pv)
    Pv = dist_map.shape[1]
    valid_orig_idx_t = torch.tensor(valid_orig_idx, device=q.device, dtype=torch.long)

    score_per_img = []
    per_img_peak = []
    per_img_area = []
    per_img_mean = []
    per_img_top_patch_idx = []
    per_img_top_patch_vals = []
    per_img_top_patch_match_idx = []
    per_img_best_patch_dist_full = []
    per_img_best_patch_match_full = []
    per_img_best_component_mask = []

    # --------------------------------------------------
    # 2) ref image별 filtered heatmap + connected component score
    # --------------------------------------------------
    for n in range(N):
        patch_dist_valid = dist_map[n]         # (Pv,)
        patch_match_valid = match_idx_map[n]   # (Pv,)

        # full map (디버그 / 시각화용)
        full_patch_dist = torch.zeros(P, device=q.device, dtype=patch_dist_valid.dtype)
        full_patch_dist[valid_orig_idx_t] = patch_dist_valid

        full_patch_match = torch.full(
            (P,),
            fill_value=-1,
            device=q.device,
            dtype=torch.long,
        )
        full_patch_match[valid_orig_idx_t] = patch_match_valid

        per_img_best_patch_dist_full.append(full_patch_dist)
        per_img_best_patch_match_full.append(full_patch_match)

        # (a) 네가 원한 heatmap 필터 유지
        filt = _build_filtered_map_from_topk(
            patch_dist_valid=patch_dist_valid,
            patch_match_valid=patch_match_valid,
            valid_orig_idx_t=valid_orig_idx_t,
            P=P,
            Hp=Hp,
            Wp=Wp,
            top_p=top_p,
            alpha=alpha,
            min_cut=min_cut
        )

        filtered_map = filt["filtered_map"]
        cut = float(filt["cut"].item())
        peak_val = float(filt["peak_val"])

        # valid 영역 밖은 제거
        filtered_map = filtered_map.clone()
        filtered_map[~valid2] = 0.0

        # (b) filtered heatmap을 connected component로 점수화
        cc = _score_connected_excess(
            filtered_map=filtered_map,
            tau=cut,
            singleton_weight=singleton_weight,
            component_min_area=component_min_area,
        )

        score_one = cc["score"]         # tensor
        selected_area = int(cc["area"])
        mean_val = float(cc["mean"])
        best_comp_mask_2d = cc["mask"]

        # best component에 속한 patch만 시각화용 top_patch로 저장
        comp_idx = torch.where(best_comp_mask_2d.view(-1))[0]
        if comp_idx.numel() > 0:
            kept_vals = full_patch_dist[comp_idx]
            kept_match_idx = full_patch_match[comp_idx]
            kept_idx = comp_idx
        else:
            kept_vals = torch.empty(0, device=q.device, dtype=full_patch_dist.dtype)
            kept_match_idx = torch.empty(0, device=q.device, dtype=torch.long)
            kept_idx = torch.empty(0, device=q.device, dtype=torch.long)

        score_per_img.append(score_one)
        per_img_peak.append(peak_val)
        per_img_area.append(selected_area)
        per_img_mean.append(mean_val)
        per_img_top_patch_idx.append(kept_idx)
        per_img_top_patch_vals.append(kept_vals)
        per_img_top_patch_match_idx.append(kept_match_idx)
        per_img_best_component_mask.append(best_comp_mask_2d)

    score_per_img = torch.stack([
        s if torch.is_tensor(s) else torch.tensor(s, device=q.device, dtype=torch.float32)
        for s in score_per_img
    ])

    # --------------------------------------------------
    # 3) 이제 score는 "클수록 이상" 이므로 그대로 topk
    # --------------------------------------------------
    k_ref = min(2, score_per_img.numel())
    best_vals, best_idxs = torch.topk(score_per_img, k=k_ref)

    score = float(best_vals.mean().item())
    best_img_idx = best_idxs[0]

    best_patch_dist_full = per_img_best_patch_dist_full[int(best_img_idx.item())]
    best_patch_match_full = per_img_best_patch_match_full[int(best_img_idx.item())]
    top_patch_idx = per_img_top_patch_idx[int(best_img_idx.item())]
    top_patch_vals = per_img_top_patch_vals[int(best_img_idx.item())]
    top_patch_match_idx = per_img_top_patch_match_idx[int(best_img_idx.item())]
    best_component_mask = per_img_best_component_mask[int(best_img_idx.item())]

    print("score =", score)
    print("selected_patch_count =", int(len(top_patch_vals)))
    print("score_mean =", float(top_patch_vals.mean().item()) if top_patch_vals.numel() > 0 else 0.0)
    print("score_peak =", float(top_patch_vals.max().item()) if top_patch_vals.numel() > 0 else 0.0)

    debug = {
        "ok": True,
        "reason": "cc_excess_after_topk_filter",
        "best_img_idx": best_img_idx,
        "score_per_img": score_per_img,
        "best_patch_dist": best_patch_dist_full,
        "best_patch_match_idx": best_patch_match_full,
        "top_patch_idx": top_patch_idx,
        "top_patch_vals": top_patch_vals,
        "top_patch_match_idx": top_patch_match_idx,
        "best_component_mask": best_component_mask,
        "valid_patch_count": valid_count,
        "local_radius": radius,
        "top_p": top_p,
        "alpha": alpha,
        "singleton_weight": singleton_weight,
        "selected_patch_count": int(len(top_patch_vals)),
        "score_mean": float(top_patch_vals.mean().item()) if top_patch_vals.numel() > 0 else 0.0,
        "score_peak": float(top_patch_vals.max().item()) if top_patch_vals.numel() > 0 else 0.0,
        "best_component_peak": float(per_img_peak[int(best_img_idx.item())]),
        "best_component_area": int(per_img_area[int(best_img_idx.item())]),
        "best_component_mean": float(per_img_mean[int(best_img_idx.item())]),
        "grid_hw": (Hp, Wp),
    }
    return score, debug
# =========================================================================================== dist 계산 함수


# ------------------------------------------------------------------------------------------------------------
# 각 모드에 맞추어 knn dist를 return
@torch.inference_mode()
def compute_knn_dist(
    q_out,                 # DINO query embedding output
    ref_bank,              # load_bank_by_place 결과
    k=3,
    repr_mode="global",
    top_p=0.1,
    preselect_m=10,
    q_img_bgr=None,
    global_model=None,     # DINO model
    local_model=None,      # CNN model
    device=None,           # 하나만 사용
    tfm=None,
    local_tfm = None,
    global_mode="patch_mean",
    sg_matcher=None,
    cfg=None,
):
    timing = {}
    pcfg = cfg.get("patchcore", {})
    alpha = float(pcfg.get("alpha", 0.6))
    min_cut = float(pcfg.get("min_cut", 0.2))
    singleton_weight = float(pcfg.get("singleton_weight", 0.25))
    component_min_area = int(pcfg.get("component_min_area", 2))

    # ---------------------------------------------------
    # global
    # ---------------------------------------------------
    if repr_mode == "global":
        with _timer(timing, "global.ref_to_gpu", q_out["global"].device):
            ref_embs_np, ref_paths = ref_bank["global"]
            q = q_out["global"]
            ref = torch.from_numpy(ref_embs_np).float().to(q.device)

        with _timer(timing, "global.dist", q.device):
            dist, debug_inner = _dist_global(q, ref, k)

        debug = {
            "timing": _stats_to_float_dict(timing),
            "inner_debug": debug_inner,
        }
        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return dist, debug, ref_paths

    # ---------------------------------------------------
    # patch
    # ---------------------------------------------------
    elif repr_mode == "patch":
        with _timer(timing, "patch.ref_to_gpu", q_out["patch"].device):
            ref_patch_np, ref_paths = ref_bank["patch"]
            q_patch = q_out["patch"]
            ref_patch = torch.from_numpy(ref_patch_np).float().to(q_patch.device)

        with _timer(timing, "patch.dist", q_patch.device):
            score, debug = _dist_patchcore(
                q_patch,
                ref_patch,
                top_p=top_p,
                k=k,
                alpha=alpha,
            )

        debug["timing"] = _stats_to_float_dict(timing)
        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return score, debug, ref_paths

    # ---------------------------------------------------
    # global + patch
    # ---------------------------------------------------
    elif repr_mode == "global_patch":
        with _timer(timing, "gp.refg_to_gpu", q_out["global"].device):
            refg_np, ref_paths = ref_bank["global"]
            refp_np, _ = ref_bank["patch"]

            qg = q_out["global"]
            qp = q_out["patch"]

            refg = torch.from_numpy(refg_np).float().to(qg.device)

        with _timer(timing, "gp.global_preselect", qg.device):
            _, (topk_sim, topk_idx) = _dist_global(
                qg,
                refg,
                min(preselect_m, refg.shape[0]),
            )
            idx = topk_idx.squeeze(0)

        with _timer(timing, "gp.refp_to_gpu", qp.device):
            refp = torch.from_numpy(refp_np).float().to(qp.device)
            refp_sel = refp[idx]

        with _timer(timing, "gp.patch_dist", qp.device):
            score, patch_debug_local = _dist_patchcore(
                qp,
                refp_sel,
                top_p=top_p,
                k=min(k, refp_sel.shape[0]),
                alpha=alpha,
            )

        topk_idx_global = idx[patch_debug_local["topk_idx"]]
        best_img_idx_global = idx[patch_debug_local["best_img_idx"]]

        debug = {
            "global_topk": (topk_sim, idx),
            "patch_topk": {
                "best_ref_img_path": str(ref_paths[int(best_img_idx_global.item())]),
                "top_patch_match_idx": patch_debug_local["best_patch_match_idx"][patch_debug_local["top_patch_idx"]],
                "topk_score": patch_debug_local["topk_score"],
                "topk_idx": topk_idx_global,
                "best_img_idx": best_img_idx_global,
                "best_patch_dist": patch_debug_local["best_patch_dist"],
                "top_patch_idx": patch_debug_local["top_patch_idx"],
                "top_patch_vals": patch_debug_local["top_patch_vals"],
            },
            "timing": _stats_to_float_dict(timing),
        }

        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return score, debug, ref_paths

    # ---------------------------------------------------
    # global + aligned local CNN

# ---------------------------------------------------
# global + aligned local CNN
# ---------------------------------------------------
    elif repr_mode == "global_patch_with_aligned":
        if sg_matcher is None:
            raise ValueError("sg_matcher is required for global_patch_with_aligned")
        if q_img_bgr is None:
            raise ValueError("q_img_bgr is required for global_patch_with_aligned")
        if tfm is None:
            raise ValueError("tfm is required for global_patch_with_aligned")
        if local_model is None or device is None:
            raise ValueError("local_model/device are required for global_patch_with_aligned")
        if cfg is None:
            raise ValueError("cfg is required for global_patch_with_aligned")

        sg_raw = cfg["superglue"]

        pcfg = cfg.get("patchcore", {})

        local_radius = int(pcfg.get("radius", 1))
        alpha = float(pcfg.get("alpha", 0.6))
        min_cut = float(pcfg.get("min_cut", 0.2))
        singleton_weight = float(pcfg.get("singleton_weight", 0.25))
        component_min_area = int(pcfg.get("component_min_area", 2))

        refg_np, ref_paths = ref_bank["global"]
        qg = q_out["global"]

        refg = torch.from_numpy(refg_np).float().to(qg.device)

        # 1) coarse global 탐색 (DINO global)
        _, (topk_sim, topk_idx) = _dist_global(
            qg,
            refg,
            min(preselect_m, refg.shape[0]),
        )
        idx = topk_idx.squeeze(0)

        candidates = []

        # 2) coarse 후보에 대해 정합 + crop + CNN grid feature 추출
        for ref_i in idx.tolist():
            ref_img_path = ref_paths[ref_i]
            ref_img_bgr = cv2.imread(str(ref_img_path), cv2.IMREAD_COLOR)
            if ref_img_bgr is None:
                print("[DEBUG] ref_img read fail:", ref_i, ref_img_path)
                continue

            match_out = sg_matcher.match_and_estimate(q_img_bgr, ref_img_bgr)
            if not match_out["ok"]:
                print("[DEBUG] match fail:", ref_i)
                continue

            warped_q_bgr, warped_mask = warp_query_to_bank(
                q_img_bgr,
                match_out["H"],
                bank_hw=ref_img_bgr.shape[:2],
            )

            warped_q_crop, ref_crop, mask_crop, crop_bbox = crop_common_safe_region(
                warped_q_bgr,
                ref_img_bgr,
                warped_mask,
                erode_kernel=11,
                erode_iter=2,
                margin=8,
                min_size=64,
            )
            if warped_q_crop is None:
                print("[DEBUG] crop fail:", ref_i)
                continue
            
            x_q_crop = local_tfm(BGR_to_RGB(warped_q_crop))
            qp_crop, q_hw = extract_grid_layers(local_model, device, x_q_crop)

            x_ref_crop = local_tfm(BGR_to_RGB(ref_crop))
            refp_crop, ref_hw = extract_grid_layers(local_model, device, x_ref_crop)

            if q_hw != ref_hw:
                print("[DEBUG] grid_hw mismatch:", ref_i, q_hw, ref_hw)
                continue

            refp_crop = refp_crop.unsqueeze(0)
            grid_h2, grid_w2 = q_hw

            patch_valid_2d = make_patch_valid_mask(
                mask_crop,
                grid_h=grid_h2,
                grid_w=grid_w2,
                thr=sg_raw["valid_patch_thr"],
            )
            patch_valid_1d = patch_valid_2d.reshape(-1)

            valid_patch_count = int(patch_valid_1d.sum())
            if valid_patch_count < 4:
                print("[DEBUG] too few valid patches:", ref_i, valid_patch_count)
                continue

            score_one, dbg_one = _dist_patchcore_masked_local(
                qp_crop,
                refp_crop,
                valid_mask_1d=patch_valid_1d,
                top_p=top_p,
                k=1,
                radius=local_radius,
                alpha=alpha,
                singleton_weight=singleton_weight,
                min_cut=min_cut,
                component_min_area=component_min_area,
                grid_hw=q_hw,
            )
            if score_one is None or not dbg_one.get("ok", True):
                print("[DEBUG] local masked dist fail:", ref_i, dbg_one.get("reason", None))
                continue

            if DEBUG_VIS:
                vis_query_bgr = _to_vis_bgr_from_rgb_tensor(x_q_crop)
                vis_ref_bgr = _to_vis_bgr_from_rgb_tensor(x_ref_crop)
            else:
                vis_query_bgr = None
                vis_ref_bgr = None


            candidates.append({
                "ref_i": ref_i,
                "ref_img_path": str(ref_img_path),
                "score": float(score_one),
                "patch_debug": dbg_one,
                "match_out": match_out,
                "crop_bbox": crop_bbox,
                "valid_patch_count": valid_patch_count,
                "crop_shape": warped_q_crop.shape[:2],
                "vis_query_bgr": vis_query_bgr,
                "vis_ref_bgr": vis_ref_bgr,
            })

        # fallback: 기존 global_patch
        if len(candidates) == 0:
            print("[DEBUG] aligned -> fallback global_patch")
            refp_np, _ = ref_bank["patch"]
            qp = q_out["patch"]
            refp = torch.from_numpy(refp_np).float().to(qp.device)
            refp_sel = refp[idx]

            score, patch_debug_local = _dist_patchcore(
                qp,
                refp_sel,
                top_p=top_p,
                k=min(k, refp_sel.shape[0]),
                alpha=alpha,
            )

            topk_idx_global = idx[patch_debug_local["topk_idx"]]
            best_img_idx_global = idx[patch_debug_local["best_img_idx"]]

            debug = {
                "fallback": True,
                "global_topk": (topk_sim, idx),
                "patch_topk": {
                    "best_ref_img_path": str(ref_paths[int(best_img_idx_global.item())]),
                    "top_patch_match_idx": patch_debug_local["best_patch_match_idx"][patch_debug_local["top_patch_idx"]],
                    "topk_score": patch_debug_local["topk_score"],
                    "topk_idx": topk_idx_global,
                    "best_img_idx": best_img_idx_global,
                    "best_patch_dist": patch_debug_local["best_patch_dist"],
                    "top_patch_idx": patch_debug_local["top_patch_idx"],
                    "top_patch_vals": patch_debug_local["top_patch_vals"],
                },
                "timing": _stats_to_float_dict(timing),
            }
            _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
            return score, debug, ref_paths

        # 최종 score는 best ref 1개 기준으로
        best = min(candidates, key=lambda x: x["score"])
        score = float(best["score"])

        topk_idx_global = torch.tensor(
            [best["ref_i"]],
            device=qg.device,
            dtype=torch.long,
        )
        topk_score = torch.tensor(
            [best["score"]],
            device=qg.device,
            dtype=torch.float32,
        )

        best_patch_debug = best["patch_debug"]
        best_match = best["match_out"]

        debug = {
            "fallback": False,
            "global_topk": (topk_sim, idx),
            "patch_topk": {
                "best_ref_img_path": best["ref_img_path"],
                "top_patch_match_idx": best_patch_debug["top_patch_match_idx"],
                "topk_score": topk_score,
                "topk_idx": topk_idx_global,
                "best_img_idx": torch.tensor(best["ref_i"], device=qg.device),
                "best_patch_dist": best_patch_debug["best_patch_dist"],
                "top_patch_idx": best_patch_debug["top_patch_idx"],
                "top_patch_vals": best_patch_debug["top_patch_vals"],
            },
            "timing": _stats_to_float_dict(timing),
        }

        if DEBUG_VIS:
            debug["align_debug"] = {
                "best_ref_i": int(best["ref_i"]),
                "best_ref_img_path": best["ref_img_path"],
                "crop_bbox": best["crop_bbox"],
                "crop_shape": best["crop_shape"],
                "grid_hw": best_patch_debug.get("grid_hw", None),
                "valid_patch_count": int(best["valid_patch_count"]),
                "H": best_match["H"],
                "inliers": best_match["inliers"],
                "inlier_ratio": best_match["inlier_ratio"],
                "reproj_error_mean": best_match["reproj_error_mean"],
                "reproj_error_median": best_match["reproj_error_median"],
                "top_patch_idx": best_patch_debug["top_patch_idx"],
                "top_patch_vals": best_patch_debug["top_patch_vals"],
                "top_patch_match_idx": best_patch_debug["top_patch_match_idx"],
                "vis_query_bgr": best["vis_query_bgr"].tolist(),
                "vis_ref_bgr": best["vis_ref_bgr"].tolist(),
            }

        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return score, debug, ref_paths


def th_percentile(scores, percentile):
    return float(np.percentile(scores, percentile))

def th_gaussian(scores, k=3.0):
    mu = scores.mean()
    sigma = scores.std()
    return float(mu + k * sigma)

def th_robust(scores, k=3.0):
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    return float(med + k * mad * 1.4826)


def calibrate_place(bank_root, plc_idx, global_model, local_model, device, sg_matcher=None, cfg=None):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    timing = {}
    if cfg is None:
        cfg = load_cfg(bank_root)

    # bank rebuild → DINO 기준
    with _timer(timing, "calibrate.rebuild_bank_bank", device):
        rebuild_bank(bank_root, plc_idx, global_model, device, mode="bank", cfg=cfg)

    with _timer(timing, "calibrate.rebuild_bank_th", device):
        rebuild_bank(bank_root, plc_idx, global_model, device, mode="th_calib", cfg=cfg)

    # threshold 계산
    with _timer(timing, "calibrate.compute_and_save_threshold", device):
        thr, scores, thr_path = compute_and_save_threshold(
            bank_root,
            plc_idx,
            cfg=cfg,
            global_model=global_model,   
            local_model=local_model,     
            device=device,
            sg_matcher=sg_matcher,
        )

    _print_timing(f"calibrate_place[{plc_idx}]", timing)
    print("[CALIB-RAW] n =", len(scores))
    print("[CALIB-RAW] min/max =", float(np.min(scores)), float(np.max(scores)))
    print("[CALIB-RAW] median =", float(np.median(scores)))
    mad = float(np.median(np.abs(scores - np.median(scores))))
    print("[CALIB-RAW] mad =", mad)
    print("[CALIB-RAW] thr =", float(thr))
    print("[CALIB-RAW] top5 =", sorted([float(x) for x in scores])[-5:])

    
    return thr, scores, thr_path

@torch.inference_mode()
def compute_and_save_threshold(
    bank_root,
    plc_idx,
    cfg: dict,
    global_model=None,   # DINO
    local_model=None,    # CNN
    device=None,
    sg_matcher=None,
):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    total_timing = {}

    r = cfg.get("repr", {})
    c = cfg.get("calib", {})

    pcfg = cfg.get("patchcore", {})
    top_p = float(pcfg.get("top_p", 0.1))
    preselect_m = int(pcfg.get("preselect_m", 10))
    local_radius = int(pcfg.get("radius", 1))
    alpha = float(pcfg.get("alpha", 0.6))
    min_cut = float(pcfg.get("min_cut", 0.2))
    singleton_weight = float(pcfg.get("singleton_weight", 0.25))
    component_min_area = int(pcfg.get("component_min_area", 2))

    repr_mode = str(r.get("repr_mode", "global"))

    k          = int(c.get("k", 3))
    percentile = int(c.get("percentile", 97))
    method     = str(c.get("method", "robust"))
    robust_k   = float(c.get("robust_k", 2.5))
    gaussian_k = float(c.get("gaussian_k", 2.5))

    bank_hash_src = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
    with open(bank_hash_src, "rb") as f:
        bank_hash = hashlib.sha1(f.read()).hexdigest()[:8]

    if repr_mode == "global":
        with _timer(total_timing, "calib.global.load_npz", device):
            bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
            th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
            bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()
            th   = torch.from_numpy(np.load(th_npz, allow_pickle=True)["embs"]).float()
            th   = F.normalize(th, dim=1)

        with _timer(total_timing, "calib.global.score_calc", device):
            k2 = min(k, bank.shape[0])
            sim = th @ bank.T
            topk_sim, _ = torch.topk(sim, k=k2, dim=1)
            scores = (1.0 - topk_sim).mean(dim=1).numpy()

    elif repr_mode == "patch":
        with _timer(total_timing, "calib.patch.load_npz", device):
            bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
            th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

            bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()
            th   = torch.from_numpy(np.load(th_npz,   allow_pickle=True)["embs"]).float()

        scores = []
        for i in range(th.shape[0]):
            with _timer(total_timing, "calib.patch.per_th_dist", device):
                q_patch = th[i]
                s, _ = _dist_patchcore(
                    q_patch,
                    bank,
                    top_p=top_p,
                    k=k,
                    alpha=alpha,
                )
                scores.append(s)

    elif repr_mode == "global_patch":
        with _timer(total_timing, "calib.gp.load_npz", device):
            bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
            th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
            bank_p_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
            th_p_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

            bank_g = torch.from_numpy(np.load(bank_g_npz, allow_pickle=True)["embs"]).float()
            th_g   = torch.from_numpy(np.load(th_g_npz,   allow_pickle=True)["embs"]).float()
            bank_p = torch.from_numpy(np.load(bank_p_npz, allow_pickle=True)["embs"]).float()
            th_p   = torch.from_numpy(np.load(th_p_npz,   allow_pickle=True)["embs"]).float()

            th_g = F.normalize(th_g, dim=1)
            scores = []
            for i in range(th_g.shape[0]):
                with _timer(total_timing, "calib.gp.per_th_global_preselect", device):
                    sim = th_g[i:i+1] @ bank_g.T
                    M = min(preselect_m, bank_g.shape[0])
                    _, idx = torch.topk(sim, k=M, dim=1)
                    idx = idx.squeeze(0)

                with _timer(total_timing, "calib.gp.per_th_patch_dist", device):
                    bank_sel = bank_p[idx]
                    s, _ = _dist_patchcore(
                        th_p[i],
                        bank_sel,
                        top_p=top_p,
                        k=k,
                        alpha=alpha,
                    )
                    scores.append(s)

        scores = np.array(scores, dtype=np.float32)

    elif repr_mode == "global_patch_with_aligned":
        if global_model is None:
            raise ValueError("global_model is required for global_patch_with_aligned")
        if local_model is None:
            raise ValueError("local_model is required for global_patch_with_aligned")

        with _timer(total_timing, "calib.aligned.load_npz", device):
            bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
            th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"

            bank_g = torch.from_numpy(
                np.load(bank_g_npz, allow_pickle=True)["embs"]
            ).float().to(device)

            th_g = torch.from_numpy(
                np.load(th_g_npz, allow_pickle=True)["embs"]
            ).float().to(device)

            bank_g = F.normalize(bank_g, dim=1)
            th_g   = F.normalize(th_g, dim=1)

            e = cfg.get("embed", {})
            img_size = int(e.get("img_size", 560))
            global_mode = str(e.get("global_mode", "patch_mean"))
            tfm = make_transform(img_size=img_size)
            local_tfm = make_aligned_local_transform(img_size=img_size)

            sg_raw = cfg["superglue"]

        with _timer(total_timing, "calib.aligned.load_ref_bank", device):
            ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")
            bank_ref_paths = ref_bank["global"][1]

        with _timer(total_timing, "calib.aligned.list_th_imgs", device):
            th_dir = bank_root / plc_idx / "th_calib"
            th_img_paths = sorted(
                [p for p in th_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            )

        scores = []

        for th_idx, th_img_path in enumerate(th_img_paths):
            loop_timing = {}

            with _timer(loop_timing, "imread", device):
                img_bgr = cv2.imread(str(th_img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            with _timer(loop_timing, "tfm", device):
                x = tfm(BGR_to_RGB(img_bgr))

            with _timer(loop_timing, "embed_global_patch", device):
                q_out = make_embed(
                    global_model,
                    device,
                    x,
                    repr_mode=repr_mode,
                    global_mode=global_mode,
                )

            qg = q_out["global"]

            with _timer(loop_timing, "global_preselect", device):
                sim = qg.unsqueeze(0) @ bank_g.T
                M = min(preselect_m, bank_g.shape[0])
                _, idx = torch.topk(sim, k=M, dim=1)
                idx = idx.squeeze(0)

            cand_scores = []

            with _timer(loop_timing, "aligned_candidates", device):
                for ref_i in idx.tolist():
                    ref_img_path = bank_ref_paths[ref_i]
                    ref_img_bgr = cv2.imread(str(ref_img_path), cv2.IMREAD_COLOR)
                    if ref_img_bgr is None:
                        continue

                    match_out = sg_matcher.match_and_estimate(img_bgr, ref_img_bgr)
                    if not match_out["ok"]:
                        continue

                    warped_q_bgr, warped_mask = warp_query_to_bank(
                        img_bgr,
                        match_out["H"],
                        bank_hw=ref_img_bgr.shape[:2],
                    )

                    warped_q_crop, ref_crop, mask_crop, crop_bbox = crop_common_safe_region(
                        warped_q_bgr,
                        ref_img_bgr,
                        warped_mask,
                        erode_kernel=9,
                        erode_iter=2,
                        margin=8,
                        min_size=64,
                    )
                    if warped_q_crop is None:
                        continue

                    x_q_crop = local_tfm(BGR_to_RGB(warped_q_crop))
                    qp_crop, q_hw = extract_grid_layers(local_model, device, x_q_crop)

                    x_ref_crop = local_tfm(BGR_to_RGB(ref_crop))
                    refp_crop, ref_hw = extract_grid_layers(local_model, device, x_ref_crop)

                    if q_hw != ref_hw:
                        continue

                    refp_crop = refp_crop.unsqueeze(0)
                    grid_h2, grid_w2 = q_hw

                    patch_valid_2d = make_patch_valid_mask(
                        mask_crop,
                        grid_h=grid_h2,
                        grid_w=grid_w2,
                        thr=sg_raw["valid_patch_thr"],
                    )
                    patch_valid_1d = patch_valid_2d.reshape(-1)

                    valid_patch_count = int(patch_valid_1d.sum())
                    if valid_patch_count < 4:
                        continue

                    score_one, dbg_one = _dist_patchcore_masked_local(
                        qp_crop,
                        refp_crop,
                        valid_mask_1d=patch_valid_1d,
                        top_p=top_p,
                        k=1,
                        radius=local_radius,
                        alpha=alpha,
                        singleton_weight=singleton_weight,
                        min_cut=min_cut,
                        component_min_area=component_min_area,
                        grid_hw=q_hw,
                    )
                    if score_one is None or not dbg_one.get("ok", True):
                        continue

                    cand_scores.append(float(score_one))

            if len(cand_scores) == 0:
                continue
            else:
                s = float(np.min(cand_scores))

            scores.append(float(s))
            _merge_timing(total_timing, {f"calib.aligned.per_img.{kk}": vv for kk, vv in loop_timing.items()})

        scores = np.array(scores, dtype=np.float32)

    else:
        raise ValueError(f"Unknown repr_mode: {repr_mode}")

    with _timer(total_timing, "calib.threshold_compute", device):
        if method == "percentile":
            thr = th_percentile(scores, percentile)
        elif method == "gaussian":
            thr = th_gaussian(scores, k=gaussian_k)
        elif method == "robust":
            thr = th_robust(scores, k=robust_k)
        else:
            raise ValueError(f"Unknown method: {method}")

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_bank_id = f"{plc_idx}_{repr_mode}_{method}_k{k}_p{percentile}_{bank_hash}_{created_at}"

    out = {
        "plc_idx": plc_idx,
        "repr_mode": repr_mode,
        "k": k,
        "method": method,
        "percentile": percentile,
        "robust_k": robust_k,
        "gaussian_k": gaussian_k,
        "threshold": float(thr),
        "num_th": int(len(scores)),
        "ref_bank_id": ref_bank_id,
        "created_at": created_at,
    }

    with _timer(total_timing, "calib.write_threshold_json", device):
        thr_path = bank_root / plc_idx / "threshold.json"
        thr_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_timing(f"compute_and_save_threshold[{plc_idx}:{repr_mode}]", total_timing)
    return float(thr), scores, thr_path

# ================================================================================= 추론

# event 단위 이상감지
@torch.inference_mode()
def infer_event(
    imgs_bgr,
    bank_root,
    plc_idx,
    cfg: dict = None,
    global_model=None,
    local_model=None,
    device=None,
    sg_matcher=None,
):
    """
    repr_mode: "global" | "patch" | "global_patch" | "global_patch_with_aligned"
    - global: global feature kNN
    - patch: PatchCore-style
    - global_patch: global preselect 후 image-level patchcore
    - global_patch_with_aligned: global preselect 후 각 이미지에 대해 정합 -> local patch별 비교 -> dist
    """

    plc_idx = str(plc_idx)
    bank_root = Path(bank_root)

    if not imgs_bgr:
        raise ValueError("imgs_bgr is empty")

    total_timing = {}

    # -------------------------
    # cfg
    # -------------------------
    with _timer(total_timing, "infer.load_cfg", device):
        if cfg is None:
            cfg = load_cfg(bank_root)

    infer_cfg = cfg.get("infer", {})
    event_rule = str(infer_cfg.get("event_rule", "max"))
    use_two_stage_vlm = bool(infer_cfg.get("use_two_stage_vlm", False))

    e = cfg.get("embed", {})
    img_size = int(e.get("img_size", 560))
    global_mode = str(e.get("global_mode", "patch_mean"))
    tfm = make_transform(img_size=img_size)
    local_tfm = make_aligned_local_transform(img_size=img_size)

    pcfg = cfg.get("patchcore", {})
    top_p = float(pcfg.get("top_p", 0.1))
    preselect_m = int(pcfg.get("preselect_m", 10))

    # -------------------------
    # threshold meta
    # -------------------------
    with _timer(total_timing, "infer.read_threshold_json", device):
        thr_path = bank_root / plc_idx / "threshold.json"
        if not thr_path.exists():
            raise FileNotFoundError(f"No threshold.json for plc_idx={plc_idx} ({thr_path})")
        meta = json.loads(thr_path.read_text(encoding="utf-8"))

    thr = float(meta["threshold"])
    k = int(meta.get("k", 3))
    ref_bank_id = str(meta.get("ref_bank_id", ""))

    repr_mode = str(meta.get("repr_mode", "global")).lower()
    if repr_mode in {"patch_global"}:
        repr_mode = "global_patch"
    if repr_mode not in {"global", "patch", "global_patch", "global_patch_with_aligned"}:
        raise ValueError(
            f"Unknown repr_mode={repr_mode} "
            "(use global|patch|global_patch|global_patch_with_aligned)"
        )

    # -------------------------
    # bank load
    # -------------------------
    with _timer(total_timing, "infer.load_ref_bank", device):
        ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")

    # -------------------------
    # frame loop
    # -------------------------
    frame_scores: List[float] = []
    frame_change_flags: List[int] = []
    topk_paths_all: List[List[str]] = []
    topk_sims_all: List[List[float]] = []
    patch_vis_all: List[Optional[Dict[str, Any]]] = []
    align_vis_all: List[Optional[Dict[str, Any]]] = []
    frame_timing_all: List[Dict[str, float]] = []

    for fi, img_bgr in enumerate(imgs_bgr):
        frame_timing = {}

        with _timer(frame_timing, "tfm", device):
            x = tfm(BGR_to_RGB(img_bgr))

        # global embedding은 DINO
        with _timer(frame_timing, "embed", device):
            q_out = make_embed(
                global_model,
                device,
                x,
                repr_mode=repr_mode,
                global_mode=global_mode,
            )

        # distance 계산: global=DINO, local aligned=CNN
        with _timer(frame_timing, "compute_knn_dist", device):
            dist, debug, ref_paths = compute_knn_dist(
                q_out,
                ref_bank,
                k=k,
                repr_mode=repr_mode,
                global_mode=global_mode,
                top_p=top_p,
                preselect_m=preselect_m,
                q_img_bgr=img_bgr,
                global_model=global_model,
                local_model=local_model,
                device=device,
                tfm=tfm,
                local_tfm=local_tfm,
                sg_matcher=sg_matcher,
                cfg=cfg,
            )

        if isinstance(debug, dict):
            _merge_timing(frame_timing, {f"knn.{kk}": vv for kk, vv in debug.get("timing", {}).items()})

        is_change = dist > thr
        frame_scores.append(float(dist))
        frame_change_flags.append(1 if is_change else 0)

        # aligned vis logging -------------------------------------
        if repr_mode == "global_patch_with_aligned":
            align_vis_all.append(debug.get("align_debug", None))
        else:
            align_vis_all.append(None)

        # topk logging ------------------------------------- 
        if repr_mode == "global":
            topk_sim, topk_idx = debug["inner_debug"]
            idx = topk_idx.squeeze(0).tolist()
            sims = topk_sim.squeeze(0).tolist()
            paths = [ref_paths[i] for i in idx]

        elif repr_mode == "patch":
            topk_score = debug["topk_score"]
            topk_idx = debug["topk_idx"]
            idx = topk_idx.tolist()
            paths = [ref_paths[i] for i in idx]
            sims = (-topk_score).tolist()

        elif repr_mode == "global_patch":
            patch_topk = debug["patch_topk"]
            topk_score = patch_topk["topk_score"]
            topk_idx = patch_topk["topk_idx"]
            idx = topk_idx.tolist()
            paths = [ref_paths[i] for i in idx]
            sims = (-topk_score).tolist()

        elif repr_mode == "global_patch_with_aligned":
            patch_topk = debug["patch_topk"]
            topk_score = patch_topk["topk_score"]
            topk_idx = patch_topk["topk_idx"]
            idx = topk_idx.tolist()
            paths = [ref_paths[i] for i in idx]
            sims = (-topk_score).tolist()

        else:
            raise ValueError(f"Unknown repr_mode={repr_mode}")

        topk_paths_all.append([str(p) for p in paths])
        topk_sims_all.append([float(s) for s in sims])

        top_patch_idx, top_patch_vals = get_top_p_patch_info(repr_mode, debug)

        if top_patch_idx is not None and top_patch_vals is not None:
            align_debug = debug.get("align_debug", None) if isinstance(debug, dict) else None
            align_data = align_debug if isinstance(align_debug, dict) else {}

            patch_vis_all.append({
                "top_patch_idx": top_patch_idx.detach().cpu().tolist()
                if hasattr(top_patch_idx, "detach") else list(top_patch_idx),

                "top_patch_vals": top_patch_vals.detach().cpu().tolist()
                if hasattr(top_patch_vals, "detach") else list(top_patch_vals),

                "top_patch_match_idx": (
                    debug["patch_topk"]["top_patch_match_idx"].detach().cpu().tolist()
                    if "patch_topk" in debug and "top_patch_match_idx" in debug["patch_topk"]
                    and hasattr(debug["patch_topk"]["top_patch_match_idx"], "detach")
                    else list(debug["patch_topk"]["top_patch_match_idx"])
                    if "patch_topk" in debug and "top_patch_match_idx" in debug["patch_topk"]
                    else []
                ),

                "best_ref_img_path": (
                    debug["patch_topk"].get("best_ref_img_path", None)
                    if "patch_topk" in debug else None
                ),

                "repr_mode": repr_mode,
                "img_size": img_size,

                # aligned 시각화용 metadata
                "H": align_data.get("H", None),
                "crop_bbox": align_data.get("crop_bbox", None),
                "grid_hw": align_data.get("grid_hw", None),
                "crop_shape": align_data.get("crop_shape", None),
                "vis_query_bgr": align_data.get("vis_query_bgr", None),
                "vis_ref_bgr": align_data.get("vis_ref_bgr", None),
            })
        else:
            patch_vis_all.append(None)

        frame_timing_all.append(_stats_to_float_dict(frame_timing))
        _print_timing(f"infer_event[{plc_idx}] frame={fi}", frame_timing)
        _merge_timing(total_timing, {f"frame.{kk}": vv for kk, vv in frame_timing.items()})

    # -------------------------
    # event aggregation
    # -------------------------
    with _timer(total_timing, "infer.event_aggregate", device):
        if event_rule == "mean":
            decision_score = float(np.mean(frame_scores))
            anomaly_flag = 1 if decision_score > thr else 0

        elif event_rule == "max":
            decision_score = float(np.max(frame_scores))
            anomaly_flag = 1 if decision_score > thr else 0

        elif event_rule == "median":
            decision_score = float(np.median(frame_scores))
            anomaly_flag = 1 if decision_score > thr else 0

        elif event_rule == "vote":
            decision_score = float(np.median(frame_scores))
            n_ab = int(sum(frame_change_flags))
            n = int(len(frame_change_flags))
            anomaly_flag = 1 if n_ab >= (n / 2) else 0

        else:
            raise ValueError(f"Unknown event_rule: {event_rule}")

        margin = decision_score - thr

        event_score = float(np.clip(
            50.0 + 50.0 * (margin / max(thr, 1e-8)),
            0.0,
            100.0
        ))

        rep_idx = int(np.argmax(frame_scores)) if len(frame_scores) > 0 else 0

    # -------------------------
    # optional: VLM gate
    # -------------------------
    if anomaly_flag == 1:
        rep, summary = None, "anomaly detect"
    else:
        rep, summary = None, "it's fine, have relax"

    if use_two_stage_vlm and anomaly_flag == 1:
        pass

    # -------------------------
    # pack
    # -------------------------
    with _timer(total_timing, "infer.pack_json", device):
        ref_topk_json = json.dumps(
            {"topk_paths": topk_paths_all, "topk_sims": topk_sims_all, "rep": rep},
            ensure_ascii=False,
        )

    _print_timing(f"infer_event[{plc_idx}] total", total_timing)

    return {
        "threshold": float(thr),
        "frame_scores": [float(x) for x in frame_scores],
        "anomaly_flag": int(anomaly_flag),
        "frame_change_flags": [int(x) for x in frame_change_flags],
        "event_score": float(event_score),
        "ref_bank_id": ref_bank_id,
        "ref_topk_json": ref_topk_json,
        "summary": summary,
        "patch_vis": {
            "frame_idx": rep_idx,
            "repr_mode": repr_mode,
            "img_size": img_size,

            # DINO 계열일 때만 사용
            "patch_size": (
                14 if repr_mode != "global_patch_with_aligned" else None
            ),

            "top_patch_idx": (
                patch_vis_all[rep_idx]["top_patch_idx"]
                if len(patch_vis_all) > rep_idx and patch_vis_all[rep_idx] is not None
                else []
            ),
            "top_patch_vals": (
                patch_vis_all[rep_idx]["top_patch_vals"]
                if len(patch_vis_all) > rep_idx and patch_vis_all[rep_idx] is not None
                else []
            ),
            "top_patch_match_idx": (
                patch_vis_all[rep_idx]["top_patch_match_idx"]
                if len(patch_vis_all) > rep_idx and patch_vis_all[rep_idx] is not None
                and "top_patch_match_idx" in patch_vis_all[rep_idx]
                else []
            ),
            "best_ref_img_path": (
                patch_vis_all[rep_idx]["best_ref_img_path"]
                if len(patch_vis_all) > rep_idx and patch_vis_all[rep_idx] is not None
                and "best_ref_img_path" in patch_vis_all[rep_idx]
                else None
            ),

            # aligned 전용 메타데이터
            "grid_hw": (
                align_vis_all[rep_idx]["grid_hw"]
                if repr_mode == "global_patch_with_aligned"
                and len(align_vis_all) > rep_idx
                and align_vis_all[rep_idx] is not None
                and "grid_hw" in align_vis_all[rep_idx]
                else None
            ),
            "crop_bbox": (
                align_vis_all[rep_idx]["crop_bbox"]
                if repr_mode == "global_patch_with_aligned"
                and len(align_vis_all) > rep_idx
                and align_vis_all[rep_idx] is not None
                and "crop_bbox" in align_vis_all[rep_idx]
                else None
            ),
            "crop_shape": (
                align_vis_all[rep_idx]["crop_shape"]
                if repr_mode == "global_patch_with_aligned"
                and len(align_vis_all) > rep_idx
                and align_vis_all[rep_idx] is not None
                and "crop_shape" in align_vis_all[rep_idx]
                else None
            ),
            "vis_query_bgr": (
                patch_vis_all[rep_idx]["vis_query_bgr"]
                if len(patch_vis_all) > rep_idx and patch_vis_all[rep_idx] is not None
                and "vis_query_bgr" in patch_vis_all[rep_idx]
                else None
            ),
            "vis_ref_bgr": (
                patch_vis_all[rep_idx]["vis_ref_bgr"]
                if len(patch_vis_all) > rep_idx and patch_vis_all[rep_idx] is not None
                and "vis_ref_bgr" in patch_vis_all[rep_idx]
                else None
            ),
        },
        "align_vis": (
            {
                "frame_idx": rep_idx,
                "data": align_vis_all[rep_idx],
            }
            if repr_mode == "global_patch_with_aligned"
            and len(align_vis_all) > rep_idx
            and align_vis_all[rep_idx] is not None
            else None
        ),
        "timing": _stats_to_float_dict(total_timing),
        "frame_timing": frame_timing_all,
    }