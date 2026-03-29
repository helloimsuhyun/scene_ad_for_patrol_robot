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
from .dino_emb import make_embed, make_transform, extract_patch_layers
from .config import load_cfg

import cv2

from .warp_utils import (
    warp_query_to_bank,
    make_patch_valid_mask,
    crop_common_valid_region,
)

LOCAL_PATCH_RADIUS = 1
ALPHA = 0.6

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


# ()
from scipy import ndimage
import math

# ---------------- timing utils ----------------
import time
from contextlib import contextmanager

ENABLE_TIMING_LOG = True


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


def _infer_patch_grid(P: int):
    side = int(math.sqrt(P))
    if side * side != P:
        raise ValueError(f"Patch count P={P} is not a perfect square.")
    return side, side


def _remove_small_blob_from_patch_dist(
    patch_dist_1d: torch.Tensor,   # (P,)
    patch_thr: float,
    min_area: int = 3,
) -> torch.Tensor:
    """
    patch_dist_1d에서 threshold 넘는 patch들 중
    connected component 크기가 min_area 미만인 blob 제거.

    return:
        keep_mask: (P,) bool tensor
    """
    P = int(patch_dist_1d.numel())
    Hp, Wp = _infer_patch_grid(P)

    score_map = patch_dist_1d.detach().cpu().numpy().reshape(Hp, Wp)
    mask = score_map > patch_thr

    labeled, n = ndimage.label(mask)
    keep = np.zeros_like(mask, dtype=bool)

    for i in range(1, n + 1):
        comp = (labeled == i)
        if comp.sum() >= min_area:
            keep |= comp

    keep_mask = torch.from_numpy(keep.reshape(-1)).to(patch_dist_1d.device)
    return keep_mask


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


USE_SMALL_BLOB_FILTER = False
BLOB_MIN_AREA = 3


def _dist_patchcore(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
    k: int = 3,
):
    q = F.normalize(q_patch, dim=1)
    # ref = F.normalize(ref_patch, dim=2) # bank쪽에서 한번에 정규화
    ref = ref_patch

    # (N,Pq,Pr)
    sim = torch.einsum("qd,npd->nqp", q, ref)
    max_sim = sim.max(dim=2).values       # (N,Pq)
    dist = 1.0 - max_sim                  # (N,Pq)

    Pq = dist.shape[1]
    m = max(1, int(Pq * top_p))

    # ref image별 상위 top_p patch 평균
    top_vals_all, _ = torch.topk(dist, k=m, dim=1)   # (N,m)
    score_per_img = top_vals_all.mean(dim=1)         # (N,)

    k2 = min(k, score_per_img.shape[0])
    best_val, best_idx = torch.topk(-score_per_img, k=k2)
    topk_score = -best_val                           # (k,)

    # 가장 좋은 ref image 1개 기준
    best_img_idx = best_idx[0]                       # scalar
    best_patch_dist = dist[best_img_idx]             # (Pq,)

    keep_mask = None

    if USE_SMALL_BLOB_FILTER:
        # 기존 top-p 후보의 하한값을 patch threshold로 사용
        raw_top_patch_vals, _ = torch.topk(best_patch_dist, k=m)
        patch_thr = float(raw_top_patch_vals[-1].item())

        keep_mask = _remove_small_blob_from_patch_dist(
            best_patch_dist,
            patch_thr=patch_thr,
            min_area=BLOB_MIN_AREA,
        )

        kept_patch_dist = best_patch_dist[keep_mask]

        if kept_patch_dist.numel() > 0:
            m_kept = max(1, int(kept_patch_dist.numel() * top_p))
            top_patch_vals, top_local_idx = torch.topk(kept_patch_dist, k=m_kept)

            kept_orig_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
            top_patch_idx = kept_orig_idx[top_local_idx]

            score = top_patch_vals.mean().item()
        else:
            top_patch_vals = torch.empty(0, device=best_patch_dist.device)
            top_patch_idx = torch.empty(0, dtype=torch.long, device=best_patch_dist.device)
            score = 0.0

    else:
        top_patch_vals, top_patch_idx = torch.topk(best_patch_dist, k=m)
        score = topk_score.mean().item()

    debug = {
        "topk_score": topk_score,            # (k,)
        "topk_idx": best_idx,               # (k,)
        "best_img_idx": best_img_idx,       # scalar
        "best_patch_dist": best_patch_dist, # (Pq,)
        "top_patch_idx": top_patch_idx,     # (m') filtering 후 달라질 수 있음
        "top_patch_vals": top_patch_vals,   # (m')
        "keep_mask": keep_mask,             # (Pq,) or None
    }
    return score, debug

def _dist_patchcore_masked_local(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    valid_mask_1d,              # (P,) bool
    top_p: float = 0.1,         
    k: int = 1,                 # 유지
    radius: int = 0,
):
    q = F.normalize(q_patch, dim=1)   # (P,D)
    ref = ref_patch                   # (N,P,D)

    valid_mask_1d = torch.as_tensor(valid_mask_1d, device=q.device, dtype=torch.bool)

    if valid_mask_1d.numel() != q.shape[0]:
        raise ValueError(f"valid_mask size mismatch: {valid_mask_1d.numel()} vs {q.shape[0]}")

    valid_count = int(valid_mask_1d.sum().item())
    if valid_count < 4:
        return None, {"ok": False, "reason": "too_few_valid_patches"}

    N, P, D = ref.shape
    Hp, Wp = _infer_patch_grid(P)

    q2 = q.view(Hp, Wp, D)
    ref2 = ref.view(N, Hp, Wp, D)
    valid2 = valid_mask_1d.view(Hp, Wp)

    best_dist_list = []
    valid_orig_idx = []

    # --------------------------------------------------
    # 1) valid patch들에 대해 local-window distance 계산
    # --------------------------------------------------
    for y in range(Hp):
        for x in range(Wp):
            if not valid2[y, x]:
                continue

            qv = q2[y, x]

            y0 = max(0, y - radius)
            y1 = min(Hp, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(Wp, x + radius + 1)

            ref_win = ref2[:, y0:y1, x0:x1, :].reshape(N, -1, D)  # (N,L,D)

            sim = torch.einsum("d,nld->nl", qv, ref_win)   # (N,L)
            max_sim = sim.max(dim=1).values                # (N,)
            dist = 1.0 - max_sim                           # (N,)

            best_dist_list.append(dist)
            valid_orig_idx.append(y * Wp + x)

    if len(best_dist_list) < 4:
        return None, {"ok": False, "reason": "too_few_valid_patches_after_local"}

    dist_map = torch.stack(best_dist_list, dim=1)   # (N, Pv)
    Pv = dist_map.shape[1]
    valid_orig_idx_t = torch.tensor(valid_orig_idx, device=q.device, dtype=torch.long)

    # --------------------------------------------------
    # 2) ref별 top-p + peak-relative threshold score 계산
    # --------------------------------------------------
    alpha = ALPHA

    score_per_img = []
    per_img_peak = []
    per_img_area = []
    per_img_top_patch_idx = []
    per_img_top_patch_vals = []
    per_img_best_patch_dist_full = []

    for n in range(N):
        patch_dist_valid = dist_map[n]  # (Pv,)

        # valid patch만 있는 1D -> full patch grid로 복원
        full_patch_dist = torch.zeros(P, device=q.device, dtype=patch_dist_valid.dtype)
        full_patch_dist[valid_orig_idx_t] = patch_dist_valid
        patch_dist_2d = full_patch_dist.view(Hp, Wp)

        # invalid 제외
        candidate_map = patch_dist_2d.clone()
        candidate_map[~valid2] = 0.0

        peak_val = float(candidate_map.max().item())

        # debug용 full map 저장
        per_img_best_patch_dist_full.append(full_patch_dist)

        # top-p
        k_top = max(1, int(np.ceil(Pv * top_p)))
        top_vals, top_local_idx = torch.topk(
            patch_dist_valid,
            k=min(k_top, patch_dist_valid.numel())
        )
        top_idx = valid_orig_idx_t[top_local_idx]

        if top_vals.numel() == 0:
            score_one = 0.0
            kept_vals = top_vals
            kept_idx = top_idx
            selected_area = 0
            peak_val = 0.0
        else:
            peak_val = float(top_vals.max().item())

            # peak-relative threshold
            keep = top_vals >= (alpha * peak_val)

            kept_vals = top_vals[keep]
            kept_idx = top_idx[keep]

            if kept_vals.numel() == 0:
                kept_vals = top_vals[:1]
                kept_idx = top_idx[:1]

            score_one = float(kept_vals.mean().item())
            selected_area = int(kept_vals.numel())

        score_per_img.append(score_one)
        per_img_peak.append(peak_val)
        per_img_area.append(selected_area)

        # debug용 저장: threshold 후 남은 patch만 넣기
        per_img_top_patch_idx.append(kept_idx)
        per_img_top_patch_vals.append(kept_vals)

    score_per_img = torch.tensor(score_per_img, device=q.device, dtype=torch.float32)

    # 가장 정상에 가까운(ref score 최소) ref 선택
    best_img_idx = torch.argmin(score_per_img)
    score = float(score_per_img[best_img_idx].item())

    best_patch_dist_full = per_img_best_patch_dist_full[int(best_img_idx.item())]
    top_patch_idx = per_img_top_patch_idx[int(best_img_idx.item())]
    top_patch_vals = per_img_top_patch_vals[int(best_img_idx.item())]

    debug = {
        "ok": True,
        "reason": "peak_connected_component",
        "best_img_idx": best_img_idx,
        "score_per_img": score_per_img,
        "best_patch_dist": best_patch_dist_full,   # (P,)
        "top_patch_idx": top_patch_idx,
        "top_patch_vals": top_patch_vals,
        "valid_patch_count": valid_count,
        "local_radius": radius,
        "top_p": top_p,
        "alpha": alpha,
        "selected_patch_count": int(len(top_patch_vals)),
        "score_mean": float(top_patch_vals.mean().item()) if top_patch_vals.numel() > 0 else 0.0,
        "score_peak": float(top_patch_vals.max().item()) if top_patch_vals.numel() > 0 else 0.0,
        "best_component_peak": float(per_img_peak[int(best_img_idx.item())]),
        "best_component_area": int(per_img_area[int(best_img_idx.item())]),
    }
    return score, debug

def _dist_patchcore_local(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
    k: int = 3,
    radius: int = 1,
):
    q = F.normalize(q_patch, dim=1)   # (P,D)
    ref = ref_patch                   # (N,P,D)

    N, P, D = ref.shape
    Hp, Wp = _infer_patch_grid(P)

    q2 = q.view(Hp, Wp, D)
    ref2 = ref.view(N, Hp, Wp, D)

    best_dist_list = []

    for y in range(Hp):
        for x in range(Wp):
            qv = q2[y, x]  # (D,)

            y0 = max(0, y - radius)
            y1 = min(Hp, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(Wp, x + radius + 1)

            # (N, wy, wx, D) -> (N, L, D)
            ref_win = ref2[:, y0:y1, x0:x1, :].reshape(N, -1, D)

            # qv 와 각 ref image의 local window patch 유사도
            # sim: (N, L)
            sim = torch.einsum("d,nld->nl", qv, ref_win)
            max_sim = sim.max(dim=1).values   # (N,)
            dist = 1.0 - max_sim              # (N,)

            best_dist_list.append(dist)

    # (N,P)
    dist_map = torch.stack(best_dist_list, dim=1)

    Pq = dist_map.shape[1]
    m = max(1, int(Pq * top_p))

    top_vals_all, _ = torch.topk(dist_map, k=m, dim=1)   # (N,m)
    score_per_img = top_vals_all.mean(dim=1)             # (N,)

    k2 = min(k, score_per_img.shape[0])
    best_val, best_idx = torch.topk(-score_per_img, k=k2)
    topk_score = -best_val

    best_img_idx = best_idx[0]
    best_patch_dist = dist_map[best_img_idx]             # (P,)

    top_patch_vals, top_patch_idx = torch.topk(best_patch_dist, k=m)
    score = topk_score.mean().item()

    debug = {
        "topk_score": topk_score,
        "topk_idx": best_idx,
        "best_img_idx": best_img_idx,
        "best_patch_dist": best_patch_dist,
        "top_patch_idx": top_patch_idx,
        "top_patch_vals": top_patch_vals,
        "local_radius": radius,
    }
    return score, debug


def _dist_patch_pool(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
):
    q = F.normalize(q_patch, dim=1)          # (P, D)
    # ref = F.normalize(ref_patch, dim=2)    # bank쪽에서 한번에 정규화
    ref = ref_patch

    N, Pr, D = ref.shape
    Pq = q.shape[0]

    # (N*Pr, D)
    ref_pool = ref.reshape(N * Pr, D)

    # (Pq, N*Pr)
    sim = q @ ref_pool.T

    # 각 query patch마다 가장 유사한 ref patch 선택
    max_sim, nn_flat_idx = sim.max(dim=1)    # (Pq,), (Pq,)
    nn_dist = 1.0 - max_sim                  # (Pq,)

    # flat idx -> (image idx, patch idx)
    nn_img_idx = torch.div(nn_flat_idx, Pr, rounding_mode="floor")
    nn_patch_idx = nn_flat_idx % Pr

    m = max(1, int(Pq * top_p))
    top_vals, top_patch_idx = torch.topk(nn_dist, k=m)

    score = top_vals.mean().item()

    debug = {
        "top_vals": top_vals,
        "top_patch_idx": top_patch_idx,
        "nn_img_idx": nn_img_idx,
        "nn_patch_idx": nn_patch_idx,
        "nn_dist": nn_dist,
    }
    return score, debug


def patch_dist_to_grid_scores(
    patch_dist_1d: torch.Tensor,   # (P,)
    grid_rows: int = 4,
    grid_cols: int = 4,
    pool: str = "topk_mean",
    top_ratio: float = 0.2,
):
    """
    return:
        patch_map: (Hp, Wp) torch.Tensor
        grid_scores: (grid_rows, grid_cols) torch.Tensor
    """
    P = int(patch_dist_1d.numel())
    Hp, Wp = _infer_patch_grid(P)

    patch_map = patch_dist_1d.view(Hp, Wp)

    # grid 경계
    y_bins = np.linspace(0, Hp, grid_rows + 1, dtype=int)
    x_bins = np.linspace(0, Wp, grid_cols + 1, dtype=int)

    out = torch.zeros((grid_rows, grid_cols), device=patch_dist_1d.device, dtype=patch_dist_1d.dtype)

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            y0, y1 = y_bins[gy], y_bins[gy + 1]
            x0, x1 = x_bins[gx], x_bins[gx + 1]

            cell = patch_map[y0:y1, x0:x1].reshape(-1)
            if cell.numel() == 0:
                out[gy, gx] = 0.0
                continue

            if pool == "mean":
                score = cell.mean()

            elif pool == "max":
                score = cell.max()

            elif pool == "topk_mean":
                m = max(1, int(cell.numel() * top_ratio))
                vals, _ = torch.topk(cell, k=m)
                score = vals.mean()

            else:
                raise ValueError(f"Unknown pool: {pool}")

            out[gy, gx] = score

    return patch_map, out


DEBUG = True

# ------------------------------------------------------------------------------------------------------------
# 각 모드에 맞추어 knn dist를 return
def compute_knn_dist(
    q_out, # q에서 나온 임베딩
    ref_bank, # bank(npz) 임베딩
    k=3,
    repr_mode="global",
    top_p=0.1,
    preselect_m=10,
    q_img_bgr=None,
    model=None,
    device=None,
    tfm=None,
    global_mode="patch_mean",
    sg_matcher=None,
    cfg=None,
):
    timing = {}

    # global emb -------------------------
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

    # patch -------------------------
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
            )

        debug["timing"] = _stats_to_float_dict(timing)
        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return score, debug, ref_paths

    # global + patch -------------------------
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
            )

        topk_idx_global = idx[patch_debug_local["topk_idx"]]
        best_img_idx_global = idx[patch_debug_local["best_img_idx"]]

        debug = {
            "global_topk": (topk_sim, idx),
            "patch_topk": {
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

    # global + patch pooled -------------------------
    elif repr_mode == "global_patch_pool":
        with _timer(timing, "gpp.refg_to_gpu", q_out["global"].device):
            refg_np, ref_paths = ref_bank["global"]
            refp_np, _ = ref_bank["patch"]

            qg = q_out["global"]
            qp = q_out["patch"]

            refg = torch.from_numpy(refg_np).float().to(qg.device)

        with _timer(timing, "gpp.global_preselect", qg.device):
            _, (topk_sim, topk_idx_global) = _dist_global(
                qg,
                refg,
                min(preselect_m, refg.shape[0]),
            )

            idx = topk_idx_global.squeeze(0)   # preselected image indices

        with _timer(timing, "gpp.refp_to_gpu", qp.device):
            refp = torch.from_numpy(refp_np).float().to(qp.device)
            refp_sel = refp[idx]               # (M,P,D)

        with _timer(timing, "gpp.pool_dist", qp.device):
            score, pool_debug = _dist_patch_pool(
                qp,
                refp_sel,
                top_p=top_p,
            )

        # 어떤 preselected image가 query patch nearest로 많이 뽑혔는지 집계
        nn_img_idx_local = pool_debug["nn_img_idx"]      # 0 ~ M-1
        M = refp_sel.shape[0]
        with _timer(timing, "gpp.vote_topk", qp.device):
            votes = torch.bincount(nn_img_idx_local, minlength=M)
            k2 = min(k, M)
            top_vote_vals, top_vote_idx_local = torch.topk(votes, k=k2)
            topk_idx = idx[top_vote_idx_local]   # 원본 ref image index로 복원

        debug = {
            "global_topk": (topk_sim, idx),
            "pool_topk": {
                "top_vote_vals": top_vote_vals,
                "topk_idx": topk_idx,
            },
            "pool_debug": pool_debug,
            "timing": _stats_to_float_dict(timing),
        }

        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return score, debug, ref_paths

    # global + patch with aligned logic -------------------------
    elif repr_mode == "global_patch_with_aligned":

        if sg_matcher is None:
            raise ValueError("sg_matcher is required for global_patch_with_aligned")
        if q_img_bgr is None:
            raise ValueError("q_img_bgr is required for global_patch_with_aligned")
        if tfm is None:
            raise ValueError("tfm is required for global_patch_with_aligned")
        if model is None or device is None:
            raise ValueError("model/device are required for global_patch_with_aligned")
        if cfg is None:
            raise ValueError("cfg is required for global_patch_with_aligned")

        sg_raw = cfg["superglue"]

        refg_np, ref_paths = ref_bank["global"]
        qg = q_out["global"]

        refg = torch.from_numpy(refg_np).float().to(qg.device)

        # 1) coarse global 탐색
        _, (topk_sim, topk_idx) = _dist_global(
            qg,
            refg,
            min(preselect_m, refg.shape[0]),
        )
        idx = topk_idx.squeeze(0)

        candidates = []

        # 2 ----- coarse 후보에 대해 정합 + crop + emb 추출
        for ref_i in idx.tolist():
            ref_img_path = ref_paths[ref_i]
            ref_img_bgr = cv2.imread(str(ref_img_path), cv2.IMREAD_COLOR)
            if ref_img_bgr is None:
                continue

            match_out = sg_matcher.match_and_estimate(q_img_bgr, ref_img_bgr)
            if not match_out["ok"]:
                continue

            # 3 ---- query -> bank warp & 공통 ROI bbox 영역 
            warped_q_bgr, warped_mask = warp_query_to_bank(
                q_img_bgr,
                match_out["H"],
                bank_hw=ref_img_bgr.shape[:2],
            )

            warped_q_crop, ref_crop, mask_crop, crop_bbox = crop_common_valid_region(
                warped_q_bgr,
                ref_img_bgr,
                warped_mask,
                margin=8,
                min_size=64,
            )

            if DEBUG and warped_q_crop is not None:
                print("[DEBUG] entered save block", ref_i, warped_q_crop.shape)
                dbg_dir = f"./debug_out/dbg_pair_{ref_i}"
                os.makedirs(dbg_dir, exist_ok=True)

                cv2.imwrite(f"{dbg_dir}/01_warped_q.jpg", warped_q_bgr)
                cv2.imwrite(f"{dbg_dir}/02_ref.jpg", ref_img_bgr)
                cv2.imwrite(f"{dbg_dir}/03_crop_q.jpg", warped_q_crop)
                cv2.imwrite(f"{dbg_dir}/04_crop_ref.jpg", ref_crop)
                cv2.imwrite(f"{dbg_dir}/05_mask.jpg", mask_crop)

            if warped_q_crop is None:
                continue

            # 5) crop된 query / ref를 patch 임베딩
            x_q_crop = tfm(BGR_to_RGB(warped_q_crop))

            qp_crop = extract_patch_layers(model,device,x_q_crop)

            """
            q_crop_out = make_embed(
                model,
                device,
                x_q_crop,
                repr_mode="patch",
                global_mode=global_mode,
            )
            qp_crop = q_crop_out["patch"]  # (P2, D)
            """

            if DEBUG and warped_q_crop is not None:
                x_dbg = x_q_crop
                dbg = (x_dbg.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                dbg = cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{dbg_dir}/06_dino_input.jpg", dbg)

            x_ref_crop = tfm(BGR_to_RGB(ref_crop))
            refp_crop = extract_patch_layers(model,device,x_ref_crop)

            """
            ref_crop_out = make_embed(
                model,
                device,
                x_ref_crop,
                repr_mode="patch",
                global_mode=global_mode,
            )
            refp_crop = ref_crop_out["patch"].unsqueeze(0)  # (1, P2, D)
            """

            P2 = qp_crop.shape[0]
            grid_h2, grid_w2 = _infer_patch_grid(P2)

            # 6) crop된 mask 기준 patch valid mask
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

            # 7) masked local patch 비교
            score_one, dbg_one = _dist_patchcore_masked_local(
                qp_crop,
                refp_crop,
                valid_mask_1d=patch_valid_1d,
                top_p=top_p,
                k=1,
                radius=LOCAL_PATCH_RADIUS,
            )

            if score_one is None or not dbg_one.get("ok", True):
                continue

            candidates.append({
                "ref_i": ref_i,
                "score": float(score_one),
                "match_out": match_out,
                "patch_debug": dbg_one,
                "ref_img_path": str(ref_img_path),
                "crop_bbox": crop_bbox,
                "valid_patch_count": valid_patch_count,
                "cropped": True,
            })

        # 정합 성공 후보가 없으면 기존 global_patch fallback
        if len(candidates) == 0:
            refp_np, _ = ref_bank["patch"]
            qp = q_out["patch"]
            refp = torch.from_numpy(refp_np).float().to(qp.device)
            refp_sel = refp[idx]

            score, patch_debug_local = _dist_patchcore(
                qp,
                refp_sel,
                top_p=top_p,
                k=min(k, refp_sel.shape[0]),
            )

            topk_idx_global = idx[patch_debug_local["topk_idx"]]
            best_img_idx_global = idx[patch_debug_local["best_img_idx"]]

            debug = {
                "fallback": True,
                "global_topk": (topk_sim, idx),
                "patch_topk": {
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

        # 8) distance는 작을수록 좋으므로 min
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
                "topk_score": topk_score,
                "topk_idx": topk_idx_global,
                "best_img_idx": torch.tensor(best["ref_i"], device=qg.device),
                "best_patch_dist": best_patch_debug["best_patch_dist"],
                "top_patch_idx": best_patch_debug["top_patch_idx"],
                "top_patch_vals": best_patch_debug["top_patch_vals"],
            },
            "align_debug": {
                "best_ref_img_path": best["ref_img_path"],
                "crop_bbox": best["crop_bbox"],
                "valid_patch_count": best["valid_patch_count"],
                "inliers": best_match["inliers"],
                "inlier_ratio": best_match["inlier_ratio"],
                "reproj_error_mean": best_match["reproj_error_mean"],
                "reproj_error_median": best_match["reproj_error_median"],
            },
            "timing": _stats_to_float_dict(timing),
        }

        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return score, debug, ref_paths

    else:
        raise ValueError("repr_mode must be global|patch|global_patch|global_patch_pool|global_patch_with_aligned")


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


@torch.inference_mode()
def compute_and_save_threshold(
    bank_root,
    plc_idx,
    cfg: dict,
    model=None,
    device=None,
    sg_matcher=None,
):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    total_timing = {}

    r = cfg.get("repr", {})
    c = cfg.get("calib", {})

    repr_mode = str(r.get("repr_mode", "global"))

    k          = int(c.get("k", 3))
    percentile = int(c.get("percentile", 97))
    method     = str(c.get("method", "robust"))
    robust_k   = float(c.get("robust_k", 2.5))
    gaussian_k = float(c.get("gaussian_k", 2.5))

    # hash는 global bank 기준 유지
    bank_hash_src = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
    with open(bank_hash_src, "rb") as f:
        bank_hash = hashlib.sha1(f.read()).hexdigest()[:8]

    # ---- load th_calib/bank ----
    if repr_mode == "global":
        with _timer(total_timing, "calib.global.load_npz", device):
            bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
            th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
            bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()
            th   = torch.from_numpy(np.load(th_npz, allow_pickle=True)["embs"]).float()
            # bank = F.normalize(bank, dim=1)
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

            bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()  # (N,P,D)
            th   = torch.from_numpy(np.load(th_npz,   allow_pickle=True)["embs"]).float()  # (T,P,D)

            top_p = float(cfg.get("patchcore", {}).get("top_p", 0.1))

        scores = []
        for i in range(th.shape[0]):
            with _timer(total_timing, "calib.patch.per_th_dist", device):
                q_patch = th[i]  # (P,D)
                s, _ = _dist_patchcore(q_patch, bank, top_p=top_p, k=k)
                scores.append(s)
        scores = np.array(scores, dtype=np.float32)

    elif repr_mode == "global_patch":
        with _timer(total_timing, "calib.gp.load_npz", device):
            # threshold는 "최종 score=patchcore score" 기준으로 잡는 게 깔끔함
            bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
            th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
            bank_p_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
            th_p_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

            bank_g = torch.from_numpy(np.load(bank_g_npz, allow_pickle=True)["embs"]).float()  # (N,D)
            th_g   = torch.from_numpy(np.load(th_g_npz,   allow_pickle=True)["embs"]).float()  # (T,D)
            bank_p = torch.from_numpy(np.load(bank_p_npz, allow_pickle=True)["embs"]).float()  # (N,P,D)
            th_p   = torch.from_numpy(np.load(th_p_npz,   allow_pickle=True)["embs"]).float()  # (T,P,D)

            # bank_g = F.normalize(bank_g, dim=1)
            th_g = F.normalize(th_g, dim=1)

            pcfg = cfg.get("patchcore", {})
            top_p = float(pcfg.get("top_p", 0.1))
            preselect_m = int(pcfg.get("preselect_m", 10))

        scores = []
        for i in range(th_g.shape[0]):
            with _timer(total_timing, "calib.gp.per_th_global_preselect", device):
                sim = th_g[i:i+1] @ bank_g.T
                M = min(preselect_m, bank_g.shape[0])
                _, idx = torch.topk(sim, k=M, dim=1)
                idx = idx.squeeze(0)

            with _timer(total_timing, "calib.gp.per_th_patch_dist", device):
                bank_sel = bank_p[idx]  # (M,P,D)
                s, _ = _dist_patchcore(th_p[i], bank_sel, top_p=top_p, k=k)
                scores.append(s)

        scores = np.array(scores, dtype=np.float32)

    elif repr_mode == "global_patch_pool":
        with _timer(total_timing, "calib.gpp.load_npz", device):
            bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
            th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
            bank_p_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
            th_p_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

            bank_g = torch.from_numpy(np.load(bank_g_npz, allow_pickle=True)["embs"]).float()  # (N,D)
            th_g   = torch.from_numpy(np.load(th_g_npz,   allow_pickle=True)["embs"]).float()  # (T,D)
            bank_p = torch.from_numpy(np.load(bank_p_npz, allow_pickle=True)["embs"]).float()  # (N,P,D)
            th_p   = torch.from_numpy(np.load(th_p_npz,   allow_pickle=True)["embs"]).float()  # (T,P,D)

            # bank_g = F.normalize(bank_g, dim=1)
            th_g = F.normalize(th_g, dim=1)

            pcfg = cfg.get("patchcore", {})
            top_p = float(pcfg.get("top_p", 0.1))
            preselect_m = int(pcfg.get("preselect_m", 10))

        scores = []
        for i in range(th_g.shape[0]):
            with _timer(total_timing, "calib.gpp.per_th_global_preselect", device):
                sim = th_g[i:i+1] @ bank_g.T
                M = min(preselect_m, bank_g.shape[0])
                _, idx = torch.topk(sim, k=M, dim=1)
                idx = idx.squeeze(0)

            with _timer(total_timing, "calib.gpp.per_th_pool_dist", device):
                bank_sel = bank_p[idx]  # (M,P,D)
                s, _ = _dist_patch_pool(
                    th_p[i],
                    bank_sel,
                    top_p=top_p,
                )
                scores.append(s)

        scores = np.array(scores, dtype=np.float32)

    elif repr_mode == "global_patch_with_aligned":
        if model is None or device is None or sg_matcher is None:
            raise ValueError("model/device/sg_matcher are required for global_patch_with_aligned")

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

            pcfg = cfg.get("patchcore", {})
            top_p = float(pcfg.get("top_p", 0.1))
            preselect_m = int(pcfg.get("preselect_m", 10))

            e = cfg.get("embed", {})
            img_size = int(e.get("img_size", 560))
            global_mode = str(e.get("global_mode", "patch_mean"))
            tfm = make_transform(img_size=img_size)

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
                    model,
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

                    warped_q_crop, ref_crop, mask_crop, crop_bbox = crop_common_valid_region(
                        warped_q_bgr,
                        ref_img_bgr,
                        warped_mask,
                        margin=8,
                        min_size=64,
                    )
                    if warped_q_crop is None:
                        continue

                    """        
                    x_q_crop = tfm(BGR_to_RGB(warped_q_crop))
                    q_crop_out = make_embed(
                        model,
                        device,
                        x_q_crop,
                        repr_mode="patch",
                        global_mode=global_mode,
                    )
                    qp_crop = q_crop_out["patch"]

                    x_ref_crop = tfm(BGR_to_RGB(ref_crop))
                    ref_crop_out = make_embed(
                        model,
                        device,
                        x_ref_crop,
                        repr_mode="patch",
                        global_mode=global_mode,
                    )
                    refp_crop = ref_crop_out["patch"].unsqueeze(0)
                    """
                    x_q_crop = tfm(BGR_to_RGB(warped_q_crop))
                    qp_crop = extract_patch_layers(model, device, x_q_crop)   # (P2, D)

                    x_ref_crop = tfm(BGR_to_RGB(ref_crop))
                    refp_crop = extract_patch_layers(model, device, x_ref_crop).unsqueeze(0)  # (1, P2, D)

                    P2 = qp_crop.shape[0]
                    grid_h2, grid_w2 = _infer_patch_grid(P2)

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
                        radius=LOCAL_PATCH_RADIUS,
                    )

                    if score_one is None or not dbg_one.get("ok", True):
                        continue

                    cand_scores.append(float(score_one))

            if len(cand_scores) == 0:
                with _timer(loop_timing, "fallback_global_patch", device):
                    s, _, _ = compute_knn_dist(
                        q_out,
                        ref_bank,
                        k=k,
                        repr_mode="global_patch",
                        top_p=top_p,
                        preselect_m=preselect_m,
                        q_img_bgr=img_bgr,
                        model=model,
                        device=device,
                        tfm=tfm,
                        global_mode=global_mode,
                        sg_matcher=sg_matcher,
                        cfg=cfg,
                    )
            else:
                s = min(cand_scores)

            scores.append(float(s))
            _merge_timing(total_timing, {f"calib.aligned.per_img.{kk}": vv for kk, vv in loop_timing.items()})

        scores = np.array(scores, dtype=np.float32)

    else:
        raise ValueError(f"Unknown repr_mode: {repr_mode}")

    # ---- threshold from scores ----
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


def calibrate_place(bank_root, plc_idx, model, device, sg_matcher=None):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    timing = {}
    cfg = load_cfg(bank_root)

    with _timer(timing, "calibrate.rebuild_bank_bank", device):
        rebuild_bank(bank_root, plc_idx, model, device, mode="bank", cfg=cfg)

    with _timer(timing, "calibrate.rebuild_bank_th", device):
        rebuild_bank(bank_root, plc_idx, model, device, mode="th_calib", cfg=cfg)

    with _timer(timing, "calibrate.compute_and_save_threshold", device):
        thr, scores, thr_path = compute_and_save_threshold(
            bank_root,
            plc_idx,
            cfg=cfg,
            model=model,
            device=device,
            sg_matcher=sg_matcher,
        )

    _print_timing(f"calibrate_place[{plc_idx}]", timing)
    return thr, scores, thr_path


# ------------------------------- 추론

# event 단위 이상감지
@torch.inference_mode()
def infer_event(
    imgs_bgr: List[np.ndarray],
    plc_idx: str,
    bank_root,
    model,
    device,
    sg_matcher=None,
) -> Dict[str, Any]:
    """
    repr_mode: "global" | "patch" | "global_patch" | "global_patch_pool"
    - global: global kNN
    - patch: PatchCore-style (image-level) using patch tokens
    - global_patch: global preselect 후 image-level patchcore
    - global_patch_pool: global preselect 후 pooled patch matching

    return:
      {
        "threshold": float,
        "frame_scores": List[float],
        "anomaly_flag": int,
        "frame_change_flags": List[int],
        "event_score": float,
        "ref_bank_id": str,
        "ref_topk_json": str(json),
        "summary": str,
      }
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
        cfg = load_cfg(bank_root)

    infer_cfg = cfg.get("infer", {})
    event_rule = str(infer_cfg.get("event_rule", "max"))
    use_two_stage_vlm = bool(infer_cfg.get("use_two_stage_vlm", False))

    e = cfg.get("embed", {})
    img_size = int(e.get("img_size", 560))
    global_mode = str(e.get("global_mode", "patch_mean"))
    tfm = make_transform(img_size=img_size)

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
    if repr_mode not in {"global", "patch", "global_patch", "global_patch_pool", "global_patch_with_aligned"}:
        raise ValueError(
            f"Unknown repr_mode={repr_mode} "
            "(use global|patch|global_patch|global_patch_pool|global_patch_with_aligned)"
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
    patch_vis_all: List[Optional[Dict[str, Any]]] = []  # vis
    frame_timing_all: List[Dict[str, float]] = []

    for fi, img_bgr in enumerate(imgs_bgr):
        frame_timing = {}

        with _timer(frame_timing, "tfm", device):
            x = tfm(BGR_to_RGB(img_bgr))

        # embed
        with _timer(frame_timing, "embed", device):
            q_out = make_embed(
                model,
                device,
                x,
                repr_mode=repr_mode,          # global|patch|global_patch
                global_mode=global_mode,
            )

        # dist
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
                model=model,
                device=device,
                tfm=tfm,
                sg_matcher=sg_matcher,
                cfg=cfg,
            )

        if isinstance(debug, dict):
            _merge_timing(frame_timing, {f"knn.{k}": v for k, v in debug.get("timing", {}).items()})

        is_change = dist > thr

        if repr_mode in {"patch", "global_patch"}:
            best_patch_dist = None

            if repr_mode == "patch":
                best_patch_dist = debug.get("best_patch_dist", None)
            else:
                best_patch_dist = debug.get("patch_topk", {}).get("best_patch_dist", None)

            if best_patch_dist is not None:
                _, grid_scores = patch_dist_to_grid_scores(
                    best_patch_dist,
                    grid_rows=4,
                    grid_cols=4,
                    pool="topk_mean",
                    top_ratio=0.3,
                )

                grid_flag = int(grid_scores.max().item() > (thr * 1.15))
                is_change = bool(is_change or grid_flag)

        frame_scores.append(float(dist))
        frame_change_flags.append(1 if is_change else 0)

        # -------------------------
        # topk logging (항상 "이미지 단위 topk")
        # -------------------------
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

        elif repr_mode == "global_patch_pool":
            pool_topk = debug["pool_topk"]
            topk_idx = pool_topk["topk_idx"]
            top_vote_vals = pool_topk["top_vote_vals"]

            idx = topk_idx.tolist()
            paths = [ref_paths[i] for i in idx]
            sims = top_vote_vals.tolist()

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

        # -- vis
        top_patch_idx, top_patch_vals = get_top_p_patch_info(repr_mode, debug)

        if top_patch_idx is not None and top_patch_vals is not None:
            patch_vis_all.append({
                "top_patch_idx": top_patch_idx.detach().cpu().tolist()
                if hasattr(top_patch_idx, "detach") else list(top_patch_idx),
                "top_patch_vals": top_patch_vals.detach().cpu().tolist()
                if hasattr(top_patch_vals, "detach") else list(top_patch_vals),
            })
        else:
            patch_vis_all.append(None)

        frame_timing_all.append(_stats_to_float_dict(frame_timing))
        _print_timing(f"infer_event[{plc_idx}] frame={fi}", frame_timing)
        _merge_timing(total_timing, {f"frame.{k}": v for k, v in frame_timing.items()})

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

        # gui표시용 event 스코어
        event_score = float(np.clip(
            50.0 + 50.0 * (margin / max(thr, 1e-8)),
            0.0,
            100.0
        ))

        rep_idx = int(np.argmax(frame_scores)) if len(frame_scores) > 0 else 0  # vis

    # -------------------------
    # optional: VLM gate
    # -------------------------
    rep, summary = None, ""
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
            "patch_size": 14,
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
        },
        "timing": _stats_to_float_dict(total_timing),
        "frame_timing": frame_timing_all,
    }