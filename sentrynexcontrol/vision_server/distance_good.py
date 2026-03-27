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

from .banker import load_bank_by_place, BGR_to_RGB, rebuild_bank
from .dino_emb import make_embed , make_transform
from .config import load_cfg

import cv2

from .warp_utils import warp_query_to_bank, make_patch_valid_mask

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

#()
from scipy import ndimage
import math

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


#global dist 계산 함수 > 거리와 제일 유사한 k개 유사도, 인덱스 반환
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


"""
#patchcore dist 계산함수
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

    score = topk_score.mean().item()

    # 가장 좋은 ref image 1개 기준으로 patch 시각화용 정보 추출
    best_img_idx = best_idx[0]                       # scalar
    best_patch_dist = dist[best_img_idx]            # (Pq,)
    top_patch_vals, top_patch_idx = torch.topk(best_patch_dist, k=m)

    debug = {
        "topk_score": topk_score,            # (k,)
        "topk_idx": best_idx,               # (k,)
        "best_img_idx": best_img_idx,       # scalar
        "best_patch_dist": best_patch_dist, # (Pq,)
        "top_patch_idx": top_patch_idx,     # (m,)
        "top_patch_vals": top_patch_vals,   # (m,)
    }
    return score, debug
"""

USE_SMALL_BLOB_FILTER = True
BLOB_MIN_AREA = 2

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
            kept_orig_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

            # blob filtering 후 남은 patch 전체를 후보로 사용
            cand_vals = kept_patch_dist
            cand_idx = kept_orig_idx

            patch_thr2 = 0.7 * float(cand_vals.max().item())
            sel_mask = cand_vals >= patch_thr2
            sel_vals = cand_vals[sel_mask]
            sel_idx = cand_idx[sel_mask]

            min_keep = min(3, cand_vals.numel())
            if sel_vals.numel() < min_keep:
                top_vals, top_local_idx = torch.topk(cand_vals, k=min_keep)
                sel_vals = top_vals
                sel_idx = cand_idx[top_local_idx]

            top_patch_vals = sel_vals
            top_patch_idx = sel_idx

            score = top_patch_vals.mean().item()
        else:
            # blob filtering 후 다 사라지면 fallback
            top_patch_vals, top_patch_idx = torch.topk(best_patch_dist, k=m)
            score = top_patch_vals.mean().item()

    else:

        # topk > peak기반(max의 0.8만 남기고 평균)

        top_patch_vals, top_patch_idx = torch.topk(best_patch_dist, k=m)
        patch_thr2 = 0.7 * float(top_patch_vals.max().item())
        sel_mask = top_patch_vals >= patch_thr2
        sel_vals = top_patch_vals[sel_mask]
        sel_idx = top_patch_idx[sel_mask]

        if sel_vals.numel() == 0:
            sel_vals = top_patch_vals[:1]
            sel_idx = top_patch_idx[:1]

        top_patch_vals = sel_vals
        top_patch_idx = sel_idx

        score = top_patch_vals.mean().item()

    debug = {
        "topk_score": topk_score,            # multi-ref 형식 유지
        "topk_idx": best_idx,
        "best_img_idx": best_img_idx,
        "best_patch_dist": best_patch_dist,
        "top_patch_idx": top_patch_idx,
        "top_patch_vals": top_patch_vals,
        "keep_mask": keep_mask,
    }
    return score, debug


def _dist_patchcore_local(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
    k: int = 3,
    radius: int = 2,
    rel_thr: float = 0.8,
):
    q = F.normalize(q_patch, dim=1)
    ref = ref_patch

    N, P, D = ref.shape
    Hp, Wp = _infer_patch_grid(P)

    q2 = q.view(Hp, Wp, D)
    ref2 = ref.view(N, Hp, Wp, D)

    best_dist_list = []

    # --- local matching
    for y in range(Hp):
        for x in range(Wp):
            qv = q2[y, x]  # (D,)

            y0 = max(0, y - radius)
            y1 = min(Hp, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(Wp, x + radius + 1)

            ref_local = ref2[:, y0:y1, x0:x1, :]     # (N,h,w,D)
            ref_local = ref_local.reshape(N, -1, D)  # (N,L,D)

            sim_local = torch.einsum("d,nld->nl", qv, ref_local)  # (N,L)
            max_sim = sim_local.max(dim=1).values                # (N,)
            dist_local = 1.0 - max_sim                           # (N,)

            best_dist_list.append(dist_local)

    # (N,P)
    dist = torch.stack(best_dist_list, dim=1)

    # --- ref별 score 계산
    Pq = dist.shape[1]
    m = max(1, int(Pq * top_p))

    top_vals_all, _ = torch.topk(dist, k=m, dim=1)
    score_per_img = top_vals_all.mean(dim=1)

    # --- ref top-k 선택
    k2 = min(k, score_per_img.shape[0])
    best_val, best_idx = torch.topk(score_per_img, k=k2)  
    topk_score = best_val

    best_img_idx = best_idx[0]
    best_patch_dist = dist[best_img_idx]

    # --- patch threshold cut (핵심)
    top_patch_vals, top_patch_idx = torch.topk(best_patch_dist, k=m)
    score = topk_score.mean().item()

    debug = {
        "topk_score": topk_score,
        "topk_idx": best_idx,
        "best_img_idx": best_img_idx,
        "best_patch_dist": best_patch_dist,
        "top_patch_idx": top_patch_idx,
        "top_patch_vals": top_patch_vals,
    }

    return score, debug



def _infer_patch_grid(P: int):
    s = int(P ** 0.5)
    if s * s != P:
        raise ValueError(f"Cannot infer square patch grid from P={P}")
    return s, s


def _dist_local_patchcore_masked(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (1,P,D)
    valid_mask_1d,              # (P,) bool
    top_p: float = 0.1,
    k: int = 1,
    radius: int = 0,            # 정합 후에는 1:1에 가깝게
    rel_thr: float = 0.8,       # hard threshold cut
):
    q = F.normalize(q_patch, dim=1)
    ref = ref_patch

    valid_mask_1d = torch.as_tensor(valid_mask_1d, device=q.device, dtype=torch.bool)
    if valid_mask_1d.numel() != q.shape[0]:
        raise ValueError(f"valid_mask size mismatch: {valid_mask_1d.numel()} vs {q.shape[0]}")

    valid_count = int(valid_mask_1d.sum().item())
    if valid_count < 4:
        return None, {"ok": False, "reason": "too_few_valid_patches"}

    valid_orig_idx = torch.nonzero(valid_mask_1d, as_tuple=False).squeeze(1)

    P, D = q.shape
    N = ref.shape[0]
    if N != 1:
        raise ValueError(f"_dist_local_patchcore_masked expects ref_patch.shape[0] == 1, got N={N}")

    Hp, Wp = _infer_patch_grid(P)

    q2 = q.view(Hp, Wp, D)
    ref2 = ref.view(N, Hp, Wp, D)

    best_dist_list = []

    for orig_idx in valid_orig_idx.tolist():
        y = orig_idx // Wp
        x = orig_idx % Wp

        qv = q2[y, x]

        y0 = max(0, y - radius)
        y1 = min(Hp, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(Wp, x + radius + 1)

        ref_local = ref2[:, y0:y1, x0:x1, :]
        local_h, local_w = ref_local.shape[1], ref_local.shape[2]
        ref_local = ref_local.reshape(N, local_h * local_w, D)

        sim_local = torch.einsum("d,nld->nl", qv, ref_local)   # (1,L)
        max_sim_local = sim_local.max(dim=1).values            # (1,)
        dist_local = 1.0 - max_sim_local                       # (1,)

        best_dist_list.append(dist_local)

    dist = torch.stack(best_dist_list, dim=1)   # (1,Pv)

    Pv = dist.shape[1]
    m = max(1, int(Pv * top_p))

    best_patch_dist_valid = dist[0]  # (Pv,)
    top_patch_vals, top_patch_idx_local = torch.topk(best_patch_dist_valid, k=m)
    top_patch_idx = valid_orig_idx[top_patch_idx_local]

    # hard threshold cut
    patch_thr = rel_thr * float(top_patch_vals.max().item())
    sel_mask = top_patch_vals >= patch_thr
    sel_vals = top_patch_vals[sel_mask]
    sel_idx_local = top_patch_idx_local[sel_mask]

    if sel_vals.numel() == 0:
        sel_vals = top_patch_vals[:1]
        sel_idx_local = top_patch_idx_local[:1]

    sel_patch_idx = valid_orig_idx[sel_idx_local]

    score_tensor = sel_vals.mean().detach()
    score = score_tensor.item()

    # debug 형식 유지
    topk_score = score_tensor.unsqueeze(0)  # (1,)
    best_idx = torch.tensor([0], device=score_tensor.device, dtype=torch.long)
    best_img_idx = 0

    debug = {
        "ok": True,
        "topk_score": topk_score,
        "topk_idx": best_idx,
        "best_img_idx": best_img_idx,
        "best_patch_dist": best_patch_dist_valid,
        "top_patch_idx": sel_patch_idx,
        "top_patch_vals": sel_vals,
        "valid_patch_count": valid_count,
    }
    return score, debug



def _dist_patch_pool(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
):

    q = F.normalize(q_patch, dim=1)          # (P, D)
    #ref = F.normalize(ref_patch, dim=2)      # (N, P, D) # bank쪽에서 한번에 정규화
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


# ------------------------------------------------------------------------------------------------------------
#각 모드에 맞추어 knn dist를 return 
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
    thr = None
):

    # global emb -------------------------

    if repr_mode == "global":

        ref_embs_np, ref_paths = ref_bank["global"]
        q = q_out["global"]
        ref = torch.from_numpy(ref_embs_np).float().to(q.device)
        dist, debug = _dist_global(q, ref, k)

        return dist, debug, ref_paths


    # patch -------------------------

    elif repr_mode == "patch":

        ref_patch_np, ref_paths = ref_bank["patch"]
        q_patch = q_out["patch"]
        ref_patch = torch.from_numpy(ref_patch_np).float().to(q_patch.device)

        score, debug = _dist_patchcore(
            q_patch,
            ref_patch,
            top_p=top_p,
            k=k,
        )

        return score, debug, ref_paths


    # global + patch -------------------------

    elif repr_mode == "global_patch":

        refg_np, ref_paths = ref_bank["global"]
        refp_np, _ = ref_bank["patch"]

        qg = q_out["global"]
        qp = q_out["patch"]

        refg = torch.from_numpy(refg_np).float().to(qg.device)

        _, (topk_sim, topk_idx) = _dist_global(
            qg,
            refg,
            min(preselect_m, refg.shape[0]),
        )

        idx = topk_idx.squeeze(0)

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
            "global_topk": (topk_sim, idx),
            "patch_topk": {
                "topk_score": patch_debug_local["topk_score"],
                "topk_idx": topk_idx_global,
                "best_img_idx": best_img_idx_global,
                "best_patch_dist": patch_debug_local["best_patch_dist"],
                "top_patch_idx": patch_debug_local["top_patch_idx"],
                "top_patch_vals": patch_debug_local["top_patch_vals"],
            },
        }

        return score, debug, ref_paths
    
    # global + patch pooled -------------------------

    elif repr_mode == "global_patch_pool":

        refg_np, ref_paths = ref_bank["global"]
        refp_np, _ = ref_bank["patch"]

        qg = q_out["global"]
        qp = q_out["patch"]

        refg = torch.from_numpy(refg_np).float().to(qg.device)

        _, (topk_sim, topk_idx_global) = _dist_global(
            qg,
            refg,
            min(preselect_m, refg.shape[0]),
        )

        idx = topk_idx_global.squeeze(0)   # preselected image indices
        refp = torch.from_numpy(refp_np).float().to(qp.device)

        refp_sel = refp[idx]               # (M,P,D)

        score, pool_debug = _dist_patch_pool(
            qp,
            refp_sel,
            top_p=top_p,
        )

        # 어떤 preselected image가 query patch nearest로 많이 뽑혔는지 집계
        nn_img_idx_local = pool_debug["nn_img_idx"]      # 0 ~ M-1
        M = refp_sel.shape[0]
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
        }

        return score, debug, ref_paths
    
    # global + patch with aligned logic -------------------------

    elif repr_mode == "global_patch_with_aligned":

        if sg_matcher is None:
            raise ValueError("sg_matcher is required for global_patch filtering")
        if q_img_bgr is None:
            raise ValueError("q_img_bgr is required for global_patch filtering")

        refg_np, ref_paths = ref_bank["global"]
        refp_np, _ = ref_bank["patch"]

        qg = q_out["global"]
        qp = q_out["patch"]

        refg = torch.from_numpy(refg_np).float().to(qg.device)

        _, (topk_sim, topk_idx) = _dist_global(
            qg,
            refg,
            min(preselect_m, refg.shape[0]),
        )

        idx = topk_idx.squeeze(0)

        refp = torch.from_numpy(refp_np).float().to(qp.device)
        refp_sel = refp[idx]

        # -------------------------
        # 1) 원래 global_patch score 계산
        # -------------------------
        score_raw, patch_debug_local = _dist_patchcore(
            qp,
            refp_sel,
            top_p=top_p,
            k=min(k, refp_sel.shape[0]),
        )

        topk_idx_global = idx[patch_debug_local["topk_idx"]]
        best_img_idx_global = idx[patch_debug_local["best_img_idx"]]

        # patchcore가 실제로 사용한 best ref
        ref_i = int(best_img_idx_global.item())
        ref_img_path = ref_paths[ref_i]
        ref_img_bgr = cv2.imread(str(ref_img_path), cv2.IMREAD_COLOR)

        # 기본 debug
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
        }

        # ref 이미지 로드 실패 시 fallback
        if ref_img_bgr is None:
            debug["align_filter"] = {
                "used": False,
                "reason": "ref_img_load_fail",
                "score_raw": float(score_raw),
                "score_final": float(score_raw),
            }
            return score_raw, debug, ref_paths

        # -------------------------
        # 2) best ref에 대해서만 정합
        # -------------------------
        match_out = sg_matcher.match_and_estimate(q_img_bgr, ref_img_bgr)
        if not match_out["ok"]:
            debug["align_filter"] = {
                "used": False,
                "reason": "match_fail",
                "score_raw": float(score_raw),
                "score_final": float(score_raw),
            }
            return score_raw, debug, ref_paths

        # query -> bank(ref) 좌표계로 warp
        warped_q_bgr, warped_mask = warp_query_to_bank(
            q_img_bgr,
            match_out["H"],
            bank_hw=ref_img_bgr.shape[:2],
        )

        # -------------------------
        # 3) patch_valid_mask 생성
        # -------------------------
        P = qp.shape[0]
        grid_h, grid_w = _infer_patch_grid(P)

        patch_valid_2d = make_patch_valid_mask(
            warped_mask,
            grid_h=grid_h,
            grid_w=grid_w,
            thr=0.8,   
        )

        patch_valid_1d = torch.from_numpy(
            patch_valid_2d.reshape(-1)
        ).to(device=qp.device, dtype=torch.bool)

        valid_ratio = float(patch_valid_1d.float().mean().item())

        # -------------------------
        # 4) 정합 품질이 좋을 때만
        #    top patch 중 invalid patch 제거
        # -------------------------

        use_align_filter = (
            match_out["reproj_error_median"] <= 3.0 and
            valid_ratio >= 0.4
        )

        if use_align_filter:
            best_patch_dist = patch_debug_local["best_patch_dist"]   # (P,)
            top_patch_idx = patch_debug_local["top_patch_idx"]       # (m,)

            top_patch_keep = patch_valid_1d[top_patch_idx]           # (m,)
            filtered_top_patch_idx = top_patch_idx[top_patch_keep]
            filtered_top_patch_vals = best_patch_dist[filtered_top_patch_idx]

            # 전부 날아가면 fallback
            if filtered_top_patch_vals.numel() > 0:
                score = filtered_top_patch_vals.mean().item()
            else:
                filtered_top_patch_idx = top_patch_idx
                filtered_top_patch_vals = patch_debug_local["top_patch_vals"]
                score = score_raw
                use_align_filter = False
                filter_reason = "all_top_patches_removed_fallback"
        else:
            filtered_top_patch_idx = patch_debug_local["top_patch_idx"]
            filtered_top_patch_vals = patch_debug_local["top_patch_vals"]
            score = score_raw
            filter_reason = ""

        # -------------------------
        # 5) debug 갱신
        # -------------------------
        debug = {
            "global_topk": (topk_sim, idx),
            "patch_topk": {
                "topk_score": patch_debug_local["topk_score"],
                "topk_idx": topk_idx_global,
                "best_img_idx": best_img_idx_global,
                "best_patch_dist": patch_debug_local["best_patch_dist"],
                "top_patch_idx": filtered_top_patch_idx,
                "top_patch_vals": filtered_top_patch_vals,
            },
            "align_filter": {
                "used": bool(use_align_filter),
                "reason": "ok" if use_align_filter else filter_reason,
                "best_ref_img_path": str(ref_img_path),
                "inliers": int(match_out["inliers"]),
                "inlier_ratio": float(match_out["inlier_ratio"]),
                "reproj_error_mean": float(match_out["reproj_error_mean"]) if match_out["reproj_error_mean"] is not None else None,
                "reproj_error_median": float(match_out["reproj_error_median"]) if match_out["reproj_error_median"] is not None else None,
                "valid_ratio": float(valid_ratio),
                "num_top_patch_before": int(patch_debug_local["top_patch_idx"].numel()),
                "num_top_patch_after": int(filtered_top_patch_idx.numel()),
                "score_raw": float(score_raw),
                "score_final": float(score),
            },
        }

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

    r = cfg.get("repr", {})
    c = cfg.get("calib", {})

    repr_mode = str(r.get("repr_mode", "global"))

    k         = int(c.get("k", 3))
    percentile= int(c.get("percentile", 97))
    method    = str(c.get("method", "robust"))
    robust_k  = float(c.get("robust_k", 2.5))
    gaussian_k= float(c.get("gaussian_k", 2.5))

    # hash는 global bank 기준 유지
    bank_hash_src = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
    with open(bank_hash_src, "rb") as f:
        bank_hash = hashlib.sha1(f.read()).hexdigest()[:8]

    # ---- load th_calib/bank ----
    if repr_mode == "global":
        bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
        th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
        bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()
        th   = torch.from_numpy(np.load(th_npz, allow_pickle=True)["embs"]).float()
        #bank = F.normalize(bank, dim=1)
        th   = F.normalize(th, dim=1)
        k2 = min(k, bank.shape[0])
        sim = th @ bank.T
        topk_sim, _ = torch.topk(sim, k=k2, dim=1)
        scores = (1.0 - topk_sim).mean(dim=1).numpy()

    elif repr_mode == "patch":
        bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
        th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

        bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()  # (N,P,D)
        th   = torch.from_numpy(np.load(th_npz,   allow_pickle=True)["embs"]).float()  # (T,P,D)

        top_p = float(cfg.get("patchcore", {}).get("top_p", 0.1))  

        scores = []
        for i in range(th.shape[0]):
            q_patch = th[i]  # (P,D)
            s, _ = _dist_patchcore(q_patch, bank, top_p=top_p, k=k)  
            scores.append(s)
        scores = np.array(scores, dtype=np.float32)

    elif repr_mode == "global_patch":
        # threshold는 "최종 score=patchcore score" 기준으로 잡는 게 깔끔함
        bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
        th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
        bank_p_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
        th_p_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

        bank_g = torch.from_numpy(np.load(bank_g_npz, allow_pickle=True)["embs"]).float()  # (N,D)
        th_g   = torch.from_numpy(np.load(th_g_npz,   allow_pickle=True)["embs"]).float()  # (T,D)
        bank_p = torch.from_numpy(np.load(bank_p_npz, allow_pickle=True)["embs"]).float()  # (N,P,D)
        th_p   = torch.from_numpy(np.load(th_p_npz,   allow_pickle=True)["embs"]).float()  # (T,P,D)

        #bank_g = F.normalize(bank_g, dim=1)
        th_g   = F.normalize(th_g, dim=1)

        pcfg = cfg.get("patchcore", {})
        top_p = float(pcfg.get("top_p", 0.1))
        preselect_m = int(pcfg.get("preselect_m", 10))

        scores = []
        for i in range(th_g.shape[0]):
            # global로 후보 이미지 M개 고르고
            sim = th_g[i:i+1] @ bank_g.T
            M = min(preselect_m, bank_g.shape[0])
            _, idx = torch.topk(sim, k=M, dim=1)
            idx = idx.squeeze(0)

            bank_sel = bank_p[idx]  # (M,P,D)

            s, _ = _dist_patchcore(th_p[i], bank_sel, top_p=top_p, k=k)
            scores.append(s)

        scores = np.array(scores, dtype=np.float32)

    elif repr_mode == "global_patch_pool":
        bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
        th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
        bank_p_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
        th_p_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

        bank_g = torch.from_numpy(np.load(bank_g_npz, allow_pickle=True)["embs"]).float()  # (N,D)
        th_g   = torch.from_numpy(np.load(th_g_npz,   allow_pickle=True)["embs"]).float()  # (T,D)
        bank_p = torch.from_numpy(np.load(bank_p_npz, allow_pickle=True)["embs"]).float()  # (N,P,D)
        th_p   = torch.from_numpy(np.load(th_p_npz,   allow_pickle=True)["embs"]).float()  # (T,P,D)

        #bank_g = F.normalize(bank_g, dim=1)
        th_g   = F.normalize(th_g, dim=1)

        pcfg = cfg.get("patchcore", {})
        top_p = float(pcfg.get("top_p", 0.1))
        preselect_m = int(pcfg.get("preselect_m", 10))

        scores = []
        for i in range(th_g.shape[0]):
            sim = th_g[i:i+1] @ bank_g.T
            M = min(preselect_m, bank_g.shape[0])
            _, idx = torch.topk(sim, k=M, dim=1)
            idx = idx.squeeze(0)

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

        bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
        th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
        bank_p_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"

        bank_g = torch.from_numpy(np.load(bank_g_npz, allow_pickle=True)["embs"]).float()
        th_g   = torch.from_numpy(np.load(th_g_npz,   allow_pickle=True)["embs"]).float()
        th_g   = F.normalize(th_g, dim=1)

        pcfg = cfg.get("patchcore", {})
        top_p = float(pcfg.get("top_p", 0.1))
        preselect_m = int(pcfg.get("preselect_m", 10))

        e = cfg.get("embed", {})
        img_size = int(e.get("img_size", 560))
        global_mode = str(e.get("global_mode", "patch_mean"))
        tfm = make_transform(img_size=img_size)

        ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")

        th_dir = bank_root / plc_idx / "th_calib"
        th_img_paths = sorted(
            [p for p in th_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )

        scores = []
        for th_img_path in th_img_paths:
            img_bgr = cv2.imread(str(th_img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            x = tfm(BGR_to_RGB(img_bgr))
            q_out = make_embed(
                model,
                device,
                x,
                repr_mode=repr_mode,
                global_mode=global_mode,
            )

            s, _, _ = compute_knn_dist(
                q_out,
                ref_bank,
                k=k,
                repr_mode=repr_mode,
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
            scores.append(s)

        scores = np.array(scores, dtype=np.float32)

    else:
        raise ValueError(f"Unknown repr_mode: {repr_mode}")

    # ---- threshold from scores ----
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

    thr_path = bank_root / plc_idx / "threshold.json"
    thr_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return float(thr), scores, thr_path


def calibrate_place(bank_root, plc_idx, model, device, sg_matcher=None):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    cfg = load_cfg(bank_root)

    rebuild_bank(bank_root, plc_idx, model, device, mode="bank",     cfg=cfg)
    rebuild_bank(bank_root, plc_idx, model, device, mode="th_calib", cfg=cfg)

    thr, scores, thr_path = compute_and_save_threshold(
        bank_root,
        plc_idx,
        cfg=cfg,
        model=model,
        device=device,
        sg_matcher=sg_matcher,
    )    
    return thr, scores, thr_path




#------------------------------- 추론

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

    # -------------------------
    # cfg
    # -------------------------
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
    if repr_mode not in {"global","patch","global_patch","global_patch_pool","global_patch_with_aligned",}:
        raise ValueError(
            f"Unknown repr_mode={repr_mode} "
            "(use global|patch|global_patch|global_patch_pool|global_patch_with_aligned)"
    )

    # -------------------------
    # bank load
    # -------------------------
    ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")

    # -------------------------
    # frame loop
    # -------------------------
    frame_scores: List[float] = []
    frame_change_flags: List[int] = []
    topk_paths_all: List[List[str]] = []
    topk_sims_all: List[List[float]] = []
    patch_vis_all: List[Optional[Dict[str, Any]]] = [] # vis

    for img_bgr in imgs_bgr:
        x = tfm(BGR_to_RGB(img_bgr))

        # embed
        q_out = make_embed(
            model,
            device,
            x,
            repr_mode=repr_mode,          # global|patch|global_patch
            global_mode=global_mode,
        )

        # dist
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
            thr=thr
        )

        is_change = dist > thr
        frame_scores.append(float(dist))
        frame_change_flags.append(1 if is_change else 0)

        # -------------------------
        # topk logging (항상 "이미지 단위 topk")
        # -------------------------
        if repr_mode == "global":
            topk_sim, topk_idx = debug
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

        # -- vis

    # -------------------------
    # event aggregation
    # -------------------------
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

    #gui표시용 event 스코어
    event_score = float(np.clip(
        50.0 + 50.0 * (margin / max(thr, 1e-8)),
        0.0,
        100.0
    ))
    
    rep_idx = int(np.argmax(frame_scores)) if len(frame_scores) > 0 else 0 # vis

    # -------------------------
    # optional: VLM gate
    # -------------------------
    rep, summary = None, ""
    if use_two_stage_vlm and anomaly_flag == 1:
        pass
    # -------------------------
    # pack
    # -------------------------
    ref_topk_json = json.dumps(
        {"topk_paths": topk_paths_all, "topk_sims": topk_sims_all, "rep": rep},
        ensure_ascii=False,
    )

    """
    return {
        "threshold": float(thr),
        "frame_scores": [float(x) for x in frame_scores],
        "anomaly_flag": int(anomaly_flag),
        "frame_change_flags": [int(x) for x in frame_change_flags],
        "event_score": float(event_score),
        "ref_bank_id": ref_bank_id,
        "ref_topk_json": ref_topk_json,
        "summary": summary,
    }
    """

    #vis
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
    }