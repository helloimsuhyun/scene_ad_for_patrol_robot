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

import json
from pathlib import Path
import numpy as np
import torch
import cv2
import torch.nn.functional as F

import cv2

from .warp_utils import (
    warp_query_to_bank,
    make_patch_valid_mask,
    crop_common_valid_region,
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

    elif repr_mode in {
        "global_patch",
        "global_patch_with_aligned",
    }:
        patch_topk = debug["patch_topk"]
        return patch_topk["top_patch_idx"], patch_topk["top_patch_vals"]

    elif repr_mode == "global_patch_with_aligned_loss_bank":
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

# =========================================================================================== dist 계산 함수

def _infer_patch_grid(P: int):
    side = int(math.sqrt(P))
    if side * side != P:
        raise ValueError(f"Patch count P={P} is not a perfect square.")
    return side, side

# =====================================================  LOSS MAP
def make_lossmap_bgr(
    warped_q_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    valid_mask: np.ndarray | None = None,
    blur_ksize: int = 5,
    use_gray: bool = True,
    normalize: bool = True,
):
    """
    정합된 query/ref crop으로부터 pixel-diff loss map 생성.
    return:
        loss_bgr  : DINO 입력용 3채널 uint8 image
        loss_gray : scalar loss map uint8
    """
    if warped_q_bgr is None or ref_bgr is None:
        raise ValueError("warped_q_bgr/ref_bgr is None")

    if warped_q_bgr.shape[:2] != ref_bgr.shape[:2]:
        raise ValueError(
            f"shape mismatch: {warped_q_bgr.shape[:2]} vs {ref_bgr.shape[:2]}"
        )

    diff = cv2.absdiff(warped_q_bgr, ref_bgr)  # (H,W,3)

    if use_gray:
        loss_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        loss_gray = diff.mean(axis=2).astype(np.uint8)

    if blur_ksize is not None and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        loss_gray = cv2.GaussianBlur(loss_gray, (blur_ksize, blur_ksize), 0)

    if valid_mask is not None:
        vm = valid_mask
        if vm.dtype != np.uint8:
            vm = vm.astype(np.uint8)
        loss_gray = np.where(vm > 0, loss_gray, 0).astype(np.uint8)

    if normalize:
        mx = int(loss_gray.max())
        if mx > 0:
            loss_gray = np.clip(
                (loss_gray.astype(np.float32) / mx) * 255.0,
                0, 255
            ).astype(np.uint8)

    loss_bgr = cv2.cvtColor(loss_gray, cv2.COLOR_GRAY2BGR)
    return loss_bgr, loss_gray


def make_zero_prototype_bgr_like(img_bgr: np.ndarray):
    """
    loss-map과 같은 shape의 zero prototype image 생성
    """
    return np.zeros_like(img_bgr, dtype=np.uint8)


def extract_loss_patch_with_zero_proto(
    model,
    device,
    tfm,
    warped_q_crop: np.ndarray,
    ref_crop: np.ndarray,
    mask_crop: np.ndarray | None = None,
    blur_ksize: int = 5,
):
    """
    return:
        qp_loss   : (P,D)
        refp_zero : (1,P,D)
        loss_bgr  : debug용
        loss_gray : debug용
    """
    loss_bgr, loss_gray = make_lossmap_bgr(
        warped_q_crop,
        ref_crop,
        valid_mask=mask_crop,
        blur_ksize=blur_ksize,
        use_gray=True,
        normalize=True,
    )

    zero_bgr = make_zero_prototype_bgr_like(loss_bgr)

    x_q_loss = tfm(BGR_to_RGB(loss_bgr))
    qp_loss = extract_patch_layers(model, device, x_q_loss)   # (P,D)

    x_ref_zero = tfm(BGR_to_RGB(zero_bgr))
    refp_zero = extract_patch_layers(model, device, x_ref_zero).unsqueeze(0)  # (1,P,D)

    return qp_loss, refp_zero, loss_bgr, loss_gray

def build_aggregated_loss_map_for_query(
    q_img_bgr: np.ndarray,
    q_global: torch.Tensor,
    ref_bank,
    preselect_m: int,
    sg_matcher,
    loss_blur_ksize: int = 5,
    loss_erode_ksize: int = 0,
    loss_erode_iter: int = 1,
    exclude_ref_paths: list[str] | None = None,
):
    """
    query 1장에 대해:
      1) global preselect
      2) 정합 성공 ref들을 query 좌표계로 warp
      3) ref별 full-canvas loss map 생성
      4) pixel-wise min aggregation
      5) optional erosion
    return:
      {
        "ok": bool,
        "reason": str,
        "topk_sim": torch.Tensor,   # (1,m)
        "topk_idx": torch.Tensor,   # (1,m)
        "aggr_loss": np.ndarray | None,   # (H,W) uint8
        "aggr_valid": np.ndarray | None,  # (H,W) uint8
        "candidates": list[dict],
        "num_candidates": int,
      }
    """
    if q_img_bgr is None:
        raise ValueError("q_img_bgr is required")
    if sg_matcher is None:
        raise ValueError("sg_matcher is required")
    if ref_bank is None or "global" not in ref_bank:
        raise ValueError("ref_bank['global'] is required")
    if q_global is None:
        raise ValueError("q_global is required")
    
    exclude_ref_paths = set(str(p) for p in (exclude_ref_paths or []))

    refg_np, ref_paths = ref_bank["global"]
    refg = torch.from_numpy(refg_np).float().to(q_global.device)

    # 1) global preselect
    _, (topk_sim, topk_idx) = _dist_global(
        q_global,
        refg,
        min(preselect_m, refg.shape[0]),
    )
    idx = topk_idx.squeeze(0)

    full_loss_list = []
    full_valid_list = []
    candidates = []

    Hq, Wq = q_img_bgr.shape[:2]

    # 2) ref loop
    for ref_i in idx.tolist():
        ref_img_path = ref_paths[ref_i]
        if ref_img_path in exclude_ref_paths:
            continue
        ref_img_bgr = cv2.imread(str(ref_img_path), cv2.IMREAD_COLOR)
        if ref_img_bgr is None:
            continue

        match_out = sg_matcher.match_and_estimate(q_img_bgr, ref_img_bgr)
        if not match_out["ok"]:
            continue

        H_q2r = match_out["H"]
        try:
            H_r2q = np.linalg.inv(H_q2r)
        except np.linalg.LinAlgError:
            continue

        # ref -> query 좌표계 warp
        warped_ref_bgr = cv2.warpPerspective(
            ref_img_bgr,
            H_r2q,
            (Wq, Hq),
        )

        # ref valid mask -> query 좌표계 warp
        ref_mask = np.ones(ref_img_bgr.shape[:2], dtype=np.uint8) * 255
        warped_ref_mask = cv2.warpPerspective(
            ref_mask,
            H_r2q,
            (Wq, Hq),
        )
        warped_ref_mask = np.where(warped_ref_mask > 127, 255, 0).astype(np.uint8)

        # full query canvas에서 loss map 생성
        _, loss_gray = make_lossmap_bgr(
            q_img_bgr,
            warped_ref_bgr,
            valid_mask=warped_ref_mask,
            blur_ksize=loss_blur_ksize,
            use_gray=True,
            normalize=False,
        )

        full_loss_list.append(loss_gray.astype(np.float32))
        full_valid_list.append(warped_ref_mask.copy())

        candidates.append({
            "ref_i": ref_i,
            "match_out": match_out,
            "ref_img_path": str(ref_img_path),
        })

    if len(full_loss_list) == 0:
        return {
            "ok": False,
            "reason": "no_aligned_candidates",
            "topk_sim": topk_sim,
            "topk_idx": idx,
            "aggr_loss": None,
            "aggr_valid": None,
            "candidates": [],
            "num_candidates": 0,
        }

    # 3) full-canvas min aggregation
    masked_stack = []
    valid_stack = []

    for loss_gray, valid_mask in zip(full_loss_list, full_valid_list):
        vm = (valid_mask > 0)
        masked = np.where(vm, loss_gray, np.inf).astype(np.float32)
        masked_stack.append(masked)
        valid_stack.append(vm.astype(np.uint8))

    masked_stack = np.stack(masked_stack, axis=0)   # (N,H,W)
    valid_stack = np.stack(valid_stack, axis=0)     # (N,H,W)

    aggr_loss = np.min(masked_stack, axis=0)
    aggr_loss[np.isinf(aggr_loss)] = 0.0
    aggr_loss = np.clip(aggr_loss, 0, 255).astype(np.uint8)

    aggr_valid = (np.sum(valid_stack, axis=0) > 0).astype(np.uint8) * 255

    # 4) optional morphology
    if loss_erode_ksize is not None and loss_erode_ksize > 1:
        kernel = np.ones((loss_erode_ksize, loss_erode_ksize), np.uint8)
        aggr_loss = cv2.erode(
            aggr_loss,
            kernel,
            iterations=loss_erode_iter,
        )

    return {
        "ok": True,
        "reason": "ok",
        "topk_sim": topk_sim,
        "topk_idx": idx,
        "aggr_loss": aggr_loss,
        "aggr_valid": aggr_valid,
        "candidates": candidates,
        "num_candidates": len(candidates),
    }

def extract_loss_features_from_aggr(
    model,
    device,
    tfm,
    aggr_loss: np.ndarray,
    aggr_valid: np.ndarray,
    valid_patch_thr: float,
):
    """
    aggregated loss map / valid map으로부터
      - DINO patch embedding
      - valid patch mask
    를 추출한다.

    return:
      {
        "ok": bool,
        "reason": str,
        "aggr_loss_bgr": np.ndarray,
        "qp_loss": torch.Tensor | None,         # (P,D)
        "patch_valid_2d": np.ndarray | None,    # (Hp,Wp)
        "patch_valid_1d": np.ndarray | None,    # (P,)
        "valid_patch_count": int,
        "grid_h": int | None,
        "grid_w": int | None,
      }
    """
    if model is None or device is None:
        raise ValueError("model/device required")
    if tfm is None:
        raise ValueError("tfm required")
    if aggr_loss is None:
        raise ValueError("aggr_loss is required")
    if aggr_valid is None:
        raise ValueError("aggr_valid is required")

    aggr_loss_bgr = cv2.cvtColor(aggr_loss, cv2.COLOR_GRAY2BGR)

    x_q_loss = tfm(BGR_to_RGB(aggr_loss_bgr))
    qp_loss = extract_patch_layers(model, device, x_q_loss)   # (P,D)

    P2 = qp_loss.shape[0]
    grid_h2, grid_w2 = _infer_patch_grid(P2)

    patch_valid_2d = make_patch_valid_mask(
        aggr_valid,
        grid_h=grid_h2,
        grid_w=grid_w2,
        thr=valid_patch_thr,
    )
    patch_valid_1d = patch_valid_2d.reshape(-1)

    valid_patch_count = int(patch_valid_1d.sum())

    if valid_patch_count < 4:
        return {
            "ok": False,
            "reason": "too_few_valid_patches_after_aggregation",
            "aggr_loss_bgr": aggr_loss_bgr,
            "qp_loss": qp_loss,
            "patch_valid_2d": patch_valid_2d,
            "patch_valid_1d": patch_valid_1d,
            "valid_patch_count": valid_patch_count,
            "grid_h": grid_h2,
            "grid_w": grid_w2,
        }

    return {
        "ok": True,
        "reason": "ok",
        "aggr_loss_bgr": aggr_loss_bgr,
        "qp_loss": qp_loss,
        "patch_valid_2d": patch_valid_2d,
        "patch_valid_1d": patch_valid_1d,
        "valid_patch_count": valid_patch_count,
        "grid_h": grid_h2,
        "grid_w": grid_w2,
    }

# LOSS BANK =====================================================
@torch.inference_mode()
@torch.inference_mode()
def build_loss_feature_bank(
    bank_root,
    plc_idx,
    cfg: dict,
    model,
    device,
    sg_matcher,
    save_name: str | None = None,
    min_valid_patches: int = 4,
):
    """
    bank 이미지들끼리 서로 비교해서
    normal global loss feature bank를 생성해 저장한다.

    저장 내용:
      - embs:        (N, D) float32
      - valid_counts:(N,) int32
      - paths:       (N,) object
      - meta_json:   str
    """
    if model is None or device is None:
        raise ValueError("model/device required")
    if sg_matcher is None:
        raise ValueError("sg_matcher required")

    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    if save_name is None:
        save_name = f"{plc_idx}_loss_global_bank.npz"

    pcfg = cfg.get("patchcore", {})
    preselect_m = int(pcfg.get("preselect_m", 10))
    loss_blur_ksize = int(pcfg.get("loss_blur_ksize", 5))
    loss_erode_ksize = int(pcfg.get("loss_erode_ksize", 0))
    loss_erode_iter = int(pcfg.get("loss_erode_iter", 1))

    sg_raw = cfg["superglue"]
    valid_patch_thr = float(sg_raw["valid_patch_thr"])

    tfm = make_transform(cfg["embed"]["img_size"])

    ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")
    refg_np, ref_paths = ref_bank["global"]

    refg = torch.from_numpy(refg_np).float().to(device)
    refg = F.normalize(refg, dim=1)

    saved_embs = []
    saved_valid_counts = []
    saved_paths = []

    num_total = len(ref_paths)
    num_skipped = 0

    for i, q_img_path in enumerate(ref_paths):
        q_img_path = str(q_img_path)

        img_bgr = cv2.imread(q_img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            num_skipped += 1
            continue

        qg = refg[i:i+1]

        agg_out = build_aggregated_loss_map_for_query(
            q_img_bgr=img_bgr,
            q_global=qg,
            ref_bank=ref_bank,
            preselect_m=preselect_m,
            sg_matcher=sg_matcher,
            loss_blur_ksize=loss_blur_ksize,
            loss_erode_ksize=loss_erode_ksize,
            loss_erode_iter=loss_erode_iter,
            exclude_ref_paths=[q_img_path],
        )

        if not agg_out["ok"]:
            num_skipped += 1
            continue

        aggr_loss = agg_out["aggr_loss"]
        aggr_valid = agg_out["aggr_valid"]

        feat_out = extract_loss_features_from_aggr(
            model=model,
            device=device,
            tfm=tfm,
            aggr_loss=aggr_loss,
            aggr_valid=aggr_valid,
            valid_patch_thr=valid_patch_thr,
        )

        if not feat_out["ok"]:
            num_skipped += 1
            continue

        valid_patch_count = int(feat_out["valid_patch_count"])
        if valid_patch_count < min_valid_patches:
            num_skipped += 1
            continue

        g_loss = extract_global_loss_feature(
            model=model,
            device=device,
            tfm=tfm,
            aggr_loss=aggr_loss,
        )  # (D,)

        saved_embs.append(g_loss.detach().cpu().numpy().astype(np.float32))
        saved_valid_counts.append(valid_patch_count)
        saved_paths.append(q_img_path)

    if len(saved_embs) == 0:
        save_path = bank_root / plc_idx / "bank" / save_name
        return {
            "ok": False,
            "save_path": str(save_path),
            "num_total": num_total,
            "num_saved": 0,
            "num_skipped": num_skipped,
            "paths": [],
        }

    embs_np = np.stack(saved_embs, axis=0)                    # (N,D)
    valid_counts_np = np.array(saved_valid_counts, np.int32)  # (N,)

    meta = {
        "repr_mode": "global_patch_with_aligned_loss_bank",
        "loss_bank_type": "global",
        "place_id": plc_idx,
        "num_total_bank_imgs": num_total,
        "num_saved_bank_imgs": len(saved_paths),
        "embed_img_size": int(cfg["embed"]["img_size"]),
        "preselect_m": preselect_m,
        "loss_blur_ksize": loss_blur_ksize,
        "loss_erode_ksize": loss_erode_ksize,
        "loss_erode_iter": loss_erode_iter,
        "valid_patch_thr": valid_patch_thr,
        "min_valid_patches": min_valid_patches,
    }

    save_path = bank_root / plc_idx / "bank" / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        embs=embs_np,
        valid_counts=valid_counts_np,
        paths=np.array(saved_paths, dtype=object),
        meta_json=json.dumps(meta, ensure_ascii=False),
    )

    return {
        "ok": True,
        "save_path": str(save_path),
        "num_total": num_total,
        "num_saved": len(saved_paths),
        "num_skipped": num_skipped,
        "paths": saved_paths,
    }
def load_loss_feature_bank(
    bank_root,
    plc_idx,
    file_name: str | None = None,
):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    if file_name is None:
        file_name = f"{plc_idx}_loss_global_bank.npz"

    npz_path = bank_root / plc_idx / "bank" / file_name
    if not npz_path.exists():
        raise FileNotFoundError(f"loss feature bank not found: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)

    meta_json = z["meta_json"]
    if isinstance(meta_json, np.ndarray):
        meta_json = meta_json.item()
    meta = json.loads(meta_json)

    return {
        "embs": z["embs"],                 # (N,D)
        "valid_counts": z["valid_counts"], # (N,)
        "paths": list(z["paths"]),
        "meta": meta,
        "npz_path": str(npz_path),
    }


def extract_global_loss_feature(
    model,
    device,
    tfm,
    aggr_loss: np.ndarray,
):
    """
    aggregated loss map -> global feature (D,)
    """
    if aggr_loss is None:
        raise ValueError("aggr_loss is required")
    if model is None or device is None:
        raise ValueError("model/device required")
    if tfm is None:
        raise ValueError("tfm required")

    aggr_loss_bgr = cv2.cvtColor(aggr_loss, cv2.COLOR_GRAY2BGR)
    x = tfm(BGR_to_RGB(aggr_loss_bgr))

    out = make_embed(
        model,
        device,
        x,
        repr_mode="global",
    )
    g = out["global"]   # (D,)
    g = F.normalize(g, dim=0)
    return g


def _dist_loss_bank_global(
    q_global_loss: torch.Tensor,   # (D,)
    ref_global_bank: torch.Tensor, # (N,D)
    metric: str = "cosine",
):
    """
    global loss feature vs global loss bank
    return:
        score: float
        debug: dict
    """
    q = F.normalize(q_global_loss, dim=0)
    ref = F.normalize(ref_global_bank, dim=1)

    if metric == "cosine":
        sim = ref @ q                    # (N,)
        dist = 1.0 - sim                 # 작을수록 정상
        best_idx = torch.argmin(dist)
        score = float(dist[best_idx].item())
        debug = {
            "metric": metric,
            "best_idx": int(best_idx.item()),
            "best_dist_full": dist.detach().cpu(),
            "best_sim": float(sim[best_idx].item()),
        }
        return score, debug

    elif metric == "l2":
        diff = ref - q.unsqueeze(0)      # (N,D)
        dist = torch.sqrt((diff * diff).mean(dim=1))
        best_idx = torch.argmin(dist)
        score = float(dist[best_idx].item())
        debug = {
            "metric": metric,
            "best_idx": int(best_idx.item()),
            "best_dist_full": dist.detach().cpu(),
        }
        return score, debug

    else:
        raise ValueError(f"Unknown metric: {metric}")
    
# ================================================================

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
    # 2) 그 안에서 peak-relative thresholding
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

        # peak-relative threshold
        peak_val = top_vals.max()
        keep = top_vals >= (alpha * peak_val)             # (m,)

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

def _dist_patchcore_masked_local(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    valid_mask_1d,              # (P,) bool
    top_p: float = 0.05,         
    k: int = 1,                 # 유지
    alpha: float = 0.6, 
    radius: int = 1,
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
    best_match_idx_list = []
    valid_orig_idx = []

    # --------------------------------------------------
    # 1) valid patch들에 대해 local-window distance 계산

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
            max_sim, best_local_idx = sim.max(dim=1)               # (N,)
            dist = 1.0 - max_sim                           # (N,)

            # local window index -> global patch index
            win_w = x1 - x0

            dy = torch.div(best_local_idx, win_w, rounding_mode="floor")
            dx = best_local_idx % win_w

            best_global_idx = (y0 + dy) * Wp + (x0 + dx)   # (N,)

            best_dist_list.append(dist)
            best_match_idx_list.append(best_global_idx)
            valid_orig_idx.append(y * Wp + x)

    if len(best_dist_list) < 4:
        return None, {"ok": False, "reason": "too_few_valid_patches_after_local"}

    dist_map = torch.stack(best_dist_list, dim=1)   # (N, Pv)
    match_idx_map = torch.stack(best_match_idx_list, dim=1)  # (N, Pv)
    Pv = dist_map.shape[1]
    valid_orig_idx_t = torch.tensor(valid_orig_idx, device=q.device, dtype=torch.long)

    # 2) bank 이미지에 대해 patch별 top-p 이상 patch  + peak-relative threshold score

    score_per_img = []
    per_img_peak = []
    per_img_area = []
    per_img_top_patch_idx = []
    per_img_top_patch_vals = []
    per_img_top_patch_match_idx = []
    per_img_best_patch_dist_full = []
    per_img_best_patch_match_full = []

    for n in range(N):
        patch_dist_valid = dist_map[n]       # (Pv,)
        patch_match_valid = match_idx_map[n] # (Pv,)

        # valid patch만 있는 1D -> full patch grid로 복원
        full_patch_dist = torch.zeros(P, device=q.device, dtype=patch_dist_valid.dtype)
        full_patch_dist[valid_orig_idx_t] = patch_dist_valid
        patch_dist_2d = full_patch_dist.view(Hp, Wp)

        full_patch_match = torch.full(
            (P,),
            fill_value=-1,
            device=q.device,
            dtype=torch.long,
        )
        full_patch_match[valid_orig_idx_t] = patch_match_valid

        # invalid 제외
        candidate_map = patch_dist_2d.clone()
        candidate_map[~valid2] = 0.0

        peak_val = float(candidate_map.max().item())

        # debug용 full map 저장
        per_img_best_patch_dist_full.append(full_patch_dist)
        per_img_best_patch_match_full.append(full_patch_match)

        # top-p
        k_top = max(1, int(np.ceil(Pv * top_p)))
        top_vals, top_local_idx = torch.topk(
            patch_dist_valid,
            k=min(k_top, patch_dist_valid.numel())
        )
        top_idx = valid_orig_idx_t[top_local_idx]
        top_match_idx = patch_match_valid[top_local_idx]

        if top_vals.numel() == 0:
            score_one = 0.0
            kept_vals = top_vals
            kept_idx = top_idx
            kept_match_idx = top_match_idx
            selected_area = 0
            peak_val = 0.0
        else:
            peak_val = float(top_vals.max().item())

            # peak-relative threshold
            keep = top_vals >= (alpha * peak_val)

            kept_vals = top_vals[keep]
            kept_idx = top_idx[keep]
            kept_match_idx = top_match_idx[keep]

            if kept_vals.numel() == 0:
                kept_vals = top_vals[:1]
                kept_idx = top_idx[:1]
                kept_match_idx = top_match_idx[:1]

            score_one = float(kept_vals.mean().item())
            selected_area = int(kept_vals.numel())

        score_per_img.append(score_one)
        per_img_peak.append(peak_val)
        per_img_area.append(selected_area)

        # debug용 저장: threshold 후 남은 patch만 넣기
        per_img_top_patch_idx.append(kept_idx)
        per_img_top_patch_vals.append(kept_vals)
        per_img_top_patch_match_idx.append(kept_match_idx)

    score_per_img = torch.tensor(score_per_img, device=q.device, dtype=torch.float32)

    # 가장 정상에 가까운(ref score 최소) ref 선택
    best_img_idx = torch.argmin(score_per_img)
    score = float(score_per_img[best_img_idx].item())

    best_patch_dist_full = per_img_best_patch_dist_full[int(best_img_idx.item())]
    best_patch_match_full = per_img_best_patch_match_full[int(best_img_idx.item())]
    top_patch_idx = per_img_top_patch_idx[int(best_img_idx.item())]
    top_patch_vals = per_img_top_patch_vals[int(best_img_idx.item())]
    top_patch_match_idx = per_img_top_patch_match_idx[int(best_img_idx.item())]

    debug = {
        "ok": True,
        "reason": "peak",
        "best_img_idx": best_img_idx,
        "score_per_img": score_per_img,
        "best_patch_dist": best_patch_dist_full,   # (P,)
        "best_patch_match_idx": best_patch_match_full,  # (P,)
        "top_patch_idx": top_patch_idx,
        "top_patch_vals": top_patch_vals,
        "top_patch_match_idx": top_patch_match_idx,
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

# =========================================================================================== dist 계산 함수


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
    bank_root=None,   
    plc_idx=None,     
):
    timing = {}
    pcfg = cfg.get("patchcore", {})
    alpha = float(pcfg.get("alpha", 0.6))

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
                alpha=alpha,
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
                alpha = alpha
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

        lp_cfg = cfg.get("patchcore", {})
        local_radius = int(lp_cfg.get("radius", 1))
        alpha = float(lp_cfg.get("alpha", 0.6))

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
                print("[DEBUG] ref_img read fail:", ref_i, ref_img_path)
                continue

            match_out = sg_matcher.match_and_estimate(q_img_bgr, ref_img_bgr)
            if not match_out["ok"]:
                print("[DEBUG] match fail:", ref_i)
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

            if warped_q_crop is None:
                print("[DEBUG] crop fail:", ref_i)
                continue

            # 5) crop된 query / ref를 patch 임베딩
            x_q_crop = tfm(BGR_to_RGB(warped_q_crop))

            qp_crop = extract_patch_layers(model,device,x_q_crop) # [P,D]

            x_ref_crop = tfm(BGR_to_RGB(ref_crop))
            refp_crop = extract_patch_layers(model,device,x_ref_crop).unsqueeze(0) # [1,P,,D]

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
                print("[DEBUG] too few valid patches:", ref_i, valid_patch_count)
                continue

            # 7) masked local patch 비교
            score_one, dbg_one = _dist_patchcore_masked_local(
                qp_crop,
                refp_crop,
                valid_mask_1d=patch_valid_1d,
                top_p=top_p,
                k=1,
                radius=local_radius,
                alpha=alpha
            )

            if score_one is None or not dbg_one.get("ok", True):
                print("[DEBUG] local patch score fail:", ref_i, dbg_one.get("reason"))
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
                alpha=alpha
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

        # 8) distance는 작을수록 좋으므로 min
        scores = [c["score"] for c in candidates]
        score = float(np.mean(scores))

        best = max(candidates, key=lambda x: x["score"])

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
                "valid_patch_count": int(best["valid_patch_count"]),
                "H": best_match["H"],
                "inliers": best_match["inliers"],
                "inlier_ratio": best_match["inlier_ratio"],
                "reproj_error_mean": best_match["reproj_error_mean"],
                "reproj_error_median": best_match["reproj_error_median"],
                "top_patch_idx": best_patch_debug["top_patch_idx"],
                "top_patch_vals": best_patch_debug["top_patch_vals"],
            }

        _print_timing(f"compute_knn_dist[{repr_mode}]", timing)
        return score, debug, ref_paths

    elif repr_mode == "global_patch_with_aligned_loss_bank":

        if sg_matcher is None:
            raise ValueError("sg_matcher is required")
        if q_img_bgr is None:
            raise ValueError("q_img_bgr is required")
        if tfm is None:
            raise ValueError("tfm is required")
        if model is None or device is None:
            raise ValueError("model/device required")
        if cfg is None:
            raise ValueError("cfg required")
        if bank_root is None or plc_idx is None:
            raise ValueError("bank_root/plc_idx required for loss_bank mode")

        sg_raw = cfg["superglue"]

        lp_cfg = cfg.get("patchcore", {})
        loss_blur_ksize = int(lp_cfg.get("loss_blur_ksize", 5))
        loss_erode_ksize = int(lp_cfg.get("loss_erode_ksize", 0))
        loss_erode_iter = int(lp_cfg.get("loss_erode_iter", 1))
        loss_metric = str(lp_cfg.get("loss_global_metric", "cosine"))

        refg_np, ref_paths = ref_bank["global"]
        qg = q_out["global"]

        agg_out = build_aggregated_loss_map_for_query(
            q_img_bgr=q_img_bgr,
            q_global=qg,
            ref_bank=ref_bank,
            preselect_m=preselect_m,
            sg_matcher=sg_matcher,
            loss_blur_ksize=loss_blur_ksize,
            loss_erode_ksize=loss_erode_ksize,
            loss_erode_iter=loss_erode_iter,
        )

        topk_sim = agg_out["topk_sim"]
        idx = agg_out["topk_idx"]
        candidates = agg_out["candidates"]

        if not agg_out["ok"]:
            print("[DEBUG] fallback global_patch")
            refp_np, _ = ref_bank["patch"]
            qp = q_out["patch"]
            refp = torch.from_numpy(refp_np).float().to(qp.device)
            refp_sel = refp[idx]

            score, _ = _dist_patchcore(
                qp,
                refp_sel,
                top_p=top_p,
                k=min(k, refp_sel.shape[0]),
            )

            return score, {
                "fallback": True,
                "reason": agg_out["reason"],
                "global_topk": (topk_sim, idx),
            }, ref_paths

        aggr_loss = agg_out["aggr_loss"]
        aggr_valid = agg_out["aggr_valid"]

        feat_out = extract_loss_features_from_aggr(
            model=model,
            device=device,
            tfm=tfm,
            aggr_loss=aggr_loss,
            aggr_valid=aggr_valid,
            valid_patch_thr=sg_raw["valid_patch_thr"],
        )

        valid_patch_count = feat_out["valid_patch_count"]

        if not feat_out["ok"]:
            return 0.0, {
                "fallback": False,
                "global_topk": (topk_sim, idx),
                "loss_agg": {
                    "aggr_loss_map": aggr_loss,
                    "aggr_valid_map": aggr_valid,
                    "valid_patch_count": valid_patch_count,
                },
            }, ref_paths

        # 🔥 global loss feature
        q_global_loss = extract_global_loss_feature(
            model=model,
            device=device,
            tfm=tfm,
            aggr_loss=aggr_loss,
        )  # (D,)

        loss_bank = load_loss_feature_bank(bank_root, plc_idx)
        ref_embs = torch.from_numpy(loss_bank["embs"]).float().to(device)   # (N,D)

        score, dbg_loss = _dist_loss_bank_global(
            q_global_loss=q_global_loss,
            ref_global_bank=ref_embs,
            metric=loss_metric,
        )

        best_bank_idx = int(dbg_loss["best_idx"])
        best_loss_bank_path = str(loss_bank["paths"][best_bank_idx])

        debug = {
            "fallback": False,
            "global_topk": (topk_sim, idx),
            "loss_agg": {
                "aggr_loss_map": aggr_loss,
                "aggr_valid_map": aggr_valid,
                "valid_patch_count": valid_patch_count,
            },
            "loss_bank": {
                "metric": loss_metric,
                "best_bank_idx": best_bank_idx,
                "best_bank_path": best_loss_bank_path,
                "best_dist_full": dbg_loss["best_dist_full"],
            },
        }

        return score, debug, ref_paths

    else:
        raise ValueError(
            "repr_mode must be "
            "global|patch|global_patch|global_patch_with_aligned|global_patch_with_aligned_loss_zero"
        )


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

    pcfg = cfg.get("patchcore", {})
    top_p = float(pcfg.get("top_p", 0.1))
    preselect_m = int(pcfg.get("preselect_m", 10))
    local_radius = int(pcfg.get("radius", 1))
    alpha = float(pcfg.get("alpha", 0.6))

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
            scores = []
            for i in range(th_g.shape[0]):
                with _timer(total_timing, "calib.gp.per_th_global_preselect", device):
                    sim = th_g[i:i+1] @ bank_g.T
                    M = min(preselect_m, bank_g.shape[0])
                    _, idx = torch.topk(sim, k=M, dim=1)
                    idx = idx.squeeze(0)

                with _timer(total_timing, "calib.gp.per_th_patch_dist", device):
                    bank_sel = bank_p[idx]  # (M,P,D)
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
                        radius=local_radius,
                        alpha = alpha
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
                s = float(np.mean(cand_scores))

            scores.append(float(s))
            _merge_timing(total_timing, {f"calib.aligned.per_img.{kk}": vv for kk, vv in loop_timing.items()})

        scores = np.array(scores, dtype=np.float32)
    
    elif repr_mode == "global_patch_with_aligned_loss_bank":

        if model is None or device is None or sg_matcher is None:
            raise ValueError("model/device/sg_matcher required")

        sg_raw = cfg["superglue"]

        lp_cfg = cfg.get("patchcore", {})
        preselect_m = int(lp_cfg.get("preselect_m", 10))
        loss_blur_ksize = int(lp_cfg.get("loss_blur_ksize", 5))
        loss_erode_ksize = int(lp_cfg.get("loss_erode_ksize", 3))
        loss_erode_iter = int(lp_cfg.get("loss_erode_iter", 1))
        loss_metric = str(lp_cfg.get("loss_global_metric", "cosine"))

        tfm = make_transform(cfg["embed"]["img_size"])

        ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")

        th_npz = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
        th_paths = list(np.load(th_npz, allow_pickle=True)["paths"])

        loss_bank = load_loss_feature_bank(bank_root, plc_idx)
        ref_embs = torch.from_numpy(loss_bank["embs"]).float().to(device)   # (N,D)

        scores = []

        for img_path in th_paths:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                scores.append(0.0)
                continue

            x = tfm(BGR_to_RGB(img_bgr))
            q_out = make_embed(model, device, x, repr_mode="global")
            qg = q_out["global"]

            agg_out = build_aggregated_loss_map_for_query(
                q_img_bgr=img_bgr,
                q_global=qg,
                ref_bank=ref_bank,
                preselect_m=preselect_m,
                sg_matcher=sg_matcher,
                loss_blur_ksize=loss_blur_ksize,
                loss_erode_ksize=loss_erode_ksize,
                loss_erode_iter=loss_erode_iter,
            )

            if not agg_out["ok"]:
                scores.append(0.0)
                continue

            feat_out = extract_loss_features_from_aggr(
                model=model,
                device=device,
                tfm=tfm,
                aggr_loss=agg_out["aggr_loss"],
                aggr_valid=agg_out["aggr_valid"],
                valid_patch_thr=sg_raw["valid_patch_thr"],
            )

            if not feat_out["ok"]:
                scores.append(0.0)
                continue

            q_global_loss = extract_global_loss_feature(
                model=model,
                device=device,
                tfm=tfm,
                aggr_loss=agg_out["aggr_loss"],
            )

            s, _ = _dist_loss_bank_global(
                q_global_loss=q_global_loss,
                ref_global_bank=ref_embs,
                metric=loss_metric,
            )
            scores.append(float(s))

        scores = np.array(scores, dtype=np.float32)

    else:
        raise ValueError(f"Unknown repr_mode: {repr_mode}")


    # ================================================== threshold from scores 
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

    repr_mode = str(cfg.get("repr", {}).get("repr_mode", "global"))

    if repr_mode == "global_patch_with_aligned_loss_bank":
        with _timer(timing, "calibrate.build_loss_feature_bank", device):
            out = build_loss_feature_bank(
                bank_root=bank_root,
                plc_idx=plc_idx,
                cfg=cfg,
                model=model,
                device=device,
                sg_matcher=sg_matcher,
            )
            if not out["ok"]:
                raise RuntimeError(
                    f"failed to build loss feature bank for plc={plc_idx}: {out}"
                )


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


# ================================================================================= 추론

# event 단위 이상감지
# ================================================================================= 추론

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
    repr_mode: "global" | "patch" | "global_patch" | "global_patch_with_align"
    - global: global feature kNN
    - patch: PatchCore-style
    - global_patch: global preselect 후 image-level patchcore
    - global_patch_with_align: global preselect후 각기 이미지에 대해 정합 -> local patch별 비교 -> dist

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
        "patch_vis": dict,
        "align_vis": dict | None,
        "loss_vis": dict | None,
        "timing": dict,
        "frame_timing": List[dict],
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

    if repr_mode not in {
        "global",
        "patch",
        "global_patch",
        "global_patch_with_aligned",
        "global_patch_with_aligned_loss_bank",
    }:
        raise ValueError(
            f"Unknown repr_mode={repr_mode} "
            "(use global|patch|global_patch|global_patch_with_aligned|global_patch_with_aligned_loss_bank)"
        )

    # -------------------------
    # bank load
    # -------------------------
    with _timer(total_timing, "infer.load_ref_bank", device):
        ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")

    # -------------------------
    # per-frame inference
    # -------------------------
    frame_scores = []
    frame_change_flags = []

    topk_paths_all = []
    topk_sims_all = []

    patch_vis_all = []
    align_vis_all = []
    loss_vis_all = []
    frame_timing_all = []

    for fi, img_bgr in enumerate(imgs_bgr):
        frame_timing = {}

        with _timer(frame_timing, "tfm", device):
            x = tfm(BGR_to_RGB(img_bgr))

        with _timer(frame_timing, "embed", device):
            q_out = make_embed(
                model,
                device,
                x,
                repr_mode=repr_mode,
                global_mode=global_mode,
            )

        with _timer(frame_timing, "compute_knn_dist", device):
            score, debug, ref_paths = compute_knn_dist(
                q_out=q_out,
                ref_bank=ref_bank,
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
                bank_root=bank_root,
                plc_idx=plc_idx,
            )

        score = float(score)
        is_change = int(score > thr)

        frame_scores.append(score)
        frame_change_flags.append(is_change)

        # ---------------------------------
        # top-k path/sim 정리
        # ---------------------------------
        if repr_mode == "global":
            topk_sim, topk_idx = debug["inner_debug"][1] if isinstance(debug.get("inner_debug"), tuple) else debug["inner_debug"]["topk"]
            idx = topk_idx.tolist() if hasattr(topk_idx, "tolist") else list(topk_idx)
            sims = topk_sim.tolist() if hasattr(topk_sim, "tolist") else list(topk_sim)
            paths = [ref_paths[i] for i in idx]

        elif repr_mode == "patch":
            topk_idx = debug["topk_idx"]
            topk_score = debug["topk_score"]
            idx = topk_idx.detach().cpu().tolist() if hasattr(topk_idx, "detach") else list(topk_idx)
            sims = [float(x) for x in (
                topk_score.detach().cpu().tolist() if hasattr(topk_score, "detach") else list(topk_score)
            )]
            paths = [ref_paths[i] for i in idx]

        elif repr_mode == "global_patch":
            _, topk_idx = debug["global_topk"]
            idx = topk_idx.tolist() if hasattr(topk_idx, "tolist") else list(topk_idx)
            paths = [ref_paths[i] for i in idx]
            sims = [0.0 for _ in idx]

        elif repr_mode == "global_patch_with_aligned":
            _, topk_idx = debug["global_topk"]
            idx = topk_idx.tolist() if hasattr(topk_idx, "tolist") else list(topk_idx)
            paths = [ref_paths[i] for i in idx]
            sims = [0.0 for _ in idx]

        elif repr_mode == "global_patch_with_aligned_loss_bank":
            _, topk_idx = debug["global_topk"]
            idx = topk_idx.tolist() if hasattr(topk_idx, "tolist") else list(topk_idx)
            paths = [ref_paths[i] for i in idx]
            sims = [0.0 for _ in idx]

        else:
            raise ValueError(f"Unknown repr_mode={repr_mode}")

        topk_paths_all.append([str(p) for p in paths])
        topk_sims_all.append([float(s) for s in sims])

        # ---------------------------------
        # patch vis
        # ---------------------------------
        top_patch_idx, top_patch_vals = get_top_p_patch_info(repr_mode, debug)

        if top_patch_idx is not None and top_patch_vals is not None:
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
            })
        else:
            patch_vis_all.append(None)

        # ---------------------------------
        # align vis
        # ---------------------------------
        if repr_mode in {"global_patch_with_aligned", "global_patch_with_aligned_loss_bank"}:
            align_vis_all.append(debug.get("align_debug", None))
        else:
            align_vis_all.append(None)

        # ---------------------------------
        # loss vis
        # ---------------------------------
        frame_loss_vis = None
        if repr_mode == "global_patch_with_aligned_loss_bank":
            loss_agg = debug.get("loss_agg", None)

            if loss_agg is not None:
                best_ref_img_path = None

                if "loss_bank" in debug:
                    best_ref_img_path = debug["loss_bank"].get("best_bank_path", None)

                if best_ref_img_path is None and "global_topk" in debug:
                    _, topk_idx_dbg = debug["global_topk"]
                    idx0 = (
                        int(topk_idx_dbg[0].item())
                        if hasattr(topk_idx_dbg[0], "item")
                        else int(topk_idx_dbg[0])
                    )
                    if 0 <= idx0 < len(ref_paths):
                        best_ref_img_path = str(ref_paths[idx0])

                frame_loss_vis = {
                    "frame_idx": int(fi),
                    "aggr_loss_map": loss_agg.get("aggr_loss_map", None),
                    "aggr_valid_map": loss_agg.get("aggr_valid_map", None),
                    "best_ref_img_path": best_ref_img_path,
                }

        loss_vis_all.append(frame_loss_vis)

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
    if anomaly_flag == 1:
        rep, summary = None, "anomaly detect"
    else:
        rep, summary = None, "it's fine, have relex"

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

    loss_vis = (
        {
            "frame_idx": rep_idx,
            "aggr_loss_map": (
                loss_vis_all[rep_idx]["aggr_loss_map"]
                if len(loss_vis_all) > rep_idx and loss_vis_all[rep_idx] is not None
                else None
            ),
            "aggr_valid_map": (
                loss_vis_all[rep_idx]["aggr_valid_map"]
                if len(loss_vis_all) > rep_idx and loss_vis_all[rep_idx] is not None
                else None
            ),
            "best_ref_img_path": (
                loss_vis_all[rep_idx]["best_ref_img_path"]
                if len(loss_vis_all) > rep_idx and loss_vis_all[rep_idx] is not None
                else None
            ),
        }
        if repr_mode == "global_patch_with_aligned_loss_bank"
        else None
    )

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
        },
        "align_vis": (
            {
                "frame_idx": rep_idx,
                "data": align_vis_all[rep_idx],
            }
            if repr_mode in {"global_patch_with_aligned", "global_patch_with_aligned_loss_bank"}
            and len(align_vis_all) > rep_idx
            and align_vis_all[rep_idx] is not None
            else None
        ),
        "loss_vis": loss_vis,
        "timing": _stats_to_float_dict(total_timing),
        "frame_timing": frame_timing_all,
    }