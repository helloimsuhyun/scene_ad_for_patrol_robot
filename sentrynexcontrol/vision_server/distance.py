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
from .dino_emb import make_embed, make_transform
from .config import load_cfg, get_cfg_bundle

import cv2

from .distance_util import (
    score_one_pair,
    build_flagged_component_regions,
    verify_bbox_with_local_search,
)


# vis helper
def get_top_p_patch_info(repr_mode: str, debug: dict):

    if repr_mode in {"global_patch", "global_patch_with_aligned"}:
        patch_topk = debug["patch_topk"]
        return patch_topk["top_patch_idx"], patch_topk["top_patch_vals"]

    else:
        return None, None

# =========================================================================================== ===dist 계산 함수

# 1. global dist 계산 함수 > 거리와 제일 유사한 k개 유사도, 인덱스 반환
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

# 2. patchcore 스타일 dist 계산
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

# 3. preselcet > 정합 > local focus위한 함수 
def run_preselect(q_preselect_emb, ref_bank, mode, top_m):

    ref_embs_np, ref_paths = ref_bank["global"]

    if mode == "dino":
        q = q_preselect_emb
        if isinstance(q, np.ndarray):
            q = torch.from_numpy(q).float()

        ref = torch.from_numpy(ref_embs_np).float().to(q.device)
        _, (_, topk_idx) = _dist_global(q, ref, min(top_m, ref.shape[0]))
        return topk_idx.squeeze(0).tolist(), ref_paths

    elif mode == "vpr":
        q = q_preselect_emb
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().numpy()

        sims = q @ ref_embs_np.T
        idx = np.argsort(-sims)[:top_m]
        return idx.tolist(), ref_paths

    else:
        raise ValueError(mode)

@torch.inference_mode()
def _run_gpa_frame(
    q_bgr: np.ndarray,
    q_out: Optional[Dict[str, Any]],
    ref_bank,
    cfg: dict,
    sg_matcher,
    cc_backbone,
    verifier_backbone,
    device: str,
    k: int,
    vpr_model=None
):
    cfgb = get_cfg_bundle(cfg)

    mode_cfg = cfgb["mode_cfg"]
    gp_cfg = cfgb["preselect"]
    cc_cfg = cfgb["cc"]
    proposal_cfg = cfgb["proposal"]
    ver_cfg = cfgb["verifier"]


    preselect_m = int(gp_cfg.get("top_m", 5))

    cc_radius = int(cc_cfg.get("radius", 1))
    cc_top_p = float(cc_cfg.get("top_p", 0.05))
    cc_alpha = float(cc_cfg.get("alpha", 0.6))
    cc_min_cut = float(cc_cfg.get("min_cut", 0.20))
    cc_singleton_weight = float(cc_cfg.get("singleton_weight", 0.25))
    cc_component_min_area = int(cc_cfg.get("component_min_area", 2))

    proposal_top_k = int(proposal_cfg.get("top_k", 5))
    patch_margin = int(proposal_cfg.get("patch_margin", 1))
    crop_margin_ratio = float(proposal_cfg.get("crop_margin_ratio", 0.20))
    min_patch_area = int(proposal_cfg.get("min_patch_area", 2))
    min_crop_size = int(proposal_cfg.get("min_crop_size", 96))

    ver_radius = int(ver_cfg.get("radius", 1))
    ver_top_p = float(ver_cfg.get("top_p", 0.10))

    tensor_device = torch.device(device)


    # 1. preselect 모드에 맞추어 emb를 추출
    preselect_mode = gp_cfg.get("mode", "dino")

    if preselect_mode == "dino":
        if q_out is None or "global" not in q_out:
            raise RuntimeError("DINO preselect인데 q_out['global']이 없습니다.")
        q_preselect_emb = q_out["global"]

    elif preselect_mode == "vpr":
        if vpr_model is None:
            raise RuntimeError("VPR mode인데 vpr_model이 없습니다.")
        q_preselect_emb = vpr_model.encode_image(q_bgr)
        if isinstance(q_preselect_emb, torch.Tensor):
            q_preselect_emb = q_preselect_emb.detach().cpu().numpy()

    else:
        raise ValueError(f"unknown preselect mode: {preselect_mode}")

    # 2. preselct emb로 상위 유사도 preselect_m개의 inx를 return
    cand_ref_idx, ref_paths  = run_preselect(
        q_preselect_emb=q_preselect_emb,
        ref_bank=ref_bank,
        mode=preselect_mode,
        top_m=preselect_m,
    )

    cand_scores = []
    cand_infos = []

    # 3. preselect된 ref들에 대해서 cc점수화 (각 이미지에서 max cc의 점수를 기준으로 정상 이미지중 max cc가 제일 낮은 이미지를 best ref로 ) ----------------
    for ref_i in cand_ref_idx:
        r_path = ref_paths[ref_i]
        r_bgr = cv2.imread(str(r_path), cv2.IMREAD_COLOR)
        if r_bgr is None:
            continue

        score, debug = score_one_pair(
            q_bgr=q_bgr,
            r_bgr=r_bgr,
            sg=sg_matcher,
            backbone=cc_backbone,
            device=device,
            radius=cc_radius,
            top_p=cc_top_p,
            alpha=cc_alpha,
            min_cut=cc_min_cut,
            singleton_weight=cc_singleton_weight,
            component_min_area=cc_component_min_area,
        )
        if score is None:
            continue

        cand_scores.append(float(score))
        cand_infos.append({
            "ref_i": int(ref_i),
            "r_path": r_path,
            "score": float(score),
            "debug": debug,
        })

    if len(cand_scores) == 0:
        return {
            "ok": False,
            "reason": "all_ref_failed",
            "ref_paths": ref_paths,
        }

    # best ref 선택 ( argmin )
    best_idx = int(np.argmin(cand_scores))
    best_score = float(cand_scores[best_idx])
    best_info = cand_infos[best_idx]
    best_debug = best_info["debug"]

    all_comps = best_debug.get("all_comp_scores", [])
    topk_comps = sorted(
        all_comps,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )[:proposal_top_k] # best ref와의 cc 결과에서 proposal_top_k개의 cc를 local verification

    # cc 영역을 bbox crop으로 변환
    flagged_regions = build_flagged_component_regions(
        flagged_comps=topk_comps,
        q_crop=best_debug["q_crop"],
        r_crop=best_debug["r_crop"],
        valid_mask=best_debug["valid_mask"],
        patch_margin=patch_margin,
        crop_margin_ratio=crop_margin_ratio,
        min_patch_area=min_patch_area,
        min_crop_size=min_crop_size,
    )

    verifier_results = []
    for reg in flagged_regions:
        out = verify_bbox_with_local_search(
            q_region=reg["q_region"],
            r_region=reg["r_region"],
            backbone=verifier_backbone,
            device=device,
            radius=ver_radius,
            top_p=ver_top_p,
        )
        verifier_results.append({
            **reg,
            "verifier_score": float(out["score"]),
            "verifier_feat_hw": out["feat_hw"],
            "verifier_top_p_thr": out["top_p_thr"],
            "verifier_top_k": out["top_k"],
            "verifier_top_p": out["top_p"],
        })

    verifier_scores_sorted = sorted(
        [r["verifier_score"] for r in verifier_results],
        reverse=True,
    )
    final_score = float(max(verifier_scores_sorted)) if verifier_scores_sorted else 0.0

    # infer_event 호환용 top-k
    cand_infos_sorted = sorted(cand_infos, key=lambda x: x["score"])
    topk_infos = cand_infos_sorted[:max(1, min(k, len(cand_infos_sorted)))]
    topk_score = torch.tensor(
        [float(x["score"]) for x in topk_infos],
        device=tensor_device,
        dtype=torch.float32,
    )
    topk_idx = torch.tensor(
        [int(x["ref_i"]) for x in topk_infos],
        device=tensor_device,
        dtype=torch.long,
    )
    best_img_idx = torch.tensor(
        int(best_info["ref_i"]),
        device=tensor_device,
        dtype=torch.long,
    )

    debug = {
        "fallback": False,
        "cc_score": float(best_score),
        "verifier_scores": verifier_scores_sorted,
        "flagged_regions": [
            {
                "score": float(r["score"]),
                "verifier_score": float(r["verifier_score"]),
                "img_bbox": tuple(map(int, r["img_bbox"])),
                "patch_bbox": tuple(map(int, r["patch_bbox"])),
                "area": int(r["area"]),
                "peak": float(r["peak"]),
                "mean": float(r["mean"]),
                "verifier_top_p": float(r.get("verifier_top_p", 0.0)),
                "verifier_top_k": int(r.get("verifier_top_k", 0)),
                "verifier_top_p_thr": float(r.get("verifier_top_p_thr", 0.0)),
            }
            for r in verifier_results
        ],
        "global_topk": (None, None),
        "patch_topk": {
            "best_ref_img_path": str(best_info["r_path"]),
            "topk_score": topk_score,
            "topk_idx": topk_idx,
            "best_img_idx": best_img_idx,
            "best_patch_dist": torch.empty(0, device=tensor_device, dtype=torch.float32),
            "top_patch_idx": torch.empty(0, device=tensor_device, dtype=torch.long),
            "top_patch_vals": torch.empty(0, device=tensor_device, dtype=torch.float32),
            "top_patch_match_idx": torch.empty(0, device=tensor_device, dtype=torch.long),
        },
        "align_debug": {
            "crop_bbox": best_debug.get("bbox"),
            "grid_hw": list(best_debug["valid_mask"].shape[:2]) if "valid_mask" in best_debug else None,
            "crop_shape": list(best_debug["q_crop"].shape[:2]) if "q_crop" in best_debug else None,
        },
        "best_debug": best_debug,
    }

    return {
        "ok": True,
        "score": float(final_score),   # 최종 score = verifier max
        "cc_score": float(best_score),
        "best_debug": best_debug,
        "verifier_results": verifier_results,
        "debug": debug,
        "ref_paths": ref_paths,
    }


# =========================================================================================== dist 계산 함수


# ------------------------------------------------------------------------------------------------------------
# 각 모드에 맞추어 knn dist를 return
@torch.inference_mode()
def compute_knn_dist(
    q_out,
    ref_bank,
    k=3,
    repr_mode="global",
    top_p=0.1,
    preselect_m=10,
    q_img_bgr=None,
    cc_backbone=None,
    verifier_backbone=None,
    device=None,
    sg_matcher=None,
    vpr_model=None,
    cfg=None,
):
    if cfg is None:
        raise ValueError("cfg required")

    cfgb = get_cfg_bundle(cfg)

    # ---------------------------------------------------
    # global preselect + patch
    # ---------------------------------------------------
    if repr_mode == "global_patch":
        patch_cfg = cfgb["mode_cfg"].get("patch_score", {})
        pre_cfg = cfgb["preselect"]

        alpha = float(patch_cfg.get("alpha", 0.6))
        top_p = float(patch_cfg.get("top_p", top_p))
        preselect_m = int(pre_cfg.get("top_m", preselect_m))

        refp_np, _ = ref_bank["patch"]

        qp = q_out["patch"]
        q_preselect_emb = q_out["global"]

        cand_ref_idx, ref_paths = run_preselect(
            q_preselect_emb=q_preselect_emb,
            ref_bank=ref_bank,
            mode=pre_cfg.get("mode", "dino"),
            top_m=preselect_m,
        )

        idx = torch.tensor(
            cand_ref_idx,
            device=qp.device,
            dtype=torch.long,
        )

        refp = torch.from_numpy(refp_np).float().to(qp.device)
        refp_sel = refp[idx]

        score, patch_debug_local = _dist_patchcore(
            qp,
            refp_sel,
            top_p=top_p,
            k=min(k, refp_sel.shape[0]),
            alpha=alpha,
        )

        topk_idx_local = patch_debug_local["topk_idx"]
        topk_idx_global = idx[topk_idx_local]
        best_img_idx_global = idx[patch_debug_local["best_img_idx"]]

        debug = {
            "global_topk": (None, idx),
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
        }

        return score, debug, ref_paths

    # ---------------------------------------------------
    # global + aligned local CNN
    # ---------------------------------------------------
    elif repr_mode == "global_patch_with_aligned":
        if sg_matcher is None:
            raise ValueError("sg_matcher required")
        if q_img_bgr is None:
            raise ValueError("q_img_bgr required")
        if cc_backbone is None or verifier_backbone is None:
            raise ValueError("cc_backbone/verifier_backbone required")
        if cfg is None:
            raise ValueError("cfg required")

        out = _run_gpa_frame(
            q_bgr=q_img_bgr,
            q_out=q_out,
            ref_bank=ref_bank,
            cfg=cfg,
            sg_matcher=sg_matcher,
            cc_backbone=cc_backbone,
            verifier_backbone=verifier_backbone,
            vpr_model=vpr_model,
            device=device,
            k=k,
        )

        if not out["ok"]:
            debug = {
                "fallback": False,
                "reason": out.get("reason", "unknown"),
                "cc_score": 0.0,
                "verifier_scores": [],
                "flagged_regions": [],
                "patch_topk": {
                    "best_ref_img_path": None,
                    "topk_score": torch.empty(0, dtype=torch.float32),
                    "topk_idx": torch.empty(0, dtype=torch.long),
                    "best_img_idx": torch.tensor(0, dtype=torch.long),
                    "best_patch_dist": torch.empty(0, dtype=torch.float32),
                    "top_patch_idx": torch.empty(0, dtype=torch.long),
                    "top_patch_vals": torch.empty(0, dtype=torch.float32),
                    "top_patch_match_idx": torch.empty(0, dtype=torch.long),
                },
                "align_debug": None,
            }
            return 0.0, debug, out["ref_paths"]

        return float(out["score"]), out["debug"], out["ref_paths"]

    else:
        raise ValueError(f"Unknown repr_mode: {repr_mode}")

# ====================================================================================== threshold

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


def calibrate_place(bank_root, plc_idx, global_model, cc_backbone, verifier_backbone ,device, sg_matcher=None, cfg=None, vpr_model=None):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    if cfg is None:
        cfg = load_cfg(bank_root)

    # bank rebuild 
    rebuild_bank(bank_root, plc_idx, global_model, device, mode="bank", cfg=cfg, vpr_model=vpr_model)
    rebuild_bank(bank_root, plc_idx, global_model, device, mode="th_calib", cfg=cfg, vpr_model=vpr_model)

    # threshold 계산
    thr, scores, thr_path = compute_and_save_threshold(
        bank_root,
        plc_idx,
        cfg=cfg,
        global_model=global_model,   
        cc_backbone=cc_backbone,
        verifier_backbone=verifier_backbone,     
        device=device,
        sg_matcher=sg_matcher,
        vpr_model = vpr_model
        )

    return thr, scores, thr_path

@torch.inference_mode()
def compute_and_save_threshold(
    bank_root,
    plc_idx,
    cfg: dict,
    global_model=None,
    cc_backbone=None,
    verifier_backbone=None,
    device=None,
    sg_matcher=None,
    vpr_model=None,
):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    cfgb = get_cfg_bundle(cfg)

    repr_mode = cfgb["repr_mode"]
    threshold_cfg = cfgb["threshold_cfg"]
    calib_cfg = cfgb["calibration"]
    mode_cfg = cfgb["mode_cfg"]
    pre_cfg = cfgb["preselect"]

    method = str(threshold_cfg.get("method", "percentile")).lower()
    k = int(threshold_cfg.get("topk_neighbors", 3))

    percentile_value = float(threshold_cfg.get("percentile_value", 97))
    robust_std_k = float(threshold_cfg.get("robust_std_k", 2.5))
    gaussian_std_k = float(threshold_cfg.get("gaussian_std_k", 2.5))

    if method == "percentile":
        default_param = percentile_value
    elif method == "robust":
        default_param = robust_std_k
    elif method == "gaussian":
        default_param = gaussian_std_k
    else:
        raise ValueError(method)

    threshold_param = float(calib_cfg.get("threshold_param", default_param))
    threshold_floor = float(calib_cfg.get("threshold_floor", 0.0))
    max_imgs = int(calib_cfg.get("max_calib_images", 0))

    # ---------------------------------------------------
    # global_patch
    # ---------------------------------------------------
    if repr_mode == "global_patch":
        patch_cfg = mode_cfg.get("patch_score", {})
        top_p = float(patch_cfg.get("top_p", 0.1))
        alpha = float(patch_cfg.get("alpha", 0.6))
        preselect_m = int(pre_cfg.get("top_m", 10))
        preselect_mode = str(pre_cfg.get("mode", "dino"))

        bank_g_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
        th_g_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
        bank_p_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
        th_p_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

        bank_g = torch.from_numpy(np.load(bank_g_npz, allow_pickle=True)["embs"]).float()
        th_g   = torch.from_numpy(np.load(th_g_npz, allow_pickle=True)["embs"]).float()
        bank_p = torch.from_numpy(np.load(bank_p_npz, allow_pickle=True)["embs"]).float()
        th_p   = torch.from_numpy(np.load(th_p_npz, allow_pickle=True)["embs"]).float()

        # dino preselect만 여기선 지원
        if preselect_mode != "dino":
            raise ValueError(f"global_patch calibration currently supports only dino preselect, got: {preselect_mode}")

        bank_g = F.normalize(bank_g, dim=1)
        th_g = F.normalize(th_g, dim=1)

        scores = []
        for i in range(th_g.shape[0]):
            sim = th_g[i:i+1] @ bank_g.T
            M = min(preselect_m, bank_g.shape[0])
            _, idx = torch.topk(sim, k=M, dim=1)
            idx = idx.squeeze(0)

            bank_sel = bank_p[idx]
            s, _ = _dist_patchcore(
                th_p[i],
                bank_sel,
                top_p=top_p,
                k=k,
                alpha=alpha,
            )
            scores.append(float(s))

        scores = np.asarray(scores, dtype=np.float32)

        if method == "robust":
            thr = max(th_robust(scores, k=threshold_param), threshold_floor)
        elif method == "gaussian":
            thr = max(th_gaussian(scores, k=threshold_param), threshold_floor)
        elif method == "percentile":
            thr = max(th_percentile(scores, threshold_param), threshold_floor)
        else:
            raise ValueError(method)

    # ---------------------------------------------------
    # global_patch_with_aligned
    # ---------------------------------------------------
    elif repr_mode == "global_patch_with_aligned":
        if sg_matcher is None:
            raise ValueError("sg_matcher required")
        if global_model is None or cc_backbone is None or verifier_backbone is None or device is None:
            raise ValueError("global_model/cc_backbone/verifier_backbone/device required")

        e = cfg.get("embed", {})
        img_size = int(e.get("img_size", 560))
        global_mode = str(e.get("global_mode", "patch_mean"))

        tfm = make_transform(img_size=img_size)
        ref_bank = load_bank_by_place(bank_root, plc_idx, mode="bank")

        th_dir = bank_root / plc_idx / "th_calib"
        th_imgs = sorted([
            p for p in th_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])

        min_th_calib_images = int(calib_cfg.get("min_th_calib_images", 5))
        if len(th_imgs) < min_th_calib_images:
            print(
                f"[WARN] th_calib images too few: "
                f"{len(th_imgs)} < {min_th_calib_images} → skip calib, use floor"
            )

            scores = np.asarray([], dtype=np.float32)
            thr = float(threshold_floor)
            method = "floor_fallback"

            created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

            out = {
                "plc_idx": plc_idx,
                "repr_mode": repr_mode,
                "k": k,
                "method": method,
                "threshold": float(thr),
                "num_th": 0,
                "created_at": created_at,
            }

            thr_path = bank_root / plc_idx / "threshold.json"
            thr_path.write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            return float(thr), scores, thr_path

        if max_imgs > 0:
            th_imgs = th_imgs[:max_imgs]

        preselect_mode = str(pre_cfg.get("mode", "dino"))
        final_scores = []

        for q_path in th_imgs:
            q_bgr = cv2.imread(str(q_path), cv2.IMREAD_COLOR)
            if q_bgr is None:
                continue

            q_out = None
            if preselect_mode == "dino":
                x = tfm(BGR_to_RGB(q_bgr))
                q_out = make_embed(
                    global_model,
                    device,
                    x,
                    repr_mode="global",
                    global_mode=global_mode,
                )

            out = _run_gpa_frame(
                q_bgr=q_bgr,
                q_out=q_out,
                ref_bank=ref_bank,
                cfg=cfg,
                sg_matcher=sg_matcher,
                cc_backbone=cc_backbone,
                verifier_backbone=verifier_backbone,
                vpr_model=vpr_model,
                device=device,
                k=1,
            )

            if not out["ok"]:
                continue

            final_scores.append(float(out["score"]))

            ver_scores = out["debug"].get("verifier_scores", [])
            flagged_regions = out["debug"].get("flagged_regions", [])

            print(
                f"[CALIB] {q_path.name} | "
                f"score={float(out['score']):.4f} | "
                f"n_regions={len(flagged_regions)} | "
                f"ver_max={(max(ver_scores) if len(ver_scores) > 0 else -1):.4f}"
            )

        if len(final_scores) == 0:
            print("[WARN] all calib failed → fallback to threshold_floor")

            scores = np.asarray([], dtype=np.float32)
            thr = float(threshold_floor)
            method = "floor_fallback"

        scores = np.asarray(final_scores, dtype=np.float32)

        if method == "robust":
            thr = max(th_robust(scores, k=threshold_param), threshold_floor)
        elif method == "gaussian":
            thr = max(th_gaussian(scores, k=threshold_param), threshold_floor)
        elif method == "percentile":
            thr = max(th_percentile(scores, threshold_param), threshold_floor)
        else:
            raise ValueError(method)

    else:
        raise ValueError(f"Unknown repr_mode: {repr_mode}")

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    out = {
        "plc_idx": plc_idx,
        "repr_mode": repr_mode,
        "k": k,
        "method": method,
        "threshold": float(thr),
        "num_th": int(len(scores)),
        "created_at": created_at,
    }

    thr_path = bank_root / plc_idx / "threshold.json"
    thr_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return float(thr), scores, thr_path

# ================================================================================= 추론

# event 단위 이상감지
@torch.inference_mode()
def infer_event(
    imgs_bgr: List[np.ndarray],
    plc_idx: str,
    bank_root,
    global_model,
    cc_backbone,
    verifier_backbone,
    device,
    sg_matcher=None,
    vpr_model=None,
    cfg=None,
):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    if cfg is None:
        cfg = load_cfg(bank_root)

    cfgb = get_cfg_bundle(cfg)

    img_size = int(cfg.get("embed", {}).get("img_size", 560))
    global_mode = str(cfg.get("embed", {}).get("global_mode", "patch_mean"))
    event_rule = str(cfg.get("infer", {}).get("event_rule", "max")).lower()
    use_two_stage_vlm = bool(cfg.get("infer", {}).get("use_two_stage_vlm", False))

    tfm = make_transform(img_size)

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

    repr_mode = str(meta.get("repr_mode", cfgb["repr_mode"])).lower()
    if repr_mode not in {"global_patch", "global_patch_with_aligned"}:
        raise ValueError(
            f"Unknown repr_mode={repr_mode} "
            "(use global_patch|global_patch_with_aligned)"
        )

    mode_cfg = cfg.get("modes", {}).get(repr_mode, {})
    pre_cfg = mode_cfg.get("preselect", {})

    top_p = 0.1
    preselect_m = int(pre_cfg.get("top_m", 10))

    if repr_mode == "global_patch":
        patch_cfg = mode_cfg.get("patch_score", {})
        top_p = float(patch_cfg.get("top_p", 0.1))

    preselect_mode = str(pre_cfg.get("mode", "dino"))

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
    patch_vis_all: List[Optional[Dict[str, Any]]] = []
    align_vis_all: List[Optional[Dict[str, Any]]] = []

    for fi, img_bgr in enumerate(imgs_bgr):
        q_out = None

        # aligned + vpr 일 때만 DINO query embedding 생략
        need_query_embed = not (
            repr_mode == "global_patch_with_aligned" and preselect_mode == "vpr"
        )

        if need_query_embed:
            x = tfm(BGR_to_RGB(img_bgr))
            q_out = make_embed(
                global_model,
                device,
                x,
                repr_mode=repr_mode if repr_mode == "global_patch" else "global",
                global_mode=global_mode,
            )

        dist, debug, ref_paths = compute_knn_dist(
            q_out,
            ref_bank,
            k=k,
            repr_mode=repr_mode,
            top_p=top_p,
            preselect_m=preselect_m,
            q_img_bgr=img_bgr,
            cc_backbone=cc_backbone,
            verifier_backbone=verifier_backbone,
            device=device,
            sg_matcher=sg_matcher,
            vpr_model=vpr_model,
            cfg=cfg,
        )

        is_change = dist > thr
        frame_scores.append(float(dist))
        frame_change_flags.append(1 if is_change else 0)

        if repr_mode == "global_patch_with_aligned":
            align_vis_all.append(debug.get("align_debug", None))
        else:
            align_vis_all.append(None)

        if repr_mode in {"global_patch", "global_patch_with_aligned"}:
            topk_score = debug["patch_topk"]["topk_score"]
            topk_idx = debug["patch_topk"]["topk_idx"]
            idx = topk_idx.tolist()
            paths = [ref_paths[i] for i in idx]
            sims = (-topk_score).tolist()
        else:
            idx, sims, paths = [], [], []

        topk_paths_all.append([str(p) for p in paths])
        topk_sims_all.append([float(s) for s in sims])

        if repr_mode in {"global_patch", "global_patch_with_aligned"}:
            top_patch_idx, top_patch_vals = get_top_p_patch_info(repr_mode, debug)

            align_data = (debug.get("align_debug") or {}) if repr_mode == "global_patch_with_aligned" else {}

            patch_vis_all.append({
                "top_patch_idx": top_patch_idx.detach().cpu().tolist()
                if isinstance(top_patch_idx, torch.Tensor) else [],
                "top_patch_vals": top_patch_vals.detach().cpu().tolist()
                if isinstance(top_patch_vals, torch.Tensor) else [],
                "top_patch_match_idx": (
                    debug["patch_topk"]["top_patch_match_idx"].detach().cpu().tolist()
                    if "patch_topk" in debug and "top_patch_match_idx" in debug["patch_topk"]
                    and isinstance(debug["patch_topk"]["top_patch_match_idx"], torch.Tensor)
                    else []
                ),
                "best_ref_img_path": (
                    debug["patch_topk"].get("best_ref_img_path", None)
                    if "patch_topk" in debug else None
                ),
                "repr_mode": repr_mode,
                "img_size": img_size,
                "H": align_data.get("H", None),
                "crop_bbox": align_data.get("crop_bbox", None),
                "grid_hw": align_data.get("grid_hw", None),
                "crop_shape": align_data.get("crop_shape", None),
                "vis_query_bgr": align_data.get("vis_query_bgr", None),
                "vis_ref_bgr": align_data.get("vis_ref_bgr", None),
            })
        else:
            patch_vis_all.append(None)

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
        n_ab = int(sum(frame_change_flags))
        n = int(len(frame_change_flags))
        anomaly_flag = 1 if n_ab > (n / 2) else 0
        decision_score = float(n_ab / max(n, 1))

    else:
        raise ValueError(f"Unknown event_rule: {event_rule}")

    margin = decision_score - thr
    event_score = float(np.clip(
        50.0 + 50.0 * (margin / max(thr, 1e-8)),
        0.0,
        100.0
    ))

    rep_idx = int(np.argmax(frame_scores)) if len(frame_scores) > 0 else 0

    if anomaly_flag == 1:
        rep, summary = None, "anomaly detect"
    else:
        rep, summary = None, "it's fine, have relax"

    if use_two_stage_vlm and anomaly_flag == 1:
        pass

    ref_topk_json = json.dumps(
        {"topk_paths": topk_paths_all, "topk_sims": topk_sims_all, "rep": rep},
        ensure_ascii=False,
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
            "patch_size": (14 if repr_mode != "global_patch_with_aligned" else None),

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
        },
    }