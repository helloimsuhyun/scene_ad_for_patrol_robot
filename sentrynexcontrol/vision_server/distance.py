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
from .config import load_cfg

import cv2

from .distance_util import (
    score_one_pair,
    build_flagged_component_regions,
    verify_bbox_with_local_search,
)


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

# =========================================================================================== dist 계산 함수

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

def run_preselect(q_out, q_img_bgr, ref_bank, mode, top_m, vpr_model=None):
    refg_np, ref_paths = ref_bank["global"]
    qg = q_out["global"]

    if mode == "dino":
        refg = torch.from_numpy(refg_np).float().to(qg.device)

        _, (_, topk_idx) = _dist_global(
            qg,
            refg,
            min(top_m, refg.shape[0]),
        )

        return topk_idx.squeeze(0).tolist()

    elif mode == "vpr":
        if vpr_model is None:
            raise ValueError("vpr_model required")

        ref_embs_np, ref_paths = ref_bank["global"]

        q_emb = vpr_model.encode_image(q_img_bgr)   
        q_emb = q_emb.detach().cpu().numpy()

        sims = q_emb @ ref_embs_np.T
        idx = np.argsort(-sims)[:top_m]

        return idx.tolist()

    else:
        raise ValueError(mode)

@torch.inference_mode()
def _run_gpa_frame(
    q_bgr: np.ndarray,
    q_out: dict,
    ref_bank,
    cfg: dict,
    sg_matcher,
    cc_backbone,
    verifier_backbone,
    device: str,
    k: int,
):
    mode_cfg = cfg.get("repr_modes", {}).get("global_patch_with_aligned", {})
    gp_cfg = mode_cfg.get("global_preselect", {})
    cc_cfg = mode_cfg.get("cc", {})
    proposal_cfg = mode_cfg.get("proposal", {})
    ver_cfg = mode_cfg.get("verifier", {})

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

    refg_np, ref_paths = ref_bank["global"]
    qg = q_out["global"]
    refg = torch.from_numpy(refg_np).float().to(qg.device)

    preselect_mode = gp_cfg.get("mode", "dino")

    cand_ref_idx = run_preselect(
        q_out=q_out,
        q_img_bgr=q_bgr,
        ref_bank=ref_bank,
        mode=preselect_mode,
        top_m=preselect_m,
        vpr_model=cfg.get("vpr_model", None),
    )

    cand_scores = []
    cand_infos = []

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

    # 실험용과 동일: argmin 선택
    best_idx = int(np.argmin(cand_scores))
    best_score = float(cand_scores[best_idx])
    best_info = cand_infos[best_idx]
    best_debug = best_info["debug"]

    all_comps = best_debug.get("all_comp_scores", [])
    topk_comps = sorted(
        all_comps,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )[:proposal_top_k]

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
        device=qg.device,
        dtype=torch.float32,
    )
    topk_idx = torch.tensor(
        [int(x["ref_i"]) for x in topk_infos],
        device=qg.device,
        dtype=torch.long,
    )
    best_img_idx = torch.tensor(int(best_info["ref_i"]), device=qg.device, dtype=torch.long)

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
            "best_patch_dist": torch.empty(0, device=qg.device, dtype=torch.float32),
            "top_patch_idx": torch.empty(0, device=qg.device, dtype=torch.long),
            "top_patch_vals": torch.empty(0, device=qg.device, dtype=torch.float32),
            "top_patch_match_idx": torch.empty(0, device=qg.device, dtype=torch.long),
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
    q_out,                 # DINO query embedding output
    ref_bank,              # load_bank_by_place 결과
    k=3,
    repr_mode="global",
    top_p=0.1,
    preselect_m=10,
    q_img_bgr=None,
    global_model=None,     # DINO model
    cc_backbone=None,      # CNN model
    verifier_backbone=None,
    device=None,           # 하나만 사용
    sg_matcher=None,
    cfg=None,
):
    pcfg = cfg.get("patchcore", {})
    alpha = float(pcfg.get("alpha", 0.6))

    # ---------------------------------------------------
    # global
    # ---------------------------------------------------
    if repr_mode == "global":
        ref_embs_np, ref_paths = ref_bank["global"]
        q = q_out["global"]
        ref = torch.from_numpy(ref_embs_np).float().to(q.device)

        dist, debug_inner = _dist_global(q, ref, k)

        debug = {
            "inner_debug": debug_inner,
        }
        return dist, debug, ref_paths

    # ---------------------------------------------------
    # patch
    # ---------------------------------------------------
    elif repr_mode == "patch":
        ref_patch_np, ref_paths = ref_bank["patch"]
        q_patch = q_out["patch"]
        ref_patch = torch.from_numpy(ref_patch_np).float().to(q_patch.device)

        score, debug = _dist_patchcore(
            q_patch,
            ref_patch,
            top_p=top_p,
            k=k,
            alpha=alpha,
            )

        return score, debug, ref_paths

    # ---------------------------------------------------
    # global + patch
    # ---------------------------------------------------
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
        }

        return score, debug, ref_paths

    # ---------------------------------------------------
    # global + aligned local CNN

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


def calibrate_place(bank_root, plc_idx, global_model, cc_backbone, verifier_backbone ,device, sg_matcher=None, cfg=None):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    if cfg is None:
        cfg = load_cfg(bank_root)

    # bank rebuild → DINO 기준
    rebuild_bank(bank_root, plc_idx, global_model, device, mode="bank", cfg=cfg)

    rebuild_bank(bank_root, plc_idx, global_model, device, mode="th_calib", cfg=cfg)

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
        )

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
    cc_backbone=None,    # CNN
    verifier_backbone=None,
    device=None,
    sg_matcher=None,
):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)


    r = cfg.get("repr", {})
    c = cfg.get("calib", {})

    pcfg = cfg.get("patchcore", {})
    top_p = float(pcfg.get("top_p", 0.1))
    preselect_m = int(pcfg.get("preselect_m", 10))
    alpha = float(pcfg.get("alpha", 0.6))


    repr_mode = str(r.get("repr_mode", "global"))

    k          = int(c.get("k", 3))
    method     = str(c.get("method", "robust"))
    robust_k   = float(c.get("robust_k", 2.5))

    if repr_mode == "global":
        bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
        th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"
        bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()
        th   = torch.from_numpy(np.load(th_npz, allow_pickle=True)["embs"]).float()
        th   = F.normalize(th, dim=1)

        k2 = min(k, bank.shape[0])
        sim = th @ bank.T
        topk_sim, _ = torch.topk(sim, k=k2, dim=1)
        scores = (1.0 - topk_sim).mean(dim=1).numpy()

    elif repr_mode == "patch":
        bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}_patch_.npz"
        th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}_patch_.npz"

        bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()
        th   = torch.from_numpy(np.load(th_npz,   allow_pickle=True)["embs"]).float()

        scores = []
        for i in range(th.shape[0]):
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
            scores.append(s)

        scores = np.array(scores, dtype=np.float32)

    elif repr_mode == "global_patch_with_aligned":
        if sg_matcher is None:
            raise ValueError("sg_matcher required")
        if global_model is None or cc_backbone is None or verifier_backbone is None or device is None:
            raise ValueError("global_model/cc_backbone/verifier_backbone/device required")

        mode_cfg = cfg.get("repr_modes", {}).get("global_patch_with_aligned", {})
        calib_cfg = mode_cfg.get("calibration", {})

        final_k = float(calib_cfg.get("final_k", robust_k))
        final_threshold_floor = float(calib_cfg.get("final_threshold_floor", 0.0))
        max_imgs = int(calib_cfg.get("max_imgs", 0))

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

        if max_imgs > 0:
            th_imgs = th_imgs[:max_imgs]

        final_scores = []

        for q_path in th_imgs:
            q_bgr = cv2.imread(str(q_path), cv2.IMREAD_COLOR)
            if q_bgr is None:
                continue

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
                device=device,
                k=1,
            )

            if not out["ok"]:
                continue

            final_scores.append(float(out["score"]))

            # (옵션) 로그
            ver_scores = out["debug"].get("verifier_scores", [])
            flagged_regions = out["debug"].get("flagged_regions", [])

            print(
                f"[CALIB] {q_path.name} | "
                f"score={float(out['score']):.4f} | "
                f"n_regions={len(flagged_regions)} | "
                f"ver_max={(max(ver_scores) if len(ver_scores) > 0 else -1):.4f}"
            )

        if len(final_scores) == 0:
            raise RuntimeError("aligned calib failed: no scores collected")

        final_scores = np.asarray(final_scores, dtype=np.float32)

        # ✅ threshold 계산
        if method == "robust":
            thr = max(th_robust(final_scores, k=final_k), final_threshold_floor)
        elif method == "gaussian":
            thr = max(th_gaussian(final_scores, k=final_k), final_threshold_floor)
        elif method == "percentile":
            thr = max(th_percentile(final_scores, final_k), final_threshold_floor)
        else:
            raise ValueError(method)

        scores = final_scores
        created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ✅ 깔끔한 threshold.json
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
    imgs_bgr,
    bank_root,
    plc_idx,
    cfg: dict = None,
    global_model=None,
    cc_backbone=None,
    verifier_backbone=None,
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


    # -------------------------
    # cfg
    # -------------------------
    if cfg is None:
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
    if repr_mode not in {"global", "patch", "global_patch", "global_patch_with_aligned"}:
        raise ValueError(
            f"Unknown repr_mode={repr_mode} "
            "(use global|patch|global_patch|global_patch_with_aligned)"
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
    patch_vis_all: List[Optional[Dict[str, Any]]] = []
    align_vis_all: List[Optional[Dict[str, Any]]] = []

    for fi, img_bgr in enumerate(imgs_bgr):

        x = tfm(BGR_to_RGB(img_bgr))

        # global embedding은 DINO
        q_out = make_embed(
            global_model,
            device,
            x,
            repr_mode=repr_mode,
            global_mode=global_mode,
        )

        # distance 계산: global=DINO, local aligned=CNN
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
            cfg=cfg,
        )

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
            sims = [float(x) for x in topk_score.tolist()]

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
    }