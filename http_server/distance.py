# distance.py
# 1. threshold 캘리브레이션 : def calibrate_place(bank_root, plc_idx, model, device):
# 2. event단위 q 이미지에 대한 추론 : def infer_event(imgs_bgr: List[np.ndarray],plc_idx: str,bank_root,model,device)
# - suhyun
# 파라미터 설정은 config.py에서 여기 파일에 의해서만 의존함

import hashlib  
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from banker import load_bank_by_place, BGR_to_RGB, rebuild_bank
from dino_emb import make_embed , make_transform
from vlm_gate import vlm_gate
from config import load_cfg

# vis helper

def get_top_p_patch_info(repr_mode: str, debug: dict):
    if repr_mode == "global_patch_pool":
        pool_debug = debug["pool_debug"]
        return pool_debug["top_patch_idx"], pool_debug["top_vals"]

    elif repr_mode == "global_patch":
        patch_topk = debug["patch_topk"]
        return patch_topk["top_patch_idx"], patch_topk["top_patch_vals"]

    else:
        return None, None



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
    ref = F.normalize(ref, dim=1)

    k = min(k, ref.shape[0])
    sim = q @ ref.T                       # (1,N)
    topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)
    dist = (1.0 - topk_sim).mean().item()
    return dist, (topk_sim, topk_idx)

#patchcore dist 계산함수
def _dist_patchcore(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
    k: int = 3,
):
    q = F.normalize(q_patch, dim=1)
    ref = F.normalize(ref_patch, dim=2)

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


def _dist_patch_pool(
    q_patch: torch.Tensor,      # (P,D)
    ref_patch: torch.Tensor,    # (N,P,D)
    top_p: float = 0.1,
):

    q = F.normalize(q_patch, dim=1)          # (P, D)
    ref = F.normalize(ref_patch, dim=2)      # (N, P, D)

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
    q_out, #q에서 나온 임베팅
    ref_bank, #bank(npz)임베딩
    k=3,
    repr_mode="global",
    top_p=0.1,
    preselect_m=10,
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

    else:
        raise ValueError("repr_mode must be global|patch|global_patch|global_patch_pool")

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

@torch.no_grad()
def compute_and_save_threshold(
    bank_root,
    plc_idx,
    cfg: dict,
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
        bank = F.normalize(bank, dim=1)
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

        bank_g = F.normalize(bank_g, dim=1)
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

        bank_g = F.normalize(bank_g, dim=1)
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


def calibrate_place(bank_root, plc_idx, model, device):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    cfg = load_cfg(bank_root)

    rebuild_bank(bank_root, plc_idx, model, device, mode="bank",     cfg=cfg)
    rebuild_bank(bank_root, plc_idx, model, device, mode="th_calib", cfg=cfg)

    thr, scores, thr_path = compute_and_save_threshold(bank_root, plc_idx, cfg=cfg)    
    return thr, scores, thr_path


@torch.no_grad()
def infer_one(img, plc_idx, bank_root, model, device):

    plc_idx = str(plc_idx)
    bank_root = Path(bank_root)

    meta = json.loads(
        (bank_root / plc_idx / "threshold.json").read_text()
    )

    thr = float(meta["threshold"])
    k = int(meta["k"])
    repr_mode = meta["repr_mode"]
    ref_bank_id = meta["ref_bank_id"]

    cfg = load_cfg(bank_root)
    embed_cfg = cfg.get("embed", {})

    img_size = int(embed_cfg.get("img_size", 560))
    global_mode = embed_cfg.get("global_mode", "patch_mean")

    tfm = make_transform(img_size=img_size)
    x = tfm(BGR_to_RGB(img))

    q_out = make_embed(
        model,
        device,
        x,
        repr_mode=repr_mode,
        global_mode=global_mode,
    )

    ref_bank = load_bank_by_place(bank_root, plc_idx)

    pcfg = cfg.get("patchcore", {})

    top_p = pcfg.get("top_p", 0.1)
    preselect_m = pcfg.get("preselect_m", 10)

    dist, debug, ref_paths = compute_knn_dist(
        q_out,
        ref_bank,
        k=k,
        repr_mode=repr_mode,
        top_p=top_p,
        preselect_m=preselect_m,
    )

    is_change = dist > thr

    # --------------------
    # topk logging
    # --------------------

    if repr_mode == "global":
        topk_sim, topk_idx = debug
        idx = topk_idx.squeeze(0).tolist()
        sims = topk_sim.squeeze(0).tolist()

    elif repr_mode == "patch":
        topk_score, topk_idx = debug
        idx = topk_idx.tolist()
        sims = (-topk_score).tolist()

    elif repr_mode == "global_patch":
        patch_topk = debug["patch_topk"]
        topk_score = patch_topk["topk_score"]
        topk_idx = patch_topk["topk_idx"]

        idx = topk_idx.tolist()
        sims = (-topk_score).tolist()

    elif repr_mode == "global_patch_pool":
        pool_topk = debug["pool_topk"]
        topk_idx = pool_topk["topk_idx"]
        top_vote_vals = pool_topk["top_vote_vals"]

        idx = topk_idx.tolist()
        sims = top_vote_vals.tolist()

    else:
        raise ValueError(f"Unknown repr_mode={repr_mode}")
    
    topk_paths = [ref_paths[i] for i in idx]

    return float(dist), thr, bool(is_change), topk_paths, sims, ref_bank_id

#------------------------------- 추론

# event 단위 이상감지
@torch.no_grad()
def infer_event(
    imgs_bgr: List[np.ndarray],
    plc_idx: str,
    bank_root,
    model,
    device,
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
    if repr_mode not in {"global", "patch", "global_patch", "global_patch_pool"}:        
        raise ValueError(f"Unknown repr_mode={repr_mode} (use global|patch|global_patch|global_patch_pool)")

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
            top_p=top_p,
            preselect_m=preselect_m,
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
            topk_score, topk_idx = debug
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
        event_score = float(np.mean(frame_scores))
        anomaly_flag = 1 if event_score > thr else 0

    elif event_rule == "max":
        event_score = float(np.max(frame_scores))
        anomaly_flag = 1 if event_score > thr else 0

    elif event_rule == "vote":
        n_ab = int(sum(frame_change_flags))
        n = int(len(frame_change_flags))
        anomaly_flag = 1 if n_ab >= (n / 2) else 0
        event_score = float(n_ab / max(1, n))

    elif event_rule == "median":
        event_score = float(np.median(frame_scores))
        anomaly_flag = 1 if event_score > thr else 0

    else:
        raise ValueError(f"Unknown event_rule: {event_rule}")
    
    rep_idx = int(np.argmax(frame_scores)) if len(frame_scores) > 0 else 0 # vis

    # -------------------------
    # optional: VLM gate
    # -------------------------
    rep, summary = None, ""
    if use_two_stage_vlm and anomaly_flag == 1:
        idx = int(np.argmax(frame_scores))
        q_img = imgs_bgr[idx]
        ref_img_path = topk_paths_all[idx][0] if topk_paths_all[idx] else ""
        rep = {"frame_idx": idx, "ref_img_path": ref_img_path}

        if ref_img_path:
            vlm_result = vlm_gate(q_img, ref_img_path)
            anomaly_flag = 1 if bool(vlm_result.get("physical_change", False)) else 0
            summary = str(vlm_result.get("description", ""))

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