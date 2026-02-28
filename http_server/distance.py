# 1. threshold 캘리브레이션 : def calibrate_place(bank_root, plc_idx, model, device, k=3, percentile=95):
# 2. q 이미지에 대한 추론 : def infer_one(img, plc_idx : str, bank_root, model, device):
# - suhyun

import hashlib  
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


from PIL import Image

from banker import load_bank_by_place, BGR_to_RGB
from dino_emb import make_embed , make_transform
from vlm_gate import vlm_gate


def compute_knn_dist(
    q_emb: torch.Tensor,
    bank_root,
    plc_idx: str,
    k: int = 3,
    ref_embs: Optional[torch.Tensor] = None,
    ref_img_paths: Optional[List[str]] = None,
):

    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)  # (1,D)

    # load bank
    if ref_embs is None or ref_img_paths is None:
        ref_embs_np, ref_img_paths = load_bank_by_place(bank_root, plc_idx)
        if ref_embs_np is None:
            raise FileNotFoundError(f"No bank found for plc_idx={plc_idx}")

        ref_embs = torch.from_numpy(ref_embs_np).float().to(q_emb.device)
    else:
        ref_embs = ref_embs.to(q_emb.device)

    #둘다 정규화 (혹시 모르니)
    q_emb = F.normalize(q_emb, dim=1)
    ref_embs = F.normalize(ref_embs, dim=1)

    k = min(k, ref_embs.shape[0])

    sim = q_emb @ ref_embs.T                  # (1,N)
    topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)  # (1,k)

    mean_dist = (1.0 - topk_sim).mean().item()
    return mean_dist, (topk_sim, topk_idx, ref_img_paths)


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
def compute_and_save_threshold(bank_root, plc_idx, k=3, percentile=95, method = "robust"):
    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
    th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"

    with open(bank_npz, "rb") as f:
        bank_hash = hashlib.sha1(f.read()).hexdigest()[:8]

    bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()  # (N,D)
    th   = torch.from_numpy(np.load(th_npz, allow_pickle=True)["embs"]).float()    # (M,D)

    bank = F.normalize(bank, dim=1)
    th   = F.normalize(th, dim=1)
    
    k = min(k, bank.shape[0])
    sim = th @ bank.T                    # (M,N)
    topk_sim, _ = torch.topk(sim, k=k, dim=1)

    scores = (1.0 - topk_sim).mean(dim=1).numpy()  # (M,)

    if method == "percentile":
        thr = th_percentile(scores, percentile)

    elif method == "gaussian":
        thr = th_gaussian(scores, k=2.5)

    elif method == "robust":
        thr = th_robust(scores, k=2.5)

    else:
        raise ValueError(f"Unknown method: {method}")
    
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_bank_id = f"{plc_idx}_{method}_k{k}_p{percentile}_{bank_hash}_{created_at}"

    out = {
        "plc_idx": plc_idx,
        "k": int(k),
        "percentile": int(percentile),
        "threshold": thr,
        "num_bank": int(bank.shape[0]),
        "num_th": int(len(scores)),
        "ref_bank_id": ref_bank_id
    }

    thr_path = bank_root / plc_idx / "threshold.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return thr, scores, thr_path

@torch.no_grad()
def infer_one(img, plc_idx : str, bank_root, model, device):
    """
    img      : numpy uint8 (H,W,3)
    plc_idx  : str
    bank_root: ref_bank 폴더
    return:
        score: float
        thr: float
        is_change: bool
        topk_paths: list[str]
        topk_sims: list[float]
    """

    plc_idx = str(plc_idx)
    bank_root = Path(bank_root)

    #img emb 추출
    tfm = make_transform()
    img = BGR_to_RGB(img)
    img_tensor = tfm(img)
    q_emb = make_embed(model, device, img_tensor)   

    
    #th load
    thr_path = bank_root / plc_idx / "threshold.json"
    if not thr_path.exists():
        raise FileNotFoundError(f"No threshold.json for plc_idx={plc_idx}")

    with open(thr_path, "r") as f:
        meta = json.load(f)

    thr = float(meta["threshold"])
    k = int(meta["k"])
    ref_bank_id = str(meta["ref_bank_id"])

    #compute_dist
    dist , debug = compute_knn_dist(q_emb,bank_root,plc_idx,k=k)

    is_change = dist > thr

    # topk 이미지 경로 , 임베딩 반환
    topk_sim , topk_idx , ref_paths = debug
    topk_idx = topk_idx.squeeze(0).tolist()
    topk_sims = topk_sim.squeeze(0).tolist()
    topk_paths = [ref_paths[i] for i in topk_idx]

    return dist, thr, is_change, topk_paths, topk_sims, ref_bank_id


def calibrate_place(bank_root, plc_idx, model, device, k=3, percentile=95):
    """
    버튼 한 번으로 캘리브레이션:
      1. ref bank rebuild
      2. th_calib rebuild
      3) threshold 계산/저장

    return:
        thr: float
        scores: np.ndarray (M,)
        thr_path: Path
    """

    from banker import rebuild_bank

    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    rebuild_bank(bank_root, plc_idx, model, device, mode="bank")
    rebuild_bank(bank_root, plc_idx, model, device, mode="th_calib")

    thr, scores, thr_path = compute_and_save_threshold(
        bank_root, plc_idx, k=k, percentile=percentile
    )

    return thr, scores, thr_path


@torch.no_grad()
def infer_event(
    imgs_bgr: List[np.ndarray],      # 여러 장: uint8 BGR
    plc_idx: str,                    # place_id / plc_idx
    bank_root,                       # bank 파일의 root
    model,
    device,
    event_rule: str = "vote",        # event 단위 이상 결정 정책
    use_two_stage_vlm = False
) -> Dict[str, Any]:

    
    plc_idx = str(plc_idx)
    bank_root = Path(bank_root)

    if not imgs_bgr:
        raise ValueError("imgs_bgr is empty")

    # threshold json load
    thr_path = bank_root / plc_idx / "threshold.json"
    if not thr_path.exists():
        raise FileNotFoundError(f"No threshold.json for plc_idx={plc_idx} ({thr_path})")

    with open(thr_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    thr = float(meta["threshold"])
    k = int(meta["k"])
    ref_bank_id = str(meta["ref_bank_id"])


    # bank emb load
    ref_embs_np, ref_img_paths = load_bank_by_place(bank_root, plc_idx)
    if ref_embs_np is None:
        raise FileNotFoundError(f"No bank found for plc_idx={plc_idx}")

    ref_embs = torch.from_numpy(ref_embs_np).float()  # (N,D)
    k = min(k, ref_embs.shape[0])

    # 각 이미지별 점수 계산
    tfm = make_transform()  
    frame_scores: List[float] = []
    frame_change_flags: List[int] = []
    topk_paths_all: List[List[str]] = []
    topk_sims_all: List[List[float]] = []

    for img_bgr in imgs_bgr:
        img_rgb = BGR_to_RGB(img_bgr)

        img_tensor = tfm(img_rgb)
        q_emb = make_embed(model, device, img_tensor)
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)

        dist, debug = compute_knn_dist(q_emb, bank_root, plc_idx, k=k,
                              ref_embs=ref_embs, ref_img_paths=ref_img_paths)

        topk_sim, topk_idx, ref_paths = debug
        idx_list = topk_idx.squeeze(0).tolist()
        sim_list = topk_sim.squeeze(0).tolist()
        topk_paths = [ref_paths[i] for i in idx_list]

        is_change = dist > thr

        frame_scores.append(float(dist))
        frame_change_flags.append(1 if is_change else 0)
        topk_paths_all.append(topk_paths)
        topk_sims_all.append([float(s) for s in sim_list])

    #event 단위 
    if event_rule == "mean":
        event_score = float(np.mean(frame_scores))
        anomaly_flag = 1 if event_score > thr else 0

    elif event_rule == "max":
        event_score = float(np.max(frame_scores))
        anomaly_flag = 1 if event_score > thr else 0

    elif event_rule == "vote":
        # 프레임별 change 여부 기반 다수결
        n_abnormal = sum(frame_change_flags)
        n_total = len(frame_change_flags)

        anomaly_flag = 1 if n_abnormal >= (n_total / 2) else 0
        event_score = float(n_abnormal / n_total)  # vote 비율을 점수로 저장

    else:
        raise ValueError(f"Unknown event_mode: {event_rule}")
    
    rep = None   
    summary = ""

    if use_two_stage_vlm:
        m2 = 0.9
        max_ratio = max(frame_scores) / (thr + 1e-12)
        #need_vlm = (anomaly_flag == 1) and (max_ratio < 1.0 + m2)
        if anomaly_flag == 1:
            need_vlm = True
            
        if need_vlm:
            idx = int(np.argmax(frame_scores))
            q_img = imgs_bgr[idx]
            ref_img_path = topk_paths_all[idx][0]
            rep = {"frame_idx": idx, "ref_img_path": ref_img_path}

            vlm_result = vlm_gate(q_img, ref_img_path)
            anomaly_flag = 1 if vlm_result["physical_change"] else 0
            summary = vlm_result["description"]   


    ref_topk_json = json.dumps(
        {"topk_paths": topk_paths_all, "topk_sims": topk_sims_all, "rep": rep},
        ensure_ascii=False,
    )

    return {
        "threshold": thr, # 해당 plc의 th
        "frame_scores": frame_scores, # 각 이미지들의 이상 score
        "anomaly_flag": int(anomaly_flag), #binary 이상 유무
        "frame_change_flags": frame_change_flags, #각 이미지별 이상 유무
        "event_score": event_score, #이벤트 단위 점수
        "ref_bank_id" : ref_bank_id,
        "ref_topk_json": ref_topk_json, #참조 이미지 경로 , 유사도
        "summary": summary, #additional 추가 확장 고려
    }

