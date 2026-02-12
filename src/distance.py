# 1. threshold 캘리브레이션 : def calibrate_place(bank_root, plc_idx, model, device, k=3, percentile=95):
# 2. q 이미지에 대한 추론 : def infer_one(img, plc_idx : str, bank_root, model, device):
# - suhyun

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import json
from PIL import Image

from banker import load_bank_by_place, BGR_to_RGB
from dino_emb import make_embed , make_transform


def compute_knn_dist(
    q_emb: torch.Tensor,
    bank_root,
    plc_idx : str,
    k: int = 3,
):
    if q_emb.dim() == 1:        # (D,)
        q_emb = q_emb.unsqueeze(0)  # (1,D)

    # load bank
    ref_embs, ref_img_paths = load_bank_by_place(bank_root,plc_idx)
    if ref_embs is None:
        raise FileNotFoundError(f"No bank found for plc_idx={plc_idx}")
    
    k = min(k, ref_embs.shape[0])
    ref_embs = torch.from_numpy(ref_embs).float()

    sim = q_emb @ ref_embs.T
    topk_sim, top_k_idx = torch.topk(sim, k=k, dim=1)
    mean_dist = (1 - topk_sim).mean().item()

    return mean_dist, (topk_sim,top_k_idx,ref_img_paths)

@torch.no_grad()
def compute_and_save_threshold(bank_root, plc_idx, k=3, percentile=95):

    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    bank_npz = bank_root / plc_idx / "bank" / f"{plc_idx}.npz"
    th_npz   = bank_root / plc_idx / "th_calib" / f"{plc_idx}.npz"

    bank = torch.from_numpy(np.load(bank_npz, allow_pickle=True)["embs"]).float()  # (N,D)
    th   = torch.from_numpy(np.load(th_npz, allow_pickle=True)["embs"]).float()    # (M,D)
    
    k = min(k, bank.shape[0])
    sim = th @ bank.T                    # (M,N)
    topk_sim, _ = torch.topk(sim, k=k, dim=1)

    scores = (1.0 - topk_sim).mean(dim=1).numpy()  # (M,)
    thr = float(np.percentile(scores, percentile))

    out = {
        "plc_idx": plc_idx,
        "k": int(k),
        "percentile": int(percentile),
        "threshold": thr,
        "num_bank": int(bank.shape[0]),
        "num_th": int(len(scores)),
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

    #compute_dist
    dist , debug = compute_knn_dist(q_emb,bank_root,plc_idx,k=k)

    is_change = dist > thr

    # topk 이미지 경로 , 임베딩 반환
    topk_sim , topk_idx , ref_paths = debug
    topk_idx = topk_idx.squeeze(0).tolist()
    topk_sims = topk_sim.squeeze(0).tolist()
    topk_paths = [ref_paths[i] for i in topk_idx]

    return dist, thr, is_change, topk_paths, topk_sims


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

