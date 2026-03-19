# 1. 들어온 img에 대해서 이미지를 저장하고 장소 npz에 임베딩을 추가  def save_one_to_bank(img, plc_idx : str, save_root):
# 2. 장소 단위로 npz를 load해서 e npy 배열들과 이미지 경로 모음 list를 return  def load_bank_by_place(save_root, plc_idx):
# 3. 장소 단위로 npz를 전부 탐색해서 해당 폴더에 존재하는 img에 대해 npz를 다시 만듬  def rebuild_bank(save_root, plc_idx):

# save_root : ref_bank 폴더
# plc_idx : 장소에 대한 고유한 idx str
# img : numpy uint8

# - suhyun

"""
def load_bank_by_place(save_root, plc_idx):
def rebuild_bank(save_root, plc_idx):
"""

import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Dict, Tuple, Optional, List

from .dino_emb import make_embed , make_transform


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def BGR_to_RGB(img_bgr_uint8):
    img_rgb = img_bgr_uint8[:, :, ::-1]          # BGR->RGB
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    return img_pil

#경로에따른 npz에서 emb와 해당하는 이미지 경로를 return
def load_bank_npz(npz_path):
    """
    npz_path: Path or str
    return:
        ref_embs: numpy array [K, D]
        paths: list[str]
    """

    npz_path = Path(npz_path)

    if not npz_path.exists():
        return None, None

    data = np.load(npz_path, allow_pickle=True)
    ref_embs = data["embs"]       
    paths = data["paths"].tolist()     

    return ref_embs, paths

# 장소 폴더 단위로 npz를 load, emb와 대응되는 paths를 return
def load_bank_by_place(save_root, plc_idx, mode="bank"):

    save_root = Path(save_root)
    plc_idx = str(plc_idx)
    place_dir = save_root / plc_idx / mode

    npz_global = place_dir / f"{plc_idx}.npz"
    npz_patch  = place_dir / f"{plc_idx}_patch_.npz"

    return {
        "global": load_bank_npz(npz_global),
        "patch":  load_bank_npz(npz_patch),
    }

#해당 장소의 npz 초기화
def rebuild_bank(save_root, plc_idx, model, device, mode="bank", cfg=None):
    


    #임베딩 추출 cfg ----
    cfg = cfg or {}

    repr_mode   = str(cfg.get("repr", {}).get("repr_mode", "global"))  # global|patch|global_patch|global_patch_pool
    global_mode = str(cfg.get("embed", {}).get("global_mode", "patch_mean"))
    img_size    = int(cfg.get("embed", {}).get("img_size", 560))

    if repr_mode not in {"global", "patch", "global_patch", "global_patch_pool"}:
        raise ValueError(f"repr_mode must be global|patch|global_patch|global_patch_pool, got {repr_mode}")
    effective_mode = "global_patch" if repr_mode == "patch" else repr_mode

    save_root = Path(save_root)
    plc_idx = str(plc_idx)

    place_dir = save_root / plc_idx / mode
    place_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in place_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS])

    if not img_paths:
        print(f"[rebuild_bank] no images found: {place_dir}")
        return None

    tfm = make_transform(img_size=img_size)

    # emb를 npz로 저장 ---------------------
    paths: List[str] = []
    global_list: List[np.ndarray] = []
    patch_list: List[np.ndarray] = []

    embs_g = None
    embs_p = None

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img)

        out = make_embed(
            model, device, x,
            repr_mode=effective_mode,      # global|patch|global_patch|global_patch_poll
            global_mode=global_mode,
        )

        paths.append(str(p))

        if "global" in out:
            global_list.append(out["global"].detach().cpu().numpy().astype(np.float32))  # (D,)
        if "patch" in out:
            patch_list.append(out["patch"].detach().cpu().numpy().astype(np.float32))   # (P,D)

    # save global
    if global_list:
        embs_g = np.stack(global_list, axis=0)  # (N,D)

        # L2 정규화
        norm_g = np.linalg.norm(embs_g, axis=1, keepdims=True)
        embs_g = embs_g / np.clip(norm_g, 1e-12, None)

        np.savez_compressed(place_dir / f"{plc_idx}.npz",
                            embs=embs_g, paths=np.array(paths, dtype=object))

    # save patch
    if patch_list:
        embs_p = np.stack(patch_list, axis=0)   # (N,P,D)

        # L2 정규화
        norm_p = np.linalg.norm(embs_p, axis=2, keepdims=True)
        embs_p = embs_p / np.clip(norm_p, 1e-12, None)

        np.savez_compressed(place_dir / f"{plc_idx}_patch_.npz",
                            embs=embs_p, paths=np.array(paths, dtype=object))

    
    return {
        "global": (embs_g if global_list else None),
        "patch":  (embs_p if patch_list else None),
        "paths": paths,
    }
    